"""Offline regressions for forwarding, quoted follow-ups and cancellation.

No QQ messages, HTTP calls or model requests are made by this test module.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


class Plain:
    type = "Plain"

    def __init__(self, text):
        self.text = text


class Image:
    @staticmethod
    def fromURL(url):
        return types.SimpleNamespace(url=url)


class Forward:
    type = "Forward"

    def __init__(self, id):
        self.id = id


class Reply:
    type = "Reply"

    def __init__(self, id):
        self.id = id
        self.message_str = "[Forward Message]"
        self.chain = []
        self.sender_id = "author"
        self.sender_nickname = "Author"


class Event:
    def __init__(
        self, chain, id="message-1", session="group-a", sender="reader", action=None
    ):
        self.message_obj = types.SimpleNamespace(
            message=list(chain), message_id=id, timestamp=1, message_str=""
        )
        self.message_str = ""
        self.unified_msg_origin = session
        self.sender = sender
        self.bot = types.SimpleNamespace(
            call_action=action if action is not None else AsyncMock(return_value={})
        )

    def get_sender_name(self):
        return self.sender

    def get_sender_id(self):
        return self.sender


def node(*segments):
    return {
        "sender": {"nickname": "Author", "user_id": "author"},
        "time": 1,
        "message": list(segments),
    }


def picture(url="https://example.invalid/image.png"):
    return {"type": "image", "data": {"url": url}}


@pytest.fixture
def media(monkeypatch):
    modules = {
        name: types.ModuleType(name)
        for name in (
            "astrbot",
            "astrbot.api",
            "astrbot.core",
            "astrbot.core.message",
            "astrbot.core.message.components",
            "astrbot.core.platform",
            "astrbot.core.platform.astr_message_event",
            "forward_test_package",
            "forward_test_package.utils",
        )
    }
    modules["astrbot.api"].logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    components = modules["astrbot.core.message.components"]
    for cls in (Plain, Image, Forward, Reply):
        setattr(components, cls.__name__, cls)
    modules["astrbot.core.platform.astr_message_event"].AstrMessageEvent = Event
    root = Path(__file__).parents[1]
    modules["forward_test_package"].__path__ = [str(root)]
    modules["forward_test_package.utils"].__path__ = [str(root / "utils")]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    loaded = {}
    for name in ("forward_message_parser", "forward_media"):
        fullname = f"forward_test_package.utils.{name}"
        spec = importlib.util.spec_from_file_location(
            fullname, root / "utils" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["forward_media"], loaded["forward_message_parser"]


def test_parser_without_renderer_preserves_text_only_default(media):
    _, parser = media

    async def scenario():
        action = AsyncMock(return_value={"messages": [node(picture())]})
        event = Event([Forward("resource")], action=action)
        assert await parser.ForwardMessageParser.try_parse_and_replace(
            event, True, True
        )
        assert "[图片]" in event.message_str
        assert "图片内容" not in event.message_str

    asyncio.run(scenario())


def test_caption_preserves_order_and_shares_limit_across_forward_components(media):
    module, _ = media

    async def scenario():
        async def action(_, **args):
            resource = args.get("message_id", args.get("id"))
            return {
                "messages": [
                    node(
                        {"type": "text", "data": {"text": resource}},
                        picture(f"https://example.invalid/{resource}.png"),
                    )
                ]
            }

        processor = module.ForwardMediaProcessor()
        describe = AsyncMock(return_value="a red diagram")
        event = Event(
            [Forward("first"), Plain("between"), Forward("second")], action=action
        )
        policy = module.ForwardMediaPolicy("vision", "describe", max_images=1)
        assert await processor.prepare(event, policy, describe)
        assert event.message_str.index("first") < event.message_str.index("between")
        assert event.message_str.index("between") < event.message_str.index("second")
        assert "a red diagram" in event.message_str
        assert "超过" in event.message_str
        assert describe.await_count == 1
        await processor.close()

    asyncio.run(scenario())


def test_quote_waits_for_exact_message_without_blocking_unrelated_traffic(media):
    module, _ = media

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def describe(_):
            entered.set()
            await release.wait()
            return "a blue chart"

        action = AsyncMock(return_value={"messages": [node(picture())]})
        processor = module.ForwardMediaProcessor()
        policy = module.ForwardMediaPolicy("vision", "describe")
        original = Event([Forward("resource")], action=action)
        first = asyncio.create_task(processor.prepare(original, policy, describe))
        await entered.wait()
        reply = Reply("message-1")
        quoted = Event([reply, Plain("read it")], id="message-2", action=action)
        second = asyncio.create_task(processor.prepare(quoted, policy, describe))
        unrelated = Event([Plain("another topic")], id="message-3", action=action)
        assert await processor.prepare(unrelated, policy, describe) is False
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        await asyncio.gather(first, second)
        assert "a blue chart" in reply.message_str
        assert "a blue chart" in reply.chain[0].text
        assert action.await_count == 1
        await processor.close()

    asyncio.run(scenario())


def test_old_quote_is_fetched_and_plain_quote_is_not_rewritten(media):
    module, _ = media

    async def scenario():
        async def action(name, **args):
            if name == "get_msg":
                segments = (
                    [{"type": "forward", "data": {"id": "resource"}}]
                    if args["message_id"] == "old"
                    else [{"type": "text", "data": {"text": "ordinary"}}]
                )
                return {"data": node(*segments)}
            return {"messages": [node(picture())]}

        processor = module.ForwardMediaProcessor()
        policy = module.ForwardMediaPolicy("vision", "describe")
        describe = AsyncMock(return_value="a green chart")
        reply = Reply("old")
        await processor.prepare(Event([reply], action=action), policy, describe)
        assert "a green chart" in reply.message_str
        plain = Reply("plain")
        await processor.prepare(Event([plain], action=action), policy, describe)
        assert plain.message_str == "[Forward Message]"
        assert describe.await_count == 1
        await processor.close()

    asyncio.run(scenario())


def test_same_message_id_in_other_group_never_reuses_content(media):
    module, _ = media

    async def scenario():
        processor = module.ForwardMediaProcessor()
        policy = module.ForwardMediaPolicy("vision", "describe")
        action = AsyncMock(return_value={"messages": [node(picture())]})
        first = Event([Forward("resource")], session="group-a", action=action)
        second = Event([Forward("resource")], session="group-b", action=action)
        await processor.prepare(
            first, policy, AsyncMock(return_value="group A picture")
        )
        await processor.prepare(
            second, policy, AsyncMock(return_value="group B picture")
        )
        assert "group B picture" in second.message_str
        assert "group A picture" not in second.message_str
        await processor.close()

    asyncio.run(scenario())


def test_direct_followup_links_once_and_only_for_same_sender(media):
    module, _ = media

    async def scenario():
        processor = module.ForwardMediaProcessor()
        policy = module.ForwardMediaPolicy("vision", "describe")
        action = AsyncMock(return_value={"messages": [node(picture())]})
        describe = AsyncMock(return_value="picture description")
        await processor.prepare(
            Event([Forward("resource")], action=action), policy, describe
        )
        stranger = Event([Plain("question")], sender="someone-else")
        await processor.prepare(stranger, policy, describe, link_followup=True)
        assert len(stranger.message_obj.message) == 1
        followup = Event([Plain("read it")])
        await processor.prepare(followup, policy, describe, link_followup=True)
        assert "picture description" in followup.message_str
        later = Event([Plain("unrelated")])
        await processor.prepare(later, policy, describe, link_followup=True)
        assert len(later.message_obj.message) == 1
        await processor.close()

    asyncio.run(scenario())


def test_duplicate_images_and_failed_images_have_explicit_outcomes(media):
    module, _ = media

    async def scenario():
        describe = AsyncMock(side_effect=["recognized", RuntimeError("offline")])
        renderer = module._ImageRenderer(describe, limit=2)
        first = await renderer(picture()["data"])
        assert await renderer(picture()["data"]) == first
        failure = await renderer(picture("https://example.invalid/b.png")["data"])
        assert "识别失败" in failure
        assert renderer.described == 1 and renderer.unread == 1
        assert describe.await_count == 2
        assert "没有可用" in await renderer({"file": "/etc/passwd"})

    asyncio.run(scenario())


def test_cancelled_quote_does_not_cancel_shared_work_and_close_cleans_up(media):
    module, _ = media

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        processor = module.ForwardMediaProcessor()

        async def load():
            entered.set()
            await release.wait()
            return module.ForwardMediaResult("result", "ready")

        key = ("group", "message", None)
        first = asyncio.create_task(processor._resolve(key, "reader", load, 90))
        await entered.wait()
        second = asyncio.create_task(processor._resolve(key, "reader", load, 90))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert not processor._entries[key].task.done()
        await processor.close()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not processor._entries

    asyncio.run(scenario())


def test_capacity_never_evicts_running_task(media):
    module, _ = media

    async def scenario():
        processor = module.ForwardMediaProcessor(max_entries=1)
        entered = asyncio.Event()

        async def load():
            entered.set()
            await asyncio.Event().wait()

        pending = asyncio.create_task(
            processor._resolve(("a", "1"), "reader", load, 90)
        )
        await entered.wait()
        result = await processor._resolve(("a", "2"), "reader", load, 90)
        assert result.status == "busy"
        assert len(processor._entries) == 1
        await processor.close()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(scenario())


def test_failed_task_has_short_ttl_and_can_retry(media):
    module, _ = media

    async def scenario():
        clock = [0.0]
        processor = module.ForwardMediaProcessor(clock=lambda: clock[0])
        timeout = AsyncMock(side_effect=asyncio.TimeoutError)
        key = ("a", "1")
        result = await processor._resolve(key, "reader", timeout, 90)
        assert result.status == "timeout"
        assert "未完整读取" in result.text
        clock[0] = 6
        success = AsyncMock(
            return_value=module.ForwardMediaResult("recovered", "ready")
        )
        result = await processor._resolve(key, "reader", success, 90)
        assert result.text == "recovered"
        await processor.close()

    asyncio.run(scenario())


def test_unsupported_platform_preserves_original_components(media):
    module, _ = media

    async def scenario():
        processor = module.ForwardMediaProcessor()
        event = Event([Forward("resource")])
        event.bot = None
        describe = AsyncMock()
        assert not await processor.prepare(
            event, module.ForwardMediaPolicy("v", "p"), describe
        )
        assert isinstance(event.message_obj.message[0], Forward)
        describe.assert_not_awaited()
        await processor.close()

    asyncio.run(scenario())


def test_unconfirmed_quote_timeout_keeps_original_reply(media, monkeypatch):
    module, parser = media

    async def scenario():
        monkeypatch.setattr(
            parser.ForwardMessageParser,
            "parse_quoted_forward",
            AsyncMock(side_effect=asyncio.TimeoutError),
        )
        processor = module.ForwardMediaProcessor()
        reply = Reply("ordinary")
        reply.chain = [Plain("original quote")]
        await processor.prepare(
            Event([reply]), module.ForwardMediaPolicy("v", "p"), AsyncMock()
        )
        assert reply.chain[0].text == "original quote"
        assert reply.message_str == "[Forward Message]"
        await processor.close()

    asyncio.run(scenario())


def test_failed_forward_fetch_is_retryable_not_cached_as_success(media):
    module, _ = media

    async def scenario():
        now = [0.0]
        processor = module.ForwardMediaProcessor(clock=lambda: now[0])
        policy = module.ForwardMediaPolicy("v", "p")
        broken = Event([Forward("resource")], action=AsyncMock(return_value={}))
        await processor.prepare(broken, policy, AsyncMock())
        assert "内容获取失败" in broken.message_str
        assert next(iter(processor._entries.values())).task.result().status == "partial"
        now[0] = 6
        restored = Event(
            [Forward("resource")],
            action=AsyncMock(return_value={"messages": [node(picture())]}),
        )
        await processor.prepare(restored, policy, AsyncMock(return_value="recovered"))
        assert "recovered" in restored.message_str
        await processor.close()

    asyncio.run(scenario())


def test_failed_quote_lookup_is_not_cached_as_an_ordinary_message(media):
    module, _ = media

    async def scenario():
        now = [0.0]
        processor = module.ForwardMediaProcessor(clock=lambda: now[0])
        policy = module.ForwardMediaPolicy("v", "p")
        reply = Reply("old")
        await processor.prepare(Event([reply]), policy, AsyncMock())
        assert reply.message_str == "[Forward Message]"
        assert next(iter(processor._entries.values())).task.result().status == "unknown"
        now[0] = 6

        async def action(name, **kwargs):
            if name == "get_msg":
                return {"message": [{"type": "forward", "data": {"id": "resource"}}]}
            return {"messages": [node(picture())]}

        await processor.prepare(
            Event([reply], action=action),
            policy,
            AsyncMock(return_value="retried caption"),
        )
        assert "retried caption" in reply.message_str
        await processor.close()

    asyncio.run(scenario())


def test_busy_processor_does_not_replace_unconfirmed_quote(media):
    module, _ = media

    async def scenario():
        processor = module.ForwardMediaProcessor(max_entries=1)
        entered = asyncio.Event()

        async def load():
            entered.set()
            await asyncio.Event().wait()

        pending = asyncio.create_task(processor._resolve(("a", "1"), "s", load, 90))
        await entered.wait()
        reply = Reply("ordinary")
        reply.chain = [Plain("original text")]
        await processor.prepare(
            Event([reply]), module.ForwardMediaPolicy("v", "p"), AsyncMock()
        )
        assert reply.chain[0].text == "original text"
        await processor.close()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(scenario())


def test_nested_forward_depth_and_cycles_do_not_repeat_caption_calls(media):
    module, _ = media

    async def scenario():
        async def action(_, **kwargs):
            resource = kwargs.get("message_id", kwargs.get("id"))
            return {
                "messages": [
                    node(
                        picture(f"https://example.invalid/{resource}.png"),
                        {"type": "forward", "data": {"id": "nested"}},
                    )
                ]
            }

        for depth, count, marker in ((0, 1, "层级过深"), (3, 2, "重复嵌套")):
            processor = module.ForwardMediaProcessor()
            describe = AsyncMock(return_value="caption")
            event = Event([Forward("root")], action=action)
            await processor.prepare(
                event, module.ForwardMediaPolicy("v", "p", max_depth=depth), describe
            )
            assert describe.await_count == count
            assert marker in event.message_str
            await processor.close()

    asyncio.run(scenario())


def test_policy_changes_and_missing_message_ids_do_not_reuse_old_results(media):
    module, _ = media

    async def scenario():
        processor = module.ForwardMediaProcessor()
        action = AsyncMock(return_value={"messages": [node(picture())]})
        describe = AsyncMock(side_effect=["one", "two", "three", "four"])
        for message_id, prompt, expected in (
            ("same", "first", "one"),
            ("same", "second", "two"),
            ("", "first", "three"),
            ("", "first", "four"),
        ):
            event = Event([Forward("resource")], id=message_id, action=action)
            await processor.prepare(
                event, module.ForwardMediaPolicy("v", prompt), describe
            )
            assert expected in event.message_str
        assert describe.await_count == 4
        await processor.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("window,elapsed", [(0, 1), (10, 11)])
def test_disabled_or_expired_followup_does_not_attach_content(media, window, elapsed):
    module, _ = media

    async def scenario():
        now = [0.0]
        processor = module.ForwardMediaProcessor(clock=lambda: now[0])
        policy = module.ForwardMediaPolicy("v", "p")
        action = AsyncMock(return_value={"messages": [node(picture())]})
        await processor.prepare(
            Event([Forward("resource")], action=action),
            policy,
            AsyncMock(return_value="caption"),
        )
        now[0] = elapsed
        followup = Event([Plain("new topic")])
        await processor.prepare(
            followup, policy, AsyncMock(), link_followup=True, followup_seconds=window
        )
        assert len(followup.message_obj.message) == 1
        await processor.close()

    asyncio.run(scenario())


def test_total_timeout_cancels_work_and_does_not_leave_background_task(media):
    module, _ = media

    async def scenario():
        processor = module.ForwardMediaProcessor()
        cancelled = asyncio.Event()

        async def load():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        result = await asyncio.wait_for(processor._resolve(("a", "1"), "s", load, 1), 3)
        assert result.status == "timeout"
        assert cancelled.is_set()
        await processor.close()

    asyncio.run(scenario())


def test_ready_results_expire_and_completed_entries_can_be_evicted(media):
    module, _ = media

    async def scenario():
        now = [0.0]
        processor = module.ForwardMediaProcessor(
            max_entries=1, ttl=10, clock=lambda: now[0]
        )
        factory = AsyncMock(return_value=module.ForwardMediaResult("ready", "ready"))
        await processor._resolve(("a", "1"), "s", factory, 90)
        await processor._resolve(("a", "1"), "s", factory, 90)
        assert factory.await_count == 1
        now[0] = 11
        await processor._resolve(("a", "1"), "s", factory, 90)
        assert factory.await_count == 2
        await processor._resolve(("a", "2"), "s", factory, 90)
        assert list(processor._entries) == [("a", "2")]
        await processor.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "scope,is_at,keyword,expected",
    [
        ("all", False, False, True),
        ("at_only", False, True, False),
        ("at_only", True, False, True),
        ("keyword_only", True, False, False),
        ("keyword_only", False, True, True),
        ("mention_only", False, False, False),
        ("mention_only", False, True, True),
    ],
)
def test_scope_policy_is_shared_by_forward_and_quote_paths(
    media, scope, is_at, keyword, expected
):
    module, _ = media
    assert module.allows_image_caption(scope, is_at, keyword) is expected
