import json
import threading
from types import SimpleNamespace

import httpx
import pytest

anthropic = pytest.importorskip("anthropic")

import agentping
from agentping.instrumentation import _anthropic as ap_inst


def _capture():
    captured: list = []
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            captured.append({
                "url": str(request.url),
                "body": json.loads(request.content) if request.content else {},
            })
        return httpx.Response(202, json={})

    return captured, httpx.MockTransport(handler)


def _events(captured, rid, event_type):
    flat = []
    for c in captured:
        if f"/v1/runs/{rid}/events" in c["url"]:
            flat.extend(c["body"]["events"])
    return [e for e in flat if e["type"] == event_type]


def _messages_cls():
    try:
        from anthropic.resources.messages import Messages
    except Exception:
        from anthropic.resources.messages.messages import Messages
    return Messages


class _FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50,
                 cache_read_input_tokens=None, cache_creation_input_tokens=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _FakeResponse:
    def __init__(self, model="claude-sonnet-4-5", usage=None, stop_reason="end_turn", content=None):
        self.model = model
        self.usage = usage or _FakeUsage()
        self.stop_reason = stop_reason
        self.content = content or []


def test_instrument_anthropic_emits_llm_call():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    Messages = _messages_cls()
    original = Messages.create
    Messages.create = lambda self, *a, **kw: _FakeResponse(
        usage=_FakeUsage(input_tokens=100, output_tokens=50, cache_read_input_tokens=20),
        stop_reason="tool_use",
        content=[SimpleNamespace(type="text"), SimpleNamespace(type="tool_use"), SimpleNamespace(type="tool_use")],
    )

    try:
        agentping.instrument_anthropic()
        client = anthropic.Anthropic(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            client.messages.create(model="claude-sonnet-4-5", max_tokens=10, messages=[])

        agentping.flush(timeout=2.0)
    finally:
        ap_inst.uninstrument()
        Messages.create = original

    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-sonnet-4-5"
    # Anthropic reports input_tokens net of the cached prefix. Ingest wants
    # the gross count and subtracts cached_input_tokens itself.
    assert data["input_tokens"] == 120
    assert data["output_tokens"] == 50
    assert data["cached_input_tokens"] == 20
    assert data["finish_reason"] == "tool_use"
    assert data["tool_calls"] == 2
    assert "latency_ms" in data
    assert "stream" not in data


def test_instrument_anthropic_emits_errored_llm_call_and_reraises():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    Messages = _messages_cls()
    original = Messages.create

    def boom(self, *a, **kw):
        raise RuntimeError("overloaded_error")

    Messages.create = boom

    try:
        agentping.instrument_anthropic()
        client = anthropic.Anthropic(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            with pytest.raises(RuntimeError):
                client.messages.create(model="claude-sonnet-4-5", max_tokens=10, messages=[])
        agentping.flush(timeout=2.0)
    finally:
        ap_inst.uninstrument()
        Messages.create = original

    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["status"] == "error"
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-sonnet-4-5"
    assert data["error"] == "overloaded_error"
    assert data["exception"] == "RuntimeError"
    assert "latency_ms" in data


def _raw_stream_events():
    return [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                model="claude-sonnet-4-5",
                usage=_FakeUsage(input_tokens=80, output_tokens=1, cache_read_input_tokens=30),
            ),
        ),
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta"),
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="tool_use")),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(output_tokens=42),
        ),
        SimpleNamespace(type="message_stop"),
    ]


def test_instrument_anthropic_stream_true_emits_one_llm_call_on_close():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    Messages = _messages_cls()
    original = Messages.create
    Messages.create = lambda self, *a, **kw: iter(_raw_stream_events())

    try:
        agentping.instrument_anthropic()
        client = anthropic.Anthropic(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            stream = client.messages.create(model="claude-sonnet-4-5", max_tokens=10, messages=[], stream=True)
            seen = [getattr(e, "type", None) for e in stream]
        agentping.flush(timeout=2.0)
    finally:
        ap_inst.uninstrument()
        Messages.create = original

    # Events pass through unchanged.
    assert seen[0] == "message_start" and seen[-1] == "message_stop"

    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["stream"] is True
    assert data["model"] == "claude-sonnet-4-5"
    assert data["input_tokens"] == 110  # 80 net + 30 cached
    assert data["cached_input_tokens"] == 30
    assert data["output_tokens"] == 42  # last message_delta wins
    assert data["finish_reason"] == "tool_use"
    assert data["tool_calls"] == 1


def test_instrument_anthropic_messages_stream_context_manager():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    Messages = _messages_cls()
    original = getattr(Messages, "stream", None)
    if original is None:
        pytest.skip("this anthropic release has no messages.stream")

    class _FakeMessageStream:
        text_stream = iter(["hel", "lo"])

        @property
        def current_message_snapshot(self):
            return _FakeResponse(
                usage=_FakeUsage(input_tokens=60, output_tokens=9),
                stop_reason="end_turn",
            )

    class _FakeManager:
        def __enter__(self):
            return _FakeMessageStream()

        def __exit__(self, *a):
            return False

    Messages.stream = lambda self, *a, **kw: _FakeManager()

    try:
        agentping.instrument_anthropic()
        client = anthropic.Anthropic(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            with client.messages.stream(model="claude-sonnet-4-5", max_tokens=10, messages=[]) as s:
                chunks = list(s.text_stream)
        agentping.flush(timeout=2.0)
    finally:
        ap_inst.uninstrument()
        Messages.stream = original

    assert chunks == ["hel", "lo"]
    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["stream"] is True
    assert data["input_tokens"] == 60
    assert data["output_tokens"] == 9
    assert data["finish_reason"] == "end_turn"


def test_instrument_anthropic_is_idempotent():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))
    Messages = _messages_cls()

    original = Messages.create
    try:
        agentping.instrument_anthropic()
        first = Messages.create
        agentping.instrument_anthropic()
        second = Messages.create
        assert first is second
    finally:
        ap_inst.uninstrument()
        Messages.create = original
