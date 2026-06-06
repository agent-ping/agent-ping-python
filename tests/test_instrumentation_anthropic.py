import json
import threading

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


class _FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50,
                 cache_read_input_tokens=None, cache_creation_input_tokens=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _FakeResponse:
    def __init__(self, model="claude-sonnet-4-5", usage=None):
        self.model = model
        self.usage = usage or _FakeUsage()


def test_instrument_anthropic_emits_llm_call():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    try:
        from anthropic.resources.messages import Messages
    except Exception:
        from anthropic.resources.messages.messages import Messages

    original = Messages.create
    Messages.create = lambda self, *a, **kw: _FakeResponse(
        usage=_FakeUsage(input_tokens=100, output_tokens=50, cache_read_input_tokens=20)
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

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    assert len(event_calls) >= 1
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-sonnet-4-5"
    assert data["input_tokens"] == 100
    assert data["output_tokens"] == 50
    assert data["cached_input_tokens"] == 20
    assert "latency_ms" in data


def test_instrument_anthropic_is_idempotent():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))
    try:
        from anthropic.resources.messages import Messages
    except Exception:
        from anthropic.resources.messages.messages import Messages

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
