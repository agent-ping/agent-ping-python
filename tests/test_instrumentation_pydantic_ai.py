import json
import threading

import httpx
import pytest

pytest.importorskip("pydantic_ai")

import agentping
from agentping.instrumentation import _pydantic_ai as pai_inst


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
    def __init__(self, input_tokens=200, output_tokens=80, cache_read_tokens=0, cache_write_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens


class _FakeResult:
    def __init__(self, usage=None):
        self.usage = usage or _FakeUsage()


def test_instrument_pydantic_ai_emits_llm_call():
    import os
    os.environ.setdefault("OPENAI_API_KEY", "x")

    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from pydantic_ai import Agent
    original_run_sync = Agent.run_sync
    Agent.run_sync = lambda self, *a, **kw: _FakeResult(
        usage=_FakeUsage(input_tokens=400, output_tokens=150)
    )

    try:
        agentping.instrument_pydantic_ai()
        agent = Agent("openai:gpt-4o-mini")
        with agentping.run("triage") as r:
            rid = r.id
            agent.run_sync("classify this")
        agentping.flush(timeout=2.0)
    finally:
        pai_inst.uninstrument()
        Agent.run_sync = original_run_sync

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o-mini"
    assert data["input_tokens"] == 400
    assert data["output_tokens"] == 150
    assert "latency_ms" in data


def test_instrument_pydantic_ai_splits_cache_read():
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "x")
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from pydantic_ai import Agent
    original = Agent.run_sync
    Agent.run_sync = lambda self, *a, **kw: _FakeResult(
        usage=_FakeUsage(input_tokens=500, output_tokens=120, cache_read_tokens=200)
    )

    try:
        agentping.instrument_pydantic_ai()
        agent = Agent("anthropic:claude-sonnet-4-5")
        with agentping.run("classifier") as r:
            rid = r.id
            agent.run_sync("hi")
        agentping.flush(timeout=2.0)
    finally:
        pai_inst.uninstrument()
        Agent.run_sync = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    data = llm_events[0]["data"]
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-sonnet-4-5"
    # 500 prompt - 200 cached = 300 fresh input
    assert data["input_tokens"] == 300
    assert data["cached_input_tokens"] == 200


def test_instrument_pydantic_ai_is_idempotent():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))

    from pydantic_ai import Agent
    original = Agent.run_sync
    try:
        agentping.instrument_pydantic_ai()
        first = Agent.run_sync
        agentping.instrument_pydantic_ai()
        second = Agent.run_sync
        assert first is second
    finally:
        pai_inst.uninstrument()
        Agent.run_sync = original
