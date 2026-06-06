import json
import threading

import httpx
import pytest

openai = pytest.importorskip("openai")

import agentping
from agentping.instrumentation import _openai as op_inst


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


class _FakeDetails:
    def __init__(self, cached_tokens=0):
        self.cached_tokens = cached_tokens


class _FakeUsage:
    def __init__(self, prompt_tokens=100, completion_tokens=50, cached_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = _FakeDetails(cached_tokens=cached_tokens)


class _FakeResponse:
    def __init__(self, model="gpt-4o-mini", usage=None):
        self.model = model
        self.usage = usage or _FakeUsage()


def test_instrument_openai_emits_llm_call():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        from openai.resources.chat.completions.completions import Completions

    original = Completions.create
    Completions.create = lambda self, *a, **kw: _FakeResponse(
        usage=_FakeUsage(prompt_tokens=100, completion_tokens=50, cached_tokens=20)
    )

    try:
        agentping.instrument_openai()
        client = openai.OpenAI(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            client.chat.completions.create(model="gpt-4o-mini", messages=[])
        agentping.flush(timeout=2.0)
    finally:
        op_inst.uninstrument()
        Completions.create = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o-mini"
    assert data["input_tokens"] == 80
    assert data["cached_input_tokens"] == 20
    assert data["output_tokens"] == 50
    assert "latency_ms" in data


def test_instrument_openai_is_idempotent():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))
    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        from openai.resources.chat.completions.completions import Completions

    original = Completions.create
    try:
        agentping.instrument_openai()
        first = Completions.create
        agentping.instrument_openai()
        second = Completions.create
        assert first is second
    finally:
        op_inst.uninstrument()
        Completions.create = original
