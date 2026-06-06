import json
import threading

import httpx
import pytest

cohere = pytest.importorskip("cohere")

import agentping
from agentping.instrumentation import _cohere as c_inst


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


class _BilledUnits:
    def __init__(self, input_tokens=150, output_tokens=60):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeUsage:
    def __init__(self, billed_units=None):
        self.billed_units = billed_units or _BilledUnits()


class _FakeResponse:
    def __init__(self, usage=None):
        self.usage = usage or _FakeUsage()


def test_instrument_cohere_emits_llm_call_with_billed_units():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    original = cohere.ClientV2.chat
    cohere.ClientV2.chat = lambda self, *a, **kw: _FakeResponse(
        usage=_FakeUsage(billed_units=_BilledUnits(input_tokens=300, output_tokens=120))
    )

    try:
        agentping.instrument_cohere()
        client = cohere.ClientV2(api_key="x")
        with agentping.run("rfp") as r:
            rid = r.id
            client.chat(model="command-r-plus-08-2024", messages=[])
        agentping.flush(timeout=2.0)
    finally:
        c_inst.uninstrument()
        cohere.ClientV2.chat = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "cohere"
    assert data["model"] == "command-r-plus-08-2024"
    assert data["input_tokens"] == 300
    assert data["output_tokens"] == 120
    assert "latency_ms" in data


def test_instrument_cohere_is_idempotent():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))

    original = cohere.ClientV2.chat
    try:
        agentping.instrument_cohere()
        first = cohere.ClientV2.chat
        agentping.instrument_cohere()
        second = cohere.ClientV2.chat
        assert first is second
    finally:
        c_inst.uninstrument()
        cohere.ClientV2.chat = original


def test_instrument_cohere_handles_missing_billed_units():
    """If a response lacks the billed_units block, no token fields are emitted but the event still lands."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    class _NoBilled:
        usage = None

    original = cohere.ClientV2.chat
    cohere.ClientV2.chat = lambda self, *a, **kw: _NoBilled()

    try:
        agentping.instrument_cohere()
        client = cohere.ClientV2(api_key="x")
        with agentping.run("classify") as r:
            rid = r.id
            client.chat(model="command-light", messages=[])
        agentping.flush(timeout=2.0)
    finally:
        c_inst.uninstrument()
        cohere.ClientV2.chat = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    assert llm_events[0]["data"]["provider"] == "cohere"
    assert llm_events[0]["data"]["model"] == "command-light"
    assert "input_tokens" not in llm_events[0]["data"]
