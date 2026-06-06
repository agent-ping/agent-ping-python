import json
import threading

import httpx
import pytest

pytest.importorskip("mistralai.client.chat")

import agentping
from agentping.instrumentation import _mistral as m_inst


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
    def __init__(self, prompt_tokens=100, completion_tokens=50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, model="mistral-large-latest", usage=None):
        self.model = model
        self.usage = usage or _FakeUsage()


def test_instrument_mistral_emits_llm_call():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from mistralai.client.chat import Chat
    original = Chat.complete
    Chat.complete = lambda self, *a, **kw: _FakeResponse(
        usage=_FakeUsage(prompt_tokens=210, completion_tokens=88)
    )

    try:
        agentping.instrument_mistral()
        from mistralai.client.sdk import Mistral
        client = Mistral(api_key="x")
        with agentping.run("doc-translator") as r:
            rid = r.id
            client.chat.complete(model="mistral-large-latest", messages=[])
        agentping.flush(timeout=2.0)
    finally:
        m_inst.uninstrument()
        Chat.complete = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "mistral"
    assert data["model"] == "mistral-large-latest"
    assert data["input_tokens"] == 210
    assert data["output_tokens"] == 88
    assert "latency_ms" in data


def test_instrument_mistral_is_idempotent():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))

    from mistralai.client.chat import Chat
    original = Chat.complete
    try:
        agentping.instrument_mistral()
        first = Chat.complete
        agentping.instrument_mistral()
        second = Chat.complete
        assert first is second
    finally:
        m_inst.uninstrument()
        Chat.complete = original


def test_instrument_mistral_falls_back_to_response_model_field():
    """If the caller didn't pass model= as a kwarg, use the model from the response."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from mistralai.client.chat import Chat
    original = Chat.complete
    Chat.complete = lambda self, *a, **kw: _FakeResponse(model="mistral-small-latest")

    try:
        agentping.instrument_mistral()
        from mistralai.client.sdk import Mistral
        client = Mistral(api_key="x")
        with agentping.run("classify") as r:
            rid = r.id
            client.chat.complete(messages=[])  # no model kwarg
        agentping.flush(timeout=2.0)
    finally:
        m_inst.uninstrument()
        Chat.complete = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert llm_events[0]["data"]["model"] == "mistral-small-latest"
