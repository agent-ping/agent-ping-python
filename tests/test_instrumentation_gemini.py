import json
import threading

import httpx
import pytest

genai = pytest.importorskip("google.genai")

import agentping
from agentping.instrumentation import _gemini as gem_inst


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
    def __init__(self, prompt_token_count=120, candidates_token_count=80, cached_content_token_count=0):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.cached_content_token_count = cached_content_token_count


class _FakeResponse:
    def __init__(self, usage=None):
        self.usage_metadata = usage or _FakeUsage()


def test_instrument_gemini_emits_llm_call():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from google.genai.models import Models
    original = Models.generate_content
    Models.generate_content = lambda self, *a, **kw: _FakeResponse(
        usage=_FakeUsage(prompt_token_count=200, candidates_token_count=60, cached_content_token_count=40)
    )

    try:
        agentping.instrument_gemini()
        client = genai.Client(api_key="x")
        with agentping.run("research") as r:
            rid = r.id
            client.models.generate_content(model="gemini-2.0-flash", contents="hi")
        agentping.flush(timeout=2.0)
    finally:
        gem_inst.uninstrument()
        Models.generate_content = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "gemini"
    assert data["model"] == "gemini-2.0-flash"
    # prompt_token_count already includes the cached content; ingest
    # subtracts cached_input_tokens itself, so the gross count is sent.
    assert data["input_tokens"] == 200
    assert data["cached_input_tokens"] == 40
    assert data["output_tokens"] == 60
    assert "latency_ms" in data


def test_instrument_gemini_is_idempotent():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))

    from google.genai.models import Models
    original = Models.generate_content
    try:
        agentping.instrument_gemini()
        first = Models.generate_content
        agentping.instrument_gemini()
        second = Models.generate_content
        assert first is second
    finally:
        gem_inst.uninstrument()
        Models.generate_content = original


def test_instrument_gemini_no_cache_block():
    """When the response has no cached_content_token_count, all tokens count as fresh input."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from google.genai.models import Models
    original = Models.generate_content
    Models.generate_content = lambda self, *a, **kw: _FakeResponse(
        usage=_FakeUsage(prompt_token_count=80, candidates_token_count=40, cached_content_token_count=0)
    )

    try:
        agentping.instrument_gemini()
        client = genai.Client(api_key="x")
        with agentping.run("research") as r:
            rid = r.id
            client.models.generate_content(model="gemini-1.5-flash", contents="hi")
        agentping.flush(timeout=2.0)
    finally:
        gem_inst.uninstrument()
        Models.generate_content = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    data = llm_events[0]["data"]
    assert data["input_tokens"] == 80
    assert "cached_input_tokens" not in data
