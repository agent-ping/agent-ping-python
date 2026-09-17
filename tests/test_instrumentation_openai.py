import json
import threading
from types import SimpleNamespace

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


def _events(captured, rid, event_type):
    flat = []
    for c in captured:
        if f"/v1/runs/{rid}/events" in c["url"]:
            flat.extend(c["body"]["events"])
    return [e for e in flat if e["type"] == event_type]


def _completions_cls():
    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        from openai.resources.chat.completions.completions import Completions
    return Completions


def _responses_cls():
    try:
        from openai.resources.responses import Responses
    except Exception:
        from openai.resources.responses.responses import Responses
    return Responses


class _FakeUsage:
    """Chat Completions usage shape."""

    def __init__(self, prompt_tokens=100, completion_tokens=50, cached_tokens=0, reasoning_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt_tokens_details = SimpleNamespace(cached_tokens=cached_tokens)
        self.completion_tokens_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)


class _FakeResponse:
    def __init__(self, model="gpt-4o-mini", usage=None, finish_reason="stop", tool_calls=None):
        self.model = model
        self.usage = usage or _FakeUsage()
        self.choices = [SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(tool_calls=tool_calls or []),
        )]


def test_instrument_openai_emits_llm_call():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    Completions = _completions_cls()
    original = Completions.create
    Completions.create = lambda self, *a, **kw: _FakeResponse(
        usage=_FakeUsage(prompt_tokens=100, completion_tokens=50, cached_tokens=20, reasoning_tokens=7),
        finish_reason="tool_calls",
        tool_calls=[SimpleNamespace(id="call_1"), SimpleNamespace(id="call_2")],
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

    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o-mini"
    # prompt_tokens is already gross; ingest subtracts cached_input_tokens itself.
    assert data["input_tokens"] == 100
    assert data["cached_input_tokens"] == 20
    assert data["output_tokens"] == 50
    assert data["reasoning_tokens"] == 7
    assert data["finish_reason"] == "tool_calls"
    assert data["tool_calls"] == 2
    assert "latency_ms" in data


def test_instrument_openai_emits_errored_llm_call_and_reraises():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    Completions = _completions_cls()
    original = Completions.create

    def boom(self, *a, **kw):
        raise ValueError("rate limited")

    Completions.create = boom

    try:
        agentping.instrument_openai()
        client = openai.OpenAI(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            with pytest.raises(ValueError):
                client.chat.completions.create(model="gpt-4o-mini", messages=[])
        agentping.flush(timeout=2.0)
    finally:
        op_inst.uninstrument()
        Completions.create = original

    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["status"] == "error"
    assert data["model"] == "gpt-4o-mini"
    assert data["error"] == "rate limited"
    assert data["exception"] == "ValueError"


def test_instrument_openai_chat_stream_emits_one_llm_call_on_close():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    def chunks():
        yield SimpleNamespace(model="gpt-4o-mini", usage=None, choices=[
            SimpleNamespace(finish_reason=None, delta=SimpleNamespace(tool_calls=[SimpleNamespace(index=0)])),
        ])
        yield SimpleNamespace(model="gpt-4o-mini", usage=None, choices=[
            SimpleNamespace(finish_reason=None, delta=SimpleNamespace(tool_calls=[SimpleNamespace(index=0)])),
        ])
        yield SimpleNamespace(model="gpt-4o-mini", usage=None, choices=[
            SimpleNamespace(finish_reason="tool_calls", delta=SimpleNamespace(tool_calls=None)),
        ])
        # Final usage chunk (stream_options include_usage).
        yield SimpleNamespace(model="gpt-4o-mini", choices=[], usage=_FakeUsage(prompt_tokens=30, completion_tokens=12))

    Completions = _completions_cls()
    original = Completions.create
    Completions.create = lambda self, *a, **kw: chunks()

    try:
        agentping.instrument_openai()
        client = openai.OpenAI(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            stream = client.chat.completions.create(
                model="gpt-4o-mini", messages=[], stream=True, stream_options={"include_usage": True},
            )
            count = sum(1 for _ in stream)
        agentping.flush(timeout=2.0)
    finally:
        op_inst.uninstrument()
        Completions.create = original

    assert count == 4
    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["stream"] is True
    assert data["input_tokens"] == 30
    assert data["output_tokens"] == 12
    assert data["finish_reason"] == "tool_calls"
    assert data["tool_calls"] == 1  # same index across two deltas is one call


class _FakeResponsesUsage:
    def __init__(self, input_tokens=100, output_tokens=50, cached_tokens=0, reasoning_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = SimpleNamespace(cached_tokens=cached_tokens)
        self.output_tokens_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)


class _FakeResponsesResponse:
    def __init__(self, model="gpt-4.1-mini", usage=None, status="completed", output=None, incomplete_reason=None):
        self.model = model
        self.usage = usage or _FakeResponsesUsage()
        self.status = status
        self.output = output or []
        self.incomplete_details = SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None


def test_instrument_openai_responses_create_emits_llm_call():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    Responses = _responses_cls()
    original = Responses.create
    Responses.create = lambda self, *a, **kw: _FakeResponsesResponse(
        usage=_FakeResponsesUsage(input_tokens=200, output_tokens=40, cached_tokens=150, reasoning_tokens=11),
        output=[SimpleNamespace(type="reasoning"), SimpleNamespace(type="function_call"), SimpleNamespace(type="message")],
    )

    try:
        agentping.instrument_openai()
        client = openai.OpenAI(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            client.responses.create(model="gpt-4.1-mini", input="hi")
        agentping.flush(timeout=2.0)
    finally:
        op_inst.uninstrument()
        Responses.create = original

    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4.1-mini"
    assert data["input_tokens"] == 200
    assert data["cached_input_tokens"] == 150
    assert data["output_tokens"] == 40
    assert data["reasoning_tokens"] == 11
    assert data["finish_reason"] == "completed"
    assert data["tool_calls"] == 1


def test_instrument_openai_responses_stream_uses_terminal_event():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    final = _FakeResponsesResponse(
        usage=_FakeResponsesUsage(input_tokens=50, output_tokens=20),
        status="incomplete",
        incomplete_reason="max_output_tokens",
    )

    def events():
        yield SimpleNamespace(type="response.created", response=SimpleNamespace(model="gpt-4.1-mini"))
        yield SimpleNamespace(type="response.output_text.delta", delta="hi")
        yield SimpleNamespace(type="response.incomplete", response=final)

    Responses = _responses_cls()
    original = Responses.create
    Responses.create = lambda self, *a, **kw: events()

    try:
        agentping.instrument_openai()
        client = openai.OpenAI(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            for _ in client.responses.create(model="gpt-4.1-mini", input="hi", stream=True):
                pass
        agentping.flush(timeout=2.0)
    finally:
        op_inst.uninstrument()
        Responses.create = original

    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["stream"] is True
    assert data["input_tokens"] == 50
    assert data["output_tokens"] == 20
    assert data["finish_reason"] == "max_output_tokens"


def test_instrument_openai_is_idempotent():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))
    Completions = _completions_cls()

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
