"""Time to first token and opt-in prompt capture for Gemini, Mistral, Cohere and Bedrock."""

import json
import threading
from types import SimpleNamespace

import httpx
import pytest

import agentping


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


def _init(transport, **kwargs):
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport, **kwargs)


def _llm_call(captured, run_id):
    flat = []
    for c in captured:
        if f"/v1/runs/{run_id}/events" in c["url"]:
            flat.extend(c["body"]["events"])
    calls = [e["data"] for e in flat if e["type"] == "llm_call"]
    assert len(calls) == 1
    return calls[0]


def _assert_ttft(data):
    assert data["stream"] is True
    assert isinstance(data["ttft_ms"], int) and 0 <= data["ttft_ms"] <= data["latency_ms"]


# -- Gemini ------------------------------------------------------------------


def _gemini_part(text):
    return SimpleNamespace(text=text, function_call=None)


def _gemini_response(text, usage=None, finish="STOP"):
    return SimpleNamespace(
        model_version="gemini-2.0-flash",
        usage_metadata=usage,
        candidates=[SimpleNamespace(finish_reason=finish, content=SimpleNamespace(parts=[_gemini_part(text)]))],
    )


def _gemini_call(method, factory, kwargs, **init_kwargs):
    genai = pytest.importorskip("google.genai")
    from google.genai.models import Models

    from agentping.instrumentation import _gemini as inst

    captured, transport = _capture()
    _init(transport, **init_kwargs)
    original = getattr(Models, method)
    setattr(Models, method, lambda self, *a, **kw: factory())
    try:
        agentping.instrument_gemini()
        client = genai.Client(api_key="x")
        with agentping.run("research") as r:
            rid = r.id
            result = getattr(client.models, method)(**kwargs)
            if method.endswith("_stream"):
                list(result)
        agentping.flush(timeout=2.0)
    finally:
        inst.uninstrument()
        setattr(Models, method, original)
    return _llm_call(captured, rid)


def test_gemini_capture_maps_contents_roles_and_system_instruction():
    data = _gemini_call(
        "generate_content",
        lambda: _gemini_response("Paris."),
        dict(
            model="gemini-2.0-flash",
            contents=[
                {"role": "user", "parts": [{"text": "Capital of France?"}]},
                {"role": "model", "parts": [{"text": "Thinking..."}, {"function_call": {"name": "lookup"}}]},
                {"role": "user", "parts": [{"text": "Well?"}]},
            ],
            config={"system_instruction": "Answer briefly."},
        ),
        capture_llm_content=True,
    )

    assert data["system_instructions"] == "Answer briefly."
    assert data["input"] == [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Thinking...\n[function_call lookup]"},
        {"role": "user", "content": "Well?"},
    ]
    assert data["output"] == "Paris."
    assert "ttft_ms" not in data


def test_gemini_capture_treats_a_bare_string_or_part_list_as_one_user_message():
    data = _gemini_call(
        "generate_content",
        lambda: _gemini_response("ok"),
        dict(model="gemini-2.0-flash", contents=["Summarise:", {"text": "the report"}]),
        capture_llm_content=True,
    )

    assert data["input"] == [{"role": "user", "content": "Summarise:\nthe report"}]


def test_gemini_stream_reports_ttft_and_joins_text_parts():
    def chunks():
        return iter([
            SimpleNamespace(model_version=None, usage_metadata=None, candidates=[]),
            _gemini_response("Par", finish=None),
            _gemini_response("is.", usage=SimpleNamespace(prompt_token_count=5, candidates_token_count=2)),
        ])

    data = _gemini_call(
        "generate_content_stream",
        chunks,
        dict(model="gemini-2.0-flash", contents="hi"),
        capture_llm_content=True,
    )

    _assert_ttft(data)
    assert data["output"] == "Paris."
    assert data["input"] == [{"role": "user", "content": "hi"}]
    assert data["input_tokens"] == 5


# -- Mistral -----------------------------------------------------------------


def _mistral_call(method, factory, kwargs, **init_kwargs):
    pytest.importorskip("mistralai")
    from mistralai.client.chat import Chat
    from mistralai.client.sdk import Mistral

    from agentping.instrumentation import _mistral as inst

    captured, transport = _capture()
    _init(transport, **init_kwargs)
    original = getattr(Chat, method)
    setattr(Chat, method, lambda self, *a, **kw: factory())
    try:
        agentping.instrument_mistral()
        client = Mistral(api_key="x")
        with agentping.run("translate") as r:
            rid = r.id
            result = getattr(client.chat, method)(**kwargs)
            if method == "stream":
                list(result)
        agentping.flush(timeout=2.0)
    finally:
        inst.uninstrument()
        setattr(Chat, method, original)
    return _llm_call(captured, rid)


def _mistral_response(content):
    return SimpleNamespace(
        model="mistral-large-latest",
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=4),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(tool_calls=None, content=content))],
    )


def test_mistral_sends_nothing_about_the_prompt_by_default():
    data = _mistral_call(
        "complete",
        lambda: _mistral_response("Bonjour"),
        dict(model="mistral-large-latest", messages=[{"role": "user", "content": "Hello"}]),
    )

    assert "input" not in data and "output" not in data and "system_instructions" not in data


def test_mistral_capture_sends_messages_and_text_chunks():
    data = _mistral_call(
        "complete",
        lambda: _mistral_response([SimpleNamespace(type="text", text="Bonjour"), SimpleNamespace(type="reference", reference_ids=[1])]),
        dict(
            model="mistral-large-latest",
            messages=[{"role": "system", "content": "Translate to French."}, {"role": "user", "content": "Hello"}],
        ),
        capture_llm_content=True,
    )

    assert data["input"] == [
        {"role": "system", "content": "Translate to French."},
        {"role": "user", "content": "Hello"},
    ]
    assert data["output"] == "Bonjour\n[reference]"


def test_mistral_stream_reports_ttft_and_captures_deltas():
    def event(content, usage=None, tool_calls=None):
        return SimpleNamespace(data=SimpleNamespace(
            model="mistral-large-latest",
            usage=usage,
            choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content=content, tool_calls=tool_calls))],
        ))

    def events():
        return iter([
            event(""),
            event("Bon"),
            event("jour", usage=SimpleNamespace(prompt_tokens=20, completion_tokens=4)),
        ])

    data = _mistral_call(
        "stream",
        events,
        dict(model="mistral-large-latest", messages=[{"role": "user", "content": "Hello"}]),
        capture_llm_content=True,
    )

    _assert_ttft(data)
    assert data["output"] == "Bonjour"
    assert data["input"] == [{"role": "user", "content": "Hello"}]


# -- Cohere ------------------------------------------------------------------


def _cohere_call(method, factory, kwargs, **init_kwargs):
    cohere = pytest.importorskip("cohere")

    from agentping.instrumentation import _cohere as inst

    captured, transport = _capture()
    _init(transport, **init_kwargs)
    original = getattr(cohere.ClientV2, method)
    setattr(cohere.ClientV2, method, lambda self, *a, **kw: factory())
    try:
        agentping.instrument_cohere()
        client = cohere.ClientV2(api_key="x")
        with agentping.run("rfp") as r:
            rid = r.id
            result = getattr(client, method)(**kwargs)
            if method == "chat_stream":
                list(result)
        agentping.flush(timeout=2.0)
    finally:
        inst.uninstrument()
        setattr(cohere.ClientV2, method, original)
    return _llm_call(captured, rid)


def test_cohere_capture_sends_messages_and_response_text():
    billed = SimpleNamespace(input_tokens=30, output_tokens=3)
    data = _cohere_call(
        "chat",
        lambda: SimpleNamespace(
            usage=SimpleNamespace(billed_units=billed),
            finish_reason="COMPLETE",
            message=SimpleNamespace(tool_calls=None, content=[SimpleNamespace(type="text", text="Yes, in scope.")]),
        ),
        dict(model="command-a-03-2025", messages=[{"role": "user", "content": "Is SSO in scope?"}]),
        capture_llm_content=True,
    )

    assert data["input"] == [{"role": "user", "content": "Is SSO in scope?"}]
    assert data["output"] == "Yes, in scope."
    assert data["input_tokens"] == 30


def test_cohere_stream_reports_ttft_from_content_start_and_joins_deltas():
    def delta(text):
        return SimpleNamespace(type="content-delta", delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text=text))))

    def events():
        return iter([
            SimpleNamespace(type="message-start"),
            SimpleNamespace(type="content-start"),
            delta("Yes, "),
            delta("in scope."),
            SimpleNamespace(type="content-end"),
            SimpleNamespace(type="message-end", delta=SimpleNamespace(
                usage=SimpleNamespace(billed_units=SimpleNamespace(input_tokens=30, output_tokens=3)),
                finish_reason="COMPLETE",
            )),
        ])

    data = _cohere_call(
        "chat_stream",
        events,
        dict(model="command-a-03-2025", messages=[{"role": "user", "content": "Is SSO in scope?"}]),
        capture_llm_content=True,
    )

    _assert_ttft(data)
    assert data["output"] == "Yes, in scope."
    assert data["finish_reason"] == "COMPLETE"


# -- Bedrock -----------------------------------------------------------------


def _bedrock_call(operation, factory, params, **init_kwargs):
    pytest.importorskip("botocore")
    from botocore.client import BaseClient

    from agentping.instrumentation import _bedrock as inst

    class _FakeMeta:
        service_model = SimpleNamespace(service_name="bedrock-runtime")

    class FakeBedrockClient(BaseClient):
        def __init__(self):
            self.meta = _FakeMeta()

    captured, transport = _capture()
    _init(transport, **init_kwargs)
    original = BaseClient._make_api_call
    BaseClient._make_api_call = lambda self, op, p: factory()
    try:
        agentping.instrument_bedrock()
        client = FakeBedrockClient()
        with agentping.run("support") as r:
            rid = r.id
            result = client._make_api_call(operation, params)
            if operation == "ConverseStream":
                list(result["stream"])
        agentping.flush(timeout=2.0)
    finally:
        inst.uninstrument()
        BaseClient._make_api_call = original
    return _llm_call(captured, rid)


_BEDROCK_PARAMS = {
    "modelId": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "system": [{"text": "Be terse."}],
    "messages": [
        {"role": "user", "content": [{"text": "Refund?"}, {"image": {"format": "png"}}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "lookup_order", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "ok"}]}}]},
    ],
}


def test_bedrock_converse_capture_renders_blocks_and_system():
    data = _bedrock_call(
        "Converse",
        lambda: {
            "output": {"message": {"role": "assistant", "content": [{"text": "Refund approved."}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 40, "outputTokens": 3},
        },
        _BEDROCK_PARAMS,
        capture_llm_content=True,
    )

    assert data["system_instructions"] == "Be terse."
    assert data["input"] == [
        {"role": "user", "content": "Refund?\n[image]"},
        {"role": "assistant", "content": "[toolUse lookup_order]"},
        {"role": "user", "content": "[toolResult]"},
    ]
    assert data["output"] == "Refund approved."
    assert "ttft_ms" not in data


def test_bedrock_converse_stream_reports_ttft_and_captures_text():
    def events():
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": "Refund "}}}
        yield {"contentBlockDelta": {"delta": {"text": "approved."}}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 40, "outputTokens": 3}}}

    data = _bedrock_call(
        "ConverseStream",
        lambda: {"stream": events()},
        _BEDROCK_PARAMS,
        capture_llm_content=True,
    )

    _assert_ttft(data)
    assert data["output"] == "Refund approved."
    assert data["system_instructions"] == "Be terse."
    assert data["input"][0] == {"role": "user", "content": "Refund?\n[image]"}
    assert data["finish_reason"] == "end_turn"


def test_bedrock_invoke_model_is_not_captured():
    data = _bedrock_call(
        "InvokeModel",
        lambda: {"ResponseMetadata": {"HTTPHeaders": {"x-amzn-bedrock-input-token-count": "7", "x-amzn-bedrock-output-token-count": "2"}}},
        {"modelId": "amazon.titan-text-lite-v1", "body": b"{}"},
        capture_llm_content=True,
    )

    assert data["input_tokens"] == 7
    assert "input" not in data and "output" not in data
