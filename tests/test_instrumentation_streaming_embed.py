"""Streaming + embedding coverage for Gemini, Mistral, Cohere wrappers."""

import json
import threading

import httpx
import pytest

import agentping
from agentping.instrumentation import _cohere as c_inst
from agentping.instrumentation import _gemini as gem_inst
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


def _collect_llm_events(captured, run_id):
    flat = []
    for c in captured:
        if f"/v1/runs/{run_id}/events" in c["url"]:
            flat.extend(c["body"]["events"])
    return [e for e in flat if e["type"] == "llm_call"]


# --- Gemini streaming -----------------------------------------------------

class _GeminiChunk:
    def __init__(self, usage=None, model_version="gemini-2.0-flash"):
        self.usage_metadata = usage
        self.model_version = model_version


class _GeminiUsage:
    def __init__(self, prompt_token_count=0, candidates_token_count=0, cached_content_token_count=0):
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.cached_content_token_count = cached_content_token_count


def test_gemini_stream_captures_final_usage():
    pytest.importorskip("google.genai")
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from google.genai.models import Models
    original = Models.generate_content_stream

    def fake_stream(self, *a, **kw):
        # 3 content chunks, then a final chunk with usage.
        yield _GeminiChunk()
        yield _GeminiChunk()
        yield _GeminiChunk(usage=_GeminiUsage(prompt_token_count=180, candidates_token_count=72))

    Models.generate_content_stream = fake_stream

    try:
        agentping.instrument_gemini()
        from google import genai
        client = genai.Client(api_key="x")
        with agentping.run("stream-research") as r:
            rid = r.id
            chunks = list(client.models.generate_content_stream(model="gemini-2.0-flash", contents="hi"))
            assert len(chunks) == 3
        agentping.flush(timeout=2.0)
    finally:
        gem_inst.uninstrument()
        Models.generate_content_stream = original

    events = _collect_llm_events(captured, rid)
    assert len(events) == 1
    assert events[0]["data"]["provider"] == "gemini"
    assert events[0]["data"]["model"] == "gemini-2.0-flash"
    assert events[0]["data"]["input_tokens"] == 180
    assert events[0]["data"]["output_tokens"] == 72


# --- Gemini embedding ----------------------------------------------------

class _GeminiEmbedResponse:
    def __init__(self, total=42):
        self.usage_metadata = type("U", (), {"total_token_count": total, "prompt_token_count": total})()


def test_gemini_embed_emits_llm_call_with_kind_embedding():
    pytest.importorskip("google.genai")
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from google.genai.models import Models
    original = Models.embed_content
    Models.embed_content = lambda self, *a, **kw: _GeminiEmbedResponse(total=120)

    try:
        agentping.instrument_gemini()
        from google import genai
        client = genai.Client(api_key="x")
        with agentping.run("rag-build") as r:
            rid = r.id
            client.models.embed_content(model="text-embedding-004", contents=["a", "b"])
        agentping.flush(timeout=2.0)
    finally:
        gem_inst.uninstrument()
        Models.embed_content = original

    events = _collect_llm_events(captured, rid)
    assert len(events) == 1
    assert events[0]["data"]["provider"] == "gemini"
    assert events[0]["data"]["model"] == "text-embedding-004"
    assert events[0]["data"]["kind"] == "embedding"
    assert events[0]["data"]["input_tokens"] == 120
    assert events[0]["data"]["output_tokens"] == 0


# --- Mistral streaming ---------------------------------------------------

class _MistralData:
    def __init__(self, usage=None, model="mistral-large-latest"):
        self.usage = usage
        self.model = model


class _MistralEvent:
    def __init__(self, data):
        self.data = data


class _MistralUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


def test_mistral_stream_captures_final_usage():
    pytest.importorskip("mistralai.client.chat")
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from mistralai.client.chat import Chat
    original = Chat.stream

    def fake_stream(self, *a, **kw):
        yield _MistralEvent(_MistralData())
        yield _MistralEvent(_MistralData(usage=_MistralUsage(prompt_tokens=310, completion_tokens=110)))

    Chat.stream = fake_stream

    try:
        agentping.instrument_mistral()
        from mistralai.client.sdk import Mistral
        client = Mistral(api_key="x")
        with agentping.run("translate-stream") as r:
            rid = r.id
            events = list(client.chat.stream(model="mistral-large-latest", messages=[]))
            assert len(events) == 2
        agentping.flush(timeout=2.0)
    finally:
        m_inst.uninstrument()
        Chat.stream = original

    events = _collect_llm_events(captured, rid)
    assert len(events) == 1
    assert events[0]["data"]["provider"] == "mistral"
    assert events[0]["data"]["model"] == "mistral-large-latest"
    assert events[0]["data"]["input_tokens"] == 310
    assert events[0]["data"]["output_tokens"] == 110


# --- Cohere streaming ----------------------------------------------------

class _CohereBilled:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _CohereDelta:
    def __init__(self, billed):
        self.usage = type("U", (), {"billed_units": billed})()


class _CohereEvent:
    def __init__(self, type, delta=None):
        self.type = type
        self.delta = delta


def test_cohere_chat_stream_captures_billed_units_from_message_end():
    pytest.importorskip("cohere")
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    import cohere
    original = cohere.ClientV2.chat_stream

    def fake_stream(self, *a, **kw):
        yield _CohereEvent("content-delta")
        yield _CohereEvent("content-delta")
        yield _CohereEvent("message-end", delta=_CohereDelta(_CohereBilled(input_tokens=290, output_tokens=105)))

    cohere.ClientV2.chat_stream = fake_stream

    try:
        agentping.instrument_cohere()
        client = cohere.ClientV2(api_key="x")
        with agentping.run("rfp-stream") as r:
            rid = r.id
            events = list(client.chat_stream(model="command-r-plus-08-2024", messages=[]))
            assert len(events) == 3
        agentping.flush(timeout=2.0)
    finally:
        c_inst.uninstrument()
        cohere.ClientV2.chat_stream = original

    events = _collect_llm_events(captured, rid)
    assert len(events) == 1
    assert events[0]["data"]["provider"] == "cohere"
    assert events[0]["data"]["model"] == "command-r-plus-08-2024"
    assert events[0]["data"]["input_tokens"] == 290
    assert events[0]["data"]["output_tokens"] == 105


# --- Cohere embed --------------------------------------------------------

class _CohereEmbedResponse:
    def __init__(self, input_tokens=300):
        self.meta = type("M", (), {"billed_units": _CohereBilled(input_tokens=input_tokens)})()


def test_cohere_embed_emits_llm_call_with_kind_embedding():
    pytest.importorskip("cohere")
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    import cohere
    original = cohere.ClientV2.embed
    cohere.ClientV2.embed = lambda self, *a, **kw: _CohereEmbedResponse(input_tokens=540)

    try:
        agentping.instrument_cohere()
        client = cohere.ClientV2(api_key="x")
        with agentping.run("rag-index") as r:
            rid = r.id
            client.embed(model="embed-english-v3.0", texts=["a", "b"], input_type="search_document")
        agentping.flush(timeout=2.0)
    finally:
        c_inst.uninstrument()
        cohere.ClientV2.embed = original

    events = _collect_llm_events(captured, rid)
    assert len(events) == 1
    assert events[0]["data"]["provider"] == "cohere"
    assert events[0]["data"]["model"] == "embed-english-v3.0"
    assert events[0]["data"]["kind"] == "embedding"
    assert events[0]["data"]["input_tokens"] == 540
    assert events[0]["data"]["output_tokens"] == 0
