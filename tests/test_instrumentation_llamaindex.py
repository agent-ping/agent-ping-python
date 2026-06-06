"""LlamaIndex native handler tests.

We construct CBEventType payloads directly and invoke the handler's
event lifecycle methods, the same way LlamaIndex's CallbackManager
would dispatch them during a real query."""

import json
import threading

import httpx
import pytest

pytest.importorskip("llama_index.core")

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


def _events_for(captured, rid, event_type):
    flat = []
    for c in captured:
        if f"/v1/runs/{rid}/events" in c["url"]:
            flat.extend(c["body"]["events"])
    return [e for e in flat if e["type"] == event_type]


def test_llm_event_emits_llm_call_with_openai_usage():
    from llama_index.core.callbacks import CBEventType, EventPayload

    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    handler = agentping.AgentPingLlamaIndexHandler()

    class FakeResponse:
        raw = {"usage": {"prompt_tokens": 220, "completion_tokens": 80}}

    with agentping.run("rag-query") as r:
        rid = r.id
        handler.on_event_start(
            CBEventType.LLM,
            payload={EventPayload.SERIALIZED: {"model": "gpt-4o-mini"}},
            event_id="ev1",
        )
        handler.on_event_end(
            CBEventType.LLM,
            payload={
                EventPayload.MODEL_NAME: "gpt-4o-mini",
                EventPayload.RESPONSE: FakeResponse(),
            },
            event_id="ev1",
        )
    agentping.flush(timeout=2.0)

    events = _events_for(captured, rid, "llm_call")
    assert len(events) == 1
    assert events[0]["data"]["provider"] == "openai"
    assert events[0]["data"]["model"] == "gpt-4o-mini"
    assert events[0]["data"]["input_tokens"] == 220
    assert events[0]["data"]["output_tokens"] == 80
    assert "latency_ms" in events[0]["data"]


def test_llm_event_reads_anthropic_usage_shape():
    from llama_index.core.callbacks import CBEventType, EventPayload

    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    class FakeResponse:
        raw = {
            "usage": {"input_tokens": 1000, "output_tokens": 50},
            "model": "claude-sonnet-4-5",
        }

    handler = agentping.AgentPingLlamaIndexHandler()
    with agentping.run("classify") as r:
        rid = r.id
        handler.on_event_start(CBEventType.LLM, payload={}, event_id="ev2")
        handler.on_event_end(
            CBEventType.LLM,
            payload={EventPayload.RESPONSE: FakeResponse()},
            event_id="ev2",
        )
    agentping.flush(timeout=2.0)

    events = _events_for(captured, rid, "llm_call")
    assert len(events) == 1
    assert events[0]["data"]["provider"] == "anthropic"
    assert events[0]["data"]["model"] == "claude-sonnet-4-5"
    assert events[0]["data"]["input_tokens"] == 1000
    assert events[0]["data"]["output_tokens"] == 50


def test_llm_event_captures_cached_input_tokens():
    from llama_index.core.callbacks import CBEventType, EventPayload

    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    class FakeResponse:
        raw = {
            "usage": {
                "prompt_tokens": 1500,
                "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 600},
            },
            "model": "gpt-4o",
        }

    handler = agentping.AgentPingLlamaIndexHandler()
    with agentping.run("cache-test") as r:
        rid = r.id
        handler.on_event_start(CBEventType.LLM, payload={}, event_id="ev3")
        handler.on_event_end(
            CBEventType.LLM,
            payload={EventPayload.RESPONSE: FakeResponse()},
            event_id="ev3",
        )
    agentping.flush(timeout=2.0)

    events = _events_for(captured, rid, "llm_call")
    assert events[0]["data"]["input_tokens"] == 1500
    assert events[0]["data"]["cached_input_tokens"] == 600


def test_embedding_event_emits_llm_call_with_embedding_kind():
    from llama_index.core.callbacks import CBEventType, EventPayload

    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    handler = agentping.AgentPingLlamaIndexHandler()
    with agentping.run("embed") as r:
        rid = r.id
        handler.on_event_start(
            CBEventType.EMBEDDING,
            payload={EventPayload.MODEL_NAME: "text-embedding-3-small"},
            event_id="emb1",
        )
        handler.on_event_end(
            CBEventType.EMBEDDING,
            payload={
                EventPayload.MODEL_NAME: "text-embedding-3-small",
                EventPayload.CHUNKS: ["a chunk of text " * 10, "another chunk " * 5],
            },
            event_id="emb1",
        )
    agentping.flush(timeout=2.0)

    events = _events_for(captured, rid, "llm_call")
    assert len(events) == 1
    assert events[0]["data"]["kind"] == "embedding"
    assert events[0]["data"]["provider"] == "openai"
    assert events[0]["data"]["model"] == "text-embedding-3-small"
    assert events[0]["data"]["input_tokens"] > 0


def test_retrieve_event_emits_retrieve_with_node_count():
    from llama_index.core.callbacks import CBEventType, EventPayload

    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    handler = agentping.AgentPingLlamaIndexHandler()
    with agentping.run("retrieve") as r:
        rid = r.id
        handler.on_event_start(CBEventType.RETRIEVE, payload={}, event_id="rt1")
        handler.on_event_end(
            CBEventType.RETRIEVE,
            payload={EventPayload.NODES: ["n1", "n2", "n3", "n4"]},
            event_id="rt1",
        )
    agentping.flush(timeout=2.0)

    events = _events_for(captured, rid, "retrieve")
    assert len(events) == 1
    assert events[0]["data"]["node_count"] == 4


def test_exception_event_emits_error():
    from llama_index.core.callbacks import CBEventType

    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    handler = agentping.AgentPingLlamaIndexHandler()
    with agentping.run("faulty") as r:
        rid = r.id
        handler.on_event_start(CBEventType.EXCEPTION, payload={}, event_id="ex1")
        handler.on_event_end(
            CBEventType.EXCEPTION,
            payload={"exception": RuntimeError("retriever timed out")},
            event_id="ex1",
        )
    agentping.flush(timeout=2.0)

    events = _events_for(captured, rid, "error")
    assert len(events) == 1
    assert "retriever timed out" in events[0]["data"]["message"]


def test_handler_resolves_active_run_when_constructed_with_no_arg():
    from llama_index.core.callbacks import CBEventType, EventPayload

    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    class FakeResponse:
        raw = {"usage": {"prompt_tokens": 12, "completion_tokens": 4}}

    handler = agentping.AgentPingLlamaIndexHandler()  # no run arg
    with agentping.run("scoped-li") as r:
        rid = r.id
        handler.on_event_start(CBEventType.LLM, payload={}, event_id="ev1")
        handler.on_event_end(
            CBEventType.LLM,
            payload={EventPayload.MODEL_NAME: "gpt-4o", EventPayload.RESPONSE: FakeResponse()},
            event_id="ev1",
        )
    agentping.flush(timeout=2.0)

    events = _events_for(captured, rid, "llm_call")
    assert events
    assert events[0]["data"]["model"] == "gpt-4o"
    assert events[0]["data"]["provider"] == "openai"
