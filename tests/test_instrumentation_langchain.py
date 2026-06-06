"""LangChain native handler tests.

Constructs LLMResult and runs callbacks directly. We don't need a real
chain — the handler is a passive consumer of the BaseCallbackHandler
protocol, and the protocol is what we contract against."""

import json
import threading
from uuid import uuid4

import httpx
import pytest

pytest.importorskip("langchain_core")

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


def _collect_events(captured, rid, event_type):
    flat = []
    for c in captured:
        if f"/v1/runs/{rid}/events" in c["url"]:
            flat.extend(c["body"]["events"])
    return [e for e in flat if e["type"] == event_type]


def test_handler_emits_llm_call_with_provider_inference():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from langchain_core.outputs import LLMResult

    handler = agentping.AgentPingCallbackHandler()
    run_id = uuid4()

    with agentping.run("rag-pipeline") as r:
        rid = r.id
        handler.on_chat_model_start(
            serialized={"kwargs": {"model_name": "gpt-4o-mini"}},
            messages=[],
            run_id=run_id,
        )
        result = LLMResult(
            generations=[[]],
            llm_output={
                "model_name": "gpt-4o-mini",
                "token_usage": {"prompt_tokens": 220, "completion_tokens": 80},
            },
        )
        handler.on_llm_end(result, run_id=run_id)
    agentping.flush(timeout=2.0)

    events = _collect_events(captured, rid, "llm_call")
    assert len(events) == 1
    assert events[0]["data"]["provider"] == "openai"
    assert events[0]["data"]["model"] == "gpt-4o-mini"
    assert events[0]["data"]["input_tokens"] == 220
    assert events[0]["data"]["output_tokens"] == 80
    assert "latency_ms" in events[0]["data"]


def test_handler_infers_provider_from_claude_model_name():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from langchain_core.outputs import LLMResult

    handler = agentping.AgentPingCallbackHandler()
    run_id = uuid4()
    with agentping.run("classify") as r:
        rid = r.id
        handler.on_chat_model_start(
            serialized={"kwargs": {"model": "claude-sonnet-4-5"}},
            messages=[],
            run_id=run_id,
        )
        result = LLMResult(
            generations=[[]],
            llm_output={"model_name": "claude-sonnet-4-5", "token_usage": {"input_tokens": 100, "output_tokens": 25}},
        )
        handler.on_llm_end(result, run_id=run_id)
    agentping.flush(timeout=2.0)

    events = _collect_events(captured, rid, "llm_call")
    assert events[0]["data"]["provider"] == "anthropic"
    assert events[0]["data"]["model"] == "claude-sonnet-4-5"


def test_handler_emits_tool_call():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    handler = agentping.AgentPingCallbackHandler()
    with agentping.run("tool-using") as r:
        rid = r.id
        handler.on_tool_start(
            serialized={"name": "search_db"},
            input_str='{"q": "pricing"}',
            run_id=uuid4(),
        )
    agentping.flush(timeout=2.0)

    events = _collect_events(captured, rid, "tool_call")
    assert events[0]["data"]["tool"] == "search_db"
    assert "pricing" in events[0]["data"]["args_preview"]


def test_handler_emits_error_event_on_chain_error():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    handler = agentping.AgentPingCallbackHandler()
    with agentping.run("failing-chain") as r:
        rid = r.id
        handler.on_chain_error(RuntimeError("upstream rate limit hit"), run_id=uuid4())
    agentping.flush(timeout=2.0)

    events = _collect_events(captured, rid, "error")
    assert "rate limit" in events[0]["data"]["message"]


def test_handler_resolves_active_run_when_constructed_with_no_arg():
    """The handler's no-arg constructor must pick up the run from the ContextVar
    set by `agentping.run(...)`'s context manager. This is the default DX."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from langchain_core.outputs import LLMResult

    handler = agentping.AgentPingCallbackHandler()  # NO run arg
    with agentping.run("scoped") as r:
        rid = r.id
        result = LLMResult(
            generations=[[]],
            llm_output={"model_name": "gpt-4o", "token_usage": {"prompt_tokens": 10, "completion_tokens": 4}},
        )
        handler.on_llm_end(result, run_id=uuid4())
    agentping.flush(timeout=2.0)

    events = _collect_events(captured, rid, "llm_call")
    assert events
    assert events[0]["data"]["model"] == "gpt-4o"
