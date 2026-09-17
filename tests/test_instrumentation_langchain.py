"""LangChain native handler tests.

Constructs LLMResult and runs callbacks directly. We don't need a real
chain: the handler is a passive consumer of the BaseCallbackHandler
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


def test_handler_reads_usage_metadata_finish_reason_and_tool_calls():
    """Modern chat models put gross usage on the AIMessage; the handler
    prefers it over the legacy llm_output token_usage block."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    handler = agentping.AgentPingCallbackHandler()
    run_id = uuid4()
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "lookup", "args": {"q": "a"}, "id": "call_1"},
            {"name": "lookup", "args": {"q": "b"}, "id": "call_2"},
        ],
        usage_metadata={
            "input_tokens": 300,
            "output_tokens": 40,
            "total_tokens": 340,
            "input_token_details": {"cache_read": 120, "cache_creation": 30},
            "output_token_details": {"reasoning": 9},
        },
        response_metadata={"model_name": "claude-sonnet-4-5"},
    )
    with agentping.run("agent") as r:
        rid = r.id
        handler.on_chat_model_start(
            serialized={"id": ["langchain", "chat_models", "anthropic", "ChatAnthropic"]},
            messages=[],
            run_id=run_id,
            metadata={"ls_provider": "anthropic", "ls_model_name": "claude-sonnet-4-5"},
        )
        result = LLMResult(
            generations=[[ChatGeneration(message=message, generation_info={"finish_reason": "tool_use"})]],
            llm_output={},
        )
        handler.on_llm_end(result, run_id=run_id)
    agentping.flush(timeout=2.0)

    data = _collect_events(captured, rid, "llm_call")[0]["data"]
    assert data["provider"] == "anthropic"
    assert data["model"] == "claude-sonnet-4-5"
    assert data["input_tokens"] == 300  # gross, as usage_metadata reports it
    assert data["output_tokens"] == 40
    assert data["cached_input_tokens"] == 120
    assert data["cache_creation_input_tokens"] == 30
    assert data["reasoning_tokens"] == 9
    assert data["finish_reason"] == "tool_use"
    assert data["tool_calls"] == 2


def test_handler_emits_errored_llm_call_on_llm_error():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    handler = agentping.AgentPingCallbackHandler()
    run_id = uuid4()
    with agentping.run("agent") as r:
        rid = r.id
        handler.on_chat_model_start(
            serialized={"kwargs": {"model_name": "gpt-4o-mini"}},
            messages=[],
            run_id=run_id,
            metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o-mini"},
        )
        handler.on_llm_error(TimeoutError("read timed out"), run_id=run_id)
    agentping.flush(timeout=2.0)

    events = _collect_events(captured, rid, "llm_call")
    assert len(events) == 1
    data = events[0]["data"]
    assert data["status"] == "error"
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o-mini"
    assert data["error"] == "read timed out"
    assert data["exception"] == "TimeoutError"


def test_handler_emits_tool_call_on_tool_end():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from langchain_core.messages import ToolMessage

    handler = agentping.AgentPingCallbackHandler()
    run_id = uuid4()
    with agentping.run("tool-using") as r:
        rid = r.id
        handler.on_tool_start(
            serialized={"name": "search_db"},
            input_str="{'q': 'pricing'}",
            run_id=run_id,
            inputs={"q": "pricing"},
        )
        handler.on_tool_end(
            ToolMessage(content='{"rows": 3}', tool_call_id="call_1"),
            run_id=run_id,
        )
    agentping.flush(timeout=2.0)

    events = _collect_events(captured, rid, "tool_call")
    assert len(events) == 1
    data = events[0]["data"]
    assert data["tool"] == "search_db"
    assert data["status"] == "success"
    assert data["tool_invocation_id"] == str(run_id)
    assert json.loads(data["input"]) == {"q": "pricing"}
    assert data["output"] == '{"rows": 3}'
    assert "latency_ms" in data


def test_handler_emits_errored_tool_call_on_tool_error():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    handler = agentping.AgentPingCallbackHandler()
    run_id = uuid4()
    with agentping.run("tool-using") as r:
        rid = r.id
        handler.on_tool_start(serialized={"name": "search_db"}, input_str="pricing", run_id=run_id)
        handler.on_tool_error(ConnectionError("db unreachable"), run_id=run_id)
    agentping.flush(timeout=2.0)

    data = _collect_events(captured, rid, "tool_call")[0]["data"]
    assert data["tool"] == "search_db"
    assert data["status"] == "error"
    assert data["input"] == "pricing"
    assert data["error"] == "db unreachable"
    assert data["exception"] == "ConnectionError"


def test_handler_can_switch_off_tool_payload_capture():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    handler = agentping.AgentPingCallbackHandler(capture_tool_payloads=False)
    run_id = uuid4()
    with agentping.run("tool-using") as r:
        rid = r.id
        handler.on_tool_start(serialized={"name": "search_db"}, input_str="secret", run_id=run_id)
        handler.on_tool_end("also secret", run_id=run_id)
    agentping.flush(timeout=2.0)

    data = _collect_events(captured, rid, "tool_call")[0]["data"]
    assert data["tool"] == "search_db"
    assert "input" not in data
    assert "output" not in data


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
    assert events[0]["data"]["exception"] == "RuntimeError"


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
