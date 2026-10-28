"""Opt-in prompt/completion capture and time to first token on the framework
integrations: LangChain, LlamaIndex, CrewAI, Pydantic AI, OpenAI Agents.

The handlers are driven with the callback payloads the frameworks send; no
real model is called."""

import asyncio
import json
import os
import threading
from types import SimpleNamespace
from uuid import uuid4

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


def _llm_calls(captured, run_id):
    flat = []
    for c in captured:
        if f"/v1/runs/{run_id}/events" in c["url"]:
            flat.extend(c["body"]["events"])
    return [e["data"] for e in flat if e["type"] == "llm_call"]


def _assert_no_content(data):
    assert "input" not in data
    assert "output" not in data
    assert "system_instructions" not in data


# -- LangChain -----------------------------------------------------------------


def _langchain_messages():
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    return [
        SystemMessage("Be terse."),
        HumanMessage("Refund request from Acme"),
        AIMessage("", tool_calls=[{"name": "lookup", "args": {"q": "acme"}, "id": "call_1"}]),
        ToolMessage('{"rows": 3}', tool_call_id="call_1"),
    ]


def _langchain_result(text="Route to billing."):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    message = AIMessage(
        content=text,
        usage_metadata={"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
        response_metadata={"model_name": "claude-haiku-5-5"},
    )
    return LLMResult(generations=[[ChatGeneration(message=message, generation_info={"finish_reason": "stop"})]], llm_output={})


def _langchain_call(tokens=None, prompts=None, **init_kwargs):
    pytest.importorskip("langchain_core")
    captured, transport = _capture()
    _init(transport, **init_kwargs)
    handler = agentping.AgentPingCallbackHandler()
    run_id = uuid4()
    with agentping.run("triage") as r:
        rid = r.id
        if prompts is not None:
            handler.on_llm_start(serialized={"kwargs": {"model": "claude-haiku-5-5"}}, prompts=prompts, run_id=run_id)
        else:
            handler.on_chat_model_start(
                serialized={"kwargs": {"model": "claude-haiku-5-5"}},
                messages=[_langchain_messages()],
                run_id=run_id,
                metadata={"ls_provider": "anthropic", "ls_model_name": "claude-haiku-5-5"},
            )
        for token, chunk in tokens or []:
            handler.on_llm_new_token(token, chunk=chunk, run_id=run_id)
        handler.on_llm_end(_langchain_result(), run_id=run_id)
    agentping.flush(timeout=2.0)
    calls = _llm_calls(captured, rid)
    assert len(calls) == 1
    return calls[0]


def test_langchain_content_is_not_sent_by_default():
    data = _langchain_call()

    _assert_no_content(data)
    assert "ttft_ms" not in data and "stream" not in data, "not a stream"
    assert data["input_tokens"] == 30, "existing fields unchanged"


def test_langchain_capture_maps_message_types_and_splits_system_off():
    data = _langchain_call(capture_llm_content=True)

    assert data["system_instructions"] == "Be terse."
    assert [m["role"] for m in data["input"]] == ["user", "assistant", "tool"]
    assert data["input"][0]["content"] == "Refund request from Acme"
    assert data["input"][2]["content"] == '{"rows": 3}'
    assert data["output"] == "Route to billing."
    assert "ttft_ms" not in data


def test_langchain_completion_prompt_is_captured_as_a_user_message():
    data = _langchain_call(prompts=["Summarise: Acme wants a refund"], capture_llm_content=True)

    assert data["input"] == [{"role": "user", "content": "Summarise: Acme wants a refund"}]
    assert data["output"] == "Route to billing."


def test_langchain_capture_clips_to_the_configured_cap():
    data = _langchain_call(capture_llm_content=True, llm_content_max_chars=10)

    assert data["input"][0]["content"] == "Refund req..."
    assert data["output"] == "Route to b..."


def test_langchain_stream_reports_ttft_from_the_first_content_token():
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    housekeeping = ChatGenerationChunk(message=AIMessageChunk(content=""))
    data = _langchain_call(
        tokens=[("", housekeeping), ("Route ", None), ("to billing.", None)],
        capture_llm_content=True,
    )

    assert data["stream"] is True
    assert isinstance(data["ttft_ms"], int) and 0 <= data["ttft_ms"] <= data["latency_ms"]
    assert data["output"] == "Route to billing."


def test_langchain_stream_counts_a_tool_call_chunk_as_content():
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    tool_chunk = ChatGenerationChunk(
        message=AIMessageChunk(content="", tool_call_chunks=[{"name": "lookup", "args": "", "id": "call_1", "index": 0}])
    )
    data = _langchain_call(tokens=[("", tool_chunk)])

    assert data["stream"] is True
    assert isinstance(data["ttft_ms"], int)


def test_langchain_stream_of_only_empty_tokens_has_no_ttft():
    data = _langchain_call(tokens=[("", None), ("", None)])

    assert data["stream"] is True
    assert "ttft_ms" not in data


# -- LlamaIndex ----------------------------------------------------------------


def _llamaindex_call(end_payload, **init_kwargs):
    pytest.importorskip("llama_index.core")
    from llama_index.core.callbacks import CBEventType, EventPayload

    captured, transport = _capture()
    _init(transport, **init_kwargs)
    handler = agentping.AgentPingLlamaIndexHandler()
    with agentping.run("rag") as r:
        rid = r.id
        handler.on_event_start(CBEventType.LLM, payload={EventPayload.SERIALIZED: {"model": "gpt-5"}}, event_id="ev1")
        handler.on_event_end(CBEventType.LLM, payload={EventPayload.MODEL_NAME: "gpt-5", **end_payload}, event_id="ev1")
    agentping.flush(timeout=2.0)
    calls = _llm_calls(captured, rid)
    assert len(calls) == 1
    return calls[0]


def _llamaindex_chat_payload():
    from llama_index.core.callbacks import EventPayload
    from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole

    response = ChatResponse(
        message=ChatMessage(role=MessageRole.ASSISTANT, content="Route to billing."),
        raw={"usage": {"prompt_tokens": 20, "completion_tokens": 4}},
    )
    return {
        EventPayload.MESSAGES: [
            ChatMessage(role=MessageRole.SYSTEM, content="Be terse."),
            ChatMessage(role=MessageRole.USER, content="Refund request from Acme"),
        ],
        EventPayload.RESPONSE: response,
    }


def test_llamaindex_content_is_not_sent_by_default():
    data = _llamaindex_call(_llamaindex_chat_payload())

    _assert_no_content(data)
    assert "ttft_ms" not in data
    assert data["input_tokens"] == 20


def test_llamaindex_capture_sends_chat_messages_system_and_response_text():
    data = _llamaindex_call(_llamaindex_chat_payload(), capture_llm_content=True)

    assert data["system_instructions"] == "Be terse."
    assert data["input"] == [{"role": "user", "content": "Refund request from Acme"}]
    assert data["output"] == "Route to billing."
    assert "ttft_ms" not in data, "the callback manager has no per-token hook"


def test_llamaindex_capture_sends_completion_prompt_and_text():
    from llama_index.core.callbacks import EventPayload
    from llama_index.core.llms import CompletionResponse

    data = _llamaindex_call(
        {
            EventPayload.PROMPT: "Summarise: Acme wants a refund",
            EventPayload.COMPLETION: CompletionResponse(text="Acme asks for a refund."),
        },
        capture_llm_content=True,
    )

    assert data["input"] == [{"role": "user", "content": "Summarise: Acme wants a refund"}]
    assert data["output"] == "Acme asks for a refund."
    assert "system_instructions" not in data


# -- CrewAI --------------------------------------------------------------------
# The crew-level payload builder is exercised directly: it only reads
# `usage_metrics` and the kickoff result, so the package need not be installed.


def _crewai_payload(**init_kwargs):
    from agentping.instrumentation import _crewai

    _init(httpx.MockTransport(lambda r: httpx.Response(202, json={})), **init_kwargs)
    metrics = SimpleNamespace(prompt_tokens=100, completion_tokens=40, cached_prompt_tokens=0, successful_requests=2)
    return _crewai._build_payload(metrics, 12, SimpleNamespace(raw="Final market brief."))


def test_crewai_output_is_not_sent_by_default():
    data = _crewai_payload()

    _assert_no_content(data)
    assert data["input_tokens"] == 100 and data["request_count"] == 2


def test_crewai_capture_sends_the_crew_result_as_output_only():
    data = _crewai_payload(capture_llm_content=True)

    assert data["output"] == "Final market brief."
    assert "input" not in data, "CrewAI exposes no per-call prompt at the kickoff level"
    assert "ttft_ms" not in data


def test_crewai_capture_tolerates_a_result_without_raw_text():
    from agentping.instrumentation import _crewai

    _init(httpx.MockTransport(lambda r: httpx.Response(202, json={})), capture_llm_content=True)

    assert "output" not in _crewai._build_payload(None, 5, SimpleNamespace(pydantic=None))
    assert _crewai._build_payload(None, 5, "plain string")["output"] == "plain string"


# -- Pydantic AI ---------------------------------------------------------------


def _pydantic_history():
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    return [
        ModelRequest(parts=[SystemPromptPart(content="Be terse."), UserPromptPart(content="Refund request from Acme")]),
        ModelResponse(parts=[TextPart(content="Checking."), ToolCallPart(tool_name="lookup", args={"q": "acme"})]),
        ModelRequest(parts=[ToolReturnPart(tool_name="lookup", content={"rows": 3}, tool_call_id="call_1")]),
        ModelResponse(parts=[TextPart(content="Route to billing.")]),
    ]


class _FakeRunResult:
    def __init__(self, output, history):
        self.output = output
        self._history = history
        self.usage = SimpleNamespace(input_tokens=200, output_tokens=30, cache_read_tokens=0, cache_write_tokens=0, requests=2, tool_calls=1)

    def all_messages(self):
        return list(self._history)


def _pydantic_call(result, **init_kwargs):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent

    from agentping.instrumentation import _pydantic_ai as pai_inst

    os.environ.setdefault("OPENAI_API_KEY", "x")
    captured, transport = _capture()
    _init(transport, **init_kwargs)
    original = Agent.run_sync
    Agent.run_sync = lambda self, *a, **kw: result
    try:
        agentping.instrument_pydantic_ai()
        agent = Agent("openai:gpt-5")
        with agentping.run("triage") as r:
            rid = r.id
            agent.run_sync("Refund request from Acme")
        agentping.flush(timeout=2.0)
    finally:
        pai_inst.uninstrument()
        Agent.run_sync = original
    calls = _llm_calls(captured, rid)
    assert len(calls) == 1
    return calls[0]


def test_pydantic_ai_content_is_not_sent_by_default():
    data = _pydantic_call(_FakeRunResult("Route to billing.", _pydantic_history()))

    _assert_no_content(data)
    assert "ttft_ms" not in data
    assert data["input_tokens"] == 200 and data["tool_calls"] == 1


def test_pydantic_ai_capture_renders_the_history_before_the_final_answer():
    data = _pydantic_call(_FakeRunResult("Route to billing.", _pydantic_history()), capture_llm_content=True)

    assert data["system_instructions"] == "Be terse."
    assert [m["role"] for m in data["input"]] == ["user", "assistant", "tool"]
    assert data["input"][0]["content"] == "Refund request from Acme"
    assert data["input"][1]["content"] == "Checking.\n[tool-call lookup]"
    assert json.loads(data["input"][2]["content"]) == {"rows": 3}
    assert data["output"] == "Route to billing."
    assert "ttft_ms" not in data, "run_sync returns after the whole loop"


def test_pydantic_ai_capture_stringifies_structured_output():
    data = _pydantic_call(_FakeRunResult({"team": "billing"}, _pydantic_history()[:1]), capture_llm_content=True)

    assert json.loads(data["output"]) == {"team": "billing"}
    assert data["input"] == [{"role": "user", "content": "Refund request from Acme"}]


# -- OpenAI Agents -------------------------------------------------------------


def _agents_response():
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=50, output_tokens=8,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
        output=[
            SimpleNamespace(type="reasoning", summary=[]),
            SimpleNamespace(type="message", role="assistant", content=[
                SimpleNamespace(type="output_text", text="Route "),
                SimpleNamespace(type="output_text", text="to billing."),
            ]),
        ],
    )


def _agents_input_items():
    return [
        {"role": "user", "content": "Refund request from Acme"},
        {"type": "function_call", "name": "lookup", "arguments": '{"q": "acme"}', "call_id": "call_1"},
        {"type": "function_call_output", "call_id": "call_1", "output": '{"rows": 3}'},
        {"type": "reasoning", "summary": []},
        {"role": "assistant", "content": [{"type": "output_text", "text": "Checking."}]},
    ]


def _agents_call(**init_kwargs):
    pytest.importorskip("agents")
    from agentping.instrumentation._openai_agents import AgentPingHooks

    captured, transport = _capture()
    _init(transport, **init_kwargs)
    hooks = AgentPingHooks()
    agent = SimpleNamespace(name="triage", model="gpt-5")

    async def drive():
        with agentping.run("triage") as r:
            rid = r.id
            await hooks.on_llm_start(context="ctx", agent=agent, system_prompt="Be terse.", input_items=_agents_input_items())
            await hooks.on_llm_end(context="ctx", agent=agent, response=_agents_response())
            return rid

    rid = asyncio.run(drive())
    agentping.flush(timeout=2.0)
    calls = _llm_calls(captured, rid)
    assert len(calls) == 1
    return calls[0]


def test_openai_agents_content_is_not_sent_by_default():
    data = _agents_call()

    _assert_no_content(data)
    assert "ttft_ms" not in data
    assert data["input_tokens"] == 50


def test_openai_agents_capture_sends_input_items_system_prompt_and_output_text():
    data = _agents_call(capture_llm_content=True)

    assert data["system_instructions"] == "Be terse."
    assert data["input"] == [
        {"role": "user", "content": "Refund request from Acme"},
        {"role": "assistant", "content": "[function_call lookup]"},
        {"role": "tool", "content": '{"rows": 3}'},
        {"role": "assistant", "content": "Checking."},
    ]
    assert data["output"] == "Route to billing."
    assert "ttft_ms" not in data, "RunHooks has no per-token hook"
