import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("agents")

import agentping
from agentping.instrumentation import _openai_agents as oaa_inst


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


class _FakeUsage:
    def __init__(self, input_tokens=120, output_tokens=40, cached_tokens=0, reasoning_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = SimpleNamespace(cached_tokens=cached_tokens)
        self.output_tokens_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)


class _FakeResponse:
    def __init__(self, usage=None, output=None):
        self.usage = usage or _FakeUsage()
        self.output = output or []


class _FakeAgent:
    def __init__(self, name="triage", model="gpt-4o-mini"):
        self.name = name
        self.model = model


class _FakeToolContext:
    """Shape of the ToolContext the Agents SDK passes to tool hooks for
    function tools."""

    def __init__(self, tool_name, tool_call_id, tool_arguments):
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.tool_arguments = tool_arguments


def test_agentping_hooks_emits_llm_call_on_llm_end():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from agentping.instrumentation._openai_agents import AgentPingHooks

    hooks = AgentPingHooks()
    agent = _FakeAgent()
    response = _FakeResponse(
        _FakeUsage(input_tokens=400, output_tokens=150, cached_tokens=256, reasoning_tokens=12),
        output=[SimpleNamespace(type="function_call"), SimpleNamespace(type="message")],
    )

    async def drive():
        with agentping.run("triage") as r:
            rid = r.id
            await hooks.on_llm_start(context="ctx", agent=agent, system_prompt=None, input_items=[])
            await hooks.on_llm_end(context="ctx", agent=agent, response=response)
            return rid

    rid = asyncio.run(drive())
    agentping.flush(timeout=2.0)

    llm_events = _events(captured, rid, "llm_call")
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o-mini"
    assert data["input_tokens"] == 400  # Responses API count, already gross
    assert data["output_tokens"] == 150
    assert data["cached_input_tokens"] == 256
    assert data["reasoning_tokens"] == 12
    assert data["tool_calls"] == 1
    assert "latency_ms" in data


def test_agentping_hooks_infers_provider_from_litellm_prefix():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from agentping.instrumentation._openai_agents import AgentPingHooks

    hooks = AgentPingHooks()
    agent = _FakeAgent(model="litellm/anthropic/claude-sonnet-4-5")

    async def drive():
        with agentping.run("triage") as r:
            rid = r.id
            await hooks.on_llm_end(context="ctx", agent=agent, response=_FakeResponse())
            return rid

    rid = asyncio.run(drive())
    agentping.flush(timeout=2.0)

    data = _events(captured, rid, "llm_call")[0]["data"]
    assert data["provider"] == "anthropic"
    assert data["model"] == "litellm/anthropic/claude-sonnet-4-5"


def test_agentping_hooks_emits_tool_call_handoff_and_agent_steps():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from agentping.instrumentation._openai_agents import AgentPingHooks

    hooks = AgentPingHooks()

    class _Tool:
        name = "fetch_orders"

    a = _FakeAgent(name="Triage")
    b = _FakeAgent(name="Billing")
    ctx = _FakeToolContext("fetch_orders", "call_abc", '{"customer_id": 42}')

    async def drive():
        with agentping.run("triage") as r:
            rid = r.id
            await hooks.on_agent_start(context="run", agent=a)
            await hooks.on_tool_start(context=ctx, agent=a, tool=_Tool())
            await hooks.on_tool_end(context=ctx, agent=a, tool=_Tool(), result={"orders": [1, 2]})
            await hooks.on_handoff(context="run", from_agent=a, to_agent=b)
            await hooks.on_agent_end(context="run", agent=a, output="done")
            return rid

    rid = asyncio.run(drive())
    agentping.flush(timeout=2.0)

    tool_events = _events(captured, rid, "tool_call")
    assert len(tool_events) == 1
    tool = tool_events[0]["data"]
    assert tool["tool"] == "fetch_orders"
    assert tool["status"] == "success"
    assert tool["tool_invocation_id"] == "call_abc"
    assert tool["input"] == '{"customer_id": 42}'
    assert json.loads(tool["output"]) == {"orders": [1, 2]}
    assert "latency_ms" in tool

    steps = [e["data"] for e in _events(captured, rid, "step")]
    assert {"kind": "handoff", "from": "Triage", "to": "Billing"} in steps
    agent_steps = [s for s in steps if s["kind"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["agent"] == "Triage"
    assert "latency_ms" in agent_steps[0]


def test_agentping_hooks_tool_call_without_tool_context():
    """Hosted tools call the hooks with a plain RunContextWrapper, so there
    is no call id or argument string to report."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from agentping.instrumentation._openai_agents import AgentPingHooks

    hooks = AgentPingHooks()

    class _Tool:
        name = "web_search"

    a = _FakeAgent(name="Research")

    async def drive():
        with agentping.run("research") as r:
            rid = r.id
            await hooks.on_tool_start(context="run", agent=a, tool=_Tool())
            await hooks.on_tool_end(context="run", agent=a, tool=_Tool(), result="3 results")
            return rid

    rid = asyncio.run(drive())
    agentping.flush(timeout=2.0)

    tool = _events(captured, rid, "tool_call")[0]["data"]
    assert tool["tool"] == "web_search"
    assert "tool_invocation_id" not in tool
    assert "input" not in tool
    assert tool["output"] == "3 results"


def test_instrument_openai_agents_auto_attaches_hooks():
    """instrument_openai_agents() patches Runner.run to inject AgentPingHooks when no hooks were passed."""
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))

    from agents import Runner
    seen_hooks = {}

    async def fake_run(starting_agent, input, *, context=None, max_turns=10, hooks=None, run_config=None, **kw):
        seen_hooks["hooks"] = hooks
        return "ok"

    original = Runner.run
    Runner.run = staticmethod(fake_run)

    try:
        agentping.instrument_openai_agents()
        # No hooks passed; the wrapper should inject AgentPingHooks.
        asyncio.run(Runner.run(starting_agent=_FakeAgent(), input="hi"))
        assert seen_hooks["hooks"].__class__.__name__ == "AgentPingHooks"
    finally:
        oaa_inst.uninstrument()
        Runner.run = staticmethod(original)
