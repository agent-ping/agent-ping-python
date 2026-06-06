import asyncio
import json
import threading

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


class _FakeUsage:
    def __init__(self, input_tokens=120, output_tokens=40):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, usage=None):
        self.usage = usage or _FakeUsage()


class _FakeAgent:
    def __init__(self, name="triage", model="gpt-4o-mini"):
        self.name = name
        self.model = model


def test_agentping_hooks_emits_llm_call_on_llm_end():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from agentping.instrumentation._openai_agents import AgentPingHooks

    hooks = AgentPingHooks()
    agent = _FakeAgent()
    response = _FakeResponse(_FakeUsage(input_tokens=400, output_tokens=150))

    async def drive():
        with agentping.run("triage") as r:
            rid = r.id
            await hooks.on_llm_start(context="ctx", agent=agent)
            await hooks.on_llm_end(context="ctx", agent=agent, response=response)
            return rid

    rid = asyncio.run(drive())
    agentping.flush(timeout=2.0)

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4o-mini"
    assert data["input_tokens"] == 400
    assert data["output_tokens"] == 150


def test_agentping_hooks_emits_tool_and_handoff_events():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from agentping.instrumentation._openai_agents import AgentPingHooks

    hooks = AgentPingHooks()

    class _Tool:
        name = "fetch_orders"

    a = _FakeAgent(name="Triage")
    b = _FakeAgent(name="Billing")

    async def drive():
        with agentping.run("triage") as r:
            rid = r.id
            await hooks.on_tool_start(context="ctx", agent=a, tool=_Tool())
            await hooks.on_handoff(context="ctx", from_agent=a, to_agent=b)
            return rid

    rid = asyncio.run(drive())
    agentping.flush(timeout=2.0)

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    tool_events = [e for e in flat if e["type"] == "tool_call"]
    handoff_events = [e for e in flat if e["type"] == "handoff"]
    assert tool_events and tool_events[0]["data"]["tool"] == "fetch_orders"
    assert handoff_events and handoff_events[0]["data"] == {"from": "Triage", "to": "Billing"}


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
