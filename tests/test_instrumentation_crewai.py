import json
import os
import threading

import httpx
import pytest

pytest.importorskip("crewai")

os.environ.setdefault("OPENAI_API_KEY", "x")

import agentping
from agentping.instrumentation import _crewai as cw_inst


def _make_crew():
    """Construct a minimal Crew. CrewAI validates non-empty agents+tasks."""
    from crewai import Agent, Crew, Task
    agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
    task = Task(description="d", expected_output="o", agent=agent)
    return Crew(agents=[agent], tasks=[task])


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


class _FakeMetrics:
    def __init__(self, prompt_tokens=1500, completion_tokens=600, cached_prompt_tokens=0, successful_requests=4):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_prompt_tokens = cached_prompt_tokens
        self.successful_requests = successful_requests


def test_instrument_crewai_emits_llm_call_from_usage_metrics():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from crewai import Crew
    original = Crew.kickoff

    def fake_kickoff(self, *a, **kw):
        # Simulate CrewAI populating usage_metrics during the run.
        self.usage_metrics = _FakeMetrics(prompt_tokens=2000, completion_tokens=850, successful_requests=6)
        return "ok"

    Crew.kickoff = fake_kickoff

    try:
        agentping.instrument_crewai()
        crew = _make_crew()
        with agentping.run("market-brief") as r:
            rid = r.id
            crew.kickoff()
        agentping.flush(timeout=2.0)
    finally:
        cw_inst.uninstrument()
        Crew.kickoff = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["provider"] == "crewai"
    assert data["input_tokens"] == 2000
    assert data["output_tokens"] == 850
    assert data["request_count"] == 6
    assert "latency_ms" in data


def test_instrument_crewai_reports_gross_prompt_tokens_with_cached_split():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from crewai import Crew
    original = Crew.kickoff

    def fake_kickoff(self, *a, **kw):
        self.usage_metrics = _FakeMetrics(prompt_tokens=3000, completion_tokens=400, cached_prompt_tokens=1200, successful_requests=3)
        return "ok"

    Crew.kickoff = fake_kickoff

    try:
        agentping.instrument_crewai()
        crew = _make_crew()
        with agentping.run("market-brief") as r:
            rid = r.id
            crew.kickoff()
        agentping.flush(timeout=2.0)
    finally:
        cw_inst.uninstrument()
        Crew.kickoff = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    data = llm_events[0]["data"]
    # prompt_tokens is gross (the cached prefix is part of it); ingest
    # subtracts cached_input_tokens itself, so nothing is subtracted here.
    assert data["input_tokens"] == 3000
    assert data["cached_input_tokens"] == 1200
    assert data["request_count"] == 3


def test_instrument_crewai_emits_errored_llm_call_and_reraises():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from crewai import Crew
    original = Crew.kickoff

    def fake_kickoff(self, *a, **kw):
        raise RuntimeError("agent loop exhausted")

    Crew.kickoff = fake_kickoff

    try:
        agentping.instrument_crewai()
        crew = _make_crew()
        with agentping.run("market-brief") as r:
            rid = r.id
            with pytest.raises(RuntimeError):
                crew.kickoff()
        agentping.flush(timeout=2.0)
    finally:
        cw_inst.uninstrument()
        Crew.kickoff = original

    event_calls = [c for c in captured if f"/v1/runs/{rid}/events" in c["url"]]
    flat = []
    for c in event_calls:
        flat.extend(c["body"]["events"])
    llm_events = [e for e in flat if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    data = llm_events[0]["data"]
    assert data["status"] == "error"
    assert data["provider"] == "crewai"
    assert data["error"] == "agent loop exhausted"
    assert data["exception"] == "RuntimeError"


def test_instrument_crewai_is_idempotent():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))

    from crewai import Crew
    original = Crew.kickoff
    try:
        agentping.instrument_crewai()
        first = Crew.kickoff
        agentping.instrument_crewai()
        second = Crew.kickoff
        assert first is second
    finally:
        cw_inst.uninstrument()
        Crew.kickoff = original
