"""AgentPing Python SDK.

Public API:
    agentping.init(...)
    agentping.run(...)         -> Run
    agentping.heartbeat(...)
    agentping.guard.check(...) -> GuardVerdict   # raises Paused on block (hard mode)
    agentping.instrument_anthropic()
    agentping.instrument_openai()
    agentping.instrument_gemini()
    agentping.instrument_mistral()
    agentping.instrument_cohere()
    agentping.instrument_bedrock()
    agentping.instrument_pydantic_ai()
    agentping.instrument_crewai()
    agentping.instrument_openai_agents()
    agentping.AgentPingHooks              # for OpenAI Agents SDK
    agentping.AgentPingCallbackHandler    # for LangChain
    agentping.AgentPingLlamaIndexHandler  # for LlamaIndex
    agentping.status()         -> dict
    agentping.flush(timeout=5)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from . import _ids, _state, guard
from ._run import Run, _iso
from .guard import GuardVerdict, Paused

__version__ = "0.2.0"

__all__ = [
    "init",
    "run",
    "active_run",
    "guard",
    "GuardVerdict",
    "Paused",
    "heartbeat",
    "instrument_anthropic",
    "instrument_openai",
    "instrument_gemini",
    "instrument_mistral",
    "instrument_cohere",
    "instrument_bedrock",
    "instrument_pydantic_ai",
    "instrument_crewai",
    "instrument_openai_agents",
    "AgentPingHooks",
    "AgentPingCallbackHandler",
    "AgentPingLlamaIndexHandler",
    "status",
    "flush",
    "Run",
]


def init(
    api_key: str | None = None,
    base_url: str | None = None,
    queue_size: int = _state.DEFAULT_QUEUE_SIZE,
    transport: httpx.BaseTransport | None = None,
    control_url: str | None = None,
) -> None:
    _state.init(
        api_key=api_key,
        base_url=base_url,
        queue_size=queue_size,
        transport=transport,
        control_url=control_url,
    )


def run(
    agent: str,
    customer_id: str | None = None,
    feature: str | None = None,
    metadata: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
    goal: str | None = None,
) -> Run:
    return Run(
        agent=agent,
        customer_id=customer_id,
        feature=feature,
        metadata=metadata,
        parent_run_id=parent_run_id,
        goal=goal,
    )


def active_run() -> Run | None:
    """The run bound to the current context (thread or asyncio task), if any.

    Set by `with agentping.run(...)` / `async with agentping.run(...)`.
    Integration packages use this to attach events to the caller's run
    instead of inventing their own.
    """
    from .instrumentation._streaming import _active_run

    return _active_run()


def heartbeat(
    agent: str,
    status: str = "ok",
    cost_usd: float | None = None,
    duration_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
) -> str:
    s = _state.state()
    region = s.region or "eu"
    run_id = _ids.new_id("run", region)

    now = datetime.now(timezone.utc)
    if finished_at is None:
        finished_at = now
    if started_at is None and duration_ms is not None:
        started_at = datetime.fromtimestamp(finished_at.timestamp() - (duration_ms / 1000.0), tz=timezone.utc)
    if started_at is None:
        started_at = finished_at

    body: dict[str, Any] = {
        "id": run_id,
        "agent": agent,
        "status": status,
        "started_at": _iso(started_at),
        "finished_at": _iso(finished_at),
    }
    if cost_usd is not None:
        body["cost_usd"] = cost_usd
    if metadata:
        body["metadata"] = metadata

    _state.enqueue("/v1/heartbeats", body)
    return run_id


def status() -> dict[str, Any]:
    return _state.status()


def flush(timeout: float = 5.0) -> None:
    _state.flush(timeout=timeout)


def instrument_anthropic() -> None:
    from .instrumentation._anthropic import instrument as _instrument
    _instrument()


def instrument_openai() -> None:
    from .instrumentation._openai import instrument as _instrument
    _instrument()


def instrument_gemini() -> None:
    from .instrumentation._gemini import instrument as _instrument
    _instrument()


def instrument_mistral() -> None:
    from .instrumentation._mistral import instrument as _instrument
    _instrument()


def instrument_cohere() -> None:
    from .instrumentation._cohere import instrument as _instrument
    _instrument()


def instrument_bedrock() -> None:
    from .instrumentation._bedrock import instrument as _instrument
    _instrument()


def instrument_pydantic_ai() -> None:
    from .instrumentation._pydantic_ai import instrument as _instrument
    _instrument()


def instrument_crewai() -> None:
    from .instrumentation._crewai import instrument as _instrument
    _instrument()


def instrument_openai_agents() -> None:
    from .instrumentation._openai_agents import instrument as _instrument
    _instrument()


def __getattr__(name: str) -> Any:
    """Lazy-expose framework-bound classes so importing agentping never
    forces a hard dependency on openai-agents or langchain being installed."""
    if name == "AgentPingHooks":
        from .instrumentation._openai_agents import AgentPingHooks
        return AgentPingHooks
    if name == "AgentPingCallbackHandler":
        from .instrumentation._langchain import AgentPingCallbackHandler
        return AgentPingCallbackHandler
    if name == "AgentPingLlamaIndexHandler":
        from .instrumentation._llamaindex import AgentPingLlamaIndexHandler
        return AgentPingLlamaIndexHandler
    raise AttributeError(f"module 'agentping' has no attribute {name!r}")
