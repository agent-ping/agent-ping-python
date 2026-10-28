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

import re
from datetime import datetime, timezone
from typing import Any

import httpx

from . import _ids, _state, guard
from ._run import Run, _iso
from .guard import GuardVerdict, Paused

_SESSION_RE = re.compile(r"[0-9a-f]{32}")

__version__ = "0.3.0"

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
    version: str | None = None,
    capture_llm_content: bool = False,
    llm_content_max_chars: int = _state.DEFAULT_LLM_CONTENT_MAX_CHARS,
) -> None:
    """Configure the SDK.

    Args:
        version: The build or prompt version this process runs, applied to
            every run that does not set its own (``AGENTPING_VERSION`` also
            works). The dashboard marks where each new version was first seen.
        capture_llm_content: Send prompt and completion text on the
            ``llm_call`` events auto-instrumentation emits (``input``,
            ``output``, ``system_instructions``). Off by default: turn it on
            only when your prompts may leave your infrastructure.
        llm_content_max_chars: Truncate each captured piece at this length.
    """
    _state.init(
        api_key=api_key,
        base_url=base_url,
        queue_size=queue_size,
        transport=transport,
        control_url=control_url,
        version=version,
        capture_llm_content=capture_llm_content,
        llm_content_max_chars=llm_content_max_chars,
    )


def run(
    agent: str,
    customer_id: str | None = None,
    feature: str | None = None,
    metadata: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
    goal: str | None = None,
    version: str | None = None,
) -> Run:
    return Run(
        agent=agent,
        customer_id=customer_id,
        feature=feature,
        metadata=metadata,
        parent_run_id=parent_run_id,
        goal=goal,
        version=version,
    )


def new_session_id() -> str:
    """A session id for `conversation()`. Keep it for the life of the chat."""
    return _ids.uuid7_hex()


def conversation(
    agent: str,
    session_id: str,
    idle_timeout_seconds: int = 900,
    idle_timeout_status: str = "success",
    customer_id: str | None = None,
    feature: str | None = None,
    metadata: dict[str, Any] | None = None,
    goal: str | None = None,
    version: str | None = None,
) -> Run:
    """One chat session as one run, with every turn appended to it.

    A conversation has no reliable end -- the user closes the tab, or simply
    stops -- so pass the same ``session_id`` on every turn and the turns collect
    on a single record instead of becoming a run per message. The run id and its
    start time are both derived from the session id, so a stateless backend can
    rebuild this on any request without storing a mapping; re-opening is
    idempotent server-side.

        chat = agentping.conversation("support-chat", session_id)
        chat.event("llm_call", provider="anthropic", model=model,
                   input_tokens=n_in, output_tokens=n_out)
        # ...and when the user closes the chat:
        chat.finish(status="success")

    If ``finish`` is never called, the idle timeout closes the run at its last
    activity.

    Args:
        session_id: 32 hex characters whose first 12 are the session's start
            time in milliseconds. Use ``new_session_id()``.
    """
    if not _SESSION_RE.fullmatch(session_id):
        raise ValueError(
            "conversation: session_id must be 32 hex characters (see new_session_id())"
        )
    started_ms = int(session_id[:12], 16)
    if started_ms <= 0:
        raise ValueError("conversation: session_id does not carry a start time")
    region = _state.state().region or "eu"
    started_at = datetime.fromtimestamp(started_ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    return Run(
        agent=agent,
        customer_id=customer_id,
        feature=feature,
        metadata=metadata,
        goal=goal,
        version=version,
        idle_timeout_seconds=idle_timeout_seconds,
        idle_timeout_status=idle_timeout_status,
        resume=(f"run_{region}_{session_id}", started_at),
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
