"""Pydantic AI instrumentation.

Wraps `Agent.run_sync` and `Agent.run`. On a successful run, emit a
single `llm_call` event with the aggregated usage from the agent loop.
Pydantic AI's `result.usage` is the right grain: it already covers
every model exchange in the loop, including tool retries.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._streaming import emit_llm_call

_PATCHED = False
_SUPPORTED_RANGE = ">=1.0.0,<2.0.0"


def _detect_version() -> str:
    try:
        from importlib.metadata import version
        return version("pydantic-ai")
    except Exception:
        try:
            from importlib.metadata import version
            return version("pydantic-ai-slim")
        except Exception:
            return ""


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return major == 1


def _resolve_provider_and_model(agent: Any) -> tuple[str | None, str | None]:
    model = getattr(agent, "model", None)
    if model is None:
        return None, None
    return getattr(model, "system", None), getattr(model, "model_name", None)


def _build_payload(result: Any, agent: Any, latency_ms: int) -> dict[str, Any]:
    provider, model_name = _resolve_provider_and_model(agent)
    payload: dict[str, Any] = {
        "provider": provider or "unknown",
        "latency_ms": latency_ms,
    }
    if model_name:
        payload["model"] = model_name

    try:
        usage = result.usage
    except Exception:
        return payload

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_read = getattr(usage, "cache_read_tokens", None)
    cache_write = getattr(usage, "cache_write_tokens", None)

    if input_tokens is not None and cache_read:
        payload["input_tokens"] = int(input_tokens) - int(cache_read)
        payload["cached_input_tokens"] = int(cache_read)
    elif input_tokens is not None:
        payload["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        payload["output_tokens"] = int(output_tokens)
    if cache_write:
        payload["cache_creation_input_tokens"] = int(cache_write)

    return payload


def instrument() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from pydantic_ai import Agent
    except ImportError:
        _warnings.warn_once(
            "pydantic_ai_missing",
            "pydantic-ai package not installed; instrument_pydantic_ai() is a no-op.",
        )
        return

    version = _detect_version()
    if not _version_supported(version):
        _warnings.warn_once(
            "pydantic_ai_version",
            f"pydantic-ai version {version} outside supported range {_SUPPORTED_RANGE}; instrumentation disabled.",
        )
        return

    original_run_sync = Agent.run_sync
    original_run = Agent.run

    def patched_run_sync(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = original_run_sync(self, *args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            payload = _build_payload(result, self, latency_ms)
            emit_llm_call(payload)
        except Exception:
            pass
        return result

    async def patched_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = await original_run(self, *args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            payload = _build_payload(result, self, latency_ms)
            emit_llm_call(payload)
        except Exception:
            pass
        return result

    Agent.run_sync = patched_run_sync
    Agent.run = patched_run
    Agent._agentping_original_run_sync = original_run_sync
    Agent._agentping_original_run = original_run

    _PATCHED = True


def uninstrument() -> None:
    global _PATCHED
    try:
        from pydantic_ai import Agent
        if hasattr(Agent, "_agentping_original_run_sync"):
            Agent.run_sync = Agent._agentping_original_run_sync
            delattr(Agent, "_agentping_original_run_sync")
        if hasattr(Agent, "_agentping_original_run"):
            Agent.run = Agent._agentping_original_run
            delattr(Agent, "_agentping_original_run")
    except Exception:
        pass
    _PATCHED = False
