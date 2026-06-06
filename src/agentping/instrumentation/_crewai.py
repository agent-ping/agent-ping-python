"""CrewAI instrumentation.

Wraps `Crew.kickoff`. After kickoff returns, read `crew.usage_metrics`
and emit a single `llm_call` event with the aggregated token totals.
The provider is reported as `crewai` because a crew can span multiple
providers; per-task pricing is left to the per-task callback pattern.
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
        return version("crewai")
    except Exception:
        return ""


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return major == 1


def _build_payload(metrics: Any, latency_ms: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "crewai",
        "latency_ms": latency_ms,
    }
    if metrics is None:
        return payload

    prompt = getattr(metrics, "prompt_tokens", None)
    completion = getattr(metrics, "completion_tokens", None)
    cached = getattr(metrics, "cached_prompt_tokens", None)
    requests = getattr(metrics, "successful_requests", None)

    if prompt is not None and cached:
        payload["input_tokens"] = int(prompt) - int(cached)
        payload["cached_input_tokens"] = int(cached)
    elif prompt is not None:
        payload["input_tokens"] = int(prompt)
    if completion is not None:
        payload["output_tokens"] = int(completion)
    if requests is not None:
        payload["request_count"] = int(requests)

    return payload


def instrument() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from crewai import Crew
    except ImportError:
        _warnings.warn_once(
            "crewai_missing",
            "crewai package not installed; instrument_crewai() is a no-op.",
        )
        return

    version = _detect_version()
    if not _version_supported(version):
        _warnings.warn_once(
            "crewai_version",
            f"crewai version {version} outside supported range {_SUPPORTED_RANGE}; instrumentation disabled.",
        )
        return

    original_kickoff = Crew.kickoff

    def patched_kickoff(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = original_kickoff(self, *args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            metrics = getattr(self, "usage_metrics", None)
            payload = _build_payload(metrics, latency_ms)
            emit_llm_call(payload)
        except Exception:
            pass
        return result

    Crew.kickoff = patched_kickoff
    Crew._agentping_original_kickoff = original_kickoff

    if hasattr(Crew, "kickoff_async"):
        async_original = Crew.kickoff_async

        async def patched_kickoff_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            result = await async_original(self, *args, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                metrics = getattr(self, "usage_metrics", None)
                payload = _build_payload(metrics, latency_ms)
                emit_llm_call(payload)
            except Exception:
                pass
            return result

        Crew.kickoff_async = patched_kickoff_async
        Crew._agentping_original_kickoff_async = async_original

    _PATCHED = True


def uninstrument() -> None:
    global _PATCHED
    try:
        from crewai import Crew
        if hasattr(Crew, "_agentping_original_kickoff"):
            Crew.kickoff = Crew._agentping_original_kickoff
            delattr(Crew, "_agentping_original_kickoff")
        if hasattr(Crew, "_agentping_original_kickoff_async"):
            Crew.kickoff_async = Crew._agentping_original_kickoff_async
            delattr(Crew, "_agentping_original_kickoff_async")
    except Exception:
        pass
    _PATCHED = False
