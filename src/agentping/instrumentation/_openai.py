"""OpenAI SDK instrumentation.

Wraps `openai.OpenAI().chat.completions.create`. On a successful response,
emit an `llm_call` event to the active run.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._streaming import emit_llm_call

_PATCHED = False
_SUPPORTED_RANGE = ">=1.30.0,<3.0.0"


def _version_supported(version: str) -> bool:
    try:
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return False
    if major < 1:
        return False
    if major == 1:
        return minor >= 30
    return major < 3


def _build_payload(response: Any, latency_ms: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "openai",
        "latency_ms": latency_ms,
    }
    model = getattr(response, "model", None)
    if model:
        payload["model"] = model

    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None

        if prompt is not None and cached:
            payload["input_tokens"] = int(prompt) - int(cached)
            payload["cached_input_tokens"] = int(cached)
        elif prompt is not None:
            payload["input_tokens"] = int(prompt)
        if completion is not None:
            payload["output_tokens"] = int(completion)

    return payload


def instrument() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        import openai
    except ImportError:
        _warnings.warn_once(
            "openai_missing",
            "openai package not installed; instrument_openai() is a no-op.",
        )
        return

    version = getattr(openai, "__version__", "")
    if not _version_supported(version):
        _warnings.warn_once(
            "openai_version",
            f"openai version {version} outside supported range {_SUPPORTED_RANGE}; instrumentation disabled.",
        )
        return

    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        try:
            from openai.resources.chat.completions.completions import Completions
        except Exception:
            _warnings.warn_once(
                "openai_layout",
                "openai package internals changed; instrumentation disabled.",
            )
            return

    original_create = Completions.create

    def patched_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = original_create(self, *args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            payload = _build_payload(result, latency_ms)
            emit_llm_call(payload)
        except Exception:
            pass
        return result

    Completions.create = patched_create
    Completions._agentping_original_create = original_create

    try:
        from openai.resources.chat.completions import AsyncCompletions
        async_original = AsyncCompletions.create

        async def patched_async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            result = await async_original(self, *args, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                payload = _build_payload(result, latency_ms)
                emit_llm_call(payload)
            except Exception:
                pass
            return result

        AsyncCompletions.create = patched_async_create
        AsyncCompletions._agentping_original_create = async_original
    except Exception:
        pass

    _PATCHED = True


def uninstrument() -> None:
    global _PATCHED
    try:
        from openai.resources.chat.completions import Completions
        if hasattr(Completions, "_agentping_original_create"):
            Completions.create = Completions._agentping_original_create
            delattr(Completions, "_agentping_original_create")
    except Exception:
        pass
    try:
        from openai.resources.chat.completions import AsyncCompletions
        if hasattr(AsyncCompletions, "_agentping_original_create"):
            AsyncCompletions.create = AsyncCompletions._agentping_original_create
            delattr(AsyncCompletions, "_agentping_original_create")
    except Exception:
        pass
    _PATCHED = False
