"""Anthropic SDK instrumentation.

Wraps `anthropic.Anthropic().messages.create` and the async variant.
On a successful response, emit an `llm_call` event to the active run.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._streaming import emit_llm_call

_PATCHED = False
_SUPPORTED_RANGE = ">=0.30.0,<2.0.0"


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    if major < 1:
        try:
            minor = int(version.split(".")[1])
        except (ValueError, IndexError):
            return False
        return minor >= 30
    return major < 2


def _build_payload(response: Any, latency_ms: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "anthropic",
        "latency_ms": latency_ms,
    }
    model = getattr(response, "model", None)
    if model:
        payload["model"] = model

    usage = getattr(response, "usage", None)
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cache_read = getattr(usage, "cache_read_input_tokens", None)
        cache_create = getattr(usage, "cache_creation_input_tokens", None)
        if input_tokens is not None:
            payload["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            payload["output_tokens"] = int(output_tokens)
        if cache_read:
            payload["cached_input_tokens"] = int(cache_read)
        if cache_create:
            payload["cache_creation_input_tokens"] = int(cache_create)

    return payload


def instrument() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        import anthropic
    except ImportError:
        _warnings.warn_once(
            "anthropic_missing",
            "anthropic package not installed; instrument_anthropic() is a no-op.",
        )
        return

    version = getattr(anthropic, "__version__", "")
    if not _version_supported(version):
        _warnings.warn_once(
            "anthropic_version",
            f"anthropic version {version} outside supported range {_SUPPORTED_RANGE}; instrumentation disabled.",
        )
        return

    try:
        from anthropic.resources.messages import Messages
    except Exception:
        try:
            from anthropic.resources.messages.messages import Messages
        except Exception:
            _warnings.warn_once(
                "anthropic_layout",
                "anthropic package internals changed; instrumentation disabled.",
            )
            return

    original_create = Messages.create

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

    Messages.create = patched_create
    Messages._agentping_original_create = original_create

    try:
        from anthropic.resources.messages import AsyncMessages
        async_original = AsyncMessages.create

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

        AsyncMessages.create = patched_async_create
        AsyncMessages._agentping_original_create = async_original
    except Exception:
        pass

    _PATCHED = True


def uninstrument() -> None:
    """For tests: undo the patch."""
    global _PATCHED
    try:
        from anthropic.resources.messages import Messages
        if hasattr(Messages, "_agentping_original_create"):
            Messages.create = Messages._agentping_original_create
            delattr(Messages, "_agentping_original_create")
    except Exception:
        pass
    try:
        from anthropic.resources.messages import AsyncMessages
        if hasattr(AsyncMessages, "_agentping_original_create"):
            AsyncMessages.create = AsyncMessages._agentping_original_create
            delattr(AsyncMessages, "_agentping_original_create")
    except Exception:
        pass
    _PATCHED = False
