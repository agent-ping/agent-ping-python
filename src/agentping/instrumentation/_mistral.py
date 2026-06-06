"""Mistral SDK instrumentation.

Wraps `mistralai.client.chat.Chat.complete` and `Chat.complete_async`. On
a successful response, emit an `llm_call` event to the active run.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._streaming import emit_llm_call

_PATCHED = False
_SUPPORTED_RANGE = ">=2.0.0,<3.0.0"


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return major == 2


def _build_payload(response: Any, model: str | None, latency_ms: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "mistral",
        "latency_ms": latency_ms,
    }
    resolved_model = model or getattr(response, "model", None)
    if resolved_model:
        payload["model"] = resolved_model

    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if prompt is not None:
            payload["input_tokens"] = int(prompt)
        if completion is not None:
            payload["output_tokens"] = int(completion)

    return payload


def _detect_version() -> str:
    try:
        from importlib.metadata import version
        return version("mistralai")
    except Exception:
        return ""


def instrument() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        import mistralai.client.chat  # noqa: F401
    except ImportError:
        _warnings.warn_once(
            "mistral_missing",
            "mistralai package not installed; instrument_mistral() is a no-op.",
        )
        return

    version = _detect_version()
    if not _version_supported(version):
        _warnings.warn_once(
            "mistral_version",
            f"mistralai version {version} outside supported range {_SUPPORTED_RANGE}; instrumentation disabled.",
        )
        return

    try:
        from mistralai.client.chat import Chat
    except Exception:
        _warnings.warn_once(
            "mistral_layout",
            "mistralai package internals changed; instrumentation disabled.",
        )
        return

    original_complete = Chat.complete

    def patched_complete(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        start = time.monotonic()
        result = original_complete(self, *args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            payload = _build_payload(result, model, latency_ms)
            emit_llm_call(payload)
        except Exception:
            pass
        return result

    Chat.complete = patched_complete
    Chat._agentping_original_complete = original_complete

    if hasattr(Chat, "complete_async"):
        async_original = Chat.complete_async

        async def patched_complete_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start = time.monotonic()
            result = await async_original(self, *args, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                payload = _build_payload(result, model, latency_ms)
                emit_llm_call(payload)
            except Exception:
                pass
            return result

        Chat.complete_async = patched_complete_async
        Chat._agentping_original_complete_async = async_original

    # stream (sync): chat.stream() yields CompletionEvent objects; the
    # final event carries usage on its .data attribute.
    if hasattr(Chat, "stream"):
        stream_original = Chat.stream

        def patched_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start = time.monotonic()
            iterator = stream_original(self, *args, **kwargs)
            return _wrap_stream_iterator(iterator, model, start)

        Chat.stream = patched_stream
        Chat._agentping_original_stream = stream_original

    _PATCHED = True


def _wrap_stream_iterator(iterator: Any, model: str | None, start: float) -> Any:
    """Wrap a Mistral streaming iterator: pass events through, capture final usage."""

    def gen():
        last_usage = None
        last_model = model
        try:
            for event in iterator:
                data = getattr(event, "data", event)
                usage = getattr(data, "usage", None)
                if usage is not None:
                    last_usage = usage
                event_model = getattr(data, "model", None)
                if event_model:
                    last_model = event_model
                yield event
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                if last_usage is not None:
                    prompt = getattr(last_usage, "prompt_tokens", None)
                    completion = getattr(last_usage, "completion_tokens", None)
                    payload: dict[str, Any] = {
                        "provider": "mistral",
                        "model": last_model or "unknown",
                        "latency_ms": latency_ms,
                    }
                    if prompt is not None:
                        payload["input_tokens"] = int(prompt)
                    if completion is not None:
                        payload["output_tokens"] = int(completion)
                    emit_llm_call(payload)
            except Exception:
                pass

    return gen()


def uninstrument() -> None:
    global _PATCHED
    try:
        from mistralai.client.chat import Chat
        for attr, original in (
            ("_agentping_original_complete", "complete"),
            ("_agentping_original_complete_async", "complete_async"),
            ("_agentping_original_stream", "stream"),
        ):
            if hasattr(Chat, attr):
                setattr(Chat, original, getattr(Chat, attr))
                delattr(Chat, attr)
    except Exception:
        pass
    _PATCHED = False
