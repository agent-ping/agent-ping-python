"""Cohere V2 SDK instrumentation.

Wraps `cohere.ClientV2.chat` and `cohere.AsyncClientV2.chat`. On a
successful response, emit an `llm_call` event to the active run.
Token counts are pulled from `usage.billed_units` (what the user pays
for) rather than the raw `tokens` block.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._streaming import emit_llm_call

_PATCHED = False
_SUPPORTED_RANGE = ">=5.0.0,<6.0.0"


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return major == 5


def _build_payload(response: Any, model: str | None, latency_ms: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "cohere",
        "latency_ms": latency_ms,
    }
    if model:
        payload["model"] = model

    usage = getattr(response, "usage", None)
    billed = getattr(usage, "billed_units", None) if usage is not None else None
    if billed is not None:
        input_tokens = getattr(billed, "input_tokens", None)
        output_tokens = getattr(billed, "output_tokens", None)
        if input_tokens is not None:
            payload["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            payload["output_tokens"] = int(output_tokens)

    return payload


def instrument() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        import cohere
    except ImportError:
        _warnings.warn_once(
            "cohere_missing",
            "cohere package not installed; instrument_cohere() is a no-op.",
        )
        return

    version = getattr(cohere, "__version__", "")
    if not _version_supported(version):
        _warnings.warn_once(
            "cohere_version",
            f"cohere version {version} outside supported range {_SUPPORTED_RANGE}; instrumentation disabled.",
        )
        return

    try:
        ClientV2 = cohere.ClientV2
        AsyncClientV2 = cohere.AsyncClientV2
    except Exception:
        _warnings.warn_once(
            "cohere_layout",
            "cohere package internals changed; instrumentation disabled.",
        )
        return

    original_chat = ClientV2.chat

    def patched_chat(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        start = time.monotonic()
        result = original_chat(self, *args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            payload = _build_payload(result, model, latency_ms)
            emit_llm_call(payload)
        except Exception:
            pass
        return result

    ClientV2.chat = patched_chat
    ClientV2._agentping_original_chat = original_chat

    # chat_stream (sync)
    if hasattr(ClientV2, "chat_stream"):
        stream_original = ClientV2.chat_stream

        def patched_chat_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start = time.monotonic()
            iterator = stream_original(self, *args, **kwargs)
            return _wrap_stream_iterator(iterator, model, start)

        ClientV2.chat_stream = patched_chat_stream
        ClientV2._agentping_original_chat_stream = stream_original

    # embed (sync)
    if hasattr(ClientV2, "embed"):
        embed_original = ClientV2.embed

        def patched_embed(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start = time.monotonic()
            result = embed_original(self, *args, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                payload = _build_embed_payload(result, model, latency_ms)
                emit_llm_call(payload)
            except Exception:
                pass
            return result

        ClientV2.embed = patched_embed
        ClientV2._agentping_original_embed = embed_original

    try:
        async_original = AsyncClientV2.chat

        async def patched_async_chat(self: Any, *args: Any, **kwargs: Any) -> Any:
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

        AsyncClientV2.chat = patched_async_chat
        AsyncClientV2._agentping_original_chat = async_original

        # async embed
        if hasattr(AsyncClientV2, "embed"):
            async_embed_original = AsyncClientV2.embed

            async def patched_async_embed(self: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model")
                start = time.monotonic()
                result = await async_embed_original(self, *args, **kwargs)
                latency_ms = int((time.monotonic() - start) * 1000)
                try:
                    payload = _build_embed_payload(result, model, latency_ms)
                    emit_llm_call(payload)
                except Exception:
                    pass
                return result

            AsyncClientV2.embed = patched_async_embed
            AsyncClientV2._agentping_original_embed = async_embed_original
    except Exception:
        pass

    _PATCHED = True


def _build_embed_payload(response: Any, model: str | None, latency_ms: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "cohere",
        "latency_ms": latency_ms,
        "kind": "embedding",
    }
    if model:
        payload["model"] = model

    # Cohere v5 embed responses carry meta.billed_units.input_tokens.
    meta = getattr(response, "meta", None)
    billed = getattr(meta, "billed_units", None) if meta is not None else None
    if billed is not None:
        input_tokens = getattr(billed, "input_tokens", None)
        if input_tokens is not None:
            payload["input_tokens"] = int(input_tokens)
            payload["output_tokens"] = 0
    return payload


def _wrap_stream_iterator(iterator: Any, model: str | None, start: float) -> Any:
    """Wrap a Cohere chat_stream iterator. The terminal 'message-end' event
    carries usage.billed_units; we capture it and emit a single llm_call
    when the stream closes."""

    def gen():
        last_billed = None
        try:
            for event in iterator:
                # message-end event carries delta.usage.billed_units
                event_type = getattr(event, "type", None)
                if event_type == "message-end":
                    delta = getattr(event, "delta", None)
                    usage = getattr(delta, "usage", None) if delta is not None else None
                    billed = getattr(usage, "billed_units", None) if usage is not None else None
                    if billed is not None:
                        last_billed = billed
                yield event
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                payload: dict[str, Any] = {
                    "provider": "cohere",
                    "model": model or "unknown",
                    "latency_ms": latency_ms,
                }
                if last_billed is not None:
                    input_tokens = getattr(last_billed, "input_tokens", None)
                    output_tokens = getattr(last_billed, "output_tokens", None)
                    if input_tokens is not None:
                        payload["input_tokens"] = int(input_tokens)
                    if output_tokens is not None:
                        payload["output_tokens"] = int(output_tokens)
                emit_llm_call(payload)
            except Exception:
                pass

    return gen()


def uninstrument() -> None:
    global _PATCHED
    try:
        import cohere
        for cls_attr in ("ClientV2", "AsyncClientV2"):
            cls = getattr(cohere, cls_attr, None)
            if cls is None:
                continue
            for attr, original in (
                ("_agentping_original_chat", "chat"),
                ("_agentping_original_chat_stream", "chat_stream"),
                ("_agentping_original_embed", "embed"),
            ):
                if hasattr(cls, attr):
                    setattr(cls, original, getattr(cls, attr))
                    delattr(cls, attr)
    except Exception:
        pass
    _PATCHED = False
