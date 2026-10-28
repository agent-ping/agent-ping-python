"""Cohere V2 SDK instrumentation.

Wraps `cohere.ClientV2.chat`, `chat_stream`, `embed` and the async client.
Every call emits one `llm_call` to the active run: on success with token
counts from `usage.billed_units` (what the user pays for) rather than the
raw `tokens` block, plus `finish_reason` and the number of tool calls; on
failure with `status: "error"` and the exception class, after which the
exception is re-raised.

Streams report `ttft_ms` from the first content or tool-call event. With
`init(capture_llm_content=True)` the call also carries `messages` as
`input` (system messages included, as the API takes them) and the
response's text as `output`.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._shared import (
    FirstChunkClock,
    content_capture_enabled,
    emit_errored_llm_call,
    latency_ms,
    message_text,
    put_content,
    put_if_positive,
    put_if_str,
)
from ._streaming import AsyncStreamProxy, StreamProxy, emit_llm_call

_PATCHED = False
_SUPPORTED_RANGE = ">=5.0.0,<6.0.0"


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return major == 5


def _billed_fields(billed: Any) -> dict[str, Any]:
    if billed is None:
        return {}
    fields: dict[str, Any] = {}
    input_tokens = getattr(billed, "input_tokens", None)
    output_tokens = getattr(billed, "output_tokens", None)
    if input_tokens is not None:
        fields["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        fields["output_tokens"] = int(output_tokens)
    return fields


def _build_payload(
    response: Any, model: str | None, latency: int, stream: bool = False, request: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "cohere",
        "latency_ms": latency,
    }
    if model:
        payload["model"] = model
    usage = getattr(response, "usage", None)
    payload.update(_billed_fields(getattr(usage, "billed_units", None) if usage is not None else None))
    put_if_str(payload, "finish_reason", getattr(response, "finish_reason", None))
    message = getattr(response, "message", None)
    put_if_positive(payload, "tool_calls", len(getattr(message, "tool_calls", None) or []))
    if stream:
        payload["stream"] = True
    request = request or {}
    put_content(payload, input=request.get("messages"), output=message_text(getattr(message, "content", None)))
    return payload


def _build_embed_payload(response: Any, model: str | None, latency: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "cohere",
        "latency_ms": latency,
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


_CONTENT_EVENTS = {"content-start", "content-delta", "tool-plan-delta", "tool-call-start", "tool-call-delta"}


class _EventAccumulator(FirstChunkClock):
    """The terminal `message-end` event carries delta.usage.billed_units and
    delta.finish_reason; each `tool-call-start` is one tool call; each
    `content-delta` carries a piece of the reply text."""

    def __init__(self, model: str | None, request: dict[str, Any] | None = None) -> None:
        self.model = model
        self.billed: Any = None
        self.finish_reason: str | None = None
        self.tool_calls = 0
        self.request = request or {}
        self.text: list[str] = []
        self.capture = content_capture_enabled()

    def observe(self, event: Any) -> None:
        event_type = getattr(event, "type", None)
        if event_type in _CONTENT_EVENTS:
            self.mark_content()
        if event_type == "content-delta" and self.capture:
            delta = getattr(event, "delta", None)
            message = getattr(delta, "message", None) if delta is not None else None
            content = getattr(message, "content", None) if message is not None else None
            text = getattr(content, "text", None) if content is not None else None
            if isinstance(text, str):
                self.text.append(text)
        if event_type == "message-end":
            delta = getattr(event, "delta", None)
            usage = getattr(delta, "usage", None) if delta is not None else None
            billed = getattr(usage, "billed_units", None) if usage is not None else None
            if billed is not None:
                self.billed = billed
            finish = getattr(delta, "finish_reason", None) if delta is not None else None
            if finish:
                self.finish_reason = finish
        elif event_type == "tool-call-start":
            self.tool_calls += 1

    def payload(self, start: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": "cohere",
            "model": self.model or "unknown",
            "latency_ms": latency_ms(start),
            "stream": True,
        }
        payload.update(_billed_fields(self.billed))
        put_if_str(payload, "finish_reason", self.finish_reason)
        put_if_positive(payload, "tool_calls", self.tool_calls)
        self.put_ttft(payload, start)
        put_content(payload, input=self.request.get("messages"), output="".join(self.text))
        return payload


def _wrap_stream(
    iterator: Any, model: str | None, start: float, is_async: bool, request: dict[str, Any] | None = None
) -> Any:
    acc = _EventAccumulator(model, request)

    def on_complete() -> None:
        emit_llm_call(acc.payload(start))

    def on_error(exc: BaseException) -> None:
        emit_errored_llm_call("cohere", acc.model, start, exc, {"stream": True})

    proxy = AsyncStreamProxy if is_async else StreamProxy
    return proxy(iterator, acc.observe, on_complete, on_error)


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
        try:
            result = original_chat(self, *args, **kwargs)
        except BaseException as exc:
            emit_errored_llm_call("cohere", model, start, exc)
            raise
        try:
            emit_llm_call(_build_payload(result, model, latency_ms(start), request=kwargs))
        except Exception:
            pass
        return result

    ClientV2.chat = patched_chat
    ClientV2._agentping_original_chat = original_chat

    if hasattr(ClientV2, "chat_stream"):
        stream_original = ClientV2.chat_stream

        def patched_chat_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start = time.monotonic()
            try:
                iterator = stream_original(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("cohere", model, start, exc, {"stream": True})
                raise
            return _wrap_stream(iterator, model, start, is_async=False, request=kwargs)

        ClientV2.chat_stream = patched_chat_stream
        ClientV2._agentping_original_chat_stream = stream_original

    if hasattr(ClientV2, "embed"):
        embed_original = ClientV2.embed

        def patched_embed(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start = time.monotonic()
            try:
                result = embed_original(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("cohere", model, start, exc, {"kind": "embedding"})
                raise
            try:
                emit_llm_call(_build_embed_payload(result, model, latency_ms(start)))
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
            try:
                result = await async_original(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("cohere", model, start, exc)
                raise
            try:
                emit_llm_call(_build_payload(result, model, latency_ms(start), request=kwargs))
            except Exception:
                pass
            return result

        AsyncClientV2.chat = patched_async_chat
        AsyncClientV2._agentping_original_chat = async_original

        if hasattr(AsyncClientV2, "chat_stream"):
            async_stream_original = AsyncClientV2.chat_stream

            def patched_async_chat_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model")
                start = time.monotonic()
                try:
                    iterator = async_stream_original(self, *args, **kwargs)
                except BaseException as exc:
                    emit_errored_llm_call("cohere", model, start, exc, {"stream": True})
                    raise
                return _wrap_stream(iterator, model, start, is_async=True, request=kwargs)

            AsyncClientV2.chat_stream = patched_async_chat_stream
            AsyncClientV2._agentping_original_chat_stream = async_stream_original

        if hasattr(AsyncClientV2, "embed"):
            async_embed_original = AsyncClientV2.embed

            async def patched_async_embed(self: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model")
                start = time.monotonic()
                try:
                    result = await async_embed_original(self, *args, **kwargs)
                except BaseException as exc:
                    emit_errored_llm_call("cohere", model, start, exc, {"kind": "embedding"})
                    raise
                try:
                    emit_llm_call(_build_embed_payload(result, model, latency_ms(start)))
                except Exception:
                    pass
                return result

            AsyncClientV2.embed = patched_async_embed
            AsyncClientV2._agentping_original_embed = async_embed_original
    except Exception:
        pass

    _PATCHED = True


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
