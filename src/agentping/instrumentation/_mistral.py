"""Mistral SDK instrumentation.

Wraps `mistralai.client.chat.Chat.complete`, `complete_async`, `stream` and
`stream_async`. Every call emits one `llm_call` to the active run: on
success with token usage, `finish_reason` and the number of tool calls; on
failure with `status: "error"` and the exception class, after which the
exception is re-raised.

Streams report `ttft_ms` from the first delta carrying content or a tool
call. With `init(capture_llm_content=True)` the call also carries
`messages` as `input` (system messages included, as the API takes them)
and the first choice's text as `output`.
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
_SUPPORTED_RANGE = ">=2.0.0,<3.0.0"


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return major == 2


def _usage_fields(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    fields: dict[str, Any] = {}
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is not None:
        fields["input_tokens"] = int(prompt)
    if completion is not None:
        fields["output_tokens"] = int(completion)
    return fields


def _build_payload(
    response: Any, model: str | None, latency: int, stream: bool = False, request: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "mistral",
        "latency_ms": latency,
    }
    resolved_model = model or getattr(response, "model", None)
    if resolved_model:
        payload["model"] = resolved_model
    payload.update(_usage_fields(getattr(response, "usage", None)))
    choices = getattr(response, "choices", None) or []
    output = ""
    if choices:
        first = choices[0]
        put_if_str(payload, "finish_reason", getattr(first, "finish_reason", None))
        message = getattr(first, "message", None)
        put_if_positive(payload, "tool_calls", len(getattr(message, "tool_calls", None) or []))
        output = message_text(getattr(message, "content", None))
    if stream:
        payload["stream"] = True
    request = request or {}
    put_content(payload, input=request.get("messages"), output=output)
    return payload


def _detect_version() -> str:
    try:
        from importlib.metadata import version

        return version("mistralai")
    except Exception:
        return ""


class _EventAccumulator(FirstChunkClock):
    """chat.stream() yields CompletionEvent objects; the final one carries usage on .data."""

    def __init__(self, model: str | None, request: dict[str, Any] | None = None) -> None:
        self.model = model
        self.usage: Any = None
        self.finish_reason: str | None = None
        self.tool_calls = 0
        self.request = request or {}
        self.text: list[str] = []
        self.capture = content_capture_enabled()

    def observe(self, event: Any) -> None:
        data = getattr(event, "data", event)
        usage = getattr(data, "usage", None)
        if usage is not None:
            self.usage = usage
        self.model = getattr(data, "model", None) or self.model
        for choice in getattr(data, "choices", None) or []:
            finish = getattr(choice, "finish_reason", None)
            if finish:
                self.finish_reason = finish
            delta = getattr(choice, "delta", None)
            tool_calls = getattr(delta, "tool_calls", None) or []
            self.tool_calls += len(tool_calls)
            content = getattr(delta, "content", None)
            if content or tool_calls:
                self.mark_content()
            if self.capture and content:
                self.text.append(message_text(content))

    def payload(self, start: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": "mistral",
            "model": self.model or "unknown",
            "latency_ms": latency_ms(start),
            "stream": True,
        }
        payload.update(_usage_fields(self.usage))
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
        emit_errored_llm_call("mistral", acc.model, start, exc, {"stream": True})

    proxy = AsyncStreamProxy if is_async else StreamProxy
    return proxy(iterator, acc.observe, on_complete, on_error)


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
        try:
            result = original_complete(self, *args, **kwargs)
        except BaseException as exc:
            emit_errored_llm_call("mistral", model, start, exc)
            raise
        try:
            emit_llm_call(_build_payload(result, model, latency_ms(start), request=kwargs))
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
            try:
                result = await async_original(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("mistral", model, start, exc)
                raise
            try:
                emit_llm_call(_build_payload(result, model, latency_ms(start), request=kwargs))
            except Exception:
                pass
            return result

        Chat.complete_async = patched_complete_async
        Chat._agentping_original_complete_async = async_original

    if hasattr(Chat, "stream"):
        stream_original = Chat.stream

        def patched_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start = time.monotonic()
            try:
                iterator = stream_original(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("mistral", model, start, exc, {"stream": True})
                raise
            return _wrap_stream(iterator, model, start, is_async=False, request=kwargs)

        Chat.stream = patched_stream
        Chat._agentping_original_stream = stream_original

    if hasattr(Chat, "stream_async"):
        stream_async_original = Chat.stream_async

        async def patched_stream_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model")
            start = time.monotonic()
            try:
                iterator = await stream_async_original(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("mistral", model, start, exc, {"stream": True})
                raise
            return _wrap_stream(iterator, model, start, is_async=True, request=kwargs)

        Chat.stream_async = patched_stream_async
        Chat._agentping_original_stream_async = stream_async_original

    _PATCHED = True


def uninstrument() -> None:
    global _PATCHED
    try:
        from mistralai.client.chat import Chat

        for attr, original in (
            ("_agentping_original_complete", "complete"),
            ("_agentping_original_complete_async", "complete_async"),
            ("_agentping_original_stream", "stream"),
            ("_agentping_original_stream_async", "stream_async"),
        ):
            if hasattr(Chat, attr):
                setattr(Chat, original, getattr(Chat, attr))
                delattr(Chat, attr)
    except Exception:
        pass
    _PATCHED = False
