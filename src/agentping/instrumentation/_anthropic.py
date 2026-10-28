"""Anthropic SDK instrumentation.

Wraps `anthropic.Anthropic().messages.create` (plain and `stream=True`),
`messages.stream(...)` and the async variants. Every call emits one
`llm_call` to the active run: on success with token usage, `finish_reason`
and the number of tool calls; on failure with `status: "error"` and the
exception class, after which the exception is re-raised.

Anthropic reports `input_tokens` net of the cached prefix. Ingest wants the
gross count (it subtracts `cached_input_tokens` itself), so the cache read
and creation counts are added back.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from .. import _warnings
from ._shared import (
    FirstChunkClock,
    content_capture_enabled,
    emit_errored_llm_call,
    latency_ms,
    put_content,
    put_if_positive,
    put_if_str,
)
from ._streaming import AsyncStreamProxy, StreamProxy, emit_llm_call

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


def _usage_fields(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    fields: dict[str, Any] = {}
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    cache_read = int(getattr(usage, "cache_read_input_tokens", None) or 0)
    cache_create = int(getattr(usage, "cache_creation_input_tokens", None) or 0)
    if input_tokens is not None:
        fields["input_tokens"] = int(input_tokens) + cache_read + cache_create
    if output_tokens is not None:
        fields["output_tokens"] = int(output_tokens)
    put_if_positive(fields, "cached_input_tokens", cache_read)
    put_if_positive(fields, "cache_creation_input_tokens", cache_create)
    return fields


def _count_tool_uses(content: Any) -> int:
    count = 0
    for block in content or []:
        if getattr(block, "type", None) == "tool_use" or (
            isinstance(block, dict) and block.get("type") == "tool_use"
        ):
            count += 1
    return count


def _text_of(content: Any) -> str:
    """The text blocks of a response's content, joined."""
    pieces: list[str] = []
    for block in content or []:
        kind = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        if kind == "text":
            text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
            if isinstance(text, str):
                pieces.append(text)
    return "".join(pieces)


def _build_payload(response: Any, latency: int, stream: bool = False, request: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "anthropic",
        "latency_ms": latency,
    }
    model = getattr(response, "model", None)
    if model:
        payload["model"] = model
    payload.update(_usage_fields(getattr(response, "usage", None)))
    put_if_str(payload, "finish_reason", getattr(response, "stop_reason", None))
    put_if_positive(payload, "tool_calls", _count_tool_uses(getattr(response, "content", None)))
    if stream:
        payload["stream"] = True
    request = request or {}
    put_content(
        payload,
        input=request.get("messages"),
        system=request.get("system"),
        output=_text_of(getattr(response, "content", None)),
    )
    return payload


class _EventAccumulator(FirstChunkClock):
    """Builds the llm_call payload from raw `stream=True` events.

    Usage arrives in two places: `message_start` carries the input side and
    `message_delta` carries the cumulative output side, so the last delta
    wins rather than being summed.
    """

    def __init__(self, model: str | None, request: dict[str, Any] | None = None) -> None:
        self.model = model
        self.usage: dict[str, int] = {}
        self.finish_reason: str | None = None
        self.tool_calls = 0
        self.request = request or {}
        self.text: list[str] = []
        self.capture = content_capture_enabled()

    def observe(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if kind in ("content_block_start", "content_block_delta"):
            self.mark_content()
        if kind == "content_block_delta" and self.capture:
            delta = getattr(event, "delta", None)
            text = getattr(delta, "text", None)
            if isinstance(text, str) and getattr(delta, "type", "text_delta") == "text_delta":
                self.text.append(text)
        if kind == "message_start":
            message = getattr(event, "message", None)
            self.model = getattr(message, "model", None) or self.model
            usage = getattr(message, "usage", None)
            self._merge_usage(usage)
        elif kind == "message_delta":
            self._merge_usage(getattr(event, "usage", None))
            delta = getattr(event, "delta", None)
            stop = getattr(delta, "stop_reason", None)
            if stop:
                self.finish_reason = stop
        elif kind == "content_block_start":
            block = getattr(event, "content_block", None)
            if getattr(block, "type", None) == "tool_use":
                self.tool_calls += 1

    def _merge_usage(self, usage: Any) -> None:
        if usage is None:
            return
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = getattr(usage, key, None)
            if value is not None:
                self.usage[key] = int(value)

    def payload(self, start: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": "anthropic",
            "model": self.model or "unknown",
            "latency_ms": latency_ms(start),
            "stream": True,
        }
        if self.usage:
            payload.update(_usage_fields(SimpleNamespace(**self.usage)))
        put_if_str(payload, "finish_reason", self.finish_reason)
        put_if_positive(payload, "tool_calls", self.tool_calls)
        self.put_ttft(payload, start)
        put_content(payload, input=self.request.get("messages"), system=self.request.get("system"), output="".join(self.text))
        return payload


def _wrap_stream(result: Any, model: str | None, start: float, is_async: bool, request: dict[str, Any] | None = None) -> Any:
    acc = _EventAccumulator(model, request)

    def on_complete() -> None:
        emit_llm_call(acc.payload(start))

    def on_error(exc: BaseException) -> None:
        emit_errored_llm_call("anthropic", acc.model, start, exc, {"stream": True})

    proxy = AsyncStreamProxy if is_async else StreamProxy
    return proxy(result, acc.observe, on_complete, on_error)


class _StreamManagerProxy:
    """Wraps `messages.stream(...)`'s context manager.

    The SDK's MessageStream keeps an accumulated `current_message_snapshot`,
    so on exit we can read the final usage, stop reason and tool uses without
    consuming anything the caller did not.
    """

    def __init__(self, inner: Any, model: str | None, start: float, request: dict[str, Any] | None = None) -> None:
        self._inner = inner
        self._model = model
        self._start = start
        self._request = request or {}
        self._stream: Any = None

    def __enter__(self) -> Any:
        self._stream = self._inner.__enter__()
        return self._stream

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        self._emit(exc)
        return self._inner.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _emit(self, exc: Any) -> None:
        try:
            snapshot = None
            if self._stream is not None:
                try:
                    snapshot = self._stream.current_message_snapshot
                except Exception:
                    snapshot = None
            if exc is not None and snapshot is None:
                emit_errored_llm_call("anthropic", self._model, self._start, exc, {"stream": True})
                return
            if snapshot is None:
                return
            payload = _build_payload(snapshot, latency_ms(self._start), stream=True, request=self._request)
            if not payload.get("model"):
                payload["model"] = self._model or "unknown"
            emit_llm_call(payload)
        except Exception:
            pass


class _AsyncStreamManagerProxy(_StreamManagerProxy):
    async def __aenter__(self) -> Any:
        self._stream = await self._inner.__aenter__()
        return self._stream

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        self._emit(exc)
        return await self._inner.__aexit__(exc_type, exc, tb)


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
        model = kwargs.get("model")
        try:
            result = original_create(self, *args, **kwargs)
        except BaseException as exc:
            emit_errored_llm_call("anthropic", model, start, exc)
            raise
        if kwargs.get("stream"):
            return _wrap_stream(result, model, start, is_async=False, request=kwargs)
        try:
            emit_llm_call(_build_payload(result, latency_ms(start), request=kwargs))
        except Exception:
            pass
        return result

    Messages.create = patched_create
    Messages._agentping_original_create = original_create

    original_stream = getattr(Messages, "stream", None)
    if original_stream is not None:

        def patched_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            model = kwargs.get("model")
            try:
                manager = original_stream(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("anthropic", model, start, exc, {"stream": True})
                raise
            return _StreamManagerProxy(manager, model, start, kwargs)

        Messages.stream = patched_stream
        Messages._agentping_original_stream = original_stream

    try:
        from anthropic.resources.messages import AsyncMessages

        async_original = AsyncMessages.create

        async def patched_async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            model = kwargs.get("model")
            try:
                result = await async_original(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("anthropic", model, start, exc)
                raise
            if kwargs.get("stream"):
                return _wrap_stream(result, model, start, is_async=True, request=kwargs)
            try:
                emit_llm_call(_build_payload(result, latency_ms(start), request=kwargs))
            except Exception:
                pass
            return result

        AsyncMessages.create = patched_async_create
        AsyncMessages._agentping_original_create = async_original

        async_original_stream = getattr(AsyncMessages, "stream", None)
        if async_original_stream is not None:

            def patched_async_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
                start = time.monotonic()
                model = kwargs.get("model")
                try:
                    manager = async_original_stream(self, *args, **kwargs)
                except BaseException as exc:
                    emit_errored_llm_call("anthropic", model, start, exc, {"stream": True})
                    raise
                return _AsyncStreamManagerProxy(manager, model, start, kwargs)

            AsyncMessages.stream = patched_async_stream
            AsyncMessages._agentping_original_stream = async_original_stream
    except Exception:
        pass

    _PATCHED = True


def _restore(cls: Any, name: str) -> None:
    saved = f"_agentping_original_{name}"
    if hasattr(cls, saved):
        setattr(cls, name, getattr(cls, saved))
        delattr(cls, saved)


def uninstrument() -> None:
    """For tests: undo the patch."""
    global _PATCHED
    for path in ("anthropic.resources.messages",):
        for cls_name in ("Messages", "AsyncMessages"):
            try:
                module = __import__(path, fromlist=[cls_name])
                cls = getattr(module, cls_name)
            except Exception:
                continue
            _restore(cls, "create")
            _restore(cls, "stream")
    _PATCHED = False
