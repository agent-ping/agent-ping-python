"""OpenAI SDK instrumentation.

Wraps `openai.OpenAI().chat.completions.create` and `responses.create`,
plain and `stream=True`, plus the async variants. Every call emits one
`llm_call` to the active run: on success with token usage (gross
`input_tokens`, cached and reasoning splits), `finish_reason` and the
number of tool calls; on failure with `status: "error"` and the exception
class, after which the exception is re-raised.

Chat streams only carry usage when the caller passes
`stream_options={"include_usage": True}`; without it the streamed llm_call
has latency and tool calls but no token counts.
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
    put_content,
    put_if_positive,
    put_if_str,
)
from ._streaming import AsyncStreamProxy, StreamProxy, emit_llm_call

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


def _chat_usage_fields(usage: Any) -> dict[str, Any]:
    """Chat Completions usage. `prompt_tokens` is already gross."""
    if usage is None:
        return {}
    fields: dict[str, Any] = {}
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is not None:
        fields["input_tokens"] = int(prompt)
    if completion is not None:
        fields["output_tokens"] = int(completion)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    put_if_positive(fields, "cached_input_tokens", getattr(prompt_details, "cached_tokens", None))
    completion_details = getattr(usage, "completion_tokens_details", None)
    put_if_positive(fields, "reasoning_tokens", getattr(completion_details, "reasoning_tokens", None))
    return fields


def _responses_usage_fields(usage: Any) -> dict[str, Any]:
    """Responses API usage. `input_tokens` is already gross."""
    if usage is None:
        return {}
    fields: dict[str, Any] = {}
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is not None:
        fields["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        fields["output_tokens"] = int(output_tokens)
    input_details = getattr(usage, "input_tokens_details", None)
    put_if_positive(fields, "cached_input_tokens", getattr(input_details, "cached_tokens", None))
    output_details = getattr(usage, "output_tokens_details", None)
    put_if_positive(fields, "reasoning_tokens", getattr(output_details, "reasoning_tokens", None))
    return fields


def _build_payload(response: Any, latency: int, stream: bool = False, request: dict[str, Any] | None = None) -> dict[str, Any]:
    """Payload for a Chat Completions response."""
    payload: dict[str, Any] = {"provider": "openai", "latency_ms": latency}
    model = getattr(response, "model", None)
    if model:
        payload["model"] = model
    payload.update(_chat_usage_fields(getattr(response, "usage", None)))
    choices = getattr(response, "choices", None) or []
    if choices:
        first = choices[0]
        put_if_str(payload, "finish_reason", getattr(first, "finish_reason", None))
        message = getattr(first, "message", None)
        put_if_positive(payload, "tool_calls", len(getattr(message, "tool_calls", None) or []))
    if stream:
        payload["stream"] = True
    request = request or {}
    output = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
    put_content(payload, input=request.get("messages"), output=output if isinstance(output, str) else None)
    return payload


def _build_responses_payload(response: Any, latency: int, stream: bool = False, request: dict[str, Any] | None = None) -> dict[str, Any]:
    """Payload for a Responses API response."""
    payload: dict[str, Any] = {"provider": "openai", "latency_ms": latency}
    model = getattr(response, "model", None)
    if model:
        payload["model"] = model
    payload.update(_responses_usage_fields(getattr(response, "usage", None)))
    status = getattr(response, "status", None)
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None)
    if status == "incomplete" and reason:
        payload["finish_reason"] = str(reason)
    elif status:
        put_if_str(payload, "finish_reason", status)
    tool_calls = 0
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            tool_calls += 1
    put_if_positive(payload, "tool_calls", tool_calls)
    if stream:
        payload["stream"] = True
    request = request or {}
    output = getattr(response, "output_text", None)
    put_content(
        payload,
        input=request.get("input"),
        system=request.get("instructions"),
        output=output if isinstance(output, str) else None,
    )
    return payload


class _ChatChunkAccumulator(FirstChunkClock):
    """Folds streamed Chat Completions chunks into one payload."""

    def __init__(self, model: str | None, request: dict[str, Any] | None = None) -> None:
        self.model = model
        self.usage: Any = None
        self.finish_reason: str | None = None
        self.tool_call_indexes: set[Any] = set()
        self.request = request or {}
        self.text: list[str] = []
        self.capture = content_capture_enabled()

    def observe(self, chunk: Any) -> None:
        self.model = getattr(chunk, "model", None) or self.model
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.usage = usage
        for choice in getattr(chunk, "choices", None) or []:
            finish = getattr(choice, "finish_reason", None)
            if finish:
                self.finish_reason = finish
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content != "":
                self.mark_content()
                if self.capture:
                    self.text.append(content)
            if getattr(delta, "tool_calls", None):
                self.mark_content()
            for call in getattr(delta, "tool_calls", None) or []:
                index = getattr(call, "index", None)
                self.tool_call_indexes.add(index if index is not None else getattr(call, "id", len(self.tool_call_indexes)))

    def payload(self, start: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": "openai",
            "model": self.model or "unknown",
            "latency_ms": latency_ms(start),
            "stream": True,
        }
        payload.update(_chat_usage_fields(self.usage))
        put_if_str(payload, "finish_reason", self.finish_reason)
        put_if_positive(payload, "tool_calls", len(self.tool_call_indexes))
        self.put_ttft(payload, start)
        put_content(payload, input=self.request.get("messages"), output="".join(self.text))
        return payload


class _ResponsesEventAccumulator(FirstChunkClock):
    """Folds streamed Responses events into one payload.

    The terminal `response.completed`, `response.incomplete` and
    `response.failed` events carry the full response, which is all we need.
    """

    def __init__(self, model: str | None, request: dict[str, Any] | None = None) -> None:
        self.model = model
        self.final: Any = None
        self.request = request or {}

    def observe(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if isinstance(kind, str) and (kind.endswith(".delta") or kind == "response.output_item.added"):
            self.mark_content()
        if kind in ("response.completed", "response.incomplete", "response.failed"):
            self.final = getattr(event, "response", None)
        elif kind == "response.created" and self.model is None:
            self.model = getattr(getattr(event, "response", None), "model", None)

    def payload(self, start: float) -> dict[str, Any]:
        if self.final is not None:
            payload = _build_responses_payload(self.final, latency_ms(start), stream=True, request=self.request)
            payload.setdefault("model", self.model or "unknown")
        else:
            payload = {
                "provider": "openai",
                "model": self.model or "unknown",
                "latency_ms": latency_ms(start),
                "stream": True,
            }
        self.put_ttft(payload, start)
        return payload


def _wrap_stream(result: Any, acc: Any, start: float, is_async: bool) -> Any:
    def on_complete() -> None:
        emit_llm_call(acc.payload(start))

    def on_error(exc: BaseException) -> None:
        emit_errored_llm_call("openai", acc.model, start, exc, {"stream": True})

    proxy = AsyncStreamProxy if is_async else StreamProxy
    return proxy(result, acc.observe, on_complete, on_error)


def _make_sync(original: Any, build: Any, accumulator: Any) -> Any:
    def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        model = kwargs.get("model")
        try:
            result = original(self, *args, **kwargs)
        except BaseException as exc:
            emit_errored_llm_call("openai", model, start, exc)
            raise
        if kwargs.get("stream"):
            return _wrap_stream(result, accumulator(model, kwargs), start, is_async=False)
        try:
            emit_llm_call(build(result, latency_ms(start), request=kwargs))
        except Exception:
            pass
        return result

    return patched


def _make_async(original: Any, build: Any, accumulator: Any) -> Any:
    async def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        model = kwargs.get("model")
        try:
            result = await original(self, *args, **kwargs)
        except BaseException as exc:
            emit_errored_llm_call("openai", model, start, exc)
            raise
        if kwargs.get("stream"):
            return _wrap_stream(result, accumulator(model, kwargs), start, is_async=True)
        try:
            emit_llm_call(build(result, latency_ms(start), request=kwargs))
        except Exception:
            pass
        return result

    return patched


def _import_class(paths: list[str], name: str) -> Any:
    for path in paths:
        try:
            module = __import__(path, fromlist=[name])
            return getattr(module, name)
        except Exception:
            continue
    return None


_CHAT_PATHS = ["openai.resources.chat.completions", "openai.resources.chat.completions.completions"]
_RESPONSES_PATHS = ["openai.resources.responses", "openai.resources.responses.responses"]


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

    completions = _import_class(_CHAT_PATHS, "Completions")
    if completions is None:
        _warnings.warn_once(
            "openai_layout",
            "openai package internals changed; instrumentation disabled.",
        )
        return

    completions._agentping_original_create = completions.create
    completions.create = _make_sync(completions._agentping_original_create, _build_payload, _ChatChunkAccumulator)

    async_completions = _import_class(_CHAT_PATHS, "AsyncCompletions")
    if async_completions is not None:
        async_completions._agentping_original_create = async_completions.create
        async_completions.create = _make_async(
            async_completions._agentping_original_create, _build_payload, _ChatChunkAccumulator
        )

    responses = _import_class(_RESPONSES_PATHS, "Responses")
    if responses is not None:
        responses._agentping_original_create = responses.create
        responses.create = _make_sync(
            responses._agentping_original_create, _build_responses_payload, _ResponsesEventAccumulator
        )

    async_responses = _import_class(_RESPONSES_PATHS, "AsyncResponses")
    if async_responses is not None:
        async_responses._agentping_original_create = async_responses.create
        async_responses.create = _make_async(
            async_responses._agentping_original_create, _build_responses_payload, _ResponsesEventAccumulator
        )

    _PATCHED = True


def uninstrument() -> None:
    global _PATCHED
    for paths, name in (
        (_CHAT_PATHS, "Completions"),
        (_CHAT_PATHS, "AsyncCompletions"),
        (_RESPONSES_PATHS, "Responses"),
        (_RESPONSES_PATHS, "AsyncResponses"),
    ):
        cls = _import_class(paths, name)
        if cls is not None and hasattr(cls, "_agentping_original_create"):
            cls.create = cls._agentping_original_create
            delattr(cls, "_agentping_original_create")
    _PATCHED = False
