"""Google Gemini SDK instrumentation.

Wraps `google.genai.models.Models.generate_content`, the streaming and
embedding methods and the async variants. Every call emits one `llm_call`
to the active run: on success with token counts from `usage_metadata`
(`prompt_token_count` is already gross), `finish_reason` and the number of
function calls; on failure with `status: "error"` and the exception class,
after which the exception is re-raised.

Streams report `ttft_ms` from the first chunk carrying a text or function
call part. With `init(capture_llm_content=True)` the call also carries
`contents` as `input` (Gemini's `model` role becomes `assistant`),
`config.system_instruction` as `system_instructions` and the candidate's
text as `output`.
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
_SUPPORTED_RANGE = ">=1.0.0,<3.0.0"


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return 1 <= major < 3


def _usage_fields(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    fields: dict[str, Any] = {}
    prompt = getattr(usage, "prompt_token_count", None)
    candidates = getattr(usage, "candidates_token_count", None)
    if prompt is not None:
        fields["input_tokens"] = int(prompt)
    if candidates is not None:
        fields["output_tokens"] = int(candidates)
    put_if_positive(fields, "cached_input_tokens", getattr(usage, "cached_content_token_count", None))
    put_if_positive(fields, "reasoning_tokens", getattr(usage, "thoughts_token_count", None))
    return fields


def _count_function_calls(response: Any) -> int:
    count = 0
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "function_call", None) is not None:
                count += 1
    return count


def _finish_reason(response: Any) -> Any:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return getattr(candidates[0], "finish_reason", None)


def _attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


_ROLE_MAP = {"model": "assistant"}


def _part_text(part: Any) -> str:
    """One Part as text: its text, or a bracketed marker for anything else."""
    if isinstance(part, str):
        return part
    text = _attr(part, "text")
    if isinstance(text, str):
        return text
    for kind in ("function_call", "function_response"):
        value = _attr(part, kind)
        if value is not None:
            name = _attr(value, "name")
            return f"[{kind}{' ' + name if isinstance(name, str) else ''}]"
    for kind in ("inline_data", "file_data", "executable_code", "code_execution_result"):
        if _attr(part, kind) is not None:
            return f"[{kind}]"
    return message_text(part)


def _render_contents(contents: Any) -> list[dict[str, str]] | None:
    """`contents` as the wire's [{role, content}]. A string, or a list of
    strings and parts, is one user message; a list of Content objects is a
    conversation, with Gemini's `model` role reported as `assistant`."""
    if contents is None:
        return None
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    if not isinstance(contents, (list, tuple)):
        contents = [contents]
    out: list[dict[str, str]] = []
    loose: list[str] = []
    for item in contents:
        parts = None if isinstance(item, str) else _attr(item, "parts")
        if parts is None and (isinstance(item, str) or _attr(item, "role") is None):
            loose.append(_part_text(item))
            continue
        if loose:
            out.append({"role": "user", "content": "\n".join(loose)})
            loose = []
        role = _attr(item, "role")
        role = _ROLE_MAP.get(role, role) if isinstance(role, str) and role else "user"
        out.append({"role": role, "content": "\n".join(_part_text(p) for p in parts or [])})
    if loose:
        out.append({"role": "user", "content": "\n".join(loose)})
    return out


def _system_of(config: Any) -> str | None:
    instruction = _attr(config, "system_instruction") if config is not None else None
    if instruction is None:
        return None
    if isinstance(instruction, str):
        return instruction
    parts = _attr(instruction, "parts")
    if parts is not None:
        return "\n".join(_part_text(p) for p in parts)
    if isinstance(instruction, (list, tuple)):
        return "\n".join(_part_text(p) for p in instruction)
    return _part_text(instruction)


def _text_of(response: Any) -> str:
    """The first candidate's text parts, joined; thought summaries excluded."""
    pieces: list[str] = []
    for candidate in (getattr(response, "candidates", None) or [])[:1]:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str) and not getattr(part, "thought", False):
                pieces.append(text)
    return "".join(pieces)


def _has_content(chunk: Any) -> bool:
    for candidate in (getattr(chunk, "candidates", None) or [])[:1]:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None) or getattr(part, "function_call", None) is not None:
                return True
    return False


def _build_payload(
    response: Any, model: str, latency: int, stream: bool = False, request: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "gemini",
        "latency_ms": latency,
    }
    model = getattr(response, "model_version", None) or model
    if model:
        payload["model"] = model
    payload.update(_usage_fields(getattr(response, "usage_metadata", None)))
    put_if_str(payload, "finish_reason", _finish_reason(response))
    put_if_positive(payload, "tool_calls", _count_function_calls(response))
    if stream:
        payload["stream"] = True
    request = request or {}
    put_content(
        payload,
        input=_render_contents(request.get("contents")),
        system=_system_of(request.get("config")),
        output=_text_of(response),
    )
    return payload


def _build_embed_payload(response: Any, model: str, latency: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "gemini",
        "latency_ms": latency,
        "kind": "embedding",
    }
    if model:
        payload["model"] = model

    # Gemini embed responses carry usage_metadata.total_token_count when present.
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        total = getattr(usage, "total_token_count", None) or getattr(usage, "prompt_token_count", None)
        if total is not None:
            payload["input_tokens"] = int(total)
            payload["output_tokens"] = 0
    return payload


class _ChunkAccumulator(FirstChunkClock):
    """Keeps the last usage_metadata and model seen; counts function calls."""

    def __init__(self, model: str, request: dict[str, Any] | None = None) -> None:
        self.model = model
        self.usage: Any = None
        self.finish_reason: Any = None
        self.tool_calls = 0
        self.request = request or {}
        self.text: list[str] = []
        self.capture = content_capture_enabled()

    def observe(self, chunk: Any) -> None:
        if _has_content(chunk):
            self.mark_content()
        if self.capture:
            self.text.append(_text_of(chunk))
        usage = getattr(chunk, "usage_metadata", None)
        if usage is not None:
            self.usage = usage
        self.model = getattr(chunk, "model_version", None) or self.model
        finish = _finish_reason(chunk)
        if finish:
            self.finish_reason = finish
        self.tool_calls += _count_function_calls(chunk)

    def payload(self, start: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": "gemini",
            "model": self.model or "unknown",
            "latency_ms": latency_ms(start),
            "stream": True,
        }
        payload.update(_usage_fields(self.usage))
        put_if_str(payload, "finish_reason", self.finish_reason)
        put_if_positive(payload, "tool_calls", self.tool_calls)
        self.put_ttft(payload, start)
        put_content(
            payload,
            input=_render_contents(self.request.get("contents")),
            system=_system_of(self.request.get("config")),
            output="".join(self.text),
        )
        return payload


def _wrap_stream(iterator: Any, model: str, start: float, is_async: bool, request: dict[str, Any] | None = None) -> Any:
    acc = _ChunkAccumulator(model, request)

    def on_complete() -> None:
        emit_llm_call(acc.payload(start))

    def on_error(exc: BaseException) -> None:
        emit_errored_llm_call("gemini", acc.model, start, exc, {"stream": True})

    proxy = AsyncStreamProxy if is_async else StreamProxy
    return proxy(iterator, acc.observe, on_complete, on_error)


def instrument() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        import google.genai as genai
    except ImportError:
        _warnings.warn_once(
            "gemini_missing",
            "google-genai package not installed; instrument_gemini() is a no-op.",
        )
        return

    version = getattr(genai, "__version__", "")
    if not _version_supported(version):
        _warnings.warn_once(
            "gemini_version",
            f"google-genai version {version} outside supported range {_SUPPORTED_RANGE}; instrumentation disabled.",
        )
        return

    try:
        from google.genai.models import AsyncModels, Models
    except Exception:
        _warnings.warn_once(
            "gemini_layout",
            "google-genai package internals changed; instrumentation disabled.",
        )
        return

    original_generate = Models.generate_content

    def patched_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "")
        start = time.monotonic()
        try:
            result = original_generate(self, *args, **kwargs)
        except BaseException as exc:
            emit_errored_llm_call("gemini", model, start, exc)
            raise
        try:
            emit_llm_call(_build_payload(result, model, latency_ms(start), request=kwargs))
        except Exception:
            pass
        return result

    Models.generate_content = patched_generate
    Models._agentping_original_generate = original_generate

    if hasattr(Models, "generate_content_stream"):
        original_stream = Models.generate_content_stream

        def patched_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "")
            start = time.monotonic()
            try:
                iterator = original_stream(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("gemini", model, start, exc, {"stream": True})
                raise
            return _wrap_stream(iterator, model, start, is_async=False, request=kwargs)

        Models.generate_content_stream = patched_stream
        Models._agentping_original_stream = original_stream

    if hasattr(Models, "embed_content"):
        original_embed = Models.embed_content

        def patched_embed(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "")
            start = time.monotonic()
            try:
                result = original_embed(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("gemini", model, start, exc, {"kind": "embedding"})
                raise
            try:
                emit_llm_call(_build_embed_payload(result, model, latency_ms(start)))
            except Exception:
                pass
            return result

        Models.embed_content = patched_embed
        Models._agentping_original_embed = original_embed

    try:
        async_original = AsyncModels.generate_content

        async def patched_async_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "")
            start = time.monotonic()
            try:
                result = await async_original(self, *args, **kwargs)
            except BaseException as exc:
                emit_errored_llm_call("gemini", model, start, exc)
                raise
            try:
                emit_llm_call(_build_payload(result, model, latency_ms(start), request=kwargs))
            except Exception:
                pass
            return result

        AsyncModels.generate_content = patched_async_generate
        AsyncModels._agentping_original_generate = async_original

        if hasattr(AsyncModels, "generate_content_stream"):
            async_original_stream = AsyncModels.generate_content_stream

            async def patched_async_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "")
                start = time.monotonic()
                try:
                    iterator = await async_original_stream(self, *args, **kwargs)
                except BaseException as exc:
                    emit_errored_llm_call("gemini", model, start, exc, {"stream": True})
                    raise
                return _wrap_stream(iterator, model, start, is_async=True, request=kwargs)

            AsyncModels.generate_content_stream = patched_async_stream
            AsyncModels._agentping_original_stream = async_original_stream

        if hasattr(AsyncModels, "embed_content"):
            async_embed_original = AsyncModels.embed_content

            async def patched_async_embed(self: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "")
                start = time.monotonic()
                try:
                    result = await async_embed_original(self, *args, **kwargs)
                except BaseException as exc:
                    emit_errored_llm_call("gemini", model, start, exc, {"kind": "embedding"})
                    raise
                try:
                    emit_llm_call(_build_embed_payload(result, model, latency_ms(start)))
                except Exception:
                    pass
                return result

            AsyncModels.embed_content = patched_async_embed
            AsyncModels._agentping_original_embed = async_embed_original
    except Exception:
        pass

    _PATCHED = True


def uninstrument() -> None:
    global _PATCHED
    for cls_name in ("Models", "AsyncModels"):
        try:
            module = __import__("google.genai.models", fromlist=[cls_name])
            cls = getattr(module, cls_name)
        except Exception:
            continue
        for attr, original in (
            ("_agentping_original_generate", "generate_content"),
            ("_agentping_original_stream", "generate_content_stream"),
            ("_agentping_original_embed", "embed_content"),
        ):
            if hasattr(cls, attr):
                setattr(cls, original, getattr(cls, attr))
                delattr(cls, attr)
    _PATCHED = False
