"""Helpers shared by every instrumentation module.

The wire shape these produce matches the Laravel and TypeScript SDKs:

    llm_call   provider, model, input_tokens (gross, including any cached
               prefix), output_tokens, cached_input_tokens,
               cache_creation_input_tokens, reasoning_tokens, latency_ms,
               ttft_ms (streams: request start to first content chunk),
               finish_reason, tool_calls (count), stream: true, and when
               init(capture_llm_content=True): input (list of {role,
               content}), output (text), system_instructions
    llm_call   status: "error", error, exception, latency_ms when the
               provider raised
    tool_call  tool, status, tool_invocation_id, latency_ms, input, output,
               error, exception
    error      message, exception

`input_tokens` is gross because ingest subtracts `cached_input_tokens`
itself when it prices a call. Providers that report the prompt net of the
cache (Anthropic, Bedrock Converse) need the cached counts added back.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .. import _state
from ._streaming import emit_event, emit_llm_call

DEFAULT_TOOL_PAYLOAD_MAX_CHARS = 4000


def latency_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


class FirstChunkClock:
    """Records when a stream's first content chunk arrived.

    Accumulators mix this in and call ``mark_content()`` from ``observe`` when
    a chunk carries content (text or a tool call), not on housekeeping events
    such as Anthropic's ``message_start``. ``ttft_ms(start)`` is the time to
    first token the Pulse page charts.
    """

    _first_content: float | None = None

    def mark_content(self) -> None:
        if self._first_content is None:
            self._first_content = time.monotonic()

    def ttft_ms(self, start: float) -> int | None:
        if self._first_content is None:
            return None
        return max(0, int((self._first_content - start) * 1000))

    def put_ttft(self, payload: dict[str, Any], start: float) -> None:
        ttft = self.ttft_ms(start)
        if ttft is not None:
            payload["ttft_ms"] = ttft


def content_capture_enabled() -> bool:
    return bool(_state.state().capture_llm_content)


def _content_max_chars() -> int:
    return int(_state.state().llm_content_max_chars)


def clip_text(text: str, max_chars: int | None = None) -> str:
    limit = _content_max_chars() if max_chars is None else max_chars
    if limit > 0 and len(text) > limit:
        return text[:limit] + "..."
    return text


def message_text(content: Any) -> str:
    """Flatten a message's content: a string as is, a list of parts joined,
    with non-text parts (images, tool calls, tool results) named in brackets."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
                continue
            text = _get(part, "text")
            if isinstance(text, str):
                pieces.append(text)
                continue
            kind = _get(part, "type")
            name = _get(part, "name") or _get(part, "tool_name") or _get(part, "id")
            if isinstance(kind, str):
                pieces.append(f"[{kind}{' ' + name if isinstance(name, str) else ''}]")
            else:
                pieces.append(stringify_payload(part, 200) or "")
        return "\n".join(pieces)
    if content is None:
        return ""
    return stringify_payload(content, 200) or ""


def render_messages(messages: Any, default_role: str = "user") -> list[dict[str, str]]:
    """A provider's message list as the wire's [{role, content}], each clipped."""
    if isinstance(messages, str):
        return [{"role": default_role, "content": clip_text(messages)}]
    out: list[dict[str, str]] = []
    for message in messages or []:
        if isinstance(message, str):
            out.append({"role": default_role, "content": clip_text(message)})
            continue
        role = _get(message, "role")
        content = _get(message, "content")
        if content is None:
            content = _get(message, "parts")
        out.append({
            "role": role if isinstance(role, str) and role else default_role,
            "content": clip_text(message_text(content)),
        })
    return out


def put_content(payload: dict[str, Any], *, input: Any = None, output: Any = None, system: Any = None) -> None:
    """Add captured prompt and completion text when the SDK is configured for it."""
    if not content_capture_enabled():
        return
    if input is not None:
        rendered = render_messages(input)
        if rendered:
            payload["input"] = rendered
    if system is not None:
        text = message_text(system) if not isinstance(system, str) else system
        if text:
            payload["system_instructions"] = clip_text(text)
    if output is not None:
        text = output if isinstance(output, str) else message_text(output)
        if text:
            payload["output"] = clip_text(text)


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def error_fields(exc: BaseException) -> dict[str, str]:
    return {
        "error": str(exc)[:500] or type(exc).__name__,
        "exception": type(exc).__name__,
    }


def put_if_positive(payload: dict[str, Any], key: str, value: Any) -> None:
    try:
        if value is not None and int(value) > 0:
            payload[key] = int(value)
    except (TypeError, ValueError):
        pass


def put_if_str(payload: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, str) and value != "":
        payload[key] = value
    elif value is not None and hasattr(value, "name") and isinstance(value.name, str):
        # Enums such as Gemini's FinishReason.
        payload[key] = value.name


def stringify_payload(value: Any, max_chars: int = DEFAULT_TOOL_PAYLOAD_MAX_CHARS) -> str | None:
    """Render a tool input or output for the timeline, capped at max_chars."""
    if value is None:
        return None
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(value, default=_json_default, ensure_ascii=False)
        except Exception:
            rendered = repr(value)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + "..."
    return rendered


def _json_default(value: Any) -> Any:
    for attr in ("model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return repr(value)


def emit_errored_llm_call(
    provider: str,
    model: str | None,
    start: float,
    exc: BaseException,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "provider": provider,
        "model": model or "unknown",
        "latency_ms": latency_ms(start),
        "status": "error",
        **error_fields(exc),
    }
    if extra:
        payload.update(extra)
    emit_llm_call(payload)


def emit_error(exc: BaseException, extra: dict[str, Any] | None = None) -> None:
    fields = error_fields(exc)
    payload: dict[str, Any] = {"message": fields["error"], "exception": fields["exception"]}
    if extra:
        payload.update(extra)
    emit_event("error", payload)


class ToolPayloadOptions:
    """Whether tool input and output are sent, and how long they may be."""

    def __init__(
        self,
        capture_tool_payloads: bool = True,
        tool_payload_max_chars: int = DEFAULT_TOOL_PAYLOAD_MAX_CHARS,
    ) -> None:
        self.capture = bool(capture_tool_payloads)
        self.max_chars = int(tool_payload_max_chars)

    def render(self, value: Any) -> str | None:
        if not self.capture:
            return None
        return stringify_payload(value, self.max_chars)
