"""Helpers shared by every instrumentation module.

The wire shape these produce matches the Laravel and TypeScript SDKs:

    llm_call   provider, model, input_tokens (gross, including any cached
               prefix), output_tokens, cached_input_tokens,
               cache_creation_input_tokens, reasoning_tokens, latency_ms,
               finish_reason, tool_calls (count), stream: true
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

from ._streaming import emit_event, emit_llm_call

DEFAULT_TOOL_PAYLOAD_MAX_CHARS = 4000


def latency_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


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
