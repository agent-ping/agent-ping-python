"""AWS Bedrock instrumentation.

Patches `botocore.client.BaseClient._make_api_call` and filters to the
`bedrock-runtime` service. Every Converse, ConverseStream, and InvokeModel
call from any boto3 / botocore-based bedrock client emits an `llm_call`
event with provider="bedrock", the model id, token counts, latency, the
stop reason and the number of tool uses. A call that raises emits an
errored `llm_call` (`status: "error"`, exception class) and re-raises.

Converse follows Anthropic semantics: `inputTokens` excludes the cached
prefix, so the cache read and write counts are added back to give ingest
the gross figure it expects.

ConverseStream reports `ttft_ms` from the first content block event. With
`init(capture_llm_content=True)` Converse and ConverseStream also carry
`messages` as `input`, `system` as `system_instructions` and the reply's
text blocks as `output`. InvokeModel bodies are provider-specific bytes and
are not captured.

Why this patch point: boto3 clients are created dynamically by service
class factories, so patching the BedrockRuntime class itself is fragile.
`_make_api_call` is the single funnel every boto3 operation goes through;
we filter by `self.meta.service_model.service_name` so we never touch
non-bedrock calls.
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
    stringify_payload,
)
from ._streaming import StreamProxy, emit_llm_call

_PATCHED = False
_SUPPORTED_RANGE = ">=1.34.0,<3.0.0"

# Operations we instrument on the bedrock-runtime service.
_CHAT_OPERATIONS = {"Converse", "InvokeModel"}
_STREAM_OPERATIONS = {"ConverseStream", "InvokeModelWithResponseStream"}


def _detect_version() -> str:
    try:
        from importlib.metadata import version

        return version("botocore")
    except Exception:
        return ""


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
        minor = int(version.split(".")[1]) if "." in version else 0
    except (ValueError, IndexError):
        return False
    if major < 1:
        return False
    if major == 1:
        return minor >= 34
    return major < 3


def _usage_fields(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    fields: dict[str, Any] = {}
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    cache_read = int(usage.get("cacheReadInputTokens") or 0)
    cache_write = int(usage.get("cacheWriteInputTokens") or 0)
    if input_tokens is not None:
        fields["input_tokens"] = int(input_tokens) + cache_read + cache_write
    if output_tokens is not None:
        fields["output_tokens"] = int(output_tokens)
    put_if_positive(fields, "cached_input_tokens", cache_read)
    put_if_positive(fields, "cache_creation_input_tokens", cache_write)
    return fields


def _count_tool_uses(response: Any) -> int:
    if not isinstance(response, dict):
        return 0
    message = (response.get("output") or {}).get("message") or {}
    return sum(1 for block in message.get("content") or [] if isinstance(block, dict) and "toolUse" in block)


def _blocks_text(blocks: Any) -> str:
    """Converse content blocks as text: `text` blocks verbatim, anything else
    (toolUse, toolResult, image, document) as a bracketed marker."""
    if isinstance(blocks, str):
        return blocks
    pieces: list[str] = []
    for block in blocks or []:
        if isinstance(block, str):
            pieces.append(block)
        elif isinstance(block, dict):
            if isinstance(block.get("text"), str):
                pieces.append(block["text"])
            elif block:
                kind = next(iter(block))
                name = (block.get(kind) or {}).get("name") if isinstance(block.get(kind), dict) else None
                pieces.append(f"[{kind}{' ' + name if isinstance(name, str) else ''}]")
        else:
            pieces.append(stringify_payload(block, 200) or "")
    return "\n".join(pieces)


def _render_messages(messages: Any) -> list[dict[str, str]] | None:
    if not isinstance(messages, list):
        return None
    out: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        out.append({
            "role": role if isinstance(role, str) and role else "user",
            "content": _blocks_text(message.get("content")),
        })
    return out


def _response_text(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    message = (response.get("output") or {}).get("message") or {}
    return "".join(
        block["text"] for block in message.get("content") or [] if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def _put_converse_content(payload: dict[str, Any], params: dict[str, Any] | None, output: str) -> None:
    params = params if isinstance(params, dict) else {}
    put_content(
        payload,
        input=_render_messages(params.get("messages")),
        system=_blocks_text(params.get("system")) if params.get("system") else None,
        output=output,
    )


def _build_converse_payload(
    model: str, response: Any, latency: int, stream: bool = False, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "bedrock",
        "latency_ms": latency,
    }
    if model:
        payload["model"] = model
    if isinstance(response, dict):
        payload.update(_usage_fields(response.get("usage")))
        put_if_str(payload, "finish_reason", response.get("stopReason"))
        put_if_positive(payload, "tool_calls", _count_tool_uses(response))
    if stream:
        payload["stream"] = True
    _put_converse_content(payload, params, _response_text(response))
    return payload


def _build_invoke_payload(model: str, response: Any, latency: int) -> dict[str, Any]:
    """For InvokeModel, token counts come from response headers."""
    payload: dict[str, Any] = {
        "provider": "bedrock",
        "latency_ms": latency,
    }
    if model:
        payload["model"] = model
    if isinstance(response, dict):
        # boto3 response includes ResponseMetadata.HTTPHeaders by default.
        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        input_tokens = headers.get("x-amzn-bedrock-input-token-count")
        output_tokens = headers.get("x-amzn-bedrock-output-token-count")
        try:
            if input_tokens is not None:
                payload["input_tokens"] = int(input_tokens)
            if output_tokens is not None:
                payload["output_tokens"] = int(output_tokens)
        except (TypeError, ValueError):
            pass
    return payload


def _is_bedrock_runtime(client: Any) -> bool:
    try:
        return client.meta.service_model.service_name == "bedrock-runtime"
    except Exception:
        return False


class _ConverseStreamAccumulator(FirstChunkClock):
    """The terminal `metadata` event carries the final usage block,
    `messageStop` the stop reason, each `contentBlockStart` with a
    `toolUse` is one tool call, and `contentBlockDelta` carries the text."""

    def __init__(self, model: str, params: dict[str, Any] | None = None) -> None:
        self.model = model
        self.usage: Any = None
        self.finish_reason: str | None = None
        self.tool_calls = 0
        self.params = params
        self.text: list[str] = []
        self.capture = content_capture_enabled()

    def observe(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        if "contentBlockStart" in event or "contentBlockDelta" in event:
            self.mark_content()
        delta = (event.get("contentBlockDelta") or {}).get("delta")
        if self.capture and isinstance(delta, dict) and isinstance(delta.get("text"), str):
            self.text.append(delta["text"])
        metadata = event.get("metadata")
        if isinstance(metadata, dict) and metadata.get("usage"):
            self.usage = metadata["usage"]
        stop = event.get("messageStop")
        if isinstance(stop, dict) and stop.get("stopReason"):
            self.finish_reason = stop["stopReason"]
        block_start = event.get("contentBlockStart")
        if isinstance(block_start, dict) and "toolUse" in (block_start.get("start") or {}):
            self.tool_calls += 1

    def payload(self, start: float) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": "bedrock",
            "model": self.model or "unknown",
            "latency_ms": latency_ms(start),
            "stream": True,
        }
        payload.update(_usage_fields(self.usage))
        put_if_str(payload, "finish_reason", self.finish_reason)
        put_if_positive(payload, "tool_calls", self.tool_calls)
        self.put_ttft(payload, start)
        _put_converse_content(payload, self.params, "".join(self.text))
        return payload


def _wrap_converse_stream(result: Any, model: str, start: float, params: dict[str, Any] | None = None) -> Any:
    """Bedrock's ConverseStream returns {"stream": <event iterator>}. Wrap
    the iterator so events pass through unchanged and one llm_call is
    emitted when the stream closes."""
    if not isinstance(result, dict) or "stream" not in result:
        return result

    acc = _ConverseStreamAccumulator(model, params)

    def on_complete() -> None:
        emit_llm_call(acc.payload(start))

    def on_error(exc: BaseException) -> None:
        emit_errored_llm_call("bedrock", model, start, exc, {"stream": True})

    result["stream"] = StreamProxy(result["stream"], acc.observe, on_complete, on_error)
    return result


def instrument() -> None:
    global _PATCHED
    if _PATCHED:
        return

    try:
        from botocore.client import BaseClient
    except ImportError:
        _warnings.warn_once(
            "bedrock_missing",
            "botocore not installed; instrument_bedrock() is a no-op.",
        )
        return

    version = _detect_version()
    if version and not _version_supported(version):
        _warnings.warn_once(
            "bedrock_version",
            f"botocore version {version} outside supported range {_SUPPORTED_RANGE}; instrumentation disabled.",
        )
        return

    original = BaseClient._make_api_call

    def patched(self: Any, operation_name: str, api_params: dict) -> Any:
        if not _is_bedrock_runtime(self):
            return original(self, operation_name, api_params)
        if operation_name not in _CHAT_OPERATIONS and operation_name not in _STREAM_OPERATIONS:
            return original(self, operation_name, api_params)

        model = api_params.get("modelId", "") if isinstance(api_params, dict) else ""
        streaming = operation_name in _STREAM_OPERATIONS
        start = time.monotonic()
        try:
            result = original(self, operation_name, api_params)
        except BaseException as exc:
            emit_errored_llm_call("bedrock", model, start, exc, {"stream": True} if streaming else None)
            raise

        if operation_name == "ConverseStream":
            return _wrap_converse_stream(result, model, start, api_params)

        try:
            if operation_name == "Converse":
                payload = _build_converse_payload(model, result, latency_ms(start), params=api_params)
            else:  # InvokeModel, InvokeModelWithResponseStream
                payload = _build_invoke_payload(model, result, latency_ms(start))
                if streaming:
                    payload["stream"] = True
            emit_llm_call(payload)
        except Exception:
            pass

        return result

    BaseClient._make_api_call = patched
    BaseClient._agentping_original_make_api_call = original

    _PATCHED = True


def uninstrument() -> None:
    global _PATCHED
    try:
        from botocore.client import BaseClient

        if hasattr(BaseClient, "_agentping_original_make_api_call"):
            BaseClient._make_api_call = BaseClient._agentping_original_make_api_call
            delattr(BaseClient, "_agentping_original_make_api_call")
    except Exception:
        pass
    _PATCHED = False
