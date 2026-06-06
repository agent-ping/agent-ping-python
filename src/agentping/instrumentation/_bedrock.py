"""AWS Bedrock instrumentation.

Patches `botocore.client.BaseClient._make_api_call` and filters to the
`bedrock-runtime` service. Every Converse, ConverseStream, and InvokeModel
call from any boto3 / botocore-based bedrock client emits an `llm_call`
event with provider="bedrock", the model id, token counts, and latency.

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
from ._streaming import emit_llm_call

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


def _build_converse_payload(model: str, response: Any, latency_ms: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "bedrock",
        "latency_ms": latency_ms,
    }
    if model:
        payload["model"] = model
    usage = response.get("usage") if isinstance(response, dict) else None
    if usage:
        input_tokens = usage.get("inputTokens")
        output_tokens = usage.get("outputTokens")
        cache_read = usage.get("cacheReadInputTokens")
        cache_write = usage.get("cacheWriteInputTokens")
        if input_tokens is not None and cache_read:
            payload["input_tokens"] = int(input_tokens) - int(cache_read)
            payload["cached_input_tokens"] = int(cache_read)
        elif input_tokens is not None:
            payload["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            payload["output_tokens"] = int(output_tokens)
        if cache_write:
            payload["cache_creation_input_tokens"] = int(cache_write)
    return payload


def _build_invoke_payload(model: str, response: Any, latency_ms: int) -> dict[str, Any]:
    """For InvokeModel, token counts come from response headers."""
    payload: dict[str, Any] = {
        "provider": "bedrock",
        "latency_ms": latency_ms,
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
        start = time.monotonic()
        result = original(self, operation_name, api_params)
        latency_ms = int((time.monotonic() - start) * 1000)

        if operation_name == "ConverseStream":
            return _wrap_converse_stream(result, model, start)

        try:
            if operation_name == "Converse":
                payload = _build_converse_payload(model, result, latency_ms)
            else:  # InvokeModel
                payload = _build_invoke_payload(model, result, latency_ms)
            emit_llm_call(payload)
        except Exception:
            pass

        return result

    BaseClient._make_api_call = patched
    BaseClient._agentping_original_make_api_call = original

    _PATCHED = True


def _wrap_converse_stream(result: Any, model: str, start: float) -> Any:
    """Bedrock's ConverseStream returns {"stream": <event iterator>}.
    The terminal "metadata" event carries the final usage block; we wrap
    the iterator so the events pass through unchanged and we capture the
    final usage to emit one llm_call event when the stream closes."""
    if not isinstance(result, dict) or "stream" not in result:
        return result

    original_stream = result["stream"]

    def wrapped():
        last_usage = None
        try:
            for event in original_stream:
                if isinstance(event, dict) and "metadata" in event:
                    metadata = event["metadata"]
                    if isinstance(metadata, dict):
                        usage = metadata.get("usage")
                        if usage:
                            last_usage = usage
                yield event
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                if last_usage:
                    payload = _build_converse_payload(
                        model,
                        {"usage": last_usage},
                        latency_ms,
                    )
                else:
                    payload = {
                        "provider": "bedrock",
                        "model": model or "unknown",
                        "latency_ms": latency_ms,
                    }
                emit_llm_call(payload)
            except Exception:
                pass

    result["stream"] = wrapped()
    return result


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
