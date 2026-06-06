"""Google Gemini SDK instrumentation.

Wraps `google.genai.models.Models.generate_content` and the async variant
plus the streaming methods. On a successful response, emit an `llm_call`
event to the active run with token counts from `usage_metadata`.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._streaming import emit_llm_call  # noqa: F401  (used inside _wrap_stream_iterator)

_PATCHED = False
_SUPPORTED_RANGE = ">=1.0.0,<3.0.0"


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return 1 <= major < 3


def _build_payload(response: Any, model: str, latency_ms: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "gemini",
        "latency_ms": latency_ms,
    }
    if model:
        payload["model"] = model

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_token_count", None)
        candidates = getattr(usage, "candidates_token_count", None)
        cached = getattr(usage, "cached_content_token_count", None)
        if prompt is not None and cached:
            payload["input_tokens"] = int(prompt) - int(cached)
            payload["cached_input_tokens"] = int(cached)
        elif prompt is not None:
            payload["input_tokens"] = int(prompt)
        if candidates is not None:
            payload["output_tokens"] = int(candidates)

    return payload


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

    # generate_content (sync)
    original_generate = Models.generate_content

    def patched_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "")
        start = time.monotonic()
        result = original_generate(self, *args, **kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            payload = _build_payload(result, model, latency_ms)
            emit_llm_call(payload)
        except Exception:
            pass
        return result

    Models.generate_content = patched_generate
    Models._agentping_original_generate = original_generate

    # generate_content_stream (sync): wrap the iterator so we accumulate
    # the final usage_metadata from the last chunk and fire one event on
    # stream close. Chunks pass through with no delay.
    if hasattr(Models, "generate_content_stream"):
        original_stream = Models.generate_content_stream

        def patched_stream(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "")
            start = time.monotonic()
            iterator = original_stream(self, *args, **kwargs)
            return _wrap_stream_iterator(iterator, model, start)

        Models.generate_content_stream = patched_stream
        Models._agentping_original_stream = original_stream

    # embed_content (sync)
    if hasattr(Models, "embed_content"):
        original_embed = Models.embed_content

        def patched_embed(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "")
            start = time.monotonic()
            result = original_embed(self, *args, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                payload = _build_embed_payload(result, model, latency_ms)
                emit_llm_call(payload)
            except Exception:
                pass
            return result

        Models.embed_content = patched_embed
        Models._agentping_original_embed = original_embed

    try:
        # generate_content (async)
        async_original = AsyncModels.generate_content

        async def patched_async_generate(self: Any, *args: Any, **kwargs: Any) -> Any:
            model = kwargs.get("model", "")
            start = time.monotonic()
            result = await async_original(self, *args, **kwargs)
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                payload = _build_payload(result, model, latency_ms)
                emit_llm_call(payload)
            except Exception:
                pass
            return result

        AsyncModels.generate_content = patched_async_generate
        AsyncModels._agentping_original_generate = async_original

        # embed_content (async)
        if hasattr(AsyncModels, "embed_content"):
            async_embed_original = AsyncModels.embed_content

            async def patched_async_embed(self: Any, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model", "")
                start = time.monotonic()
                result = await async_embed_original(self, *args, **kwargs)
                latency_ms = int((time.monotonic() - start) * 1000)
                try:
                    payload = _build_embed_payload(result, model, latency_ms)
                    emit_llm_call(payload)
                except Exception:
                    pass
                return result

            AsyncModels.embed_content = patched_async_embed
            AsyncModels._agentping_original_embed = async_embed_original
    except Exception:
        pass

    _PATCHED = True


def _build_embed_payload(response: Any, model: str, latency_ms: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "gemini",
        "latency_ms": latency_ms,
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


def _wrap_stream_iterator(iterator: Any, model: str, start: float) -> Any:
    """Wrap a Gemini streaming iterator: pass chunks through, capture final usage."""

    def gen():
        last_usage = None
        last_model = model
        try:
            for chunk in iterator:
                usage = getattr(chunk, "usage_metadata", None)
                if usage is not None:
                    last_usage = usage
                chunk_model = getattr(chunk, "model_version", None)
                if chunk_model:
                    last_model = chunk_model
                yield chunk
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            try:
                if last_usage is not None:
                    prompt = getattr(last_usage, "prompt_token_count", None)
                    candidates = getattr(last_usage, "candidates_token_count", None)
                    cached = getattr(last_usage, "cached_content_token_count", None)
                    payload: dict[str, Any] = {
                        "provider": "gemini",
                        "model": last_model or "unknown",
                        "latency_ms": latency_ms,
                    }
                    if prompt is not None and cached:
                        payload["input_tokens"] = int(prompt) - int(cached)
                        payload["cached_input_tokens"] = int(cached)
                    elif prompt is not None:
                        payload["input_tokens"] = int(prompt)
                    if candidates is not None:
                        payload["output_tokens"] = int(candidates)
                    emit_llm_call(payload)
            except Exception:
                pass

    return gen()


def uninstrument() -> None:
    global _PATCHED
    try:
        from google.genai.models import Models
        for attr, original in (
            ("_agentping_original_generate", "generate_content"),
            ("_agentping_original_stream", "generate_content_stream"),
            ("_agentping_original_embed", "embed_content"),
        ):
            if hasattr(Models, attr):
                setattr(Models, original, getattr(Models, attr))
                delattr(Models, attr)
    except Exception:
        pass
    try:
        from google.genai.models import AsyncModels
        for attr, original in (
            ("_agentping_original_generate", "generate_content"),
            ("_agentping_original_embed", "embed_content"),
        ):
            if hasattr(AsyncModels, attr):
                setattr(AsyncModels, original, getattr(AsyncModels, attr))
                delattr(AsyncModels, attr)
    except Exception:
        pass
    _PATCHED = False
