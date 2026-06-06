"""LlamaIndex integration.

Provides `AgentPingLlamaIndexHandler`, a `BaseCallbackHandler` subclass
that emits AgentPing events for every LLM call, embedding call, retrieve
event, and exception inside a LlamaIndex pipeline. Attach via:

    from llama_index.core.callbacks import CallbackManager
    Settings.callback_manager = CallbackManager([
        agentping.AgentPingLlamaIndexHandler(),
    ])

Constructor argument `run` is optional. When omitted the handler
resolves the active run via the ContextVar set by `agentping.run(...)`.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._streaming import _active_run, emit_llm_call


_SUPPORTED_RANGE = ">=0.10.0,<2.0.0"


def _detect_version() -> str:
    for name in ("llama-index-core", "llama-index"):
        try:
            from importlib.metadata import version
            return version(name)
        except Exception:
            continue
    return ""


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
        minor = int(version.split(".")[1]) if "." in version else 0
    except (ValueError, IndexError):
        return False
    if major == 0:
        return minor >= 10
    return major < 2


def _make_handler_class() -> Any:
    """Lazy-build the handler class so importing agentping never forces
    a hard dependency on llama-index being installed."""
    try:
        from llama_index.core.callbacks import CBEventType, EventPayload
        from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    except ImportError:
        _warnings.warn_once(
            "llamaindex_missing",
            "llama-index-core not installed; AgentPingLlamaIndexHandler will fail if instantiated.",
        )
        return None

    version = _detect_version()
    if version and not _version_supported(version):
        _warnings.warn_once(
            "llamaindex_version",
            f"llama-index version {version} outside supported range {_SUPPORTED_RANGE}; handler may not work.",
        )

    class AgentPingLlamaIndexHandler(BaseCallbackHandler):
        """LlamaIndex BaseCallbackHandler that emits AgentPing events.

        Attach via `Settings.callback_manager = CallbackManager([handler])`,
        or per-query via `callback_manager=CallbackManager([handler])`.

        Emits:
          - `llm_call` on every LLM call (with usage from response.raw or response.usage)
          - `llm_call` with kind="embedding" on every embedding call
          - `retrieve` on every retriever query
          - `error` on caught exceptions
        """

        def __init__(self, run: Any = None) -> None:
            super().__init__(
                event_starts_to_ignore=[],
                event_ends_to_ignore=[],
            )
            self._explicit_run = run
            self._event_starts: dict[str, tuple[float, str | None]] = {}

        @property
        def _run(self) -> Any:
            return self._explicit_run if self._explicit_run is not None else _active_run()

        def start_trace(self, trace_id: str | None = None) -> None:
            # We don't open a child run per trace; events attach to the
            # active AgentPing run set by the surrounding `with agentping.run(...)`.
            pass

        def end_trace(
            self,
            trace_id: str | None = None,
            trace_map: dict | None = None,
        ) -> None:
            pass

        def on_event_start(
            self,
            event_type: Any,
            payload: dict | None = None,
            event_id: str = "",
            parent_id: str = "",
            **kwargs: Any,
        ) -> str:
            model = self._extract_model_name(payload)
            self._event_starts[event_id] = (time.monotonic(), model)
            return event_id

        def on_event_end(
            self,
            event_type: Any,
            payload: dict | None = None,
            event_id: str = "",
            **kwargs: Any,
        ) -> None:
            start_info = self._event_starts.pop(event_id, None)
            start = start_info[0] if start_info else time.monotonic()
            requested_model = start_info[1] if start_info else None
            latency_ms = int((time.monotonic() - start) * 1000)

            run = self._run
            if run is None:
                return

            try:
                if event_type == CBEventType.LLM:
                    self._emit_llm(payload, latency_ms, requested_model)
                elif event_type == CBEventType.EMBEDDING:
                    self._emit_embedding(payload, latency_ms, requested_model)
                elif event_type == CBEventType.RETRIEVE and payload:
                    nodes = payload.get(EventPayload.NODES, [])
                    run.event("retrieve", {"node_count": len(nodes)})
                elif event_type == CBEventType.EXCEPTION and payload:
                    exc = payload.get("exception")
                    if exc:
                        run.event("error", {"message": str(exc)[:500]})
            except Exception:
                pass

        def _emit_llm(self, payload: dict | None, latency_ms: int, requested_model: str | None) -> None:
            if not payload:
                return

            response = payload.get(EventPayload.RESPONSE)
            messages = payload.get(EventPayload.MESSAGES)
            model = (
                payload.get(EventPayload.MODEL_NAME)
                or requested_model
                or self._extract_model_from_response(response)
                or "unknown"
            )
            provider = self._infer_provider(model)

            usage = self._extract_usage(response)
            data: dict[str, Any] = {
                "provider": provider,
                "model": model,
                "latency_ms": latency_ms,
            }
            if usage.get("input_tokens") is not None:
                data["input_tokens"] = int(usage["input_tokens"])
            if usage.get("output_tokens") is not None:
                data["output_tokens"] = int(usage["output_tokens"])
            if usage.get("cached_input_tokens"):
                data["cached_input_tokens"] = int(usage["cached_input_tokens"])

            emit_llm_call(data)

        def _emit_embedding(self, payload: dict | None, latency_ms: int, requested_model: str | None) -> None:
            model = (
                (payload or {}).get(EventPayload.MODEL_NAME)
                or requested_model
                or "unknown"
            )
            provider = self._infer_provider(model)

            # Count chunks/embeddings as a proxy for input units. The
            # rate card prices by token; if the LlamaIndex embedder hands
            # back a tokens_used hint use that, otherwise leave the cost
            # to be reconciled when the provider invoice lands.
            input_tokens = 0
            if payload:
                chunks = payload.get(EventPayload.CHUNKS) or []
                if chunks:
                    # Rough: 1 token ≈ 4 chars. This is a fallback when
                    # the embedder doesn't return usage info.
                    input_tokens = sum(len(str(c)) // 4 for c in chunks)

            data: dict[str, Any] = {
                "provider": provider,
                "model": model,
                "kind": "embedding",
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": 0,
            }
            emit_llm_call(data)

        @staticmethod
        def _extract_model_name(payload: dict | None) -> str | None:
            if not payload:
                return None
            value = payload.get(EventPayload.MODEL_NAME)
            if value:
                return str(value)
            serialized = payload.get(EventPayload.SERIALIZED)
            if isinstance(serialized, dict):
                for key in ("model", "model_name", "deployment_name"):
                    v = serialized.get(key)
                    if v:
                        return str(v)
            return None

        @staticmethod
        def _extract_model_from_response(response: Any) -> str | None:
            if response is None:
                return None
            raw = getattr(response, "raw", None)
            if isinstance(raw, dict):
                value = raw.get("model") or raw.get("model_name")
                if value:
                    return str(value)
            return None

        @staticmethod
        def _extract_usage(response: Any) -> dict[str, Any]:
            """LlamaIndex's response objects carry usage in different
            places depending on the underlying LLM. Try them in order."""
            usage: dict[str, Any] = {}
            if response is None:
                return usage

            # 1. response.raw is a dict from the underlying provider.
            raw = getattr(response, "raw", None)
            if isinstance(raw, dict):
                u = raw.get("usage") or {}
                if isinstance(u, dict):
                    # OpenAI / Mistral / Cohere shape
                    if "prompt_tokens" in u:
                        usage["input_tokens"] = u["prompt_tokens"]
                    if "completion_tokens" in u:
                        usage["output_tokens"] = u["completion_tokens"]
                    # Anthropic shape
                    if "input_tokens" in u:
                        usage["input_tokens"] = u["input_tokens"]
                    if "output_tokens" in u:
                        usage["output_tokens"] = u["output_tokens"]
                    # Cached input
                    details = u.get("prompt_tokens_details") or {}
                    cached = details.get("cached_tokens") if isinstance(details, dict) else None
                    if cached:
                        usage["cached_input_tokens"] = cached

            # 2. response.additional_kwargs sometimes holds usage too.
            additional = getattr(response, "additional_kwargs", None)
            if isinstance(additional, dict):
                u = additional.get("usage") or {}
                if isinstance(u, dict) and "input_tokens" not in usage:
                    if "prompt_tokens" in u:
                        usage["input_tokens"] = u["prompt_tokens"]
                    if "completion_tokens" in u:
                        usage["output_tokens"] = u["completion_tokens"]

            return usage

        @staticmethod
        def _infer_provider(model: str | None) -> str:
            if not model:
                return "llamaindex"
            m = model.lower()
            if m.startswith("claude") or "anthropic" in m:
                return "anthropic"
            if (
                m.startswith("gpt")
                or m.startswith("o1")
                or m.startswith("o3")
                or m.startswith("text-embedding")
                or "openai" in m
            ):
                return "openai"
            if m.startswith("gemini") or "google" in m:
                return "gemini"
            if m.startswith("mistral") or "mixtral" in m:
                return "mistral"
            if m.startswith("command") or m.startswith("embed-") or "cohere" in m:
                return "cohere"
            return "llamaindex"

    return AgentPingLlamaIndexHandler


AgentPingLlamaIndexHandler = _make_handler_class()
