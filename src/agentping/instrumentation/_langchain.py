"""LangChain integration.

Provides `AgentPingCallbackHandler`, a `BaseCallbackHandler` subclass
that emits AgentPing events for every LLM call, tool invocation, and
chain error inside a chain. Attach via:

    chain.invoke(input, config={"callbacks": [agentping.AgentPingCallbackHandler()]})

Constructor argument `run` is optional. When omitted the handler resolves
the active run via the ContextVar set by `agentping.run(...)`, so under a
typical `with agentping.run("agent") as r:` block you can simply pass
`AgentPingCallbackHandler()` with no arg.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from .. import _warnings
from ._streaming import _active_run, emit_llm_call


_SUPPORTED_RANGE = ">=0.3.0,<2.0.0"


def _detect_version() -> str:
    for name in ("langchain-core", "langchain_core", "langchain"):
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
        return minor >= 3
    return major < 2


def _make_handler_class() -> Any:
    """Lazy-build the handler class so importing agentping doesn't require
    langchain to be installed."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        _warnings.warn_once(
            "langchain_missing",
            "langchain-core not installed; AgentPingCallbackHandler will fail if instantiated.",
        )
        return None

    version = _detect_version()
    if version and not _version_supported(version):
        _warnings.warn_once(
            "langchain_version",
            f"langchain version {version} outside supported range {_SUPPORTED_RANGE}; handler may not work.",
        )

    class AgentPingCallbackHandler(BaseCallbackHandler):
        """LangChain BaseCallbackHandler that emits AgentPing events.

        Attach to a chain via `config={"callbacks": [AgentPingCallbackHandler()]}`.
        Captures every LLM call (cost), tool invocation (trace), and chain
        error (status) inside the chain. Provider + model are read from the
        LLMResult's llm_output; if a custom LLM doesn't populate llm_output,
        the event still lands with `provider: "langchain"`.
        """

        def __init__(self, run: Any = None) -> None:
            super().__init__()
            self._explicit_run = run
            self._llm_starts: dict[UUID, tuple[float, str | None]] = {}

        @property
        def _run(self) -> Any:
            return self._explicit_run if self._explicit_run is not None else _active_run()

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            model_name = self._extract_model_name(serialized, kwargs)
            self._llm_starts[run_id] = (time.monotonic(), model_name)

        def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[Any]],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            model_name = self._extract_model_name(serialized, kwargs)
            self._llm_starts[run_id] = (time.monotonic(), model_name)

        def on_llm_end(
            self,
            response: Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            start_info = self._llm_starts.pop(run_id, None)
            start = start_info[0] if start_info else time.monotonic()
            requested_model = start_info[1] if start_info else None
            latency_ms = int((time.monotonic() - start) * 1000)

            run = self._run
            if run is None:
                return

            try:
                llm_output = getattr(response, "llm_output", None) or {}
                usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
                model = (
                    llm_output.get("model_name")
                    or llm_output.get("model")
                    or requested_model
                    or "unknown"
                )
                provider = self._infer_provider(model, llm_output)

                payload: dict[str, Any] = {
                    "provider": provider,
                    "model": model,
                    "latency_ms": latency_ms,
                }
                # Token-usage shape varies across providers; cover the common ones.
                input_tokens = (
                    usage.get("input_tokens")
                    or usage.get("prompt_tokens")
                    or usage.get("promptTokens")
                )
                output_tokens = (
                    usage.get("output_tokens")
                    or usage.get("completion_tokens")
                    or usage.get("completionTokens")
                )
                if input_tokens is not None:
                    payload["input_tokens"] = int(input_tokens)
                if output_tokens is not None:
                    payload["output_tokens"] = int(output_tokens)

                emit_llm_call(payload)
            except Exception:
                pass

        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            run = self._run
            if run is None:
                return
            try:
                run.event("tool_call", {
                    "tool": serialized.get("name", "unknown"),
                    "args_preview": str(input_str)[:200],
                })
            except Exception:
                pass

        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            run = self._run
            if run is None:
                return
            try:
                run.event("error", {"message": str(error)[:500]})
            except Exception:
                pass

        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            run = self._run
            if run is None:
                return
            try:
                run.event("llm_call_error", {"message": str(error)[:500]})
            except Exception:
                pass

        @staticmethod
        def _extract_model_name(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str | None:
            # LangChain's serialized payload sometimes nests model in id[-1].
            if isinstance(serialized, dict):
                kwargs_dict = serialized.get("kwargs", {}) if isinstance(serialized.get("kwargs"), dict) else {}
                for key in ("model", "model_name", "deployment_name"):
                    value = kwargs_dict.get(key)
                    if value:
                        return str(value)
            invocation = kwargs.get("invocation_params") or {}
            if isinstance(invocation, dict):
                for key in ("model", "model_name"):
                    value = invocation.get(key)
                    if value:
                        return str(value)
            return None

        @staticmethod
        def _infer_provider(model: str | None, llm_output: dict[str, Any]) -> str:
            # Explicit provider in llm_output wins.
            for key in ("system_fingerprint", "provider"):
                value = llm_output.get(key)
                if isinstance(value, str) and value:
                    if "anthropic" in value.lower():
                        return "anthropic"
                    if "openai" in value.lower() or value.startswith("fp_"):
                        return "openai"
                    return value.lower()
            if not model:
                return "langchain"
            m = model.lower()
            if m.startswith("claude") or "anthropic" in m:
                return "anthropic"
            if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3") or "openai" in m:
                return "openai"
            if m.startswith("gemini") or "google" in m:
                return "gemini"
            if m.startswith("mistral") or "mixtral" in m:
                return "mistral"
            if m.startswith("command") or "cohere" in m:
                return "cohere"
            return "langchain"

    return AgentPingCallbackHandler


AgentPingCallbackHandler = _make_handler_class()
