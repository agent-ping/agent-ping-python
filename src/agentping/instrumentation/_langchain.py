"""LangChain integration.

Provides `AgentPingCallbackHandler`, a `BaseCallbackHandler` subclass
that emits AgentPing events for every model call, tool invocation and
chain error inside a chain. Attach via:

    chain.invoke(input, config={"callbacks": [agentping.AgentPingCallbackHandler()]})

Constructor argument `run` is optional. When omitted the handler resolves
the active run via the ContextVar set by `agentping.run(...)`, so under a
typical `with agentping.run("agent") as r:` block you can simply pass
`AgentPingCallbackHandler()` with no arg.

Emits:
  - `llm_call` on every model completion, with provider, model, gross
    input tokens, output tokens, cached and reasoning tokens where the
    model reports them, `finish_reason` and the number of tool calls
  - `llm_call` with `status: "error"` when a model call raises
  - on streamed calls (`on_llm_new_token` fired) `stream: true` and
    `ttft_ms`, the time from the model start to the first token that
    carried text or a tool-call chunk
  - with `agentping.init(capture_llm_content=True)`: `input` (the chat
    messages as `{role, content}`, LangChain types mapped human→user,
    ai→assistant, tool→tool), `system_instructions` (the system
    messages) and `output` (the generation's text)
  - `tool_call` when a tool finishes or raises, with its input, output,
    latency and `tool_invocation_id` (the LangChain run id)
  - `error` when a chain raises

Tool input and output are sent by default, capped at 4000 characters.
Pass `capture_tool_payloads=False` to drop them or
`tool_payload_max_chars=` to change the cap.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from .. import _warnings
from ._shared import (
    DEFAULT_TOOL_PAYLOAD_MAX_CHARS,
    FirstChunkClock,
    ToolPayloadOptions,
    content_capture_enabled,
    error_fields,
    latency_ms,
    message_text,
    put_content,
    put_if_positive,
    put_if_str,
)
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


_ROLE_BY_MESSAGE_TYPE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


class _LlmStart(FirstChunkClock):
    """What `on_llm_start` / `on_chat_model_start` knew, kept until the end
    callback: the clock, the model and, when content capture is on, the
    prompt. `streamed` flips when `on_llm_new_token` fires."""

    def __init__(
        self,
        model: str | None,
        provider: str | None,
        input_value: Any = None,
        system: str | None = None,
    ) -> None:
        self.start = time.monotonic()
        self.model = model
        self.provider = provider
        self.input = input_value
        self.system = system
        self.streamed = False


class _ToolStart:
    __slots__ = ("start", "tool", "input")

    def __init__(self, tool: str, input_value: Any) -> None:
        self.start = time.monotonic()
        self.tool = tool
        self.input = input_value


def _get(obj: Any, key: str) -> Any:
    """Read `key` from a dict or an attribute on an object."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


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
        Provider and model come from LangChain's `ls_provider` /
        `ls_model_name` metadata, the model class name, or the response; a
        custom model that reports none of them lands with
        `provider: "langchain"`.
        """

        def __init__(
            self,
            run: Any = None,
            capture_tool_payloads: bool = True,
            tool_payload_max_chars: int = DEFAULT_TOOL_PAYLOAD_MAX_CHARS,
        ) -> None:
            super().__init__()
            self._explicit_run = run
            self._options = ToolPayloadOptions(capture_tool_payloads, tool_payload_max_chars)
            self._llm_starts: dict[UUID, _LlmStart] = {}
            self._tool_starts: dict[UUID, _ToolStart] = {}

        @property
        def _run(self) -> Any:
            return self._explicit_run if self._explicit_run is not None else _active_run()

        # Model calls

        def on_llm_start(
            self,
            serialized: dict[str, Any],
            prompts: list[str],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            input_value = None
            if content_capture_enabled() and prompts:
                input_value = prompts[0]
            self._remember_llm_start(serialized, run_id, kwargs, input_value, None)

        def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[Any]],
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            input_value, system = None, None
            if content_capture_enabled() and messages:
                try:
                    input_value, system = self._convert_messages(messages[0])
                except Exception:
                    input_value, system = None, None
            self._remember_llm_start(serialized, run_id, kwargs, input_value, system)

        def _remember_llm_start(
            self,
            serialized: Any,
            run_id: UUID,
            kwargs: dict[str, Any],
            input_value: Any = None,
            system: str | None = None,
        ) -> None:
            metadata = kwargs.get("metadata") if isinstance(kwargs.get("metadata"), dict) else {}
            model = metadata.get("ls_model_name") or self._extract_model_name(serialized, kwargs)
            provider = metadata.get("ls_provider") or self._provider_from_class(serialized)
            self._llm_starts[run_id] = _LlmStart(
                str(model) if model else None,
                str(provider).lower() if provider else None,
                input_value,
                system,
            )

        def on_llm_new_token(
            self,
            token: Any,
            *,
            chunk: Any = None,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            # Only streamed calls deliver tokens. The first one that carries
            # text or a tool-call chunk is the time to first token; empty
            # housekeeping chunks (Anthropic's message_start) are not.
            start_info = self._llm_starts.get(run_id)
            if start_info is None:
                return
            try:
                start_info.streamed = True
                if token or self._chunk_has_tool_call(chunk):
                    start_info.mark_content()
            except Exception:
                pass

        def on_llm_end(
            self,
            response: Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            start_info = self._llm_starts.pop(run_id, None)
            run = self._run
            if run is None:
                return

            try:
                llm_output = getattr(response, "llm_output", None) or {}
                generation = self._first_generation(response)
                message = getattr(generation, "message", None)
                response_meta = getattr(message, "response_metadata", None) or {}

                model = (
                    llm_output.get("model_name")
                    or llm_output.get("model")
                    or response_meta.get("model_name")
                    or response_meta.get("model")
                    or (start_info.model if start_info else None)
                    or "unknown"
                )
                provider = self._infer_provider(model, llm_output, start_info)

                payload: dict[str, Any] = {
                    "provider": provider,
                    "model": str(model),
                    "latency_ms": latency_ms(start_info.start) if start_info else 0,
                }

                # usage_metadata is LangChain's normalised, gross count;
                # token_usage is the older per-provider summary.
                usage = getattr(message, "usage_metadata", None) or {}
                legacy = llm_output.get("token_usage") or llm_output.get("usage") or {}
                input_tokens = self._first_present(usage, legacy, ("input_tokens", "prompt_tokens", "promptTokens"))
                output_tokens = self._first_present(
                    usage, legacy, ("output_tokens", "completion_tokens", "completionTokens")
                )
                if input_tokens is not None:
                    payload["input_tokens"] = int(input_tokens)
                if output_tokens is not None:
                    payload["output_tokens"] = int(output_tokens)
                input_details = _get(usage, "input_token_details") or {}
                output_details = _get(usage, "output_token_details") or {}
                put_if_positive(payload, "cached_input_tokens", _get(input_details, "cache_read"))
                put_if_positive(payload, "cache_creation_input_tokens", _get(input_details, "cache_creation"))
                put_if_positive(payload, "reasoning_tokens", _get(output_details, "reasoning"))

                generation_info = getattr(generation, "generation_info", None) or {}
                put_if_str(
                    payload,
                    "finish_reason",
                    generation_info.get("finish_reason")
                    or response_meta.get("finish_reason")
                    or response_meta.get("stop_reason"),
                )
                put_if_positive(payload, "tool_calls", len(getattr(message, "tool_calls", None) or []))
                if start_info is not None and start_info.streamed:
                    payload["stream"] = True
                    start_info.put_ttft(payload, start_info.start)
                put_content(
                    payload,
                    input=start_info.input if start_info else None,
                    system=start_info.system if start_info else None,
                    output=self._output_text(generation, message),
                )

                emit_llm_call(payload)
            except Exception:
                pass

        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            start_info = self._llm_starts.pop(run_id, None)
            run = self._run
            if run is None:
                return
            try:
                model = (start_info.model if start_info else None) or "unknown"
                payload: dict[str, Any] = {
                    "provider": self._infer_provider(model, {}, start_info),
                    "model": model,
                    "latency_ms": latency_ms(start_info.start) if start_info else 0,
                    "status": "error",
                    **error_fields(error),
                }
                emit_llm_call(payload)
            except Exception:
                pass

        # Tools

        def on_tool_start(
            self,
            serialized: dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            name = "unknown"
            if isinstance(serialized, dict):
                ids = serialized.get("id")
                name = serialized.get("name") or (ids[-1] if isinstance(ids, list) and ids else None) or "unknown"
            # `inputs` is the structured form LangChain adds for tools that
            # take more than one argument; it beats the stringified one.
            input_value = kwargs.get("inputs") if kwargs.get("inputs") is not None else input_str
            self._tool_starts[run_id] = _ToolStart(str(name), input_value)

        def on_tool_end(
            self,
            output: Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            start_info = self._tool_starts.pop(run_id, None)
            run = self._run
            if run is None:
                return
            try:
                payload = self._tool_call_data(start_info, run_id, "success")
                rendered = self._options.render(self._tool_output(output))
                if rendered is not None:
                    payload["output"] = rendered
                run.event("tool_call", payload)
            except Exception:
                pass

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            start_info = self._tool_starts.pop(run_id, None)
            run = self._run
            if run is None:
                return
            try:
                payload = self._tool_call_data(start_info, run_id, "error")
                payload.update(error_fields(error))
                run.event("tool_call", payload)
            except Exception:
                pass

        def _tool_call_data(self, start_info: _ToolStart | None, run_id: UUID, status: str) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "tool": start_info.tool if start_info else "unknown",
                "status": status,
                "tool_invocation_id": str(run_id),
                "latency_ms": latency_ms(start_info.start) if start_info else 0,
            }
            if start_info is not None:
                rendered = self._options.render(start_info.input)
                if rendered is not None:
                    payload["input"] = rendered
            return payload

        @staticmethod
        def _tool_output(output: Any) -> Any:
            # Tools may return a ToolMessage; its content is what the model saw.
            content = getattr(output, "content", None)
            return content if content is not None else output

        # Chains

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
                fields = error_fields(error)
                run.event("error", {"message": fields["error"], "exception": fields["exception"]})
            except Exception:
                pass

        # Helpers

        @staticmethod
        def _first_generation(response: Any) -> Any:
            generations = getattr(response, "generations", None) or []
            if generations and generations[0]:
                return generations[0][0]
            return None

        @staticmethod
        def _convert_messages(messages: Any) -> tuple[list[dict[str, str]], str | None]:
            """LangChain BaseMessages as the wire's [{role, content}], with the
            system messages split off into `system_instructions`."""
            rendered: list[dict[str, str]] = []
            system: list[str] = []
            for message in messages or []:
                if isinstance(message, str):
                    rendered.append({"role": "user", "content": message})
                    continue
                kind = _get(message, "type")
                role = _ROLE_BY_MESSAGE_TYPE.get(kind) if isinstance(kind, str) else None
                if role is None:
                    explicit = _get(message, "role")
                    role = explicit if isinstance(explicit, str) and explicit else (kind if isinstance(kind, str) else "user")
                content = message_text(_get(message, "content"))
                if role == "system":
                    if content:
                        system.append(content)
                    continue
                rendered.append({"role": role, "content": content})
            return rendered, ("\n".join(system) if system else None)

        @staticmethod
        def _chunk_has_tool_call(chunk: Any) -> bool:
            message = getattr(chunk, "message", None)
            if message is None:
                return False
            return bool(getattr(message, "tool_call_chunks", None) or getattr(message, "tool_calls", None))

        @staticmethod
        def _output_text(generation: Any, message: Any) -> str | None:
            """The generation's text: the chat message's content when there
            is one (it may be a list of blocks), else the plain `text`."""
            if generation is None:
                return None
            if message is not None:
                return message_text(getattr(message, "content", None))
            text = getattr(generation, "text", None)
            return text if isinstance(text, str) else None

        @staticmethod
        def _first_present(usage: Any, legacy: Any, keys: tuple[str, ...]) -> Any:
            for source in (usage, legacy):
                for key in keys:
                    value = _get(source, key)
                    if value is not None:
                        return value
            return None

        @staticmethod
        def _extract_model_name(serialized: Any, kwargs: dict[str, Any]) -> str | None:
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
        def _provider_from_class(serialized: Any) -> str | None:
            """`serialized["id"]` ends with the model class, e.g. ChatAnthropic."""
            if not isinstance(serialized, dict):
                return None
            ids = serialized.get("id")
            if not isinstance(ids, list) or not ids:
                return None
            class_name = str(ids[-1]).lower()
            for needle, provider in (
                ("anthropic", "anthropic"),
                ("openai", "openai"),
                ("google", "gemini"),
                ("gemini", "gemini"),
                ("vertex", "gemini"),
                ("mistral", "mistral"),
                ("cohere", "cohere"),
                ("bedrock", "bedrock"),
            ):
                if needle in class_name:
                    return provider
            return None

        @staticmethod
        def _infer_provider(model: str | None, llm_output: dict[str, Any], start_info: _LlmStart | None) -> str:
            if start_info is not None and start_info.provider:
                return start_info.provider
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
            m = str(model).lower()
            if m.startswith("claude") or "anthropic" in m:
                return "anthropic"
            if m.startswith(("gpt", "o1", "o3", "o4")) or "openai" in m:
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
