"""OpenAI Agents SDK instrumentation.

Exposes two surfaces:

1. `AgentPingHooks` - a ready-made `RunHooks` subclass that emits one
   `llm_call` per model call, one `tool_call` per finished tool, and a
   `step` for every handoff and agent turn. Attach via
   `Runner.run(..., hooks=AgentPingHooks())`.

2. `instrument()` - patches `Runner.run` and `Runner.run_sync` so they
   auto-attach `AgentPingHooks` when the caller did not pass any hooks.

Emits:
  - `llm_call` with provider, model, gross input tokens, output tokens,
    cached and reasoning tokens, latency and the number of tool calls the
    model asked for
  - `tool_call` when a tool finishes, with `tool_invocation_id`, its
    arguments, its result and latency (arguments and results can be
    turned off or capped with `capture_tool_payloads` and
    `tool_payload_max_chars`)
  - `step` with `kind: "handoff"` on every agent-to-agent handoff
  - `step` with `kind: "agent"` when an agent finishes its turn

The Agents SDK does not surface a failed model call or a tool exception
through hooks; those land on the run when it fails.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._shared import (
    DEFAULT_TOOL_PAYLOAD_MAX_CHARS,
    ToolPayloadOptions,
    latency_ms,
    put_if_positive,
    put_if_str,
)
from ._streaming import _active_run, emit_event, emit_llm_call

_PATCHED = False
_SUPPORTED_RANGE = ">=0.1.0,<2.0.0"


def _detect_version() -> str:
    try:
        from importlib.metadata import version
        return version("openai-agents")
    except Exception:
        return ""


def _version_supported(version: str) -> bool:
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        return False
    return major < 2


def _model_name(agent: Any) -> str:
    model = getattr(agent, "model", None)
    if isinstance(model, str) and model:
        return model
    for attr in ("model", "model_id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _provider_for(model: str) -> str:
    """The Agents SDK is OpenAI-first; other providers show up as a model prefix
    such as `litellm/anthropic/claude-sonnet-4-5`."""
    parts = model.split("/")
    if len(parts) >= 3 and parts[0] == "litellm":
        return parts[1]
    if len(parts) >= 2 and parts[0]:
        return parts[0]
    return "openai"


def _build_payload(agent: Any, response: Any) -> dict[str, Any]:
    model = _model_name(agent)
    payload: dict[str, Any] = {
        "provider": _provider_for(model),
        "model": model,
    }

    usage = getattr(response, "usage", None)
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        # `input_tokens` is the Responses API count, already gross.
        if input_tokens is not None:
            payload["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            payload["output_tokens"] = int(output_tokens)
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        put_if_positive(payload, "cached_input_tokens", getattr(input_details, "cached_tokens", None))
        put_if_positive(payload, "reasoning_tokens", getattr(output_details, "reasoning_tokens", None))

    tool_calls = 0
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            tool_calls += 1
    put_if_positive(payload, "tool_calls", tool_calls)

    return payload


def _make_hooks_class() -> Any:
    """Lazy-import the hooks base class so this module imports cleanly
    when openai-agents is missing."""
    try:
        from agents import RunHooks
    except ImportError:
        return None

    class AgentPingHooks(RunHooks):
        """RunHooks implementation that emits AgentPing events.

        Attach to Runner.run via `hooks=AgentPingHooks()`. Each model call
        lands as an `llm_call` event, each finished tool as a `tool_call`,
        and handoffs and agent turns as `step` events. Pass `run=` to pin
        the hooks to one run instead of the active one.
        """

        def __init__(
            self,
            run: Any = None,
            capture_tool_payloads: bool = True,
            tool_payload_max_chars: int = DEFAULT_TOOL_PAYLOAD_MAX_CHARS,
        ) -> None:
            self._explicit_run = run
            self._options = ToolPayloadOptions(capture_tool_payloads, tool_payload_max_chars)
            self._llm_starts: dict[int, float] = {}
            self._tool_starts: dict[Any, float] = {}
            self._agent_starts: dict[tuple[int, str], float] = {}

        @property
        def _run(self) -> Any:
            return self._explicit_run if self._explicit_run is not None else _active_run()

        def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
            if self._explicit_run is not None:
                self._explicit_run.event(event_type, payload)
            elif event_type == "llm_call":
                emit_llm_call(payload)
            else:
                emit_event(event_type, payload)

        # Model calls

        async def on_llm_start(
            self,
            context: Any,
            agent: Any,
            system_prompt: Any = None,
            input_items: Any = None,
        ) -> None:
            self._llm_starts[id(context)] = time.monotonic()

        async def on_llm_end(
            self,
            context: Any,
            agent: Any,
            response: Any,
        ) -> None:
            start = self._llm_starts.pop(id(context), None)
            try:
                payload = _build_payload(agent, response)
                payload["latency_ms"] = latency_ms(start) if start else 0
                self._emit("llm_call", payload)
            except Exception:
                pass

        # Tools

        async def on_tool_start(
            self,
            context: Any,
            agent: Any,
            tool: Any,
        ) -> None:
            self._tool_starts[self._tool_key(context, tool)] = time.monotonic()

        async def on_tool_end(
            self,
            context: Any,
            agent: Any,
            tool: Any,
            result: Any,
        ) -> None:
            start = self._tool_starts.pop(self._tool_key(context, tool), None)
            try:
                payload: dict[str, Any] = {
                    "tool": self._tool_name(context, tool),
                    "status": "success",
                    "latency_ms": latency_ms(start) if start else 0,
                }
                put_if_str(payload, "tool_invocation_id", getattr(context, "tool_call_id", None))
                # Function tools run with a ToolContext that carries the raw
                # arguments string; hosted tools do not.
                rendered = self._options.render(getattr(context, "tool_arguments", None))
                if rendered is not None:
                    payload["input"] = rendered
                rendered = self._options.render(result)
                if rendered is not None:
                    payload["output"] = rendered
                self._emit("tool_call", payload)
            except Exception:
                pass

        @staticmethod
        def _tool_name(context: Any, tool: Any) -> str:
            name = getattr(tool, "name", None) or getattr(context, "tool_name", None)
            return str(name) if name else "unknown"

        @staticmethod
        def _tool_key(context: Any, tool: Any) -> Any:
            call_id = getattr(context, "tool_call_id", None)
            if isinstance(call_id, str) and call_id:
                return call_id
            return (id(context), getattr(tool, "name", None) or "unknown")

        # Handoffs and agent turns

        async def on_handoff(
            self,
            context: Any,
            from_agent: Any,
            to_agent: Any,
        ) -> None:
            try:
                self._emit("step", {
                    "kind": "handoff",
                    "from": getattr(from_agent, "name", None) or "unknown",
                    "to": getattr(to_agent, "name", None) or "unknown",
                })
            except Exception:
                pass

        async def on_agent_start(self, context: Any, agent: Any) -> None:
            name = getattr(agent, "name", None) or "unknown"
            self._agent_starts[(id(context), name)] = time.monotonic()

        async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
            name = getattr(agent, "name", None) or "unknown"
            start = self._agent_starts.pop((id(context), name), None)
            try:
                payload: dict[str, Any] = {"kind": "agent", "agent": name}
                if start:
                    payload["latency_ms"] = latency_ms(start)
                self._emit("step", payload)
            except Exception:
                pass

    return AgentPingHooks


# Expose the hooks class at module load (None if package missing).
AgentPingHooks = _make_hooks_class()


def instrument() -> None:
    """Patch Runner.run / Runner.run_sync so hooks are auto-attached."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from agents import Runner
    except ImportError:
        _warnings.warn_once(
            "openai_agents_missing",
            "openai-agents package not installed; instrument_openai_agents() is a no-op.",
        )
        return

    version = _detect_version()
    if not _version_supported(version):
        _warnings.warn_once(
            "openai_agents_version",
            f"openai-agents version {version} outside supported range {_SUPPORTED_RANGE}; instrumentation disabled.",
        )
        return

    hooks_cls = AgentPingHooks
    if hooks_cls is None:
        _warnings.warn_once(
            "openai_agents_layout",
            "openai-agents internals changed; instrumentation disabled.",
        )
        return

    original_run = Runner.run

    async def patched_run(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("hooks") is None:
            kwargs["hooks"] = hooks_cls()
        return await original_run(*args, **kwargs)

    Runner.run = staticmethod(patched_run)
    Runner._agentping_original_run = original_run

    if hasattr(Runner, "run_sync"):
        original_run_sync = Runner.run_sync

        def patched_run_sync(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("hooks") is None:
                kwargs["hooks"] = hooks_cls()
            return original_run_sync(*args, **kwargs)

        Runner.run_sync = staticmethod(patched_run_sync)
        Runner._agentping_original_run_sync = original_run_sync

    _PATCHED = True


def uninstrument() -> None:
    global _PATCHED
    try:
        from agents import Runner
        if hasattr(Runner, "_agentping_original_run"):
            Runner.run = staticmethod(Runner._agentping_original_run)
            delattr(Runner, "_agentping_original_run")
        if hasattr(Runner, "_agentping_original_run_sync"):
            Runner.run_sync = staticmethod(Runner._agentping_original_run_sync)
            delattr(Runner, "_agentping_original_run_sync")
    except Exception:
        pass
    _PATCHED = False
