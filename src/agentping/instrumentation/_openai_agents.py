"""OpenAI Agents SDK instrumentation.

Exposes two surfaces:

1. `AgentPingHooks` - a ready-made `RunHooks` subclass that emits
   `llm_call`, `tool_call`, and `handoff` events for every step in an
   agent run. Attach via `Runner.run(..., hooks=AgentPingHooks())`.

2. `instrument()` - patches `Runner.run` and `Runner.run_sync` so they
   auto-attach `AgentPingHooks` when the caller did not pass any hooks.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _warnings
from ._streaming import emit_llm_call

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


def _build_payload(agent: Any, response: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "openai",
    }
    model = getattr(agent, "model", None)
    if model:
        payload["model"] = model if isinstance(model, str) else getattr(model, "model", "")

    usage = getattr(response, "usage", None)
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is not None:
            payload["input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            payload["output_tokens"] = int(output_tokens)

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

        Attach to Runner.run via `hooks=AgentPingHooks()`. Each LLM call
        the agent makes lands as an `llm_call` event; tool invocations
        land as `tool_call`; handoffs land as `handoff`.
        """

        def __init__(self) -> None:
            self._llm_starts: dict[int, float] = {}

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
            latency_ms = int((time.monotonic() - start) * 1000) if start else 0
            try:
                payload = _build_payload(agent, response)
                payload["latency_ms"] = latency_ms
                emit_llm_call(payload)
            except Exception:
                pass

        async def on_tool_start(
            self,
            context: Any,
            agent: Any,
            tool: Any,
        ) -> None:
            try:
                from ._streaming import _active_run
                active = _active_run()
                if active is not None:
                    active.event("tool_call", {
                        "tool": getattr(tool, "name", "unknown"),
                    })
            except Exception:
                pass

        async def on_handoff(
            self,
            context: Any,
            from_agent: Any,
            to_agent: Any,
        ) -> None:
            try:
                from ._streaming import _active_run
                active = _active_run()
                if active is not None:
                    active.event("handoff", {
                        "from": getattr(from_agent, "name", "unknown"),
                        "to": getattr(to_agent, "name", "unknown"),
                    })
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
