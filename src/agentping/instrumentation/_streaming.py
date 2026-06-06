"""Generic helper for emitting llm_call events after a streamed response completes."""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any, Callable, Iterator

from .. import _ids, _state
from .._run import _iso


def emit_llm_call(payload: dict[str, Any]) -> None:
    """Enqueue a free-standing llm_call event against the active run.

    When the LLM call happens outside an `agentping.run(...)` context, we have
    nowhere to attach the event. We skip silently rather than crash; the user
    can wrap their call in `agentping.run(...)` to capture it.

    Active-run propagation is via `contextvars.ContextVar`, which is correct
    under both threads and asyncio tasks. Each concurrent task gets its own
    active-run view; events never leak between tasks.
    """
    active = _active_run()
    if active is None:
        return
    active.event("llm_call", payload)


_active: ContextVar[Any] = ContextVar("agentping_active_run", default=None)


def set_active_run(r: Any) -> Any:
    """Set the active run for the current context. Returns a token for reset.

    The token must be passed to `reset_active_run`. Pairs with the Run
    context manager / async context manager exit.
    """
    return _active.set(r)


def reset_active_run(token: Any) -> None:
    """Restore the previous active-run binding using the token returned by set."""
    if token is None:
        return
    try:
        _active.reset(token)
    except (LookupError, ValueError):
        # Token from a different context, or already reset; safe to ignore.
        pass


def _active_run() -> Any:
    return _active.get()


class StreamingTokenAccumulator:
    """Counts tokens off a streamed iterator without delaying yields."""

    def __init__(
        self,
        inner: Iterator[Any],
        extract_usage: Callable[[Any], dict[str, int] | None],
        on_complete: Callable[[dict[str, int], float], None],
    ) -> None:
        self._inner = inner
        self._extract = extract_usage
        self._on_complete = on_complete
        self._usage: dict[str, int] = {}
        self._start = time.monotonic()

    def __iter__(self) -> "StreamingTokenAccumulator":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._inner)
        except StopIteration:
            self._finalize()
            raise
        u = self._extract(chunk)
        if u:
            for k, v in u.items():
                self._usage[k] = self._usage.get(k, 0) + v
        return chunk

    def _finalize(self) -> None:
        latency_ms = int((time.monotonic() - self._start) * 1000)
        try:
            self._on_complete(self._usage, latency_ms)
        except Exception:
            pass
