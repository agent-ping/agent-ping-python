"""Active-run propagation and helpers for emitting events from instrumentation.

Also holds the stream proxies that wrap provider `Stream` objects. The proxies
forward iteration, attribute access, `close()` and context-manager entry to
the wrapped stream, so a caller who uses `with client.messages.create(
stream=True) as stream:` or `stream.close()` still can.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Any, AsyncIterator, Callable, Iterator

_log = logging.getLogger("agentping")
_warned_no_run = False


def emit_event(event_type: str, payload: dict[str, Any]) -> None:
    """Enqueue an event against the active run.

    When the call happens outside an `agentping.run(...)` context there is
    nowhere to attach the event. We log once at debug and drop it rather than
    crash; the user can wrap their call in `agentping.run(...)` to capture it.

    Active-run propagation is via `contextvars.ContextVar`, which is correct
    under both threads and asyncio tasks. Each concurrent task gets its own
    active-run view; events never leak between tasks.
    """
    global _warned_no_run
    active = _active_run()
    if active is None:
        if not _warned_no_run:
            _warned_no_run = True
            _log.debug(
                "agentping: dropped a %s event because no run is active; "
                "wrap the call in agentping.run(...) to record it",
                event_type,
            )
        return
    active.event(event_type, payload)


def emit_llm_call(payload: dict[str, Any]) -> None:
    """Enqueue a free-standing llm_call event against the active run."""
    emit_event("llm_call", payload)


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


def _reset_warnings_for_tests() -> None:
    global _warned_no_run
    _warned_no_run = False


class StreamProxy:
    """Wraps a sync provider stream, observing each chunk as it is yielded.

    `on_chunk` is called with every chunk; `on_complete` once, when the
    stream is exhausted, closed or its context exits; `on_error` if
    iteration raises. Each callback fires at most once except `on_chunk`.
    Anything the wrapped stream exposes (`close()`, `response`, ...) is
    reachable through attribute delegation.
    """

    def __init__(
        self,
        inner: Any,
        on_chunk: Callable[[Any], None],
        on_complete: Callable[[], None],
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._inner = inner
        self._on_chunk = on_chunk
        self._on_complete = on_complete
        self._on_error = on_error
        self._done = False
        self._iter: Iterator[Any] | None = None

    def __iter__(self) -> "StreamProxy":
        return self

    def __next__(self) -> Any:
        if self._iter is None:
            self._iter = iter(self._inner)
        try:
            chunk = next(self._iter)
        except StopIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._fail(exc)
            raise
        try:
            self._on_chunk(chunk)
        except Exception:
            pass
        return chunk

    def __enter__(self) -> "StreamProxy":
        enter = getattr(self._inner, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        exit_ = getattr(self._inner, "__exit__", None)
        if callable(exit_):
            exit_(exc_type, exc, tb)
        self._finish()

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
        self._finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._on_complete()
        except Exception:
            pass

    def _fail(self, exc: BaseException) -> None:
        if self._done:
            return
        self._done = True
        if self._on_error is None:
            return
        try:
            self._on_error(exc)
        except Exception:
            pass


class AsyncStreamProxy:
    """Async twin of `StreamProxy`."""

    def __init__(
        self,
        inner: Any,
        on_chunk: Callable[[Any], None],
        on_complete: Callable[[], None],
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._inner = inner
        self._on_chunk = on_chunk
        self._on_complete = on_complete
        self._on_error = on_error
        self._done = False
        self._iter: AsyncIterator[Any] | None = None

    def __aiter__(self) -> "AsyncStreamProxy":
        return self

    async def __anext__(self) -> Any:
        if self._iter is None:
            self._iter = self._inner.__aiter__()
        try:
            chunk = await self._iter.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as exc:
            self._fail(exc)
            raise
        try:
            self._on_chunk(chunk)
        except Exception:
            pass
        return chunk

    async def __aenter__(self) -> "AsyncStreamProxy":
        enter = getattr(self._inner, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        exit_ = getattr(self._inner, "__aexit__", None)
        if callable(exit_):
            await exit_(exc_type, exc, tb)
        self._finish()

    async def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result
        self._finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._on_complete()
        except Exception:
            pass

    def _fail(self, exc: BaseException) -> None:
        if self._done:
            return
        self._done = True
        if self._on_error is None:
            return
        try:
            self._on_error(exc)
        except Exception:
            pass


class StreamingTokenAccumulator:
    """Counts tokens off a streamed iterator without delaying yields.

    Kept for backwards compatibility; new instrumentation uses StreamProxy.
    """

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
