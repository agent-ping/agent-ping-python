"""Bounded queue with drop-oldest semantics.

We can't use `queue.Queue(maxsize=N)` directly because it blocks on `.put()`
when full. We need to drop the oldest entry, increment a counter, and return
immediately without blocking the user thread.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


class BoundedQueue:
    def __init__(self, maxsize: int = 1000) -> None:
        self._maxsize = maxsize
        self._buf: deque[Any] = deque()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._dropped = 0

    def put(self, item: Any) -> None:
        with self._cond:
            if len(self._buf) >= self._maxsize:
                self._buf.popleft()
                self._dropped += 1
            self._buf.append(item)
            self._cond.notify()

    def drain(self, max_items: int) -> list[Any]:
        with self._lock:
            n = min(max_items, len(self._buf))
            out = [self._buf.popleft() for _ in range(n)]
        return out

    def push_front(self, items: list[Any]) -> None:
        """Return items to the front of the queue (for retry)."""
        with self._cond:
            for item in reversed(items):
                if len(self._buf) >= self._maxsize:
                    self._buf.pop()
                    self._dropped += 1
                self._buf.appendleft(item)

    def wait_for_items(self, timeout: float) -> None:
        with self._cond:
            if self._buf:
                return
            self._cond.wait(timeout=timeout)

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def add_dropped(self, n: int) -> None:
        with self._lock:
            self._dropped += n
