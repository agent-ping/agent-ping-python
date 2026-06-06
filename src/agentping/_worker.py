"""Background flush worker.

Batches telemetry messages into HTTP POSTs. Up to 50 per batch. Flushes
every 2s or when the buffer reaches 50, whichever first. Exponential
back-off on transport errors and 5xx. Respects Retry-After.

Telemetry messages are dicts with shape:
    {"path": "/v1/runs", "body": {...}, "attempts": int}
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx

from . import _warnings
from ._client import Client
from ._queue import BoundedQueue

BATCH_SIZE = 50
FLUSH_INTERVAL = 2.0
MAX_ATTEMPTS = 5
RATE_LIMIT_INTERVAL = 1.0


class Worker:
    def __init__(self, queue: BoundedQueue, client: Client) -> None:
        self._queue = queue
        self._client = client
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_flush_ts: float | None = None
        self._last_error: str | None = None
        self._backoff_until: float = 0.0
        self._backoff_secs: float = 1.0
        self._min_next_send: float = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="agentping-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._queue.wake()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                self._last_error = repr(exc)

    def _tick(self) -> None:
        now = time.monotonic()
        if now < self._backoff_until or now < self._min_next_send:
            wait = max(self._backoff_until, self._min_next_send) - now
            self._queue.wait_for_items(timeout=max(wait, 0.05))
            return

        if len(self._queue) < BATCH_SIZE:
            self._queue.wait_for_items(timeout=FLUSH_INTERVAL)

        items = self._queue.drain(BATCH_SIZE)
        if not items:
            return

        self._send(items)

    def flush_now(self, deadline_monotonic: float) -> None:
        first = True
        while time.monotonic() < deadline_monotonic:
            # Honour backoff between retries (e.g. an event waiting for its
            # run-start to be persisted) but never let a stale backoff block
            # the first attempt of an explicit flush.
            now = time.monotonic()
            if not first and now < self._backoff_until:
                wait = min(self._backoff_until, deadline_monotonic) - now
                if wait > 0:
                    time.sleep(wait)
            first = False

            items = self._queue.drain(BATCH_SIZE)
            if not items:
                return
            self._send(items, override_backoff=False)

    def _send(self, items: list[dict[str, Any]], override_backoff: bool = False) -> None:
        failed: list[dict[str, Any]] = []
        dropped = 0
        for item in items:
            try:
                resp = self._client.post(item["path"], item["body"])
            except (httpx.HTTPError, Exception) as exc:
                self._last_error = repr(exc)
                item["attempts"] = item.get("attempts", 0) + 1
                if item["attempts"] >= MAX_ATTEMPTS:
                    dropped += 1
                else:
                    failed.append(item)
                continue

            if resp.status_code in (200, 202):
                continue

            if resp.status_code == 429:
                if resp.retry_after is not None and not override_backoff:
                    self._backoff_until = time.monotonic() + resp.retry_after
                item["attempts"] = item.get("attempts", 0) + 1
                if item["attempts"] >= MAX_ATTEMPTS:
                    dropped += 1
                else:
                    failed.append(item)
                continue

            if 500 <= resp.status_code < 600:
                self._last_error = f"server_status_{resp.status_code}"
                item["attempts"] = item.get("attempts", 0) + 1
                if item["attempts"] >= MAX_ATTEMPTS:
                    dropped += 1
                else:
                    failed.append(item)
                continue

            if resp.status_code == 404:
                # Run-start is persisted asynchronously, so an event/finish can
                # arrive before its run row exists. Retry with backoff instead
                # of dropping; a genuinely missing run drops after MAX_ATTEMPTS.
                self._last_error = "run_not_ready_404"
                item["attempts"] = item.get("attempts", 0) + 1
                if item["attempts"] >= MAX_ATTEMPTS:
                    dropped += 1
                else:
                    failed.append(item)
                continue

            if resp.status_code in (400, 401, 403, 409, 422):
                self._last_error = f"client_status_{resp.status_code}"
                continue

            self._last_error = f"status_{resp.status_code}"
            item["attempts"] = item.get("attempts", 0) + 1
            if item["attempts"] >= MAX_ATTEMPTS:
                dropped += 1
            else:
                failed.append(item)

        self._last_flush_ts = time.time()
        self._min_next_send = time.monotonic() + RATE_LIMIT_INTERVAL

        if failed:
            if not override_backoff:
                self._backoff_until = max(
                    self._backoff_until, time.monotonic() + self._backoff_secs
                )
                self._backoff_secs = min(self._backoff_secs * 2, 30.0)
            self._queue.push_front(failed)
        else:
            self._backoff_secs = 1.0

        if dropped:
            self._queue.add_dropped(dropped)
            _warnings.warn_once(
                "network_error",
                "telemetry batch exhausted retries; events dropped.",
            )

    @property
    def last_flush_ts(self) -> float | None:
        return self._last_flush_ts

    @property
    def last_error(self) -> str | None:
        return self._last_error
