"""Process-wide singleton holding the live SDK state."""

from __future__ import annotations

import atexit
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import _ids, _warnings
from ._client import Client
from ._queue import BoundedQueue
from ._worker import Worker

KNOWN_REGIONS = ("eu", "us")
DEFAULT_REGION = "eu"
DEFAULT_QUEUE_SIZE = 1000


def default_base_url_for(region: str) -> str:
    """Regional ingest hostname for the SDK to call (spec §4.1, §2.9).

    Falls back to the EU host if the region is unknown; the edge will
    return 401 region_mismatch if the credential actually targets a
    different region, which surfaces the misconfiguration clearly.
    """
    r = region if region in KNOWN_REGIONS else DEFAULT_REGION
    return f"https://{r}.ingest.agentping.io"


def default_control_url_for(region: str) -> str:
    """Control-plane hostname for the SDK to call (guard-checks-spec).

    The guard gate is served by the control plane, not the ingest edge, so
    it has its own base URL. The EU control plane is served at the apex;
    other regions are subdomained.
    """
    r = region if region in KNOWN_REGIONS else DEFAULT_REGION
    return "https://agentping.io" if r == "eu" else f"https://{r}.agentping.io"


@dataclass
class State:
    api_key: str = ""
    base_url: str = ""
    control_url: str = ""
    region: str = DEFAULT_REGION
    queue: BoundedQueue | None = None
    worker: Worker | None = None
    client: Client | None = None
    control_client: Client | None = None
    initialized: bool = False
    transport: httpx.BaseTransport | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = State()
_atexit_registered = False


def state() -> State:
    return _state


def init(
    api_key: str | None = None,
    base_url: str | None = None,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    transport: httpx.BaseTransport | None = None,
    control_url: str | None = None,
) -> None:
    global _atexit_registered

    with _state.lock:
        key = api_key or os.environ.get("AGENTPING_API_KEY", "")
        region = _ids.extract_region(key)
        url = (
            base_url
            or os.environ.get("AGENTPING_BASE_URL")
            or default_base_url_for(region)
        )
        control = (
            control_url
            or os.environ.get("AGENTPING_CONTROL_URL")
            or default_control_url_for(region)
        )

        if not key:
            _warnings.warn_once(
                "auth_error",
                "no API key provided; telemetry disabled. Set AGENTPING_API_KEY or call init(api_key=...).",
            )

        if _state.initialized:
            _shutdown_locked()

        _state.api_key = key
        _state.base_url = url
        _state.control_url = control
        _state.region = region
        _state.queue = BoundedQueue(maxsize=queue_size)
        _state.transport = transport
        _state.client = Client(api_key=key, base_url=url, transport=transport)
        _state.control_client = Client(api_key=key, base_url=control, transport=transport)
        _state.worker = Worker(queue=_state.queue, client=_state.client)
        _state.initialized = True

        if key:
            _state.worker.start()

    if not _atexit_registered:
        atexit.register(_final_flush)
        _atexit_registered = True


def enqueue(path: str, body: dict[str, Any]) -> None:
    s = _state
    if not s.initialized:
        return
    if not s.api_key:
        return
    assert s.queue is not None
    s.queue.put({"path": path, "body": body, "attempts": 0})


def flush(timeout: float = 5.0) -> None:
    s = _state
    if not s.initialized or s.worker is None:
        return
    deadline = time.monotonic() + timeout
    s.worker.flush_now(deadline)


def status() -> dict[str, Any]:
    s = _state
    if not s.initialized or s.queue is None or s.worker is None:
        return {
            "queue_size": 0,
            "dropped_count": 0,
            "last_flush_ts": None,
            "last_error": None,
        }
    return {
        "queue_size": len(s.queue),
        "dropped_count": s.queue.dropped,
        "last_flush_ts": s.worker.last_flush_ts,
        "last_error": s.worker.last_error,
    }


def _shutdown_locked() -> None:
    if _state.worker is not None:
        _state.worker.stop(timeout=1.0)
    if _state.client is not None:
        _state.client.close()
    if _state.control_client is not None:
        _state.control_client.close()
    _state.worker = None
    _state.client = None
    _state.control_client = None
    _state.queue = None
    _state.initialized = False


def shutdown_for_tests() -> None:
    with _state.lock:
        _shutdown_locked()
    _warnings.reset_for_tests()


def _final_flush() -> None:
    try:
        flush(timeout=5.0)
    except Exception:
        pass
    with _state.lock:
        if _state.worker is not None:
            _state.worker.stop(timeout=1.0)
        if _state.client is not None:
            _state.client.close()
