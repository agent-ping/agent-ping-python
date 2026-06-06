"""One-shot stderr warning per error class."""

from __future__ import annotations

import sys
import threading

_lock = threading.Lock()
_warned: set[str] = set()


def warn_once(error_class: str, message: str) -> None:
    with _lock:
        if error_class in _warned:
            return
        _warned.add(error_class)
    sys.stderr.write(f"[agentping] {message}\n")
    sys.stderr.flush()


def reset_for_tests() -> None:
    with _lock:
        _warned.clear()
