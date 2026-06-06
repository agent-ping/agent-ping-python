"""UUIDv7 generation and prefixed-ID construction."""

from __future__ import annotations

import os
import re
import threading
import time

_REGION_PATTERN = re.compile(r"^apk_([a-z]{2})_[0-9a-f]{32}$")
_ID_PATTERN = re.compile(r"^[a-z]+_[a-z]{2}_[0-9a-f]{32}$")

_uuid_lock = threading.Lock()
_last_ms = 0
_last_seq = 0


def uuid7_hex() -> str:
    """Return a UUIDv7 as 32 lowercase hex characters (no dashes).

    Layout: 48-bit ms timestamp, 4 version bits (0111), 12 rand bits,
    2 variant bits (10), 62 rand bits. Monotonic within the same ms via
    a sub-ms counter folded into the random tail.
    """
    global _last_ms, _last_seq
    with _uuid_lock:
        ms = int(time.time() * 1000)
        if ms <= _last_ms:
            ms = _last_ms
            _last_seq += 1
        else:
            _last_ms = ms
            _last_seq = 0
        seq = _last_seq & 0x0FFF

    rand_b = os.urandom(8)
    rand_int = int.from_bytes(rand_b, "big")
    rand_int = (rand_int & ~(0b11 << 62)) | (0b10 << 62)

    high = (ms & 0xFFFFFFFFFFFF) << 80
    high |= 0x7 << 76
    high |= seq << 64
    full = high | rand_int

    return f"{full:032x}"


def extract_region(api_key: str) -> str:
    """Pull the two-letter region from an `apk_<region>_<32 hex>` key.

    Falls back to `eu` if the key doesn't match (validation happens
    elsewhere; we never crash user code from here).
    """
    if not api_key:
        return "eu"
    m = _REGION_PATTERN.match(api_key)
    if not m:
        return "eu"
    return m.group(1)


def new_id(prefix: str, region: str) -> str:
    """Build an `<prefix>_<region>_<32 hex>` identifier."""
    return f"{prefix}_{region}_{uuid7_hex()}"


def is_valid_id(value: str) -> bool:
    return bool(_ID_PATTERN.match(value or ""))
