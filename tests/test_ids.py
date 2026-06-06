import re

from agentping import _ids


def test_uuid7_hex_is_32_lowercase_hex():
    h = _ids.uuid7_hex()
    assert len(h) == 32
    assert re.match(r"^[0-9a-f]{32}$", h)


def test_uuid7_version_bits_are_7():
    h = _ids.uuid7_hex()
    assert h[12] == "7"


def test_uuid7_monotonic_within_ms():
    ids = [_ids.uuid7_hex() for _ in range(100)]
    assert len(set(ids)) == 100


def test_new_id_format():
    id_ = _ids.new_id("run", "eu")
    assert _ids.is_valid_id(id_)
    assert id_.startswith("run_eu_")
    assert len(id_) == len("run_eu_") + 32


def test_extract_region_from_key():
    assert _ids.extract_region("apk_eu_" + "a" * 32) == "eu"
    assert _ids.extract_region("apk_us_" + "f" * 32) == "us"


def test_extract_region_falls_back_to_eu():
    assert _ids.extract_region("") == "eu"
    assert _ids.extract_region("invalid") == "eu"


def test_is_valid_id_rejects_garbage():
    assert not _ids.is_valid_id("run_eu_short")
    assert not _ids.is_valid_id("RUN_EU_" + "a" * 32)
    assert not _ids.is_valid_id("evt_eu_" + "g" * 32)
