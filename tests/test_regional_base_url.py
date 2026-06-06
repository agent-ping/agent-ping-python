"""Region-aware default base_url.

When init() is called without an explicit base_url, the SDK derives the
ingest host from the credential's region segment (spec §2.9, §4.1)."""

import os

import pytest

import agentping
from agentping import _state


@pytest.fixture(autouse=True)
def _shutdown_between_tests():
    yield
    _state.shutdown_for_tests()
    os.environ.pop("AGENTPING_BASE_URL", None)


def test_eu_key_picks_eu_ingest_host():
    agentping.init(api_key="apk_eu_" + "a" * 32)
    assert _state.state().base_url == "https://eu.ingest.agentping.io"
    assert _state.state().region == "eu"


def test_us_key_picks_us_ingest_host():
    agentping.init(api_key="apk_us_" + "f" * 32)
    assert _state.state().base_url == "https://us.ingest.agentping.io"
    assert _state.state().region == "us"


def test_explicit_base_url_overrides_region_default():
    agentping.init(api_key="apk_us_" + "f" * 32, base_url="https://api.test")
    assert _state.state().base_url == "https://api.test"


def test_env_var_overrides_region_default():
    os.environ["AGENTPING_BASE_URL"] = "https://override.example"
    agentping.init(api_key="apk_us_" + "f" * 32)
    assert _state.state().base_url == "https://override.example"


def test_missing_key_falls_back_to_eu_host():
    agentping.init(api_key=None)
    assert _state.state().base_url == "https://eu.ingest.agentping.io"
    assert _state.state().region == "eu"


def test_helper_returns_known_region_url():
    assert _state.default_base_url_for("eu") == "https://eu.ingest.agentping.io"
    assert _state.default_base_url_for("us") == "https://us.ingest.agentping.io"
    assert _state.default_base_url_for("zz") == "https://eu.ingest.agentping.io"
