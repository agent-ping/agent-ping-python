"""Shared fixtures."""

from __future__ import annotations

import pytest

from agentping import _state, _warnings


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _state.shutdown_for_tests()
    _warnings.reset_for_tests()
