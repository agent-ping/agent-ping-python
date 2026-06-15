"""Guard gate (guard-checks-spec): hard/soft modes, fail-closed default, the
Paused exception, the tier-inactive warning, and 429-as-non-verdict."""

from __future__ import annotations

import httpx
import pytest

import agentping
from agentping import guard


def _transport(handler):
    return httpx.MockTransport(handler)


def _init(handler):
    agentping.init(
        api_key="apk_eu_" + "a" * 32,
        base_url="https://ingest.test",
        control_url="https://control.test",
        transport=_transport(handler),
    )


def _verdict(decision="allow", **extra):
    body = {
        "decision": decision,
        "checked_at": "2026-06-15T09:14:22Z",
        "blocked_by": ["grl_eu_x"] if decision == "block" else [],
        "guard_check_id": "gck_eu_1" if decision == "block" else None,
        "rules": [],
        "stale_as_of": "2026-06-15T09:13:50Z",
        "active": True,
        "inactive_reason": None,
    }
    body.update(extra)
    return body


def test_allow_returns_a_verdict():
    _init(lambda req: httpx.Response(200, json=_verdict("allow")))
    v = guard.check(agent="nightly")
    assert v.decision == "allow"
    assert v.blocked is False


def test_hard_mode_raises_paused_on_block():
    _init(lambda req: httpx.Response(200, json=_verdict("block")))
    with pytest.raises(agentping.Paused) as exc:
        guard.check(agent="nightly")
    assert exc.value.verdict.blocked is True
    assert exc.value.verdict.blocked_by == ["grl_eu_x"]


def test_soft_mode_returns_the_block_verdict_without_raising():
    _init(lambda req: httpx.Response(200, json=_verdict("block")))
    v = guard.check(agent="nightly", mode="soft")
    assert v.blocked is True
    assert v.guard_check_id == "gck_eu_1"


def test_the_call_targets_the_control_plane_with_the_dimensions():
    captured = {}

    def handler(req):
        captured["url"] = str(req.url)
        captured["body"] = req.content.decode()
        return httpx.Response(200, json=_verdict("allow"))

    _init(handler)
    guard.check(customer_ref="cus_abc", agent="nightly", function="run")
    assert captured["url"] == "https://control.test/v1/guard/check"
    assert '"customer_ref":"cus_abc"' in captured["body"].replace(" ", "")
    assert '"agent":"nightly"' in captured["body"].replace(" ", "")


def test_an_inactive_plan_warns_and_allows(capsys):
    _init(lambda req: httpx.Response(200, json=_verdict("allow", active=False, inactive_reason="tier")))
    v = guard.check(agent="nightly")  # hard mode, but inactive is a no-op allow
    assert v.decision == "allow"
    assert v.active is False
    err = capsys.readouterr().err
    assert "unguarded" in err.lower()


def test_unreachable_fails_closed_by_default():
    def handler(req):
        raise httpx.ConnectError("boom")

    _init(handler)
    with pytest.raises(agentping.Paused):
        guard.check(agent="nightly")  # default on_unreachable="block"


def test_unreachable_can_fail_open():
    def handler(req):
        raise httpx.ConnectError("boom")

    _init(handler)
    v = guard.check(agent="nightly", on_unreachable="allow")
    assert v.decision == "allow"


def test_a_429_leans_allow_even_when_fail_closed():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(429, json={"error": "rate_limited"}, headers={"Retry-After": "0"})

    _init(handler)
    v = guard.check(agent="nightly", on_unreachable="block")
    assert v.decision == "allow"  # a throttle is our condition, not a block
    assert calls["n"] == 2  # one retry after a short backoff


def test_a_5xx_follows_on_unreachable():
    _init(lambda req: httpx.Response(503, json={}))
    with pytest.raises(agentping.Paused):
        guard.check(agent="nightly")
