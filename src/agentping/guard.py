"""The guard gate: a top-of-script safety net (guard-checks-spec).

One synchronous call you put at the top of a script (and optionally at other
checkpoints). It asks AgentPing whether to run, given your configured rules and
your real cumulative spend, and either returns a verdict (soft mode) or raises
``Paused`` (hard mode, the default).

Defaults invert the usual observability advice on purpose: a guard you bolt to
the top of a script should fail closed. ``mode`` defaults to ``"hard"`` (raise
on block) and ``on_unreachable`` to ``"block"`` (do not run unless confirmed
safe). This couples your scheduled runs to AgentPing's uptime; set
``on_unreachable="allow"`` if you would rather run when the gate is unreachable.

Honest limitation: this gates the *next invocation* starting, not a single
invocation that passes the check and then loops forever. Drop ``check()`` at
several points to turn it into a series of gates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import _state, _warnings

_PATH = "/v1/guard/check"


@dataclass
class GuardVerdict:
    """The full server verdict (guard-checks-spec, "API contract"). Returned by
    soft mode and carried by the ``Paused`` exception in hard mode."""

    decision: str  # "allow" | "block"
    blocked_by: list[str] = field(default_factory=list)
    rules: list[dict[str, Any]] = field(default_factory=list)
    stale_as_of: str | None = None
    active: bool = True
    inactive_reason: str | None = None
    guard_check_id: str | None = None
    checked_at: str | None = None

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GuardVerdict:
        return cls(
            decision=d.get("decision", "allow"),
            blocked_by=list(d.get("blocked_by") or []),
            rules=list(d.get("rules") or []),
            stale_as_of=d.get("stale_as_of"),
            active=bool(d.get("active", True)),
            inactive_reason=d.get("inactive_reason"),
            guard_check_id=d.get("guard_check_id"),
            checked_at=d.get("checked_at"),
        )


class Paused(Exception):
    """Raised by ``check()`` in hard mode when the gate blocks. Carries the full
    verdict so the traceback can say what blocked and by how much."""

    def __init__(self, verdict: GuardVerdict) -> None:
        self.verdict = verdict
        reason = ", ".join(verdict.blocked_by) or "a guard rule"
        super().__init__(f"AgentPing guard blocked this run ({reason}).")


def check(
    customer_ref: str | None = None,
    agent: str | None = None,
    function: str | None = None,
    environment: str | None = None,
    *,
    mode: str = "hard",
    on_unreachable: str = "block",
) -> GuardVerdict:
    """Ask the gate whether to run.

    In hard mode (default) raises ``Paused`` on a block and returns the verdict
    on an allow. In soft mode always returns the verdict; check ``.blocked``.

    A rate-limited gate (HTTP 429) is our condition, not a safety signal: it is
    retried once and then leans allow even under ``on_unreachable="block"``, so
    our throttle can never halt a legitimate agent. A gate that is unreachable
    follows ``on_unreachable``. An inactive plan is a no-op allow plus a loud,
    one-time warning that the script is running unguarded.
    """
    if mode not in ("hard", "soft"):
        raise ValueError('mode must be "hard" or "soft"')
    if on_unreachable not in ("allow", "block"):
        raise ValueError('on_unreachable must be "allow" or "block"')

    body = {
        k: v
        for k, v in {
            "customer_ref": customer_ref,
            "agent": agent,
            "function": function,
            "environment": environment,
        }.items()
        if v is not None
    }

    s = _state.state()
    client = s.control_client if s.initialized else None
    if client is None or not s.api_key:
        return _unreachable("not_initialized", mode, on_unreachable)

    try:
        r = client.post_raw(_PATH, body)
    except httpx.HTTPError:
        return _unreachable("network", mode, on_unreachable)

    if r.status_code == 429:
        _backoff(r)
        try:
            r = client.post_raw(_PATH, body)
        except httpx.HTTPError:
            r = None
        if r is None or r.status_code == 429:
            # A throttle is our condition, not a safety signal: lean allow.
            return GuardVerdict(decision="allow")

    if r.status_code != 200:
        return _unreachable(f"http_{r.status_code}", mode, on_unreachable)

    try:
        data = r.json()
    except ValueError:
        return _unreachable("bad_response", mode, on_unreachable)
    if not isinstance(data, dict) or "decision" not in data:
        return _unreachable("bad_response", mode, on_unreachable)

    verdict = GuardVerdict.from_dict(data)

    if not verdict.active:
        _warnings.warn_once(
            "guard_inactive_on_plan",
            "guard is inactive on your plan; this script is running UNGUARDED. "
            "Upgrade to the Team or Business plan to enable it.",
        )

    if verdict.blocked and mode == "hard":
        raise Paused(verdict)
    return verdict


def _unreachable(reason: str, mode: str, on_unreachable: str) -> GuardVerdict:
    _warnings.warn_once(
        "guard_unreachable",
        f"guard gate unreachable ({reason}); applying on_unreachable={on_unreachable!r}.",
    )
    decision = "block" if on_unreachable == "block" else "allow"
    verdict = GuardVerdict(
        decision=decision,
        blocked_by=["unreachable"] if decision == "block" else [],
    )
    if verdict.blocked and mode == "hard":
        raise Paused(verdict)
    return verdict


def _backoff(r: httpx.Response) -> None:
    delay = 0.5
    ra = r.headers.get("Retry-After")
    if ra:
        try:
            delay = min(float(ra), 1.0)
        except ValueError:
            pass
    time.sleep(delay)
