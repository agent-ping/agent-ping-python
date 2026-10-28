"""Run lifecycle. Run.id is set immediately, no network round-trip."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from . import _ids, _state


def _iso(ts: datetime | None = None) -> str:
    dt = ts or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"


class Run:
    def __init__(
        self,
        agent: str,
        customer_id: str | None = None,
        feature: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_run_id: str | None = None,
        goal: str | None = None,
        idle_timeout_seconds: int | None = None,
        idle_timeout_status: str | None = None,
        resume: tuple[str, str] | None = None,
        version: str | None = None,
    ) -> None:
        s = _state.state()
        region = s.region or "eu"
        self._region = region
        self.id: str = resume[0] if resume else _ids.new_id("run", region)
        self.agent: str = agent
        self.customer_id = customer_id
        self.feature = feature
        self.goal = goal
        # A run's own version beats the init()-level default.
        self.version: str | None = (version if version is not None else s.version).strip() or None
        self.metadata = metadata or {}
        self.parent_run_id = parent_run_id or os.environ.get("AGENTPING_PARENT_RUN")
        self.started_at = resume[1] if resume else _iso()
        self._finished = False

        body: dict[str, Any] = {
            "id": self.id,
            "agent": agent,
            "started_at": self.started_at,
        }
        if customer_id is not None:
            body["customer_id"] = customer_id
        if feature is not None:
            body["feature"] = feature
        if goal is not None:
            body["goal"] = goal
        if self.version is not None:
            body["version"] = self.version
        if metadata:
            body["metadata"] = metadata
        if self.parent_run_id:
            body["parent_run_id"] = self.parent_run_id
        if idle_timeout_seconds is not None:
            body["idle_timeout_seconds"] = idle_timeout_seconds
            if idle_timeout_status is not None:
                body["idle_timeout_status"] = idle_timeout_status

        _state.enqueue("/v1/runs", body)

    def event(self, type: str, data: dict[str, Any] | None = None, turn: int | None = None) -> str:
        """Append an event to the run.

        Args:
            turn: Which exchange of a conversation this belongs to, counting from
                1. One turn is often several events -- call the model, run a tool,
                call it again -- and this is what groups them. Omit for work that
                is not a conversation.
        """
        evt_id = _ids.new_id("evt", self._region)
        entry: dict[str, Any] = {
            "external_id": evt_id,
            "type": type,
            "ts": _iso(),
            "data": data or {},
        }
        if turn is not None:
            entry["turn"] = turn
        body = {"events": [entry]}
        _state.enqueue(f"/v1/runs/{self.id}/events", body)
        return evt_id

    def finish(
        self,
        status: str = "success",
        output: dict[str, Any] | None = None,
        scores: dict[str, Any] | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        body: dict[str, Any] = {
            "status": status,
            "finished_at": _iso(),
        }
        if output is not None:
            body["output"] = output
        if scores is not None:
            body["scores"] = scores
        _state.enqueue(f"/v1/runs/{self.id}/finish", body)

    def __enter__(self) -> "Run":
        from .instrumentation._streaming import set_active_run
        self._active_token = set_active_run(self)
        return self

    def _fail_with(self, exc: BaseException) -> None:
        """An unhandled exception inside the `with` block lands as an
        `error` event on the timeline, then the run finishes failed."""
        from .instrumentation._shared import error_fields
        fields = error_fields(exc)
        self.event("error", {"message": fields["error"], "exception": fields["exception"]})
        self.finish(status="failed", output={"error": repr(exc)})

    def __exit__(self, exc_type, exc, tb) -> None:
        from .instrumentation._streaming import reset_active_run
        reset_active_run(getattr(self, "_active_token", None))
        if self._finished:
            return
        if exc is not None:
            self._fail_with(exc)
        else:
            self.finish(status="success")

    async def __aenter__(self) -> "Run":
        from .instrumentation._streaming import set_active_run
        self._active_token = set_active_run(self)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        from .instrumentation._streaming import reset_active_run
        reset_active_run(getattr(self, "_active_token", None))
        if self._finished:
            return
        if exc is not None:
            self._fail_with(exc)
        else:
            self.finish(status="success")
