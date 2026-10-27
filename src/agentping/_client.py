"""HTTP client wrapper. 2-second hard timeout per request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from . import _warnings

TIMEOUT_SECONDS = 2.0


@dataclass
class Response:
    status_code: int
    json_body: dict[str, Any]
    retry_after: float | None = None


class Client:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            timeout=httpx.Timeout(TIMEOUT_SECONDS),
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "agentping-python/0.3.0",
            },
        )

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def post_raw(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """POST and return the raw httpx response, letting transport errors
        propagate and emitting no warnings. The guard gate uses this so it can
        own its own allow/block/unreachable translation (guard-checks-spec)
        rather than the telemetry warn-and-retry semantics of post().
        """
        return self._http.post(f"{self._base_url}{path}", json=body)

    def post(self, path: str, body: dict[str, Any]) -> Response:
        url = f"{self._base_url}{path}"
        try:
            r = self._http.post(url, json=body)
        except httpx.TimeoutException:
            _warnings.warn_once(
                "network_error",
                "request timed out; telemetry will retry in the background.",
            )
            raise
        except httpx.HTTPError as exc:
            _warnings.warn_once(
                "network_error",
                f"network error contacting AgentPing ({exc.__class__.__name__}); will retry.",
            )
            raise

        retry_after: float | None = None
        ra = r.headers.get("Retry-After")
        if ra:
            try:
                retry_after = float(ra)
            except ValueError:
                retry_after = None

        if r.status_code in (401, 403):
            _warnings.warn_once(
                "auth_error",
                "API key rejected by AgentPing; telemetry will be dropped.",
            )
        elif 500 <= r.status_code < 600:
            _warnings.warn_once(
                "server_error",
                f"AgentPing returned HTTP {r.status_code}; will retry.",
            )

        try:
            body_json = r.json() if r.content else {}
        except ValueError:
            body_json = {}

        return Response(status_code=r.status_code, json_body=body_json, retry_after=retry_after)
