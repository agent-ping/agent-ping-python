import json
import threading

import httpx

import agentping


def _capture():
    captured: list = []
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            captured.append({
                "url": str(request.url),
                "body": json.loads(request.content) if request.content else {},
            })
        return httpx.Response(202, json={})

    return captured, httpx.MockTransport(handler)


def test_heartbeat_posts_to_heartbeats_endpoint():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    rid = agentping.heartbeat("daily-summary", status="ok", cost_usd=0.084,
                              duration_ms=12300, metadata={"items": 47})
    agentping.flush(timeout=2.0)

    assert rid.startswith("run_eu_")
    hb = [c for c in captured if "/v1/heartbeats" in c["url"]]
    assert len(hb) == 1
    body = hb[0]["body"]
    assert body["id"] == rid
    assert body["agent"] == "daily-summary"
    assert body["status"] == "ok"
    assert body["cost_usd"] == 0.084
    assert body["metadata"] == {"items": 47}
    assert "started_at" in body
    assert "finished_at" in body
