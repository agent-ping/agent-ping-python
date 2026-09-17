import threading
import time

import httpx

import agentping
from agentping import _ids


def _make_capturing_transport(events: list):
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            body = request.content.decode() if request.content else ""
            events.append({"url": str(request.url), "method": request.method, "body": body})
        return httpx.Response(202, json={"accepted": 1})

    return httpx.MockTransport(handler)


def test_run_id_available_immediately():
    captured: list = []
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))

    r = agentping.run("triage")
    assert r.id.startswith("run_eu_")
    assert _ids.is_valid_id(r.id)


def test_run_event_and_finish_enqueue_requests():
    captured: list = []
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))

    r = agentping.run("triage", customer_id="acme", feature="route")
    r.event("log", {"msg": "hi"})
    r.event("llm_call", {"provider": "anthropic", "model": "claude", "input_tokens": 10,
                         "output_tokens": 5, "latency_ms": 200})
    r.finish(status="success", output={"x": 1})

    agentping.flush(timeout=2.0)

    paths = [evt["url"] for evt in captured]
    assert any("/v1/runs" == p.split("https://api.test")[1] for p in paths)
    assert any(f"/v1/runs/{r.id}/events" in p for p in paths)
    assert any(f"/v1/runs/{r.id}/finish" in p for p in paths)


def test_context_manager_finishes_success():
    captured: list = []
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))

    with agentping.run("triage") as r:
        rid = r.id
        r.event("log", {"msg": "in"})

    agentping.flush(timeout=2.0)
    finish_calls = [c for c in captured if f"/v1/runs/{rid}/finish" in c["url"]]
    assert len(finish_calls) == 1
    import json
    assert json.loads(finish_calls[0]["body"])["status"] == "success"


def test_context_manager_finishes_failed_on_exception():
    captured: list = []
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))

    try:
        with agentping.run("triage") as r:
            rid = r.id
            raise ValueError("oops")
    except ValueError:
        pass

    agentping.flush(timeout=2.0)
    finish_calls = [c for c in captured if f"/v1/runs/{rid}/finish" in c["url"]]
    assert len(finish_calls) == 1
    import json
    assert json.loads(finish_calls[0]["body"])["status"] == "failed"


def test_context_manager_records_error_event_before_finishing_failed():
    """The exception that ends the block lands on the timeline as an
    `error` event (message plus exception class) so the dashboard can
    show why the run failed, not just that it did."""
    import json

    captured: list = []
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))

    try:
        with agentping.run("triage") as r:
            rid = r.id
            raise ValueError("upstream returned 503")
    except ValueError:
        pass

    agentping.flush(timeout=2.0)
    events = []
    for c in captured:
        if f"/v1/runs/{rid}/events" in c["url"]:
            events.extend(json.loads(c["body"])["events"])
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["data"]["message"] == "upstream returned 503"
    assert errors[0]["data"]["exception"] == "ValueError"

    finish = [c for c in captured if f"/v1/runs/{rid}/finish" in c["url"]]
    assert json.loads(finish[0]["body"])["status"] == "failed"


def test_status_returns_dict():
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport([]))
    s = agentping.status()
    assert set(s.keys()) == {"queue_size", "dropped_count", "last_flush_ts", "last_error"}


def test_run_uses_us_region_from_key():
    captured: list = []
    agentping.init(api_key="apk_us_" + "b" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))
    r = agentping.run("triage")
    assert r.id.startswith("run_us_")


def test_finish_is_idempotent():
    captured: list = []
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))
    r = agentping.run("triage")
    r.finish(status="success")
    r.finish(status="failed")
    agentping.flush(timeout=2.0)
    finish_calls = [c for c in captured if f"/v1/runs/{r.id}/finish" in c["url"]]
    assert len(finish_calls) == 1


def test_run_goal_included_in_run_start():
    captured: list = []
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))

    r = agentping.run("digest", goal="Send the daily digest to #support")
    assert r.goal == "Send the daily digest to #support"
    r.finish(status="success")
    agentping.flush(timeout=2.0)

    import json as _json
    start = next(evt for evt in captured if evt["url"].endswith("/v1/runs"))
    body = _json.loads(start["body"])
    assert body["goal"] == "Send the daily digest to #support"


def test_run_without_goal_omits_the_field():
    captured: list = []
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))

    r = agentping.run("digest")
    r.finish(status="success")
    agentping.flush(timeout=2.0)

    import json as _json
    start = next(evt for evt in captured if evt["url"].endswith("/v1/runs"))
    body = _json.loads(start["body"])
    assert "goal" not in body


def test_active_run_exposed_publicly():
    captured: list = []
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=_make_capturing_transport(captured))

    assert agentping.active_run() is None
    with agentping.run("triage") as r:
        assert agentping.active_run() is r
    assert agentping.active_run() is None
