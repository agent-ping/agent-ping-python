import json
import threading

import httpx
import pytest

import agentping


def _capture():
    captured = []
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            body = json.loads(request.content) if request.content else {}
            captured.append({"url": str(request.url), "body": body})
        return httpx.Response(202, json={"accepted": 1})

    return captured, httpx.MockTransport(handler)


def _starts(captured):
    return [c["body"] for c in captured if c["url"].endswith("/v1/runs")]


def test_turns_collect_on_one_run():
    """Each turn rebuilds the conversation from the session id alone, as a
    stateless request handler would. They must land on one run."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    session = agentping.new_session_id()
    for n in (1, 2, 3):
        chat = agentping.conversation("support-chat", session)
        chat.event("llm_call", {"provider": "anthropic", "model": "claude-sonnet-4-6",
                                 "input_tokens": 900 * n, "output_tokens": 40})
    agentping.flush(timeout=2.0)

    starts = _starts(captured)
    assert starts, "no run starts recorded"
    assert {s["id"] for s in starts} == {f"run_eu_{session}"}
    # A differing started_at across turns is a 409 on the server.
    assert len({s["started_at"] for s in starts}) == 1


def test_start_time_comes_from_the_session_id_not_the_clock():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    session = agentping.new_session_id()
    agentping.conversation("support-chat", session)
    agentping.flush(timeout=2.0)

    started_ms = int(session[:12], 16)
    body = _starts(captured)[0]
    assert body["started_at"].startswith(
        __import__("datetime").datetime.fromtimestamp(
            started_ms / 1000.0, tz=__import__("datetime").timezone.utc
        ).isoformat()[:19]
    )


def test_opts_into_the_sweep_closing_as_success():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    agentping.conversation("support-chat", agentping.new_session_id())
    agentping.flush(timeout=2.0)

    body = _starts(captured)[0]
    assert body["idle_timeout_seconds"] == 900
    # Leaving a chat is not a failure; the server would otherwise default to timeout.
    assert body["idle_timeout_status"] == "success"


def test_ordinary_run_stays_out_of_the_sweep():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    agentping.run("nightly-report")
    agentping.flush(timeout=2.0)

    assert "idle_timeout_seconds" not in _starts(captured)[0]


def test_rejects_a_session_id_that_is_not_32_hex():
    _, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    with pytest.raises(ValueError, match="32 hex"):
        agentping.conversation("support-chat", "not-a-session")
