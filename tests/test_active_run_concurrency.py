"""Regression: the active-run must isolate concurrent asyncio tasks.

Before the ContextVar migration this used threading.local() which is task-blind,
so two concurrent agentping.run() blocks under asyncio would clobber each other's
active run and route LLM events to the wrong run."""

import asyncio
import json
import threading

import httpx
import pytest

import agentping
from agentping.instrumentation._streaming import emit_llm_call


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


def test_concurrent_async_runs_do_not_leak_events():
    """Two concurrent runs each emit one llm_call; each event must land on its own run."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    async def driver(name: str, payload: dict) -> str:
        async with agentping.run(name) as r:
            await asyncio.sleep(0)  # yield control so the other task runs
            emit_llm_call(payload)
            await asyncio.sleep(0)
            return r.id

    async def main():
        a_id, b_id = await asyncio.gather(
            driver("task-a", {"provider": "openai", "model": "gpt-4o-mini", "input_tokens": 100, "output_tokens": 10}),
            driver("task-b", {"provider": "anthropic", "model": "claude-sonnet-4-5", "input_tokens": 200, "output_tokens": 20}),
        )
        return a_id, b_id

    a_id, b_id = asyncio.run(main())
    agentping.flush(timeout=2.0)

    # Collect every llm_call event grouped by which run it landed on.
    by_run: dict[str, list[dict]] = {}
    for call in captured:
        if "/events" not in call["url"]:
            continue
        run_id = call["url"].split("/v1/runs/")[1].split("/events")[0]
        for event in call["body"]["events"]:
            if event["type"] == "llm_call":
                by_run.setdefault(run_id, []).append(event["data"])

    # Run A must have exactly the OpenAI event; Run B must have exactly the Anthropic one.
    assert a_id != b_id
    assert len(by_run.get(a_id, [])) == 1, f"task-a got {by_run.get(a_id, [])}"
    assert len(by_run.get(b_id, [])) == 1, f"task-b got {by_run.get(b_id, [])}"
    assert by_run[a_id][0]["provider"] == "openai"
    assert by_run[a_id][0]["input_tokens"] == 100
    assert by_run[b_id][0]["provider"] == "anthropic"
    assert by_run[b_id][0]["input_tokens"] == 200


def test_threaded_runs_do_not_leak_events():
    """ContextVar isolates threads too (each thread gets its own context by default)."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    results: dict[str, str] = {}
    barrier = threading.Barrier(2)

    def worker(name: str, payload: dict) -> None:
        with agentping.run(name) as r:
            barrier.wait()  # both threads enter their `with` block before emitting
            emit_llm_call(payload)
            results[name] = r.id

    t1 = threading.Thread(target=worker, args=("t1", {"provider": "openai", "input_tokens": 50}))
    t2 = threading.Thread(target=worker, args=("t2", {"provider": "anthropic", "input_tokens": 60}))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    agentping.flush(timeout=2.0)

    by_run: dict[str, list[dict]] = {}
    for call in captured:
        if "/events" not in call["url"]:
            continue
        run_id = call["url"].split("/v1/runs/")[1].split("/events")[0]
        for event in call["body"]["events"]:
            if event["type"] == "llm_call":
                by_run.setdefault(run_id, []).append(event["data"])

    assert len(by_run.get(results["t1"], [])) == 1
    assert len(by_run.get(results["t2"], [])) == 1
    assert by_run[results["t1"]][0]["provider"] == "openai"
    assert by_run[results["t2"]][0]["provider"] == "anthropic"


def test_async_context_manager_supports_aenter_aexit():
    """`async with agentping.run(...)` is supported."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    async def main():
        async with agentping.run("async-only") as r:
            return r.id

    rid = asyncio.run(main())
    agentping.flush(timeout=2.0)

    # The async exit path should write a finish record.
    finish_calls = [c for c in captured if c["url"].endswith(f"/v1/runs/{rid}/finish")]
    assert finish_calls, "async __aexit__ did not call finish()"
    assert finish_calls[0]["body"]["status"] == "success"


def test_emit_outside_a_run_drops_and_logs_once_at_debug(caplog):
    """An instrumented call outside `agentping.run(...)` has nowhere to go.
    The event is dropped, and the drop is logged once per process at debug
    so it is diagnosable without being noisy."""
    import logging

    from agentping.instrumentation import _streaming

    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                   transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))
    _streaming._reset_warnings_for_tests()

    with caplog.at_level(logging.DEBUG, logger="agentping"):
        emit_llm_call({"provider": "openai", "model": "gpt-4o-mini", "input_tokens": 1, "output_tokens": 1})
        emit_llm_call({"provider": "openai", "model": "gpt-4o-mini", "input_tokens": 1, "output_tokens": 1})

    dropped = [r for r in caplog.records if "no run is active" in r.getMessage()]
    assert len(dropped) == 1
    assert dropped[0].levelno == logging.DEBUG
