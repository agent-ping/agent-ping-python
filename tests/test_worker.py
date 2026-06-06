import threading
import time

import httpx

from agentping._client import Client
from agentping._queue import BoundedQueue
from agentping._worker import Worker


def _client_with_handler(handler):
    transport = httpx.MockTransport(handler)
    return Client(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)


def test_worker_batches_and_sends():
    received = []
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            received.append(str(request.url))
        return httpx.Response(202, json={"accepted": 1})

    q = BoundedQueue(maxsize=100)
    client = _client_with_handler(handler)
    worker = Worker(q, client)

    for i in range(5):
        q.put({"path": "/v1/runs", "body": {"i": i}, "attempts": 0})

    worker.flush_now(time.monotonic() + 2.0)
    assert len(received) == 5
    assert len(q) == 0
    client.close()


def test_worker_retries_on_500_and_gives_up_after_5():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(500, json={})

    q = BoundedQueue(maxsize=100)
    client = _client_with_handler(handler)
    worker = Worker(q, client)

    q.put({"path": "/v1/runs", "body": {"i": 1}, "attempts": 0})

    for _ in range(10):
        worker.flush_now(time.monotonic() + 0.5)

    assert attempts["count"] == 5
    assert q.dropped == 1
    assert len(q) == 0
    client.close()


def test_worker_does_not_retry_4xx_client_errors():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(409, json={})

    q = BoundedQueue(maxsize=100)
    client = _client_with_handler(handler)
    worker = Worker(q, client)

    q.put({"path": "/v1/runs", "body": {}, "attempts": 0})
    worker.flush_now(time.monotonic() + 1.0)
    assert attempts["count"] == 1
    assert len(q) == 0
    client.close()


def test_worker_handles_network_error_and_requeues():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ConnectError("boom")

    q = BoundedQueue(maxsize=100)
    client = _client_with_handler(handler)
    worker = Worker(q, client)

    q.put({"path": "/v1/runs", "body": {}, "attempts": 0})
    for _ in range(10):
        worker.flush_now(time.monotonic() + 0.5)
    assert attempts["count"] == 5
    assert q.dropped == 1
    client.close()
