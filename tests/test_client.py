import httpx

from agentping._client import Client


def test_client_posts_with_auth_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = request.content
        return httpx.Response(202, json={"accepted": 1})

    transport = httpx.MockTransport(handler)
    client = Client(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    resp = client.post("/v1/runs", {"id": "run_eu_abc"})

    assert resp.status_code == 202
    assert resp.json_body == {"accepted": 1}
    assert captured["url"] == "https://api.test/v1/runs"
    assert captured["auth"] == "Bearer apk_eu_" + "a" * 32
    client.close()


def test_client_returns_retry_after():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "7"}, json={})

    transport = httpx.MockTransport(handler)
    client = Client(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)
    resp = client.post("/v1/runs", {})
    assert resp.status_code == 429
    assert resp.retry_after == 7.0
    client.close()


def test_client_handles_500_warning_once(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.MockTransport(handler)
    client = Client(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)
    client.post("/v1/runs", {})
    client.post("/v1/runs", {})
    err = capsys.readouterr().err
    assert err.count("HTTP 500") == 1
    client.close()
