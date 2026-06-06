"""Bedrock native auto-instrumentation tests."""

import json
import threading

import httpx
import pytest

pytest.importorskip("botocore")

import agentping
from agentping.instrumentation import _bedrock as bedrock_inst


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


def _collect_llm_events(captured, rid):
    flat = []
    for c in captured:
        if f"/v1/runs/{rid}/events" in c["url"]:
            flat.extend(c["body"]["events"])
    return [e for e in flat if e["type"] == "llm_call"]


def test_converse_emits_llm_call_with_usage():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from botocore.client import BaseClient
    original = BaseClient._make_api_call

    # Replace _make_api_call to mimic Bedrock Converse without hitting AWS.
    class _FakeServiceModel:
        service_name = "bedrock-runtime"

    class _FakeMeta:
        service_model = _FakeServiceModel()

    def fake_make_api_call(self, op, params):
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "usage": {"inputTokens": 420, "outputTokens": 88},
            "stopReason": "end_turn",
        }

    BaseClient._make_api_call = fake_make_api_call

    try:
        agentping.instrument_bedrock()

        # Build a fake "client" that the patched code can inspect.
        class FakeBedrockClient(BaseClient):
            def __init__(self):
                self.meta = _FakeMeta()

        client = FakeBedrockClient()

        with agentping.run("order-classifier") as r:
            rid = r.id
            client._make_api_call("Converse", {
                "modelId": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "messages": [],
            })
        agentping.flush(timeout=2.0)
    finally:
        bedrock_inst.uninstrument()
        BaseClient._make_api_call = original

    events = _collect_llm_events(captured, rid)
    assert len(events) == 1
    data = events[0]["data"]
    assert data["provider"] == "bedrock"
    assert data["model"] == "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
    assert data["input_tokens"] == 420
    assert data["output_tokens"] == 88
    assert "latency_ms" in data


def test_converse_splits_cache_read_into_cached_input_tokens():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from botocore.client import BaseClient
    original = BaseClient._make_api_call

    class _FakeServiceModel:
        service_name = "bedrock-runtime"

    class _FakeMeta:
        service_model = _FakeServiceModel()

    def fake_make_api_call(self, op, params):
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "usage": {
                "inputTokens": 1000,
                "outputTokens": 120,
                "cacheReadInputTokens": 400,
            },
        }

    BaseClient._make_api_call = fake_make_api_call

    try:
        agentping.instrument_bedrock()

        class FakeBedrockClient(BaseClient):
            def __init__(self):
                self.meta = _FakeMeta()

        client = FakeBedrockClient()
        with agentping.run("classify") as r:
            rid = r.id
            client._make_api_call("Converse", {
                "modelId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "messages": [],
            })
        agentping.flush(timeout=2.0)
    finally:
        bedrock_inst.uninstrument()
        BaseClient._make_api_call = original

    data = _collect_llm_events(captured, rid)[0]["data"]
    assert data["input_tokens"] == 600  # 1000 - 400 cached
    assert data["cached_input_tokens"] == 400
    assert data["output_tokens"] == 120


def test_invoke_model_reads_token_counts_from_response_headers():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from botocore.client import BaseClient
    original = BaseClient._make_api_call

    class _FakeServiceModel:
        service_name = "bedrock-runtime"

    class _FakeMeta:
        service_model = _FakeServiceModel()

    def fake_make_api_call(self, op, params):
        return {
            "body": b"{}",
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-amzn-bedrock-input-token-count": "256",
                    "x-amzn-bedrock-output-token-count": "44",
                }
            },
        }

    BaseClient._make_api_call = fake_make_api_call

    try:
        agentping.instrument_bedrock()

        class FakeBedrockClient(BaseClient):
            def __init__(self):
                self.meta = _FakeMeta()

        client = FakeBedrockClient()
        with agentping.run("invoke") as r:
            rid = r.id
            client._make_api_call("InvokeModel", {
                "modelId": "amazon.titan-text-lite-v1",
            })
        agentping.flush(timeout=2.0)
    finally:
        bedrock_inst.uninstrument()
        BaseClient._make_api_call = original

    data = _collect_llm_events(captured, rid)[0]["data"]
    assert data["provider"] == "bedrock"
    assert data["model"] == "amazon.titan-text-lite-v1"
    assert data["input_tokens"] == 256
    assert data["output_tokens"] == 44


def test_non_bedrock_calls_are_not_intercepted():
    """The patch must filter by service_name. S3, DynamoDB, etc. should pass straight through."""
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from botocore.client import BaseClient
    original = BaseClient._make_api_call

    class _FakeServiceModelS3:
        service_name = "s3"

    class _FakeMetaS3:
        service_model = _FakeServiceModelS3()

    call_count = {"n": 0}

    def fake_make_api_call(self, op, params):
        call_count["n"] += 1
        return {"Buckets": []}

    BaseClient._make_api_call = fake_make_api_call

    try:
        agentping.instrument_bedrock()

        class FakeS3Client(BaseClient):
            def __init__(self):
                self.meta = _FakeMetaS3()

        client = FakeS3Client()
        with agentping.run("non-bedrock") as r:
            rid = r.id
            client._make_api_call("ListBuckets", {})
        agentping.flush(timeout=2.0)
    finally:
        bedrock_inst.uninstrument()
        BaseClient._make_api_call = original

    # The S3 call ran (call_count == 1) but no AgentPing event was emitted for it.
    assert call_count["n"] == 1
    assert _collect_llm_events(captured, rid) == []


def test_converse_stream_captures_final_usage_from_metadata_event():
    captured, transport = _capture()
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport)

    from botocore.client import BaseClient
    original = BaseClient._make_api_call

    class _FakeServiceModel:
        service_name = "bedrock-runtime"

    class _FakeMeta:
        service_model = _FakeServiceModel()

    def stream_events():
        yield {"contentBlockDelta": {"delta": {"text": "Hello"}}}
        yield {"contentBlockDelta": {"delta": {"text": " world"}}}
        yield {"metadata": {"usage": {"inputTokens": 180, "outputTokens": 24}}}

    def fake_make_api_call(self, op, params):
        return {"stream": stream_events()}

    BaseClient._make_api_call = fake_make_api_call

    try:
        agentping.instrument_bedrock()

        class FakeBedrockClient(BaseClient):
            def __init__(self):
                self.meta = _FakeMeta()

        client = FakeBedrockClient()
        with agentping.run("stream-converse") as r:
            rid = r.id
            response = client._make_api_call("ConverseStream", {
                "modelId": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "messages": [],
            })
            # Drain the stream like real callers do.
            list(response["stream"])
        agentping.flush(timeout=2.0)
    finally:
        bedrock_inst.uninstrument()
        BaseClient._make_api_call = original

    data = _collect_llm_events(captured, rid)[0]["data"]
    assert data["provider"] == "bedrock"
    assert data["input_tokens"] == 180
    assert data["output_tokens"] == 24


def test_instrument_bedrock_is_idempotent():
    from botocore.client import BaseClient
    original = BaseClient._make_api_call
    try:
        agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test",
                       transport=httpx.MockTransport(lambda r: httpx.Response(202, json={})))
        agentping.instrument_bedrock()
        first = BaseClient._make_api_call
        agentping.instrument_bedrock()
        second = BaseClient._make_api_call
        assert first is second
    finally:
        bedrock_inst.uninstrument()
        BaseClient._make_api_call = original
