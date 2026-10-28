"""Run version labels, time to first token and opt-in prompt capture."""

import json
import threading
from types import SimpleNamespace

import httpx
import pytest

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


def _init(transport, **kwargs):
    agentping.init(api_key="apk_eu_" + "a" * 32, base_url="https://api.test", transport=transport, **kwargs)


def _run_start(captured, run_id):
    return next(c["body"] for c in captured if c["url"].endswith("/v1/runs") and c["body"].get("id") == run_id)


def _llm_calls(captured, run_id):
    flat = []
    for c in captured:
        if f"/v1/runs/{run_id}/events" in c["url"]:
            flat.extend(c["body"]["events"])
    return [e["data"] for e in flat if e["type"] == "llm_call"]


# -- version -----------------------------------------------------------------


def test_run_version_defaults_from_init_and_a_run_can_override_it():
    captured, transport = _capture()
    _init(transport, version=" 2026.10.27 ")

    default = agentping.run("triage")
    override = agentping.run("triage", version="prompt-v7")
    default.finish(status="success")
    override.finish(status="success")
    agentping.flush(timeout=2.0)

    assert default.version == "2026.10.27"
    assert _run_start(captured, default.id)["version"] == "2026.10.27"
    assert _run_start(captured, override.id)["version"] == "prompt-v7"


def test_run_version_comes_from_the_environment_and_is_omitted_when_unset(monkeypatch):
    captured, transport = _capture()
    monkeypatch.setenv("AGENTPING_VERSION", "git-abc123")
    _init(transport)
    from_env = agentping.run("triage")
    from_env.finish(status="success")

    monkeypatch.delenv("AGENTPING_VERSION")
    _init(transport)
    unset = agentping.run("triage")
    unset.finish(status="success")
    agentping.flush(timeout=2.0)

    assert _run_start(captured, from_env.id)["version"] == "git-abc123"
    assert "version" not in _run_start(captured, unset.id)


def test_conversation_carries_the_version_too():
    captured, transport = _capture()
    _init(transport)

    chat = agentping.conversation("support-chat", agentping.new_session_id(), version="v3")
    chat.finish(status="success")
    agentping.flush(timeout=2.0)

    assert _run_start(captured, chat.id)["version"] == "v3"


# -- Anthropic: content and time to first token --------------------------------

anthropic = pytest.importorskip("anthropic")
from agentping.instrumentation import _anthropic as ap_inst  # noqa: E402


def _messages_cls():
    try:
        from anthropic.resources.messages import Messages
    except Exception:
        from anthropic.resources.messages.messages import Messages
    return Messages


def _anthropic_response():
    return SimpleNamespace(
        model="claude-haiku-5-5",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=None, cache_creation_input_tokens=None),
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="Route to billing."), SimpleNamespace(type="tool_use", name="lookup")],
    )


def _anthropic_call(kwargs, response_factory, **init_kwargs):
    captured, transport = _capture()
    _init(transport, **init_kwargs)
    Messages = _messages_cls()
    original = Messages.create
    Messages.create = lambda self, *a, **kw: response_factory()
    try:
        agentping.instrument_anthropic()
        client = anthropic.Anthropic(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            result = client.messages.create(**kwargs)
            if kwargs.get("stream"):
                list(result)
        agentping.flush(timeout=2.0)
    finally:
        ap_inst.uninstrument()
        Messages.create = original
    calls = _llm_calls(captured, rid)
    assert len(calls) == 1
    return calls[0]


def test_anthropic_prompt_and_completion_are_not_sent_by_default():
    data = _anthropic_call(
        dict(model="claude-haiku-5-5", max_tokens=10, system="Be terse.", messages=[{"role": "user", "content": "hi"}]),
        _anthropic_response,
    )

    assert "input" not in data and "output" not in data and "system_instructions" not in data


def test_anthropic_capture_sends_messages_system_and_text_blocks_clipped():
    data = _anthropic_call(
        dict(
            model="claude-haiku-5-5", max_tokens=10,
            system=[{"type": "text", "text": "Be terse."}],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Refund request from Acme " * 10}, {"type": "image", "source": {}}]},
                {"role": "assistant", "content": "Checking."},
            ],
        ),
        _anthropic_response,
        capture_llm_content=True,
        llm_content_max_chars=40,
    )

    assert data["system_instructions"] == "Be terse."
    assert [m["role"] for m in data["input"]] == ["user", "assistant"]
    assert data["input"][0]["content"].endswith("...") and len(data["input"][0]["content"]) == 43
    assert "[image]" not in data["input"][0]["content"], "clipped before the image marker"
    assert data["input"][1]["content"] == "Checking."
    assert data["output"] == "Route to billing."


def test_anthropic_stream_reports_time_to_first_token_and_captured_text():
    def events():
        return iter([
            SimpleNamespace(type="message_start", message=SimpleNamespace(model="claude-haiku-5-5", usage=SimpleNamespace(input_tokens=8, output_tokens=1, cache_read_input_tokens=None, cache_creation_input_tokens=None))),
            SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="Route ")),
            SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="to billing.")),
            SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"), usage=SimpleNamespace(output_tokens=4)),
            SimpleNamespace(type="message_stop"),
        ])

    data = _anthropic_call(
        dict(model="claude-haiku-5-5", max_tokens=10, messages=[{"role": "user", "content": "hi"}], stream=True),
        events,
        capture_llm_content=True,
    )

    assert data["stream"] is True
    assert isinstance(data["ttft_ms"], int) and 0 <= data["ttft_ms"] <= data["latency_ms"]
    assert data["output"] == "Route to billing."
    assert data["input"] == [{"role": "user", "content": "hi"}]


def test_anthropic_stream_without_content_has_no_ttft():
    def events():
        return iter([
            SimpleNamespace(type="message_start", message=SimpleNamespace(model="claude-haiku-5-5", usage=None)),
            SimpleNamespace(type="message_stop"),
        ])

    data = _anthropic_call(dict(model="claude-haiku-5-5", max_tokens=10, messages=[], stream=True), events)

    assert "ttft_ms" not in data


# -- OpenAI --------------------------------------------------------------------

openai = pytest.importorskip("openai")
from agentping.instrumentation import _openai as op_inst  # noqa: E402


def _completions_cls():
    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        from openai.resources.chat.completions.completions import Completions
    return Completions


def _responses_cls():
    try:
        from openai.resources.responses import Responses
    except Exception:
        from openai.resources.responses.responses import Responses
    return Responses


def _openai_usage():
    return SimpleNamespace(
        prompt_tokens=30, completion_tokens=12,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )


def test_openai_chat_stream_reports_ttft_and_captures_deltas():
    captured, transport = _capture()
    _init(transport, capture_llm_content=True)

    def chunks():
        yield SimpleNamespace(model="gpt-5", usage=None, choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content="", tool_calls=None))])
        yield SimpleNamespace(model="gpt-5", usage=None, choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content="Route ", tool_calls=None))])
        yield SimpleNamespace(model="gpt-5", usage=None, choices=[SimpleNamespace(finish_reason="stop", delta=SimpleNamespace(content="to billing.", tool_calls=None))])
        yield SimpleNamespace(model="gpt-5", choices=[], usage=_openai_usage())

    Completions = _completions_cls()
    original = Completions.create
    Completions.create = lambda self, *a, **kw: chunks()
    try:
        agentping.instrument_openai()
        client = openai.OpenAI(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            messages = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}]
            list(client.chat.completions.create(model="gpt-5", messages=messages, stream=True))
        agentping.flush(timeout=2.0)
    finally:
        op_inst.uninstrument()
        Completions.create = original

    data = _llm_calls(captured, rid)[0]
    assert isinstance(data["ttft_ms"], int) and 0 <= data["ttft_ms"] <= data["latency_ms"]
    assert data["output"] == "Route to billing."
    assert data["input"] == [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}]


def test_openai_responses_capture_input_instructions_and_output_text():
    captured, transport = _capture()
    _init(transport, capture_llm_content=True)

    response = SimpleNamespace(
        model="gpt-5", status="completed", incomplete_details=None, output=[],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, input_tokens_details=SimpleNamespace(cached_tokens=0), output_tokens_details=SimpleNamespace(reasoning_tokens=0)),
        output_text="Route to billing.",
    )
    Responses = _responses_cls()
    original = Responses.create
    Responses.create = lambda self, *a, **kw: response
    try:
        agentping.instrument_openai()
        client = openai.OpenAI(api_key="sk-fake")
        with agentping.run("triage") as r:
            rid = r.id
            client.responses.create(model="gpt-5", instructions="Be terse.", input="Refund request from Acme")
        agentping.flush(timeout=2.0)
    finally:
        op_inst.uninstrument()
        Responses.create = original

    data = _llm_calls(captured, rid)[0]
    assert data["system_instructions"] == "Be terse."
    assert data["input"] == [{"role": "user", "content": "Refund request from Acme"}]
    assert data["output"] == "Route to billing."
    assert "ttft_ms" not in data, "not a stream"
