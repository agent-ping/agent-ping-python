# agentping-sdk

Production observability for AI agents. Cost attribution, live
monitoring, quality scoring. [agentping.io](https://agentping.io) ·
[docs](https://agentping.io/docs/sdks/python).

The PyPI distribution is `agentping-sdk`. The Python import name is
`agentping`.

## Install

```bash
pip install agentping-sdk
```

## Quickstart

```python
import agentping

agentping.init()  # reads AGENTPING_API_KEY

with agentping.run("support-triage", customer_id="acme-corp") as r:
    r.event("log", {"message": "starting"})
    r.event("llm_call", {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "input_tokens": 100,
        "output_tokens": 50,
        "latency_ms": 1234,
    })
```

`run.id` is generated locally before any network call; safe to log or
return immediately. The context manager exits with `status="success"` on
clean exit and `status="failed"` if an exception escapes.

## Heartbeat shortcut

For cron-shaped jobs that run to completion as a single unit:

```python
agentping.heartbeat("daily-summary", status="ok", cost_usd=0.084,
                    duration_ms=12300, metadata={"items": 47})
```

## Auto-instrumentation

One-line patches for each supported provider and framework. Calls made
inside `agentping.run(...)` emit `llm_call` events automatically. All
patches are idempotent.

**LLM providers** (install the provider's package alongside the matching extra):

```python
agentping.instrument_anthropic()   # pip install "agentping-sdk[anthropic]"
agentping.instrument_openai()      # pip install "agentping-sdk[openai]"
agentping.instrument_gemini()      # pip install "agentping-sdk[gemini]"
agentping.instrument_mistral()     # pip install "agentping-sdk[mistral]"
agentping.instrument_cohere()      # pip install "agentping-sdk[cohere]"
agentping.instrument_bedrock()     # pip install "agentping-sdk[bedrock]"
```

Streaming, embeddings, and prompt-cache attribution are captured.
Bedrock covers Converse, ConverseStream, InvokeModel, and
InvokeModelWithResponseStream across every model family.

**Agent frameworks:**

```python
# LangChain (callback handler)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(callbacks=[agentping.AgentPingCallbackHandler()])

# LlamaIndex (callback manager)
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
Settings.callback_manager = CallbackManager([agentping.AgentPingLlamaIndexHandler()])

# Pydantic AI
agentping.instrument_pydantic_ai()

# CrewAI
agentping.instrument_crewai()

# OpenAI Agents SDK
from agents import Runner
result = await Runner.run(my_agent, "...", hooks=agentping.AgentPingHooks())
```

Per-integration docs:
[Anthropic](https://agentping.io/docs/providers/anthropic) ·
[OpenAI](https://agentping.io/docs/providers/openai) ·
[Gemini](https://agentping.io/docs/providers/gemini) ·
[Mistral](https://agentping.io/docs/providers/mistral) ·
[Cohere](https://agentping.io/docs/providers/cohere) ·
[Bedrock](https://agentping.io/docs/providers/bedrock) ·
[LangChain](https://agentping.io/docs/frameworks/langchain) ·
[LlamaIndex](https://agentping.io/docs/frameworks/llamaindex) ·
[Pydantic AI](https://agentping.io/docs/frameworks/pydantic-ai) ·
[CrewAI](https://agentping.io/docs/frameworks/crewai) ·
[OpenAI Agents](https://agentping.io/docs/frameworks/openai-agents).

## Contract guarantees

- Run IDs are generated locally before any network call: `run.id` is
  available the instant `agentping.run(...)` returns.
- Every HTTP request has a 2-second hard timeout. The SDK never blocks
  user code.
- Telemetry queues are bounded (default 1000); when full, the oldest
  event is dropped and counted in `dropped_count`.
- One stderr warning per error class for the life of the process.
- `atexit` flush has a 5-second deadline, then exits regardless.

## Inspect

```python
agentping.status()
# {"queue_size": 0, "dropped_count": 0, "last_flush_ts": 1747000000.0, "last_error": None}

agentping.flush(timeout=5.0)
```

## Environment

| Var                    | Default                                                                |
| ---------------------- | ---------------------------------------------------------------------- |
| `AGENTPING_API_KEY`    | (required)                                                             |
| `AGENTPING_BASE_URL`   | Derived from the key's region: `https://{region}.ingest.agentping.io`. |
| `AGENTPING_PARENT_RUN` | Optional. Parent run id for nested agents.                             |

The default base URL is region-aware. `apk_eu_*` keys route to
`https://eu.ingest.agentping.io`; `apk_us_*` keys route to
`https://us.ingest.agentping.io`. Override `AGENTPING_BASE_URL` only
when self-hosting or testing.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## License

MIT.
