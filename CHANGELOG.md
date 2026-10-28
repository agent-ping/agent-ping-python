# Changelog

All notable changes to the AgentPing Python SDK are documented here. The
format is based on [Keep a Changelog](https://keepachangelog.com/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `init(version=...)` and `run(..., version=...)` label every run with the build
  or prompt version that produced it (`AGENTPING_VERSION` also works). The
  dashboard's Spend and Verify charts mark where each new version was first
  seen. `conversation()` takes it too.
- Streamed provider calls (Anthropic, OpenAI, Gemini, Mistral, Cohere and
  Bedrock ConverseStream) report `ttft_ms`, the time from the request to the
  first content chunk, charted on Pulse as time to first token. Anthropic
  `messages.stream()` context managers do not, because the SDK iterates the
  stream itself.
- `init(capture_llm_content=True, llm_content_max_chars=4000)` sends the prompt
  and completion on every auto-instrumented provider `llm_call` (Anthropic,
  OpenAI, Gemini, Mistral, Cohere and Bedrock Converse): `input` as a list of
  `{role, content}` messages, `output` as text, and `system_instructions`. Off
  by default; each piece is clipped at the cap. Non-text parts appear as
  bracketed markers such as `[image]` or `[tool_use lookup]`, Gemini's `model`
  role is reported as `assistant`, and Bedrock InvokeModel bodies are not
  captured. The run page renders them as role-labelled messages.
- The framework integrations honour `capture_llm_content` too. LangChain's
  `AgentPingCallbackHandler` sends the chat messages (`human`/`ai`/`tool` mapped
  to `user`/`assistant`/`tool`, system messages as `system_instructions`) and the
  generation text, and on streamed calls `stream: true` plus `ttft_ms` from the
  first `on_llm_new_token` that carried text or a tool-call chunk. LlamaIndex
  sends the chat messages or formatted prompt and the response text; OpenAI
  Agents sends the input items, the system prompt and the `output_text` parts;
  Pydantic AI sends the run's message history before the final answer and
  `result.output`; CrewAI sends the crew's final answer as `output` (no prompt
  is available at kickoff level). None of these but LangChain has a per-token
  hook, so they report no `ttft_ms`.
- Events carry `turn`, and `conversation()` models a chat session as one run
  (from the previous unreleased commits).

## [0.2.0] - 2026-09-17

Timeline depth. Every instrumentation now reports what the run's
dashboard timeline needs to explain a run, not just price it.

### Fixed

- **`input_tokens` was net of the cached prefix for every provider that
  reports prompt caching** (Anthropic, OpenAI chat and Responses, Gemini,
  Bedrock Converse, Pydantic AI, CrewAI, LangChain). Ingest subtracts
  `cached_input_tokens` itself, so the SDK was double-discounting cache
  hits and under-reporting cost on cached calls. `input_tokens` is now
  the gross prompt count on every path; ingest does the split.

### Added

- `llm_call` events carry `finish_reason` and `tool_calls` (the count of
  tool-use blocks in the response) for every provider that exposes them,
  plus `reasoning_tokens` when the model reports them (OpenAI, Gemini,
  LangChain `usage_metadata`).
- **Errored LLM calls are recorded.** When a patched provider call, a
  Pydantic AI run or a CrewAI kickoff raises, the SDK emits an
  `llm_call` with `status: "error"`, `error`, `exception` (the exception
  class name) and `latency_ms` before re-raising. Timed-out and
  rate-limited calls now show up on the timeline instead of vanishing.
- **Tool calls on the timeline.** `AgentPingHooks` (OpenAI Agents) and
  `AgentPingCallbackHandler` (LangChain) emit `tool_call` events with
  `tool`, `status`, `tool_invocation_id`, `latency_ms`, and the
  stringified `input` and `output` (capped at 4000 characters). Pass
  `capture_tool_payloads=False` to keep arguments and results out of
  the payload. Tool failures land as `status: "error"` with `error` and
  `exception`.
- OpenAI Agents: handoffs are `step` events with
  `kind: "handoff"`, `from` and `to`; agent start and end are
  `step` events with `kind: "agent"`, `agent` and `latency_ms`.
  `on_llm_end` reads `cached_input_tokens` and `reasoning_tokens` from
  the Responses usage details and infers the provider from
  `litellm/<provider>/<model>` model names.
- OpenAI Responses API (`client.responses.create`) is instrumented
  alongside chat completions, including streaming.
- Anthropic `client.messages.stream(...)` context manager is
  instrumented; the `llm_call` is emitted when the stream closes with
  usage from the final message snapshot.
- Streamed calls (every provider) emit one `llm_call` when the stream is
  fully consumed, with `stream: true`.
- An unhandled exception inside `with agentping.run(...)` now lands as
  an `error` event (`message`, `exception`) on the timeline before the
  run finishes `failed`. LlamaIndex `EXCEPTION` callbacks and LangChain
  `on_chain_error` include the `exception` class name too.

### Changed

- `AgentPingHooks.on_tool_start` no longer emits; the `tool_call` is
  emitted on `on_tool_end` with the measured latency and result.
- The PyPI distribution is `agentping-io`. The 0.1.0 notes named it
  `agentping-sdk`, but that name on PyPI belongs to an unrelated
  project and 0.1.0 was never uploaded there. Install with
  `pip install agentping-io`; the import name stays `agentping`.

### Wire format

Field names match the TypeScript and Laravel SDKs at 0.2.0. Ingest
accepts events from 0.1.x unchanged, but cost on cached calls is only
correct from 0.2.0.

## [0.1.0] - 2026-05-17

Initial public release.

### Added

- Core run lifecycle. `agentping.init()`, `agentping.run(name, ...)`
  context manager, `r.event(type, payload)`, automatic
  `status="success"` / `status="failed"` on context exit.
- Client-generated UUIDv7 run IDs. `run.id` is available the instant
  `agentping.run(...)` returns, before any network call.
- `agentping.heartbeat(agent, status=..., cost_usd=..., duration_ms=...,
  metadata={...})` for cron-shaped jobs that run as a single unit.
- Bounded background queue (default 1000), drop-oldest on overflow,
  exposed via `agentping.status()`.
- `atexit` flush with a 5-second deadline. `agentping.flush(timeout=...)`
  for explicit drains.
- Region-aware default base URL. `apk_eu_*` keys route to
  `https://eu.ingest.agentping.io`; `apk_us_*` keys route to
  `https://us.ingest.agentping.io`. Override via `AGENTPING_BASE_URL`
  or the `base_url` init option.

### Auto-instrumentation

- LLM providers: `instrument_anthropic()`, `instrument_openai()`,
  `instrument_gemini()`, `instrument_mistral()`, `instrument_cohere()`,
  `instrument_bedrock()`. Streaming, embeddings, and prompt-cache
  attribution are captured. Bedrock covers Converse, ConverseStream,
  InvokeModel, and InvokeModelWithResponseStream across every model
  family.
- Frameworks: `AgentPingCallbackHandler` (LangChain),
  `AgentPingLlamaIndexHandler`, `instrument_pydantic_ai()`,
  `instrument_crewai()`, `AgentPingHooks` (OpenAI Agents SDK).

### Distribution

- Distribution name on PyPI was planned as `agentping-sdk` (see 0.2.0:
  it is `agentping-io`). The Python import name stays `agentping`. The
  `agentping` name on PyPI is held by an unrelated project.

## Notes on stability

The 0.x line is pre-1.0. Public API may change before 1.0.0. We do not
break the wire format between SDK and ingest without a version bump and
a migration note here.

[Unreleased]: https://github.com/agent-ping/agent-ping-python/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/agent-ping/agent-ping-python/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/agent-ping/agent-ping-python/releases/tag/v0.1.0
