# Changelog

All notable changes to the AgentPing Python SDK are documented here. The
format is based on [Keep a Changelog](https://keepachangelog.com/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

- Distribution name on PyPI is `agentping-sdk`. The Python import name
  stays `agentping`. The `agentping` name on PyPI is held by an
  unrelated project.

## Notes on stability

The 0.x line is pre-1.0. Public API may change before 1.0.0. We do not
break the wire format between SDK and ingest without a version bump and
a migration note here.

[Unreleased]: https://github.com/agent-ping/agent-ping-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/agent-ping/agent-ping-python/releases/tag/v0.1.0
