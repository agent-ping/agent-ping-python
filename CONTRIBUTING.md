# Contributing

Thanks for considering a contribution to the AgentPing Python SDK.

## Reporting a bug

Open a GitHub issue. Useful issues include:

- The version of `agentping-sdk` you're running (`pip show agentping-sdk`).
- Your Python version and OS.
- A minimal reproduction. Snip the surrounding application code; we
  only need the AgentPing calls and the provider client.
- What you expected to happen, what actually happened.
- Anything visible from `agentping.status()` if the bug is about
  telemetry not landing.

Security issues should not go on the public tracker. Email
`security@agentping.io` instead.

## Proposing a feature

Open an issue before opening a PR. Most feature work touches the wire
contract with the AgentPing ingest server, so we want to design the
interface together before you spend time implementing it.

If you're adding auto-instrumentation for a provider we don't cover
yet, see [Auto-instrumentation guidelines](#auto-instrumentation-guidelines)
below.

## Local development

```bash
git clone https://github.com/agent-ping/agent-ping-python
cd agentping-python
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Python 3.10 or later. The SDK has zero runtime dependencies beyond the
standard library; the provider extras (`anthropic`, `openai`, ...) pull
in the matching client.

## Tests

```bash
.venv/bin/pytest                    # full suite
.venv/bin/pytest tests/test_run.py  # one file
.venv/bin/pytest -k heartbeat       # filter by name
```

We expect every PR to add or update tests. The HTTP layer mocks the
ingest server, so tests do not require credentials or a live endpoint.

## Style

- Type hints on every public function. Run `mypy` locally if you have
  it; CI runs it on PRs.
- Format with `ruff format`. Lint with `ruff check`. Both run in CI.
- No em-dashes in source, docstrings, comments, or commit messages. Use
  commas, semicolons, or sentence breaks. This is a project-wide rule.

## Auto-instrumentation guidelines

When adding an `instrument_*` function for a new provider:

1. The patch must be idempotent. Calling it twice should be a no-op.
2. Capture token counts from the actual response, not from request
   parameters. Tokens reported pre-call are estimates and produce
   wrong spend numbers.
3. Streaming wrappers must preserve the original iterator's behaviour:
   each chunk flows to the caller without buffering, and the
   `llm_call` event is emitted once when the stream closes with the
   accumulated counts.
4. Prompt-cache token attribution: split cached and uncached input
   tokens into distinct fields if the provider exposes them. The
   server rate card prices the two differently.
5. Embedding calls and chat calls go to the same `llm_call` event
   type. The `model` field distinguishes them downstream.

## Commit messages and PRs

- Subject in imperative mood (`add Bedrock InvokeModel coverage`, not
  `added Bedrock InvokeModel coverage`).
- One topic per PR. Bug fixes, refactors, and new instrumentation
  should not be bundled.
- If the PR changes the wire contract with ingest, link the matching
  issue in the AgentPing platform tracker.

## Releasing

Maintainers only. Release flow lives in
`agent-ping/LAUNCH.md` in the AgentPing platform monorepo.
