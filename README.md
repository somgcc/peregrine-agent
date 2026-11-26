# Peregrine

A fast, efficient, responsive agent framework. Rust. MIT OR Apache-2.0.

Peregrine is a minimal agent core with a streaming-first loop: tokens flow to
the user as they arrive, the loop never buffers a whole response before
acting, and the loop never waits on the renderer. Providers, tools, UI, and
the sandbox are all downstream of two traits — `LlmClient` and `Tool`.

It is built by studying two excellent references and taking the best of each:

- **Pi** (`earendil-works/pi`, MIT) — radical minimalism: a tiny core, four
  tools, a system prompt under 1,000 tokens, definition-first tools, JSONL
  sessions, and a loop that depends only on a stream function and a tool.
- **Orca** (`stablyai/orca`, MIT; `orca-ae/orca-agent-engine`, Apache-2.0) —
  the event log as the spine of the system, guardrails enforced *outside* the
  agent (monotonic `allow < ask < deny`, fail-closed budgets), credential-free
  execution snapshots, and a `mock` harness that validates the whole plumbing
  with no model and no cost.

See [`DESIGN.md`](DESIGN.md) for the full design.

## Quickstart

```sh
cargo build --release
./target/release/peregrine --print --provider mock "list the files"
```

The mock provider needs no key and no network — it runs a full turn with a
real tool call (`bash ls -la`) and answers from the result.

```sh
# Interactive REPL (streaming, /exit to quit)
./target/release/peregrine

# NDJSON events on stdout (machine-readable, RPC-friendly)
./target/release/peregrine --json --provider mock "list the files"

# A real model via any OpenAI-compatible endpoint (Ollama works too)
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.openai.com/v1 \
  ./target/release/peregrine --print --provider openai-compatible "fix the failing test"

# Anthropic
ANTHROPIC_API_KEY=sk-ant-... \
  ./target/release/peregrine --print --provider anthropic "review this diff"

# Resume a session
./target/release/peregrine --print --provider mock --session <id> "and now edit it"
```

## Architecture

```
peregrine (CLI: interactive / --print / --json)
  └─ peregrine-tools  (built-in tools, definition-first, Ops trait)
  └─ peregrine-llm    (provider abstraction: OpenAI-compatible, Anthropic, mock)
  └─ peregrine-core   (loop, transcript, tools trait, guardrails, sessions)
       depends on NOTHING provider- or tool-specific
```

Core never imports a provider or a tool implementation. It only knows
`LlmClient` (streams assistant events) and `Tool` (executes).

### The loop

```
loop:
  pending = drain steering queue            # typed while the agent works: enqueue, don't block
  while has_tool_calls or pending:
    pending += prepare_next_turn()          # hooks may swap context/model
    append pending (+ tool-loadout delta as system message)
    msg = stream_assistant_response()       # LlmClient::stream, events forwarded to the event channel
    if msg.stop_reason in {error, aborted}: emit turn_end, agent_end; return
    if msg has tool_calls:
      if stop_reason == length: fail ALL calls   # args may be truncated; never execute them
      else: execute_tool_calls(msg)              # prepare sequentially, execute concurrently,
                                                 # results emitted in the assistant's source order
      append tool-result messages
    finish_turn() -> end | continue
    pending = drain steering queue
  follow_ups = drain follow-up queue
  if follow_ups: pending = follow_ups; continue
  break
emit agent_end
```

Key properties, all tested:

- **Failure-as-data.** Unknown tools, validation errors, provider errors,
  and tool errors all become `is_error` results or `Error` events. The loop
  never crashes on a tool.
- **Parallel tool batches.** Sequential prepare (validation, guardrail,
  permission hook — each exactly once), concurrent execute, source-ordered
  results. Per-file mutation mutex serializes same-file edits while
  different files mutate in parallel.
- **Guardrails outside the agent.** Monotonic verdict lattice
  `allow < ask < deny`; budgets fail closed (`max_turns`, `max_tool_calls_per_session`
  default 100, turn timeout default 600 s, loop detection: identical
  tool+args ≥ 3 in a window of 10 → deny). `ask` in non-interactive mode is
  a deny. The `before_tool_call` hook is the only permission gate, and it
  exists from day one — there is no "trust the terminal user" default for
  unattended use.
- **Events over a bounded channel.** The loop never awaits the renderer;
  backpressure is by coalescing (update events drop first; drops are counted
  in the run outcome).
- **Sessions are an append-only JSONL event log** (one file per session,
  `<timestamp>_<session_id>.jsonl`, header line first). Resume is reload +
  replay; the system prompt and tool loadout live in the transcript as
  replayable state. See [`examples/session_transcript.jsonl`](examples/session_transcript.jsonl)
  for a real one.

### Tools (v0.1: Pi's four)

| Tool | Parameters | Notes |
|---|---|---|
| `read` | `{path, offset?, limit?}` | Line-based offset/limit, 1 MB cap |
| `write` | `{path, content}` | Creates parent dirs, whole-file overwrite |
| `edit` | `{path, edits: [{old_text, new_text}]}` | All edits matched against the original file; `old_text` must be unique; returns a unified diff |
| `bash` | `{command, timeout?}` | stdout+stderr merged, streamed; timeout kills the process group; output over 100 KB spills to a temp file |

Tools are definition-first: one struct derives `serde` + `schemars`, so the
JSON Schema the model sees and the types the tool parses can never drift.
All paths resolve under the working directory (symlink-aware via
canonicalization). See [`examples/custom_tool.rs`](examples/custom_tool.rs)
for a from-scratch custom tool (`word_count`).

### Providers

One `LlmClient` trait; providers are data (base URL + auth scheme + wire
format), not code. Auth resolution: explicit override → stored credential
(`~/.config/peregrine/auth.json`, mode `0600`) → env (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`). Usage comes from the provider
response; there are no char-count token estimates.

## Extension roadmap

- **Phase 2: lightweight Docker/sandbox extension for tool execution.**
  The seam already exists and is the *only* seam: `Ops`. Tools depend on
  `Ops`, never on the filesystem or spawning directly. `SandboxOps` will
  implement `Ops` over Docker / `bwrap` / gVisor-style runtimes, selected by
  config, with no core or tool changes. Credentials never enter the sandbox
  (opaque short-lived tokens only, resolved server-side at egress).
  Guardrails stay outside the agent.
- v0.2: compaction with a real local tokenizer, `ratatui`/`crossterm` TUI
  (16 ms render coalescing, line-diff output), interactive approval prompts
  for `ask` verdicts, MCP as an extension (not core).
- Later: sub-agents, plan mode, multi-agent fleet (durable mail + worktrees,
  human-reviewed merge — no auto-merge), web dashboard.

Non-goals for v0.1 (deliberate): sub-agents, plan mode, MCP, fleet,
auto-merge, web dashboard.

## Perf targets (gated, not hoped for)

- Binary startup < 50 ms; first-token passthrough (no buffering before the
  first stream event); zero CPU when idle (no polling).
- CI bench: streaming throughput (events/s), turn latency with the mock
  provider, memory slope per session, stalls > 100 ms = FAIL.

## Development

```sh
cargo test --workspace            # 49 tests, all green
cargo clippy --workspace -- -D warnings
cargo run -p peregrine --example custom_tool
```

## License

MIT OR Apache-2.0. See `LICENSE-MIT` and `LICENSE-APACHE`.
