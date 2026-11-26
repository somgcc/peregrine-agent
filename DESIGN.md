# Peregrine — design (v0.1)

Fast, efficient, responsive agent framework. Rust. MIT OR Apache-2.0.
Repo: `peregrine-agent`. Crate `peregrine`, binary `peregrine`.
References: Pi (`earendil-works/pi` v1.6.1, MIT) and Orca
(`stablyai/orca` v1.4.214, MIT; `orca-ae/orca-agent-engine` v0.5.0, Apache-2.0).
Study notes: Pi <https://github.com/earendil-works/pi>, Orca
<https://github.com/stablyai/orca>, orca-agent-engine
<https://github.com/orca-ae/orca-agent-engine>.

## Goals

- Minimal core, streaming-first: tokens flow to the user as they arrive; the loop
  never buffers a whole response before acting.
- Fast: single static binary, lazy provider loading, zero CPU when idle (no polling),
  bounded everything (truncation, 1 MB caps, 16 ms render floor).
- Responsive: the loop never awaits the renderer; events go over a bounded channel
  and the UI drains at its own pace.
- Composable: the loop depends only on two traits (`LlmClient`, `Tool`).
  Providers, tools, UI, sandbox are all downstream.

## Non-goals (v0.1)

Sub-agents, plan mode, MCP, multi-agent fleet, auto-merge, web dashboard.
Phase 2 (planned): lightweight Docker/sandbox extension for tool execution.

## Layering (borrowed from Pi)

```
peregrine (CLI: interactive / --print / --json)
  └─ peregrine-tools  (built-in tools, definition-first, Ops trait)
  └─ peregrine-llm    (provider abstraction: OpenAI-compatible, Anthropic, mock)
  └─ peregrine-core   (loop, transcript, tools trait, guardrails, sessions)
       depends on NOTHING provider- or tool-specific
```

Core never imports a provider or a tool implementation. It only knows
`LlmClient` (streams assistant events) and `Tool` (executes).

## Core types

```rust
// Transcript: system prompt + tool declarations live IN the transcript as
// replayable state (sections + tools_added/removed), not as call parameters.
// Resume, fork, compaction, model switching all fall out of one mechanism.
enum Message {
    System { sections: Vec<SystemSection> },   // named, diffed per turn
    User { content: Vec<UserContent> },
    Assistant { content: Vec<AssistantContent> },
    ToolResult { call_id: String, result: ToolResult },
}
enum AssistantContent { Text(String), Thinking(String), ToolCall { id: String, name: String, args: serde_json::Value } }

// Failure-as-data: errors are values in the stream, never panics. The loop is total.
enum LlmEvent { TextDelta(String), ThinkingDelta(String), ToolCallDelta { id: String, name: Option<String>, args_delta: String }, Done { usage: Usage, stop_reason: StopReason }, Error(String) }
enum StopReason { Stop, Length, ToolUse, Error, Aborted }

trait LlmClient: Send + Sync {
    fn stream(&self, req: LlmRequest) -> impl futures::Stream<Item = LlmEvent> + Send;
    // one call the loop uses; providers normalize wire differences here
}

trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> schemars::Schema;   // serde + schemars: one definition, types + JSON Schema
    fn execute(&self, ctx: &ToolCtx, args: serde_json::Value) -> impl Future<Output = ToolResult> + Send;
}
struct ToolResult { content: String, is_error: bool }

// Ops: the sandbox seam. Tools depend on Ops, never on fs/spawn directly.
// LocalOps runs on the host (v0.1 default). SandboxOps (Docker/bwrap) plugs in here in phase 2.
trait Ops: Send + Sync {
    async fn read_file(&self, path: &Path, offset: Option<usize>, limit: Option<usize>) -> io::Result<String>;
    async fn write_file(&self, path: &Path, content: &str) -> io::Result<()>;
    async fn edit_file(&self, path: &Path, edits: &[(String, String)]) -> io::Result<String>; // unified diff out
    async fn run_command(&self, cmd: &str, timeout: Option<Duration>) -> impl Stream<Item = CmdChunk> + Send;
}
```

## Agent loop (adapted from Pi `runLoop`)

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

- `Agent` owns state (`model`, `tools`, `messages`, `is_streaming`) + two queues
  (steer: injected after the current turn; follow-up: run only when the agent
  would stop). One `CancellationToken` per run. `prompt()` errors if a run is
  active; use `steer()` / `follow_up()` instead.
- Hooks (all total, must not fail the loop): `before_tool_call` (may block with
  a reason — the ONLY permission gate), `after_tool_call` (result override),
  `finish_turn`, `prepare_request`, `prepare_next_turn`, `transform_context`.
- Unknown tool / validation error / thrown error all become `is_error: true`
  results. The loop never crashes on a tool.
- Nested tool calls re-enter the same pipeline (permission hooks apply) and are
  recorded only as a bounded list on the parent result, never in the transcript.
- Per-file mutation queue: a mutex keyed by canonical path serializes mutations
  to the same file while different files mutate in parallel.
- Events (`agent_start/end`, `turn_start/end`, `message_start/update/end`,
  `tool_execution_start/update/end`) are sent over a bounded `tokio::mpsc`
  channel. The loop never awaits the renderer; backpressure by coalescing.

## Providers (borrowed from Pi, without the sprawl)

- One `Api` trait; providers are data (base URL + auth scheme + wire format),
  not code. v0.1 ships 3: `openai-compatible` (OpenAI, Ollama, any compatible
  endpoint), `anthropic` (Messages API), `mock` (deterministic, LLM-free —
  build the mock FIRST; it makes every test fast and free).
- Auth resolution: explicit override → stored credential → env. Stored
  credentials are file-backed (`~/.config/peregrine/auth.json`, mode `0600`).
  Credentials never enter tool execution; the sandbox phase keeps that invariant.
- Usage comes from the provider response; context accounting tracks it strictly.
  No char-count token estimates (Pi's most common failure class). Compaction is
  v0.2 with a real local tokenizer.

## Tools (v0.1: Pi's 4)

`read {path, offset?, limit?}` (streams in chunks, line/byte caps),
`write {path, content}` (creates parent dirs, whole-file overwrite),
`edit {path, edits: [{old_text, new_text}]}` (all edits matched against the
original file; `old_text` must be unique; returns a unified diff),
`bash {command, timeout?}` (stdout+stderr merged, streamed, truncated with full
output spilled to a temp file; timeout/abort kills the process tree).
Tools are definition-first: the registry holds definitions (schema + prompt
snippet + renderers); the core adapts them.
Path confinement: `read`, `write`, and `edit` resolve all paths under the
working directory (symlink-aware via canonicalization). `bash` is not
path-confined in v0.1 — it runs with the user's shell privileges and is gated
behind the approval hook; true confinement comes with the Phase 2 sandbox
(`SandboxOps`).

## Guardrails (borrowed from Orca engine, enforced outside the agent)

Pure evaluator, monotonic verdict lattice `allow < ask < deny` — a rule can only
tighten, never loosen. Every phase must have a concrete enforcement hook.
v0.1 budgets (fail closed): `max_turns`, `max_tool_calls_per_session`
(default 100), turn timeout (default 600 s), loop detection (identical
tool+args cycle ≥ 3 in a window of 10 → deny), cost/token budget hooks
(unpriced model is never priced at $0 — unknown cost asks, never passes).
`ask` in non-interactive mode is a deny. Permission policy hook
(`before_tool_call` → allow/deny/ask) exists from day one — no
"trust the terminal user" default for unattended use.

## Sessions (borrowed from Pi + Orca)

Append-only JSONL event log, one file per session
(`<timestamp>_<session_id>.jsonl`), header line `{type: "session", version, id, cwd}`.
The event log is the spine: session state, UI projections, and recovery all
derive from replay. Resume = reload + replay. Fork/branch = new file sharing a
prefix. SSE-style streaming with cursor resume (`Last-Event-ID` semantics) for
the future server mode.

## CLI modes (borrowed from Pi)

- Interactive: REPL with streaming tokens, 16 ms render coalescing, input
  preempts rendering, line-diff output (ratatui/crossterm in v0.2; simple
  line-based REPL in v0.1).
- `--print`: run once, print final text, exit.
- `--json`: NDJSON events on stdout (machine-readable, RPC-friendly).
- `--provider`, `--model`, `--session` (resume), `--cwd`.

## Phase 2: sandbox extension

Lightweight Docker/sandbox extension for tool execution. Design constraint on
v0.1: `Ops` is the only seam — `SandboxOps` implements `Ops` over
Docker / `bwrap` / gVisor-style runtimes, selected by config, no core changes.
Credentials never enter the sandbox (opaque short-lived tokens only, resolved
server-side at egress). Guardrails stay outside the agent.

## Perf targets (gated, not hoped for)

- Binary startup < 50 ms; first token passthrough (no buffering before the
  first stream event); zero CPU when idle.
- CI bench: streaming throughput (events/s), turn latency with the mock
  provider, memory slope per session, stalls > 100 ms = FAIL.

## v0.1 deviations (as built, 2026-11-26)

- `Tool::execute` / `LlmClient::stream` / `Ops` methods return
  `Pin<Box<dyn Future/Stream>>`, not `impl Future/Stream` in trait position:
  RPITIT traits are not dyn-compatible and the loop needs `Box<dyn Tool>` /
  `Arc<dyn LlmClient>`. Semantics identical.
- Session log lines are adjacently tagged (`{"type":"session","data":{...}}`)
  rather than flat: `AgentEvent` is itself internally tagged.
- Nested tool calls not implemented (no v0.1 tool needs them; `ToolCtx` can gain
  them without core changes). JSON Schema validation is light only (object +
  required fields); full validation needs a validator crate.
- `ask` verdict denies in all modes, including interactive; approval prompting
  needs a UI prompt path (the `before_tool_call` hook exists for it).
- No cost budgets in v0.1 (no pricing source; nothing is priced at $0 because
  there is no cost accounting at all). No `--fork` CLI flag yet (format
  supports it). Stored-credential write path not implemented (read only).
- No 16 ms render coalescing in the v0.1 REPL (deltas print as they arrive;
  coalescing is v0.2 ratatui work). No CI bench harness yet.
- Known v0.1 gaps (fixed in v0.1.1): steering/interrupt is implemented in
  the core (`steer()` / `cancel()`) but not wired in the CLI, so Ctrl-C and
  mid-run input do nothing in the REPL; `bash` buffers its full output and
  returns it at the end instead of streaming chunks live; the turn timeout
  bounds only the model stream, not tool execution.
- Anthropic provider verified by unit tests only; OpenAI-compatible verified
  against a local fake SSE endpoint only (no live key used).

## v0.1 acceptance

- `cargo test --workspace` green; `cargo clippy -- -D warnings` clean.
- `peregrine --print --provider mock "list the files"` runs a full turn with a
  tool call against the mock provider, no network.
- `peregrine --print --provider openai-compatible` works against a local
  OpenAI-compatible endpoint (env `OPENAI_API_KEY` / `OPENAI_BASE_URL`).
- README with quickstart, architecture diagram (text), extension + sandbox
  roadmap; `LICENSE-MIT`, `LICENSE-APACHE`; example session + example custom tool.
