//! The agent loop (adapted from Pi `runLoop`).
//!
//! `Agent` owns state (model, tools, messages) + two queues (steer: injected
//! after the current turn; follow-up: run only when the agent would stop)
//! and one cancellation flag per run. `prompt()` errors if a run is active;
//! use `steer()` / `follow_up()` instead.
//!
//! Events go over a bounded `tokio::mpsc` channel. The loop never awaits
//! the renderer; backpressure is by coalescing (update events are dropped
//! first, structural events are counted as dropped if the channel is full).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::guardrails::{GuardrailAction, Guardrails, Verdict};
use crate::llm::{LlmClient, LlmEvent, LlmRequest, StopReason, StreamAccumulator, Usage};
use crate::tool::{ToolCtx, ToolRegistry, ToolResult};
use crate::transcript::{Message, SystemSection, ToolDef};

/// How a run ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Completed,
    MaxTurnsExceeded,
    Denied(String),
    Error(String),
    Aborted,
}

/// The outcome of one `run()`.
#[derive(Debug, Clone)]
pub struct AgentOutcome {
    pub status: RunStatus,
    pub usage: Usage,
    pub turns: usize,
    pub tool_calls: usize,
    /// Events dropped because the event channel was full (coalescing).
    pub dropped_events: usize,
}

/// Error from `prompt()` / setup — never from the loop itself, which is total.
#[derive(Debug)]
pub enum AgentError {
    AlreadyRunning,
    NoMessages,
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::AlreadyRunning => write!(f, "an agent run is already active"),
            AgentError::NoMessages => write!(f, "no messages to run on"),
        }
    }
}

impl std::error::Error for AgentError {}

/// Events emitted by the loop. The UI is just one subscriber.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    AgentStart { model: String },
    AgentEnd { status: RunStatus, usage: Usage },
    TurnStart { turn: usize },
    TurnEnd { turn: usize },
    MessageStart,
    MessageUpdate { text_delta: String, thinking: bool },
    MessageEnd { text: String },
    ToolExecutionStart { id: String, name: String },
    ToolExecutionUpdate { id: String, text: String },
    ToolExecutionEnd { id: String, name: String, is_error: bool },
}

/// What `before_tool_call` may decide. This is the ONLY permission gate.
#[derive(Debug, Clone)]
pub enum BeforeToolCall {
    Allow,
    Block { reason: String },
}

/// What `finish_turn` may decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDecision {
    End,
    Continue,
}

/// Hook type aliases (kept short so the `Hooks` struct stays readable).
pub type BeforeToolCallHook =
    Arc<dyn Fn(&str, &serde_json::Value) -> BeforeToolCall + Send + Sync>;
pub type AfterToolCallHook = Arc<dyn Fn(&str, &ToolResult) -> Option<ToolResult> + Send + Sync>;
pub type FinishTurnHook = Arc<dyn Fn() -> TurnDecision + Send + Sync>;
pub type PrepareRequestHook = Arc<dyn Fn(&mut Vec<Message>) + Send + Sync>;
pub type PrepareNextTurnHook = Arc<dyn Fn() -> Vec<Message> + Send + Sync>;
pub type TransformContextHook = Arc<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>;

/// Hooks. All total: they return values, never `Result`, and must not fail
/// the loop. All optional; `Hooks::default()` is a plain loop.
#[derive(Default)]
pub struct Hooks {
    /// The permission gate. Called after guardrail evaluation, before
    /// execution. `Block` becomes an `is_error` result.
    pub before_tool_call: Option<BeforeToolCallHook>,
    /// May override a tool result field-by-field.
    pub after_tool_call: Option<AfterToolCallHook>,
    /// Decides whether the run ends after a turn.
    pub finish_turn: Option<FinishTurnHook>,
    /// May mutate the request messages in place (e.g. compaction).
    pub prepare_request: Option<PrepareRequestHook>,
    /// May add messages before the next turn (e.g. context/model swaps).
    pub prepare_next_turn: Option<PrepareNextTurnHook>,
    /// May prune/inject at the transcript level before a request.
    pub transform_context: Option<TransformContextHook>,
}

/// Capacity of the bounded event channel.
pub const EVENT_CHANNEL_CAPACITY: usize = 256;

pub struct Agent {
    client: Arc<dyn LlmClient>,
    tools: Arc<ToolRegistry>,
    ctx: ToolCtx,
    model: String,
    messages: Vec<Message>,
    guardrails: Guardrails,
    hooks: Hooks,
    event_tx: Option<mpsc::Sender<AgentEvent>>,
    steer_queue: VecDeque<Message>,
    follow_up_queue: VecDeque<Message>,
    running: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    usage: Usage,
    turns: usize,
    tool_calls_made: AtomicUsize,
    dropped_events: Arc<AtomicUsize>,
    last_tool_defs: Vec<ToolDef>,
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: Arc<dyn LlmClient>,
        tools: Arc<ToolRegistry>,
        ctx: ToolCtx,
        model: String,
        system_sections: Vec<SystemSection>,
        guardrails: Guardrails,
        hooks: Hooks,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> Self {
        let defs = tools.defs();
        let messages = vec![Message::System {
            sections: system_sections,
            tools_added: defs.clone(),
            tools_removed: vec![],
        }];
        Self {
            client,
            tools,
            ctx,
            model,
            messages,
            guardrails,
            hooks,
            event_tx,
            steer_queue: VecDeque::new(),
            follow_up_queue: VecDeque::new(),
            running: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
            usage: Usage::default(),
            turns: 0,
            tool_calls_made: AtomicUsize::new(0),
            dropped_events: Arc::new(AtomicUsize::new(0)),
            last_tool_defs: defs,
        }
    }

    /// The current transcript (cloned).
    pub fn messages(&self) -> Vec<Message> {
        self.messages.clone()
    }

    /// Append messages directly (used for resume).
    pub fn append_messages(&mut self, messages: Vec<Message>) {
        self.messages.extend(messages);
    }

    /// Replace the transcript wholesale (used for session resume: the
    /// replayed transcript already carries its system message and tool
    /// loadout). Recomputes the loadout snapshot from the replay.
    pub fn set_messages(&mut self, messages: Vec<Message>) {
        self.last_tool_defs = crate::transcript::ToolLoadout::replay(&messages).tools;
        self.messages = messages;
    }

    /// Queue a message to be injected after the current turn.
    pub fn steer(&mut self, msg: Message) {
        self.steer_queue.push_back(msg);
    }

    /// Queue a message to run only when the agent would otherwise stop.
    pub fn follow_up(&mut self, msg: Message) {
        self.follow_up_queue.push_back(msg);
    }

    /// Abort the current run. The in-flight stream is dropped; the loop
    /// observes the flag between turns and at stream poll time.
    pub fn abort(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Send one user prompt and run the loop. Errors if a run is active.
    pub async fn prompt(&mut self, text: String) -> Result<AgentOutcome, AgentError> {
        if self
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(AgentError::AlreadyRunning);
        }
        self.cancelled.store(false, Ordering::SeqCst);
        self.messages.push(Message::user_text(text));
        let outcome = self.run_loop().await;
        self.running.store(false, Ordering::SeqCst);
        Ok(outcome)
    }

    /// Run the loop over the current transcript and queues (used after
    /// resume, or after `steer`/`follow_up` without a new prompt).
    pub async fn run(&mut self) -> Result<AgentOutcome, AgentError> {
        if self
            .running
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(AgentError::AlreadyRunning);
        }
        if self.messages.is_empty() {
            self.running.store(false, Ordering::SeqCst);
            return Err(AgentError::NoMessages);
        }
        self.cancelled.store(false, Ordering::SeqCst);
        let outcome = self.run_loop().await;
        self.running.store(false, Ordering::SeqCst);
        Ok(outcome)
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(tx) = &self.event_tx {
            match tx.try_send(event) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    // Backpressure by coalescing: the event is dropped; the
                    // loop never awaits the renderer. Count it so the
                    // outcome reports the loss honestly.
                    self.dropped_events.fetch_add(1, Ordering::Relaxed);
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {}
            }
        }
    }

    fn drain_steer(&mut self) -> Vec<Message> {
        self.steer_queue.drain(..).collect()
    }

    fn drain_follow_up(&mut self) -> Vec<Message> {
        self.follow_up_queue.drain(..).collect()
    }

    /// Append a system message describing the tool-loadout delta, if the
    /// registry changed since the last turn.
    fn append_tool_loadout_delta(&mut self) {
        let current = self.tools.defs();
        if current == self.last_tool_defs {
            return;
        }
        let added: Vec<ToolDef> = current
            .iter()
            .filter(|d| !self.last_tool_defs.iter().any(|o| o.name == d.name))
            .cloned()
            .collect();
        let removed: Vec<String> = self
            .last_tool_defs
            .iter()
            .filter(|o| !current.iter().any(|d| d.name == o.name))
            .map(|o| o.name.clone())
            .collect();
        self.messages.push(Message::System {
            sections: vec![],
            tools_added: added,
            tools_removed: removed,
        });
        self.last_tool_defs = current;
    }

    async fn run_loop(&mut self) -> AgentOutcome {
        self.emit(AgentEvent::AgentStart {
            model: self.model.clone(),
        });
        let mut status = RunStatus::Completed;
        let mut pending: Vec<Message> = self.drain_steer();
        let mut need_turn = true;

        'outer: loop {
            while need_turn {
                need_turn = false;

                // prepare_next_turn hooks may add messages / swap context.
                if let Some(hook) = &self.hooks.prepare_next_turn {
                    pending.extend(hook());
                }
                self.append_tool_loadout_delta();
                if !pending.is_empty() {
                    self.messages.append(&mut pending);
                }

                if self.cancelled.load(Ordering::SeqCst) {
                    status = RunStatus::Aborted;
                    break 'outer;
                }

                self.turns += 1;
                let turn = self.turns;
                if self.guardrails.evaluate(GuardrailAction::TurnStart { turn })
                    == Verdict::Deny
                {
                    status = RunStatus::MaxTurnsExceeded;
                    break 'outer;
                }
                self.emit(AgentEvent::TurnStart { turn });

                if let Some(hook) = &self.hooks.prepare_request {
                    hook(&mut self.messages);
                }
                let request_messages = match &self.hooks.transform_context {
                    Some(hook) => hook(self.messages.clone()),
                    None => self.messages.clone(),
                };
                let req = LlmRequest::new(
                    self.model.clone(),
                    request_messages,
                    self.tools.defs(),
                );
                let turn_timeout = self.guardrails.budgets.turn_timeout;

                self.emit(AgentEvent::MessageStart);
                let acc = match tokio::time::timeout(turn_timeout, self.collect_stream(&req)).await
                {
                    Ok(acc) => acc,
                    Err(_) => {
                        self.emit(AgentEvent::TurnEnd { turn });
                        status = RunStatus::Error(format!(
                            "turn {turn} timed out after {}s",
                            turn_timeout.as_secs()
                        ));
                        break 'outer;
                    }
                };
                self.usage.add(acc.usage);

                match acc.stop_reason {
                    Some(StopReason::Error) => {
                        let msg = acc.error.unwrap_or_else(|| "provider error".to_string());
                        self.emit(AgentEvent::MessageEnd { text: String::new() });
                        self.emit(AgentEvent::TurnEnd { turn });
                        status = RunStatus::Error(msg);
                        break 'outer;
                    }
                    Some(StopReason::Aborted) => {
                        self.emit(AgentEvent::MessageEnd { text: String::new() });
                        self.emit(AgentEvent::TurnEnd { turn });
                        status = RunStatus::Aborted;
                        break 'outer;
                    }
                    _ => {}
                }

                let stop_reason = acc.stop_reason;
                let assistant_msg = acc.into_message();
                let text = assistant_msg.text().unwrap_or_default();
                self.emit(AgentEvent::MessageEnd { text });
                // Own the call data before moving the message into the
                // transcript: results must be appended in source order.
                let owned_calls: Vec<(String, String, serde_json::Value)> = assistant_msg
                    .tool_calls()
                    .into_iter()
                    .map(|(id, name, args)| (id.to_string(), name.to_string(), args.clone()))
                    .collect();
                let had_tool_calls = !owned_calls.is_empty();
                self.messages.push(assistant_msg);

                if had_tool_calls {
                    if stop_reason == Some(StopReason::Length) {
                        // Args may be truncated: never execute them.
                        for (id, _, _) in &owned_calls {
                            self.messages.push(Message::ToolResult {
                                call_id: id.clone(),
                                result: ToolResult::error(
                                    "refused to execute tool call: response hit the length limit, arguments may be truncated",
                                ),
                            });
                        }
                    } else {
                        let call_refs: Vec<(&str, &str, &serde_json::Value)> = owned_calls
                            .iter()
                            .map(|(id, name, args)| (id.as_str(), name.as_str(), args))
                            .collect();
                        let results = self.execute_tool_calls(&call_refs).await;
                        for ((id, _, _), result) in owned_calls.iter().zip(results) {
                            self.messages.push(Message::ToolResult {
                                call_id: id.clone(),
                                result,
                            });
                        }
                    }
                }

                self.emit(AgentEvent::TurnEnd { turn });

                if let Some(hook) = &self.hooks.finish_turn {
                    if hook() == TurnDecision::End {
                        break 'outer;
                    }
                }

                pending = self.drain_steer();
                if had_tool_calls || !pending.is_empty() {
                    need_turn = true;
                }
            }

            // The agent would stop: run follow-ups, if any.
            let follow_ups = self.drain_follow_up();
            if follow_ups.is_empty() {
                break 'outer;
            }
            pending = follow_ups;
            need_turn = true;
        }

        self.emit(AgentEvent::AgentEnd {
            status: status.clone(),
            usage: self.usage,
        });
        AgentOutcome {
            status,
            usage: self.usage,
            turns: self.turns,
            tool_calls: self.tool_calls_made.load(Ordering::SeqCst),
            dropped_events: self.dropped_events.load(Ordering::Relaxed),
        }
    }

    /// Stream one assistant response, forwarding deltas as events and
    /// accumulating the message. Never panics on provider failure.
    async fn collect_stream(&self, req: &LlmRequest) -> StreamAccumulator {
        let mut stream = self.client.stream(req.clone());
        let mut acc = StreamAccumulator::new();
        while let Some(event) = stream.next().await {
            if self.cancelled.load(Ordering::SeqCst) {
                let _ = acc.feed(LlmEvent::Done {
                    usage: acc.usage,
                    stop_reason: StopReason::Aborted,
                });
                break;
            }
            match &event {
                LlmEvent::TextDelta { text } => self.emit(AgentEvent::MessageUpdate {
                    text_delta: text.clone(),
                    thinking: false,
                }),
                LlmEvent::ThinkingDelta { text } => self.emit(AgentEvent::MessageUpdate {
                    text_delta: text.clone(),
                    thinking: true,
                }),
                _ => {}
            }
            if !acc.feed(event) {
                break;
            }
        }
        acc
    }

    /// Execute one batch of tool calls: prepare sequentially (validation,
    /// guardrail evaluation, permission hook — each exactly once, in source
    /// order), execute concurrently, results in the assistant's source order.
    async fn execute_tool_calls(
        &mut self,
        calls: &[(&str, &str, &serde_json::Value)],
    ) -> Vec<ToolResult> {
        enum Prepared<'a> {
            Ready {
                tool: Arc<dyn crate::tool::Tool>,
                name: &'a str,
                args: &'a serde_json::Value,
            },
            Failed(ToolResult),
        }
        let mut prepared: Vec<Prepared<'_>> = Vec::with_capacity(calls.len());
        for (_id, name, args) in calls {
            let Some(tool) = self.tools.get(name) else {
                prepared.push(Prepared::Failed(ToolResult::error(format!(
                    "unknown tool: {name}"
                ))));
                continue;
            };
            match validate_args(tool.parameters(), args) {
                Err(e) => prepared.push(Prepared::Failed(ToolResult::error(e))),
                Ok(()) => {
                    // Guardrail evaluation is sequential, exactly once per
                    // call, in source order.
                    let verdict = self
                        .guardrails
                        .evaluate(GuardrailAction::ToolCall { name, args });
                    let verdict = self.guardrails.resolve_ask(verdict);
                    match verdict {
                        Verdict::Deny => {
                            prepared.push(Prepared::Failed(ToolResult::error(format!(
                                "denied by guardrail: tool call budget, loop detection, or turn limit (tool: {name})"
                            ))));
                        }
                        Verdict::Ask => {
                            prepared.push(Prepared::Failed(ToolResult::error(format!(
                                "tool call requires approval (ask) and no approval was given (tool: {name})"
                            ))));
                        }
                        Verdict::Allow => {
                            // The permission hook is the ONLY other gate.
                            let blocked = match &self.hooks.before_tool_call {
                                Some(hook) => match hook(name, args) {
                                    BeforeToolCall::Allow => None,
                                    BeforeToolCall::Block { reason } => Some(reason),
                                },
                                None => None,
                            };
                            match blocked {
                                Some(reason) => {
                                    prepared.push(Prepared::Failed(ToolResult::error(format!(
                                        "blocked by hook: {reason}"
                                    ))));
                                }
                                None => prepared.push(Prepared::Ready {
                                    tool: tool.clone(),
                                    name,
                                    args,
                                }),
                            }
                        }
                    }
                }
            }
        }

        // Borrow self immutably for the concurrent phase: the async blocks
        // capture a shared reference, so the `map` closure stays `FnMut`.
        let this: &Agent = self;
        let futures: Vec<_> = prepared
            .iter()
            .map(|p| async move {
                match p {
                    Prepared::Failed(result) => result.clone(),
                    Prepared::Ready { tool, name, args } => {
                        this.execute_one(tool, name, args).await
                    }
                }
            })
            .collect();
        futures::future::join_all(futures).await
    }

    /// One prepared tool call: execute (concurrent with its batch), then the
    /// after-hook. Every failure mode becomes an `is_error` result.
    async fn execute_one(
        &self,
        tool: &Arc<dyn crate::tool::Tool>,
        name: &str,
        args: &serde_json::Value,
    ) -> ToolResult {
        let seq = self.tool_calls_made.fetch_add(1, Ordering::SeqCst);
        let event_id = format!("{name}:{seq}");
        self.emit(AgentEvent::ToolExecutionStart {
            id: event_id.clone(),
            name: name.to_string(),
        });
        let mut result = tool.execute(&self.ctx, args.clone()).await;
        if let Some(hook) = &self.hooks.after_tool_call {
            if let Some(overridden) = hook(name, &result) {
                result = overridden;
            }
        }
        self.emit(AgentEvent::ToolExecutionEnd {
            id: event_id,
            name: name.to_string(),
            is_error: result.is_error,
        });
        result
    }
}

/// Light argument validation against a tool's JSON Schema: the args must be
/// an object and must carry every `required` property. Full JSON Schema
/// validation (types, formats, nested constraints) is a documented v0.1
/// limitation; the schema is still the single source of truth for the model.
pub fn validate_args(schema: schemars::Schema, args: &serde_json::Value) -> Result<(), String> {
    let obj = args
        .as_object()
        .ok_or_else(|| "tool arguments must be a JSON object".to_string())?;
    let schema_json = serde_json::to_value(schema).unwrap_or(serde_json::Value::Null);
    if let Some(required) = schema_json.get("required").and_then(|r| r.as_array()) {
        for field in required {
            if let Some(field) = field.as_str() {
                if !obj.contains_key(field) {
                    return Err(format!("missing required argument: {field}"));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmEvent;
    use crate::tool::Tool;
    use futures::stream;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// A scripted provider: each `stream()` call plays the next script.
    struct ScriptedClient {
        scripts: Mutex<VecDeque<Vec<LlmEvent>>>,
    }

    impl ScriptedClient {
        fn new(scripts: Vec<Vec<LlmEvent>>) -> Self {
            Self {
                scripts: Mutex::new(scripts.into()),
            }
        }
    }

    impl LlmClient for ScriptedClient {
        fn stream(
            &self,
            _req: LlmRequest,
        ) -> std::pin::Pin<Box<dyn futures::Stream<Item = LlmEvent> + Send + '_>> {
            let events = self
                .scripts
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| {
                    vec![LlmEvent::Done {
                        usage: Usage::default(),
                        stop_reason: StopReason::Stop,
                    }]
                });
            Box::pin(stream::iter(events))
        }
    }

    struct FakeBash {
        output: String,
    }

    impl Tool for FakeBash {
        fn name(&self) -> &str {
            "bash"
        }

        fn description(&self) -> &str {
            "Fake bash."
        }

        fn parameters(&self) -> schemars::Schema {
            schemars::schema_for!(FakeBashArgs)
        }

        fn execute<'a>(
            &'a self,
            _ctx: &'a ToolCtx,
            _args: serde_json::Value,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ToolResult> + Send + 'a>> {
            let output = self.output.clone();
            Box::pin(async move { ToolResult::ok(output) })
        }
    }

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)] // the fake ignores its args; the field proves schema derivation
    struct FakeBashArgs {
        command: String,
    }

    struct FakeOps;

    impl crate::ops::Ops for FakeOps {
        fn read_file(
            &self,
            _path: &std::path::Path,
            _offset: Option<usize>,
            _limit: Option<usize>,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = std::io::Result<String>> + Send + '_>>
        {
            Box::pin(async { Ok(String::new()) })
        }

        fn write_file(
            &self,
            _path: &std::path::Path,
            _content: &str,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = std::io::Result<()>> + Send + '_>>
        {
            Box::pin(async { Ok(()) })
        }

        fn edit_file(
            &self,
            _path: &std::path::Path,
            _edits: &[(String, String)],
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = std::io::Result<String>> + Send + '_>>
        {
            Box::pin(async { Ok(String::new()) })
        }

        fn run_command(
            &self,
            _cmd: &str,
            _timeout: Option<std::time::Duration>,
        ) -> std::pin::Pin<Box<dyn futures::Stream<Item = crate::ops::CmdChunk> + Send + '_>>
        {
            Box::pin(stream::empty())
        }
    }

    fn test_agent(scripts: Vec<Vec<LlmEvent>>) -> Agent {
        let mut tools = ToolRegistry::new();
        tools.register(Arc::new(FakeBash {
            output: "a.txt\nb.txt".to_string(),
        }));
        let ctx = ToolCtx::new("/tmp".into(), Arc::new(FakeOps));
        Agent::new(
            Arc::new(ScriptedClient::new(scripts)),
            Arc::new(tools),
            ctx,
            "mock-1".to_string(),
            vec![SystemSection::new("rules", "Be concise.")],
            Guardrails::new(crate::guardrails::Budgets::default(), false),
            Hooks::default(),
            None,
        )
    }

    fn tool_call_script() -> Vec<LlmEvent> {
        vec![
            LlmEvent::ToolCallDelta {
                id: "c1".into(),
                name: Some("bash".into()),
                args_delta: "{\"command\": \"ls\"}".into(),
            },
            LlmEvent::Done {
                usage: Usage {
                    input_tokens: 5,
                    output_tokens: 5,
                },
                stop_reason: StopReason::ToolUse,
            },
        ]
    }

    fn text_script(text: &str) -> Vec<LlmEvent> {
        vec![
            LlmEvent::TextDelta {
                text: text.to_string(),
            },
            LlmEvent::Done {
                usage: Usage::default(),
                stop_reason: StopReason::Stop,
            },
        ]
    }

    #[tokio::test]
    async fn full_turn_with_tool_call() {
        let mut agent = test_agent(vec![
            tool_call_script(),
            text_script("Here are the files: a.txt, b.txt."),
        ]);
        let outcome = agent.prompt("list the files".to_string()).await.unwrap();
        assert_eq!(outcome.status, RunStatus::Completed);
        assert_eq!(outcome.turns, 2);
        assert_eq!(outcome.tool_calls, 1);
        let msgs = agent.messages();
        let results: Vec<_> = msgs
            .iter()
            .filter_map(|m| match m {
                Message::ToolResult { result, .. } => Some(result),
                _ => None,
            })
            .collect();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "a.txt\nb.txt");
        assert!(!results[0].is_error);
        assert_eq!(outcome.usage.input_tokens, 5);
    }

    #[tokio::test]
    async fn length_stop_reason_fails_tool_calls() {
        let mut agent = test_agent(vec![vec![
            LlmEvent::ToolCallDelta {
                id: "c1".into(),
                name: Some("bash".into()),
                args_delta: "{\"command\": \"ls\"}".into(),
            },
            LlmEvent::Done {
                usage: Usage::default(),
                stop_reason: StopReason::Length,
            },
        ]]);
        let outcome = agent.prompt("list".to_string()).await.unwrap();
        assert_eq!(outcome.status, RunStatus::Completed);
        assert_eq!(outcome.tool_calls, 0, "truncated calls must not execute");
        let msgs = agent.messages();
        let result = msgs.iter().find_map(|m| match m {
            Message::ToolResult { result, .. } => Some(result),
            _ => None,
        });
        let result = result.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("length limit"));
    }

    #[tokio::test]
    async fn unknown_tool_becomes_error_result() {
        let mut agent = test_agent(vec![
            vec![
                LlmEvent::ToolCallDelta {
                    id: "c9".into(),
                    name: Some("nope".into()),
                    args_delta: "{}".into(),
                },
                LlmEvent::Done {
                    usage: Usage::default(),
                    stop_reason: StopReason::ToolUse,
                },
            ],
            text_script("done"),
        ]);
        let outcome = agent.prompt("go".to_string()).await.unwrap();
        assert_eq!(outcome.status, RunStatus::Completed);
        let msgs = agent.messages();
        let result = msgs.iter().find_map(|m| match m {
            Message::ToolResult { result, .. } => Some(result),
            _ => None,
        });
        assert!(result.unwrap().is_error);
    }

    #[tokio::test]
    async fn loop_detection_denies_repeated_calls() {
        // The model keeps issuing the identical bash call; the third
        // identical call must be denied by the guardrail.
        let script = || tool_call_script();
        let mut agent = test_agent(vec![
            script(),
            script(),
            script(),
            text_script("giving up"),
        ]);
        let outcome = agent.prompt("ls".to_string()).await.unwrap();
        assert_eq!(outcome.status, RunStatus::Completed);
        assert_eq!(outcome.tool_calls, 2, "third identical call denied");
        let msgs = agent.messages();
        let errors: Vec<_> = msgs
            .iter()
            .filter_map(|m| match m {
                Message::ToolResult { result, .. } if result.is_error => Some(result),
                _ => None,
            })
            .collect();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].content.contains("denied by guardrail"));
    }

    #[tokio::test]
    async fn follow_up_queue_runs_after_stop() {
        let mut agent = test_agent(vec![text_script("first"), text_script("second")]);
        agent.follow_up(Message::user_text("and then?"));
        let outcome = agent.prompt("start".to_string()).await.unwrap();
        assert_eq!(outcome.status, RunStatus::Completed);
        assert_eq!(outcome.turns, 2);
    }

    #[tokio::test]
    async fn prompt_errors_when_run_active() {
        let mut agent = test_agent(vec![text_script("x")]);
        agent.running.store(true, Ordering::SeqCst);
        assert!(matches!(
            agent.prompt("y".to_string()).await,
            Err(AgentError::AlreadyRunning)
        ));
        agent.running.store(false, Ordering::SeqCst);
    }

    #[tokio::test]
    async fn provider_error_ends_run() {
        let mut agent = test_agent(vec![vec![LlmEvent::Error {
            message: "model exploded".into(),
        }]]);
        let outcome = agent.prompt("hi".to_string()).await.unwrap();
        assert_eq!(outcome.status, RunStatus::Error("model exploded".into()));
    }

    #[tokio::test]
    async fn before_tool_call_hook_can_block() {
        let mut agent = test_agent(vec![tool_call_script(), text_script("ok")]);
        agent.hooks.before_tool_call = Some(Arc::new(|_, _| BeforeToolCall::Block {
            reason: "no bash today".into(),
        }));
        let outcome = agent.prompt("ls".to_string()).await.unwrap();
        assert_eq!(outcome.status, RunStatus::Completed);
        assert_eq!(outcome.tool_calls, 0);
        let msgs = agent.messages();
        let result = msgs.iter().find_map(|m| match m {
            Message::ToolResult { result, .. } => Some(result),
            _ => None,
        });
        assert!(result.unwrap().content.contains("no bash today"));
    }
}
