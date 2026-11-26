//! `peregrine-core`: the agent loop, transcript, tool trait, `Ops` trait,
//! guardrails, and JSONL sessions.
//!
//! This crate depends on NOTHING provider- or tool-specific. It only knows
//! `LlmClient` (streams assistant events) and `Tool` (executes). Providers,
//! tools, UI, and the sandbox are all downstream.

pub mod agent;
pub mod guardrails;
pub mod llm;
pub mod ops;
pub mod session;
pub mod tool;
pub mod transcript;

pub use agent::{Agent, AgentError, AgentEvent, AgentOutcome, BeforeToolCall, Hooks, RunStatus, TurnDecision, EVENT_CHANNEL_CAPACITY};
pub use guardrails::{Budgets, GuardrailAction, Guardrails, Verdict};
pub use llm::{LlmClient, LlmEvent, LlmRequest, StopReason, Usage};
pub use ops::{CmdChunk, Ops};
pub use session::{SessionLog, default_session_dir, replay};
pub use tool::{Tool, ToolCtx, ToolRegistry, ToolResult};
pub use transcript::{AssistantContent, Message, SystemSection, ToolDef, ToolLoadout, UserContent};
