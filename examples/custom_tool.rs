//! Example: a custom tool.
//!
//! Defines a `word_count` tool (counts the words in a file) from scratch,
//! registers it alongside the built-ins, and drives it with a scripted
//! provider that asks for a word count and then answers from the result.
//!
//! Run with:
//! ```sh
//! cargo run -p peregrine --example custom_tool
//! ```
//!
//! This shows the whole extension surface a new tool needs: one struct,
//! one argument struct (serde + schemars derive the JSON Schema the model
//! sees), and `register`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::stream;
use peregrine_core::{
    Agent, Budgets, Guardrails, Hooks, LlmClient, LlmEvent, LlmRequest, Message, RunStatus,
    StopReason, SystemSection, Tool, ToolCtx, ToolRegistry, ToolResult, Usage,
};
use peregrine_tools::{LocalOps, register_builtin};
use schemars::Schema;
use serde::Deserialize;

/// `word_count {path}` — counts whitespace-separated words in a file.
pub struct WordCountTool;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct WordCountArgs {
    /// Path to count, relative to the working directory.
    pub path: String,
}

impl Tool for WordCountTool {
    fn name(&self) -> &str {
        "word_count"
    }

    fn description(&self) -> &str {
        "Count the whitespace-separated words in a file."
    }

    fn parameters(&self) -> Schema {
        schemars::schema_for!(WordCountArgs)
    }

    fn execute<'a>(
        &'a self,
        ctx: &'a ToolCtx,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        Box::pin(async move {
            let args: WordCountArgs = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::error(format!("invalid args: {e}")),
            };
            match ctx
                .ops
                .read_file(std::path::Path::new(&args.path), None, None)
                .await
            {
                Ok(text) => {
                    let words = text.split_whitespace().count();
                    ToolResult::ok(format!("{} has {words} words.", args.path))
                }
                Err(e) => ToolResult::error(format!("word_count failed: {e}")),
            }
        })
    }
}

/// A scripted provider: first turn calls `word_count`, second turn answers.
struct ScriptedClient;

impl LlmClient for ScriptedClient {
    fn stream(
        &self,
        req: LlmRequest,
    ) -> Pin<Box<dyn futures::Stream<Item = LlmEvent> + Send + '_>> {
        let last_is_tool_result = matches!(req.messages.last(), Some(Message::ToolResult { .. }));
        let events = if last_is_tool_result {
            let summary = req
                .messages
                .last()
                .and_then(|m| m.text())
                .unwrap_or_default();
            vec![
                LlmEvent::TextDelta {
                    text: format!("Done — {summary}"),
                },
                LlmEvent::Done {
                    usage: Usage::default(),
                    stop_reason: StopReason::Stop,
                },
            ]
        } else {
            vec![
                LlmEvent::ToolCallDelta {
                    id: "call-1".into(),
                    name: Some("word_count".into()),
                    args_delta: "{\"path\": \"notes.txt\"}".into(),
                },
                LlmEvent::Done {
                    usage: Usage::default(),
                    stop_reason: StopReason::ToolUse,
                },
            ]
        };
        Box::pin(stream::iter(events))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A scratch working directory with something to count.
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("notes.txt"), "one two three four five\n")?;

    let ops = Arc::new(LocalOps::new(dir.path().to_path_buf())?);
    let ctx = ToolCtx::new(dir.path().to_path_buf(), ops);
    let mut registry = ToolRegistry::new();
    register_builtin(&mut registry);
    registry.register(Arc::new(WordCountTool));

    let mut agent = Agent::new(
        Arc::new(ScriptedClient),
        Arc::new(registry),
        ctx,
        "scripted-1".to_string(),
        vec![SystemSection::new("rules", "Be concise.")],
        Guardrails::new(Budgets::default(), false),
        Hooks::default(),
        None,
    );
    let outcome = agent.prompt("count the words in notes.txt".to_string()).await?;
    assert_eq!(outcome.status, RunStatus::Completed);
    assert_eq!(outcome.tool_calls, 1);
    let text = agent
        .messages()
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::Assistant { .. } => m.text(),
            _ => None,
        })
        .unwrap_or_default();
    println!("{text}");
    Ok(())
}
