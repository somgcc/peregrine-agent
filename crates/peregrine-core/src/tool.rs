//! The `Tool` trait and the tool registry.
//!
//! Tools are definition-first: one definition gives the name, description,
//! JSON Schema (via `schemars`, derived from the same struct `serde` uses)
//! and the executor. The core adapts definitions; it never knows a concrete
//! tool.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use schemars::Schema;
use serde::{Deserialize, Serialize};

use crate::ops::Ops;

/// The result of one tool execution. Errors are values (`is_error`), never
/// panics — the loop turns them into `ToolResult` messages and continues.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// What a tool execution may use: the working directory and the `Ops` seam.
/// Tools depend on `Ops`, never on the filesystem or process spawning
/// directly — `SandboxOps` plugs in here in phase 2 without touching tools.
#[derive(Clone)]
pub struct ToolCtx {
    pub cwd: std::path::PathBuf,
    pub ops: Arc<dyn Ops>,
    /// Optional sink for live output chunks (e.g. `bash` stdout/stderr).
    /// The agent sets one per tool call; it forwards chunks to the event
    /// channel as `AgentEvent::ToolExecutionUpdate`. `None` (the default)
    /// means the tool buffers silently and only the final result is seen.
    /// The sink must never block: it is called from the tool's own task.
    pub output_sink: Option<ToolOutputSink>,
}

/// A callback receiving one live output chunk from a running tool.
pub type ToolOutputSink = Arc<dyn Fn(&str) + Send + Sync>;

impl ToolCtx {
    pub fn new(cwd: std::path::PathBuf, ops: Arc<dyn Ops>) -> Self {
        Self {
            cwd,
            ops,
            output_sink: None,
        }
    }

    /// Attach a live-output sink (used by the agent, per tool call).
    pub fn with_output_sink(mut self, sink: ToolOutputSink) -> Self {
        self.output_sink = Some(sink);
        self
    }

    /// Emit one live output chunk, if a sink is attached. No-op otherwise.
    pub fn emit_output(&self, text: &str) {
        if let Some(sink) = &self.output_sink {
            sink(text);
        }
    }
}

/// A tool the agent can call.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for the tool's arguments object, derived with `schemars`
    /// from the same struct `serde` deserializes.
    fn parameters(&self) -> Schema;
    fn execute<'a>(
        &'a self,
        ctx: &'a ToolCtx,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>>;
}

/// A registry of tools, keyed by name. Lookup is by exact name; unknown
/// tools become `is_error` results at call time, never a loop crash.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Tool declarations as the model sees them, sorted by name for a stable
    /// prompt.
    pub fn defs(&self) -> Vec<crate::transcript::ToolDef> {
        let mut defs: Vec<_> = self
            .tools
            .values()
            .map(|t| crate::transcript::ToolDef {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: serde_json::to_value(t.parameters())
                    .unwrap_or(serde_json::json!({"type": "object"})),
            })
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes its input."
        }

        fn parameters(&self) -> Schema {
            schemars::schema_for!(EchoArgs)
        }

        fn execute<'a>(
            &'a self,
            _ctx: &'a ToolCtx,
            args: serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
            Box::pin(async move {
                match serde_json::from_value::<EchoArgs>(args) {
                    Ok(parsed) => ToolResult::ok(parsed.text),
                    Err(e) => ToolResult::error(format!("invalid args: {e}")),
                }
            })
        }
    }

    #[derive(Deserialize, schemars::JsonSchema)]
    struct EchoArgs {
        text: String,
    }

    #[test]
    fn registry_defs_are_sorted_and_schema_derived() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool));
        let defs = reg.defs();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
        assert_eq!(defs[0].parameters["properties"]["text"]["type"], "string");
        assert!(defs[0].parameters["required"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("text")));
    }
}
