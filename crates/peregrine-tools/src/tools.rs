//! The four built-in tools (Pi's set): `read`, `write`, `edit`, `bash`.
//!
//! Tools are definition-first: each argument struct derives `serde` +
//! `schemars` from one definition, so the JSON Schema the model sees and
//! the types the tool parses can never drift apart. Tools depend on `Ops`,
//! never on the filesystem or spawning directly.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use peregrine_core::{CmdChunk, Tool, ToolCtx, ToolRegistry, ToolResult};
use schemars::Schema;
use serde::Deserialize;

/// Output larger than this is truncated, with the full output spilled to a
/// temp file whose path is reported in the result.
pub const BASH_OUTPUT_CAP: usize = 100_000;

/// Default timeout for `bash` when the caller gives none (2 minutes).
pub const BASH_DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// Register all four built-in tools into a registry.
pub fn register_builtin(registry: &mut ToolRegistry) {
    registry.register(Arc::new(ReadTool));
    registry.register(Arc::new(WriteTool));
    registry.register(Arc::new(EditTool));
    registry.register(Arc::new(BashTool));
}

/// All four built-in tools, as a vector.
pub fn builtin_tools() -> Vec<Arc<dyn Tool>> {
    let mut registry = ToolRegistry::new();
    register_builtin(&mut registry);
    let mut tools = Vec::new();
    for name in registry.names() {
        tools.push(registry.get(name).unwrap().clone());
    }
    tools
}

fn parse_args<T>(args: serde_json::Value) -> Result<T, ToolResult>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(args).map_err(|e| ToolResult::error(format!("invalid args: {e}")))
}

// ---------------------------------------------------------------- read

pub struct ReadTool;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct ReadArgs {
    /// Path to read, relative to the working directory.
    pub path: String,
    /// First line to return (0-based). Defaults to 0.
    #[serde(default)]
    pub offset: Option<usize>,
    /// Maximum lines to return. Defaults to the whole file.
    #[serde(default)]
    pub limit: Option<usize>,
}

impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read a file as text, optionally starting at a line offset and limited to a number of lines. Paths resolve under the working directory."
    }

    fn parameters(&self) -> Schema {
        schemars::schema_for!(ReadArgs)
    }

    fn execute<'a>(
        &'a self,
        ctx: &'a ToolCtx,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        let args = match parse_args::<ReadArgs>(args) {
            Ok(a) => a,
            Err(e) => return Box::pin(async move { e }),
        };
        Box::pin(async move {
            let path = std::path::Path::new(&args.path);
            match ctx.ops.read_file(path, args.offset, args.limit).await {
                Ok(text) => ToolResult::ok(text),
                Err(e) => ToolResult::error(format!("read failed: {e}")),
            }
        })
    }
}

// ---------------------------------------------------------------- write

pub struct WriteTool;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct WriteArgs {
    /// Path to write, relative to the working directory. Parent directories are created.
    pub path: String,
    /// Whole new file content (overwrites).
    pub content: String,
}

impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write a whole file, creating parent directories. Overwrites existing content."
    }

    fn parameters(&self) -> Schema {
        schemars::schema_for!(WriteArgs)
    }

    fn execute<'a>(
        &'a self,
        ctx: &'a ToolCtx,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        let args = match parse_args::<WriteArgs>(args) {
            Ok(a) => a,
            Err(e) => return Box::pin(async move { e }),
        };
        Box::pin(async move {
            let path = std::path::Path::new(&args.path);
            match ctx.ops.write_file(path, &args.content).await {
                Ok(()) => ToolResult::ok(format!("wrote {}", args.path)),
                Err(e) => ToolResult::error(format!("write failed: {e}")),
            }
        })
    }
}

// ---------------------------------------------------------------- edit

pub struct EditTool;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct EditArgs {
    /// Path to edit, relative to the working directory.
    pub path: String,
    /// Edits to apply. Every `old_text` is matched against the original
    /// file and must occur exactly once.
    pub edits: Vec<Edit>,
}

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}

impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing exact text. Each old_text must occur exactly once in the original file. Returns a unified diff of the change."
    }

    fn parameters(&self) -> Schema {
        schemars::schema_for!(EditArgs)
    }

    fn execute<'a>(
        &'a self,
        ctx: &'a ToolCtx,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        let args = match parse_args::<EditArgs>(args) {
            Ok(a) => a,
            Err(e) => return Box::pin(async move { e }),
        };
        Box::pin(async move {
            if args.edits.is_empty() {
                return ToolResult::error("edit failed: no edits given");
            }
            let path = std::path::Path::new(&args.path);
            let pairs: Vec<(String, String)> = args
                .edits
                .into_iter()
                .map(|e| (e.old_text, e.new_text))
                .collect();
            match ctx.ops.edit_file(path, &pairs).await {
                Ok(diff) => ToolResult::ok(diff),
                Err(e) => ToolResult::error(format!("edit failed: {e}")),
            }
        })
    }
}

// ---------------------------------------------------------------- bash

pub struct BashTool;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct BashArgs {
    /// Shell command to run (via `sh -c`).
    pub command: String,
    /// Timeout in seconds. Defaults to 120. On timeout the process group
    /// is killed.
    #[serde(default)]
    pub timeout: Option<u64>,
}

impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Run a shell command. stdout and stderr are merged. Output over 100KB is truncated, with the full output spilled to a temp file."
    }

    fn parameters(&self) -> Schema {
        schemars::schema_for!(BashArgs)
    }

    fn execute<'a>(
        &'a self,
        ctx: &'a ToolCtx,
        args: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        let args = match parse_args::<BashArgs>(args) {
            Ok(a) => a,
            Err(e) => return Box::pin(async move { e }),
        };
        Box::pin(async move {
            let timeout = args
                .timeout
                .map(Duration::from_secs)
                .unwrap_or(BASH_DEFAULT_TIMEOUT);
            let mut output = String::new();
            let mut exit_code: Option<Option<i32>> = None;
            let mut stream = ctx.ops.run_command(&args.command, Some(timeout));
            while let Some(chunk) = stream.next().await {
                match chunk {
                    CmdChunk::Stdout(text) | CmdChunk::Stderr(text) => {
                        output.push_str(&text);
                    }
                    CmdChunk::Exit(code) => exit_code = Some(code),
                }
            }
            let code = exit_code.flatten();
            let truncated = output.len() > BASH_OUTPUT_CAP;
            let shown: String = output.chars().take(BASH_OUTPUT_CAP).collect();
            let mut content = shown;
            if truncated {
                let spill = std::env::temp_dir().join(format!(
                    "peregrine-bash-{}-{}.log",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0)
                ));
                if std::fs::write(&spill, &output).is_ok() {
                    content.push_str(&format!(
                        "\n[output truncated at {} chars; full output spilled to {}]",
                        BASH_OUTPUT_CAP,
                        spill.display()
                    ));
                }
            }
            match code {
                Some(0) => ToolResult::ok(content),
                Some(code) => ToolResult::error(format!("{content}\n[exit code {code}]")),
                None => ToolResult::error(format!("{content}\n[command killed or did not exit]")),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_ops::LocalOps;

    fn ctx_in(dir: &std::path::Path) -> ToolCtx {
        ToolCtx::new(
            dir.to_path_buf(),
            Arc::new(LocalOps::new(dir.to_path_buf()).unwrap()),
        )
    }

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let dir = tempdir();
        let ctx = ctx_in(dir.path());
        let write = WriteTool;
        let r = write
            .execute(
                &ctx,
                serde_json::json!({"path": "a/b.txt", "content": "hello\nworld\n"}),
            )
            .await;
        assert!(!r.is_error, "{}", r.content);
        let read = ReadTool;
        let r = read
            .execute(&ctx, serde_json::json!({"path": "a/b.txt"}))
            .await;
        assert_eq!(r.content, "hello\nworld");
        // offset/limit are line-based.
        let r = read
            .execute(
                &ctx,
                serde_json::json!({"path": "a/b.txt", "offset": 1, "limit": 1}),
            )
            .await;
        assert_eq!(r.content, "world");
    }

    #[tokio::test]
    async fn edit_returns_diff_and_applies() {
        let dir = tempdir();
        let ctx = ctx_in(dir.path());
        WriteTool
            .execute(
                &ctx,
                serde_json::json!({"path": "x.txt", "content": "one\ntwo\nthree\n"}),
            )
            .await;
        let r = EditTool
            .execute(
                &ctx,
                serde_json::json!({"path": "x.txt", "edits": [{"old_text": "two", "new_text": "TWO"}]}),
            )
            .await;
        assert!(!r.is_error, "{}", r.content);
        assert!(r.content.contains("-two"));
        assert!(r.content.contains("+TWO"));
        let r = ReadTool
            .execute(&ctx, serde_json::json!({"path": "x.txt"}))
            .await;
        assert_eq!(r.content, "one\nTWO\nthree");
    }

    #[tokio::test]
    async fn edit_rejects_non_unique_old_text() {
        let dir = tempdir();
        let ctx = ctx_in(dir.path());
        WriteTool
            .execute(
                &ctx,
                serde_json::json!({"path": "x.txt", "content": "a\na\n"}),
            )
            .await;
        let r = EditTool
            .execute(
                &ctx,
                serde_json::json!({"path": "x.txt", "edits": [{"old_text": "a", "new_text": "b"}]}),
            )
            .await;
        assert!(r.is_error);
        assert!(r.content.contains("not unique"));
    }

    #[tokio::test]
    async fn edit_rejects_missing_old_text() {
        let dir = tempdir();
        let ctx = ctx_in(dir.path());
        WriteTool
            .execute(
                &ctx,
                serde_json::json!({"path": "x.txt", "content": "a\n"}),
            )
            .await;
        let r = EditTool
            .execute(
                &ctx,
                serde_json::json!({"path": "x.txt", "edits": [{"old_text": "zzz", "new_text": "b"}]}),
            )
            .await;
        assert!(r.is_error);
        assert!(r.content.contains("not found"));
    }

    #[tokio::test]
    async fn bash_runs_and_reports_exit_code() {
        let dir = tempdir();
        let ctx = ctx_in(dir.path());
        let r = BashTool
            .execute(&ctx, serde_json::json!({"command": "echo hi; echo err >&2"}))
            .await;
        assert!(!r.is_error, "{}", r.content);
        assert!(r.content.contains("hi"));
        assert!(r.content.contains("err"));
        let r = BashTool
            .execute(&ctx, serde_json::json!({"command": "exit 3"}))
            .await;
        assert!(r.is_error);
        assert!(r.content.contains("exit code 3"));
    }

    #[tokio::test]
    async fn bash_timeout_kills_quickly() {
        let dir = tempdir();
        let ctx = ctx_in(dir.path());
        let start = std::time::Instant::now();
        let r = BashTool
            .execute(
                &ctx,
                serde_json::json!({"command": "sleep 30", "timeout": 1}),
            )
            .await;
        assert!(start.elapsed() < Duration::from_secs(10));
        assert!(r.is_error);
        assert!(r.content.contains("killed") || r.content.contains("did not exit"));
    }

    #[tokio::test]
    async fn path_confinement_blocks_escape() {
        let dir = tempdir();
        let ctx = ctx_in(dir.path());
        let r = ReadTool
            .execute(&ctx, serde_json::json!({"path": "../outside.txt"}))
            .await;
        assert!(r.is_error);
        assert!(r.content.contains("escapes"));
        let r = WriteTool
            .execute(
                &ctx,
                serde_json::json!({"path": "../evil.txt", "content": "x"}),
            )
            .await;
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn read_missing_file_is_error() {
        let dir = tempdir();
        let ctx = ctx_in(dir.path());
        let r = ReadTool
            .execute(&ctx, serde_json::json!({"path": "nope.txt"}))
            .await;
        assert!(r.is_error);
    }

    #[tokio::test]
    async fn parallel_edits_to_same_file_serialize() {
        let dir = tempdir();
        let ctx = ctx_in(dir.path());
        WriteTool
            .execute(
                &ctx,
                serde_json::json!({"path": "x.txt", "content": "alpha\nbeta\n"}),
            )
            .await;
        let ctx = Arc::new(ctx);
        let c1 = ctx.clone();
        let c2 = ctx.clone();
        let (r1, r2) = tokio::join!(
            EditTool.execute(
                &c1,
                serde_json::json!({"path": "x.txt", "edits": [{"old_text": "alpha", "new_text": "ALPHA"}]}),
            ),
            EditTool.execute(
                &c2,
                serde_json::json!({"path": "x.txt", "edits": [{"old_text": "beta", "new_text": "BETA"}]}),
            ),
        );
        assert!(!r1.is_error, "{}", r1.content);
        assert!(!r2.is_error, "{}", r2.content);
        let r = ReadTool
            .execute(&ctx, serde_json::json!({"path": "x.txt"}))
            .await;
        assert!(r.content.contains("ALPHA"));
        assert!(r.content.contains("BETA"));
    }
}
