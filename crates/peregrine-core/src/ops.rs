//! `Ops`: the sandbox seam.
//!
//! Tools depend on `Ops`, never on the filesystem or process spawning
//! directly. `LocalOps` runs on the host (v0.1 default). `SandboxOps`
//! (Docker / `bwrap` / gVisor-style runtimes) implements the same trait in
//! phase 2 — no core or tool changes. Credentials never enter `Ops`.

use std::future::Future;
use std::io;
use std::path::Path;
use std::pin::Pin;
use std::time::Duration;

use futures::Stream;

/// One chunk of a running command's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CmdChunk {
    Stdout(String),
    Stderr(String),
    /// The process exited (code `None` when killed by a signal).
    Exit(Option<i32>),
}

/// Filesystem + process operations, behind a trait so the sandbox can replace
/// them. All paths are resolved under a working directory by the
/// implementation (path confinement); `Ops` itself takes paths as given.
pub trait Ops: Send + Sync {
    /// Read a file as text. `offset`/`limit` are 0-based line numbers;
    /// `None` means from the start / to the end.
    fn read_file(
        &self,
        path: &Path,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Pin<Box<dyn Future<Output = io::Result<String>> + Send + '_>>;

    /// Write a whole file, creating parent directories.
    fn write_file(
        &self,
        path: &Path,
        content: &str,
    ) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + '_>>;

    /// Apply `edits` (old_text → new_text) against the original file.
    /// Every `old_text` must occur exactly once. Returns a unified diff of
    /// the change.
    fn edit_file(
        &self,
        path: &Path,
        edits: &[(String, String)],
    ) -> Pin<Box<dyn Future<Output = io::Result<String>> + Send + '_>>;

    /// Run a shell command, streaming its output. `timeout` kills the
    /// process (and its process group) when it elapses.
    fn run_command(
        &self,
        cmd: &str,
        timeout: Option<Duration>,
    ) -> Pin<Box<dyn Stream<Item = CmdChunk> + Send + '_>>;
}
