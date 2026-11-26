//! `LocalOps`: the host implementation of `Ops`.
//!
//! Path confinement: every path resolves under the working directory,
//! symlink-aware via canonicalization. Per-file mutation queue: a mutex
//! keyed by canonical path serializes mutations to the same file while
//! different files mutate in parallel.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;
use peregrine_core::{CmdChunk, Ops};
use tokio::io::AsyncBufReadExt;
use tokio::sync::Mutex;

/// Maximum file size `LocalOps` will read or write in one call (1 MB).
pub const MAX_FILE_BYTES: u64 = 1_000_000;

/// A mutex per canonical path, so mutations to the same file serialize
/// while different files mutate in parallel.
#[derive(Default)]
struct FileLocks {
    inner: Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>,
}

impl FileLocks {
    async fn lock(&self, path: &Path) -> tokio::sync::OwnedMutexGuard<()> {
        let entry = {
            let mut map = self.inner.lock().await;
            map.entry(path.to_path_buf())
                .or_default()
                .clone()
        };
        entry.lock_owned().await
    }
}

pub struct LocalOps {
    cwd: PathBuf,
    locks: FileLocks,
}

impl LocalOps {
    /// Canonicalize the working directory once; all later resolutions are
    /// checked against it.
    pub fn new(cwd: PathBuf) -> io::Result<Self> {
        Ok(Self {
            cwd: cwd.canonicalize()?,
            locks: FileLocks::default(),
        })
    }

    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Resolve a path under the working directory, symlink-aware. A path
    /// that does not exist yet resolves through its nearest existing
    /// ancestor (so `write` can create new files in new directories).
    /// Anything escaping the working directory is refused.
    pub fn resolve(&self, path: &Path) -> io::Result<PathBuf> {
        let joined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        };
        let normalized = normalize_lexical(&joined);
        let canonical = match normalized.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                // Walk up to the nearest existing ancestor, then re-append
                // the missing components (which contain no `..` after
                // lexical normalization, so the result stays under the
                // canonical ancestor).
                let mut ancestor = normalized.as_path();
                let mut missing: Vec<std::ffi::OsString> = Vec::new();
                while !ancestor.exists() {
                    match ancestor.file_name() {
                        Some(name) => missing.push(name.to_os_string()),
                        None => {
                            return Err(io::Error::new(
                                io::ErrorKind::NotFound,
                                "path has no existing ancestor",
                            ));
                        }
                    }
                    ancestor = ancestor.parent().ok_or_else(|| {
                        io::Error::new(io::ErrorKind::NotFound, "path has no parent")
                    })?;
                }
                let mut canonical = ancestor.canonicalize()?;
                for comp in missing.iter().rev() {
                    canonical.push(comp);
                }
                canonical
            }
        };
        if !canonical.starts_with(&self.cwd) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("path escapes the working directory: {}", path.display()),
            ));
        }
        Ok(canonical)
    }
}

impl Ops for LocalOps {
    fn read_file(
        &self,
        path: &Path,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = io::Result<String>> + Send + '_>>
    {
        let resolved = self.resolve(path);
        Box::pin(async move {
            let path = resolved?;
            let meta = tokio::fs::metadata(&path).await?;
            if meta.len() > MAX_FILE_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::FileTooLarge,
                    format!(
                        "file is {} bytes, over the {} byte cap",
                        meta.len(),
                        MAX_FILE_BYTES
                    ),
                ));
            }
            let text = tokio::fs::read_to_string(&path).await?;
            let lines: Vec<&str> = text.lines().collect();
            let start = offset.unwrap_or(0).min(lines.len());
            let end = match limit {
                Some(n) => (start + n).min(lines.len()),
                None => lines.len(),
            };
            Ok(lines[start..end].join("\n"))
        })
    }

    fn write_file(
        &self,
        path: &Path,
        content: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = io::Result<()>> + Send + '_>> {
        let resolved = self.resolve(path);
        let content = content.to_string();
        Box::pin(async move {
            if content.len() as u64 > MAX_FILE_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::FileTooLarge,
                    format!("content is over the {} byte cap", MAX_FILE_BYTES),
                ));
            }
            let path = resolved?;
            let _guard = self.locks.lock(&path).await;
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&path, content).await
        })
    }

    fn edit_file(
        &self,
        path: &Path,
        edits: &[(String, String)],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = io::Result<String>> + Send + '_>>
    {
        let resolved = self.resolve(path);
        let edits: Vec<(String, String)> = edits.to_vec();
        Box::pin(async move {
            let path = resolved?;
            let _guard = self.locks.lock(&path).await;
            let original = tokio::fs::read_to_string(&path).await?;
            // Every edit is matched against the ORIGINAL file: each
            // old_text must occur exactly once there.
            for (old, _) in &edits {
                let count = original.matches(old.as_str()).count();
                if count == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("old_text not found in file: {old:?}"),
                    ));
                }
                if count > 1 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("old_text is not unique in file (occurs {count} times): {old:?}"),
                    ));
                }
            }
            let mut updated = original.clone();
            for (old, new) in &edits {
                updated = updated.replacen(old.as_str(), new.as_str(), 1);
            }
            tokio::fs::write(&path, &updated).await?;
            Ok(unified_diff(&path, &original, &updated))
        })
    }

    fn run_command(
        &self,
        cmd: &str,
        timeout: Option<Duration>,
    ) -> std::pin::Pin<Box<dyn Stream<Item = CmdChunk> + Send + '_>> {
        let cmd = cmd.to_string();
        let cwd = self.cwd.clone();
        Box::pin(async_stream::stream! {
            let mut child = match spawn_shell(&cmd, &cwd) {
                Ok(c) => c,
                Err(e) => {
                    yield CmdChunk::Stderr(format!("failed to spawn shell: {e}\n"));
                    yield CmdChunk::Exit(None);
                    return;
                }
            };
            let pid = child.id();
            let stdout = child.stdout.take();
            let stderr = child.stderr.take();
            let (tx, mut rx) = tokio::sync::mpsc::channel::<CmdChunk>(64);
            if let Some(out) = stdout {
                let tx = tx.clone();
                tokio::spawn(async move { pump_stream(out, false, tx).await });
            }
            if let Some(err) = stderr {
                let tx = tx.clone();
                tokio::spawn(async move { pump_stream(err, true, tx).await });
            }
            drop(tx);
            let deadline = timeout.map(|t| tokio::time::Instant::now() + t);
            loop {
                let next = match deadline {
                    Some(d) => match tokio::time::timeout_at(d, rx.recv()).await {
                        Ok(c) => c,
                        Err(_) => {
                            // Timeout: kill the whole process group.
                            kill_process_group(pid);
                            let _ = child.kill().await;
                            yield CmdChunk::Exit(None);
                            return;
                        }
                    },
                    None => rx.recv().await,
                };
                match next {
                    Some(chunk) => yield chunk,
                    None => break,
                }
            }
            match child.wait().await {
                Ok(status) => yield CmdChunk::Exit(status.code()),
                Err(_) => yield CmdChunk::Exit(None),
            }
        })
    }
}

async fn pump_stream<R>(reader: R, is_stderr: bool, tx: tokio::sync::mpsc::Sender<CmdChunk>)
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    let mut lines = tokio::io::BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let chunk = if is_stderr {
            CmdChunk::Stderr(format!("{line}\n"))
        } else {
            CmdChunk::Stdout(format!("{line}\n"))
        };
        if tx.send(chunk).await.is_err() {
            break;
        }
    }
}

#[cfg(unix)]
fn spawn_shell(cmd: &str, cwd: &Path) -> io::Result<tokio::process::Child> {
    tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .process_group(0)
        .spawn()
}

#[cfg(not(unix))]
fn spawn_shell(cmd: &str, cwd: &Path) -> io::Result<tokio::process::Child> {
    tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(cwd)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
}

/// Kill a process group (Unix) so a timed-out command's children die too.
#[cfg(unix)]
fn kill_process_group(pid: Option<u32>) {
    if let Some(pid) = pid {
        unsafe {
            libc::killpg(pid as i32, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: Option<u32>) {}

/// Resolve `.` and `..` lexically (no filesystem access). The result is
/// absolute when the input is absolute.
fn normalize_lexical(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// A minimal unified diff: one hunk covering the change, with 3 lines of
/// context. Enough for a model (or a human) to see what an edit did.
pub fn unified_diff(path: &Path, old: &str, new: &str) -> String {
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();
    let mut prefix = 0;
    while prefix < old_lines.len()
        && prefix < new_lines.len()
        && old_lines[prefix] == new_lines[prefix]
    {
        prefix += 1;
    }
    let mut suffix = 0;
    while suffix < old_lines.len() - prefix
        && suffix < new_lines.len() - prefix
        && old_lines[old_lines.len() - 1 - suffix] == new_lines[new_lines.len() - 1 - suffix]
    {
        suffix += 1;
    }
    let old_start = prefix.saturating_sub(3);
    let new_start = prefix.saturating_sub(3);
    let old_end = (old_lines.len() - suffix + 3).min(old_lines.len());
    let new_end = (new_lines.len() - suffix + 3).min(new_lines.len());
    let old_count = old_end - old_start;
    let new_count = new_end - new_start;

    let mut out = String::new();
    out.push_str(&format!("--- a/{}\n", path.display()));
    out.push_str(&format!("+++ b/{}\n", path.display()));
    out.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        old_start + 1,
        old_count,
        new_start + 1,
        new_count
    ));
    for line in &old_lines[old_start..prefix] {
        out.push_str(&format!(" {line}\n"));
    }
    for line in &old_lines[prefix..old_lines.len() - suffix] {
        out.push_str(&format!("-{line}\n"));
    }
    for line in &new_lines[prefix..new_lines.len() - suffix] {
        out.push_str(&format!("+{line}\n"));
    }
    for line in &old_lines[old_lines.len() - suffix..old_end] {
        out.push_str(&format!(" {line}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_shows_changed_lines_with_context() {
        let old = "a\nb\nc\nd\ne\nf\ng\n";
        let new = "a\nb\nc\nD\ne\nf\ng\n";
        let diff = unified_diff(Path::new("x.txt"), old, new);
        assert!(diff.contains("--- a/x.txt"));
        assert!(diff.contains("+++ b/x.txt"));
        assert!(diff.contains("-d\n"));
        assert!(diff.contains("+D\n"));
        assert!(diff.contains(" c\n"));
        assert!(diff.contains(" e\n"));
    }

    #[test]
    fn diff_of_identical_files_is_empty_hunk() {
        let diff = unified_diff(Path::new("x.txt"), "a\nb\n", "a\nb\n");
        assert!(diff.contains("@@ -1,2 +1,2 @@"));
        assert!(!diff.contains("-a"));
    }
}
