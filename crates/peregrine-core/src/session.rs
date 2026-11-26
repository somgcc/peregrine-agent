//! Sessions: append-only JSONL event log, one file per session
//! (`<timestamp>_<session_id>.jsonl`), header line
//! `{"type": "session", "data": {"version": 1, "id": ..., "cwd": ...}}`.
//!
//! The event log is the spine: session state, UI projections and recovery
//! all derive from replay. Resume = reload + replay. Fork/branch = a new
//! file sharing a prefix.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::agent::AgentEvent;
use crate::transcript::Message;

/// The header line of a session file. Serialized through `SessionEntry`,
/// which supplies the `"type": "session"` tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionHeader {
    pub version: u32,
    pub id: String,
    pub cwd: String,
    pub created_at: u64,
}

impl SessionHeader {
    pub fn new(id: String, cwd: &Path) -> Self {
        Self {
            version: 1,
            id,
            cwd: cwd.to_string_lossy().into_owned(),
            created_at: unix_secs(),
        }
    }
}

pub fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// One line of a session file. Adjacently tagged: the outer `type` says
/// which entry this is, and `data` carries the payload — which may itself
/// be internally tagged (e.g. `AgentEvent`), so the two tags never collide.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SessionEntry {
    Session(SessionHeader),
    Message(Message),
    Event(AgentEvent),
}

/// The result of replaying a session file.
#[derive(Debug, Default)]
pub struct ReplayedSession {
    pub header: Option<SessionHeader>,
    pub messages: Vec<Message>,
    pub events: Vec<AgentEvent>,
}

/// Append-only writer for one session file.
pub struct SessionLog {
    path: PathBuf,
    header: SessionHeader,
    file: Mutex<File>,
}

impl SessionLog {
    /// Create a new session file in `dir`. Returns the log and its header.
    pub fn create(dir: &Path, cwd: &Path) -> io::Result<(Self, SessionHeader)> {
        std::fs::create_dir_all(dir)?;
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let id = format!("{millis:x}-{}", std::process::id());
        let filename = format!("{}_{id}.jsonl", unix_secs());
        let path = dir.join(filename);
        let header = SessionHeader::new(id, cwd);
        let mut file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)?;
        writeln!(file, "{}", serde_json::to_string(&SessionEntry::Session(header.clone()))?)?;
        file.flush()?;
        Ok((
            Self {
                path,
                header: header.clone(),
                file: Mutex::new(file),
            },
            header,
        ))
    }

    /// Open an existing session file for appending (resume).
    pub fn open(path: &Path) -> io::Result<Self> {
        let replayed = replay(path)?;
        let header = replayed.header.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "session file has no header")
        })?;
        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            header,
            file: Mutex::new(file),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &SessionHeader {
        &self.header
    }

    pub fn id(&self) -> &str {
        &self.header.id
    }

    pub fn append_message(&self, msg: &Message) -> io::Result<()> {
        self.append(&SessionEntry::Message(msg.clone()))
    }

    pub fn append_event(&self, event: &AgentEvent) -> io::Result<()> {
        self.append(&SessionEntry::Event(event.clone()))
    }

    fn append(&self, entry: &SessionEntry) -> io::Result<()> {
        let line = serde_json::to_string(entry)?;
        let mut file = self.file.lock().unwrap_or_else(|e| e.into_inner());
        writeln!(file, "{line}")?;
        file.flush()
    }
}

/// Replay a session file: header, messages and events in order.
pub fn replay(path: &Path) -> io::Result<ReplayedSession> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = ReplayedSession::default();
    for (n, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let entry: SessionEntry = serde_json::from_str(&line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("session line {}: {e}", n + 1),
            )
        })?;
        match entry {
            SessionEntry::Session(h) => out.header = Some(h),
            SessionEntry::Message(m) => out.messages.push(m),
            SessionEntry::Event(e) => out.events.push(e),
        }
    }
    Ok(out)
}

/// Default session directory: `~/.config/peregrine/sessions`.
pub fn default_session_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join(".config")
            .join("peregrine")
            .join("sessions")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentEvent;

    #[test]
    fn create_append_replay_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "peregrine-session-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let cwd = Path::new("/tmp");
        let (log, header) = SessionLog::create(&dir, cwd).unwrap();
        log.append_message(&Message::user_text("hello")).unwrap();
        log.append_event(&AgentEvent::AgentStart {
            model: "mock-1".into(),
        })
        .unwrap();
        log.append_message(&Message::assistant_text("hi")).unwrap();
        drop(log);

        let replayed = replay(&dir.join(format!("{}_{}.jsonl", header.created_at, header.id)))
            .unwrap();
        assert_eq!(replayed.header.as_ref().unwrap().id, header.id);
        assert_eq!(replayed.messages.len(), 2);
        assert_eq!(replayed.messages[0].text().as_deref(), Some("hello"));
        assert_eq!(replayed.events.len(), 1);
        assert!(matches!(
            replayed.events[0],
            AgentEvent::AgentStart { .. }
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_appends_to_existing_session() {
        let dir = std::env::temp_dir().join(format!(
            "peregrine-session-test2-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let (log, header) = SessionLog::create(&dir, Path::new("/tmp")).unwrap();
        let path = log.path().to_path_buf();
        drop(log);
        let log = SessionLog::open(&path).unwrap();
        assert_eq!(log.id(), header.id);
        log.append_message(&Message::user_text("again")).unwrap();
        drop(log);
        let replayed = replay(&path).unwrap();
        assert_eq!(replayed.messages.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
