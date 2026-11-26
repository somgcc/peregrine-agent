//! Peregrine CLI: interactive REPL, `--print`, `--json`.
//!
//! Modes (borrowed from Pi):
//! * interactive REPL (default): streaming tokens, input preempts rendering
//!   (line-based in v0.1; ratatui/crossterm in v0.2).
//! * `--print`: run once, print the final text, exit.
//! * `--json`: NDJSON events on stdout (machine-readable, RPC-friendly).
//! * `--provider`, `--model`, `--session` (resume), `--cwd`.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use peregrine_core::{
    Agent, AgentEvent, AgentOutcome, Budgets, Guardrails, Hooks, RunStatus, SessionLog,
    SystemSection, ToolCtx, ToolRegistry, default_session_dir, EVENT_CHANNEL_CAPACITY,
};
use peregrine_llm::{ClientOptions, ProviderKind, create_client};
use peregrine_tools::{LocalOps, register_builtin};
use tokio::io::AsyncBufReadExt;
use tokio::sync::mpsc;

const VERSION: &str = env!("CARGO_PKG_VERSION");

const USAGE: &str = "peregrine — a fast, efficient, responsive agent framework

USAGE:
    peregrine [OPTIONS] [PROMPT]
    peregrine --print [OPTIONS] <PROMPT>
    peregrine --json [OPTIONS] <PROMPT>

OPTIONS:
    --provider <name>   mock | openai-compatible | anthropic (default: mock)
    --model <model>     model id (default: provider default)
    --session <id|path> resume a session (default: start a new one)
    --cwd <dir>         working directory (default: current directory)
    --print             run once, print the final text, exit
    --json              stream NDJSON events on stdout
    --help              print this help
    --version           print the version

EXAMPLES:
    peregrine --print --provider mock \"list the files\"
    peregrine --provider openai-compatible --print \"fix the failing test\"
    peregrine                       # interactive REPL
";

#[derive(Debug, Default)]
struct Args {
    provider: String,
    model: Option<String>,
    session: Option<String>,
    cwd: Option<PathBuf>,
    print: bool,
    json: bool,
    prompt: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        provider: "mock".to_string(),
        ..Default::default()
    };
    let mut positional: Vec<String> = Vec::new();
    let mut it = std::env::args().skip(1).peekable();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--version" | "-V" => {
                println!("peregrine {VERSION}");
                std::process::exit(0);
            }
            "--print" => args.print = true,
            "--json" => args.json = true,
            "--provider" => {
                args.provider = it
                    .next()
                    .ok_or_else(|| "--provider needs a value".to_string())?;
            }
            "--model" => {
                args.model = Some(
                    it.next()
                        .ok_or_else(|| "--model needs a value".to_string())?,
                );
            }
            "--session" => {
                args.session = Some(
                    it.next()
                        .ok_or_else(|| "--session needs a value".to_string())?,
                );
            }
            "--cwd" => {
                args.cwd = Some(PathBuf::from(
                    it.next().ok_or_else(|| "--cwd needs a value".to_string())?,
                ));
            }
            _ if arg.starts_with("--provider=") => {
                args.provider = arg.trim_start_matches("--provider=").to_string();
            }
            _ if arg.starts_with("--model=") => {
                args.model = Some(arg.trim_start_matches("--model=").to_string());
            }
            _ if arg.starts_with("--session=") => {
                args.session = Some(arg.trim_start_matches("--session=").to_string());
            }
            _ if arg.starts_with("--cwd=") => {
                args.cwd = Some(PathBuf::from(arg.trim_start_matches("--cwd=")));
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown flag: {arg}"));
            }
            _ => positional.push(arg),
        }
    }
    if !positional.is_empty() {
        args.prompt = Some(positional.join(" "));
    }
    Ok(args)
}

fn default_sections() -> Vec<SystemSection> {
    vec![
        SystemSection::new(
            "identity",
            "You are Peregrine, a fast, efficient coding assistant. Help the user inspect files, run commands, and modify code.",
        ),
        SystemSection::new(
            "rules",
            "Be concise. Use the tools to inspect before you change anything. Prefer small, exact edits over rewrites.",
        ),
    ]
}

#[tokio::main]
async fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("peregrine: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(args).await {
        eprintln!("peregrine: {e}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let cwd = match &args.cwd {
        Some(dir) => dir.canonicalize()?,
        None => std::env::current_dir()?,
    };
    let ops = Arc::new(LocalOps::new(cwd.clone())?);
    let ctx = ToolCtx::new(cwd.clone(), ops);
    let mut registry = ToolRegistry::new();
    register_builtin(&mut registry);

    let kind = ProviderKind::parse(&args.provider)?;
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| kind.default_model().to_string());
    let client = create_client(
        kind,
        ClientOptions {
            model: Some(model.clone()),
            base_url: None,
            api_key: None,
        },
    )?;

    let interactive = args.prompt.is_none() && !args.print && !args.json;
    let (tx, rx) = mpsc::channel::<AgentEvent>(EVENT_CHANNEL_CAPACITY);
    let mut agent = Agent::new(
        client,
        Arc::new(registry),
        ctx,
        model.clone(),
        default_sections(),
        Guardrails::new(Budgets::default(), interactive),
        Hooks::default(),
        Some(tx),
    );

    // Session: resume or create. The log is shared with the event drain
    // task (events are appended as they arrive).
    let session_dir = default_session_dir();
    let session_log: Option<Arc<SessionLog>> = match (&args.session, &session_dir) {
        (Some(spec), _) => {
            let path = resolve_session_path(spec, session_dir.as_deref())?;
            let log = SessionLog::open(&path)?;
            let replayed = peregrine_core::replay(&path)?;
            if !replayed.messages.is_empty() {
                agent.set_messages(replayed.messages);
            }
            eprintln!("resumed session {}", log.id());
            Some(Arc::new(log))
        }
        (None, Some(dir)) => {
            let (log, _header) = SessionLog::create(dir, &cwd)?;
            let log = Arc::new(log);
            for msg in agent.messages() {
                log.append_message(&msg)?;
            }
            Some(log)
        }
        (None, None) => None,
    };

    let drain_log = session_log.clone();
    let drain_json = args.json;
    let drain_interactive = interactive;
    // The drain task acks each AgentEnd so the REPL can wait until all
    // streamed output is flushed before printing its status line.
    let (turn_done_tx, mut turn_done_rx) = mpsc::channel::<()>(8);
    let drain = tokio::spawn(async move {
        drain_events(rx, drain_json, drain_interactive, drain_log, turn_done_tx).await;
    });

    let one_shot = args.print || args.json || args.prompt.is_some();
    if one_shot {
        let prompt = args
            .prompt
            .clone()
            .ok_or("--print/--json needs a prompt")?;
        let before = agent.messages().len();
        let outcome = agent.prompt(prompt).await?;
        log_new_messages(&session_log, &agent, before)?;
        if !args.json {
            print_final_text(&outcome, &agent);
        }
        drop(agent);
        drain.await?;
        check_outcome(&outcome)?;
    } else {
        repl(&mut agent, &session_log, &mut turn_done_rx).await?;
        drop(agent);
        drain.await?;
    }
    Ok(())
}

/// Find a session file by id (`<dir>/*_<id>.jsonl`) or by path.
fn resolve_session_path(
    spec: &str,
    session_dir: Option<&std::path::Path>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let direct = PathBuf::from(spec);
    if direct.exists() {
        return Ok(direct);
    }
    let dir = session_dir.ok_or_else(|| format!("no session directory; cannot find id {spec}"))?;
    let suffix = format!("_{spec}.jsonl");
    let mut matches = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(&suffix) {
            matches.push(entry.path());
        }
    }
    match matches.len() {
        1 => Ok(matches.into_iter().next().unwrap()),
        0 => Err(format!("no session with id {spec}").into()),
        _ => Err(format!("ambiguous session id {spec} ({} matches)", matches.len()).into()),
    }
}

fn log_new_messages(
    log: &Option<Arc<SessionLog>>,
    agent: &Agent,
    before: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(log) = log {
        for msg in agent.messages().iter().skip(before) {
            log.append_message(msg)?;
        }
    }
    Ok(())
}

fn print_final_text(outcome: &AgentOutcome, agent: &Agent) {
    let text = agent
        .messages()
        .iter()
        .rev()
        .find_map(|m| match m {
            peregrine_core::Message::Assistant { .. } => m.text(),
            _ => None,
        })
        .unwrap_or_default();
    if !text.is_empty() {
        println!("{text}");
    }
    if !matches!(outcome.status, RunStatus::Completed) {
        eprintln!("peregrine: run ended with {:?}", outcome.status);
    }
}

fn check_outcome(outcome: &AgentOutcome) -> Result<(), Box<dyn std::error::Error>> {
    match &outcome.status {
        RunStatus::Completed | RunStatus::MaxTurnsExceeded | RunStatus::Denied(_) => Ok(()),
        RunStatus::Error(msg) => Err(format!("run failed: {msg}").into()),
        RunStatus::Aborted => Err("run aborted".into()),
    }
}

/// Drain agent events: NDJSON on stdout in `--json` mode, live rendering
/// in interactive mode, silence (but still logged) in `--print` mode.
async fn drain_events(
    mut rx: mpsc::Receiver<AgentEvent>,
    json: bool,
    interactive: bool,
    log: Option<Arc<SessionLog>>,
    turn_done_tx: mpsc::Sender<()>,
) {
    while let Some(event) = rx.recv().await {
        if let Some(log) = &log {
            let _ = log.append_event(&event);
        }
        let is_end = matches!(event, AgentEvent::AgentEnd { .. });
        if json {
            if let Ok(line) = serde_json::to_string(&event) {
                println!("{line}");
            }
        } else if interactive {
            match event {
                AgentEvent::MessageUpdate {
                    text_delta,
                    thinking,
                } => {
                    if thinking {
                        print!("\u{1b}[2m{text_delta}\u{1b}[0m");
                    } else {
                        print!("{text_delta}");
                    }
                    let _ = std::io::stdout().flush();
                }
                AgentEvent::MessageEnd { .. } => {
                    println!();
                }
                AgentEvent::ToolExecutionStart { name, .. } => {
                    println!("\n[tool: {name}]");
                }
                AgentEvent::ToolExecutionEnd {
                    name, is_error, ..
                } => {
                    println!(
                        "[tool {name} {}]",
                        if is_error { "error" } else { "ok" }
                    );
                }
                _ => {}
            }
        }
        if is_end {
            let _ = turn_done_tx.try_send(());
        }
    }
}

async fn repl(
    agent: &mut Agent,
    log: &Option<Arc<SessionLog>>,
    turn_done_rx: &mut mpsc::Receiver<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Peregrine {} — type /exit to quit.", VERSION);
    let stdin = tokio::io::stdin();
    let mut reader = tokio::io::BufReader::new(stdin);
    let mut line = String::new();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break; // EOF
        }
        let text = line.trim();
        if text == "/exit" || text == "/quit" {
            break;
        }
        if text.is_empty() {
            continue;
        }
        let before = agent.messages().len();
        let outcome = agent.prompt(text.to_string()).await?;
        log_new_messages(log, agent, before)?;
        // Wait until the drain task has flushed every event of this turn
        // (bounded: never block the REPL on the renderer for long).
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            turn_done_rx.recv(),
        )
        .await;
        println!(
            "\n[{} | turns: {} | tokens: {} in / {} out]",
            match &outcome.status {
                RunStatus::Completed => "done".to_string(),
                other => format!("{other:?}"),
            },
            outcome.turns,
            outcome.usage.input_tokens,
            outcome.usage.output_tokens
        );
    }
    Ok(())
}
