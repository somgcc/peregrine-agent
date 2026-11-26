//! Integration tests for the `peregrine` binary. All tests use the mock
//! provider: no network, no key.

use std::process::Command;

fn run_cli(dir: &std::path::Path, args: &[&str]) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_peregrine");
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("HOME", dir.join("home"))
        .output()
        .unwrap()
}

#[test]
fn print_mock_runs_full_turn_with_tool_call() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "hi").unwrap();
    let out = run_cli(
        dir.path(),
        &["--print", "--provider", "mock", "list the files"],
    );
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("hello.txt"),
        "expected the tool result in the final text, got: {stdout}"
    );
}

#[test]
fn json_mode_emits_ndjson_events() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_cli(
        dir.path(),
        &["--json", "--provider", "mock", "hello"],
    );
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut saw_start = false;
    let mut saw_end = false;
    for line in stdout.lines() {
        let value: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|_| panic!("not JSON: {line}"));
        match value.get("type").and_then(|t| t.as_str()) {
            Some("agent_start") => saw_start = true,
            Some("agent_end") => saw_end = true,
            _ => {}
        }
    }
    assert!(saw_start, "no agent_start event in:\n{stdout}");
    assert!(saw_end, "no agent_end event in:\n{stdout}");
}

#[test]
fn unknown_provider_fails() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_cli(dir.path(), &["--print", "--provider", "nope", "hi"]);
    assert!(!out.status.success());
}

#[test]
fn print_without_prompt_fails() {
    let dir = tempfile::tempdir().unwrap();
    let out = run_cli(dir.path(), &["--print", "--provider", "mock"]);
    assert!(!out.status.success());
}
