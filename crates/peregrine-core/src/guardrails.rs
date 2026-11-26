//! Guardrails: enforced outside the agent (borrowed from Orca engine).
//!
//! Pure evaluator, monotonic verdict lattice `allow < ask < deny` — a rule
//! can only tighten, never loosen. Budgets fail closed: when a budget is
//! exceeded, or usage is unknown, the answer is `deny`, never "proceed and
//! hope". `ask` in non-interactive mode is a deny.

use std::collections::VecDeque;
use std::time::Duration;

/// Verdict lattice. Declaration order is the lattice order:
/// `Allow < Ask < Deny`, so `Ord` composes verdicts by `max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Verdict {
    Allow,
    Ask,
    Deny,
}

impl Verdict {
    /// Compose two verdicts monotonically: the tighter one wins.
    pub fn compose(self, other: Verdict) -> Verdict {
        self.max(other)
    }
}

/// Budgets for one run. All fail closed.
#[derive(Debug, Clone)]
pub struct Budgets {
    /// Maximum assistant turns per run.
    pub max_turns: usize,
    /// Maximum tool calls per session (Orca default: 100).
    pub max_tool_calls_per_session: usize,
    /// Maximum wall-clock time for one turn (Orca default: 600 s).
    pub turn_timeout: Duration,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            max_turns: 50,
            max_tool_calls_per_session: 100,
            turn_timeout: Duration::from_secs(600),
        }
    }
}

/// What is being evaluated.
#[derive(Debug, Clone)]
pub enum GuardrailAction<'a> {
    /// A turn is about to start.
    TurnStart { turn: usize },
    /// A tool call is about to execute.
    ToolCall { name: &'a str, args: &'a serde_json::Value },
}

/// Mutable state the evaluator carries across a run. Stateful budgets are
/// persisted through the session log; a run that loses its counters refuses
/// rather than proceeding (fail closed).
#[derive(Debug)]
pub struct Guardrails {
    pub budgets: Budgets,
    pub interactive: bool,
    turns_used: usize,
    tool_calls_used: usize,
    /// Recent tool calls (name, args) for loop detection, newest last.
    recent_calls: VecDeque<(String, serde_json::Value)>,
}

/// Loop detection (Orca `detect_loop`): an identical tool+args call
/// repeating `threshold` times inside a window of `window` calls is a loop.
pub const LOOP_WINDOW: usize = 10;
pub const LOOP_THRESHOLD: usize = 3;

impl Guardrails {
    pub fn new(budgets: Budgets, interactive: bool) -> Self {
        Self {
            budgets,
            interactive,
            turns_used: 0,
            tool_calls_used: 0,
            recent_calls: VecDeque::new(),
        }
    }

    /// Evaluate one action. Pure apart from the counters it advances:
    /// a `ToolCall` that is allowed is recorded (callers must evaluate
    /// before executing, exactly once per call).
    ///
    /// The turn timeout (`budgets.turn_timeout`) is a wall-clock bound
    /// enforced by the agent loop with `tokio::time::timeout` around the
    /// stream; it is not evaluated here.
    pub fn evaluate(&mut self, action: GuardrailAction<'_>) -> Verdict {
        match action {
            GuardrailAction::TurnStart { turn } => {
                self.turns_used = self.turns_used.max(turn);
                if turn > self.budgets.max_turns {
                    Verdict::Deny
                } else {
                    Verdict::Allow
                }
            }
            GuardrailAction::ToolCall { name, args } => {
                let mut verdict = Verdict::Allow;
                if self.tool_calls_used >= self.budgets.max_tool_calls_per_session {
                    verdict = verdict.compose(Verdict::Deny);
                }
                if detect_loop(&self.recent_calls, name, args) {
                    verdict = verdict.compose(Verdict::Deny);
                }
                if verdict == Verdict::Allow {
                    self.tool_calls_used += 1;
                    self.recent_calls.push_back((name.to_string(), args.clone()));
                    while self.recent_calls.len() > LOOP_WINDOW {
                        self.recent_calls.pop_front();
                    }
                }
                verdict
            }
        }
    }

    /// Resolve `ask` for the current mode: interactive mode may ask the
    /// user (the caller prompts); non-interactive mode denies.
    pub fn resolve_ask(&self, verdict: Verdict) -> Verdict {
        if verdict == Verdict::Ask && !self.interactive {
            Verdict::Deny
        } else {
            verdict
        }
    }

    pub fn turns_used(&self) -> usize {
        self.turns_used
    }

    pub fn tool_calls_used(&self) -> usize {
        self.tool_calls_used
    }
}

/// True when executing this call would complete a loop: the identical
/// (name, args) call already appears `LOOP_THRESHOLD - 1` times among the
/// last `LOOP_WINDOW - 1` calls, so this one makes `LOOP_THRESHOLD`
/// identical calls inside a window of `LOOP_WINDOW`.
pub fn detect_loop(
    recent: &VecDeque<(String, serde_json::Value)>,
    name: &str,
    args: &serde_json::Value,
) -> bool {
    let window_start = recent.len().saturating_sub(LOOP_WINDOW - 1);
    let count = recent
        .iter()
        .skip(window_start)
        .filter(|(n, a)| n == name && a == args)
        .count();
    count + 1 >= LOOP_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn verdict_lattice_is_monotonic() {
        assert!(Verdict::Allow < Verdict::Ask);
        assert!(Verdict::Ask < Verdict::Deny);
        assert_eq!(Verdict::Allow.compose(Verdict::Deny), Verdict::Deny);
        assert_eq!(Verdict::Ask.compose(Verdict::Allow), Verdict::Ask);
    }

    #[test]
    fn tool_call_budget_fails_closed_at_100() {
        let mut g = Guardrails::new(Budgets::default(), false);
        let args = json!({"command": "ls"});
        for _ in 0..100 {
            // Vary args so loop detection does not fire first.
            let args = json!({"command": format!("echo {}", g.tool_calls_used())});
            assert_eq!(
                g.evaluate(GuardrailAction::ToolCall {
                    name: "bash",
                    args: &args
                }),
                Verdict::Allow
            );
        }
        assert_eq!(
            g.evaluate(GuardrailAction::ToolCall {
                name: "bash",
                args: &args
            }),
            Verdict::Deny
        );
        assert_eq!(g.tool_calls_used(), 100);
    }

    #[test]
    fn loop_detection_denies_third_identical_call_in_window() {
        let mut g = Guardrails::new(Budgets::default(), false);
        let args = json!({"command": "ls"});
        assert_eq!(
            g.evaluate(GuardrailAction::ToolCall {
                name: "bash",
                args: &args
            }),
            Verdict::Allow
        );
        assert_eq!(
            g.evaluate(GuardrailAction::ToolCall {
                name: "bash",
                args: &args
            }),
            Verdict::Allow
        );
        assert_eq!(
            g.evaluate(GuardrailAction::ToolCall {
                name: "bash",
                args: &args
            }),
            Verdict::Deny
        );
    }

    #[test]
    fn loop_detection_allows_different_args() {
        let mut g = Guardrails::new(Budgets::default(), false);
        for i in 0..5 {
            let args = json!({"command": format!("echo {i}")});
            assert_eq!(
                g.evaluate(GuardrailAction::ToolCall {
                    name: "bash",
                    args: &args
                }),
                Verdict::Allow
            );
        }
    }

    #[test]
    fn max_turns_denies() {
        let mut g = Guardrails::new(
            Budgets {
                max_turns: 2,
                ..Default::default()
            },
            false,
        );
        assert_eq!(
            g.evaluate(GuardrailAction::TurnStart { turn: 1 }),
            Verdict::Allow
        );
        assert_eq!(
            g.evaluate(GuardrailAction::TurnStart { turn: 2 }),
            Verdict::Allow
        );
        assert_eq!(
            g.evaluate(GuardrailAction::TurnStart { turn: 3 }),
            Verdict::Deny
        );
    }

    #[test]
    fn ask_is_deny_when_not_interactive() {
        let g = Guardrails::new(Budgets::default(), false);
        assert_eq!(g.resolve_ask(Verdict::Ask), Verdict::Deny);
        let g = Guardrails::new(Budgets::default(), true);
        assert_eq!(g.resolve_ask(Verdict::Ask), Verdict::Ask);
    }
}
