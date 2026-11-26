//! The mock provider: deterministic, LLM-free, no network, no key.
//!
//! Built first — it makes every loop test fast and free. Behaviour is
//! scripted off the transcript:
//!
//! * transcript ends with a `ToolResult` → a final text answer summarizing
//!   the result;
//! * user asks to list files → a `bash {ls -la}` tool call;
//! * user asks to `read <path>` → a `read {path}` tool call;
//! * otherwise → a plain text answer.
//!
//! Reported usage counts whitespace-separated words (a mock-only
//! convention, not a token estimate).

use std::pin::Pin;

use futures::Stream;
use peregrine_core::{LlmClient, LlmEvent, LlmRequest, Message, StopReason, Usage};

pub struct MockClient;

impl Default for MockClient {
    fn default() -> Self {
        Self::new()
    }
}

impl MockClient {
    pub fn new() -> Self {
        Self
    }

    /// Decide what the mock does for this request. Pure and testable.
    fn plan(req: &LlmRequest) -> MockPlan {
        let last = req.messages.last();
        match last {
            Some(Message::ToolResult { result, .. }) => {
                let summary: String = result
                    .content
                    .chars()
                    .take(800)
                    .collect::<String>()
                    .replace('\n', " ");
                MockPlan::Text(format!("Here is what the tool returned: {summary}"))
            }
            Some(Message::User { .. }) => {
                let text = last.map(|m| m.text().unwrap_or_default()).unwrap_or_default();
                let lower = text.to_lowercase();
                if lower.contains("list") || lower.contains("files") {
                    MockPlan::ToolCall {
                        id: "mock-call-1".to_string(),
                        name: "bash".to_string(),
                        args: serde_json::json!({"command": "ls -la"}),
                    }
                } else if let Some(path) = read_path(&lower) {
                    MockPlan::ToolCall {
                        id: "mock-call-1".to_string(),
                        name: "read".to_string(),
                        args: serde_json::json!({"path": path}),
                    }
                } else {
                    MockPlan::Text(format!("Mock response to: {text}"))
                }
            }
            _ => MockPlan::Text("Mock response: nothing to do.".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MockPlan {
    Text(String),
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
}

/// Extract the path after a `read <path>` in the user's text.
fn read_path(lower: &str) -> Option<String> {
    let rest = lower.split_once("read ")?.1.trim();
    let path = rest.split_whitespace().next()?;
    if path.is_empty() {
        None
    } else {
        Some(path.to_string())
    }
}

fn word_count(text: &str) -> u64 {
    text.split_whitespace().count() as u64
}

impl LlmClient for MockClient {
    fn stream(&self, req: LlmRequest) -> Pin<Box<dyn Stream<Item = LlmEvent> + Send + '_>> {
        let plan = Self::plan(&req);
        let input_tokens: u64 = req
            .messages
            .iter()
            .filter_map(|m| m.text())
            .map(|t| word_count(&t))
            .sum();
        let events: Vec<LlmEvent> = match plan {
            MockPlan::Text(text) => {
                let output_tokens = word_count(&text);
                let mut events = Vec::new();
                // Stream in small chunks so the loop's streaming path is
                // exercised, with no delay (the mock is instant).
                for chunk in text.as_bytes().chunks(16) {
                    events.push(LlmEvent::TextDelta {
                        text: String::from_utf8_lossy(chunk).into_owned(),
                    });
                }
                events.push(LlmEvent::Done {
                    usage: Usage {
                        input_tokens,
                        output_tokens,
                    },
                    stop_reason: StopReason::Stop,
                });
                events
            }
            MockPlan::ToolCall { id, name, args } => {
                let args_json = serde_json::to_string(&args).unwrap_or_else(|_| "{}".into());
                let (first, second) = args_json.split_at(args_json.len() / 2);
                vec![
                    LlmEvent::ToolCallDelta {
                        id: id.clone(),
                        name: Some(name),
                        args_delta: first.to_string(),
                    },
                    LlmEvent::ToolCallDelta {
                        id,
                        name: None,
                        args_delta: second.to_string(),
                    },
                    LlmEvent::Done {
                        usage: Usage {
                            input_tokens,
                            output_tokens: 8,
                        },
                        stop_reason: StopReason::ToolUse,
                    },
                ]
            }
        };
        Box::pin(futures::stream::iter(events))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use peregrine_core::{SystemSection, ToolDef};

    fn req_with_user(text: &str) -> LlmRequest {
        LlmRequest::new(
            "mock-1",
            vec![
                Message::System {
                    sections: vec![SystemSection::new("rules", "Be concise.")],
                    tools_added: vec![],
                    tools_removed: vec![],
                },
                Message::user_text(text),
            ],
            vec![ToolDef {
                name: "bash".into(),
                description: String::new(),
                parameters: serde_json::json!({"type": "object"}),
            }],
        )
    }

    #[tokio::test]
    async fn list_files_yields_bash_tool_call() {
        let client = MockClient::new();
        let events: Vec<_> = client
            .stream(req_with_user("list the files"))
            .collect()
            .await;
        assert!(matches!(
            events.last(),
            Some(LlmEvent::Done {
                stop_reason: StopReason::ToolUse,
                ..
            })
        ));
        let args: String = events
            .iter()
            .filter_map(|e| match e {
                LlmEvent::ToolCallDelta { args_delta, .. } => Some(args_delta.clone()),
                _ => None,
            })
            .collect();
        assert!(args.contains("ls -la"));
    }

    #[tokio::test]
    async fn tool_result_yields_final_text() {
        let mut req = req_with_user("list the files");
        req.messages.push(Message::ToolResult {
            call_id: "mock-call-1".into(),
            result: peregrine_core::ToolResult::ok("a.txt\nb.txt"),
        });
        let client = MockClient::new();
        let events: Vec<_> = client.stream(req).collect().await;
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                LlmEvent::TextDelta { text } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(text.contains("a.txt"));
        assert!(matches!(
            events.last(),
            Some(LlmEvent::Done {
                stop_reason: StopReason::Stop,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn plain_prompt_yields_text() {
        let client = MockClient::new();
        let events: Vec<_> = client.stream(req_with_user("hello")).collect().await;
        assert!(matches!(
            events.last(),
            Some(LlmEvent::Done {
                stop_reason: StopReason::Stop,
                ..
            })
        ));
    }
}
