//! OpenAI-compatible provider (`/chat/completions`, SSE).
//!
//! Speaks to OpenAI, Ollama, or any compatible endpoint. Base URL and key
//! come from env (`OPENAI_BASE_URL` / `OPENAI_API_KEY`), stored
//! credentials, or explicit options — never from tool execution.

use std::collections::HashMap;
use std::pin::Pin;

use futures::Stream;
use peregrine_core::{LlmClient, LlmEvent, LlmRequest, Message, StopReason, Usage};
use reqwest::Client;

use crate::sse::SseParser;

pub struct OpenAiClient {
    base_url: String,
    api_key: String,
    model: String,
    http: Client,
}

impl OpenAiClient {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            http: Client::new(),
        }
    }
}

/// Convert the transcript to OpenAI chat messages.
pub fn to_openai_messages(messages: &[Message]) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    for msg in messages {
        match msg {
            Message::System { sections, .. } => {
                let text = sections
                    .iter()
                    .map(|s| format!("{}:\n{}", s.name, s.content))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                if !text.is_empty() {
                    out.push(serde_json::json!({"role": "system", "content": text}));
                }
            }
            Message::User { .. } => {
                let text = msg.text().unwrap_or_default();
                out.push(serde_json::json!({"role": "user", "content": text}));
            }
            Message::Assistant { content } => {
                let text: String = content
                    .iter()
                    .filter_map(|c| match c {
                        peregrine_core::AssistantContent::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                let tool_calls: Vec<serde_json::Value> = content
                    .iter()
                    .filter_map(|c| match c {
                        peregrine_core::AssistantContent::ToolCall { id, name, args } => {
                            Some(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": serde_json::to_string(args).unwrap_or_default(),
                                },
                            }))
                        }
                        _ => None,
                    })
                    .collect();
                let mut m = serde_json::json!({"role": "assistant"});
                if !text.is_empty() {
                    m["content"] = serde_json::Value::String(text);
                } else if tool_calls.is_empty() {
                    m["content"] = serde_json::Value::Null;
                }
                if !tool_calls.is_empty() {
                    m["tool_calls"] = serde_json::Value::Array(tool_calls);
                }
                out.push(m);
            }
            Message::ToolResult { call_id, result } => {
                out.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result.content,
                }));
            }
        }
    }
    out
}

/// The result of interpreting one streamed chunk object.
#[derive(Debug, Default)]
struct ChunkEvents {
    text: Option<String>,
    tool_deltas: Vec<ToolDelta>,
    finish_reason: Option<String>,
    usage: Option<Usage>,
}

#[derive(Debug)]
struct ToolDelta {
    index: usize,
    id: Option<String>,
    name: Option<String>,
    args_delta: String,
}

/// Interpret one OpenAI SSE payload (a parsed JSON object) into events.
/// Pure — unit-testable without a network.
fn interpret_chunk(
    value: &serde_json::Value,
    tool_ids: &mut HashMap<usize, String>,
) -> ChunkEvents {
    let mut out = ChunkEvents::default();
    if let Some(usage) = value.get("usage") {
        out.usage = Some(Usage {
            input_tokens: usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        });
    }
    let Some(choice) = value.get("choices").and_then(|c| c.as_array()).and_then(|a| a.first())
    else {
        return out;
    };
    if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
        out.finish_reason = Some(reason.to_string());
    }
    let Some(delta) = choice.get("delta") else {
        return out;
    };
    if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
        if !text.is_empty() {
            out.text = Some(text.to_string());
        }
    }
    if let Some(calls) = delta.get("tool_calls").and_then(|c| c.as_array()) {
        for call in calls {
            let index = call.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
            let id = call.get("id").and_then(|i| i.as_str()).map(str::to_string);
            if let Some(id) = &id {
                tool_ids.insert(index, id.clone());
            }
            let function = call.get("function");
            out.tool_deltas.push(ToolDelta {
                index,
                id: tool_ids.get(&index).cloned().or(id),
                name: function
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .map(str::to_string),
                args_delta: function
                    .and_then(|f| f.get("arguments"))
                    .and_then(|a| a.as_str())
                    .unwrap_or("")
                    .to_string(),
            });
        }
    }
    out
}

fn finish_to_stop(reason: &str, had_tool_calls: bool) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "length" => StopReason::Length,
        "tool_calls" => StopReason::ToolUse,
        _ => {
            if had_tool_calls {
                StopReason::ToolUse
            } else {
                StopReason::Stop
            }
        }
    }
}

impl LlmClient for OpenAiClient {
    fn stream(&self, req: LlmRequest) -> Pin<Box<dyn Stream<Item = LlmEvent> + Send + '_>> {
        let base_url = self.base_url.clone();
        let api_key = self.api_key.clone();
        let model = if req.model.is_empty() {
            self.model.clone()
        } else {
            req.model.clone()
        };
        let http = self.http.clone();
        Box::pin(async_stream::stream! {
            let body = serde_json::json!({
                "model": model,
                "messages": to_openai_messages(&req.messages),
                "tools": req.tools.iter().map(|t| serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                })).collect::<Vec<_>>(),
                "stream": true,
                "stream_options": {"include_usage": true},
            });
            let resp = match http
                .post(format!("{base_url}/chat/completions"))
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield LlmEvent::Error { message: format!("request failed: {e}") };
                    return;
                }
            };
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                yield LlmEvent::Error {
                    message: format!("provider returned {status}: {text}"),
                };
                return;
            }
            let mut parser = SseParser::new();
            let mut tool_ids: HashMap<usize, String> = HashMap::new();
            let mut usage = Usage::default();
            let mut finish_reason: Option<String> = None;
            let mut had_tool_calls = false;
            let mut byte_stream = resp.bytes_stream();
            use futures::StreamExt;
            while let Some(chunk) = byte_stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        yield LlmEvent::Error { message: format!("stream failed: {e}") };
                        return;
                    }
                };
                for event in parser.feed(&bytes) {
                    if event.data.trim() == "[DONE]" {
                        continue;
                    }
                    let Ok(value) = serde_json::from_str::<serde_json::Value>(&event.data) else {
                        continue;
                    };
                    let parsed = interpret_chunk(&value, &mut tool_ids);
                    if let Some(u) = parsed.usage {
                        usage = u;
                    }
                    if let Some(reason) = parsed.finish_reason {
                        finish_reason = Some(reason);
                    }
                    if let Some(text) = parsed.text {
                        yield LlmEvent::TextDelta { text };
                    }
                    for delta in parsed.tool_deltas {
                        had_tool_calls = true;
                        yield LlmEvent::ToolCallDelta {
                            id: delta.id.unwrap_or_else(|| format!("call_{}", delta.index)),
                            name: delta.name,
                            args_delta: delta.args_delta,
                        };
                    }
                }
            }
            if let Some(trailing) = parser.finish() {
                if trailing.data.trim() != "[DONE]" {
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&trailing.data) {
                        let parsed = interpret_chunk(&value, &mut tool_ids);
                        if let Some(u) = parsed.usage {
                            usage = u;
                        }
                        if let Some(reason) = parsed.finish_reason {
                            finish_reason = Some(reason);
                        }
                        if let Some(text) = parsed.text {
                            yield LlmEvent::TextDelta { text };
                        }
                        for delta in parsed.tool_deltas {
                            had_tool_calls = true;
                            yield LlmEvent::ToolCallDelta {
                                id: delta.id.unwrap_or_else(|| format!("call_{}", delta.index)),
                                name: delta.name,
                                args_delta: delta.args_delta,
                            };
                        }
                    }
                }
            }
            yield LlmEvent::Done {
                usage,
                stop_reason: finish_to_stop(
                    finish_reason.as_deref().unwrap_or(""),
                    had_tool_calls,
                ),
            };
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peregrine_core::{AssistantContent, SystemSection};

    #[test]
    fn converts_transcript_to_openai_messages() {
        let msgs = vec![
            Message::System {
                sections: vec![SystemSection::new("rules", "Be concise.")],
                tools_added: vec![],
                tools_removed: vec![],
            },
            Message::user_text("list files"),
            Message::Assistant {
                content: vec![AssistantContent::ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    args: serde_json::json!({"command": "ls"}),
                }],
            },
            Message::ToolResult {
                call_id: "c1".into(),
                result: peregrine_core::ToolResult::ok("a.txt"),
            },
        ];
        let out = to_openai_messages(&msgs);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0]["role"], "system");
        assert!(out[0]["content"].as_str().unwrap().contains("Be concise."));
        assert_eq!(out[1]["role"], "user");
        assert_eq!(out[2]["role"], "assistant");
        assert_eq!(out[2]["tool_calls"][0]["function"]["name"], "bash");
        assert_eq!(out[2]["tool_calls"][0]["function"]["arguments"], "{\"command\":\"ls\"}");
        assert_eq!(out[3]["role"], "tool");
        assert_eq!(out[3]["tool_call_id"], "c1");
    }

    #[test]
    fn interprets_text_and_tool_deltas() {
        let mut ids = HashMap::new();
        let value = serde_json::json!({
            "choices": [{
                "delta": {
                    "content": "hi",
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "bash", "arguments": "{\"command\":"}
                    }]
                },
                "finish_reason": null
            }]
        });
        let parsed = interpret_chunk(&value, &mut ids);
        assert_eq!(parsed.text.as_deref(), Some("hi"));
        assert_eq!(parsed.tool_deltas.len(), 1);
        assert_eq!(parsed.tool_deltas[0].id.as_deref(), Some("call_1"));
        assert_eq!(parsed.tool_deltas[0].name.as_deref(), Some("bash"));

        // A later chunk for the same index reuses the id.
        let value = serde_json::json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": " \"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let parsed = interpret_chunk(&value, &mut ids);
        assert_eq!(parsed.tool_deltas[0].id.as_deref(), Some("call_1"));
        assert_eq!(parsed.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(
            finish_to_stop("tool_calls", false),
            StopReason::ToolUse
        );
        assert_eq!(finish_to_stop("length", false), StopReason::Length);
        assert_eq!(finish_to_stop("", true), StopReason::ToolUse);
        assert_eq!(finish_to_stop("", false), StopReason::Stop);
    }
}
