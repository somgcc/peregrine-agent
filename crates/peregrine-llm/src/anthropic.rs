//! Anthropic provider (Messages API, SSE).
//!
//! Speaks the Anthropic Messages API with streaming. Key from env
//! (`ANTHROPIC_API_KEY`), stored credentials, or explicit options.

use std::collections::HashMap;
use std::pin::Pin;

use futures::Stream;
use peregrine_core::{LlmClient, LlmEvent, LlmRequest, Message, StopReason, Usage};
use reqwest::Client;

use crate::sse::{SseEvent, SseParser};

pub struct AnthropicClient {
    base_url: String,
    api_key: String,
    model: String,
    http: Client,
}

impl AnthropicClient {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            http: Client::new(),
        }
    }
}

/// Convert the transcript to Anthropic messages. System sections become the
/// separate `system` parameter; tool results become `tool_result` blocks in
/// user messages. Adjacent same-role messages are merged (Anthropic
/// requires alternating roles).
pub fn to_anthropic_parts(messages: &[Message]) -> (String, Vec<serde_json::Value>) {
    let mut system_parts: Vec<String> = Vec::new();
    let mut out: Vec<serde_json::Value> = Vec::new();
    let push = |role: &str, block: serde_json::Value, out: &mut Vec<serde_json::Value>| {
        if let Some(last) = out.last_mut() {
            if last["role"] == role {
                last["content"]
                    .as_array_mut()
                    .expect("anthropic content is an array")
                    .push(block);
                return;
            }
        }
        out.push(serde_json::json!({"role": role, "content": [block]}));
    };

    for msg in messages {
        match msg {
            Message::System { sections, .. } => {
                for s in sections {
                    system_parts.push(format!("{}:\n{}", s.name, s.content));
                }
            }
            Message::User { .. } => {
                let text = msg.text().unwrap_or_default();
                if !text.is_empty() {
                    push("user", serde_json::json!({"type": "text", "text": text}), &mut out);
                }
            }
            Message::Assistant { content } => {
                for c in content {
                    match c {
                        peregrine_core::AssistantContent::Text { text } => {
                            if !text.is_empty() {
                                push(
                                    "assistant",
                                    serde_json::json!({"type": "text", "text": text}),
                                    &mut out,
                                );
                            }
                        }
                        peregrine_core::AssistantContent::Thinking { text } => {
                            push(
                                "assistant",
                                serde_json::json!({"type": "thinking", "thinking": text}),
                                &mut out,
                            );
                        }
                        peregrine_core::AssistantContent::ToolCall { id, name, args } => {
                            push(
                                "assistant",
                                serde_json::json!({
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": args,
                                }),
                                &mut out,
                            );
                        }
                    }
                }
            }
            Message::ToolResult { call_id, result } => {
                push(
                    "user",
                    serde_json::json!({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": result.content,
                        "is_error": result.is_error,
                    }),
                    &mut out,
                );
            }
        }
    }
    (system_parts.join("\n\n"), out)
}

/// Stateful interpretation of one Anthropic SSE event. Pure apart from the
/// index→tool-call-id map it maintains.
#[derive(Debug, Default)]
pub struct AnthropicInterpreter {
    tool_ids: HashMap<usize, String>,
    usage: Usage,
    stop_reason: Option<String>,
    had_tool_calls: bool,
}

impl AnthropicInterpreter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Interpret one event into `LlmEvent`s (usually 0 or 1).
    pub fn interpret(&mut self, event: &SseEvent) -> Vec<LlmEvent> {
        let mut out = Vec::new();
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&event.data) else {
            return out;
        };
        match event.event.as_deref() {
            Some("message_start") => {
                if let Some(input) = value
                    .get("message")
                    .and_then(|m| m.get("usage"))
                    .and_then(|u| u.get("input_tokens"))
                    .and_then(|v| v.as_u64())
                {
                    self.usage.input_tokens = input;
                }
            }
            Some("content_block_start") => {
                let index = value.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                if let Some(block) = value.get("content_block") {
                    if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        let id = block
                            .get("id")
                            .and_then(|i| i.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("")
                            .to_string();
                        if !id.is_empty() {
                            self.tool_ids.insert(index, id.clone());
                        }
                        self.had_tool_calls = true;
                        out.push(LlmEvent::ToolCallDelta {
                            id,
                            name: Some(name),
                            args_delta: String::new(),
                        });
                    }
                }
            }
            Some("content_block_delta") => {
                let index = value.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                let Some(delta) = value.get("delta") else {
                    return out;
                };
                match delta.get("type").and_then(|t| t.as_str()) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                out.push(LlmEvent::TextDelta {
                                    text: text.to_string(),
                                });
                            }
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial) = delta.get("partial_json").and_then(|p| p.as_str())
                        {
                            let id = self
                                .tool_ids
                                .get(&index)
                                .cloned()
                                .unwrap_or_else(|| format!("call_{index}"));
                            out.push(LlmEvent::ToolCallDelta {
                                id,
                                name: None,
                                args_delta: partial.to_string(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                if let Some(reason) = value
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|r| r.as_str())
                {
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(output) = value
                    .get("usage")
                    .and_then(|u| u.get("output_tokens"))
                    .and_then(|v| v.as_u64())
                {
                    self.usage.output_tokens = output;
                }
            }
            _ => {}
        }
        out
    }

    pub fn finish(&self) -> LlmEvent {
        let stop_reason = match self.stop_reason.as_deref() {
            Some("end_turn") => StopReason::Stop,
            Some("max_tokens") => StopReason::Length,
            Some("tool_use") => StopReason::ToolUse,
            _ => {
                if self.had_tool_calls {
                    StopReason::ToolUse
                } else {
                    StopReason::Stop
                }
            }
        };
        LlmEvent::Done {
            usage: self.usage,
            stop_reason,
        }
    }
}

impl LlmClient for AnthropicClient {
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
            let (system, messages) = to_anthropic_parts(&req.messages);
            let body = serde_json::json!({
                "model": model,
                "max_tokens": 4096,
                "system": system,
                "messages": messages,
                "tools": req.tools.iter().map(|t| serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })).collect::<Vec<_>>(),
                "stream": true,
            });
            let resp = match http
                .post(format!("{base_url}/v1/messages"))
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
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
            let mut interp = AnthropicInterpreter::new();
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
                    for out in interp.interpret(&event) {
                        yield out;
                    }
                }
            }
            if let Some(trailing) = parser.finish() {
                for out in interp.interpret(&trailing) {
                    yield out;
                }
            }
            yield interp.finish();
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peregrine_core::{AssistantContent, SystemSection};

    fn sse(event: &str, data: serde_json::Value) -> SseEvent {
        SseEvent {
            event: Some(event.to_string()),
            data: serde_json::to_string(&data).unwrap(),
        }
    }

    #[test]
    fn converts_transcript_with_tool_result_merge() {
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
        let (system, out) = to_anthropic_parts(&msgs);
        assert!(system.contains("Be concise."));
        // user + tool_result merge into one user message (alternating roles).
        assert_eq!(out.len(), 3);
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[1]["content"][0]["type"], "tool_use");
        assert_eq!(out[2]["role"], "user");
        assert_eq!(out[2]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn interprets_tool_use_stream() {
        let mut interp = AnthropicInterpreter::new();
        let events = interp.interpret(&sse(
            "content_block_start",
            serde_json::json!({
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "bash"}
            }),
        ));
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            LlmEvent::ToolCallDelta { id, name, .. } if id == "toolu_1" && name.as_deref() == Some("bash")
        ));
        let events = interp.interpret(&sse(
            "content_block_delta",
            serde_json::json!({
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"ls\"}"}
            }),
        ));
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            LlmEvent::ToolCallDelta { id, args_delta, .. } if id == "toolu_1" && args_delta.contains("ls")
        ));
        interp.interpret(&sse(
            "message_delta",
            serde_json::json!({"delta": {"stop_reason": "tool_use"}}),
        ));
        assert!(matches!(
            interp.finish(),
            LlmEvent::Done { stop_reason: StopReason::ToolUse, .. }
        ));
    }

    #[test]
    fn interprets_text_and_usage() {
        let mut interp = AnthropicInterpreter::new();
        interp.interpret(&sse(
            "message_start",
            serde_json::json!({"message": {"usage": {"input_tokens": 12}}}),
        ));
        let events = interp.interpret(&sse(
            "content_block_delta",
            serde_json::json!({
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"}
            }),
        ));
        assert_eq!(events.len(), 1);
        interp.interpret(&sse(
            "message_delta",
            serde_json::json!({"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}),
        ));
        match interp.finish() {
            LlmEvent::Done { usage, stop_reason } => {
                assert_eq!(usage.input_tokens, 12);
                assert_eq!(usage.output_tokens, 4);
                assert_eq!(stop_reason, StopReason::Stop);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
