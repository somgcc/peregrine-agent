//! Transcript: typed messages + system sections.
//!
//! The system prompt and the tool loadout live IN the transcript as replayable
//! state (named sections, tools added/removed), not as call parameters.
//! Resume, fork, compaction and model switching all fall out of this one
//! mechanism (borrowed from Pi's normalized context).

use serde::{Deserialize, Serialize};

/// One named section of the system prompt. Sections are diffed per turn, so
/// only changed sections are re-sent to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemSection {
    pub name: String,
    pub content: String,
}

impl SystemSection {
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
        }
    }
}

/// A tool declaration as the model sees it. The core adapts tool definitions
/// into these; providers serialize them into their own wire format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's arguments object.
    pub parameters: serde_json::Value,
}

/// Content of a user message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContent {
    Text { text: String },
}

/// Content of an assistant message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContent {
    Text { text: String },
    Thinking { text: String },
    ToolCall {
        id: String,
        name: String,
        args: serde_json::Value,
    },
}

impl AssistantContent {
    /// The tool calls carried by this content item, if any.
    pub fn as_tool_call(&self) -> Option<(&str, &str, &serde_json::Value)> {
        match self {
            AssistantContent::ToolCall { id, name, args } => Some((id, name, args)),
            _ => None,
        }
    }
}

/// A transcript message. `System` carries the prompt sections AND the
/// tool-loadout delta (tools added/removed since the previous system
/// message), so replaying the transcript reconstructs both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System {
        sections: Vec<SystemSection>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools_added: Vec<ToolDef>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tools_removed: Vec<String>,
    },
    User {
        content: Vec<UserContent>,
    },
    Assistant {
        content: Vec<AssistantContent>,
    },
    ToolResult {
        call_id: String,
        result: crate::tool::ToolResult,
    },
}

impl Message {
    pub fn user_text(text: impl Into<String>) -> Self {
        Message::User {
            content: vec![UserContent::Text { text: text.into() }],
        }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Message::Assistant {
            content: vec![AssistantContent::Text { text: text.into() }],
        }
    }

    /// All tool calls in this message, in source order.
    pub fn tool_calls(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        match self {
            Message::Assistant { content } => {
                content.iter().filter_map(|c| c.as_tool_call()).collect()
            }
            _ => Vec::new(),
        }
    }

    /// The plain text of this message, if it carries any.
    pub fn text(&self) -> Option<String> {
        match self {
            Message::User { content } => Some(
                content
                    .iter()
                    .map(|c| match c {
                        UserContent::Text { text } => text.as_str(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            Message::Assistant { content } => Some(
                content
                    .iter()
                    .filter_map(|c| match c {
                        AssistantContent::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            ),
            Message::ToolResult { result, .. } => Some(result.content.clone()),
            Message::System { sections, .. } => Some(
                sections
                    .iter()
                    .map(|s| s.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
        }
    }
}

/// The live tool loadout, reconstructed by replaying system messages.
#[derive(Debug, Default, Clone)]
pub struct ToolLoadout {
    pub tools: Vec<ToolDef>,
}

impl ToolLoadout {
    /// Apply one system message's loadout delta.
    pub fn apply(&mut self, msg: &Message) {
        if let Message::System {
            tools_added,
            tools_removed,
            ..
        } = msg
        {
            self.tools.retain(|t| !tools_removed.contains(&t.name));
            for def in tools_added {
                if let Some(existing) = self.tools.iter_mut().find(|t| t.name == def.name) {
                    *existing = def.clone();
                } else {
                    self.tools.push(def.clone());
                }
            }
        }
    }

    /// Reconstruct the loadout by replaying a whole transcript.
    pub fn replay(messages: &[Message]) -> Self {
        let mut loadout = Self::default();
        for msg in messages {
            loadout.apply(msg);
        }
        loadout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(name: &str) -> ToolDef {
        ToolDef {
            name: name.into(),
            description: String::new(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    #[test]
    fn loadout_replay_adds_and_removes() {
        let msgs = vec![
            Message::System {
                sections: vec![SystemSection::new("tools", "x")],
                tools_added: vec![def("read"), def("bash")],
                tools_removed: vec![],
            },
            Message::System {
                sections: vec![],
                tools_added: vec![def("edit")],
                tools_removed: vec!["bash".to_string()],
            },
        ];
        let loadout = ToolLoadout::replay(&msgs);
        let names: Vec<_> = loadout.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["read", "edit"]);
    }

    #[test]
    fn message_roundtrips_through_json() {
        let msg = Message::Assistant {
            content: vec![
                AssistantContent::Text {
                    text: "hi".into(),
                },
                AssistantContent::ToolCall {
                    id: "c1".into(),
                    name: "bash".into(),
                    args: serde_json::json!({"command": "ls"}),
                },
            ],
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, back);
    }
}
