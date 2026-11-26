//! Minimal SSE (Server-Sent Events) parsing shared by the streaming
//! providers. Events are separated by blank lines; an event may carry
//! several `data:` lines (joined with `\n`) and an `event:` line.

/// One parsed SSE event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, if present.
    pub event: Option<String>,
    /// The joined `data:` lines, without the `data: ` prefix.
    pub data: String,
}

/// Incremental SSE parser: feed raw bytes, get complete events.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: String,
    current_data: Vec<String>,
    current_event: Option<String>,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; returns all events completed by this chunk.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buf.push_str(&String::from_utf8_lossy(bytes));
        let mut events = Vec::new();
        while let Some(pos) = self.buf.find('\n') {
            let line = self.buf[..pos].trim_end_matches('\r').to_string();
            self.buf.drain(..=pos);
            if line.is_empty() {
                if !self.current_data.is_empty() {
                    events.push(SseEvent {
                        event: self.current_event.take(),
                        data: std::mem::take(&mut self.current_data).join("\n"),
                    });
                } else {
                    self.current_event = None;
                }
                continue;
            }
            if let Some(data) = line.strip_prefix("data:") {
                self.current_data.push(data.strip_prefix(' ').unwrap_or(data).to_string());
            } else if let Some(event) = line.strip_prefix("event:") {
                self.current_event = Some(event.trim().to_string());
            }
            // Other fields (`id:`, `retry:`) are ignored.
        }
        events
    }

    /// Flush a trailing unterminated event, if any.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if self.current_data.is_empty() {
            None
        } else {
            Some(SseEvent {
                event: self.current_event.take(),
                data: std::mem::take(&mut self.current_data).join("\n"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_events_across_chunk_boundaries() {
        let mut p = SseParser::new();
        let out = p.feed(b"data: {\"a\": 1}\n\ndata: {\"b\": ");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data, "{\"a\": 1}");
        assert_eq!(out[0].event, None);
        let out = p.feed(b"2}\n\n");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data, "{\"b\": 2}");
    }

    #[test]
    fn keeps_event_type_and_joins_multi_line_data() {
        let mut p = SseParser::new();
        let out = p.feed(b"event: message_delta\ndata: line1\ndata: line2\n\n");
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].event.as_deref(), Some("message_delta"));
        assert_eq!(out[0].data, "line1\nline2");
    }

    #[test]
    fn finish_flushes_trailing_event() {
        let mut p = SseParser::new();
        p.feed(b"data: {\"x\": 1}\n");
        assert_eq!(p.finish().map(|e| e.data).as_deref(), Some("{\"x\": 1}"));
        assert_eq!(p.finish(), None);
    }
}
