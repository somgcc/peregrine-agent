//! Auth resolution: explicit override → stored credential → env.
//!
//! Stored credentials are file-backed (`~/.config/peregrine/auth.json`,
//! mode `0600`), a JSON map of provider name to API key. Credentials never
//! enter tool execution; the sandbox phase keeps that invariant.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::ProviderKind;

/// Resolve an API key for a provider: explicit override, then the stored
/// credential file, then the provider's env var.
pub fn resolve_api_key(kind: ProviderKind, explicit: Option<String>) -> Option<String> {
    if let Some(key) = explicit.filter(|k| !k.is_empty()) {
        return Some(key);
    }
    let name = match kind {
        ProviderKind::Mock => return None,
        ProviderKind::OpenAiCompatible => "openai-compatible",
        ProviderKind::Anthropic => "anthropic",
    };
    if let Some(stored) = read_stored_key(name) {
        return Some(stored);
    }
    kind
        .api_key_env()
        .and_then(|env| std::env::var(env).ok())
        .filter(|k| !k.is_empty())
}

/// Read one provider's key from the stored-credentials file, if present.
pub fn read_stored_key(provider: &str) -> Option<String> {
    let path = auth_file_path()?;
    let text = std::fs::read_to_string(path).ok()?;
    let map: HashMap<String, String> = serde_json::from_str(&text).ok()?;
    map.get(provider).cloned().filter(|k| !k.is_empty())
}

fn auth_file_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join(".config")
            .join("peregrine")
            .join("auth.json")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_key_wins_over_everything() {
        assert_eq!(
            resolve_api_key(
                ProviderKind::OpenAiCompatible,
                Some("sk-explicit".to_string())
            ),
            Some("sk-explicit".to_string())
        );
    }

    #[test]
    fn mock_has_no_key() {
        assert_eq!(resolve_api_key(ProviderKind::Mock, None), None);
    }
}
