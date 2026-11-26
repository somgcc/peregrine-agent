//! Provider abstraction: one `LlmClient` trait, providers are data
//! (base URL + auth scheme + wire format), not code.
//!
//! v0.1 ships three providers: `mock` (deterministic, LLM-free — build it
//! first; it makes every test fast and free), `openai-compatible`
//! (OpenAI, Ollama, any compatible endpoint), and `anthropic` (Messages
//! API). Auth resolution: explicit override → stored credential → env.

pub mod anthropic;
pub mod auth;
pub mod mock;
pub mod openai;
pub mod sse;

use std::sync::Arc;

use peregrine_core::LlmClient;

/// Which provider to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Mock,
    OpenAiCompatible,
    Anthropic,
}

impl ProviderKind {
    pub fn parse(s: &str) -> Result<Self, LlmError> {
        match s {
            "mock" => Ok(ProviderKind::Mock),
            "openai-compatible" | "openai" => Ok(ProviderKind::OpenAiCompatible),
            "anthropic" => Ok(ProviderKind::Anthropic),
            other => Err(LlmError::UnknownProvider(other.to_string())),
        }
    }

    /// Default model for the provider.
    pub fn default_model(self) -> &'static str {
        match self {
            ProviderKind::Mock => "mock-1",
            ProviderKind::OpenAiCompatible => "gpt-4o-mini",
            ProviderKind::Anthropic => "claude-sonnet-4-5-20250929",
        }
    }

    /// Default base URL (mock has none).
    pub fn default_base_url(self) -> Option<&'static str> {
        match self {
            ProviderKind::Mock => None,
            ProviderKind::OpenAiCompatible => Some("https://api.openai.com/v1"),
            ProviderKind::Anthropic => Some("https://api.anthropic.com"),
        }
    }

    /// Env var carrying this provider's API key.
    pub fn api_key_env(self) -> Option<&'static str> {
        match self {
            ProviderKind::Mock => None,
            ProviderKind::OpenAiCompatible => Some("OPENAI_API_KEY"),
            ProviderKind::Anthropic => Some("ANTHROPIC_API_KEY"),
        }
    }
}

/// Options for building a client. `base_url` / `api_key` override env and
/// stored credentials; `model` overrides the provider default.
#[derive(Debug, Clone, Default)]
pub struct ClientOptions {
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
}

#[derive(Debug)]
pub enum LlmError {
    UnknownProvider(String),
    MissingApiKey(String),
    InvalidConfig(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::UnknownProvider(p) => write!(f, "unknown provider: {p}"),
            LlmError::MissingApiKey(p) => write!(
                f,
                "no API key for provider {p}: set the env var or add it to ~/.config/peregrine/auth.json"
            ),
            LlmError::InvalidConfig(m) => write!(f, "invalid provider config: {m}"),
        }
    }
}

impl std::error::Error for LlmError {}

/// Build a provider client. The mock needs no key; the network providers
/// resolve their key at build time (explicit → stored → env) and fail here
/// — not mid-run — when no key exists.
pub fn create_client(
    kind: ProviderKind,
    opts: ClientOptions,
) -> Result<Arc<dyn LlmClient>, LlmError> {
    match kind {
        ProviderKind::Mock => Ok(Arc::new(mock::MockClient::new())),
        ProviderKind::OpenAiCompatible => {
            let base_url = opts
                .base_url
                .clone()
                .or_else(|| {
                    std::env::var("OPENAI_BASE_URL")
                        .ok()
                        .or_else(|| kind.default_base_url().map(str::to_string))
                })
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            let api_key = auth::resolve_api_key(kind, opts.api_key.clone());
            let api_key = api_key.ok_or_else(|| {
                LlmError::MissingApiKey("openai-compatible".to_string())
            })?;
            Ok(Arc::new(openai::OpenAiClient::new(
                base_url,
                api_key,
                opts.model
                    .unwrap_or_else(|| kind.default_model().to_string()),
            )))
        }
        ProviderKind::Anthropic => {
            let base_url = opts
                .base_url
                .clone()
                .or_else(|| kind.default_base_url().map(str::to_string))
                .unwrap_or_else(|| "https://api.anthropic.com".to_string());
            let api_key = auth::resolve_api_key(kind, opts.api_key.clone());
            let api_key =
                api_key.ok_or_else(|| LlmError::MissingApiKey("anthropic".to_string()))?;
            Ok(Arc::new(anthropic::AnthropicClient::new(
                base_url,
                api_key,
                opts.model
                    .unwrap_or_else(|| kind.default_model().to_string()),
            )))
        }
    }
}
