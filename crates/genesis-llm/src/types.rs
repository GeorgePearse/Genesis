//! The types every model backend and every caller agree on.

use async_trait::async_trait;
use genesis_core::CostKind;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// One completion request. `messages` is the conversation so far and ends
/// with the user turn to answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LlmRequest {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    /// `disabled`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max`;
    /// each backend maps it to its own reasoning controls.
    pub reasoning_effort: Option<String>,
    /// Ask for a single JSON document where the backend can enforce it.
    pub json_output: bool,
    /// What the call is for (`proposal`, `novelty`, `idea`, ...). Set by
    /// [`crate::LlmClient::query`]; cost events and mocks key on it.
    pub stage: String,
}

impl LlmRequest {
    pub fn new(
        model: impl Into<String>,
        system: impl Into<String>,
        user: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            system: system.into(),
            messages: vec![Message::user(user)],
            ..Self::default()
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    pub fn json(mut self) -> Self {
        self.json_output = true;
        self
    }

    /// The last user turn, which is what a single-shot prompt consists of.
    pub fn last_user(&self) -> &str {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == Role::User)
            .map(|message| message.content.as_str())
            .unwrap_or("")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Usage {
    pub input_tokens: u64,
    /// Visible output, excluding reasoning.
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub thinking_tokens: u64,
}

impl Usage {
    /// Add `other` in place. Counts saturate instead of wrapping.
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(other.cached_input_tokens);
        self.thinking_tokens = self.thinking_tokens.saturating_add(other.thinking_tokens);
    }

    /// Everything billed as output: visible tokens plus reasoning.
    pub fn billed_output_tokens(&self) -> u64 {
        self.output_tokens.saturating_add(self.thinking_tokens)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct LlmResponse {
    pub content: String,
    /// Reasoning summary, when the backend returns one.
    pub thought: String,
    pub model: String,
    pub provider: String,
    pub usage: Usage,
    pub cost_usd: f64,
    pub cost_kind: CostKind,
    pub latency_secs: f64,
    /// Why the backend stopped early (`max_output_tokens`, ...), if it did.
    pub incomplete_reason: Option<String>,
    /// Billed calls behind this response, counting retried empty replies.
    pub calls: u32,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum LlmError {
    #[error("API budget exhausted: ${spent:.4} spent of ${budget:.4}")]
    BudgetExceeded { spent: f64, budget: f64 },
    /// The endpoint cannot serve this model at all: bad credentials, no
    /// quota, unknown model. Retrying the same model will not help.
    #[error("{provider} endpoint failure (HTTP {status}): {message}")]
    Endpoint {
        provider: String,
        status: u16,
        message: String,
    },
    /// The request itself was refused.
    #[error("{provider} rejected the request (HTTP {status}): {message}")]
    Rejected {
        provider: String,
        status: u16,
        message: String,
    },
    #[error("{provider} returned an empty response")]
    Empty {
        provider: String,
        incomplete_reason: Option<String>,
        cost_usd: f64,
    },
    /// Timeouts, connection errors and 5xx, after the retry budget.
    #[error("transient failure: {0}")]
    Transient(String),
    #[error("{0}")]
    Config(String),
}

impl LlmError {
    pub fn is_budget(&self) -> bool {
        matches!(self, Self::BudgetExceeded { .. })
    }

    pub fn is_endpoint(&self) -> bool {
        matches!(self, Self::Endpoint { .. } | Self::Config(_))
    }
}

/// A model backend: one HTTP API, one CLI, or a test double.
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError>;
}
