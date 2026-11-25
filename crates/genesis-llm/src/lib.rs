//! Model access for Genesis: hosted APIs, coding-agent CLIs, embeddings,
//! pricing, and the bandits that choose between models.
//!
//! Everything goes through [`LlmClient`]: it resolves a model name to a
//! backend ([`resolve_model`] lists the spellings), enforces the run's budget,
//! retries, prices the call and records it in the cost ledger.
//!
//! ```
//! use genesis_llm::{LlmClient, LlmRequest, ClientOptions};
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let client = LlmClient::new(ClientOptions::default());
//! let reply = client.query("demo", LlmRequest::new("mock/echo", "", "hello")).await.unwrap();
//! assert_eq!(reply.content, "hello");
//! # });
//! ```

pub mod agent;
pub mod bandit;
pub mod client;
pub mod embed;
pub mod extract;
pub mod kwargs;
pub mod pricing;
mod providers;
pub mod resolve;
#[cfg(test)]
mod testgen;
pub mod types;

pub use agent::{agent_available, run_agent, AgentOutcome, AgentSpec, AgentTask};
pub use bandit::{
    build_bandit, load_bandit_state, save_bandit_state, AsymmetricUcb, Bandit, FixedSampler,
    ThompsonOptions, ThompsonSampler, UcbOptions,
};
pub use client::{ClientOptions, LlmClient};
pub use embed::{cosine_similarity, local_hash_embedding};
pub use extract::{extract_fenced, extract_json, extract_tag};
pub use kwargs::SamplingKwargs;
pub use providers::mock::MockProvider;
pub use resolve::{missing_credentials, resolve_model, Route};
pub use types::{LlmError, LlmRequest, LlmResponse, Message, Provider, Role, Usage};
