//! The one door to every model.
//!
//! [`LlmClient`] resolves a model name to a backend, enforces the spending
//! cap before anything is sent, retries what is worth retrying, prices the
//! result and writes it to the run's cost ledger. Clones share one spend
//! counter, so a budget holds across every phase of a run.

use crate::agent::AgentSpec;
use crate::embed::{
    clip, embed_gemini, embed_openai, local_hash_embedding, resolve_embedding, EmbedRoute,
    Embedded, Endpoint,
};
use crate::extract::extract_json;
use crate::providers::anthropic::{self, Anthropic, Credential};
use crate::providers::cli::CliBackend;
use crate::providers::gemini::{self, Gemini};
use crate::providers::mock::BuiltinMock;
use crate::providers::openai::{Flavour, OpenAiCompat};
use crate::providers::{Backend, Billed, Call, External, Failure};
use crate::resolve::{resolve_model, Route};
use crate::types::{LlmError, LlmRequest, LlmResponse, Message, Provider, Usage};
use genesis_core::{CostEvent, CostKind, CostLedger};
use rand::Rng;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MAX_BACKOFF: Duration = Duration::from_secs(20);

/// How an [`LlmClient`] behaves. Everything has a default.
#[derive(Clone)]
pub struct ClientOptions {
    /// Receives one event per billed call.
    pub ledger: Option<Arc<CostLedger>>,
    /// Hard cap, in dollars, across every call made through this client and
    /// its clones. A call is refused once the cap is reached.
    pub budget_usd: Option<f64>,
    /// The ledger's `phase` tag.
    pub phase: String,
    /// Extra attempts after a transient failure or an empty reply.
    pub max_retries: u32,
    pub timeout_secs: u64,
    /// When set, every model name resolves to this provider.
    pub mock: Option<Arc<dyn Provider>>,
    /// Where `cli/` models keep their prompt files and logs. Without it they
    /// use a scratch directory that is cleaned up after each call.
    pub work_dir: Option<PathBuf>,
    /// Base URL per provider (`openai`, `anthropic`, `google`, `openrouter`,
    /// `deepseek`, `azure_openai`), overriding the environment and defaults.
    pub base_urls: BTreeMap<String, String>,
    /// API key per provider, overriding the environment.
    pub api_keys: BTreeMap<String, String>,
    /// First retry delay; it doubles per attempt up to 20 seconds.
    pub backoff_base_ms: u64,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            ledger: None,
            budget_usd: None,
            phase: "investigation".to_string(),
            max_retries: 3,
            timeout_secs: 1200,
            mock: None,
            work_dir: None,
            base_urls: BTreeMap::new(),
            api_keys: BTreeMap::new(),
            backoff_base_ms: 1000,
        }
    }
}

struct Shared {
    options: ClientOptions,
    spent: Mutex<f64>,
    http: reqwest::Client,
    cli_gate: Arc<tokio::sync::Mutex<()>>,
}

/// Cheap to clone; clones share the spend counter, budget and ledger.
#[derive(Clone)]
pub struct LlmClient {
    shared: Arc<Shared>,
    phase: String,
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Charges of attempts that were billed without yielding the returned reply.
#[derive(Default)]
struct Carried {
    usage: Usage,
    cost_usd: f64,
    calls: u32,
}

impl LlmClient {
    pub fn new(options: ClientOptions) -> Self {
        let phase = options.phase.clone();
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            shared: Arc::new(Shared {
                options,
                spent: Mutex::new(0.0),
                http,
                cli_gate: Arc::new(tokio::sync::Mutex::new(())),
            }),
            phase,
        }
    }

    /// A client whose every model is answered by `provider`.
    pub fn mock(provider: Arc<dyn Provider>) -> Self {
        Self::new(ClientOptions {
            mock: Some(provider),
            backoff_base_ms: 1,
            ..ClientOptions::default()
        })
    }

    /// The same client, tagging its ledger events with another phase.
    pub fn with_phase(&self, phase: &str) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            phase: phase.to_string(),
        }
    }

    pub fn spent_usd(&self) -> f64 {
        *self.shared.spent.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn budget_exhausted(&self) -> bool {
        match self.shared.options.budget_usd {
            Some(budget) => {
                let spent = self.spent_usd();
                spent >= budget || (spent - budget).abs() <= 1e-12 * budget.abs().max(spent.abs())
            }
            None => false,
        }
    }

    pub fn ledger(&self) -> Option<Arc<CostLedger>> {
        self.shared.options.ledger.clone()
    }

    fn admit(&self) -> Result<(), LlmError> {
        if self.budget_exhausted() {
            return Err(LlmError::BudgetExceeded {
                spent: self.spent_usd(),
                budget: self.shared.options.budget_usd.unwrap_or(0.0),
            });
        }
        Ok(())
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.shared.options.timeout_secs.max(1))
    }

    fn base_url(&self, provider: &str, env: Option<&str>, default: &str) -> String {
        let options = &self.shared.options;
        let url = options
            .base_urls
            .get(provider)
            .cloned()
            .or_else(|| env.and_then(env_value))
            .unwrap_or_else(|| default.to_string());
        url.trim_end_matches('/').to_string()
    }

    fn api_key(&self, provider: &str, env: &[&str], model: &str) -> Result<String, LlmError> {
        if let Some(key) = self.shared.options.api_keys.get(provider) {
            return Ok(key.clone());
        }
        env.iter().find_map(|name| env_value(name)).ok_or_else(|| {
            LlmError::Config(format!("`{model}` needs {} to be set", env.join(" or ")))
        })
    }

    fn openai_compat(
        &self,
        flavour: Flavour,
        base_url: String,
        api_key: Option<String>,
    ) -> OpenAiCompat {
        OpenAiCompat {
            http: self.shared.http.clone(),
            flavour,
            base_url,
            api_key,
        }
    }

    fn local_key(&self, api_key_env: Option<&str>) -> String {
        api_key_env
            .and_then(env_value)
            .or_else(|| env_value("LOCAL_OPENAI_API_KEY"))
            .unwrap_or_else(|| "local".to_string())
    }

    fn azure_base(&self, model: &str) -> Result<String, LlmError> {
        if let Some(url) = self.shared.options.base_urls.get("azure_openai") {
            return Ok(url.trim_end_matches('/').to_string());
        }
        let endpoint = env_value("AZURE_API_ENDPOINT").ok_or_else(|| {
            LlmError::Config(format!("`{model}` needs AZURE_API_ENDPOINT to be set"))
        })?;
        Ok(format!("{}/openai/v1", endpoint.trim_end_matches('/')))
    }

    fn openai_like(&self, route: &Route, model: &str) -> Result<Option<OpenAiCompat>, LlmError> {
        Ok(Some(match route {
            Route::OpenAi { .. } => self.openai_compat(
                Flavour::OpenAi,
                self.base_url(
                    "openai",
                    Some("OPENAI_BASE_URL"),
                    "https://api.openai.com/v1",
                ),
                Some(self.api_key("openai", &["OPENAI_API_KEY"], model)?),
            ),
            Route::Azure { .. } => self.openai_compat(
                Flavour::Azure,
                self.azure_base(model)?,
                Some(self.api_key("azure_openai", &["AZURE_OPENAI_API_KEY"], model)?),
            ),
            Route::OpenRouter { .. } => self.openai_compat(
                Flavour::OpenRouter,
                self.base_url("openrouter", None, "https://openrouter.ai/api/v1"),
                Some(self.api_key("openrouter", &["OPENROUTER_API_KEY"], model)?),
            ),
            Route::DeepSeek { .. } => self.openai_compat(
                Flavour::DeepSeek,
                self.base_url("deepseek", None, "https://api.deepseek.com"),
                Some(self.api_key("deepseek", &["DEEPSEEK_API_KEY"], model)?),
            ),
            Route::Local {
                base_url,
                api_key_env,
                ..
            } => self.openai_compat(
                Flavour::Local,
                base_url.clone(),
                Some(self.local_key(api_key_env.as_deref())),
            ),
            _ => return Ok(None),
        }))
    }

    fn cli_backend(&self, spec: AgentSpec) -> CliBackend {
        let (work_dir, keep_files) = match &self.shared.options.work_dir {
            Some(dir) => (dir.clone(), true),
            None => (std::env::temp_dir().join("genesis-llm"), false),
        };
        CliBackend {
            spec,
            work_dir,
            keep_files,
            gate: Arc::clone(&self.shared.cli_gate),
        }
    }

    /// The backend for `model` and the name its API expects.
    fn backend(&self, model: &str) -> Result<(Arc<dyn Backend>, String), LlmError> {
        if let Some(mock) = &self.shared.options.mock {
            return Ok((Arc::new(External(Arc::clone(mock))), model.to_string()));
        }
        let route = resolve_model(model)?;
        let api_model = route.api_model().to_string();
        if let Some(compat) = self.openai_like(&route, model)? {
            return Ok((Arc::new(compat), api_model));
        }
        let backend: Arc<dyn Backend> = match route {
            Route::Mock { behaviour } => Arc::new(BuiltinMock { behaviour }),
            Route::Cli { spec } => Arc::new(self.cli_backend(spec)),
            Route::Anthropic { .. } => {
                let base_url = self.base_url(
                    "anthropic",
                    Some("ANTHROPIC_BASE_URL"),
                    anthropic::DEFAULT_BASE_URL,
                );
                let credential = match self.shared.options.api_keys.get("anthropic").cloned().or_else(|| env_value("ANTHROPIC_API_KEY")) {
                    Some(key) => Credential::ApiKey(key),
                    None => Credential::AuthToken(env_value("ANTHROPIC_AUTH_TOKEN").ok_or_else(|| {
                        LlmError::Config(format!("`{model}` needs ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN to be set"))
                    })?),
                };
                let first_party = base_url == anthropic::DEFAULT_BASE_URL;
                let opted_out =
                    env_value("GENESIS_ANTHROPIC_FALLBACKS").is_some_and(|value| value == "0");
                Arc::new(Anthropic {
                    http: self.shared.http.clone(),
                    base_url,
                    credential,
                    server_side_fallbacks: first_party && !opted_out,
                })
            }
            Route::Gemini { .. } => Arc::new(Gemini {
                http: self.shared.http.clone(),
                base_url: self.base_url("google", None, gemini::DEFAULT_BASE_URL),
                api_key: self.api_key("google", &["GEMINI_API_KEY", "GOOGLE_API_KEY"], model)?,
            }),
            _ => unreachable!("every OpenAI-compatible route was handled above"),
        };
        Ok((backend, api_model))
    }

    fn backoff(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        let base = Duration::from_millis(self.shared.options.backoff_base_ms);
        let nominal = base
            .saturating_mul(2u32.saturating_pow(attempt.min(16)))
            .min(MAX_BACKOFF);
        let jittered = nominal.mul_f64(rand::thread_rng().gen_range(0.5..=1.0));
        jittered.max(retry_after.unwrap_or(Duration::ZERO))
    }

    fn record(&self, stage: &str, source: &str, provider: &str, model: &str, billed: &Billed) {
        if billed.cost_usd > 0.0 {
            *self.shared.spent.lock().unwrap_or_else(|p| p.into_inner()) += billed.cost_usd;
        }
        if let Some(ledger) = &self.shared.options.ledger {
            ledger.record(CostEvent {
                phase: self.phase.clone(),
                stage: stage.to_string(),
                source: source.to_string(),
                provider: provider.to_string(),
                model: model.to_string(),
                input_tokens: billed.usage.input_tokens,
                output_tokens: billed.usage.output_tokens,
                cached_input_tokens: billed.usage.cached_input_tokens,
                thinking_tokens: billed.usage.thinking_tokens,
                cost_usd: billed.cost_usd,
                cost_kind: billed.cost_kind,
                calls: 1,
                ..CostEvent::default()
            });
        }
    }

    /// Send one request and return the reply.
    ///
    /// The budget is checked before the request is sent and before each
    /// retry. Transient failures and empty replies are retried up to
    /// `max_retries` times; what those attempts were billed is added to the
    /// reply that is finally returned.
    pub async fn query(
        &self,
        stage: &str,
        mut request: LlmRequest,
    ) -> Result<LlmResponse, LlmError> {
        request.stage = stage.to_string();
        let (backend, api_model) = self.backend(&request.model)?;
        let provider_hint = resolve_model(&request.model)
            .map(|route| route.provider())
            .unwrap_or("mock");
        let mut carried = Carried::default();
        let mut attempt = 0u32;
        loop {
            self.admit()?;
            let call = Call {
                request: &request,
                api_model: &api_model,
                timeout: self.timeout(),
            };
            match backend.attempt(&call).await {
                Ok(mut response) => {
                    let billed = Billed {
                        usage: response.usage,
                        cost_usd: response.cost_usd,
                        cost_kind: response.cost_kind,
                    };
                    self.record(
                        stage,
                        backend.source(),
                        &response.provider,
                        &request.model,
                        &billed,
                    );
                    response.usage.add(&carried.usage);
                    response.cost_usd += carried.cost_usd;
                    response.calls = response.calls.max(1) + carried.calls;
                    return Ok(response);
                }
                Err(Failure {
                    error,
                    retryable,
                    retry_after,
                    billed,
                }) => {
                    if let Some(billed) = &billed {
                        self.record(
                            stage,
                            backend.source(),
                            provider_hint,
                            &request.model,
                            billed,
                        );
                        carried.usage.add(&billed.usage);
                        carried.cost_usd += billed.cost_usd;
                        carried.calls += 1;
                    }
                    if !retryable || attempt >= self.shared.options.max_retries {
                        return Err(match error {
                            LlmError::Empty {
                                provider,
                                incomplete_reason,
                                ..
                            } => LlmError::Empty {
                                provider,
                                incomplete_reason,
                                cost_usd: carried.cost_usd,
                            },
                            other => other,
                        });
                    }
                    tracing::debug!(model = %request.model, attempt, "retrying after: {error}");
                    tokio::time::sleep(self.backoff(attempt, retry_after)).await;
                    attempt += 1;
                }
            }
        }
    }

    /// Like [`LlmClient::query`], for a reply that must be a JSON document.
    ///
    /// A reply that cannot be read as JSON is sent back once with the
    /// complaint; if the second reply cannot be read either the call fails
    /// with [`LlmError::Empty`] and `incomplete_reason` `invalid_json`. The
    /// returned response carries the cost and usage of both calls.
    pub async fn query_json(
        &self,
        stage: &str,
        mut request: LlmRequest,
    ) -> Result<(serde_json::Value, LlmResponse), LlmError> {
        request.json_output = true;
        let first = self.query(stage, request.clone()).await?;
        if let Some(value) = extract_json(&first.content) {
            return Ok((value, first));
        }
        request
            .messages
            .push(Message::assistant(first.content.clone()));
        request.messages.push(Message::user(
            "That reply could not be read as JSON: no complete, valid JSON object or array was found in it. \
             Reply again with only the JSON document and nothing else.",
        ));
        let mut second = self.query(stage, request).await?;
        second.usage.add(&first.usage);
        second.cost_usd += first.cost_usd;
        second.calls += first.calls;
        match extract_json(&second.content) {
            Some(value) => Ok((value, second)),
            None => Err(LlmError::Empty {
                provider: second.provider,
                incomplete_reason: Some("invalid_json".to_string()),
                cost_usd: second.cost_usd,
            }),
        }
    }

    async fn embed_once(
        &self,
        route: &EmbedRoute,
        model: &str,
        text: &str,
    ) -> Result<Embedded, Failure> {
        let http = &self.shared.http;
        let timeout = self.timeout().min(Duration::from_secs(600));
        let bearer = |key: String| vec![("authorization", format!("Bearer {key}"))];
        let fatal = Failure::fatal;
        match route {
            EmbedRoute::LocalHash { dim } => Ok(Embedded {
                vector: local_hash_embedding(text, *dim),
                billed: Billed {
                    usage: Usage::default(),
                    cost_usd: 0.0,
                    cost_kind: CostKind::UsageOnly,
                },
            }),
            EmbedRoute::OpenAi { model: api_model } => {
                let base_url = self.base_url(
                    "openai",
                    Some("OPENAI_BASE_URL"),
                    "https://api.openai.com/v1",
                );
                let key = self
                    .api_key("openai", &["OPENAI_API_KEY"], model)
                    .map_err(fatal)?;
                let endpoint = Endpoint {
                    http,
                    provider: "openai",
                    base_url: &base_url,
                    headers: bearer(key),
                    timeout,
                };
                embed_openai(&endpoint, model, api_model, text).await
            }
            EmbedRoute::Azure { deployment } => {
                let base_url = self.azure_base(model).map_err(fatal)?;
                let key = self
                    .api_key("azure_openai", &["AZURE_OPENAI_API_KEY"], model)
                    .map_err(fatal)?;
                let headers = vec![("api-key", key)];
                let endpoint = Endpoint {
                    http,
                    provider: "azure_openai",
                    base_url: &base_url,
                    headers,
                    timeout,
                };
                embed_openai(&endpoint, model, deployment, text).await
            }
            EmbedRoute::OpenRouter { model: api_model } => {
                let base_url = self.base_url("openrouter", None, "https://openrouter.ai/api/v1");
                let key = self
                    .api_key("openrouter", &["OPENROUTER_API_KEY"], model)
                    .map_err(fatal)?;
                let endpoint = Endpoint {
                    http,
                    provider: "openrouter",
                    base_url: &base_url,
                    headers: bearer(key),
                    timeout,
                };
                embed_openai(&endpoint, model, api_model, text).await
            }
            EmbedRoute::Local {
                model: api_model,
                base_url,
                api_key_env,
            } => {
                let headers = bearer(self.local_key(api_key_env.as_deref()));
                let endpoint = Endpoint {
                    http,
                    provider: "local_openai",
                    base_url,
                    headers,
                    timeout,
                };
                embed_openai(&endpoint, model, api_model, text).await
            }
            EmbedRoute::Gemini { model: api_model } => {
                let base_url = self.base_url("google", None, gemini::DEFAULT_BASE_URL);
                let key = self
                    .api_key("google", &["GEMINI_API_KEY", "GOOGLE_API_KEY"], model)
                    .map_err(fatal)?;
                let headers = vec![("x-goog-api-key", key)];
                let endpoint = Endpoint {
                    http,
                    provider: "google",
                    base_url: &base_url,
                    headers,
                    timeout,
                };
                embed_gemini(&endpoint, api_model, text).await
            }
        }
    }

    /// Embed the first 10,000 characters of `text` with `model`.
    pub async fn embed(&self, stage: &str, model: &str, text: &str) -> Result<Vec<f32>, LlmError> {
        let route = resolve_embedding(model)?;
        let text = clip(text);
        let provider = match &route {
            EmbedRoute::LocalHash { .. } => "local_hash",
            EmbedRoute::OpenAi { .. } => "openai",
            EmbedRoute::Azure { .. } => "azure_openai",
            EmbedRoute::OpenRouter { .. } => "openrouter",
            EmbedRoute::Local { .. } => "local_openai",
            EmbedRoute::Gemini { .. } => "google",
        };
        let mut attempt = 0u32;
        loop {
            if !matches!(route, EmbedRoute::LocalHash { .. }) {
                self.admit()?;
            }
            match self.embed_once(&route, model, text).await {
                Ok(embedded) => {
                    if !matches!(route, EmbedRoute::LocalHash { .. }) {
                        self.record(stage, "embedding", provider, model, &embedded.billed);
                    }
                    return Ok(embedded.vector);
                }
                Err(failure) => {
                    if !failure.retryable || attempt >= self.shared.options.max_retries {
                        return Err(failure.error);
                    }
                    tokio::time::sleep(self.backoff(attempt, failure.retry_after)).await;
                    attempt += 1;
                }
            }
        }
    }
}
