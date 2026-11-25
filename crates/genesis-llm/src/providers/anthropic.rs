//! Anthropic's Messages API over raw HTTP (there is no official Rust SDK).
//!
//! Requests are sent with `stream: true` and the event stream is read to its
//! end: a long generation keeps bytes flowing, so the connection is not
//! dropped as idle the way a large non-streaming request can be.
//!
//! Thinking and effort follow the per-model rules of the current API:
//! adaptive thinking with `output_config.effort` from Claude 4.6 on, a token
//! budget before that, and no sampling parameters at all (API version
//! `2026-10-21` rejects `temperature` on every model).

use super::{list_price, Backend, Billed, Call, Failure, TOKEN_LIMIT};
use crate::kwargs::{enabled_effort, thinking_budget};
use crate::pricing::catalog;
use crate::types::{LlmError, LlmRequest, LlmResponse, Role, Usage};
use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::time::Instant;

pub(crate) const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2026-10-21";
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const OAUTH_BETA: &str = "oauth-2025-04-20";
const DEFAULT_MAX_TOKENS: u32 = 16_000;

pub(crate) enum Credential {
    ApiKey(String),
    /// An OAuth access token, sent as a bearer token.
    AuthToken(String),
}

pub(crate) struct Anthropic {
    pub http: reqwest::Client,
    pub base_url: String,
    pub credential: Credential,
    /// Ask the API to re-run a request its safety classifiers decline on the
    /// model Anthropic recommends for that category.
    pub server_side_fallbacks: bool,
}

/// How a model's thinking is controlled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Generation {
    /// Thinking cannot be turned off; effort is the only control.
    AlwaysOn,
    /// Claude Sonnet 5.5: `between_tools` stands in for "off".
    BetweenTools,
    /// Adaptive by default; `disabled` accepted at effort `high` or below.
    AdaptiveCanDisable,
    /// Adaptive when asked for, off when `thinking` is omitted.
    AdaptiveOptIn { xhigh: bool },
    /// Before Claude 4.6: a thinking-token budget.
    Budget,
    /// No extended thinking.
    None,
}

fn generation(model: &str) -> Generation {
    let starts = |prefixes: &[&str]| prefixes.iter().any(|prefix| model.starts_with(prefix));
    if starts(&["claude-fable-5", "claude-mythos-5", "claude-opus-5"]) {
        // Claude Opus 5 accepts `disabled`, but then tends to leak reasoning
        // into the reply; like its successors it gets low effort instead.
        Generation::AlwaysOn
    } else if starts(&["claude-sonnet-5-5"]) {
        Generation::BetweenTools
    } else if starts(&["claude-haiku-5-5", "claude-sonnet-5"]) {
        Generation::AdaptiveCanDisable
    } else if starts(&["claude-opus-4-8", "claude-opus-4-7"]) {
        Generation::AdaptiveOptIn { xhigh: true }
    } else if starts(&["claude-opus-4-6", "claude-sonnet-4-6"]) {
        Generation::AdaptiveOptIn { xhigh: false }
    } else if catalog().lookup(model).is_some_and(|row| row.is_reasoning)
        || starts(&["claude-haiku-4-5"])
    {
        Generation::Budget
    } else {
        Generation::None
    }
}

/// Models whose requests opt into server-side refusal fallbacks by default.
fn takes_fallbacks(model: &str) -> bool {
    [
        "claude-fable-5-2",
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
    ]
    .contains(&model)
}

pub(crate) fn build_body(request: &LlmRequest, api_model: &str, fallbacks: bool) -> Value {
    let generation = generation(api_model);
    let effort = enabled_effort(request.reasoning_effort.as_deref());
    let level = |effort: &'static str, xhigh: bool| match effort {
        "minimal" => "low",
        "xhigh" if !xhigh => "high",
        other => other,
    };
    let cap = if matches!(generation, Generation::Budget | Generation::None) {
        64_000
    } else {
        128_000
    };
    let mut max_tokens = request
        .max_tokens
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .clamp(1, cap);

    let mut body = Map::new();
    body.insert("model".into(), json!(api_model));
    if !request.system.is_empty() {
        body.insert("system".into(), json!(request.system));
    }
    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(|message| {
            let role = match message.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            json!({"role": role, "content": message.content})
        })
        .collect();
    body.insert("messages".into(), Value::Array(messages));
    body.insert("stream".into(), json!(true));

    let adaptive = json!({"type": "adaptive", "display": "summarized"});
    match (generation, effort) {
        (Generation::AlwaysOn, effort) => {
            body.insert("thinking".into(), adaptive);
            body.insert(
                "output_config".into(),
                json!({"effort": level(effort.unwrap_or("low"), true)}),
            );
        }
        (Generation::BetweenTools, None) => {
            body.insert("thinking".into(), json!({"type": "between_tools"}));
            body.insert("output_config".into(), json!({"effort": "low"}));
        }
        (Generation::AdaptiveCanDisable, None) => {
            body.insert("thinking".into(), json!({"type": "disabled"}));
            body.insert("output_config".into(), json!({"effort": "low"}));
        }
        (Generation::BetweenTools | Generation::AdaptiveCanDisable, Some(effort)) => {
            body.insert("thinking".into(), adaptive);
            body.insert(
                "output_config".into(),
                json!({"effort": level(effort, true)}),
            );
        }
        (Generation::AdaptiveOptIn { xhigh }, Some(effort)) => {
            body.insert("thinking".into(), adaptive);
            body.insert(
                "output_config".into(),
                json!({"effort": level(effort, xhigh)}),
            );
        }
        (Generation::Budget, Some(effort)) => {
            // The budget must leave room for the answer and be at least 1024.
            max_tokens = max_tokens.max(2048);
            let wanted = thinking_budget(effort);
            let budget = if wanted < max_tokens { wanted } else { 1024 };
            body.insert(
                "thinking".into(),
                json!({"type": "enabled", "budget_tokens": budget}),
            );
        }
        (Generation::AdaptiveOptIn { .. } | Generation::Budget | Generation::None, _) => {}
    }
    body.insert("max_tokens".into(), json!(max_tokens));
    if fallbacks && takes_fallbacks(api_model) {
        body.insert("fallbacks".into(), json!("default"));
    }
    Value::Object(body)
}

#[derive(Default)]
struct StreamState {
    text: String,
    thought: String,
    served_by: Option<String>,
    stop_reason: Option<String>,
    refusal_category: Option<String>,
    input_tokens: u64,
    cache_read: u64,
    cache_creation: u64,
    output_tokens: u64,
    thinking_tokens: u64,
    finished: bool,
    error: Option<(String, String)>,
}

impl StreamState {
    fn usage(&mut self, usage: &Value) {
        let take = |slot: &mut u64, pointer: &str| {
            if let Some(count) = usage.pointer(pointer).and_then(Value::as_u64) {
                *slot = count.min(super::MAX_PLAUSIBLE_TOKENS);
            }
        };
        take(&mut self.input_tokens, "/input_tokens");
        take(&mut self.cache_read, "/cache_read_input_tokens");
        take(&mut self.cache_creation, "/cache_creation_input_tokens");
        take(&mut self.output_tokens, "/output_tokens");
        take(
            &mut self.thinking_tokens,
            "/output_tokens_details/thinking_tokens",
        );
    }

    fn event(&mut self, event: &Value) {
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                self.served_by = event
                    .pointer("/message/model")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                if let Some(usage) = event.pointer("/message/usage") {
                    self.usage(usage);
                }
            }
            Some("content_block_start") => {
                // A non-streamed block may arrive whole in its start event.
                if let Some(text) = event.pointer("/content_block/text").and_then(Value::as_str) {
                    self.text.push_str(text);
                }
            }
            Some("content_block_delta") => {
                match event.pointer("/delta/type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        self.text.push_str(
                            event
                                .pointer("/delta/text")
                                .and_then(Value::as_str)
                                .unwrap_or(""),
                        );
                    }
                    Some("thinking_delta") => {
                        self.thought.push_str(
                            event
                                .pointer("/delta/thinking")
                                .and_then(Value::as_str)
                                .unwrap_or(""),
                        );
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                if let Some(reason) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(reason.to_string());
                }
                let details = event
                    .pointer("/delta/stop_details")
                    .or_else(|| event.get("stop_details"));
                if let Some(category) = details
                    .and_then(|d| d.get("category"))
                    .and_then(Value::as_str)
                {
                    self.refusal_category = Some(category.to_string());
                }
                if let Some(usage) = event.get("usage") {
                    self.usage(usage);
                }
            }
            Some("message_stop") => self.finished = true,
            Some("error") => {
                let kind = event
                    .pointer("/error/type")
                    .and_then(Value::as_str)
                    .unwrap_or("error");
                let message = event
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                self.error = Some((kind.to_string(), message.to_string()));
            }
            _ => {}
        }
    }
}

/// Fold a server-sent event stream into its final state. Only `data:` lines
/// carry information; every payload names its own type.
fn read_stream(body: &str) -> StreamState {
    let mut state = StreamState::default();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        if let Ok(event) = serde_json::from_str::<Value>(data.trim()) {
            state.event(&event);
        }
    }
    state
}

impl Anthropic {
    fn headers(&self, api_model: &str) -> Vec<(&'static str, String)> {
        let mut betas = Vec::new();
        let mut headers = vec![("anthropic-version", API_VERSION.to_string())];
        match &self.credential {
            Credential::ApiKey(key) => headers.push(("x-api-key", key.clone())),
            Credential::AuthToken(token) => {
                headers.push(("authorization", format!("Bearer {token}")));
                betas.push(OAUTH_BETA);
            }
        }
        if self.server_side_fallbacks && takes_fallbacks(api_model) {
            betas.push(FALLBACK_BETA);
        }
        if !betas.is_empty() {
            headers.push(("anthropic-beta", betas.join(",")));
        }
        headers
    }

    pub(crate) fn parse(
        &self,
        call: &Call<'_>,
        text: &str,
        latency_secs: f64,
    ) -> Result<LlmResponse, Failure> {
        let state = read_stream(text);
        let total_input = state.input_tokens + state.cache_read + state.cache_creation;
        let thinking_tokens = state.thinking_tokens.min(state.output_tokens);
        let usage = Usage {
            input_tokens: total_input,
            output_tokens: state.output_tokens - thinking_tokens,
            cached_input_tokens: state.cache_read,
            thinking_tokens,
        };
        // A fallback model bills at its own rates.
        let served_by = state.served_by.clone().unwrap_or_default();
        let (cost_usd, cost_kind) =
            list_price(&[&served_by, call.api_model, &call.request.model], &usage);
        let billed = (total_input + state.output_tokens > 0).then(|| {
            Box::new(Billed {
                usage,
                cost_usd,
                cost_kind,
            })
        });
        let provider = "anthropic".to_string();

        if let Some((kind, message)) = state.error {
            let retryable = matches!(
                kind.as_str(),
                "overloaded_error" | "api_error" | "rate_limit_error"
            );
            let error = if retryable {
                LlmError::Transient(format!("anthropic stream failed with {kind}: {message}"))
            } else {
                LlmError::Rejected {
                    provider,
                    status: 200,
                    message: format!("{kind}: {message}"),
                }
            };
            return Err(Failure {
                error,
                retryable,
                retry_after: None,
                billed,
            });
        }
        if state.stop_reason.as_deref() == Some("refusal") {
            let category = state.refusal_category.as_deref().unwrap_or("unspecified");
            let message = format!("the model declined the request (refusal, category {category})");
            return Err(Failure {
                error: LlmError::Rejected {
                    provider,
                    status: 200,
                    message,
                },
                retryable: false,
                retry_after: None,
                billed,
            });
        }
        if !state.finished && state.stop_reason.is_none() {
            return Err(Failure {
                error: LlmError::Transient(
                    "anthropic stream ended before the message was complete".into(),
                ),
                retryable: true,
                retry_after: None,
                billed,
            });
        }
        let incomplete_reason = match state.stop_reason.as_deref() {
            Some("max_tokens") => Some(TOKEN_LIMIT.to_string()),
            Some("end_turn" | "stop_sequence") | None => None,
            Some(other) => Some(other.to_string()),
        };
        if state.text.trim().is_empty() {
            return Err(Failure {
                retryable: !super::is_token_limit(incomplete_reason.as_deref()),
                error: LlmError::Empty {
                    provider,
                    incomplete_reason,
                    cost_usd,
                },
                retry_after: None,
                billed,
            });
        }
        Ok(LlmResponse {
            content: state.text,
            thought: state.thought,
            model: call.request.model.clone(),
            provider,
            usage,
            cost_usd,
            cost_kind,
            latency_secs,
            incomplete_reason,
            calls: 1,
        })
    }
}

#[async_trait]
impl Backend for Anthropic {
    async fn attempt(&self, call: &Call<'_>) -> Result<LlmResponse, Failure> {
        let body = build_body(call.request, call.api_model, self.server_side_fallbacks);
        let url = format!("{}/v1/messages", self.base_url);
        let started = Instant::now();
        let text = super::post_json(
            &self.http,
            "anthropic",
            &url,
            &self.headers(call.api_model),
            &body,
            call.timeout,
        )
        .await?;
        self.parse(call, &text, started.elapsed().as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(model: &str, effort: Option<&str>) -> Value {
        let mut request = LlmRequest::new(model, "sys", "hi")
            .temperature(0.7)
            .max_tokens(4000);
        request.reasoning_effort = effort.map(str::to_string);
        build_body(&request, model, true)
    }

    #[test]
    fn no_request_carries_sampling_parameters() {
        for model in [
            "claude-fable-5-2",
            "claude-opus-5-5",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "claude-3-5-haiku-20241022",
        ] {
            let b = body(model, Some("high"));
            assert!(
                b.get("temperature").is_none() && b.get("top_p").is_none(),
                "{model}"
            );
            assert_eq!(b["stream"], json!(true));
            assert_eq!(b["system"], json!("sys"));
        }
    }

    #[test]
    fn always_on_models_lower_effort_instead_of_disabling() {
        for model in [
            "claude-fable-5-2",
            "claude-mythos-5-2",
            "claude-opus-5-5",
            "claude-opus-5",
        ] {
            let off = body(model, Some("disabled"));
            assert_eq!(
                off["thinking"],
                json!({"type": "adaptive", "display": "summarized"}),
                "{model}"
            );
            assert_eq!(off["output_config"], json!({"effort": "low"}), "{model}");
            assert_eq!(
                body(model, Some("xhigh"))["output_config"],
                json!({"effort": "xhigh"})
            );
            assert!(off["thinking"].get("budget_tokens").is_none());
        }
    }

    #[test]
    fn sonnet_and_haiku_five_five_turn_thinking_off_their_own_way() {
        let sonnet = body("claude-sonnet-5-5", None);
        assert_eq!(sonnet["thinking"], json!({"type": "between_tools"}));
        assert_eq!(sonnet["output_config"], json!({"effort": "low"}));
        let haiku = body("claude-haiku-5-5", Some("none"));
        assert_eq!(haiku["thinking"], json!({"type": "disabled"}));
        assert_eq!(
            body("claude-sonnet-5", None)["thinking"],
            json!({"type": "disabled"})
        );
        let on = body("claude-sonnet-5-5", Some("minimal"));
        assert_eq!(on["thinking"]["type"], json!("adaptive"));
        assert_eq!(on["output_config"], json!({"effort": "low"}));
    }

    #[test]
    fn opt_in_and_budget_generations() {
        let off = body("claude-opus-4-8", None);
        assert!(off.get("thinking").is_none() && off.get("output_config").is_none());
        assert_eq!(
            body("claude-opus-4-7", Some("xhigh"))["output_config"],
            json!({"effort": "xhigh"})
        );
        assert_eq!(
            body("claude-sonnet-4-6", Some("xhigh"))["output_config"],
            json!({"effort": "high"})
        );

        let legacy = body("claude-haiku-4-5-20251001", Some("low"));
        assert_eq!(
            legacy["thinking"],
            json!({"type": "enabled", "budget_tokens": 2048})
        );
        assert!(legacy.get("output_config").is_none());
        let squeezed = body("claude-sonnet-4-5-20250929", Some("max"));
        assert_eq!(
            squeezed["thinking"]["budget_tokens"],
            json!(1024),
            "the budget must fit under max_tokens"
        );
        assert!(body("claude-3-5-haiku-20241022", Some("high"))
            .get("thinking")
            .is_none());
        assert!(body("claude-haiku-4-5-20251001", None)
            .get("thinking")
            .is_none());
    }

    #[test]
    fn max_tokens_defaults_and_caps() {
        let request = LlmRequest::new("claude-opus-5-5", "", "hi");
        let b = build_body(&request, "claude-opus-5-5", false);
        assert_eq!(b["max_tokens"], json!(16_000));
        assert!(b.get("system").is_none() && b.get("fallbacks").is_none());
        let huge = LlmRequest::new("claude-opus-5-5", "", "hi").max_tokens(900_000);
        assert_eq!(
            build_body(&huge, "claude-opus-5-5", false)["max_tokens"],
            json!(128_000)
        );
        let old = LlmRequest::new("claude-3-5-haiku-20241022", "", "hi").max_tokens(900_000);
        assert_eq!(
            build_body(&old, "claude-3-5-haiku-20241022", false)["max_tokens"],
            json!(64_000)
        );
    }

    #[test]
    fn refusal_fallbacks_are_opted_into_only_where_supported() {
        assert_eq!(body("claude-opus-5-5", None)["fallbacks"], json!("default"));
        assert_eq!(
            body("claude-sonnet-5-5", None)["fallbacks"],
            json!("default")
        );
        assert!(body("claude-haiku-5-5", None).get("fallbacks").is_none());
        assert!(body("claude-opus-4-8", None).get("fallbacks").is_none());
    }

    #[test]
    fn the_event_stream_folds_into_text_thought_and_usage() {
        let stream = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"model\":\"claude-opus-5-5\",\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":5,\"cache_creation_input_tokens\":2,\"output_tokens\":1}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n",
            "event: ping\ndata: {\"type\":\"ping\"}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":40,\"output_tokens_details\":{\"thinking_tokens\":30}}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let state = read_stream(stream);
        assert_eq!(
            (state.text.as_str(), state.thought.as_str()),
            ("Hello", "hmm")
        );
        assert_eq!(
            (state.input_tokens, state.cache_read, state.cache_creation),
            (10, 5, 2)
        );
        assert_eq!((state.output_tokens, state.thinking_tokens), (40, 30));
        assert!(state.finished && state.error.is_none());
        assert_eq!(state.stop_reason.as_deref(), Some("end_turn"));
    }
}
