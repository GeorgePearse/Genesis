//! Model backends and the HTTP plumbing they share.
//!
//! A backend makes exactly one attempt per call and says whether a failure is
//! worth repeating; [`crate::LlmClient`] owns the retry loop, the budget and
//! the ledger.

pub mod anthropic;
pub mod cli;
pub mod gemini;
pub mod mock;
pub mod openai;

use crate::pricing::catalog;
use crate::types::{LlmError, LlmRequest, LlmResponse, Provider, Usage};
use async_trait::async_trait;
use genesis_core::CostKind;
use reqwest::header::HeaderMap;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// A failed attempt, with what the retry loop needs to decide what next.
#[derive(Debug)]
pub(crate) struct Failure {
    pub error: LlmError,
    pub retryable: bool,
    pub retry_after: Option<Duration>,
    /// What the failed attempt was still billed, when the backend says.
    pub billed: Option<Box<Billed>>,
}

#[derive(Debug, Clone)]
pub(crate) struct Billed {
    pub usage: Usage,
    pub cost_usd: f64,
    pub cost_kind: CostKind,
}

impl Failure {
    pub fn fatal(error: LlmError) -> Self {
        Self {
            error,
            retryable: false,
            retry_after: None,
            billed: None,
        }
    }

    pub fn transient(message: impl Into<String>) -> Self {
        Self {
            error: LlmError::Transient(message.into()),
            retryable: true,
            retry_after: None,
            billed: None,
        }
    }
}

/// One resolved call: the caller's request and the name the API expects.
pub(crate) struct Call<'a> {
    pub request: &'a LlmRequest,
    pub api_model: &'a str,
    pub timeout: Duration,
}

#[async_trait]
pub(crate) trait Backend: Send + Sync {
    /// `llm_api` or `agent_cli`: the ledger's `source`.
    fn source(&self) -> &'static str {
        "llm_api"
    }

    async fn attempt(&self, call: &Call<'_>) -> Result<LlmResponse, Failure>;
}

/// Adapts a caller-supplied [`Provider`] (usually a test double).
pub(crate) struct External(pub Arc<dyn Provider>);

#[async_trait]
impl Backend for External {
    async fn attempt(&self, call: &Call<'_>) -> Result<LlmResponse, Failure> {
        self.0.complete(call.request).await.map_err(|error| {
            let retryable = match &error {
                LlmError::Transient(_) => true,
                LlmError::Empty {
                    incomplete_reason, ..
                } => !is_token_limit(incomplete_reason.as_deref()),
                _ => false,
            };
            let billed = match &error {
                LlmError::Empty { cost_usd, .. } if *cost_usd > 0.0 => Some(Box::new(Billed {
                    usage: Usage::default(),
                    cost_usd: *cost_usd,
                    cost_kind: CostKind::Reported,
                })),
                _ => None,
            };
            Failure {
                error,
                retryable,
                retry_after: None,
                billed,
            }
        })
    }
}

pub(crate) const TOKEN_LIMIT: &str = "max_output_tokens";

pub(crate) fn is_token_limit(reason: Option<&str>) -> bool {
    reason == Some(TOKEN_LIMIT)
}

/// Price `usage` from the catalog under the first of `names` it knows.
/// Output is billed with reasoning included.
pub(crate) fn list_price(names: &[&str], usage: &Usage) -> (f64, CostKind) {
    let billed_output = usage.billed_output_tokens();
    for name in names {
        if let Some((input, output)) = catalog().cost(name, usage.input_tokens, billed_output) {
            return (input + output, CostKind::Estimated);
        }
    }
    (0.0, CostKind::UsageOnly)
}

fn header_seconds(headers: &HeaderMap, name: &str, scale: f64) -> Option<Duration> {
    let seconds = headers
        .get(name)?
        .to_str()
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()?
        * scale;
    (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds.min(600.0)))
}

fn retry_after(headers: &HeaderMap, body: &Value) -> Option<Duration> {
    header_seconds(headers, "retry-after-ms", 1e-3)
        .or_else(|| header_seconds(headers, "retry-after", 1.0))
        .or_else(|| {
            let seconds = body.get("retry_after").and_then(Value::as_f64)?;
            (seconds.is_finite() && seconds >= 0.0)
                .then(|| Duration::from_secs_f64(seconds.min(600.0)))
        })
}

/// Machine-readable error codes a provider may put in its error body.
fn error_codes(body: &Value) -> Vec<String> {
    let error = body.get("error").unwrap_or(body);
    let mut codes = Vec::new();
    for key in ["code", "type", "status"] {
        if let Some(text) = error.get(key).and_then(Value::as_str) {
            codes.push(text.to_string());
        }
    }
    if let Some(text) = error.pointer("/details/error_code").and_then(Value::as_str) {
        codes.push(text.to_string());
    }
    if let Some(details) = error.get("details").and_then(Value::as_array) {
        codes.extend(
            details
                .iter()
                .filter_map(|d| d.get("reason").and_then(Value::as_str))
                .map(str::to_string),
        );
    }
    codes
}

fn error_message(body: &Value, raw: &str) -> String {
    let message = body
        .pointer("/error/message")
        .or_else(|| body.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| raw.trim().to_string());
    genesis_core::fsutil::truncate(&message, 1000)
}

const DEAD_ENDPOINT_CODES: &[&str] = &[
    "API_KEY_INVALID",
    "invalid_api_key",
    "insufficient_quota",
    "credit_balance_too_low",
    "enforced_spend_limit_reached",
    "authentication_error",
    "billing_error",
    "permission_error",
];

/// Decide what a non-2xx reply means.
///
/// An endpoint that cannot serve the model at all (bad key, no quota, unknown
/// model) is an [`LlmError::Endpoint`]; a request the provider refuses is
/// [`LlmError::Rejected`]; rate limits, server errors and anything the
/// provider marks `x-should-retry: true` are transient.
pub(crate) fn classify_status(
    provider: &str,
    host: &str,
    status: u16,
    headers: &HeaderMap,
    raw: &str,
) -> Failure {
    let body: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
    let message = error_message(&body, raw);
    let mut codes = error_codes(&body);
    if message
        .to_ascii_lowercase()
        .contains("credit balance is too low")
    {
        codes.push("credit_balance_too_low".to_string());
    }
    let dead_code = codes
        .iter()
        .any(|code| DEAD_ENDPOINT_CODES.contains(&code.as_str()));
    let dead_status =
        matches!(status, 401..=404) && !(status == 403 && host.ends_with("openrouter.ai"));
    if dead_code || dead_status {
        return Failure::fatal(LlmError::Endpoint {
            provider: provider.to_string(),
            status,
            message,
        });
    }
    let should_retry = headers
        .get("x-should-retry")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| match value.trim() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        });
    let retryable =
        should_retry.unwrap_or(matches!(status, 408 | 409 | 425 | 429 | 499) || status >= 500);
    if retryable {
        return Failure {
            error: LlmError::Transient(format!("{provider} returned HTTP {status}: {message}")),
            retryable: true,
            retry_after: retry_after(headers, &body),
            billed: None,
        };
    }
    Failure::fatal(LlmError::Rejected {
        provider: provider.to_string(),
        status,
        message,
    })
}

pub(crate) fn host_of(url: &str) -> String {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .unwrap_or_default()
}

/// POST `body` as JSON and return the response body of a 2xx reply.
pub(crate) async fn post_json(
    http: &reqwest::Client,
    provider: &str,
    url: &str,
    headers: &[(&str, String)],
    body: &Value,
    timeout: Duration,
) -> Result<String, Failure> {
    let mut request = http.post(url).timeout(timeout).json(body);
    for (name, value) in headers {
        request = request.header(*name, value);
    }
    let response = request.send().await.map_err(|err| {
        Failure::transient(format!("{provider} request failed: {}", describe(err)))
    })?;
    let status = response.status();
    let response_headers = response.headers().clone();
    let text = response.text().await.map_err(|err| {
        Failure::transient(format!(
            "{provider} response was cut short: {}",
            describe(err)
        ))
    })?;
    if status.is_success() {
        Ok(text)
    } else {
        Err(classify_status(
            provider,
            &host_of(url),
            status.as_u16(),
            &response_headers,
            &text,
        ))
    }
}

/// A transport error without its URL, which can carry a key in its query.
fn describe(err: reqwest::Error) -> String {
    if err.is_timeout() {
        return "timed out".to_string();
    }
    let connect = err.is_connect();
    let detail = err.without_url().to_string();
    if connect {
        format!("could not connect ({detail})")
    } else {
        detail
    }
}

pub(crate) fn parse_body(provider: &str, text: &str) -> Result<Value, Failure> {
    serde_json::from_str(text).map_err(|err| {
        Failure::transient(format!(
            "{provider} returned a body that is not JSON ({err}): {}",
            genesis_core::fsutil::truncate(text, 200)
        ))
    })
}

/// No single call uses a trillion tokens. A count beyond this is output gone
/// wrong, and is cut down so that sums of counts can never overflow.
pub(crate) const MAX_PLAUSIBLE_TOKENS: u64 = 1 << 40;

/// The token count at `pointer`, or 0 when it is absent or not a count.
pub(crate) fn u64_at(value: &Value, pointer: &str) -> u64 {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(MAX_PLAUSIBLE_TOKENS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(status: u16, body: &str, headers: &[(&'static str, &'static str)]) -> Failure {
        let mut map = HeaderMap::new();
        for (name, value) in headers {
            map.insert(*name, value.parse().unwrap());
        }
        classify_status("openai", "api.openai.com", status, &map, body)
    }

    fn replies() -> ephyra::BoxGen<(&'static str, String)> {
        use crate::testgen::{near, near_lines};
        use ephyra::prelude::*;
        use serde_json::json;
        let openai = near(json!({
            "choices": [{"message": {"content": "hi", "reasoning_content": "because"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.01,
                      "prompt_tokens_details": {"cached_tokens": 2},
                      "completion_tokens_details": {"reasoning_tokens": 1}}
        }))
        .map(|body| ("openai", body.to_string()));
        let gemini = near(json!({
            "candidates": [{"content": {"parts": [{"text": "plan", "thought": true}, {"text": "hi"}]}, "finishReason": "STOP"}],
            "promptFeedback": {"blockReason": "SAFETY"},
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4, "thoughtsTokenCount": 3, "cachedContentTokenCount": 1}
        }))
        .map(|body| ("google", body.to_string()));
        let anthropic = near_lines(vec![
            json!({"type": "message_start", "message": {"model": "claude-opus-5-5", "usage": {"input_tokens": 9, "cache_read_input_tokens": 1, "cache_creation_input_tokens": 2, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_details": {"category": "cyber"}}, "usage": {"output_tokens": 7, "output_tokens_details": {"thinking_tokens": 3}}}),
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
            json!({"type": "message_stop"}),
        ])
        .map(|lines| ("anthropic", lines.lines().map(|line| format!("data: {line}\n\n")).collect::<String>()));
        ephyra::gens::one_of(vec![openai.boxed(), gemini.boxed(), anthropic.boxed()]).boxed()
    }

    /// A provider's reply is untrusted. Whatever it holds, parsing must end
    /// in a response with text and a sane bill, or in an error; never a panic.
    #[ephyra::test(gen = replies())]
    fn provider_replies_parse_to_a_sane_response_or_an_error(reply: (&'static str, String)) {
        let (provider, body) = reply;
        let http = reqwest::Client::new();
        let request = LlmRequest::new("claude-opus-5-5", "", "hi");
        let call = Call {
            request: &request,
            api_model: "claude-opus-5-5",
            timeout: Duration::from_secs(1),
        };
        for text in [body.clone(), body.chars().take(body.len() / 2).collect()] {
            let parsed = match provider {
                "openai" => openai::OpenAiCompat {
                    http: http.clone(),
                    flavour: openai::Flavour::OpenRouter,
                    base_url: String::new(),
                    api_key: None,
                }
                .parse(&call, &text, 0.0),
                "google" => gemini::Gemini {
                    http: http.clone(),
                    base_url: String::new(),
                    api_key: String::new(),
                }
                .parse(&call, &text, 0.0),
                _ => anthropic::Anthropic {
                    http: http.clone(),
                    base_url: String::new(),
                    credential: anthropic::Credential::ApiKey(String::new()),
                    server_side_fallbacks: false,
                }
                .parse(&call, &text, 0.0),
            };
            let sane = |cost: f64, usage: &Usage| {
                assert!(cost.is_finite() && cost >= 0.0, "cost {cost} from {text}");
                for count in [
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cached_input_tokens,
                    usage.thinking_tokens,
                ] {
                    assert!(
                        count <= 4 * MAX_PLAUSIBLE_TOKENS,
                        "count {count} from {text}"
                    );
                }
            };
            match parsed {
                Ok(response) => {
                    ephyra::cover(true, "parsed to a response");
                    assert!(
                        !response.content.trim().is_empty(),
                        "an empty reply was accepted: {text}"
                    );
                    sane(response.cost_usd, &response.usage);
                }
                Err(failure) => {
                    if let Some(billed) = &failure.billed {
                        sane(billed.cost_usd, &billed.usage);
                    }
                    if let LlmError::Empty { cost_usd, .. } = &failure.error {
                        assert!(cost_usd.is_finite() && *cost_usd >= 0.0);
                    }
                    assert!(
                        !failure.retryable
                            || matches!(
                                failure.error,
                                LlmError::Transient(_) | LlmError::Empty { .. }
                            )
                    );
                }
            }
        }
    }

    #[ephyra::test(gen = (
        ephyra::gens::int(100u16..=599),
        crate::testgen::near(serde_json::json!({"error": {"message": "nope", "type": "invalid_request_error", "code": "bad", "status": "DENIED", "details": [{"reason": "API_KEY_INVALID"}]}, "retry_after": 3})),
        ephyra::gens::pick_from(vec!["", "2", "-1", "1e400", "NaN", "soon", "0.25"]),
        ephyra::gens::pick_from(vec!["", "true", "false", "maybe"]),
    ))]
    fn any_error_reply_is_classified_without_panicking(
        input: (u16, Value, &'static str, &'static str),
    ) {
        let (status, body, retry_header, should_retry) = input;
        let mut headers = HeaderMap::new();
        if !retry_header.is_empty() {
            headers.insert("retry-after", retry_header.parse().unwrap());
        }
        if !should_retry.is_empty() {
            headers.insert("x-should-retry", should_retry.parse().unwrap());
        }
        let failure = classify_status(
            "openai",
            "api.openai.com",
            status,
            &headers,
            &body.to_string(),
        );
        match &failure.error {
            LlmError::Transient(_) => assert!(failure.retryable),
            LlmError::Endpoint {
                status: reported, ..
            }
            | LlmError::Rejected {
                status: reported, ..
            } => {
                assert!(!failure.retryable && *reported == status);
            }
            other => panic!("unexpected classification {other}"),
        }
        assert!(failure
            .retry_after
            .is_none_or(|wait| wait <= Duration::from_secs(600)));
        assert!(failure.billed.is_none());
    }

    #[test]
    fn credential_and_quota_failures_are_endpoint_errors() {
        for status in [401, 402, 403, 404] {
            assert!(
                matches!(classify(status, "{}", &[]).error, LlmError::Endpoint { .. }),
                "{status}"
            );
        }
        let quota = classify(
            429,
            r#"{"error": {"code": "insufficient_quota", "message": "no quota"}}"#,
            &[],
        );
        assert!(matches!(quota.error, LlmError::Endpoint { status: 429, .. }) && !quota.retryable);
        let credit = classify(
            400,
            r#"{"error": {"type": "invalid_request_error", "message": "Your credit balance is too low"}}"#,
            &[],
        );
        assert!(matches!(credit.error, LlmError::Endpoint { .. }));
        let gemini = classify(
            400,
            r#"{"error": {"status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}]}}"#,
            &[],
        );
        assert!(matches!(gemini.error, LlmError::Endpoint { .. }));
        let spend_cap = classify(
            429,
            r#"{"error": {"type": "rate_limit_error", "details": {"error_code": "enforced_spend_limit_reached"}}}"#,
            &[],
        );
        assert!(!spend_cap.retryable);
    }

    #[test]
    fn openrouter_403_is_a_rejection_not_a_dead_endpoint() {
        let failure = classify_status("openrouter", "openrouter.ai", 403, &HeaderMap::new(), "{}");
        assert!(matches!(
            failure.error,
            LlmError::Rejected { status: 403, .. }
        ));
    }

    #[test]
    fn rate_limits_and_server_errors_retry_and_honour_retry_after() {
        let limited = classify(
            429,
            r#"{"error": {"message": "slow down"}}"#,
            &[("retry-after", "2.5")],
        );
        assert!(limited.retryable && limited.retry_after == Some(Duration::from_secs_f64(2.5)));
        assert!(matches!(limited.error, LlmError::Transient(_)));
        let ms = classify(503, "upstream down", &[("retry-after-ms", "250")]);
        assert_eq!(ms.retry_after, Some(Duration::from_millis(250)));
        let from_body = classify(500, r#"{"retry_after": 4}"#, &[]);
        assert_eq!(from_body.retry_after, Some(Duration::from_secs(4)));
        for status in [408, 409, 425, 499, 500, 529] {
            assert!(classify(status, "", &[]).retryable, "{status}");
        }
    }

    #[test]
    fn other_client_errors_are_rejections_unless_the_provider_says_retry() {
        let rejected = classify(400, r#"{"error": {"message": "bad schema"}}"#, &[]);
        assert!(
            matches!(&rejected.error, LlmError::Rejected { status: 400, message, .. } if message == "bad schema")
        );
        assert!(classify(422, "", &[])
            .error
            .to_string()
            .contains("rejected"));
        assert!(classify(400, "", &[("x-should-retry", "true")]).retryable);
        assert!(!classify(500, "", &[("x-should-retry", "false")]).retryable);
    }
}
