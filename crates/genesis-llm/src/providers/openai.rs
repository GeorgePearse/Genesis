//! The chat-completions wire format, as spoken by OpenAI, Azure OpenAI,
//! OpenRouter, DeepSeek and any local server that imitates it.

use super::{
    list_price, parse_body, post_json, u64_at, Backend, Billed, Call, Failure, TOKEN_LIMIT,
};
use crate::kwargs::enabled_effort;
use crate::pricing::catalog;
use crate::types::{LlmError, LlmRequest, LlmResponse, Role, Usage};
use async_trait::async_trait;
use genesis_core::CostKind;
use serde_json::{json, Map, Value};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flavour {
    OpenAi,
    Azure,
    OpenRouter,
    DeepSeek,
    Local,
}

impl Flavour {
    pub fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Azure => "azure_openai",
            Self::OpenRouter => "openrouter",
            Self::DeepSeek => "deepseek",
            Self::Local => "local_openai",
        }
    }
}

pub(crate) struct OpenAiCompat {
    pub http: reqwest::Client,
    pub flavour: Flavour,
    /// Ends before `/chat/completions`, without a trailing slash.
    pub base_url: String,
    pub api_key: Option<String>,
}

/// GPT-6 models take the full effort scale, `none` included.
const FULL_SCALE_MODELS: &[&str] = &["gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-luna"];
const NO_TEMPERATURE_MODELS: &[&str] = &["deepseek-v4-flash", "deepseek-v4-pro"];

struct Traits {
    is_reasoning: bool,
    think_temp_fixed: bool,
    requires_reasoning: bool,
}

fn looks_like_openai_reasoning(model: &str) -> bool {
    let mut chars = model.chars();
    let o_series = chars.next() == Some('o') && chars.next().is_some_and(|c| c.is_ascii_digit());
    o_series
        || ["gpt-5", "gpt-6", "gpt-7"]
            .iter()
            .any(|prefix| model.starts_with(prefix))
}

fn traits(flavour: Flavour, requested: &str, api_model: &str) -> Traits {
    let row = catalog()
        .lookup(requested)
        .or_else(|| catalog().lookup(api_model));
    match row {
        Some(row) => Traits {
            is_reasoning: row.is_reasoning,
            think_temp_fixed: row.think_temp_fixed,
            requires_reasoning: row.requires_reasoning,
        },
        None => Traits {
            is_reasoning: matches!(flavour, Flavour::OpenAi | Flavour::Azure)
                && looks_like_openai_reasoning(api_model),
            think_temp_fixed: false,
            requires_reasoning: false,
        },
    }
}

fn mentions_json(request: &LlmRequest) -> bool {
    request.system.to_ascii_lowercase().contains("json")
        || request
            .messages
            .iter()
            .any(|m| m.content.to_ascii_lowercase().contains("json"))
}

/// The request body for one call. Kept apart from the transport so the
/// provider-specific parameter rules can be tested without a server.
pub(crate) fn build_body(flavour: Flavour, request: &LlmRequest, api_model: &str) -> Value {
    let traits = traits(flavour, &request.model, api_model);
    let effort = enabled_effort(request.reasoning_effort.as_deref());
    let mut body = Map::new();
    body.insert("model".into(), json!(api_model));

    let mut messages = Vec::new();
    if !request.system.is_empty() {
        messages.push(json!({"role": "system", "content": request.system}));
    }
    for message in &request.messages {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        messages.push(json!({"role": role, "content": message.content}));
    }
    body.insert("messages".into(), Value::Array(messages));

    if let Some(max_tokens) = request.max_tokens {
        let key = match flavour {
            Flavour::OpenAi | Flavour::Azure => "max_completion_tokens",
            _ => "max_tokens",
        };
        body.insert(key.into(), json!(max_tokens));
    }

    let mut temperature = request.temperature;
    match flavour {
        Flavour::OpenAi | Flavour::Azure if traits.is_reasoning => {
            temperature = None;
            let full_scale = FULL_SCALE_MODELS.contains(&api_model);
            let level = match effort {
                None if full_scale && !traits.requires_reasoning => Some("none"),
                None if full_scale || traits.requires_reasoning => Some("low"),
                None => None,
                Some("minimal") => Some("low"),
                Some("max") if !full_scale => Some("high"),
                Some(other) => Some(other),
            };
            if let Some(level) = level {
                body.insert("reasoning_effort".into(), json!(level));
            }
        }
        Flavour::OpenRouter => {
            let level = match effort {
                None if traits.requires_reasoning => Some("low"),
                None => None,
                Some("max") => Some("xhigh"),
                Some(other) => Some(other),
            };
            if let Some(level) = level {
                body.insert("reasoning".into(), json!({"effort": level}));
                if traits.think_temp_fixed {
                    temperature = None;
                }
            }
            body.insert("usage".into(), json!({"include": true}));
        }
        Flavour::DeepSeek if traits.is_reasoning => {
            match effort {
                None => {
                    body.insert("thinking".into(), json!({"type": "disabled"}));
                }
                Some(level) => {
                    body.insert("thinking".into(), json!({"type": "enabled"}));
                    let level = if level == "max" { "max" } else { "high" };
                    body.insert("reasoning_effort".into(), json!(level));
                }
            }
            if NO_TEMPERATURE_MODELS.contains(&api_model) {
                temperature = None;
            }
        }
        _ => {}
    }
    if let Some(temperature) = temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    // OpenAI refuses JSON mode unless the conversation mentions JSON.
    if request.json_output && mentions_json(request) {
        body.insert("response_format".into(), json!({"type": "json_object"}));
    }
    Value::Object(body)
}

fn text_of(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

impl OpenAiCompat {
    fn headers(&self) -> Vec<(&'static str, String)> {
        let Some(key) = &self.api_key else {
            return Vec::new();
        };
        match self.flavour {
            Flavour::Azure => vec![("api-key", key.clone())],
            _ => vec![("authorization", format!("Bearer {key}"))],
        }
    }

    pub(crate) fn parse(
        &self,
        call: &Call<'_>,
        text: &str,
        latency_secs: f64,
    ) -> Result<LlmResponse, Failure> {
        let provider = self.flavour.name();
        let body = parse_body(provider, text)?;
        let choice = body.pointer("/choices/0").cloned().unwrap_or(Value::Null);
        let content = text_of(choice.pointer("/message/content"));
        let thought = text_of(
            choice
                .pointer("/message/reasoning_content")
                .or_else(|| choice.pointer("/message/reasoning")),
        );
        let incomplete_reason = match choice.get("finish_reason").and_then(Value::as_str) {
            Some("length") => Some(TOKEN_LIMIT.to_string()),
            Some("content_filter") => Some("content_filter".to_string()),
            _ => None,
        };

        let completion_tokens = u64_at(&body, "/usage/completion_tokens");
        let thinking_tokens = u64_at(&body, "/usage/completion_tokens_details/reasoning_tokens")
            .min(completion_tokens);
        let usage = Usage {
            input_tokens: u64_at(&body, "/usage/prompt_tokens"),
            output_tokens: completion_tokens - thinking_tokens,
            cached_input_tokens: u64_at(&body, "/usage/prompt_tokens_details/cached_tokens"),
            thinking_tokens,
        };
        let reported = body
            .pointer("/usage/cost")
            .and_then(Value::as_f64)
            .filter(|cost| cost.is_finite() && *cost >= 0.0);
        let (cost_usd, cost_kind) = match reported {
            Some(cost) => (cost, CostKind::Reported),
            None => list_price(&[&call.request.model, call.api_model], &usage),
        };

        if content.trim().is_empty() {
            return Err(Failure {
                retryable: !super::is_token_limit(incomplete_reason.as_deref()),
                error: LlmError::Empty {
                    provider: provider.to_string(),
                    incomplete_reason,
                    cost_usd,
                },
                retry_after: None,
                billed: Some(Box::new(Billed {
                    usage,
                    cost_usd,
                    cost_kind,
                })),
            });
        }
        Ok(LlmResponse {
            content,
            thought,
            model: call.request.model.clone(),
            provider: provider.to_string(),
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
impl Backend for OpenAiCompat {
    async fn attempt(&self, call: &Call<'_>) -> Result<LlmResponse, Failure> {
        let body = build_body(self.flavour, call.request, call.api_model);
        let url = format!("{}/chat/completions", self.base_url);
        let started = Instant::now();
        let text = post_json(
            &self.http,
            self.flavour.name(),
            &url,
            &self.headers(),
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

    fn body(flavour: Flavour, model: &str, api_model: &str, effort: Option<&str>) -> Value {
        let mut request = LlmRequest::new(model, "sys", "hi")
            .temperature(0.5)
            .max_tokens(100);
        request.reasoning_effort = effort.map(str::to_string);
        build_body(flavour, &request, api_model)
    }

    #[test]
    fn plain_chat_models_get_temperature_and_no_reasoning() {
        let b = body(
            Flavour::OpenAi,
            "gpt-4.1-mini",
            "gpt-4.1-mini",
            Some("high"),
        );
        assert_eq!(b["temperature"], json!(0.5));
        assert_eq!(b["max_completion_tokens"], json!(100));
        assert!(b.get("reasoning_effort").is_none() && b.get("max_tokens").is_none());
        assert_eq!(
            b["messages"][0],
            json!({"role": "system", "content": "sys"})
        );
        assert_eq!(b["messages"][1], json!({"role": "user", "content": "hi"}));
    }

    #[test]
    fn openai_reasoning_models_drop_temperature_and_map_effort() {
        let off = body(
            Flavour::OpenAi,
            "gpt-6-luna",
            "gpt-6-luna",
            Some("disabled"),
        );
        assert_eq!(off["reasoning_effort"], json!("none"));
        assert!(off.get("temperature").is_none());
        assert_eq!(
            body(Flavour::OpenAi, "gpt-6-astra", "gpt-6-astra", None)["reasoning_effort"],
            json!("low")
        );
        assert_eq!(
            body(
                Flavour::OpenAi,
                "gpt-6.1-terra",
                "gpt-6.1-terra",
                Some("max")
            )["reasoning_effort"],
            json!("max")
        );
        assert_eq!(
            body(
                Flavour::OpenAi,
                "gpt-6.1-terra",
                "gpt-6.1-terra",
                Some("min")
            )["reasoning_effort"],
            json!("low")
        );
        // gpt-5 requires reasoning; o4-mini does not and keeps its default when unset.
        assert_eq!(
            body(Flavour::OpenAi, "gpt-5", "gpt-5", None)["reasoning_effort"],
            json!("low")
        );
        assert!(body(Flavour::OpenAi, "o4-mini", "o4-mini", None)
            .get("reasoning_effort")
            .is_none());
        assert_eq!(
            body(Flavour::OpenAi, "o4-mini", "o4-mini", Some("max"))["reasoning_effort"],
            json!("high")
        );
        assert_eq!(
            body(Flavour::Azure, "azure-gpt-5", "gpt-5", Some("medium"))["reasoning_effort"],
            json!("medium")
        );
        // A model the catalog has never heard of is recognised by its name.
        assert!(
            body(Flavour::OpenAi, "gpt-7-nova", "gpt-7-nova", Some("high"))
                .get("temperature")
                .is_none()
        );
    }

    #[test]
    fn deepseek_openrouter_and_local_use_their_own_parameters() {
        let ds_off = body(
            Flavour::DeepSeek,
            "deepseek-v4-pro",
            "deepseek-v4-pro",
            None,
        );
        assert_eq!(ds_off["thinking"], json!({"type": "disabled"}));
        assert!(ds_off.get("temperature").is_none());
        assert_eq!(ds_off["max_tokens"], json!(100));
        let ds_on = body(
            Flavour::DeepSeek,
            "deepseek-reasoner",
            "deepseek-reasoner",
            Some("max"),
        );
        assert_eq!(
            (&ds_on["thinking"], &ds_on["reasoning_effort"]),
            (&json!({"type": "enabled"}), &json!("max"))
        );
        assert_eq!(ds_on["temperature"], json!(0.5));
        assert!(body(
            Flavour::DeepSeek,
            "deepseek-chat",
            "deepseek-chat",
            Some("high")
        )
        .get("thinking")
        .is_none());

        let or = body(
            Flavour::OpenRouter,
            "openrouter/qwen/qwen3-32b",
            "qwen/qwen3-32b",
            Some("max"),
        );
        assert_eq!(or["reasoning"], json!({"effort": "xhigh"}));
        assert_eq!(or["usage"], json!({"include": true}));
        assert!(
            or.get("temperature").is_none(),
            "this model fixes its temperature while reasoning"
        );
        let or_forced = body(
            Flavour::OpenRouter,
            "openrouter/x-ai/grok-4",
            "x-ai/grok-4",
            None,
        );
        assert_eq!(or_forced["reasoning"], json!({"effort": "low"}));
        let or_plain = body(
            Flavour::OpenRouter,
            "openrouter/qwen/qwen3-coder",
            "qwen/qwen3-coder",
            None,
        );
        assert!(or_plain.get("reasoning").is_none());
        assert_eq!(or_plain["temperature"], json!(0.5));

        let local = body(Flavour::Local, "local/phi@http://x/v1", "phi", Some("high"));
        assert_eq!(local["max_tokens"], json!(100));
        assert!(local.get("reasoning").is_none() && local.get("reasoning_effort").is_none());
    }

    #[test]
    fn json_mode_is_requested_only_when_the_prompt_mentions_json() {
        let with = build_body(
            Flavour::OpenAi,
            &LlmRequest::new("gpt-4.1", "", "Reply in JSON.").json(),
            "gpt-4.1",
        );
        assert_eq!(with["response_format"], json!({"type": "json_object"}));
        assert_eq!(
            with["messages"].as_array().unwrap().len(),
            1,
            "an empty system prompt is not sent"
        );
        let without = build_body(
            Flavour::OpenAi,
            &LlmRequest::new("gpt-4.1", "", "Reply.").json(),
            "gpt-4.1",
        );
        assert!(without.get("response_format").is_none());
    }
}
