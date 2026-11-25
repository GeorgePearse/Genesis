//! Google's Gemini API (`generateContent`) with an API key.

use super::{
    list_price, parse_body, post_json, u64_at, Backend, Billed, Call, Failure, TOKEN_LIMIT,
};
use crate::kwargs::{enabled_effort, thinking_budget};
use crate::types::{LlmError, LlmRequest, LlmResponse, Role, Usage};
use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::time::Instant;

pub(crate) const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

pub(crate) struct Gemini {
    pub http: reqwest::Client,
    pub base_url: String,
    pub api_key: String,
}

/// Models steered by a thinking level rather than a token budget.
const THINKING_LEVEL_MODELS: &[&str] = &[
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-4-argon",
    "gemini-4-hydrogen",
];
/// Models that cannot think with a zero budget.
const MINIMUM_BUDGET_MODELS: &[&str] = &["gemini-2.5-pro", "gemini-3-pro-preview"];

pub(crate) fn build_body(request: &LlmRequest, api_model: &str) -> Value {
    let effort = enabled_effort(request.reasoning_effort.as_deref());
    let max_tokens = request.max_tokens.unwrap_or(2048);
    let mut config = Map::new();
    config.insert("maxOutputTokens".into(), json!(max_tokens));

    let mut thinking = Map::new();
    thinking.insert("includeThoughts".into(), json!(true));
    if THINKING_LEVEL_MODELS.contains(&api_model) {
        let level = match effort {
            None => None,
            Some("minimal") if api_model == "gemini-3.5-flash-lite" => Some("minimal"),
            Some("minimal" | "low") => Some("low"),
            Some("medium") => Some("medium"),
            Some(_) => Some("high"),
        };
        if let Some(level) = level {
            thinking.insert("thinkingLevel".into(), json!(level));
        }
    } else {
        if let Some(temperature) = request.temperature {
            config.insert("temperature".into(), json!(temperature));
        }
        let budget = match effort {
            None if MINIMUM_BUDGET_MODELS.contains(&api_model) => 128,
            None => 0,
            Some(effort) => {
                let wanted = thinking_budget(effort);
                if wanted < max_tokens {
                    wanted
                } else {
                    1024
                }
            }
        };
        thinking.insert("thinkingBudget".into(), json!(budget));
    }
    config.insert("thinkingConfig".into(), Value::Object(thinking));
    if request.json_output {
        config.insert("responseMimeType".into(), json!("application/json"));
    }

    let contents: Vec<Value> = request
        .messages
        .iter()
        .map(|message| {
            let role = match message.role {
                Role::User => "user",
                Role::Assistant => "model",
            };
            json!({"role": role, "parts": [{"text": message.content}]})
        })
        .collect();
    let mut body = Map::new();
    if !request.system.is_empty() {
        body.insert(
            "systemInstruction".into(),
            json!({"parts": [{"text": request.system}]}),
        );
    }
    body.insert("contents".into(), Value::Array(contents));
    body.insert("generationConfig".into(), Value::Object(config));
    Value::Object(body)
}

impl Gemini {
    pub(crate) fn parse(
        &self,
        call: &Call<'_>,
        text: &str,
        latency_secs: f64,
    ) -> Result<LlmResponse, Failure> {
        let body = parse_body("google", text)?;
        let (mut content, mut thought) = (String::new(), String::new());
        if let Some(parts) = body
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
        {
            for part in parts {
                let Some(piece) = part.get("text").and_then(Value::as_str) else {
                    continue;
                };
                if part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                {
                    thought.push_str(piece);
                } else {
                    content.push_str(piece);
                }
            }
        }
        let finish = body
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str);
        let blocked = body
            .pointer("/promptFeedback/blockReason")
            .and_then(Value::as_str);
        let incomplete_reason = match (finish, blocked) {
            (Some("MAX_TOKENS"), _) => Some(TOKEN_LIMIT.to_string()),
            (Some("STOP") | None, None) => None,
            (Some(other), _) => Some(other.to_ascii_lowercase()),
            (None, Some(reason)) => Some(format!("blocked_{}", reason.to_ascii_lowercase())),
        };
        let usage = Usage {
            input_tokens: u64_at(&body, "/usageMetadata/promptTokenCount"),
            output_tokens: u64_at(&body, "/usageMetadata/candidatesTokenCount"),
            cached_input_tokens: u64_at(&body, "/usageMetadata/cachedContentTokenCount"),
            thinking_tokens: u64_at(&body, "/usageMetadata/thoughtsTokenCount"),
        };
        let (cost_usd, cost_kind) = list_price(&[&call.request.model, call.api_model], &usage);
        let provider = "google".to_string();
        if content.trim().is_empty() {
            return Err(Failure {
                retryable: !super::is_token_limit(incomplete_reason.as_deref()),
                error: LlmError::Empty {
                    provider,
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
impl Backend for Gemini {
    async fn attempt(&self, call: &Call<'_>) -> Result<LlmResponse, Failure> {
        let body = build_body(call.request, call.api_model);
        let url = format!(
            "{}/models/{}:generateContent",
            self.base_url, call.api_model
        );
        let headers = [("x-goog-api-key", self.api_key.clone())];
        let started = Instant::now();
        let text = post_json(&self.http, "google", &url, &headers, &body, call.timeout).await?;
        self.parse(call, &text, started.elapsed().as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(model: &str, effort: Option<&str>) -> Value {
        let mut request = LlmRequest::new(model, "sys", "hi")
            .temperature(0.4)
            .max_tokens(4000);
        request
            .messages
            .push(crate::types::Message::assistant("earlier"));
        request.messages.push(crate::types::Message::user("again"));
        request.reasoning_effort = effort.map(str::to_string);
        build_body(&request, model)
    }

    #[test]
    fn thinking_level_models_take_a_level_and_no_temperature() {
        let b = body("gemini-3.8-flash", Some("xhigh"));
        assert_eq!(
            b["generationConfig"]["thinkingConfig"],
            json!({"includeThoughts": true, "thinkingLevel": "high"})
        );
        assert!(b["generationConfig"].get("temperature").is_none());
        let off = body("gemini-4-argon", None);
        assert_eq!(
            off["generationConfig"]["thinkingConfig"],
            json!({"includeThoughts": true})
        );
        let lite = body("gemini-3.5-flash-lite", Some("min"));
        assert_eq!(
            lite["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            json!("minimal")
        );
        assert_eq!(
            body("gemini-3.7-flash", Some("min"))["generationConfig"]["thinkingConfig"]
                ["thinkingLevel"],
            json!("low")
        );
    }

    #[test]
    fn budget_models_take_a_budget_and_a_temperature() {
        let b = body("gemini-2.5-flash", Some("low"));
        assert_eq!(
            b["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            json!(2048)
        );
        assert_eq!(b["generationConfig"]["temperature"], json!(0.4));
        assert_eq!(
            body("gemini-2.5-flash", None)["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            json!(0)
        );
        assert_eq!(
            body("gemini-2.5-pro", None)["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            json!(128)
        );
        assert_eq!(
            body("gemini-3.5-flash", Some("max"))["generationConfig"]["thinkingConfig"]
                ["thinkingBudget"],
            json!(1024)
        );
    }

    #[test]
    fn roles_system_instruction_and_json_mode_are_mapped() {
        let b = body("gemini-2.5-flash", None);
        assert_eq!(b["systemInstruction"], json!({"parts": [{"text": "sys"}]}));
        let roles: Vec<&str> = b["contents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user", "model", "user"]);
        assert_eq!(b["generationConfig"]["maxOutputTokens"], json!(4000));
        let json_request = LlmRequest::new("gemini-2.5-flash", "", "x").json();
        let j = build_body(&json_request, "gemini-2.5-flash");
        assert_eq!(
            j["generationConfig"]["responseMimeType"],
            json!("application/json")
        );
        assert!(j.get("systemInstruction").is_none());
    }
}
