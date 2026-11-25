//! Text embeddings, used to tell near-duplicate programs apart.
//!
//! Hosted models go through the OpenAI-compatible `/embeddings` endpoint or
//! Gemini's `embedContent`. `local-hash` (or `local-hash-<dim>`) needs no
//! network or key: it hashes character n-grams into a fixed-size vector,
//! which is enough to rank programs by textual similarity.

use crate::pricing::catalog;
use crate::providers::{parse_body, post_json, Billed, Failure};
use crate::types::{LlmError, Usage};
use genesis_core::CostKind;
use serde_json::{json, Value};
use std::time::Duration;

/// Longest prefix of a text that is embedded, in characters.
pub const MAX_EMBED_CHARS: usize = 10_000;
pub const LOCAL_HASH_DEFAULT_DIM: usize = 256;

pub(crate) fn clip(text: &str) -> &str {
    match text.char_indices().nth(MAX_EMBED_CHARS) {
        Some((offset, _)) => &text[..offset],
        None => text,
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A deterministic, L2-normalised embedding from feature-hashed character
/// 3-grams and 5-grams. Runs of whitespace count as one space, so
/// reformatting alone barely moves the vector. Text with no n-grams maps to
/// the zero vector.
pub fn local_hash_embedding(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut normalised: Vec<char> = Vec::with_capacity(text.len());
    for ch in clip(text).chars() {
        if ch.is_whitespace() {
            if normalised.last() != Some(&' ') {
                normalised.push(' ');
            }
        } else {
            normalised.push(ch);
        }
    }
    let mut vector = vec![0f64; dim];
    let mut buffer = String::new();
    for width in [3usize, 5] {
        for window in normalised.windows(width) {
            buffer.clear();
            buffer.extend(window.iter());
            let hash = fnv1a(buffer.as_bytes());
            let sign = if hash >> 63 == 0 { 1.0 } else { -1.0 };
            vector[(hash % dim as u64) as usize] += sign;
        }
    }
    let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return vec![0.0; dim];
    }
    vector.into_iter().map(|x| (x / norm) as f32).collect()
}

/// Cosine similarity; 0 when either vector is empty, zero, or of another length.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut norm_a, mut norm_b) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum EmbedRoute {
    LocalHash {
        dim: usize,
    },
    OpenAi {
        model: String,
    },
    Azure {
        deployment: String,
    },
    OpenRouter {
        model: String,
    },
    Local {
        model: String,
        base_url: String,
        api_key_env: Option<String>,
    },
    Gemini {
        model: String,
    },
}

pub(crate) fn resolve_embedding(name: &str) -> Result<EmbedRoute, LlmError> {
    let name = name.trim();
    if name == "local-hash" {
        return Ok(EmbedRoute::LocalHash {
            dim: LOCAL_HASH_DEFAULT_DIM,
        });
    }
    if let Some(dim) = name.strip_prefix("local-hash-") {
        return match dim.parse::<usize>() {
            Ok(dim) if (1..=65_536).contains(&dim) => Ok(EmbedRoute::LocalHash { dim }),
            _ => Err(LlmError::Config(format!(
                "`{name}`: the dimension must be between 1 and 65536"
            ))),
        };
    }
    if name.starts_with("local/") {
        return match crate::resolve::resolve_model(name)? {
            crate::resolve::Route::Local {
                model,
                base_url,
                api_key_env,
            } => Ok(EmbedRoute::Local {
                model,
                base_url,
                api_key_env,
            }),
            _ => unreachable!("a local/ name resolves to a local route"),
        };
    }
    let strip = |prefix: &str| {
        name.strip_prefix(prefix)
            .filter(|rest| !rest.is_empty())
            .map(str::to_string)
    };
    if let Some(model) = strip("openrouter/") {
        return Ok(EmbedRoute::OpenRouter { model });
    }
    if let Some(deployment) = strip("azure-") {
        return Ok(EmbedRoute::Azure { deployment });
    }
    if let Some(model) = strip("openai/") {
        return Ok(EmbedRoute::OpenAi { model });
    }
    if let Some(model) = strip("google/").or_else(|| strip("models/")) {
        return Ok(EmbedRoute::Gemini { model });
    }
    if name.starts_with("text-embedding-") {
        return Ok(EmbedRoute::OpenAi {
            model: name.to_string(),
        });
    }
    if name.starts_with("gemini-embedding") || name.starts_with("embedding-") {
        return Ok(EmbedRoute::Gemini {
            model: name.to_string(),
        });
    }
    Err(LlmError::Config(format!(
        "unknown embedding model `{name}`; accepted: local-hash[-<dim>], text-embedding-*, openai/<model>, \
         azure-<deployment>, gemini-embedding-* or google/<model>, openrouter/<id>, local/<model>@http(s)://host/v1"
    )))
}

pub(crate) struct Embedded {
    pub vector: Vec<f32>,
    pub billed: Billed,
}

fn vector_of(values: Option<&Value>, provider: &str) -> Result<Vec<f32>, Failure> {
    let vector: Vec<f32> = values
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_f64)
                .map(|x| x as f32)
                .collect()
        })
        .unwrap_or_default();
    if vector.is_empty() {
        return Err(Failure::transient(format!(
            "{provider} returned no embedding"
        )));
    }
    Ok(vector)
}

/// Where and how to reach an embedding API.
pub(crate) struct Endpoint<'a> {
    pub http: &'a reqwest::Client,
    pub provider: &'a str,
    pub base_url: &'a str,
    pub headers: Vec<(&'static str, String)>,
    pub timeout: Duration,
}

fn priced(names: &[&str], tokens: u64) -> Billed {
    let cost = names
        .iter()
        .find_map(|name| catalog().embedding_cost(name, tokens));
    let (cost_usd, cost_kind) = match cost {
        Some(cost) => (cost, CostKind::Estimated),
        None => (0.0, CostKind::UsageOnly),
    };
    Billed {
        usage: Usage {
            input_tokens: tokens,
            ..Usage::default()
        },
        cost_usd,
        cost_kind,
    }
}

/// One request to an OpenAI-compatible `/embeddings` endpoint.
pub(crate) async fn embed_openai(
    endpoint: &Endpoint<'_>,
    requested: &str,
    api_model: &str,
    text: &str,
) -> Result<Embedded, Failure> {
    let body = json!({"model": api_model, "input": [text], "encoding_format": "float"});
    let url = format!("{}/embeddings", endpoint.base_url);
    let reply = post_json(
        endpoint.http,
        endpoint.provider,
        &url,
        &endpoint.headers,
        &body,
        endpoint.timeout,
    )
    .await?;
    let reply = parse_body(endpoint.provider, &reply)?;
    let vector = vector_of(reply.pointer("/data/0/embedding"), endpoint.provider)?;
    let tokens = reply
        .pointer("/usage/total_tokens")
        .or_else(|| reply.pointer("/usage/prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(crate::providers::MAX_PLAUSIBLE_TOKENS);
    Ok(Embedded {
        vector,
        billed: priced(&[requested, api_model], tokens),
    })
}

/// One request to Gemini's `embedContent`. The API does not return a token
/// count, so usage is estimated at four characters per token.
pub(crate) async fn embed_gemini(
    endpoint: &Endpoint<'_>,
    api_model: &str,
    text: &str,
) -> Result<Embedded, Failure> {
    let body = json!({"content": {"parts": [{"text": text}]}});
    let url = format!("{}/models/{api_model}:embedContent", endpoint.base_url);
    let reply = post_json(
        endpoint.http,
        endpoint.provider,
        &url,
        &endpoint.headers,
        &body,
        endpoint.timeout,
    )
    .await?;
    let reply = parse_body(endpoint.provider, &reply)?;
    let vector = vector_of(reply.pointer("/embedding/values"), endpoint.provider)?;
    let tokens = text.chars().count().div_ceil(4) as u64;
    Ok(Embedded {
        vector,
        billed: priced(&[api_model], tokens),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_hash_is_normalised_deterministic_and_similarity_aware() {
        let a = local_hash_embedding("def area(r):\n    return 3.14159 * r * r\n", 256);
        let reformatted =
            local_hash_embedding("def area(r):\n\n        return 3.14159 * r * r", 256);
        let other = local_hash_embedding(
            "SELECT name FROM users WHERE id = 7 ORDER BY created_at",
            256,
        );
        assert_eq!(a.len(), 256);
        assert!((a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() - 1.0).abs() < 1e-6);
        assert_eq!(
            a,
            local_hash_embedding("def area(r):\n    return 3.14159 * r * r\n", 256)
        );
        assert!(cosine_similarity(&a, &a) > 0.999_999);
        assert!(cosine_similarity(&a, &reformatted) > 0.95);
        assert!(cosine_similarity(&a, &other) < 0.3);
    }

    #[test]
    fn degenerate_inputs_do_not_divide_by_zero() {
        assert_eq!(local_hash_embedding("", 8), vec![0.0; 8]);
        assert_eq!(local_hash_embedding("ab", 8), vec![0.0; 8]);
        assert_eq!(local_hash_embedding("abc", 0).len(), 1);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn only_the_first_ten_thousand_characters_count() {
        let long = "é".repeat(MAX_EMBED_CHARS + 50);
        assert_eq!(clip(&long).chars().count(), MAX_EMBED_CHARS);
        let tail_changed = format!("{}{}", "x".repeat(MAX_EMBED_CHARS), "tail one");
        let other_tail = format!(
            "{}{}",
            "x".repeat(MAX_EMBED_CHARS),
            "another ending entirely"
        );
        assert_eq!(
            local_hash_embedding(&tail_changed, 64),
            local_hash_embedding(&other_tail, 64)
        );
    }

    #[test]
    fn embedding_model_names_resolve() {
        assert_eq!(
            resolve_embedding("local-hash").unwrap(),
            EmbedRoute::LocalHash { dim: 256 }
        );
        assert_eq!(
            resolve_embedding("local-hash-64").unwrap(),
            EmbedRoute::LocalHash { dim: 64 }
        );
        assert_eq!(
            resolve_embedding("text-embedding-3-small").unwrap(),
            EmbedRoute::OpenAi {
                model: "text-embedding-3-small".into()
            }
        );
        assert_eq!(
            resolve_embedding("azure-text-embedding-3-large").unwrap(),
            EmbedRoute::Azure {
                deployment: "text-embedding-3-large".into()
            }
        );
        assert_eq!(
            resolve_embedding("gemini-embedding-001").unwrap(),
            EmbedRoute::Gemini {
                model: "gemini-embedding-001".into()
            }
        );
        assert_eq!(
            resolve_embedding("openrouter/qwen/qwen3-embedding").unwrap(),
            EmbedRoute::OpenRouter {
                model: "qwen/qwen3-embedding".into()
            }
        );
        assert!(matches!(
            resolve_embedding("local/bge@http://localhost:9/v1").unwrap(),
            EmbedRoute::Local { .. }
        ));
        for bad in ["local-hash-0", "local-hash-x", "bert", ""] {
            assert!(resolve_embedding(bad).is_err(), "{bad}");
        }
    }
}
