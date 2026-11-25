//! The HTTP backends against a local server that replays scripted replies
//! and records what it was sent.

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response, Uri};
use axum::Router;
use genesis_core::{CostKind, CostLedger};
use genesis_llm::{ClientOptions, LlmClient, LlmError, LlmRequest};
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Reply {
    status: u16,
    headers: Vec<(&'static str, &'static str)>,
    body: String,
    delay_ms: u64,
}

fn reply(status: u16, body: Value) -> Reply {
    Reply {
        status,
        headers: Vec::new(),
        body: body.to_string(),
        delay_ms: 0,
    }
}

struct Seen {
    path: String,
    headers: HeaderMap,
    body: Value,
}

#[derive(Default)]
struct Script {
    replies: Mutex<VecDeque<Reply>>,
    seen: Mutex<Vec<Seen>>,
}

struct Server {
    url: String,
    script: Arc<Script>,
}

impl Server {
    async fn start(replies: Vec<Reply>) -> Self {
        let script = Arc::new(Script {
            replies: Mutex::new(replies.into()),
            seen: Mutex::new(Vec::new()),
        });
        let app = Router::new()
            .fallback(handle)
            .with_state(Arc::clone(&script));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self { url, script }
    }

    fn requests(&self) -> usize {
        self.script.seen.lock().unwrap().len()
    }

    fn body(&self, index: usize) -> Value {
        self.script.seen.lock().unwrap()[index].body.clone()
    }

    fn header(&self, index: usize, name: &str) -> Option<String> {
        self.script.seen.lock().unwrap()[index]
            .headers
            .get(name)
            .map(|v| v.to_str().unwrap().to_string())
    }

    fn path(&self, index: usize) -> String {
        self.script.seen.lock().unwrap()[index].path.clone()
    }

    /// A client whose `provider` talks to this server, retrying without delay.
    fn client(&self, provider: &str, options: ClientOptions) -> LlmClient {
        LlmClient::new(ClientOptions {
            base_urls: BTreeMap::from([(provider.to_string(), self.url.clone())]),
            api_keys: BTreeMap::from([(provider.to_string(), "test-key".to_string())]),
            backoff_base_ms: 1,
            ..options
        })
    }
}

async fn handle(
    State(script): State<Arc<Script>>,
    uri: Uri,
    headers: HeaderMap,
    body: String,
) -> Response<Body> {
    script.seen.lock().unwrap().push(Seen {
        path: uri.path().to_string(),
        headers,
        body: serde_json::from_str(&body).unwrap_or(Value::Null),
    });
    let next = script.replies.lock().unwrap().pop_front();
    let Some(next) = next else {
        return Response::builder()
            .status(500)
            .body(Body::from("the test scripted no further reply"))
            .unwrap();
    };
    if next.delay_ms > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(next.delay_ms)).await;
    }
    let mut response = Response::builder().status(next.status);
    for (name, value) in &next.headers {
        response = response.header(*name, *value);
    }
    response.body(Body::from(next.body)).unwrap()
}

fn chat(content: &str, prompt_tokens: u64, completion_tokens: u64) -> Value {
    json!({
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    })
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
}

#[tokio::test]
async fn openai_request_shape_usage_cost_and_ledger() {
    let server = Server::start(vec![reply(
        200,
        json!({
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500,
                      "prompt_tokens_details": {"cached_tokens": 200},
                      "completion_tokens_details": {"reasoning_tokens": 100}}
        }),
    )])
    .await;
    let dir = tempfile::tempdir().unwrap();
    let ledger = Arc::new(CostLedger::open(dir.path()).unwrap());
    let client = server.client(
        "openai",
        ClientOptions {
            ledger: Some(Arc::clone(&ledger)),
            ..ClientOptions::default()
        },
    );
    let request = LlmRequest::new("gpt-4.1-mini", "be brief", "hi")
        .temperature(0.25)
        .max_tokens(64);
    let response = client.query("proposal", request).await.unwrap();

    assert_eq!(server.path(0), "/chat/completions");
    assert_eq!(
        server.header(0, "authorization").as_deref(),
        Some("Bearer test-key")
    );
    let sent = server.body(0);
    assert_eq!(sent["model"], json!("gpt-4.1-mini"));
    assert_eq!(
        sent["messages"],
        json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
    );
    assert_eq!(
        (&sent["temperature"], &sent["max_completion_tokens"]),
        (&json!(0.25), &json!(64))
    );

    assert_eq!(
        (
            response.content.as_str(),
            response.provider.as_str(),
            response.calls
        ),
        ("hello", "openai", 1)
    );
    assert_eq!(response.model, "gpt-4.1-mini");
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cached_input_tokens,
            response.usage.thinking_tokens
        ),
        (1000, 400, 200, 100)
    );
    // gpt-4.1-mini lists at $0.40 in and $1.60 out per million tokens.
    let expected = 1000.0 * 0.4e-6 + 500.0 * 1.6e-6;
    assert!(close(response.cost_usd, expected) && response.cost_kind == CostKind::Estimated);
    assert!(close(client.spent_usd(), expected));

    let summary = ledger.summary();
    assert!(close(summary.estimated_cost_usd, expected));
    assert_eq!(summary.by_stage["investigation/proposal"].calls, 1);
    assert_eq!(summary.by_source["llm_api"].input_tokens, 1000);
    assert_eq!(summary.by_model["gpt-4.1-mini"].output_tokens, 400);
    assert_eq!(summary.by_model["gpt-4.1-mini"].thinking_tokens, 100);
}

#[tokio::test]
async fn rate_limits_and_server_errors_are_retried_then_succeed() {
    let mut limited = reply(
        429,
        json!({"error": {"message": "slow down", "type": "rate_limit_exceeded"}}),
    );
    limited.headers = vec![("retry-after-ms", "5")];
    let server = Server::start(vec![
        limited,
        reply(500, json!({"error": {"message": "oops"}})),
        reply(200, chat("ok", 10, 2)),
    ])
    .await;
    let client = server.client("openai", ClientOptions::default());
    let response = client
        .query("proposal", LlmRequest::new("gpt-4.1", "", "hi"))
        .await
        .unwrap();
    assert_eq!((response.content.as_str(), response.calls), ("ok", 1));
    assert_eq!(server.requests(), 3);
}

#[tokio::test]
async fn the_retry_budget_is_finite() {
    let down = reply(503, json!({"error": {"message": "unavailable"}}));
    let server = Server::start(vec![down.clone(), down.clone(), down]).await;
    let client = server.client(
        "openai",
        ClientOptions {
            max_retries: 2,
            ..ClientOptions::default()
        },
    );
    let error = client
        .query("proposal", LlmRequest::new("gpt-4.1", "", "hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, LlmError::Transient(message) if message.contains("503") && message.contains("unavailable")),
        "{error}"
    );
    assert_eq!(server.requests(), 3);
}

#[tokio::test]
async fn bad_credentials_and_bad_requests_are_not_retried() {
    let server = Server::start(vec![
        reply(401, json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}})),
        reply(400, json!({"error": {"message": "max_tokens is too large", "type": "invalid_request_error"}})),
    ])
    .await;
    let client = server.client("openai", ClientOptions::default());
    let unauthorised = client
        .query("proposal", LlmRequest::new("gpt-4.1", "", "hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(&unauthorised, LlmError::Endpoint { status: 401, provider, .. } if provider == "openai"),
        "{unauthorised}"
    );
    assert!(unauthorised.is_endpoint());
    let rejected = client
        .query("proposal", LlmRequest::new("gpt-4.1", "", "hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(&rejected, LlmError::Rejected { status: 400, message, .. } if message == "max_tokens is too large"),
        "{rejected}"
    );
    assert_eq!(server.requests(), 2);
    assert_eq!(client.spent_usd(), 0.0);
}

#[tokio::test]
async fn an_empty_reply_is_billed_retried_and_added_to_the_success() {
    let server = Server::start(vec![
        reply(200, chat("", 1000, 0)),
        reply(200, chat("done", 1000, 10)),
    ])
    .await;
    let client = server.client("openai", ClientOptions::default());
    let response = client
        .query("proposal", LlmRequest::new("gpt-4.1-mini", "", "hi"))
        .await
        .unwrap();
    assert_eq!((response.content.as_str(), response.calls), ("done", 2));
    assert_eq!(response.usage.input_tokens, 2000);
    let expected = 2000.0 * 0.4e-6 + 10.0 * 1.6e-6;
    assert!(close(response.cost_usd, expected) && close(client.spent_usd(), expected));
}

#[tokio::test]
async fn a_reply_cut_off_by_the_token_limit_is_not_retried() {
    let truncated = json!({
        "choices": [{"message": {"content": null}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 50}}
    });
    let server = Server::start(vec![reply(200, truncated)]).await;
    let client = server.client("openai", ClientOptions::default());
    let error = client
        .query("proposal", LlmRequest::new("gpt-5", "", "hi"))
        .await
        .unwrap_err();
    match error {
        LlmError::Empty {
            incomplete_reason,
            cost_usd,
            ..
        } => {
            assert_eq!(incomplete_reason.as_deref(), Some("max_output_tokens"));
            assert!(cost_usd > 0.0);
        }
        other => panic!("{other}"),
    }
    assert_eq!(server.requests(), 1);
    assert!(client.spent_usd() > 0.0, "the empty reply was still billed");

    let always_empty = Server::start(vec![reply(200, chat(" ", 10, 0)); 4]).await;
    let client = always_empty.client(
        "openai",
        ClientOptions {
            max_retries: 3,
            ..ClientOptions::default()
        },
    );
    assert!(matches!(
        client
            .query("p", LlmRequest::new("gpt-4.1", "", "hi"))
            .await,
        Err(LlmError::Empty { .. })
    ));
    assert_eq!(always_empty.requests(), 4);
}

#[tokio::test]
async fn the_budget_refuses_calls_before_they_are_sent() {
    let server = Server::start(vec![
        reply(200, chat("first", 1_000_000, 0)),
        reply(200, chat("second", 1, 1)),
    ])
    .await;
    let client = server.client(
        "openai",
        ClientOptions {
            budget_usd: Some(0.3),
            ..ClientOptions::default()
        },
    );
    assert!(!client.budget_exhausted());
    let first = client
        .query("proposal", LlmRequest::new("gpt-4.1-mini", "", "hi"))
        .await
        .unwrap();
    assert!(close(first.cost_usd, 0.4));
    assert!(client.budget_exhausted());
    let other_phase = client.with_phase("paper");
    let refused = other_phase
        .query("draft", LlmRequest::new("gpt-4.1-mini", "", "hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(refused, LlmError::BudgetExceeded { spent, budget } if close(spent, 0.4) && close(budget, 0.3))
    );
    assert!(refused.is_budget());
    assert_eq!(
        server.requests(),
        1,
        "a refused call never reaches the provider"
    );
    assert!(matches!(
        client.embed("novelty", "text-embedding-3-small", "x").await,
        Err(LlmError::BudgetExceeded { .. })
    ));
    assert!(
        client
            .embed("novelty", "local-hash", "fn main() {}")
            .await
            .is_ok(),
        "offline embeddings cost nothing"
    );
}

#[tokio::test]
async fn query_json_feeds_a_parse_failure_back_once() {
    let server = Server::start(vec![
        reply(200, chat("Sure! The answer is forty-two.", 100, 10)),
        reply(200, chat("```json\n{\"answer\": 42,}\n```", 150, 10)),
        reply(200, chat("no json", 10, 1)),
        reply(200, chat("still no json", 10, 1)),
    ])
    .await;
    let client = server.client("openai", ClientOptions::default());
    let request = LlmRequest::new("gpt-4.1-mini", "Reply in JSON.", "What is six times seven?");
    let (value, response) = client.query_json("idea", request.clone()).await.unwrap();
    assert_eq!(value, json!({"answer": 42}));
    assert_eq!((response.calls, response.usage.input_tokens), (2, 250));
    assert!(close(response.cost_usd, 250.0 * 0.4e-6 + 20.0 * 1.6e-6));

    assert_eq!(
        server.body(0)["response_format"],
        json!({"type": "json_object"})
    );
    let retry = server.body(1)["messages"].as_array().unwrap().clone();
    assert_eq!(retry.len(), 4);
    assert_eq!(
        retry[2],
        json!({"role": "assistant", "content": "Sure! The answer is forty-two."})
    );
    assert!(retry[3]["content"]
        .as_str()
        .unwrap()
        .contains("could not be read as JSON"));

    let failed = client.query_json("idea", request).await.unwrap_err();
    assert!(
        matches!(&failed, LlmError::Empty { incomplete_reason: Some(reason), cost_usd, .. } if reason == "invalid_json" && *cost_usd > 0.0),
        "{failed}"
    );
    assert_eq!(server.requests(), 4);
}

fn sse(events: &[Value]) -> Reply {
    let body: String = events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect();
    Reply {
        status: 200,
        headers: vec![("content-type", "text/event-stream")],
        body,
        delay_ms: 0,
    }
}

fn anthropic_stream(text: &str, stop_reason: &str) -> Reply {
    sse(&[
        json!({"type": "message_start", "message": {"model": "claude-opus-5-5", "usage": {"input_tokens": 90, "cache_read_input_tokens": 10, "cache_creation_input_tokens": 0, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "weighing options"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text}}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 50, "output_tokens_details": {"thinking_tokens": 20}}}),
        json!({"type": "message_stop"}),
    ])
}

#[tokio::test]
async fn anthropic_streams_with_current_headers_and_thinking_controls() {
    let server = Server::start(vec![anthropic_stream("Bonjour", "end_turn")]).await;
    let client = server.client("anthropic", ClientOptions::default());
    let request = LlmRequest::new("claude-opus-5-5", "be brief", "hi")
        .temperature(0.9)
        .max_tokens(2000)
        .reasoning_effort("high");
    let response = client.query("proposal", request).await.unwrap();

    assert_eq!(server.path(0), "/v1/messages");
    assert_eq!(server.header(0, "x-api-key").as_deref(), Some("test-key"));
    assert_eq!(
        server.header(0, "anthropic-version").as_deref(),
        Some("2026-10-21")
    );
    assert_eq!(
        server.header(0, "anthropic-beta"),
        None,
        "a non-default endpoint gets no beta features"
    );
    let sent = server.body(0);
    assert_eq!(sent["stream"], json!(true));
    assert_eq!(sent["system"], json!("be brief"));
    assert_eq!(sent["messages"], json!([{"role": "user", "content": "hi"}]));
    assert_eq!(
        sent["thinking"],
        json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(sent["output_config"], json!({"effort": "high"}));
    assert_eq!(sent["max_tokens"], json!(2000));
    assert!(sent.get("temperature").is_none() && sent.get("fallbacks").is_none());

    assert_eq!(
        (response.content.as_str(), response.thought.as_str()),
        ("Bonjour", "weighing options")
    );
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.cached_input_tokens,
            response.usage.output_tokens,
            response.usage.thinking_tokens
        ),
        (100, 10, 30, 20)
    );
    // claude-opus-5-5 lists at $4 in and $20 out per million tokens.
    assert!(close(response.cost_usd, 100.0 * 4.0e-6 + 50.0 * 20.0e-6));
    assert_eq!(response.incomplete_reason, None);
}

#[tokio::test]
async fn anthropic_mid_stream_errors_refusals_and_truncation() {
    let overloaded = sse(&[
        json!({"type": "message_start", "message": {"model": "claude-haiku-5-5", "usage": {"input_tokens": 5, "output_tokens": 0}}}),
        json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
    ]);
    let refusal = sse(&[
        json!({"type": "message_start", "message": {"model": "claude-haiku-5-5", "usage": {"input_tokens": 5, "output_tokens": 0}}}),
        json!({"type": "message_delta", "delta": {"stop_reason": "refusal", "stop_details": {"type": "refusal", "category": "cyber"}}, "usage": {"output_tokens": 0}}),
        json!({"type": "message_stop"}),
    ]);
    let server = Server::start(vec![
        overloaded,
        anthropic_stream("recovered", "end_turn"),
        refusal,
        anthropic_stream("partial answ", "max_tokens"),
    ])
    .await;
    let client = server.client("anthropic", ClientOptions::default());
    let request = LlmRequest::new("claude-haiku-5-5", "", "hi");
    let recovered = client.query("proposal", request.clone()).await.unwrap();
    assert_eq!(
        (recovered.content.as_str(), recovered.calls),
        ("recovered", 2)
    );
    assert_eq!(server.requests(), 2);

    let refused = client.query("proposal", request.clone()).await.unwrap_err();
    assert!(
        matches!(&refused, LlmError::Rejected { message, .. } if message.contains("category cyber")),
        "{refused}"
    );
    assert_eq!(server.requests(), 3, "a refusal is not retried");

    let truncated = client.query("proposal", request).await.unwrap();
    assert_eq!(truncated.content, "partial answ");
    assert_eq!(
        truncated.incomplete_reason.as_deref(),
        Some("max_output_tokens")
    );
    assert_eq!(
        server.body(3)["thinking"],
        json!({"type": "disabled"}),
        "haiku 5.5 with no effort asked for"
    );
}

#[tokio::test]
async fn gemini_request_shape_thoughts_and_usage() {
    let server = Server::start(vec![reply(
        200,
        json!({
            "candidates": [{"content": {"parts": [{"text": "planning", "thought": true}, {"text": "Hallo"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 40, "thoughtsTokenCount": 60, "cachedContentTokenCount": 20}
        }),
    )])
    .await;
    let client = server.client("google", ClientOptions::default());
    let request = LlmRequest::new("gemini-2.5-flash", "be brief", "hi")
        .temperature(0.3)
        .max_tokens(9000)
        .reasoning_effort("medium");
    let response = client.query("proposal", request).await.unwrap();

    assert_eq!(server.path(0), "/models/gemini-2.5-flash:generateContent");
    assert_eq!(
        server.header(0, "x-goog-api-key").as_deref(),
        Some("test-key")
    );
    let sent = server.body(0);
    assert_eq!(
        sent["systemInstruction"],
        json!({"parts": [{"text": "be brief"}]})
    );
    assert_eq!(
        sent["generationConfig"]["thinkingConfig"],
        json!({"includeThoughts": true, "thinkingBudget": 4096})
    );
    assert_eq!(
        (
            response.content.as_str(),
            response.thought.as_str(),
            response.provider.as_str()
        ),
        ("Hallo", "planning", "google")
    );
    assert_eq!(
        (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.thinking_tokens
        ),
        (200, 40, 60)
    );
    // gemini-2.5-flash lists at $0.30 in and $2.50 out per million; thoughts bill as output.
    assert!(close(response.cost_usd, 200.0 * 0.3e-6 + 100.0 * 2.5e-6));
}

#[tokio::test]
async fn openrouter_reports_its_own_charge_and_azure_and_local_authenticate_their_way() {
    let mut priced = chat("via router", 10, 5);
    priced["usage"]["cost"] = json!(0.0123);
    let router = Server::start(vec![reply(200, priced)]).await;
    let client = router.client("openrouter", ClientOptions::default());
    let response = client
        .query(
            "proposal",
            LlmRequest::new("openrouter/qwen/qwen3-coder", "", "hi"),
        )
        .await
        .unwrap();
    assert_eq!(
        (response.cost_usd, response.cost_kind),
        (0.0123, CostKind::Reported)
    );
    assert_eq!(router.body(0)["model"], json!("qwen/qwen3-coder"));
    assert_eq!(router.body(0)["usage"], json!({"include": true}));
    assert_eq!(response.model, "openrouter/qwen/qwen3-coder");

    let azure = Server::start(vec![reply(200, chat("via azure", 10, 5))]).await;
    let client = azure.client("azure_openai", ClientOptions::default());
    client
        .query("proposal", LlmRequest::new("azure-gpt-4.1", "", "hi"))
        .await
        .unwrap();
    assert_eq!(azure.header(0, "api-key").as_deref(), Some("test-key"));
    assert_eq!(azure.body(0)["model"], json!("gpt-4.1"));

    let local = Server::start(vec![reply(
        200,
        json!({"choices": [{"message": {"content": "local reply"}}]}),
    )])
    .await;
    let client = LlmClient::new(ClientOptions::default());
    let model = format!("local/tiny@{}/v1", local.url);
    let response = client
        .query("proposal", LlmRequest::new(model, "", "hi").max_tokens(8))
        .await
        .unwrap();
    assert_eq!(local.path(0), "/v1/chat/completions");
    assert_eq!(local.body(0)["model"], json!("tiny"));
    assert_eq!(local.body(0)["max_tokens"], json!(8));
    assert_eq!(
        (
            response.cost_usd,
            response.cost_kind,
            response.usage.input_tokens
        ),
        (0.0, CostKind::UsageOnly, 0)
    );
}

#[tokio::test]
async fn embeddings_are_fetched_priced_and_logged() {
    let server = Server::start(vec![
        reply(
            200,
            json!({"data": [{"embedding": [0.6, -0.8]}], "usage": {"total_tokens": 1000}}),
        ),
        reply(429, json!({"error": {"message": "slow down"}})),
        reply(200, json!({"embedding": {"values": [1.0, 0.0, 0.0]}})),
    ])
    .await;
    let dir = tempfile::tempdir().unwrap();
    let ledger = Arc::new(CostLedger::open(dir.path()).unwrap());
    let openai = server.client(
        "openai",
        ClientOptions {
            ledger: Some(Arc::clone(&ledger)),
            ..ClientOptions::default()
        },
    );
    let long = "x".repeat(20_000);
    assert_eq!(
        openai
            .embed("novelty", "text-embedding-3-small", &long)
            .await
            .unwrap(),
        [0.6, -0.8]
    );
    assert_eq!(server.path(0), "/embeddings");
    assert_eq!(server.body(0)["input"][0].as_str().unwrap().len(), 10_000);
    assert!(close(openai.spent_usd(), 1000.0 * 0.02e-6));
    assert_eq!(ledger.summary().by_source["embedding"].input_tokens, 1000);

    let google = server.client("google", ClientOptions::default());
    assert_eq!(
        google
            .embed("novelty", "gemini-embedding-001", "abcdefgh")
            .await
            .unwrap(),
        [1.0, 0.0, 0.0]
    );
    assert_eq!(
        server.requests(),
        3,
        "the rate-limited embedding call was retried"
    );
    assert_eq!(server.path(2), "/models/gemini-embedding-001:embedContent");
    assert!(close(google.spent_usd(), 2.0 * 0.15e-6));
}

#[tokio::test]
async fn a_request_that_outlives_the_timeout_is_a_transient_failure() {
    let mut slow = reply(200, chat("too late", 1, 1));
    slow.delay_ms = 3000;
    let server = Server::start(vec![slow]).await;
    let client = server.client(
        "openai",
        ClientOptions {
            timeout_secs: 1,
            max_retries: 0,
            ..ClientOptions::default()
        },
    );
    let error = client
        .query("proposal", LlmRequest::new("gpt-4.1", "", "hi"))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, LlmError::Transient(message) if message.contains("timed out")),
        "{error}"
    );
}

#[tokio::test]
async fn offline_models_unknown_names_and_missing_keys() {
    let client = LlmClient::new(ClientOptions::default());
    let echo = client
        .query("demo", LlmRequest::new("mock/echo", "", "ping"))
        .await
        .unwrap();
    assert_eq!(
        (echo.content.as_str(), echo.cost_kind),
        ("ping", CostKind::UsageOnly)
    );
    let program =
        "# Current program\n\n```python\n# EVOLVE-BLOCK-START\nx = 0.5\n# EVOLVE-BLOCK-END\n```";
    let mutated = client
        .query(
            "proposal",
            LlmRequest::new("mock/perturb", "Answer with <CODE>.", program),
        )
        .await
        .unwrap();
    assert!(mutated.content.contains("<CODE>") && !mutated.content.contains("x = 0.5\n"));
    assert!(matches!(
        client
            .query("p", LlmRequest::new("mock/nope", "", "x"))
            .await,
        Err(LlmError::Config(_))
    ));
    assert!(matches!(
        client
            .query("p", LlmRequest::new("llama-local", "", "x"))
            .await,
        Err(LlmError::Config(_))
    ));
    let keyless = LlmClient::new(ClientOptions {
        base_urls: BTreeMap::from([("deepseek".to_string(), "http://127.0.0.1:9".to_string())]),
        ..ClientOptions::default()
    });
    if std::env::var("DEEPSEEK_API_KEY").is_err() {
        let error = keyless
            .query("p", LlmRequest::new("deepseek-chat", "", "x"))
            .await
            .unwrap_err();
        assert!(
            matches!(&error, LlmError::Config(message) if message.contains("DEEPSEEK_API_KEY")),
            "{error}"
        );
    }
}

#[tokio::test]
async fn a_mock_provider_sees_the_stage_and_can_bill() {
    let mock = genesis_llm::MockProvider::costing(0.25, |request| {
        Ok(format!("{}:{}", request.stage, request.model))
    });
    let dir = tempfile::tempdir().unwrap();
    let ledger = Arc::new(CostLedger::open(dir.path()).unwrap());
    let client = LlmClient::new(ClientOptions {
        mock: Some(mock.clone()),
        ledger: Some(Arc::clone(&ledger)),
        budget_usd: Some(0.5),
        phase: "ablations".to_string(),
        ..ClientOptions::default()
    });
    assert_eq!(
        client
            .query("variant", LlmRequest::new("anything-at-all", "", "x"))
            .await
            .unwrap()
            .content,
        "variant:anything-at-all"
    );
    client
        .with_phase("paper")
        .query("draft", LlmRequest::new("m", "", "x"))
        .await
        .unwrap();
    assert!(matches!(
        client.query("variant", LlmRequest::new("m", "", "x")).await,
        Err(LlmError::BudgetExceeded { .. })
    ));
    assert_eq!(mock.calls().len(), 2);
    assert_eq!(mock.calls()[1].stage, "draft");
    let summary = ledger.summary();
    assert!(
        close(summary.by_phase["ablations"].cost_usd, 0.25)
            && close(summary.by_phase["paper"].cost_usd, 0.25)
    );
}
