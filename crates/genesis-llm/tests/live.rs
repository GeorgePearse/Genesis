//! Smoke tests against real services. They are ignored by default because
//! they need credentials or a logged-in CLI and spend a little money:
//!
//! ```bash
//! cargo test -p genesis-llm --test live -- --ignored --test-threads 1
//! ```
//!
//! `GENESIS_LIVE_OPENAI_MODEL` picks the chat model (default `gpt-6-luna`).

use genesis_core::CostKind;
use genesis_llm::{run_agent, AgentSpec, AgentTask, ClientOptions, LlmClient, LlmRequest};
use std::time::Duration;

fn client(dir: &std::path::Path) -> LlmClient {
    LlmClient::new(ClientOptions {
        budget_usd: Some(0.25),
        max_retries: 1,
        timeout_secs: 180,
        work_dir: Some(dir.to_path_buf()),
        ..ClientOptions::default()
    })
}

fn says_ok(reply: &str) -> bool {
    reply
        .trim()
        .trim_end_matches('.')
        .eq_ignore_ascii_case("ok")
}

const PROMPT: &str = "Reply with exactly the single word OK.";

#[tokio::test]
#[ignore = "live: needs OPENAI_API_KEY and spends a fraction of a cent"]
async fn openai_answers_a_real_request() {
    let dir = tempfile::tempdir().unwrap();
    let model =
        std::env::var("GENESIS_LIVE_OPENAI_MODEL").unwrap_or_else(|_| "gpt-6-luna".to_string());
    let client = client(dir.path());
    let request = LlmRequest::new(model.clone(), "You are terse.", PROMPT)
        .temperature(0.0)
        .max_tokens(64)
        .reasoning_effort("disabled");
    let response = client.query("live", request).await.unwrap();
    println!(
        "{model}: {:?} usage={:?} cost=${:.6} ({:?})",
        response.content, response.usage, response.cost_usd, response.cost_kind
    );
    assert!(says_ok(&response.content), "{:?}", response.content);
    assert!(response.usage.input_tokens > 0 && response.usage.output_tokens > 0);
    assert!(response.cost_usd > 0.0 && response.cost_usd < 0.01);
    assert_eq!(response.cost_kind, CostKind::Estimated);
    assert!((client.spent_usd() - response.cost_usd).abs() < 1e-12);

    let (value, _) = client
        .query_json(
            "live",
            LlmRequest::new(
                model,
                "Reply in JSON.",
                "Return the JSON object {\"ok\": true}.",
            )
            .max_tokens(64),
        )
        .await
        .unwrap();
    assert_eq!(value, serde_json::json!({"ok": true}));
}

#[tokio::test]
#[ignore = "live: needs OPENAI_API_KEY and spends a fraction of a cent"]
async fn openai_embeds_text() {
    let dir = tempfile::tempdir().unwrap();
    let client = client(dir.path());
    let vector = client
        .embed(
            "live",
            "text-embedding-3-small",
            "fn main() { println!(\"hi\"); }",
        )
        .await
        .unwrap();
    println!(
        "text-embedding-3-small: {} dimensions, cost=${:.8}",
        vector.len(),
        client.spent_usd()
    );
    assert_eq!(vector.len(), 1536);
    assert!(client.spent_usd() > 0.0 && client.spent_usd() < 0.001);
}

#[tokio::test]
#[ignore = "live: needs a logged-in claude CLI"]
async fn claude_cli_answers_as_a_completion_backend() {
    let dir = tempfile::tempdir().unwrap();
    let response = client(dir.path())
        .query(
            "live",
            LlmRequest::new("cli/claude@haiku", "You are terse.", PROMPT),
        )
        .await
        .unwrap();
    println!(
        "cli/claude@haiku: {:?} usage={:?} cost=${:.6} ({:?})",
        response.content, response.usage, response.cost_usd, response.cost_kind
    );
    assert!(says_ok(&response.content), "{:?}", response.content);
    assert_eq!(response.provider, "cli/claude");
    assert!(response.usage.output_tokens > 0);
    assert_eq!(
        response.cost_kind,
        CostKind::Estimated,
        "a subscription's self-reported price is an estimate"
    );
    assert_eq!(
        std::fs::read_dir(dir.path().join("headless_prompts"))
            .unwrap()
            .count(),
        1
    );
}

#[tokio::test]
#[ignore = "live: needs a logged-in claude CLI; the agent runs with full permissions in a temp dir"]
async fn claude_agent_edits_a_file_in_its_workspace() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("work")).unwrap();
    std::fs::write(dir.path().join("work/main.py"), "VALUE = 1\n").unwrap();
    let spec = AgentSpec {
        name: "claude".into(),
        model: Some("haiku".into()),
        ..AgentSpec::default()
    };
    let task = AgentTask {
        prompt: "Edit main.py in the current directory so that VALUE is 2 instead of 1. Change nothing else, then reply DONE.".into(),
        cwd: dir.path().join("work"),
        timeout: Duration::from_secs(180),
        allow_edits: true,
        log_path: dir.path().join("agent.log"),
    };
    let outcome = run_agent(&spec, &task).await.unwrap();
    println!("agent:claude@haiku: {outcome:?}");
    assert!(outcome.succeeded(), "{outcome:?}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("work/main.py"))
            .unwrap()
            .trim(),
        "VALUE = 2"
    );
    assert!(outcome.usage.output_tokens > 0 && outcome.model.contains("haiku"));

    let read_only = AgentTask {
        prompt: "Change VALUE in main.py to 3. If you are unable to modify files, reply CANNOT."
            .into(),
        allow_edits: false,
        ..task
    };
    let outcome = run_agent(&spec, &read_only).await.unwrap();
    println!("agent:claude@haiku read-only: {outcome:?}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("work/main.py"))
            .unwrap()
            .trim(),
        "VALUE = 2",
        "a read-only agent changed a file"
    );
}

#[tokio::test]
#[ignore = "live: needs a logged-in codex CLI"]
async fn codex_cli_answers_read_only() {
    let dir = tempfile::tempdir().unwrap();
    let response = client(dir.path())
        .query("live", LlmRequest::new("cli/codex", "", PROMPT))
        .await
        .unwrap();
    println!(
        "cli/codex: {:?} usage={:?} cost=${:.6} ({:?})",
        response.content, response.usage, response.cost_usd, response.cost_kind
    );
    assert!(says_ok(&response.content), "{:?}", response.content);
    assert!(response.usage.input_tokens > 0);
}

#[tokio::test]
#[ignore = "live: needs a logged-in pi CLI"]
async fn pi_cli_answers_without_tools() {
    let dir = tempfile::tempdir().unwrap();
    let response = client(dir.path())
        .query("live", LlmRequest::new("cli/pi?effort=low", "", PROMPT))
        .await
        .unwrap();
    println!(
        "cli/pi: {:?} usage={:?} cost=${:.6} ({:?})",
        response.content, response.usage, response.cost_usd, response.cost_kind
    );
    assert!(says_ok(&response.content), "{:?}", response.content);
    assert!(response.usage.output_tokens > 0);
}
