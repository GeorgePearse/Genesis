//! Live tests against real models and a real coding agent. They cost money
//! or subscription quota, so they are ignored by default:
//!
//! ```bash
//! GENESIS_LIVE_MODEL=gpt-6-luna cargo test -p genesis-scientist --test live -- --ignored --nocapture
//! ```
//!
//! They need `rustc` and `python3` for the circle-packing example.

use genesis_core::{fsutil, ConsoleMode, CostLedger, EventLog, Task};
use genesis_evolve::{DatabaseConfig, EvolutionConfig};
use genesis_llm::{ClientOptions, LlmClient};
use genesis_scientist::ideas::IdeaConfig;
use genesis_scientist::{run_investigation, InvestigationConfig, PhaseContext, RunPaths};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn example_task() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/circle_packing_rs")
}

fn context(run_dir: &Path, budget_usd: f64) -> PhaseContext {
    let paths = RunPaths::new(run_dir);
    let task = Task::load(&example_task())
        .unwrap()
        .snapshot(&paths.task())
        .unwrap();
    let ledger = Arc::new(CostLedger::open(&paths.run_dir).unwrap());
    let llm = LlmClient::new(ClientOptions {
        ledger: Some(ledger),
        budget_usd: Some(budget_usd),
        phase: "investigation".to_string(),
        work_dir: Some(paths.run_dir.join(".llm_scratch")),
        ..ClientOptions::default()
    });
    PhaseContext {
        events: EventLog::open(&paths.events(), ConsoleMode::Human, false),
        paths,
        task,
        llm,
        agent: None,
        cancel: Arc::new(AtomicBool::new(false)),
        seed: 1,
    }
}

fn config(arms: Vec<String>, generations: u32, jobs: usize) -> InvestigationConfig {
    let text: Vec<String> = arms
        .iter()
        .filter(|arm| !arm.starts_with("agent:"))
        .cloned()
        .collect();
    let helper = text.first().cloned().unwrap_or_else(|| {
        std::env::var("GENESIS_LIVE_MODEL").unwrap_or_else(|_| "gpt-6-luna".into())
    });
    InvestigationConfig {
        engine: "hybrid".into(),
        evolution: EvolutionConfig {
            num_generations: generations,
            max_evaluation_jobs: jobs,
            llm_models: arms,
            max_patch_attempts: 2,
            seed: Some(1),
            ..EvolutionConfig::default()
        },
        database: DatabaseConfig {
            num_islands: 1,
            ..DatabaseConfig::default()
        },
        ideas: IdeaConfig {
            models: vec![helper.clone()],
            ..IdeaConfig::default()
        },
        conclusions_models: vec![helper],
        conclusions_interval: 2,
        agent_timeout_secs: 600,
        ..InvestigationConfig::default()
    }
}

fn report(run_dir: &Path) {
    let investigation = run_dir.join("investigation");
    let summary: Value = fsutil::read_json(&investigation.join("run_summary.json")).unwrap();
    let attempts: Vec<Value> = fsutil::read_jsonl(&investigation.join("attempts_history.jsonl"));
    eprintln!(
        "stop={} generations={} failed={} best={} cost=${:.4}",
        summary["stop_reason"],
        summary["generations_completed"],
        summary["failed_proposals"],
        summary["best"]["combined_score"],
        summary["cost_usd"].as_f64().unwrap_or(0.0)
    );
    for attempt in attempts {
        eprintln!(
            "  cycle {} {} {} score={} correct={} rubrics={}",
            attempt["cycle_index"],
            attempt["idea_id"],
            attempt["attempt_status"],
            attempt["metrics"]["combined_score"],
            attempt["metrics"]["correct"],
            attempt["rubric_overall_pass"]
        );
    }
    eprintln!(
        "conclusions: {}",
        std::fs::read_to_string(investigation.join("conclusions.md"))
            .unwrap_or_default()
            .trim()
    );
}

#[tokio::test]
#[ignore = "calls a real model"]
async fn live_hybrid_investigation_on_circle_packing() {
    let model = std::env::var("GENESIS_LIVE_MODEL").unwrap_or_else(|_| "gpt-6-luna".to_string());
    let dir = tempfile::tempdir().unwrap();
    let run_dir = std::env::var("GENESIS_LIVE_RUN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.path().join("run"));
    let context = context(&run_dir, 1.0);
    let outcome = run_investigation(&context, &config(vec![model], 6, 2))
        .await
        .unwrap();
    report(&run_dir);
    assert!(
        outcome.completed,
        "stop reason {}",
        outcome.summary.stop_reason
    );
    assert!(
        outcome.summary.generations_completed >= 3,
        "most proposals should apply and evaluate"
    );
    assert!(
        outcome.manifest.combined_score.unwrap() >= 1.82 - 1e-9,
        "never worse than the seed"
    );
}

#[tokio::test]
#[ignore = "runs a real coding agent with edit permissions in a scratch directory"]
async fn live_agent_arm_on_circle_packing() {
    let agent =
        std::env::var("GENESIS_LIVE_AGENT").unwrap_or_else(|_| "agent:claude@haiku".to_string());
    let dir = tempfile::tempdir().unwrap();
    let run_dir = std::env::var("GENESIS_LIVE_RUN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.path().join("run"));
    let context = context(&run_dir, 1.0);
    let mut config = config(vec![agent], 3, 1);
    config.ideas.enabled = false;
    config.engine = "evolve".into();
    let outcome = run_investigation(&context, &config).await.unwrap();
    report(&run_dir);
    assert!(outcome.completed);
    assert!(
        outcome.summary.generations_completed >= 2,
        "the agent should deliver at least one evaluated edit"
    );
}
