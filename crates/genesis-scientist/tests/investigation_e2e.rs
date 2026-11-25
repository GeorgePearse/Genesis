//! End-to-end test of the investigation phase: ideas, a model arm and a
//! coding-agent arm, harness-checked rubrics, memory, distilled conclusions,
//! and the receipts the downstream phases rely on.

use genesis_core::idea::RubricStatus;
use genesis_core::{fsutil, ConsoleMode, CostLedger, EventLog, Task};
use genesis_evolve::{DatabaseConfig, EvolutionConfig, ProgramDb};
use genesis_llm::{AgentSpec, ClientOptions, LlmClient, LlmRequest, MockProvider};
use genesis_scientist::artifacts::{
    BestProgramManifest, InvestigationCompletion, OUTCOME_NO_WINNER, OUTCOME_WINNER,
};
use genesis_scientist::ideas::IdeaConfig;
use genesis_scientist::selection::{freeze_selection, load_selection};
use genesis_scientist::{run_investigation, InvestigationConfig, PhaseContext, RunPaths};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

const EVALUATOR: &str = r#"
program="$1"; results="$2"
x=$(sed -n 's/^x = \(-\{0,1\}[0-9.]*\)$/\1/p' "$program" | head -n 1)
if [ -z "$x" ]; then
  echo '{"combined_score": 0.0}' > "$results/metrics.json"
  echo '{"correct": false, "error": "x is not assigned a number"}' > "$results/correct.json"
  exit 0
fi
score=$(awk -v x="$x" 'BEGIN { printf "%.6f", 10 - (x - 3) * (x - 3) }')
echo "{\"combined_score\": $score, \"public\": {\"x\": $x}}" > "$results/metrics.json"
echo '{"correct": true}' > "$results/correct.json"
"#;

const SEED: &str = "# EVOLVE-BLOCK-START\nx = 1.0\n# EVOLVE-BLOCK-END\n";

fn context(dir: &Path, provider: Arc<MockProvider>) -> PhaseContext {
    let source = dir.join("source_task");
    std::fs::create_dir_all(&source).unwrap();
    std::fs::write(source.join("task.md"), "# Goal\nBring x to 3.\n").unwrap();
    std::fs::write(source.join("initial.py"), SEED).unwrap();
    std::fs::write(source.join("evaluate.sh"), EVALUATOR).unwrap();
    let paths = RunPaths::new(dir.join("run"));
    let task = Task::load(&source)
        .unwrap()
        .snapshot(&paths.task())
        .unwrap();
    let ledger = Arc::new(CostLedger::open(&paths.run_dir).unwrap());
    let llm = LlmClient::new(ClientOptions {
        mock: Some(provider),
        ledger: Some(ledger),
        backoff_base_ms: 1,
        ..ClientOptions::default()
    });
    PhaseContext {
        events: EventLog::open(&paths.events(), ConsoleMode::Quiet, false),
        paths,
        task,
        llm,
        agent: None,
        cancel: Arc::new(AtomicBool::new(false)),
        seed: 5,
    }
}

fn rewrite(x: f64, name: &str) -> String {
    format!(
        "<NAME>\n{name}\n</NAME>\n<DESCRIPTION>\nraise x with step(x)\n</DESCRIPTION>\n<CODE>\n```python\n\
         # EVOLVE-BLOCK-START\nx = {x:.2}\n# step(x)\n# EVOLVE-BLOCK-END\n```\n</CODE>"
    )
}

/// Answers every stage of the hybrid engine.
fn scripted_models(step: f64) -> Arc<MockProvider> {
    let proposals = AtomicUsize::new(0);
    let ideas = AtomicUsize::new(0);
    MockProvider::new(move |request: &LlmRequest| match request.stage.as_str() {
        "idea" => {
            let n = ideas.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(json!({
                "idea_id": format!("raise_x_{n}"),
                "title": format!("Raise x step {n}"),
                "hypothesis": "x closer to 3 scores higher",
                "implementation_plan": ["increase x"],
                "target_rubrics": [
                    {"id": "assigns_x", "description": "assigns x", "type": "implementation",
                     "verification": {"type": "source_contains", "patterns": ["x ="]}},
                    {"id": "calls_step", "description": "calls a step function", "type": "implementation",
                     "verification": {"type": "source_contains", "patterns": ["step(x)"]}},
                    {"id": "good_score", "description": "combined_score >= 6", "type": "metric"}
                ],
                "notes_for_agent": "keep the markers"
            })
            .to_string())
        }
        "proposal" => {
            assert!(
                request.system.contains("# Active Idea"),
                "the idea reaches the implementer"
            );
            let n = proposals.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(rewrite(
                (1.0 + step * n as f64).min(3.0),
                &format!("model_step_{n}"),
            ))
        }
        "conclusions" => Ok(json!({
            "gate": "updated", "trigger": "b", "reason": "raising x is a new, working direction",
            "conclusions": "Raising x toward three keeps improving the score; do not lower it."
        })
        .to_string()),
        other => panic!("unexpected stage {other}"),
    })
}

fn config(generations: u32, arms: &[&str]) -> InvestigationConfig {
    let fake_agent = AgentSpec {
        name: "fake".into(),
        // Edits inside the block only, to a fresh value each time so the program always changes.
        command: Some(vec!["bash".into(), "-c".into(), "sed -i \"s/^x = .*$/x = 2.9$RANDOM/\" main.py; echo 'NAME: agent_jump'; echo 'DESCRIPTION: jumped x to 2.9 # step(x)'".into()]),
        ..AgentSpec::default()
    };
    InvestigationConfig {
        engine: "hybrid".into(),
        evolution: EvolutionConfig {
            patch_types: vec!["full".into()],
            patch_type_probs: vec![1.0],
            num_generations: generations,
            max_evaluation_jobs: 1,
            llm_models: arms.iter().map(|arm| arm.to_string()).collect(),
            seed: Some(11),
            ..EvolutionConfig::default()
        },
        database: DatabaseConfig {
            num_islands: 1,
            ..DatabaseConfig::default()
        },
        ideas: IdeaConfig {
            models: vec!["idea-model".into()],
            ..IdeaConfig::default()
        },
        conclusions_interval: 2,
        agents: BTreeMap::from([("fake".to_string(), fake_agent)]),
        ..InvestigationConfig::default()
    }
}

#[tokio::test]
async fn a_hybrid_investigation_writes_every_artifact_and_finds_a_winner() {
    let dir = tempfile::tempdir().unwrap();
    let provider = scripted_models(0.5);
    let context = context(dir.path(), provider.clone());
    let outcome = run_investigation(&context, &config(7, &["model-a", "agent:fake"]))
        .await
        .unwrap();

    assert!(outcome.completed);
    assert_eq!(outcome.outcome, OUTCOME_WINNER);
    assert_eq!(outcome.summary.generations_completed, 7);
    assert!(outcome.manifest.combined_score.unwrap() > 9.0);

    let investigation = context.paths.investigation();
    let db = ProgramDb::open(
        Some(&investigation.join("programs.sqlite")),
        DatabaseConfig {
            num_islands: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let programs = db.all().unwrap();
    let children: Vec<_> = programs.iter().filter(|p| p.generation > 0).collect();
    assert_eq!(children.len(), 6);
    let by_agent: Vec<_> = children
        .iter()
        .filter(|p| p.metadata["operator"] == "agent")
        .collect();
    let by_model: Vec<_> = children
        .iter()
        .filter(|p| p.metadata["operator"] == "patch")
        .collect();
    assert!(
        !by_agent.is_empty() && !by_model.is_empty(),
        "both arms were used"
    );

    // Rubrics were checked by the harness against the stored code.
    for child in &children {
        let status: RubricStatus =
            serde_json::from_value(child.metadata["rubric_status"].clone()).unwrap();
        let passed = |id: &str| status.rubrics.iter().find(|r| r.id == id).map(|r| r.pass);
        assert_eq!(passed("assigns_x"), Some(true));
        // `step(x)` exists only inside a comment in both arms' programs.
        assert_eq!(
            passed("calls_step"),
            Some(false),
            "a comment must not satisfy a source rubric"
        );
        assert_eq!(passed("good_score"), Some(child.combined_score >= 6.0));
        assert!(status
            .rubrics
            .iter()
            .any(|r| r.id == "incumbent_score_improvement"));
        assert!(!status.overall_pass);
        assert_eq!(
            child.metadata["idea"]["hypothesis"],
            "x closer to 3 scores higher"
        );
    }
    for agent_child in &by_agent {
        assert_eq!(agent_child.meta_str("patch_name"), Some("agent_jump"));
        assert!(agent_child.code.contains("x = 2.9"));
    }

    // Histories, per-idea files, memory and conclusions.
    let ideas: Vec<Value> = fsutil::read_jsonl(&investigation.join("ideas_history.jsonl"));
    let attempts: Vec<Value> = fsutil::read_jsonl(&investigation.join("attempts_history.jsonl"));
    assert_eq!((ideas.len(), attempts.len()), (6, 6));
    assert!(attempts
        .iter()
        .all(|a| a["attempt_status"] == "fail" && a["failure_reason"] == "rubrics"));
    assert!(attempts.iter().any(|a| a["best_updated"] == true));
    let first_dir = ideas[0]["idea_dir_name"].as_str().unwrap();
    for file in [
        "idea_bundle.json",
        "target_rubrics.json",
        "rubric_status.json",
    ] {
        assert!(
            investigation
                .join("ideas")
                .join(first_dir)
                .join(file)
                .is_file(),
            "missing {file}"
        );
    }
    let memory: Vec<Value> = fsutil::read_jsonl(&investigation.join("memory.jsonl"));
    assert_eq!(memory.iter().filter(|e| e["step"] == "idea").count(), 6);
    assert_eq!(memory.iter().filter(|e| e["step"] == "evaluate").count(), 6);
    assert!(investigation.join("memory.md").is_file());
    let conclusions = std::fs::read_to_string(investigation.join("conclusions.md")).unwrap();
    assert!(conclusions.starts_with("Raising x toward three"));
    let history: Vec<Value> = fsutil::read_jsonl(&investigation.join("conclusions_history.jsonl"));
    assert_eq!(history.len(), 3, "one update every two evaluated proposals");
    // Later ideas and proposals saw the conclusions.
    let later_idea = provider
        .calls()
        .into_iter()
        .rfind(|c| c.stage == "idea")
        .unwrap();
    assert!(later_idea.last_user().contains("Raising x toward three"));

    // Receipts.
    let manifest: BestProgramManifest =
        fsutil::read_json(&investigation.join("best_program_manifest.json")).unwrap();
    assert_eq!(manifest, outcome.manifest);
    assert!(context
        .paths
        .resolve(manifest.program_path.as_deref().unwrap())
        .is_file());
    let receipt: InvestigationCompletion =
        fsutil::read_json(&context.paths.completion_receipt()).unwrap();
    assert_eq!(
        (receipt.status.as_str(), receipt.outcome.as_deref()),
        ("completed", Some(OUTCOME_WINNER))
    );
    assert!(investigation.join("outer_state.json").is_file());

    // Costs were attributed to the phase, including the agent's attempts.
    let ledger = context.llm.ledger().unwrap().summary();
    assert!(
        ledger.by_stage.contains_key("investigation/idea")
            || ledger.by_stage.keys().any(|k| k.ends_with("/idea"))
    );
    assert_eq!(ledger.by_source["agent_cli"].calls as usize, by_agent.len());

    // The downstream selection freezes the winner.
    let selected = freeze_selection(&context.paths, &context.task, &db, "best_available")
        .unwrap()
        .unwrap();
    assert_eq!(selected.selection.selection_kind, "winner");
    assert_eq!(
        selected.selection.combined_score,
        outcome.manifest.combined_score.unwrap()
    );
    assert_eq!(load_selection(&context.paths).unwrap().unwrap(), selected);
}

#[tokio::test]
async fn a_search_that_never_beats_the_seed_completes_without_a_winner() {
    let dir = tempfile::tempdir().unwrap();
    // Every proposal moves x away from 3.
    let context = context(dir.path(), scripted_models(-0.5));
    let mut config = config(4, &["model-a"]);
    config.engine = "scientist".into();
    let outcome = run_investigation(&context, &config).await.unwrap();
    assert!(outcome.completed);
    assert_eq!(outcome.outcome, OUTCOME_NO_WINNER);
    assert_eq!(outcome.manifest.source, "seed");
    let receipt: InvestigationCompletion =
        fsutil::read_json(&context.paths.completion_receipt()).unwrap();
    assert_eq!(receipt.outcome.as_deref(), Some(OUTCOME_NO_WINNER));

    // The scientist engine starts every attempt from the incumbent: all children descend from the seed.
    let db = ProgramDb::open(
        Some(&context.paths.investigation().join("programs.sqlite")),
        DatabaseConfig {
            num_islands: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let programs = db.all().unwrap();
    let seed = programs.iter().find(|p| p.generation == 0).unwrap();
    assert!(programs
        .iter()
        .filter(|p| p.generation > 0)
        .all(|p| p.parent_id.as_deref() == Some(seed.id.as_str())));

    let selected = freeze_selection(&context.paths, &context.task, &db, "best_available")
        .unwrap()
        .unwrap();
    assert_eq!(selected.selection.selection_kind, "non_winner_candidate");
    assert_eq!(selected.selection.attempt_counts.scored, 3);
}

#[tokio::test]
async fn the_evolve_engine_makes_no_idea_or_conclusion_calls() {
    let dir = tempfile::tempdir().unwrap();
    let provider = MockProvider::new(|request: &LlmRequest| {
        assert_eq!(request.stage, "proposal", "the evolve engine only proposes");
        assert!(!request.system.contains("# Active Idea"));
        Ok(rewrite(2.0, "plain"))
    });
    let context = context(dir.path(), provider);
    let mut config = config(3, &["model-a"]);
    config.engine = "evolve".into();
    let outcome = run_investigation(&context, &config).await.unwrap();
    assert_eq!(outcome.summary.generations_completed, 3);
    let investigation = context.paths.investigation();
    assert!(!investigation.join("ideas_history.jsonl").exists());
    // The attempt history and memory are still kept, keyed by patch name.
    let attempts: Vec<Value> = fsutil::read_jsonl(&investigation.join("attempts_history.jsonl"));
    assert_eq!(attempts.len(), 2);
    assert!(attempts
        .iter()
        .all(|a| a["idea_id"] == "plain" && a["attempt_status"] == "pass"));
}
