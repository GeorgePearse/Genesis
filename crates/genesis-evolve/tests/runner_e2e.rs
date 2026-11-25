//! End-to-end tests of the evolution loop: a real evaluator process, a real
//! run directory and database, and scripted model replies.

use async_trait::async_trait;
use genesis_core::idea::Rubric;
use genesis_core::{fsutil, ConsoleMode, EventLog, IdeaBundle, Program, Task};
use genesis_evolve::{
    DatabaseConfig, EvalHook, EvaluatedProposal, EvolutionConfig, EvolutionRunner, IdeaSource,
    Operator, Proposal, ProposalContext, ProposalError, RunSummary, RunnerDeps,
};
use genesis_llm::{ClientOptions, LlmClient, LlmError, LlmRequest, MockProvider};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Scores `10 - ((x - 3)^2 + (y + 1)^2)`; a program that does not assign
/// both numbers is incorrect.
const EVALUATOR: &str = r#"
program="$1"; results="$2"
x=$(sed -n 's/^x = \(-\{0,1\}[0-9.]*\)$/\1/p' "$program" | head -n 1)
y=$(sed -n 's/^y = \(-\{0,1\}[0-9.]*\)$/\1/p' "$program" | head -n 1)
if [ -z "$x" ] || [ -z "$y" ]; then
  echo '{"combined_score": 0.0}' > "$results/metrics.json"
  echo '{"correct": false, "error": "x or y is not assigned a number"}' > "$results/correct.json"
  echo "missing assignment" >&2
  exit 0
fi
score=$(awk -v x="$x" -v y="$y" 'BEGIN { printf "%.6f", 10 - ((x - 3) * (x - 3) + (y + 1) * (y + 1)) }')
echo "{\"combined_score\": $score, \"public\": {\"x\": $x, \"y\": $y}}" > "$results/metrics.json"
echo '{"correct": true}' > "$results/correct.json"
"#;

fn write_task(dir: &Path, body: &str) -> Task {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("task.md"), "Bring x to 3 and y to -1.\n").unwrap();
    std::fs::write(
        dir.join("initial.py"),
        format!("# EVOLVE-BLOCK-START\n{body}# EVOLVE-BLOCK-END\n"),
    )
    .unwrap();
    std::fs::write(dir.join("evaluate.sh"), EVALUATOR).unwrap();
    Task::load(dir).unwrap()
}

fn rewrite(x: f64, y: f64, name: &str) -> String {
    format!(
        "<NAME>\n{name}\n</NAME>\n<DESCRIPTION>\nmove toward the target\n</DESCRIPTION>\n<CODE>\n```python\n\
         # EVOLVE-BLOCK-START\nx = {x:.2}\ny = {y:.2}\n# EVOLVE-BLOCK-END\n```\n</CODE>"
    )
}

/// A model that walks x from 1.4 toward 3 in steps of 0.4, one step per call.
fn climbing_model() -> Arc<MockProvider> {
    let calls = AtomicUsize::new(0);
    MockProvider::new(move |_request: &LlmRequest| {
        let step = calls.fetch_add(1, Ordering::SeqCst) + 1;
        let x = (1.0 + 0.4 * step as f64).min(3.0);
        Ok(rewrite(x, -1.0, &format!("step_{step}")))
    })
}

fn config(generations: u32) -> EvolutionConfig {
    EvolutionConfig {
        patch_types: vec!["full".into()],
        patch_type_probs: vec![1.0],
        num_generations: generations,
        max_evaluation_jobs: 2,
        llm_models: vec!["model-a".into(), "model-b".into()],
        seed: Some(3),
        ..EvolutionConfig::default()
    }
}

struct Harness {
    _dir: tempfile::TempDir,
    task: Task,
    results: PathBuf,
}

impl Harness {
    fn new(seed_body: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let task = write_task(&dir.path().join("task"), seed_body);
        let results = dir.path().join("results");
        Self {
            _dir: dir,
            task,
            results,
        }
    }

    fn deps(&self, llm: LlmClient) -> RunnerDeps {
        RunnerDeps {
            task: self.task.clone(),
            results_dir: self.results.clone(),
            llm,
            events: EventLog::open(
                &self.results.join("events.jsonl"),
                ConsoleMode::Quiet,
                false,
            ),
            agent_operator: None,
            idea_source: None,
            eval_hooks: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            prior_cost_usd: 0.0,
        }
    }

    async fn run(&self, config: EvolutionConfig, deps: RunnerDeps) -> (RunSummary, Vec<Program>) {
        let runner = EvolutionRunner::new(config, DatabaseConfig::default(), deps).unwrap();
        let summary = runner.run().await.unwrap();
        let programs = runner.db().all().unwrap();
        (summary, programs)
    }

    fn events(&self) -> Vec<String> {
        fsutil::read_jsonl::<Value>(&self.results.join("events.jsonl"))
            .iter()
            .map(|event| event["event"].as_str().unwrap_or("").to_string())
            .collect()
    }
}

fn own(programs: &[Program]) -> Vec<&Program> {
    programs
        .iter()
        .filter(|p| !p.metadata.contains_key("_is_island_copy"))
        .collect()
}

#[tokio::test]
async fn a_run_improves_on_the_seed_and_leaves_a_complete_run_directory() {
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let provider = climbing_model();
    let (summary, programs) = harness
        .run(config(6), harness.deps(LlmClient::mock(provider.clone())))
        .await;

    assert_eq!(summary.stop_reason, "completed");
    assert_eq!(
        (summary.generations_completed, summary.failed_proposals),
        (6, 0)
    );
    assert!(!summary.evolution_failed);
    let own_programs = own(&programs);
    assert_eq!(own_programs.len(), 6);
    let seed = own_programs.iter().find(|p| p.generation == 0).unwrap();
    assert_eq!(seed.combined_score, 6.0, "10 - (1 - 3)^2");
    assert!(own_programs.iter().all(|p| p.correct));
    let best = summary.best.clone().unwrap();
    assert!(
        best.combined_score > 9.0,
        "five steps of 0.4 reach x = 3: {}",
        best.combined_score
    );
    assert_eq!(best.public_metrics["x"], serde_json::json!(3.0));

    // Every child records where it came from and what it cost.
    for child in own_programs.iter().filter(|p| p.generation > 0) {
        assert!(
            child.parent_id.is_some()
                && child
                    .code_diff
                    .as_deref()
                    .is_some_and(|diff| diff.contains("+x = "))
        );
        assert_eq!(child.metadata["patch_type"], "full");
        assert!(child.meta_str("patch_name").unwrap().starts_with("step_"));
        assert!(["model-a", "model-b"].contains(&child.meta_str("arm").unwrap()));
        assert!(
            !child.embedding.is_empty(),
            "the offline embedding model was used"
        );
        assert!(child.complexity > 0.0);
    }

    let results = &harness.results;
    for file in [
        "programs.sqlite",
        "bandit_state.json",
        "run_summary.json",
        "events.jsonl",
        "gen_0/main.py",
        "gen_0/results/metrics.json",
        "gen_3/main.py",
        "gen_3/original.py",
        "gen_3/edit.diff",
        "gen_3/rewrite.txt",
        "gen_3/results/correct.json",
        "gen_3/results/job_log.out",
        "best/main.py",
        "best/best_program.json",
    ] {
        assert!(results.join(file).exists(), "missing {file}");
    }
    assert!(results
        .join("gen_3/attempts/novelty_0/resample_0/patch_0/llm_response.txt")
        .is_file());
    assert_eq!(
        std::fs::read_to_string(results.join("best/main.py")).unwrap(),
        best.code
    );
    let stored: RunSummary = fsutil::read_json(&results.join("run_summary.json")).unwrap();
    assert_eq!(stored, summary);

    let events = harness.events();
    assert_eq!(
        (
            events.first().map(String::as_str),
            events.last().map(String::as_str)
        ),
        (Some("run_start"), Some("run_end"))
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| *event == "proposal_evaluated")
            .count(),
        5
    );
    assert!(events.contains(&"seed_evaluated".to_string()));

    // The prompt carried the task, the parent and its metrics.
    let call = &provider.calls()[0];
    assert_eq!(call.stage, "proposal");
    assert!(call.system.starts_with("Bring x to 3 and y to -1."));
    assert!(
        call.last_user().contains("x = 1.0")
            && call
                .last_user()
                .contains("Combined score to maximize: 6.00")
    );
}

#[tokio::test]
async fn a_run_resumes_without_repeating_settled_generations() {
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let provider = climbing_model();
    let llm = LlmClient::mock(provider.clone());
    let (first, _) = harness.run(config(3), harness.deps(llm.clone())).await;
    assert_eq!(first.generations_completed, 3);
    let seed_metrics = std::fs::metadata(harness.results.join("gen_0/results/metrics.json"))
        .unwrap()
        .modified()
        .unwrap();
    let calls_before = provider.calls().len();

    let (second, programs) = harness.run(config(6), harness.deps(llm)).await;
    assert_eq!(
        (second.stop_reason.as_str(), second.generations_completed),
        ("completed", 6)
    );
    let mut generations: Vec<u32> = own(&programs).iter().map(|p| p.generation).collect();
    generations.sort();
    assert_eq!(
        generations,
        [0, 1, 2, 3, 4, 5],
        "each generation exists exactly once"
    );
    assert_eq!(
        provider.calls().len() - calls_before,
        3,
        "only the three new generations were proposed"
    );
    let after = std::fs::metadata(harness.results.join("gen_0/results/metrics.json"))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(seed_metrics, after, "the seed was not evaluated again");
    assert!(harness.events().contains(&"run_resumed".to_string()));

    // Nothing left to do: a third session proposes nothing.
    let calls_before = provider.calls().len();
    let (third, _) = harness
        .run(config(6), harness.deps(LlmClient::mock(provider.clone())))
        .await;
    assert_eq!(third.generations_completed, 6);
    assert_eq!(provider.calls().len(), calls_before);
}

#[tokio::test]
async fn unusable_replies_fail_their_generation_and_the_run_goes_on() {
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let calls = AtomicUsize::new(0);
    let provider = MockProvider::new(move |request: &LlmRequest| {
        let call = calls.fetch_add(1, Ordering::SeqCst);
        // The error fed back to the model is visible on the retry turn.
        if request.messages.len() > 1 {
            assert!(request
                .last_user()
                .starts_with("The previous edit was not successful. Error: "));
        }
        if call < 4 {
            Ok("I would rather describe the change than write code.".to_string())
        } else {
            Ok(rewrite(2.0, -1.0, "late_success"))
        }
    });
    let mut config = config(4);
    config.max_evaluation_jobs = 1;
    config.max_novelty_attempts = 1;
    config.max_patch_resamples = 2;
    config.max_patch_attempts = 2;
    let (summary, programs) = harness
        .run(
            config.clone(),
            harness.deps(LlmClient::mock(provider.clone())),
        )
        .await;

    // Generation 1 used four calls (two draws of two turns) and failed; 2 and 3 succeeded.
    assert_eq!(
        (summary.failed_proposals, summary.generations_completed),
        (1, 3)
    );
    assert_eq!(summary.stop_reason, "completed");
    assert!(!own(&programs).iter().any(|p| p.generation == 1));
    let failure: Value = fsutil::read_json(&harness.results.join("gen_1/failure.json")).unwrap();
    assert_eq!(failure["failure_class"], "patch_apply_failed");
    assert!(harness
        .results
        .join("gen_1/attempts/novelty_0/resample_1/patch_1/patch_error.txt")
        .is_file());
    assert!(harness.events().contains(&"proposal_failed".to_string()));

    // The failure is terminal: a resumed run does not propose generation 1 again.
    let before = provider.calls().len();
    let (resumed, _) = harness
        .run(config, harness.deps(LlmClient::mock(provider.clone())))
        .await;
    assert_eq!(provider.calls().len(), before);
    assert_eq!(resumed.failed_proposals, 1);
}

#[tokio::test]
async fn an_evaluator_that_cannot_score_the_seed_stops_the_run_before_any_model_call() {
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    std::fs::write(
        harness.task.dir.join("evaluate.sh"),
        "echo 'no such interpreter' >&2\nexit 127\n",
    )
    .unwrap();
    let provider = MockProvider::new(|_: &LlmRequest| panic!("no proposal may be paid for"));
    let (summary, programs) = harness
        .run(config(5), harness.deps(LlmClient::mock(provider.clone())))
        .await;
    assert_eq!(summary.stop_reason, "seed_evaluation_failed");
    assert!(summary.evolution_failed);
    assert_eq!((own(&programs).len(), summary.failed_proposals), (1, 0));
    assert!(provider.calls().is_empty());

    // Resuming the same run stops the same way: its task snapshot is unchanged.
    let (resumed, _) = harness
        .run(config(5), harness.deps(LlmClient::mock(provider.clone())))
        .await;
    assert_eq!(resumed.stop_reason, "seed_evaluation_failed");
    assert!(provider.calls().is_empty());
}

#[tokio::test]
async fn an_incorrect_seed_is_repaired_in_fix_mode() {
    let harness = Harness::new("x = one\ny = -1.0\n");
    let provider = MockProvider::new(|request: &LlmRequest| {
        if request
            .system
            .contains("You are debugging and fixing an incorrect program")
        {
            assert!(request
                .last_user()
                .contains("### Standard Error (stderr):\n\n```\nmissing assignment\n```"));
            Ok(rewrite(1.0, -1.0, "assign_a_number"))
        } else {
            Ok(rewrite(2.5, -1.0, "improve"))
        }
    });
    let mut config = config(4);
    config.max_evaluation_jobs = 1;
    let (summary, programs) = harness
        .run(config, harness.deps(LlmClient::mock(provider)))
        .await;
    let own_programs = own(&programs);
    let seed = own_programs.iter().find(|p| p.generation == 0).unwrap();
    assert!(!seed.correct);
    assert_eq!(seed.text_feedback, "x or y is not assigned a number");
    let fix = own_programs.iter().find(|p| p.generation == 1).unwrap();
    assert_eq!(
        (fix.metadata["patch_type"].as_str(), fix.correct),
        (Some("fix"), true)
    );
    assert_eq!(
        own_programs
            .iter()
            .filter(|p| p.metadata["patch_type"] == "full")
            .count(),
        2
    );
    assert!(summary.best.unwrap().combined_score > 9.0);
}

#[tokio::test]
async fn the_budget_stops_new_proposals() {
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let provider =
        MockProvider::costing(0.4, |_request: &LlmRequest| Ok(rewrite(2.0, -1.0, "paid")));
    let llm = LlmClient::new(ClientOptions {
        mock: Some(provider.clone()),
        ..ClientOptions::default()
    });
    let mut config = config(20);
    config.max_api_costs = Some(1.0);
    let (summary, _) = harness.run(config, harness.deps(llm)).await;
    assert_eq!(summary.stop_reason, "budget_exhausted");
    assert!(
        summary.cost_usd >= 0.8 && summary.cost_usd <= 1.2 + 1e-9,
        "spent {}",
        summary.cost_usd
    );
    assert!(
        provider.calls().len() <= 3,
        "{} calls",
        provider.calls().len()
    );
    assert!(summary.generations_completed >= 3 && summary.generations_completed < 20);
}

#[tokio::test]
async fn a_dead_endpoint_fails_over_then_stops_the_run() {
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let dead = |model: &str| LlmError::Endpoint {
        provider: "mock".into(),
        status: 401,
        message: format!("{model}: bad key"),
    };
    let provider = MockProvider::new(move |request: &LlmRequest| {
        if request.model == "model-a" {
            Err(dead("model-a"))
        } else {
            Ok(rewrite(2.0, -1.0, "from_b"))
        }
    });
    let (summary, programs) = harness
        .run(config(4), harness.deps(LlmClient::mock(provider)))
        .await;
    assert_eq!(
        (summary.stop_reason.as_str(), summary.generations_completed),
        ("completed", 4)
    );
    assert!(own(&programs)
        .iter()
        .filter(|p| p.generation > 0)
        .all(|p| p.meta_str("arm") == Some("model-b")));
    assert!(harness
        .events()
        .contains(&"model_endpoint_failed".to_string()));

    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let all_dead = MockProvider::new(move |request: &LlmRequest| {
        Err(LlmError::Endpoint {
            provider: "mock".into(),
            status: 401,
            message: format!("{}: bad key", request.model),
        })
    });
    let (summary, _) = harness
        .run(config(10), harness.deps(LlmClient::mock(all_dead)))
        .await;
    assert_eq!(summary.stop_reason, "llm_endpoint_error");
    assert!(summary.evolution_failed);
    assert_eq!(summary.generations_completed, 1);
}

struct FixedIdea;

#[async_trait]
impl IdeaSource for FixedIdea {
    async fn generate(&self, context: &ProposalContext) -> anyhow::Result<(IdeaBundle, f64)> {
        Ok((
            IdeaBundle {
                idea_id: format!("idea_{}", context.generation),
                title: "Move x".into(),
                hypothesis: "x closer to 3 scores higher".into(),
                implementation_plan: vec!["raise x".into()],
                target_rubrics: vec![Rubric {
                    id: "moves_x".into(),
                    kind: "implementation".into(),
                    ..Rubric::default()
                }],
                notes_for_agent: String::new(),
            },
            0.0,
        ))
    }

    fn conclusions(&self) -> Option<String> {
        Some("Raising x has helped.".to_string())
    }
}

struct RecordingHook {
    evaluations: AtomicUsize,
    failures: AtomicUsize,
}

#[async_trait]
impl EvalHook for RecordingHook {
    async fn after_evaluation(
        &self,
        program: &mut Program,
        evaluated: &EvaluatedProposal,
    ) -> anyhow::Result<()> {
        self.evaluations.fetch_add(1, Ordering::SeqCst);
        assert_eq!(evaluated.idea.as_ref().unwrap().title, "Move x");
        assert!(evaluated.gen_dir.join("results/metrics.json").is_file());
        program.metadata.insert(
            "hook_note".into(),
            serde_json::json!(format!("generation {}", evaluated.generation)),
        );
        Ok(())
    }

    async fn after_failure(
        &self,
        _generation: u32,
        _idea: Option<&IdeaBundle>,
        _class: &str,
        _message: &str,
    ) {
        self.failures.fetch_add(1, Ordering::SeqCst);
    }
}

struct ScriptedAgent;

#[async_trait]
impl Operator for ScriptedAgent {
    async fn propose(&self, context: &ProposalContext) -> Result<Proposal, ProposalError> {
        assert_eq!(context.arm, "agent:scripted");
        assert!(context.idea.is_some());
        Ok(Proposal {
            code: "# EVOLVE-BLOCK-START\nx = 3.0\ny = -1.0\n# EVOLVE-BLOCK-END\n".to_string(),
            patch_type: "agent".into(),
            name: "agent_edit".into(),
            description: "set x to its optimum".into(),
            num_applied: 1,
            llm_calls: 1,
            ..Proposal::default()
        })
    }
}

#[tokio::test]
async fn ideas_hooks_and_agent_arms_plug_into_the_same_loop() {
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let provider = MockProvider::new(|request: &LlmRequest| {
        assert!(
            request.system.contains("# Active Idea")
                && request.system.contains("- [implementation] moves_x")
        );
        assert!(
            request.system.contains("# Distilled Conclusions")
                && request.system.contains("Raising x has helped.")
        );
        Ok(rewrite(2.0, -1.0, "model_edit"))
    });
    let hook = Arc::new(RecordingHook {
        evaluations: AtomicUsize::new(0),
        failures: AtomicUsize::new(0),
    });
    let mut deps = harness.deps(LlmClient::mock(provider));
    deps.idea_source = Some(Arc::new(FixedIdea));
    deps.eval_hooks = vec![hook.clone()];
    deps.agent_operator = Some(Arc::new(ScriptedAgent));
    let mut config = config(9);
    config.llm_models = vec!["model-a".into(), "agent:scripted".into()];
    let (summary, programs) = harness.run(config.clone(), deps).await;

    assert_eq!(summary.generations_completed, 9);
    assert_eq!(hook.evaluations.load(Ordering::SeqCst), 8);
    assert_eq!(hook.failures.load(Ordering::SeqCst), 0);
    let children: Vec<&Program> = own(&programs)
        .into_iter()
        .filter(|p| p.generation > 0)
        .collect();
    let by_agent = children
        .iter()
        .filter(|p| p.metadata["operator"] == "agent")
        .count();
    assert!(
        by_agent > 0 && by_agent < 8,
        "the bandit tried both operators: {by_agent} of 8 by the agent"
    );
    for child in &children {
        assert_eq!(child.metadata["idea"]["title"], "Move x");
        assert_eq!(
            child.metadata["hook_note"],
            serde_json::json!(format!("generation {}", child.generation))
        );
    }
    assert_eq!(summary.best.unwrap().combined_score, 10.0);

    // An agent arm without an agent operator is refused before anything runs.
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let error = EvolutionRunner::new(
        config,
        DatabaseConfig::default(),
        harness.deps(LlmClient::mock(climbing_model())),
    )
    .err()
    .expect("the runner must refuse")
    .to_string();
    assert!(
        error.contains("agent:scripted") && error.contains("no agent operator"),
        "{error}"
    );
}

#[tokio::test]
async fn the_offline_mutator_runs_the_loop_with_no_credentials() {
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let mut config = config(8);
    config.llm_models = vec!["mock/perturb".into()];
    config.patch_types = vec!["diff".into(), "full".into()];
    config.patch_type_probs = vec![0.5, 0.5];
    let (summary, programs) = harness
        .run(
            config,
            harness.deps(LlmClient::new(ClientOptions::default())),
        )
        .await;
    assert_eq!(summary.stop_reason, "completed");
    let children: Vec<&Program> = own(&programs)
        .into_iter()
        .filter(|p| p.generation > 0)
        .collect();
    assert!(
        children.len() >= 4,
        "most perturbations apply: {} of 7",
        children.len()
    );
    assert!(children
        .iter()
        .all(|p| p.code.contains("EVOLVE-BLOCK-START")
            && p.code != "# EVOLVE-BLOCK-START\nx = 1.0\ny = -1.0\n# EVOLVE-BLOCK-END\n"));
    assert_eq!(summary.cost_usd, 0.0);
}

#[tokio::test]
async fn cancelling_stops_the_run_and_leaves_it_resumable() {
    let harness = Harness::new("x = 1.0\ny = -1.0\n");
    let mut deps = harness.deps(LlmClient::mock(climbing_model()));
    let cancel = Arc::clone(&deps.cancel);
    let hook_cancel = Arc::clone(&cancel);
    struct CancelAfterFirst(Arc<AtomicBool>);
    #[async_trait]
    impl EvalHook for CancelAfterFirst {
        async fn after_evaluation(
            &self,
            _program: &mut Program,
            _evaluated: &EvaluatedProposal,
        ) -> anyhow::Result<()> {
            self.0.store(true, Ordering::SeqCst);
            Ok(())
        }
    }
    deps.eval_hooks = vec![Arc::new(CancelAfterFirst(hook_cancel))];
    let mut config = config(12);
    config.max_evaluation_jobs = 1;
    let (summary, _) = harness.run(config.clone(), deps).await;
    assert_eq!(summary.stop_reason, "interrupted");
    assert!(summary.generations_completed < 12);
    assert!(cancel.load(Ordering::SeqCst));

    let (resumed, programs) = harness
        .run(config, harness.deps(LlmClient::mock(climbing_model())))
        .await;
    assert_eq!(
        (resumed.stop_reason.as_str(), resumed.generations_completed),
        ("completed", 12)
    );
    assert_eq!(own(&programs).len(), 12);
}
