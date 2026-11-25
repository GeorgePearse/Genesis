//! The sealed final evaluation end to end, with real evaluator processes.

mod common;

use common::{build, by_stage, deterministic, stochastic, Run};
use genesis_core::fsutil;
use genesis_evidence::final_eval::{NOT_QUALIFIED, QUALIFIED};
use genesis_evidence::{load_final_manifest, run_final_evaluation, FinalEvalConfig};
use genesis_llm::LlmClient;
use genesis_scientist::artifacts::{AblationSummary, FinalManifest};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

const SEED: &str = "x = 1.0\ny = -1.0\nbonus = 0\n";
const WINNER: &str = "x = 3.0\ny = -1.0\nbonus = 1\n";

/// A final evaluator that only accepts programs without a bonus term: the
/// search evaluator rewarded something the held-out one forbids.
const STRICT_FINAL: &str = r#"
program="$1"; results="$2"
echo "final $program seed=$GENESIS_EVAL_SEED" >> "$(dirname "$0")/../eval_count.log"
if grep -q '^bonus = 1$' "$program"; then
  echo '{"combined_score": 0.0}' > "$results/metrics.json"
  echo '{"correct": false, "error": "the bonus term is not allowed in the final evaluation"}' > "$results/correct.json"
else
  echo '{"combined_score": 6.5}' > "$results/metrics.json"
  echo '{"correct": true}' > "$results/correct.json"
fi
"#;

fn run_with(evaluator: &str, final_evaluator: Option<&str>) -> Run {
    build(
        evaluator,
        final_evaluator,
        &[(0, SEED, 6.0, true), (1, WINNER, 11.0, true)],
        LlmClient::mock(by_stage(|_, _| None)),
    )
}

fn config(replications: usize) -> FinalEvalConfig {
    FinalEvalConfig {
        num_replications: replications,
        ..FinalEvalConfig::default()
    }
}

#[tokio::test]
async fn a_winner_qualifies_on_held_out_seeds_and_the_result_is_sealed() {
    let run = run_with(&stochastic(), None);
    let manifest = run_final_evaluation(&run.context, &config(4))
        .await
        .unwrap();

    assert_eq!(
        (
            manifest.status.as_str(),
            manifest.outcome.as_str(),
            manifest.correct
        ),
        ("completed", QUALIFIED, true)
    );
    assert_eq!(manifest.evaluator, "search_evaluator_heldout_seeds");
    assert_eq!(manifest.seeds, [1_000_003, 1_000_004, 1_000_005, 1_000_006]);
    assert_eq!(
        (manifest.episodes.len(), manifest.baseline_episodes.len()),
        (4, 4)
    );
    assert!(manifest
        .episodes
        .iter()
        .zip(&manifest.baseline_episodes)
        .all(|(c, b)| c.seed == b.seed && c.correct && b.correct));
    let comparison = manifest.comparison.as_ref().unwrap();
    assert!(
        (comparison.mean_diff - 5.0).abs() < 1e-9,
        "the same seeds give both programs the same offsets"
    );
    assert!(manifest.improvement_supported && comparison.diff_ci95_low.unwrap() > 0.0);
    assert!(manifest.limitations.is_empty());
    let selection = run.json(".downstream_selection.json");
    assert_eq!(
        manifest.program_sha256,
        selection["program_sha256"].as_str().unwrap()
    );
    assert_eq!(
        manifest.task_fingerprint,
        selection["task_fingerprint"].as_str().unwrap()
    );
    assert_eq!(run.evaluations(), 8);

    let invocation = run.json("final_evaluation/invocation.json");
    assert_eq!(invocation["status"], "started");
    assert_eq!(invocation["seeds"], json!(manifest.seeds));
    let metrics = run
        .run_dir()
        .join("final_evaluation/results/candidate/seed_1000003/metrics.json");
    assert!(metrics.is_file());
    assert!(std::fs::metadata(&metrics)
        .unwrap()
        .permissions()
        .readonly());
    assert!(std::fs::write(
        run.run_dir()
            .join("final_evaluation/results/candidate/new_file"),
        "x"
    )
    .is_err());

    // A second call returns the sealed result and evaluates nothing.
    let again = run_final_evaluation(&run.context, &config(4))
        .await
        .unwrap();
    assert_eq!(again, manifest);
    assert_eq!(run.evaluations(), 8);
    assert_eq!(load_final_manifest(run.paths()).unwrap(), Some(manifest));
    let events = run.events();
    assert!(
        events.contains(&"final_evaluation_end".to_string())
            && events.contains(&"final_evaluation_reused".to_string())
    );
}

#[tokio::test]
async fn final_seeds_avoid_the_seeds_an_ablation_study_used() {
    let run = run_with(&deterministic(), None);
    let used = AblationSummary {
        seeds: vec![1_000_003, 1_000_004, 7],
        ..Default::default()
    };
    fsutil::write_json(
        &run.paths().ablations().join("pipeline_summary.json"),
        &used,
    )
    .unwrap();
    let manifest = run_final_evaluation(&run.context, &config(2))
        .await
        .unwrap();
    assert_eq!(manifest.seeds, [1_000_005, 1_000_006]);
    // A deterministic evaluator: no interval, the improvement rests on the means.
    assert!(manifest
        .comparison
        .as_ref()
        .unwrap()
        .diff_ci95_low
        .is_none());
    assert!(manifest.improvement_supported);
    assert_eq!(manifest.candidate.as_ref().unwrap().mean, 11.0);
}

#[tokio::test]
async fn a_task_final_evaluator_decides_and_a_failure_is_reported_with_its_limits() {
    let run = run_with(&deterministic(), Some(STRICT_FINAL));
    let manifest = run_final_evaluation(&run.context, &config(2))
        .await
        .unwrap();
    assert_eq!(manifest.evaluator, "evaluate_final");
    assert_eq!(
        (manifest.outcome.as_str(), manifest.correct),
        (NOT_QUALIFIED, false)
    );
    assert!(
        manifest.candidate.is_none(),
        "no correct episode, no statistics"
    );
    assert_eq!(manifest.baseline.as_ref().unwrap().mean, 6.5);
    assert!(manifest.comparison.is_none() && !manifest.improvement_supported);
    assert_eq!(manifest.limitations.len(), 3);
    assert!(manifest.limitations[0].contains("did not qualify"));
    assert!(manifest.limitations[2].contains("separate evidence"));
    let log = run.text("eval_count.log");
    assert_eq!(
        log.lines()
            .filter(|line| line.starts_with("final "))
            .count(),
        4,
        "only the final evaluator ran"
    );
    // Not qualified is a finished evaluation: it loads, and it cannot be redone.
    assert_eq!(
        load_final_manifest(run.paths()).unwrap().unwrap().outcome,
        NOT_QUALIFIED
    );
}

#[tokio::test]
async fn the_baseline_as_selection_supports_no_improvement() {
    let run = build(
        &deterministic(),
        None,
        &[(0, SEED, 6.0, true), (1, "x = 1.0\n", 0.0, false)],
        LlmClient::mock(by_stage(|_, _| None)),
    );
    let manifest = run_final_evaluation(&run.context, &config(2))
        .await
        .unwrap();
    assert_eq!(manifest.outcome, QUALIFIED);
    assert_eq!(manifest.baseline_episodes, manifest.episodes);
    assert!(!manifest.improvement_supported);
    assert!(manifest.limitations[0].contains("the baseline itself"));
    assert_eq!(
        run.evaluations(),
        2,
        "the one program is evaluated once per seed"
    );
}

#[tokio::test]
async fn a_started_evaluation_is_never_run_again() {
    let run = run_with(&deterministic(), None);
    fsutil::write_json(
        &run.paths().final_evaluation().join("invocation.json"),
        &json!({"status": "started"}),
    )
    .unwrap();
    let error = run_final_evaluation(&run.context, &config(2))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("final evaluation already started; refusing rerun"),
        "{error}"
    );
    assert_eq!(run.evaluations(), 0);
    assert_eq!(load_final_manifest(run.paths()).unwrap(), None);
}

#[tokio::test]
async fn a_changed_task_or_a_tampered_program_stops_it_before_anything_is_spent() {
    let run = run_with(&deterministic(), None);
    std::fs::write(
        run.paths().task().join("evaluate.sh"),
        format!("{}\n# more lenient now\n", deterministic()),
    )
    .unwrap();
    let error = run_final_evaluation(&run.context, &config(2))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("the task has changed since the selection was frozen"),
        "{error}"
    );
    assert!(
        !run.paths()
            .final_evaluation()
            .join("invocation.json")
            .exists(),
        "nothing was sealed"
    );
    assert_eq!(run.evaluations(), 0);

    let run = run_with(&deterministic(), None);
    let frozen = run.paths().downstream_program().join("main.py");
    std::fs::write(&frozen, common::block("x = 3.0\ny = -1.0\nbonus = 9\n")).unwrap();
    let error = run_final_evaluation(&run.context, &config(2))
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("no longer matches the digest"), "{error}");
    assert_eq!(run.evaluations(), 0);

    let run = run_with(&deterministic(), None);
    std::fs::remove_file(run.paths().downstream_selection()).unwrap();
    assert!(run_final_evaluation(&run.context, &config(2))
        .await
        .unwrap_err()
        .to_string()
        .contains("no program has been selected"));
    let run = run_with(&deterministic(), None);
    assert!(run_final_evaluation(&run.context, &config(0))
        .await
        .is_err());
    run.context.cancel.store(true, Ordering::SeqCst);
    assert!(run_final_evaluation(&run.context, &config(1))
        .await
        .unwrap_err()
        .to_string()
        .contains("cancelled"));
    assert!(!run
        .paths()
        .final_evaluation()
        .join("invocation.json")
        .exists());
}

#[tokio::test]
async fn a_manifest_that_contradicts_the_selection_or_itself_is_refused() {
    let run = run_with(&deterministic(), None);
    let manifest = run_final_evaluation(&run.context, &config(1))
        .await
        .unwrap();
    let path = run.paths().final_evaluation().join("final_manifest.json");

    let mut foreign = manifest.clone();
    foreign.program_sha256 = "0".repeat(64);
    fsutil::write_json(&path, &foreign).unwrap();
    let error = load_final_manifest(run.paths()).unwrap_err().to_string();
    assert!(
        error.contains("different program than the frozen selection"),
        "{error}"
    );
    assert!(
        run_final_evaluation(&run.context, &config(1))
            .await
            .is_err(),
        "a bad manifest is not silently replaced"
    );

    let mut inconsistent = manifest.clone();
    inconsistent.outcome = NOT_QUALIFIED.to_string();
    fsutil::write_json(&path, &inconsistent).unwrap();
    assert!(load_final_manifest(run.paths())
        .unwrap_err()
        .to_string()
        .contains("do not support"));

    let mut flipped: Value = serde_json::to_value(&manifest).unwrap();
    flipped["episodes"][0]["correct"] = json!(false);
    fsutil::write_json(&path, &flipped).unwrap();
    assert!(load_final_manifest(run.paths()).is_err());

    fsutil::write_json(&path, &manifest).unwrap();
    let restored: FinalManifest = load_final_manifest(run.paths()).unwrap().unwrap();
    assert_eq!(restored, manifest);
}
