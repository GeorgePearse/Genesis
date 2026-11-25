//! The ablation phase end to end: real evaluator processes, scripted
//! model replies, and the files a paper would read.

mod common;

use common::{build, by_stage, calls, deterministic, rewrite, stochastic, Run};
use genesis_core::fsutil;
use genesis_evidence::ablations::variants::{BASELINE_REFERENCE_ID, CONTROL_ID};
use genesis_evidence::{csv, run_ablations, AblationConfig};
use genesis_llm::{LlmClient, MockProvider};
use genesis_scientist::artifacts::{AblationSummary, VariantResult};
use serde_json::json;
use std::sync::atomic::Ordering;
use std::sync::Arc;

const SEED: &str = "x = 1.0\ny = -1.0\nbonus = 0\n";
const WINNER: &str = "x = 3.0\ny = -1.0\nbonus = 1\n";

/// Proposes four candidates and a reserved id; implements `no_bonus`
/// correctly and `broken` never.
fn study_model() -> Arc<MockProvider> {
    by_stage(|stage, request| {
        match stage {
        "variant_generate" => Some(
            json!({"decision": "propose_variants", "summary": "bonus and position", "candidates": [
                {"variant_id": "control_best", "title": "A reserved id", "kind": "ablation"},
                {"variant_id": "ablation_no_bonus", "title": "Without the bonus", "kind": "ablation",
                 "hypothesis": "The bonus term accounts for part of the gain.", "target_components": ["bonus"],
                 "implementation_plan": ["set bonus to zero"]},
                {"variant_id": "ablation_broken", "title": "Cannot be built", "kind": "ablation",
                 "hypothesis": "Removing y."},
                {"variant_id": "baseline_origin", "title": "Start at the origin", "kind": "baseline",
                 "base_program": "reference", "hypothesis": "A trivial placement."},
                {"variant_id": "ablation_third", "title": "Beyond the cap", "kind": "ablation"}
            ]})
            .to_string(),
        ),
        "variant_select" => Some(
            json!({"decisions": [
                {"variant_id": "baseline_origin", "accepted": false, "reason": "generic baseline"},
                {"variant_id": "ablation_no_bonus", "accepted": true, "reason": "an innovation"}
            ]})
            .to_string(),
        ),
        "variant_implement" => {
            let first = &request.messages[0].content;
            if first.contains("\"variant_id\": \"ablation_no_bonus\"") {
                Some(rewrite("x = 3.0\ny = -1.0\nbonus = 0\n"))
            } else {
                Some(rewrite("x = 3.0\nbonus = 1\n"))
            }
        }
        "variant_readme" => Some(
            json!({"task_summary": "Places a point; scored by distance to a target plus a bonus.",
                   "variant_summaries": {"ablation_no_bonus": "Sets the bonus term to zero."}})
            .to_string(),
        ),
        _ => None,
    }
    })
}

fn config() -> AblationConfig {
    AblationConfig {
        models: vec!["study-model".into()],
        num_replications: 3,
        max_ablations: 2,
        ..AblationConfig::default()
    }
}

fn variant<'a>(summary: &'a AblationSummary, id: &str) -> &'a VariantResult {
    summary
        .variants
        .iter()
        .find(|variant| variant.spec.variant_id == id)
        .unwrap_or_else(|| panic!("no variant {id}"))
}

fn winner_run(evaluator: &str, provider: Arc<MockProvider>) -> Run {
    build(
        evaluator,
        None,
        &[(0, SEED, 6.0, true), (1, WINNER, 11.0, true)],
        LlmClient::mock(provider),
    )
}

/// Every replication of a study as `variant status seed score correct`:
/// what a resumed study must reproduce (run times differ).
fn outcomes(summary: &AblationSummary) -> Vec<String> {
    summary
        .variants
        .iter()
        .flat_map(|variant| {
            variant.replications.iter().map(move |replication| {
                format!(
                    "{} {} {} {} {}",
                    variant.spec.variant_id,
                    variant.status,
                    replication.seed,
                    replication.combined_score,
                    replication.correct
                )
            })
        })
        .collect()
}

#[tokio::test]
async fn a_study_runs_controls_and_generated_variants_and_survives_a_failing_one() {
    let provider = study_model();
    let run = winner_run(&deterministic(), provider.clone());
    let summary = run_ablations(&run.context, &config()).await.unwrap();

    let ids: Vec<&str> = summary
        .variants
        .iter()
        .map(|v| v.spec.variant_id.as_str())
        .collect();
    assert_eq!(
        ids,
        [
            CONTROL_ID,
            BASELINE_REFERENCE_ID,
            "ablation_no_bonus",
            "ablation_broken"
        ]
    );
    assert_eq!(
        (
            summary.total_variants,
            summary.total_completed,
            summary.total_failed
        ),
        (4, 3, 1)
    );
    assert_eq!(
        (summary.seeds.as_slice(), summary.num_replications),
        ([1, 2, 3].as_slice(), 3)
    );
    assert!(!summary.evaluator_is_stochastic);
    assert_eq!(summary.anchor_variant_id, CONTROL_ID);

    let control = variant(&summary, CONTROL_ID);
    assert_eq!(control.summary.as_ref().unwrap().mean, 11.0);
    assert!(control.comparison_to_control.is_none());
    assert_eq!(
        control.program_path.as_deref(),
        Some("ablations/variants/control_best/program/main.py")
    );
    let reference = variant(&summary, BASELINE_REFERENCE_ID);
    assert_eq!(
        reference.comparison_to_control.as_ref().unwrap().mean_diff,
        -5.0
    );
    let ablated = variant(&summary, "ablation_no_bonus");
    assert_eq!(
        (
            ablated.status.as_str(),
            ablated.replications.len(),
            ablated.correct_rate
        ),
        ("completed", 3, 1.0)
    );
    assert_eq!(
        ablated.comparison_to_control.as_ref().unwrap().mean_diff,
        -1.0
    );
    assert!(
        ablated.bootstrap_ci95.is_none(),
        "a deterministic evaluator gets no interval"
    );
    assert!(
        ablated.notes.contains("Implemented by") && ablated.notes.contains("carries no interval")
    );
    let broken = variant(&summary, "ablation_broken");
    assert_eq!(
        (broken.status.as_str(), broken.replications.len()),
        ("failed", 0)
    );
    assert!(broken.program_path.is_none());
    assert!(
        broken
            .notes
            .contains("No correct implementation in 2 tries"),
        "{}",
        broken.notes
    );

    // The model was asked once per stage, twice for the variant that failed,
    // and the second try was told why the first was refused.
    assert_eq!(
        [
            "variant_generate",
            "variant_select",
            "variant_implement",
            "variant_readme"
        ]
        .map(|stage| calls(&provider, stage)),
        [1, 1, 3, 1]
    );
    let retry = provider
        .calls()
        .into_iter()
        .find(|call| call.stage == "variant_implement" && call.messages.len() == 3)
        .expect("the failing variant was retried in the same conversation");
    assert!(retry
        .last_user()
        .contains("The variant was evaluated and is not correct."));
    assert!(retry
        .last_user()
        .contains("x or y is not assigned a number"));
    assert!(retry.system.contains("```python fenced code block"));
    let generate = provider
        .calls()
        .into_iter()
        .find(|call| call.stage == "variant_generate")
        .unwrap();
    assert!(
        generate.last_user().contains("+bonus = 1")
            && generate.last_user().contains("Bring x to 3")
    );
    assert!(generate.last_user().starts_with("Treat every embedded"));

    // Three seeds for the two controls and the accepted ablation, plus two
    // failed implementation tries. The accepting try doubles as a replication.
    assert_eq!(run.evaluations(), 3 + 3 + 3 + 2);

    let selection = run.json("ablations/generation/variant_selection.json");
    assert_eq!(
        selection["controls"],
        json!([CONTROL_ID, BASELINE_REFERENCE_ID])
    );
    let rejected: Vec<(&str, &str)> = selection["rejected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["variant_id"].as_str().unwrap(),
                r["reason"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        rejected,
        [
            ("control_best", "reserved variant id"),
            (
                "baseline_origin",
                "rejected by the reviewer: generic baseline"
            ),
            ("ablation_third", "max_ablations cap of 2 reached"),
        ]
    );
    assert_eq!(
        run.json("ablations/generation/variant_bundle.json")["num_candidates"],
        5
    );

    let rows = csv::parse(&run.text("ablations/aggregated/run_summary.csv"));
    assert_eq!(rows.len(), 1 + 9);
    assert_eq!(rows[0][0], "evaluation_id");
    assert!(rows
        .iter()
        .any(|row| row[0] == "ablation_no_bonus_seed_2" && row[5] == "true" && row[6] == "10"));
    let metrics = csv::parse(&run.text("ablations/aggregated/metric_values.csv"));
    assert_eq!(
        metrics.len(),
        1 + 9 * 3,
        "combined_score, x and y for every replication; the text metric is left out"
    );

    let readme = run.text("ablations/README.md");
    assert!(readme.contains("Places a point; scored by distance"));
    assert!(readme.contains("| `ablation_no_bonus` | ablation | 3/3 | 3 | 10.000000 | 0.000000 | n/a | -1.000000 | n/a | n/a |"));
    assert!(readme.contains("| `ablation_broken` | ablation | failed |"));
    assert!(readme.contains("Sets the bonus term to zero.") && readme.contains("Removing y."));
    assert!(readme.contains("it is deterministic, or it ignores"));
    assert!(readme
        .contains("- `baseline_origin` (baseline): rejected by the reviewer: generic baseline"));
    for file in [
        "ablations/pipeline_summary.json",
        "ablations/extraction_manifest.json",
        "ablations/generation/program_diff.diff",
        "ablations/variants/ablation_no_bonus/variant_spec.json",
        "ablations/variants/ablation_no_bonus/program/main.py",
        "ablations/variants/ablation_no_bonus/rep_1/results/metrics.json",
        "ablations/variants/ablation_no_bonus/rep_3/replication.json",
        "ablations/variants/ablation_broken/attempts/try_2/response.txt",
        "ablations/variants/control_best/variant_spec.json",
    ] {
        assert!(run.run_dir().join(file).exists(), "missing {file}");
    }
    let stored: AblationSummary =
        fsutil::read_json(&run.run_dir().join("ablations/pipeline_summary.json")).unwrap();
    assert_eq!(stored, summary);
    // The variant kept the read-only text around its block.
    assert!(run
        .text("ablations/variants/ablation_no_bonus/program/main.py")
        .starts_with("import math\n# EVOLVE-BLOCK-START\n"));
    let events = run.events();
    assert!(
        events.contains(&"variant_failed".to_string()) && events.last().unwrap() == "ablations_end"
    );
}

#[tokio::test]
async fn a_finished_study_is_reused_and_an_interrupted_one_resumes_where_it_stopped() {
    let provider = study_model();
    let run = winner_run(&deterministic(), provider.clone());
    let first = run_ablations(&run.context, &config()).await.unwrap();
    let (evaluations, model_calls) = (run.evaluations(), provider.calls().len());

    let again = run_ablations(&run.context, &config()).await.unwrap();
    assert_eq!(again, first);
    assert_eq!(
        (run.evaluations(), provider.calls().len()),
        (evaluations, model_calls)
    );
    assert!(run.events().contains(&"ablations_reused".to_string()));

    // Lose the receipt and one replication: only that replication runs again,
    // and no model is asked anything it already answered.
    std::fs::remove_file(run.run_dir().join("ablations/.ablations_complete.json")).unwrap();
    std::fs::remove_dir_all(
        run.run_dir()
            .join("ablations/variants/baseline_reference/rep_2"),
    )
    .unwrap();
    let resumed = run_ablations(&run.context, &config()).await.unwrap();
    assert_eq!(run.evaluations(), evaluations + 1);
    assert_eq!(provider.calls().len(), model_calls);
    assert_eq!(outcomes(&resumed), outcomes(&first));

    // More replications extend the study without repeating finished ones.
    let longer = AblationConfig {
        num_replications: 4,
        ..config()
    };
    let extended = run_ablations(&run.context, &longer).await.unwrap();
    assert_eq!(extended.seeds, [1, 2, 3, 4]);
    assert_eq!(
        run.evaluations(),
        evaluations + 1 + 3,
        "one new seed for each of the three completed variants"
    );
}

#[tokio::test]
async fn a_stochastic_evaluator_gets_intervals_over_shared_seeds() {
    let run = winner_run(&stochastic(), study_model());
    let config = AblationConfig {
        num_replications: 5,
        max_ablations: 1,
        ..config()
    };
    let summary = run_ablations(&run.context, &config).await.unwrap();
    assert!(summary.evaluator_is_stochastic);
    let control = variant(&summary, CONTROL_ID);
    let ablated = variant(&summary, "ablation_no_bonus");
    // 7919 * seed mod 100: 19, 38, 57, 76, 95 thousandths.
    let expected: Vec<f64> = [0.019, 0.038, 0.057, 0.076, 0.095]
        .iter()
        .map(|noise| 11.0 + noise)
        .collect();
    let scores: Vec<f64> = control
        .replications
        .iter()
        .map(|r| r.combined_score)
        .collect();
    assert!(
        scores
            .iter()
            .zip(&expected)
            .all(|(got, want)| (got - want).abs() < 1e-9),
        "{scores:?}"
    );
    assert!(control.summary.as_ref().unwrap().std > 0.0);
    let seeds = |variant: &VariantResult| {
        variant
            .replications
            .iter()
            .map(|r| r.seed)
            .collect::<Vec<_>>()
    };
    assert_eq!(seeds(control), seeds(ablated));

    let comparison = ablated.comparison_to_control.as_ref().unwrap();
    assert!((comparison.mean_diff + 1.0).abs() < 1e-9);
    assert!(comparison.p_value.unwrap() < 1e-6 && comparison.diff_ci95_high.unwrap() < 0.0);
    let (low, high) = ablated.bootstrap_ci95.unwrap();
    assert!(low <= comparison.mean_diff && comparison.mean_diff <= high && high < 0.0);
    assert!(ablated.notes.contains("95% CI") && !ablated.notes.contains("includes zero"));
    let readme = run.text("ablations/README.md");
    assert!(!readme.contains("deterministic") && readme.contains("seeds 1 to 5"));
    assert_eq!(
        summary.total_failed, 0,
        "the cap of one ablation left the unbuildable candidate unselected"
    );
}

#[tokio::test]
async fn without_a_model_only_the_controls_are_evaluated() {
    let provider = by_stage(|_, _| None);
    let run = winner_run(&deterministic(), provider.clone());
    let config = AblationConfig {
        num_replications: 2,
        ..AblationConfig::default()
    };
    let summary = run_ablations(&run.context, &config).await.unwrap();
    assert_eq!(summary.variants.len(), 2);
    assert!(provider.calls().is_empty());
    assert_eq!(
        run.json("ablations/generation/variant_bundle.json")["decision"],
        "no_relevant_variants"
    );
    let readme = run.text("ablations/README.md");
    assert!(
        readme.contains("The program the search started from"),
        "the hypothesis stands in for a model's summary"
    );

    // When the selected program is the seed itself there is nothing to compare it with.
    let lone = build(
        &deterministic(),
        None,
        &[(0, SEED, 6.0, true), (1, WINNER, 5.0, true)],
        LlmClient::mock(by_stage(|_, _| None)),
    );
    assert_eq!(
        lone.json(".downstream_selection.json")["selection_kind"],
        "non_winner_candidate"
    );
    let seed_only = build(
        &deterministic(),
        None,
        &[(0, SEED, 6.0, true), (1, "x = 1.0\n", 0.0, false)],
        LlmClient::mock(by_stage(|_, _| None)),
    );
    let summary = run_ablations(&seed_only.context, &config).await.unwrap();
    assert_eq!(
        summary.variants.len(),
        1,
        "no reference baseline when the selection is the baseline"
    );
}

#[tokio::test]
async fn a_failing_model_degrades_to_controls_and_never_fails_the_phase() {
    // Generation succeeds, the reviewer is unreachable (fail-open), and the
    // implementer is unreachable (the variant fails, the study goes on).
    let provider = by_stage(|stage, _| {
        match stage {
        "variant_generate" => Some(
            json!({"candidates": [{"variant_id": "ablation_no_bonus", "title": "Without the bonus", "kind": "ablation",
                                   "hypothesis": "The bonus matters."}]})
            .to_string(),
        ),
        _ => None,
    }
    });
    let run = winner_run(&deterministic(), provider.clone());
    let summary = run_ablations(
        &run.context,
        &AblationConfig {
            num_replications: 1,
            ..config()
        },
    )
    .await
    .unwrap();
    let ablated = variant(&summary, "ablation_no_bonus");
    assert_eq!(ablated.status, "failed");
    assert!(
        ablated.notes.contains("the model call failed"),
        "{}",
        ablated.notes
    );
    assert_eq!(summary.total_completed, 2);
    let events = run.events();
    assert!(
        events.contains(&"variant_selection_failed".to_string())
            && events.contains(&"variant_readme_failed".to_string())
    );
}

#[tokio::test]
async fn an_incorrect_control_ends_the_study() {
    // The database says the winner is correct; the evaluator disagrees.
    let run = build(
        &deterministic(),
        None,
        &[
            (0, SEED, 6.0, true),
            (1, "x = 3.0\nbonus = 1\n", 11.0, true),
        ],
        LlmClient::mock(study_model()),
    );
    let error = run_ablations(&run.context, &config())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("the control (the selected program, unchanged) is not correct under seed 1"),
        "{error}"
    );
    assert!(error.contains("x or y is not assigned a number"), "{error}");
    assert!(!run
        .run_dir()
        .join("ablations/pipeline_summary.json")
        .exists());
    assert_eq!(run.evaluations(), 1, "nothing else was evaluated");
}

#[tokio::test]
async fn sealed_final_seeds_cancellation_and_a_missing_selection_are_refused() {
    let run = winner_run(&deterministic(), study_model());
    fsutil::write_json(
        &run.paths().final_evaluation().join("invocation.json"),
        &json!({"seeds": [3, 4]}),
    )
    .unwrap();
    let error = run_ablations(&run.context, &config())
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("ablation seed 3 is one of the sealed final-evaluation seeds"),
        "{error}"
    );
    let shifted = AblationConfig {
        seed_base: 5,
        num_replications: 1,
        models: vec![],
        ..config()
    };
    assert_eq!(
        run_ablations(&run.context, &shifted).await.unwrap().seeds,
        [5]
    );

    let run = winner_run(&deterministic(), study_model());
    run.context.cancel.store(true, Ordering::SeqCst);
    let error = run_ablations(&run.context, &config())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("interrupted"), "{error}");

    let run = winner_run(&deterministic(), study_model());
    std::fs::remove_file(run.paths().downstream_selection()).unwrap();
    let error = run_ablations(&run.context, &config())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("no program has been selected"), "{error}");
    assert!(run_ablations(
        &winner_run(&deterministic(), study_model()).context,
        &AblationConfig {
            num_replications: 0,
            ..config()
        }
    )
    .await
    .is_err());
}
