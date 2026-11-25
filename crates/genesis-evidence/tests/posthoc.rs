//! Post-hoc analysis end to end over a real run database.

mod common;

use common::{build, by_stage, calls, deterministic, Run};
use genesis_core::fsutil;
use genesis_evidence::posthoc::grounding::claim_is_grounded;
use genesis_evidence::{csv, run_posthoc, PosthocConfig};
use genesis_llm::{LlmClient, MockProvider};
use genesis_scientist::artifacts::PosthocManifest;
use serde_json::json;
use std::sync::atomic::Ordering;
use std::sync::Arc;

const SEED: &str = "x = 1.0\ny = -1.0\nbonus = 0\n";

/// A seed and four children: two improvements, one worse, one incorrect.
fn searched(provider: Arc<MockProvider>) -> Run {
    build(
        &deterministic(),
        None,
        &[
            (0, SEED, 6.0, true),
            (1, "x = 2.0\ny = -1.0\n", 9.0, true),
            (2, "x = 0.0\ny = -1.0\n", 1.0, true),
            (3, "x = 2.0\n", 0.0, false),
            (4, "x = 3.0\ny = -1.0\nbonus = 1\n", 11.0, true),
        ],
        LlmClient::mock(provider),
    )
}

#[tokio::test]
async fn without_a_model_the_families_propose_and_state_their_own_findings() {
    let provider = by_stage(|_, _| None);
    let run = searched(provider.clone());
    let manifest = run_posthoc(&run.context, &PosthocConfig::default())
        .await
        .unwrap();
    assert!(provider.calls().is_empty());

    // Rubrics were never recorded, so five families have data; the cap keeps three.
    let families: Vec<&str> = manifest
        .valid_analyses
        .iter()
        .map(|a| a.spec.family.as_str())
        .collect();
    assert_eq!(
        families,
        [
            "search_trajectory",
            "operator_effectiveness",
            "lineage_of_best"
        ]
    );
    let reasons: Vec<(&str, &str)> = manifest
        .rejected_analyses
        .iter()
        .map(|r| (r.analysis_id.as_str(), r.reason.as_str()))
        .collect();
    assert_eq!(
        reasons,
        [
            ("cost_efficiency", "max_selected_analyses cap reached"),
            ("island_dynamics", "max_selected_analyses cap reached")
        ]
    );

    let trajectory = &manifest.valid_analyses[0];
    assert_eq!(
        trajectory.claim,
        "The best correct score rose from 6.0000 to 11.0000 over 5 evaluated programs, in 2 improvement steps; the \
         last improvement came at generation 4."
    );
    assert_eq!(trajectory.key_numbers["total_improvement"], 5.0);
    assert_eq!(
        trajectory.analysis_dir,
        "posthoc/analyses/search_trajectory"
    );
    assert_eq!(
        trajectory.outputs,
        [
            "posthoc/analyses/search_trajectory/outputs/search_trajectory.csv",
            "posthoc/analyses/search_trajectory/outputs/search_trajectory.svg"
        ]
    );
    for analysis in &manifest.valid_analyses {
        assert!(
            analysis.grounded && claim_is_grounded(&analysis.claim, &analysis.key_numbers),
            "{}",
            analysis.claim
        );
        assert!(run.run_dir().join(&analysis.summary_path).is_file());
        for output in &analysis.outputs {
            assert!(run.run_dir().join(output).is_file(), "missing {output}");
        }
    }
    let table = csv::parse(&run.text(&trajectory.outputs[0]));
    assert_eq!(table.len(), 1 + 5);
    assert_eq!(
        table[0],
        [
            "generation",
            "combined_score",
            "correct",
            "best_so_far",
            "improved_best"
        ]
    );
    assert_eq!(table[5], ["4", "11", "true", "11", "true"]);
    assert!(run.text(&trajectory.outputs[1]).starts_with("<svg"));
    let summary = run.text(&trajectory.summary_path);
    assert!(
        summary.contains("## Finding\n\nThe best correct score rose")
            && summary.contains("| `final_best_score` | 11 |")
    );
    let lineage = &manifest.valid_analyses[2];
    assert_eq!(lineage.key_numbers["edits"], 1.0);

    let analysis_manifest = run.json("posthoc/analyses/search_trajectory/analysis_manifest.json");
    assert_eq!(analysis_manifest["reran_evaluator"], false);
    assert_eq!(analysis_manifest["claim_source"], "template");
    assert_eq!(
        run.json("posthoc/generation/posthoc_bundle.json")["source"],
        "registry"
    );
    assert_eq!(
        run.json("posthoc/pipeline_summary.json")["status"],
        "completed"
    );
    assert_eq!(
        run.json("posthoc/ranking.json")["valid"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    let latest: PosthocManifest =
        fsutil::read_json(&run.run_dir().join("posthoc/latest_manifest.json")).unwrap();
    assert_eq!(latest, manifest);
    assert_eq!(
        run.evaluations(),
        0,
        "post-hoc analysis never evaluates a program"
    );

    // A finished directory is returned as it is.
    assert_eq!(
        run_posthoc(&run.context, &PosthocConfig::default())
            .await
            .unwrap(),
        manifest
    );
    assert!(run.events().contains(&"posthoc_reused".to_string()));
    // A different cap is a different request: selection is redone, the bundle is reused.
    let wider = run_posthoc(
        &run.context,
        &PosthocConfig {
            max_selected_analyses: 5,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(wider.valid_analyses.len(), 5);
}

fn proposing_model() -> Arc<MockProvider> {
    by_stage(|stage, request| {
        match stage {
        "posthoc_generate" => Some(
            json!({"decision": "propose_analyses", "summary": "three angles", "analyses": [
                {"id": "remove_bonus", "title": "Drop the bonus", "family": "component_ablation"},
                {"id": "retrained", "title": "Train again", "family": "search_trajectory", "requires_retrain": true},
                {"id": "rerun", "title": "Evaluate again", "family": "cost_efficiency",
                 "implementation_plan": ["invoke evaluate.py on the best program"]},
                {"id": "clusters", "title": "Error clusters", "family": "error_clustering"},
                {"id": "how_it_climbed", "title": "How the score climbed", "family": "search_trajectory",
                 "hypothesis": "Gains came in a few steps.", "expected_outputs": ["a prose description"]},
                {"id": "who_improved", "title": "Which operators improved", "family": "operator_effectiveness"},
                {"id": "its_lineage", "title": "Where the winner came from", "family": "lineage_of_best"},
                {"id": "rubrics", "title": "Rubrics against scores", "family": "rubric_outcomes"}
            ]})
            .to_string(),
        ),
        "posthoc_select" => Some(
            json!({"decisions": [{"id": "its_lineage", "accepted": false, "reason": "duplicates the trajectory"}]}).to_string(),
        ),
        "posthoc_interpret" => {
            let user = request.last_user();
            if user.contains("How the score climbed") {
                // Every number is a key number: the claim and the paragraph are kept.
                Some(
                    json!({"claim": "The search evaluated 5 programs and the best score reached 11.0.",
                           "interpretation": "Two improvement steps were enough; 80% of the programs were correct."})
                    .to_string(),
                )
            } else {
                // 87.3 is not a key number: the template claim is used instead.
                Some(json!({"claim": "Diffs improved 87.3% of the time.", "interpretation": "All 4 proposals were tabulated."}).to_string())
            }
        }
        _ => None,
    }
    })
}

#[tokio::test]
async fn a_model_proposes_code_applies_the_rules_and_numbers_stay_grounded() {
    let provider = proposing_model();
    let run = searched(provider.clone());
    let config = PosthocConfig {
        models: vec!["analyst".into()],
        ..Default::default()
    };
    let manifest = run_posthoc(&run.context, &config).await.unwrap();

    let ids: Vec<&str> = manifest
        .valid_analyses
        .iter()
        .map(|a| a.spec.id.as_str())
        .collect();
    assert_eq!(ids, ["how_it_climbed", "who_improved"]);
    let reasons: Vec<(&str, &str)> = manifest
        .rejected_analyses
        .iter()
        .map(|r| (r.analysis_id.as_str(), r.reason.as_str()))
        .collect();
    assert_eq!(
        reasons,
        [
            ("remove_bonus", "ablations belong to the ablation study"),
            ("retrained", "requires retraining"),
            ("rerun", "the plan retrains or evaluates again; post-hoc analysis uses existing artifacts only"),
            ("clusters", "unknown analysis family `error_clustering`"),
            ("its_lineage", "rejected by the reviewer: duplicates the trajectory"),
            ("rubrics", "not_executed: this run holds no data for the family"),
        ]
    );
    assert_eq!(
        ["posthoc_generate", "posthoc_select", "posthoc_interpret"]
            .map(|stage| calls(&provider, stage)),
        [1, 1, 2]
    );

    let climbed = &manifest.valid_analyses[0];
    assert_eq!(
        climbed.claim,
        "The search evaluated 5 programs and the best score reached 11.0."
    );
    assert_eq!(
        climbed.spec.expected_outputs,
        [
            "outputs/search_trajectory.csv",
            "outputs/search_trajectory.svg"
        ]
    );
    assert!(run
        .text(&climbed.summary_path)
        .contains("Two improvement steps were enough; 80% of the programs were correct."));
    assert_eq!(
        run.json("posthoc/analyses/how_it_climbed/analysis_manifest.json")["claim_source"],
        "model"
    );

    let operators = &manifest.valid_analyses[1];
    assert!(
        operators.claim.starts_with("Of 4 evaluated proposals,"),
        "{}",
        operators.claim
    );
    assert!(!operators.claim.contains("87.3"));
    let summary = run.text(&operators.summary_path);
    assert!(
        summary.contains("All 4 proposals were tabulated."),
        "a grounded paragraph is kept on its own"
    );
    assert_eq!(
        run.json("posthoc/analyses/who_improved/analysis_manifest.json")["claim_source"],
        "template"
    );
    assert!(run
        .events()
        .contains(&"posthoc_text_not_grounded".to_string()));

    let interpret = provider
        .calls()
        .into_iter()
        .find(|call| call.stage == "posthoc_interpret")
        .unwrap();
    assert!(interpret.last_user().contains("\"final_best_score\": 11.0"));
    let generate = provider
        .calls()
        .into_iter()
        .find(|call| call.stage == "posthoc_generate")
        .unwrap();
    assert!(generate
        .last_user()
        .contains("- `rubric_outcomes` (no data in this run)"));
    assert!(generate
        .last_user()
        .contains("- `search_trajectory` (has data)"));
}

#[tokio::test]
async fn nothing_valid_is_a_result_and_publishes_no_manifest() {
    let provider = by_stage(|stage, _| {
        match stage {
        "posthoc_generate" => Some(json!({"decision": "no_relevant_analyses", "summary": "nothing to add", "analyses": []}).to_string()),
        _ => None,
    }
    });
    let run = searched(provider);
    let config = PosthocConfig {
        models: vec!["analyst".into()],
        ..Default::default()
    };
    let manifest = run_posthoc(&run.context, &config).await.unwrap();
    assert!(manifest.valid_analyses.is_empty() && manifest.rejected_analyses.is_empty());
    assert!(!run.run_dir().join("posthoc/latest_manifest.json").exists());
    assert_eq!(
        run.json("posthoc/pipeline_summary.json")["status"],
        "no_valid_analyses"
    );
    assert_eq!(
        run.json("posthoc/generation/posthoc_bundle.json")["decision"],
        "no_relevant_analyses"
    );
    assert_eq!(
        run_posthoc(&run.context, &config).await.unwrap(),
        manifest,
        "the empty result is reused too"
    );
}

#[tokio::test]
async fn an_unreachable_model_falls_back_to_the_registry() {
    let provider = by_stage(|_, _| None);
    let run = searched(provider.clone());
    let config = PosthocConfig {
        models: vec!["analyst".into()],
        max_selected_analyses: 1,
        ..Default::default()
    };
    let manifest = run_posthoc(&run.context, &config).await.unwrap();
    assert_eq!(manifest.valid_analyses.len(), 1);
    assert_eq!(manifest.valid_analyses[0].spec.id, "search_trajectory");
    assert_eq!(
        run.json("posthoc/generation/posthoc_bundle.json")["source"],
        "registry"
    );
    let events = run.events();
    assert!(
        events.contains(&"posthoc_generation_failed".to_string())
            && events.contains(&"posthoc_selection_failed".to_string())
    );
}

#[tokio::test]
async fn a_missing_database_a_missing_selection_and_cancellation_are_errors() {
    let run = searched(by_stage(|_, _| None));
    run.context.cancel.store(true, Ordering::SeqCst);
    let error = run_posthoc(&run.context, &PosthocConfig::default())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("interrupted"), "{error}");

    let run = searched(by_stage(|_, _| None));
    let investigation = run.paths().investigation();
    for entry in std::fs::read_dir(&investigation).unwrap().flatten() {
        std::fs::remove_file(entry.path()).unwrap();
    }
    let error = run_posthoc(&run.context, &PosthocConfig::default())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("needs the run database"), "{error}");

    let run = searched(by_stage(|_, _| None));
    std::fs::remove_file(run.paths().downstream_selection()).unwrap();
    assert!(run_posthoc(&run.context, &PosthocConfig::default())
        .await
        .unwrap_err()
        .to_string()
        .contains("no program has been selected"));
}
