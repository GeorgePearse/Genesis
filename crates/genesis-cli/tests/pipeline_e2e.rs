//! The whole pipeline end to end: a real evaluator process, a scripted reply
//! for every model stage, and the files each phase leaves behind.

use genesis_cli::pipeline::{run_pipeline_with, PipelineConfig};
use genesis_core::fsutil;
use genesis_llm::{LlmError, LlmRequest, MockProvider};
use genesis_scientist::artifacts::{
    AblationSummary, FinalManifest, InvestigationCompletion, PaperManifest, PosthocManifest,
    Review, OUTCOME_WINNER,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Scores `10 - ((x - 3)^2 + (y + 1)^2) + bonus`; a program that does not
/// assign `x` and `y` is incorrect.
const EVALUATOR: &str = r#"
program="$1"; results="$2"
x=$(sed -n 's/^x = \(-\{0,1\}[0-9.]*\)$/\1/p' "$program" | head -n 1)
y=$(sed -n 's/^y = \(-\{0,1\}[0-9.]*\)$/\1/p' "$program" | head -n 1)
bonus=$(sed -n 's/^bonus = \([0-9.]*\)$/\1/p' "$program" | head -n 1)
if [ -z "$x" ] || [ -z "$y" ]; then
  echo '{"combined_score": 0.0}' > "$results/metrics.json"
  echo '{"correct": false, "error": "x or y is not assigned a number"}' > "$results/correct.json"
  exit 0
fi
score=$(awk -v x="$x" -v y="$y" -v b="${bonus:-0}" 'BEGIN { printf "%.6f", 10 - ((x - 3) * (x - 3) + (y + 1) * (y + 1)) + b }')
echo "{\"combined_score\": $score, \"public\": {\"x\": $x, \"y\": $y}}" > "$results/metrics.json"
echo '{"correct": true}' > "$results/correct.json"
"#;

fn program(body: &str) -> String {
    format!("# EVOLVE-BLOCK-START\n{body}# EVOLVE-BLOCK-END\n")
}

fn proposal(body: &str, name: &str) -> String {
    format!(
        "<NAME>\n{name}\n</NAME>\n<DESCRIPTION>\nmove toward the target\n</DESCRIPTION>\n<CODE>\n```python\n{}```\n</CODE>",
        program(body)
    )
}

fn review_form(overall: f64, decision: &str) -> String {
    let form = json!({
        "Summary": "The paper reports an automated program search on one task.", "Strengths": ["clear protocol"],
        "Weaknesses": ["a single task"], "Originality": 2, "Quality": 3, "Clarity": 3, "Significance": 2,
        "Questions": ["Does it transfer?"], "Limitations": ["scope"], "Ethical Concerns": false, "Soundness": 3,
        "Presentation": 3, "Contribution": 2, "Overall": overall, "Confidence": 4, "Decision": decision
    });
    format!("THOUGHT:\nThe evidence is narrow.\n\nREVIEW JSON:\n```json\n{form}\n```")
}

const SUPPLEMENT: &str = "### Protocol\nEvery candidate program was scored by the task evaluator, and the harness, not the proposing model, ran it. {{REF:TABLE:run_summary}} lists the counts of the search.";

fn draft(abstract_text: &str) -> String {
    let claims = json!({"claims": [{
        "claim": "The search raised the combined score from 6 to 11",
        "source_ids": ["result.seed_score", "result.best_correct_score"],
        "citation_keys": []
    }]});
    format!(
        "<TITLE>Automated search improves a three-parameter program</TITLE>\n<ABSTRACT>{abstract_text}</ABSTRACT>\n\
         <INTRODUCTION>We study whether an automated search can improve a small program without human edits.</INTRODUCTION>\n\
         <RELATED_WORK>Program search with language models is an active area.</RELATED_WORK>\n\
         <METHOD>The search proposes edits to the program and keeps those the evaluator scores higher.</METHOD>\n\
         <RESULTS>The best program reaches 11 ({{{{REF:TABLE:results}}}}).\n\n{{{{FIGURE:improvement}}}}</RESULTS>\n\
         <LIMITATIONS>The study covers a single task.</LIMITATIONS>\n\
         <CONCLUSION>The search improved the program on this task.</CONCLUSION>\n\
         <SUPPLEMENTARY>{SUPPLEMENT}</SUPPLEMENTARY>\n<CLAIMS>\n{claims}\n</CLAIMS>"
    )
}

/// Answers every stage of every phase. An unscripted stage is an error the
/// client does not retry, so a new call site fails this test by name.
fn scripted() -> Arc<MockProvider> {
    let proposals = AtomicUsize::new(0);
    let ideas = AtomicUsize::new(0);
    MockProvider::new(move |request: &LlmRequest| {
        let reply = match request.stage.as_str() {
            "idea" => {
                let n = ideas.fetch_add(1, Ordering::SeqCst) + 1;
                json!({
                    "idea_id": format!("toward_target_{n}"),
                    "title": format!("Move toward the target, step {n}"),
                    "hypothesis": "x closer to 3 scores higher, and the bonus adds to the score",
                    "implementation_plan": ["increase x", "set the bonus"],
                    "target_rubrics": [
                        {"id": "assigns_x", "description": "assigns x", "type": "implementation",
                         "verification": {"type": "source_contains", "patterns": ["x ="]}},
                        {"id": "good_score", "description": "combined_score >= 6", "type": "metric"}
                    ],
                    "notes_for_agent": "keep the markers"
                })
                .to_string()
            }
            "proposal" => match proposals.fetch_add(1, Ordering::SeqCst) + 1 {
                1 => proposal("x = 2.0\ny = -1.0\nbonus = 0\n", "step_one"),
                2 => proposal("x = 3.0\ny = -1.0\nbonus = 0\n", "step_two"),
                n => proposal(&format!("x = 3.0\ny = -1.0\nbonus = 1\n# attempt {n}\n"), "with_bonus"),
            },
            "conclusions" => json!({
                "gate": "updated", "trigger": "b", "reason": "moving x toward three is a new, working direction",
                "conclusions": "Moving x toward three and setting the bonus both raise the score; do not lower x."
            })
            .to_string(),
            "variant_generate" => json!({"decision": "propose_variants", "summary": "the bonus term", "candidates": [
                {"variant_id": "ablation_no_bonus", "title": "Without the bonus", "kind": "ablation",
                 "hypothesis": "The bonus term accounts for part of the gain.", "target_components": ["bonus"],
                 "implementation_plan": ["set bonus to zero"]}
            ]})
            .to_string(),
            "variant_select" => json!({"decisions": [
                {"variant_id": "ablation_no_bonus", "accepted": true, "reason": "isolates one change"}
            ]})
            .to_string(),
            "variant_implement" => {
                format!("Changed one thing.\n```python\n{}```\n", program("x = 3.0\ny = -1.0\nbonus = 0\n"))
            }
            "variant_readme" => json!({
                "task_summary": "Places a point; scored by distance to a target plus a bonus.",
                "variant_summaries": {"ablation_no_bonus": "Sets the bonus term to zero."}
            })
            .to_string(),
            "posthoc_generate" => json!({"decision": "propose_analyses", "summary": "one angle", "analyses": [
                {"id": "how_it_climbed", "title": "How the score climbed", "family": "search_trajectory",
                 "hypothesis": "Gains came in a few steps."}
            ]})
            .to_string(),
            "posthoc_select" => {
                json!({"decisions": [{"id": "how_it_climbed", "accepted": true, "reason": "bears on the result"}]}).to_string()
            }
            "posthoc_interpret" => json!({
                "claim": "The best correct score reached 11.0.",
                "interpretation": "Each improvement moved the point closer to the target."
            })
            .to_string(),
            "reference_plan" => json!({"central_topic": "program search", "queries": [
                {"query": "LLM program evolution", "category": "core_method", "rationale": "the method"}
            ]})
            .to_string(),
            "paper_strategy" => json!({
                "recommended_title": "Automated search improves a three-parameter program",
                "central_question": "Does the search improve the program?",
                "central_thesis": "The search found a higher-scoring correct program.",
                "narrative_rationale": "report the measured gain", "results_focus": ["result.best_correct_score"],
                "notes_for_writer": []
            })
            .to_string(),
            "paper_draft" => draft("An automated program search raised the combined score from 6 to 11."),
            "paper_revision" => {
                "<ABSTRACT>After revision: on this single task, an automated search raised the combined score from 6 to 11.</ABSTRACT>"
                    .to_string()
            }
            "review" if request.last_user().contains("After revision:") => review_form(6.0, "Accept"),
            "meta_review" if request.last_user().contains("\"Overall\": 6") => review_form(6.0, "Accept"),
            "review" | "meta_review" => review_form(4.0, "Reject"),
            other => {
                return Err(LlmError::Rejected {
                    provider: "mock".into(),
                    status: 400,
                    message: format!("no scripted reply for stage `{other}`"),
                })
            }
        };
        Ok(reply)
    })
}

fn task(dir: &Path) -> PathBuf {
    let task = dir.join("point_task");
    std::fs::create_dir_all(&task).unwrap();
    std::fs::write(
        task.join("task.md"),
        "Bring x to 3 and y to -1; a bonus adds to the score.\n",
    )
    .unwrap();
    std::fs::write(
        task.join("initial.py"),
        program("x = 1.0\ny = -1.0\nbonus = 0\n"),
    )
    .unwrap();
    std::fs::write(task.join("evaluate.sh"), EVALUATOR).unwrap();
    task
}

fn overrides(dir: &Path, task: &Path, phases: &str) -> Vec<String> {
    [
        format!("task.dir={}", task.display()),
        format!("paths.results_root={}", dir.join("results").display()),
        format!("execution.phases=[{phases}]"),
        "execution.seed=3".to_string(),
        "logging.format=quiet".to_string(),
        "models=[scripted-model]".to_string(),
        "investigation.evolution.patch_types=[full]".to_string(),
        "investigation.evolution.patch_type_probs=[1.0]".to_string(),
        "investigation.evolution.num_generations=5".to_string(),
        "investigation.evolution.max_evaluation_jobs=1".to_string(),
        "investigation.database.num_islands=1".to_string(),
        "investigation.conclusions_interval=2".to_string(),
        "ablations.num_replications=3".to_string(),
        "final_evaluation.num_replications=3".to_string(),
        "paper.retrieve_references=false".to_string(),
        "paper.compile_pdf=false".to_string(),
        "review.reviewer_count=2".to_string(),
    ]
    .to_vec()
}

fn stage_calls(provider: &MockProvider, stage: &str) -> usize {
    provider
        .calls()
        .iter()
        .filter(|call| call.stage == stage)
        .count()
}

#[tokio::test]
async fn every_phase_runs_and_leaves_its_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let task = task(dir.path());
    let phases = "task_drafting, investigation, ablations, final_evaluation, posthoc, paper, review, revision";
    let config = PipelineConfig::load(None, &overrides(dir.path(), &task, phases)).unwrap();
    let provider = scripted();
    let report = run_pipeline_with(
        config,
        Arc::new(AtomicBool::new(false)),
        Some(provider.clone()),
    )
    .await
    .unwrap();

    let statuses: Vec<(String, String, String)> = report
        .phases
        .iter()
        .map(|phase| {
            (
                phase.phase.clone(),
                phase.status.label().to_string(),
                phase.status.reason().unwrap_or_default().to_string(),
            )
        })
        .collect();
    let labels: Vec<(&str, &str)> = statuses
        .iter()
        .map(|(phase, label, _)| (phase.as_str(), label.as_str()))
        .collect();
    assert_eq!(
        labels,
        [
            ("task_drafting", "skipped"),
            ("investigation", "completed"),
            ("ablations", "completed"),
            ("final_evaluation", "completed"),
            ("posthoc", "completed"),
            ("paper", "completed"),
            ("review", "completed"),
            ("revision", "completed"),
        ],
        "{statuses:#?}"
    );
    assert_eq!(report.status, "completed");
    assert_eq!(report.exit_code(), 0);

    let run = report.run_dir.clone();
    let read = |relative: &str| -> Value { fsutil::read_json(&run.join(relative)).unwrap() };

    // The search: the seed scores 6, the third proposal 11.
    let receipt: InvestigationCompletion =
        fsutil::read_json(&run.join(".investigation_completion.json")).unwrap();
    assert_eq!(
        (receipt.status.as_str(), receipt.outcome.as_deref()),
        ("completed", Some(OUTCOME_WINNER))
    );
    let summary = read("investigation/run_summary.json");
    assert_eq!(summary["best"]["combined_score"], 11.0);
    assert_eq!(summary["best"]["generation"], 3);
    assert!(run.join("investigation/conclusions.md").is_file());
    let selection = read(".downstream_selection.json");
    assert_eq!(
        (
            selection["combined_score"].clone(),
            selection["baseline_score"].clone()
        ),
        (json!(11.0), json!(6.0))
    );

    // Ablations: removing the bonus costs exactly one point, in every replication.
    let ablations: AblationSummary =
        fsutil::read_json(&run.join("ablations/pipeline_summary.json")).unwrap();
    let no_bonus = ablations
        .variants
        .iter()
        .find(|variant| variant.spec.variant_id == "ablation_no_bonus")
        .expect("the proposed ablation ran");
    assert_eq!(no_bonus.replications.len(), 3);
    assert!(no_bonus
        .replications
        .iter()
        .all(|replication| replication.correct && replication.combined_score == 10.0));

    // The final evaluation ran once, on seeds the search never used.
    let final_manifest: FinalManifest =
        fsutil::read_json(&run.join("final_evaluation/final_manifest.json")).unwrap();
    assert_eq!(final_manifest.outcome, "qualified");
    assert!(run.join("final_evaluation/invocation.json").is_file());

    let posthoc: PosthocManifest =
        fsutil::read_json(&run.join("posthoc/latest_manifest.json")).unwrap();
    assert_eq!(posthoc.valid_analyses.len(), 1);
    assert_eq!(
        posthoc.valid_analyses[0].claim,
        "The best correct score reached 11.0."
    );

    // The paper passed its audit, was reviewed, and the revision was promoted.
    let manifest: PaperManifest =
        fsutil::read_json(&run.join("paper/latest_manifest.json")).unwrap();
    assert!(manifest.validation_ok);
    assert_eq!(manifest.evidence_audit_verdict, "pass");
    let paper = std::fs::read_to_string(run.join("paper/paper.md")).unwrap();
    assert!(
        paper.contains("After revision: on this single task"),
        "the promoted revision is the published paper"
    );
    let before: Review = fsutil::read_json(&run.join("paper/before_revision.json")).unwrap();
    let after: Review = fsutil::read_json(&run.join("paper/after_revision.json")).unwrap();
    assert_eq!((before.overall, after.overall), (4.0, 6.0));
    assert_eq!(
        manifest
            .revision
            .as_ref()
            .map(|record| record.status.as_str()),
        Some("promoted")
    );

    // The run's own records.
    let status = read("pipeline_status.json");
    assert_eq!(status["status"], "completed");
    assert!(run.join("run_config.yaml").is_file());
    let events: Vec<Value> = fsutil::read_jsonl(&run.join("events.jsonl"));
    let ended = events
        .iter()
        .filter(|event| event["event"] == "phase_end")
        .count();
    assert_eq!(ended, 8);
    for stage in [
        "idea",
        "proposal",
        "conclusions",
        "variant_generate",
        "variant_implement",
        "posthoc_interpret",
        "paper_draft",
        "review",
        "paper_revision",
    ] {
        assert!(
            stage_calls(&provider, stage) > 0,
            "stage `{stage}` was never called"
        );
    }
}

#[tokio::test]
async fn a_finished_run_resumes_without_repeating_paid_work() {
    let dir = tempfile::tempdir().unwrap();
    let task = task(dir.path());
    let first = PipelineConfig::load(
        None,
        &overrides(dir.path(), &task, "investigation, final_evaluation"),
    )
    .unwrap();
    let provider = scripted();
    let report = run_pipeline_with(
        first,
        Arc::new(AtomicBool::new(false)),
        Some(provider.clone()),
    )
    .await
    .unwrap();
    assert_eq!(report.status, "completed");
    let proposals = stage_calls(&provider, "proposal");
    let invocation =
        std::fs::read(report.run_dir.join("final_evaluation/invocation.json")).unwrap();

    let mut again = overrides(dir.path(), &task, "investigation, final_evaluation, paper");
    again.push(format!("paths.run_dir={}", report.run_dir.display()));
    let second = PipelineConfig::load(None, &again).unwrap();
    let resumed = run_pipeline_with(
        second,
        Arc::new(AtomicBool::new(false)),
        Some(provider.clone()),
    )
    .await
    .unwrap();
    assert_eq!(resumed.run_dir, report.run_dir);
    assert_eq!(
        stage_calls(&provider, "proposal"),
        proposals,
        "a completed search proposes nothing more"
    );
    assert_eq!(
        std::fs::read(report.run_dir.join("final_evaluation/invocation.json")).unwrap(),
        invocation,
        "the sealed evaluation is never invoked twice"
    );
    assert!(report.run_dir.join("paper/paper.md").is_file());
    assert_eq!(resumed.exit_code(), 0);
}

#[tokio::test]
async fn a_failed_search_stops_the_pipeline_before_any_evidence_phase() {
    let dir = tempfile::tempdir().unwrap();
    let task = task(dir.path());
    // The evaluator cannot run, so the seed never evaluates.
    std::fs::write(
        task.join("task.toml"),
        "eval_command = [\"definitely-not-a-command\", \"{program_path}\", \"{results_dir}\"]\n",
    )
    .unwrap();
    let config = PipelineConfig::load(
        None,
        &overrides(dir.path(), &task, "investigation, ablations, paper"),
    )
    .unwrap();
    let provider = scripted();
    let report = run_pipeline_with(
        config,
        Arc::new(AtomicBool::new(false)),
        Some(provider.clone()),
    )
    .await
    .unwrap();
    let labels: Vec<(&str, &str)> = report
        .phases
        .iter()
        .map(|phase| (phase.phase.as_str(), phase.status.label()))
        .collect();
    assert_eq!(
        labels,
        [
            ("investigation", "failed"),
            ("ablations", "skipped"),
            ("paper", "skipped")
        ]
    );
    assert_eq!(report.status, "failed");
    assert_eq!(report.exit_code(), 2);
    assert_eq!(stage_calls(&provider, "paper_draft"), 0);
}
