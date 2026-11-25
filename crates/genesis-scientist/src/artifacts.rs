//! The files phases hand to each other. Each struct is the schema of one
//! JSON artifact in the run directory; a phase reads only these, never
//! another phase's internals.

use genesis_core::stats::{Comparison, Summary};
use genesis_core::JsonMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// `investigation/best_program_manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct BestProgramManifest {
    pub program_id: Option<String>,
    pub generation: Option<u32>,
    /// Program file, relative to the run directory.
    pub program_path: Option<String>,
    pub combined_score: Option<f64>,
    pub correct: Option<bool>,
    /// `seed` when the best program is the task's own, else `generation_<n>`.
    pub source: String,
    pub patch_name: Option<String>,
    pub updated_at: String,
}

/// `.investigation_completion.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct InvestigationCompletion {
    pub schema_version: u32,
    /// `running` or `completed`.
    pub status: String,
    /// `winner` when a generated program is correct and beats the seed;
    /// `completed_no_winner` otherwise.
    pub outcome: Option<String>,
    /// SHA-256 of the best-program manifest the outcome was judged on.
    pub manifest_sha256: Option<String>,
    pub stop_reason: Option<String>,
}

pub const OUTCOME_WINNER: &str = "winner";
pub const OUTCOME_NO_WINNER: &str = "completed_no_winner";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct AttemptCounts {
    pub total: usize,
    /// Proposals that were evaluated and produced a score.
    pub scored: usize,
    /// Proposals that never reached a score: missing evidence, not losses.
    pub unscored: usize,
}

/// `.downstream_selection.json`: the program ablations, final evaluation,
/// post-hoc analysis and the paper are about. Frozen once written.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct DownstreamSelection {
    pub schema_version: u32,
    pub investigation_outcome: String,
    /// `winner`, `non_winner_candidate` or `baseline`.
    pub selection_kind: String,
    pub program_id: String,
    pub generation: u32,
    pub combined_score: f64,
    pub correct: bool,
    /// Frozen copy, relative to the run directory.
    pub program_path: String,
    pub program_sha256: String,
    pub baseline_path: String,
    pub baseline_sha256: String,
    pub baseline_score: Option<f64>,
    pub baseline_correct: bool,
    pub attempt_counts: AttemptCounts,
    pub task_fingerprint: String,
    pub selected_at: String,
}

/// One program an ablation study evaluates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct VariantSpec {
    pub variant_id: String,
    pub title: String,
    /// `control` (the selected program unchanged), `baseline` (a reference
    /// method) or `ablation` (the selected program with one component removed).
    pub kind: String,
    /// `best` or `reference`: what the variant was derived from.
    pub base_program: String,
    pub hypothesis: String,
    pub relevance_reason: String,
    pub feasibility_reason: String,
    pub target_components: Vec<String>,
    pub implementation_plan: Vec<String>,
    pub expected_effect: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ReplicationResult {
    pub seed: u64,
    pub combined_score: f64,
    pub correct: bool,
    pub evaluation_failed: bool,
    pub runtime_secs: f64,
    pub public_metrics: JsonMap,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct VariantResult {
    pub spec: VariantSpec,
    /// `completed`, `failed` (no runnable program was produced) or
    /// `not_executed`.
    pub status: String,
    /// Program file, relative to the run directory.
    pub program_path: Option<String>,
    pub replications: Vec<ReplicationResult>,
    /// Statistics of `combined_score` over correct replications.
    pub summary: Option<Summary>,
    /// Share of replications the evaluator judged correct.
    pub correct_rate: f64,
    /// This variant against the control, Welch's test; `mean_diff` is
    /// variant minus control.
    pub comparison_to_control: Option<Comparison>,
    /// Percentile-bootstrap 95% interval of the same difference.
    pub bootstrap_ci95: Option<(f64, f64)>,
    pub notes: String,
}

/// `ablations/pipeline_summary.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct AblationSummary {
    pub schema_version: u32,
    pub task: String,
    pub primary_metric_name: String,
    /// The variant every other one is compared with: `control_best`.
    pub anchor_variant_id: String,
    pub num_replications: usize,
    pub seeds: Vec<u64>,
    /// False when every replication of the control returned the same score:
    /// the evaluator ignores the seed and intervals are not meaningful.
    pub evaluator_is_stochastic: bool,
    pub total_variants: usize,
    pub total_completed: usize,
    pub total_failed: usize,
    pub variants: Vec<VariantResult>,
    pub generated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct FinalEpisode {
    pub seed: u64,
    pub score: f64,
    pub correct: bool,
    pub duration_seconds: f64,
}

/// `final_evaluation/final_manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct FinalManifest {
    pub schema_version: u32,
    /// `completed` once the evaluation ran to the end, whatever it found.
    pub status: String,
    pub started_at: String,
    pub completed_at: String,
    /// True when every final episode of the selected program was correct.
    pub correct: bool,
    /// `qualified` or `not_qualified`.
    pub outcome: String,
    /// `evaluate_final` when the task ships a final evaluator, else
    /// `search_evaluator_heldout_seeds`.
    pub evaluator: String,
    pub seeds: Vec<u64>,
    pub episodes: Vec<FinalEpisode>,
    pub baseline_episodes: Vec<FinalEpisode>,
    pub candidate: Option<Summary>,
    pub baseline: Option<Summary>,
    /// Candidate against baseline on the final seeds.
    pub comparison: Option<Comparison>,
    /// True when the 95% interval of candidate minus baseline excludes zero
    /// from below.
    pub improvement_supported: bool,
    pub program_sha256: String,
    pub task_fingerprint: String,
    pub limitations: Vec<String>,
}

/// One post-hoc analysis, proposed and then carried out.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct AnalysisSpec {
    pub id: String,
    pub title: String,
    pub family: String,
    pub hypothesis: String,
    pub relevance_reason: String,
    pub feasibility_reason: String,
    pub required_artifacts: Vec<String>,
    pub requires_retrain: bool,
    pub implementation_plan: Vec<String>,
    pub expected_claim: String,
    pub expected_outputs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct AnalysisResult {
    pub spec: AnalysisSpec,
    /// Directory of the analysis, relative to the run directory.
    pub analysis_dir: String,
    pub summary_path: String,
    /// Output files, relative to the run directory.
    pub outputs: Vec<String>,
    /// The one-sentence finding, grounded in `key_numbers`.
    pub claim: String,
    pub grounded: bool,
    /// Every number the analysis reports, by name.
    pub key_numbers: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct RejectedAnalysis {
    pub analysis_id: String,
    pub title: String,
    pub family: String,
    pub reason: String,
}

/// `posthoc/latest_manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct PosthocManifest {
    pub manifest_version: u32,
    pub created_at: String,
    pub valid_analyses: Vec<AnalysisResult>,
    pub rejected_analyses: Vec<RejectedAnalysis>,
}

/// `paper/latest_manifest.json`. Paths are relative to the run directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct PaperManifest {
    pub generated_at: String,
    pub paper_format: String,
    pub paper_dir: String,
    pub paper_tex: String,
    pub paper_md: String,
    /// Absent when no TeX toolchain was available to compile it.
    pub paper_pdf: Option<String>,
    pub paper_data: String,
    pub references_bib: String,
    pub claims_grounding: String,
    pub evidence_audit: String,
    /// `pass` or `fail`.
    pub evidence_audit_verdict: String,
    pub validation_report: String,
    pub validation_ok: bool,
    /// Outcome of the review phases, keyed `ai_review`, `before_revision`,
    /// `after_revision`.
    pub reviews: BTreeMap<String, ReviewRecord>,
    pub revision: Option<RevisionRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ReviewRecord {
    /// `completed` or `failed`.
    pub status: String,
    pub review_json: Option<String>,
    pub overall: Option<f64>,
    pub decision: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct RevisionRecord {
    /// `promoted` when the revised paper replaced the original.
    pub status: String,
    pub attempt_dir: String,
    pub original_snapshot: String,
    pub overall_before: Option<f64>,
    pub overall_after: Option<f64>,
    pub decision_before: Option<String>,
    pub decision_after: Option<String>,
}

/// The reviewer's form, with the field names of the NeurIPS-style review
/// JSON used by the Agentic Scientist.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Review {
    #[serde(rename = "Summary")]
    pub summary: String,
    #[serde(rename = "Strengths")]
    pub strengths: Vec<String>,
    #[serde(rename = "Weaknesses")]
    pub weaknesses: Vec<String>,
    /// 1 to 4.
    #[serde(rename = "Originality")]
    pub originality: f64,
    #[serde(rename = "Quality")]
    pub quality: f64,
    #[serde(rename = "Clarity")]
    pub clarity: f64,
    #[serde(rename = "Significance")]
    pub significance: f64,
    #[serde(rename = "Questions")]
    pub questions: Vec<String>,
    #[serde(rename = "Limitations")]
    pub limitations: Vec<String>,
    #[serde(rename = "Ethical Concerns")]
    pub ethical_concerns: bool,
    #[serde(rename = "Soundness")]
    pub soundness: f64,
    #[serde(rename = "Presentation")]
    pub presentation: f64,
    #[serde(rename = "Contribution")]
    pub contribution: f64,
    /// 1 to 10.
    #[serde(rename = "Overall")]
    pub overall: f64,
    /// 1 to 5.
    #[serde(rename = "Confidence")]
    pub confidence: f64,
    /// `Accept` or `Reject`.
    #[serde(rename = "Decision")]
    pub decision: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_review_form_uses_the_published_field_names() {
        let review: Review = serde_json::from_str(
            r#"{"Summary": "s", "Strengths": ["a"], "Weaknesses": [], "Originality": 2, "Quality": 3,
                "Clarity": 3, "Significance": 2, "Questions": [], "Limitations": ["l"],
                "Ethical Concerns": false, "Soundness": 3, "Presentation": 3, "Contribution": 2,
                "Overall": 4, "Confidence": 4, "Decision": "Reject"}"#,
        )
        .unwrap();
        assert_eq!((review.overall, review.decision.as_str()), (4.0, "Reject"));
        let text = serde_json::to_string(&review).unwrap();
        assert!(text.contains("\"Ethical Concerns\":false") && text.contains("\"Overall\":4.0"));
    }

    #[test]
    fn artifacts_tolerate_missing_fields() {
        let summary: AblationSummary = serde_json::from_str(r#"{"task": "t"}"#).unwrap();
        assert_eq!(summary.task, "t");
        assert!(summary.variants.is_empty());
        let manifest: FinalManifest = serde_json::from_str("{}").unwrap();
        assert!(!manifest.correct && manifest.episodes.is_empty());
        let selection: DownstreamSelection = serde_json::from_str(
            r#"{"selection_kind": "baseline", "attempt_counts": {"total": 3}}"#,
        )
        .unwrap();
        assert_eq!(selection.attempt_counts.total, 3);
    }
}
