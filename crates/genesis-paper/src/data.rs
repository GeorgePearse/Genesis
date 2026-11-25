//! The grounding data of a paper: everything the run established, gathered
//! into `paper_data.json`.
//!
//! Every number the manuscript may state is a grounding source with a
//! stable id (`result.best_correct_score`, `ablation.variant_<id>…`,
//! `final.metric.…`). The id scheme follows the Agentic Scientist's
//! `paper_data.json` (Sakana AI, Apache-2.0). Nothing a model writes is
//! ever added to it.

use anyhow::{bail, Context, Result};
use genesis_core::{fsutil, Program};
use genesis_evolve::ProgramDb;
use genesis_scientist::artifacts::{
    AblationSummary, DownstreamSelection, FinalManifest, PosthocManifest,
};
use genesis_scientist::selection::{load_selection, own_programs};
use genesis_scientist::RunPaths;
use regex::Regex;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// At most this many distinct numeric constants of a program are grounded.
const MAX_SOURCE_CONSTANTS: usize = 150;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GroundingSource {
    pub id: String,
    /// `metric`, `count`, `derived`, `cost`, `task_constant`,
    /// `source_constant`, `final_qualification`, `study_context`,
    /// `posthoc_data` or `posthoc_analysis`.
    #[serde(rename = "type")]
    pub kind: String,
    pub value: Value,
    pub description: String,
}

/// The id-keyed set of grounding sources. An id is never overwritten: a
/// second source with the same id gets a numeric suffix.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Grounding {
    sources: BTreeMap<String, GroundingSource>,
    inserted: usize,
}

impl Grounding {
    pub fn sources(&self) -> &BTreeMap<String, GroundingSource> {
        &self.sources
    }

    /// How many sources were accepted; equal to `sources().len()` because no
    /// insert replaces another.
    pub fn inserted(&self) -> usize {
        self.inserted
    }

    fn insert(&mut self, id: &str, kind: &str, value: Value, description: String) {
        let mut unique = id.to_string();
        let mut suffix = 2;
        while self.sources.contains_key(&unique) {
            unique = format!("{id}_{suffix}");
            suffix += 1;
        }
        self.sources.insert(
            unique.clone(),
            GroundingSource {
                id: unique,
                kind: kind.to_string(),
                value,
                description,
            },
        );
        self.inserted += 1;
    }

    /// Add a measured number. A non-finite value is not evidence and is
    /// left out.
    pub fn number(&mut self, id: &str, kind: &str, value: f64, description: impl Into<String>) {
        if value.is_finite() {
            self.insert(id, kind, json!(value), description.into());
        }
    }

    pub fn count(&mut self, id: &str, value: usize, description: impl Into<String>) {
        self.insert(id, "count", json!(value), description.into());
    }

    pub fn fact(&mut self, id: &str, kind: &str, value: Value, description: impl Into<String>) {
        self.insert(id, kind, value, description.into());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct ProgramRow {
    pub program_id: String,
    pub generation: u32,
    pub name: String,
    pub combined_score: f64,
    pub correct: bool,
    /// `patch` or `agent`, when recorded.
    pub operator: Option<String>,
}

impl ProgramRow {
    fn of(program: &Program) -> Self {
        Self {
            program_id: program.id.clone(),
            generation: program.generation,
            name: program
                .meta_str("patch_name")
                .unwrap_or("initial_program")
                .to_string(),
            combined_score: program.combined_score,
            correct: program.correct,
            operator: program.meta_str("operator").map(str::to_string),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct TraceStep {
    /// The grounding id of this step's score.
    pub id: String,
    pub generation: u32,
    pub combined_score: f64,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Totals {
    pub ideas: usize,
    pub attempts: usize,
    pub evaluated_programs: usize,
    pub correct_programs: usize,
    pub failed_proposals: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct EvidenceSource {
    /// `included`, `missing` (the phase did not run) or `degraded` (it ran
    /// but left nothing usable).
    pub status: String,
    pub path: Option<String>,
    pub note: Option<String>,
}

/// `paper/paper_data.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct PaperData {
    pub generated_at: String,
    pub task_name: String,
    pub task_goal: String,
    pub language: String,
    pub selection: DownstreamSelection,
    /// The program the paper is about.
    pub selected_code: String,
    pub best_correct: Option<ProgramRow>,
    pub seed: Option<ProgramRow>,
    pub totals: Totals,
    pub top_programs: Vec<ProgramRow>,
    pub improvement_trace: Vec<TraceStep>,
    pub ablations: Option<AblationSummary>,
    pub final_evaluation: Option<FinalManifest>,
    pub posthoc: Option<PosthocManifest>,
    pub distilled_conclusions: Option<String>,
    pub grounding_sources: BTreeMap<String, GroundingSource>,
    /// Keyed `ablations`, `final_evaluation`, `posthoc`, `references`.
    pub evidence_sources: BTreeMap<String, EvidenceSource>,
    pub limitations: Vec<String>,
}

impl PaperData {
    /// Every numeric grounding value.
    pub fn numeric_values(&self) -> Vec<f64> {
        self.grounding_sources
            .values()
            .filter_map(|source| source.value.as_f64())
            .collect()
    }

    /// The numeric values of the given ids; unknown ids contribute nothing.
    pub fn values_of(&self, ids: &[String]) -> Vec<f64> {
        ids.iter()
            .filter_map(|id| self.grounding_sources.get(id))
            .filter_map(|source| source.value.as_f64())
            .collect()
    }

    pub fn is_negative_result(&self) -> bool {
        self.selection.selection_kind != "winner"
    }
}

fn evaluation_failed(program: &Program) -> bool {
    program
        .metadata
        .get("evaluation_failed")
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Distinct numeric literals of a text, in order of first appearance.
fn constants(text: &str) -> Vec<f64> {
    static NUMBER: OnceLock<Regex> = OnceLock::new();
    let number =
        NUMBER.get_or_init(|| Regex::new(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?").expect("valid regex"));
    let mut seen: Vec<f64> = Vec::new();
    for found in number.find_iter(text) {
        if let Ok(value) = found.as_str().parse::<f64>() {
            if value.is_finite() && !seen.contains(&value) {
                seen.push(value);
                if seen.len() == MAX_SOURCE_CONSTANTS {
                    break;
                }
            }
        }
    }
    seen
}

fn relative_percent(delta: f64, reference: f64) -> Option<f64> {
    (reference != 0.0 && reference.is_finite()).then(|| 100.0 * delta / reference.abs())
}

/// Ground what the search itself established. Returns the best correct
/// program, the seed, the top programs and the improvement trace.
pub fn ground_search(
    grounding: &mut Grounding,
    programs: &[Program],
    totals: &Totals,
    selection: &DownstreamSelection,
) -> (
    Option<ProgramRow>,
    Option<ProgramRow>,
    Vec<ProgramRow>,
    Vec<TraceStep>,
) {
    let seed = programs.iter().find(|program| program.generation == 0);
    let mut correct: Vec<&Program> = programs.iter().filter(|program| program.correct).collect();
    correct.sort_by(|a, b| {
        b.combined_score
            .total_cmp(&a.combined_score)
            .then(a.generation.cmp(&b.generation))
    });
    let best = correct.first().copied();

    if let Some(best) = best {
        grounding.number(
            "result.best_correct_score",
            "metric",
            best.combined_score,
            "best combined score among correct programs found by the search",
        );
        grounding.count(
            "result.best_generation",
            best.generation as usize,
            "generation of the best correct program",
        );
    }
    if let Some(seed) = seed.filter(|seed| seed.correct) {
        grounding.number(
            "result.seed_score",
            "metric",
            seed.combined_score,
            "combined score of the task's seed program in the search evaluation",
        );
        if let Some(best) = best {
            let delta = best.combined_score - seed.combined_score;
            grounding.number(
                "result.improvement_abs",
                "derived",
                delta,
                "best correct score minus the seed score",
            );
            if let Some(percent) = relative_percent(delta, seed.combined_score) {
                grounding.number(
                    "result.improvement_rel_percent",
                    "derived",
                    percent,
                    "improvement over the seed score, in percent of the seed score",
                );
            }
        }
    }
    grounding.number(
        "result.selected_score",
        "metric",
        selection.combined_score,
        "search-time combined score of the program this paper studies",
    );
    grounding.count(
        "result.total_ideas",
        totals.ideas,
        "ideas generated during the search",
    );
    grounding.count(
        "result.total_attempts",
        totals.attempts,
        "proposals attempted, including those that produced no program",
    );
    grounding.count(
        "result.total_evaluated_programs",
        totals.evaluated_programs,
        "programs evaluated, the seed included",
    );
    grounding.count(
        "result.total_correct_programs",
        totals.correct_programs,
        "evaluated programs the evaluator judged correct",
    );
    grounding.count(
        "result.total_failed_proposals",
        totals.failed_proposals,
        "proposals that produced no evaluated program; missing evidence, not measured losses",
    );

    let top: Vec<ProgramRow> = correct
        .iter()
        .take(10)
        .map(|program| ProgramRow::of(program))
        .collect();
    for (rank, row) in top.iter().enumerate() {
        grounding.number(
            &format!("result.top_program_{:02}", rank + 1),
            "metric",
            row.combined_score,
            format!(
                "combined score of the rank-{} correct program (`{}`, generation {})",
                rank + 1,
                row.name,
                row.generation
            ),
        );
    }

    let mut ordered: Vec<&Program> = correct.clone();
    ordered.sort_by(|a, b| {
        a.generation
            .cmp(&b.generation)
            .then(a.timestamp.total_cmp(&b.timestamp))
    });
    let mut trace = Vec::new();
    let mut running: Option<f64> = None;
    for program in ordered {
        if running.is_none_or(|best| program.combined_score > best) {
            running = Some(program.combined_score);
            let id = format!("result.improvement_{:03}", trace.len() + 1);
            grounding.number(
                &id,
                "metric",
                program.combined_score,
                format!("running best after generation {}", program.generation),
            );
            grounding.count(
                &format!("{id}.generation"),
                program.generation as usize,
                "generation at which this running best was reached",
            );
            trace.push(TraceStep {
                id,
                generation: program.generation,
                combined_score: program.combined_score,
                name: program
                    .meta_str("patch_name")
                    .unwrap_or("initial_program")
                    .to_string(),
            });
        }
    }
    (
        best.map(ProgramRow::of),
        seed.map(ProgramRow::of),
        top,
        trace,
    )
}

/// Ground the frozen selection of a run that found no winner.
pub fn ground_study(grounding: &mut Grounding, selection: &DownstreamSelection) {
    grounding.fact(
        "study.selection_kind",
        "study_context",
        json!(selection.selection_kind),
        "how the studied program was chosen",
    );
    grounding.fact(
        "study.investigation_outcome",
        "study_context",
        json!(selection.investigation_outcome),
        "outcome of the search",
    );
    grounding.number(
        "study.search_score",
        "study_context",
        selection.combined_score,
        "search-time score of the studied program",
    );
    if let Some(baseline) = selection.baseline_score {
        grounding.number(
            "study.baseline_score",
            "study_context",
            baseline,
            "search-time score of the seed program",
        );
        grounding.number(
            "study.search_score_delta",
            "study_context",
            selection.combined_score - baseline,
            "studied program minus seed, search-time scores",
        );
    }
    grounding.count(
        "study.attempts.total",
        selection.attempt_counts.total,
        "proposals attempted",
    );
    grounding.count(
        "study.attempts.scored",
        selection.attempt_counts.scored,
        "proposals that were evaluated and scored",
    );
    grounding.count(
        "study.attempts.unscored",
        selection.attempt_counts.unscored,
        "proposals with no score: missing evidence, not measured losses",
    );
}

/// Ground the optional evidence phases. Non-finite statistics are skipped
/// and colliding ids are kept apart.
pub fn ground_evidence(
    grounding: &mut Grounding,
    ablations: Option<&AblationSummary>,
    final_manifest: Option<&FinalManifest>,
    posthoc: Option<&PosthocManifest>,
) {
    if let Some(summary) = ablations {
        grounding.count(
            "ablation.num_replications",
            summary.num_replications,
            "replications per ablation variant",
        );
        grounding.count(
            "ablation.total_variants",
            summary.total_variants,
            "variants in the ablation study",
        );
        grounding.count(
            "ablation.total_completed",
            summary.total_completed,
            "variants that completed",
        );
        let anchor = fsutil::slugify(&summary.anchor_variant_id);
        let anchor_mean = summary
            .variants
            .iter()
            .find(|variant| variant.spec.variant_id == summary.anchor_variant_id)
            .and_then(|variant| variant.summary.as_ref())
            .map(|stats| stats.mean);
        for variant in &summary.variants {
            let slug = fsutil::slugify(&variant.spec.variant_id);
            let label = format!(
                "variant `{}` ({})",
                variant.spec.variant_id, variant.spec.kind
            );
            grounding.number(
                &format!("ablation.variant_{slug}.correct_rate"),
                "metric",
                variant.correct_rate,
                format!("share of correct replications, {label}"),
            );
            if let Some(stats) = &variant.summary {
                let base = format!("ablation.variant_{slug}.combined_score");
                for (suffix, value, what) in [
                    ("mean", stats.mean, "mean"),
                    ("std", stats.std, "sample standard deviation"),
                    ("min", stats.min, "minimum"),
                    ("max", stats.max, "maximum"),
                    ("ci95_low", stats.ci95_low, "lower end of the 95% interval"),
                    (
                        "ci95_high",
                        stats.ci95_high,
                        "upper end of the 95% interval",
                    ),
                ] {
                    grounding.number(
                        &format!("{base}.{suffix}"),
                        "metric",
                        value,
                        format!("{what} of combined score over correct replications, {label}"),
                    );
                }
                grounding.count(
                    &format!("{base}.n"),
                    stats.n,
                    format!("correct replications, {label}"),
                );
            }
            if let Some(comparison) = &variant.comparison_to_control {
                let base = format!("ablation.compare_{anchor}_vs_{slug}.combined_score");
                let what = format!(
                    "`{}` minus `{}`",
                    variant.spec.variant_id, summary.anchor_variant_id
                );
                grounding.number(
                    &format!("{base}.mean_delta"),
                    "derived",
                    comparison.mean_diff,
                    format!("mean combined score, {what}"),
                );
                if let Some(percent) =
                    anchor_mean.and_then(|mean| relative_percent(comparison.mean_diff, mean))
                {
                    grounding.number(
                        &format!("{base}.rel_percent"),
                        "derived",
                        percent,
                        format!("{what}, in percent of the control mean"),
                    );
                }
                for (suffix, value, name) in [
                    (
                        "p_value",
                        comparison.p_value,
                        "two-sided p-value of Welch's test",
                    ),
                    (
                        "ci95_low",
                        comparison.diff_ci95_low,
                        "lower end of the 95% interval of the difference",
                    ),
                    (
                        "ci95_high",
                        comparison.diff_ci95_high,
                        "upper end of the 95% interval of the difference",
                    ),
                ] {
                    if let Some(value) = value {
                        grounding.number(
                            &format!("{base}.{suffix}"),
                            "derived",
                            value,
                            format!("{name}, {what}"),
                        );
                    }
                }
            }
        }
    }

    if let Some(manifest) = final_manifest {
        grounding.fact(
            "final.qualification.status",
            "final_qualification",
            json!(manifest.status),
            "status of the final evaluation",
        );
        grounding.fact(
            "final.qualification.correct",
            "final_qualification",
            json!(manifest.correct),
            "whether every final episode of the studied program was correct",
        );
        grounding.fact(
            "final.qualification.outcome",
            "final_qualification",
            json!(manifest.outcome),
            "qualification outcome of the final evaluation",
        );
        grounding.fact(
            "final.improvement_supported",
            "final_qualification",
            json!(manifest.improvement_supported),
            "whether the 95% interval of studied minus baseline lies above zero",
        );
        grounding.count(
            "final.evidence.episodes",
            manifest.episodes.len(),
            "final-evaluation episodes of the studied program",
        );
        grounding.count(
            "final.evidence.correct_episodes",
            manifest
                .episodes
                .iter()
                .filter(|episode| episode.correct)
                .count(),
            "final episodes of the studied program judged correct",
        );
        grounding.count(
            "final.evidence.baseline_episodes",
            manifest.baseline_episodes.len(),
            "final-evaluation episodes of the baseline",
        );
        for (prefix, stats, who) in [
            (
                "final.metric.combined_score",
                &manifest.candidate,
                "studied program",
            ),
            (
                "final.metric.baseline_combined_score",
                &manifest.baseline,
                "baseline",
            ),
        ] {
            if let Some(stats) = stats {
                grounding.number(
                    prefix,
                    "metric",
                    stats.mean,
                    format!("mean final combined score, {who}"),
                );
                for (suffix, value, what) in [
                    ("std", stats.std, "sample standard deviation"),
                    ("min", stats.min, "minimum"),
                    ("max", stats.max, "maximum"),
                    ("ci95_low", stats.ci95_low, "lower end of the 95% interval"),
                    (
                        "ci95_high",
                        stats.ci95_high,
                        "upper end of the 95% interval",
                    ),
                ] {
                    grounding.number(
                        &format!("{prefix}_{suffix}"),
                        "metric",
                        value,
                        format!("{what} of final combined score, {who}"),
                    );
                }
            }
        }
        if let Some(comparison) = &manifest.comparison {
            grounding.number(
                "final.comparison.mean_diff",
                "derived",
                comparison.mean_diff,
                "final combined score, studied program minus baseline",
            );
            let baseline_mean = manifest.baseline.as_ref().map(|stats| stats.mean);
            if let Some(percent) =
                baseline_mean.and_then(|mean| relative_percent(comparison.mean_diff, mean))
            {
                grounding.number(
                    "final.comparison.rel_percent",
                    "derived",
                    percent,
                    "final difference in percent of the baseline mean",
                );
            }
            for (suffix, value, name) in [
                (
                    "p_value",
                    comparison.p_value,
                    "two-sided p-value of Welch's test on the final scores",
                ),
                (
                    "ci95_low",
                    comparison.diff_ci95_low,
                    "lower end of the 95% interval of the final difference",
                ),
                (
                    "ci95_high",
                    comparison.diff_ci95_high,
                    "upper end of the 95% interval of the final difference",
                ),
                (
                    "cohens_d",
                    comparison.cohens_d,
                    "Cohen's d of the final difference",
                ),
            ] {
                if let Some(value) = value {
                    grounding.number(
                        &format!("final.comparison.{suffix}"),
                        "derived",
                        value,
                        name,
                    );
                }
            }
        }
    }

    if let Some(manifest) = posthoc {
        for analysis in &manifest.valid_analyses {
            let slug = fsutil::slugify(&analysis.spec.id);
            grounding.fact(
                &format!("posthoc.{slug}.claim"),
                "posthoc_analysis",
                json!(analysis.claim),
                format!("finding of the post-hoc analysis `{}`", analysis.spec.title),
            );
            for (key, value) in &analysis.key_numbers {
                grounding.number(
                    &format!("posthoc.{slug}.{}", fsutil::slugify(key)),
                    "posthoc_data",
                    *value,
                    format!(
                        "`{key}` from the post-hoc analysis `{}`",
                        analysis.spec.title
                    ),
                );
            }
        }
    }
}

/// Whether the final outcome policy lets this run be written up.
///
/// A winner that failed its final qualification is published only under
/// `allow_not_qualified`. A run without a winner may always be written up:
/// it is a negative-results paper and says so.
pub fn publication_allowed(
    selection_kind: &str,
    final_manifest: Option<&FinalManifest>,
    policy: &str,
) -> Result<()> {
    match final_manifest {
        Some(manifest)
            if manifest.outcome != "qualified"
                && selection_kind == "winner"
                && policy != "allow_not_qualified" =>
        {
            bail!(
                "the final evaluation did not qualify the selected program (outcome `{}`); set \
                 paper.final_outcome_policy=allow_not_qualified to write it up with that limitation stated",
                manifest.outcome
            )
        }
        _ => Ok(()),
    }
}

fn read_optional<T: DeserializeOwned>(
    paths: &RunPaths,
    path: &Path,
) -> (Option<T>, EvidenceSource) {
    if !path.is_file() {
        return (
            None,
            EvidenceSource {
                status: "missing".to_string(),
                path: None,
                note: None,
            },
        );
    }
    let relative = Some(paths.relative(path));
    match fsutil::read_json::<T>(path) {
        Ok(value) => (
            Some(value),
            EvidenceSource {
                status: "included".to_string(),
                path: relative,
                note: None,
            },
        ),
        Err(error) => (
            None,
            EvidenceSource {
                status: "degraded".to_string(),
                path: relative,
                note: Some(format!(
                    "the result file is unreadable: {}",
                    error.root_cause()
                )),
            },
        ),
    }
}

/// Why an optional phase contributes nothing, as the start of a limitation
/// sentence.
fn absence(source: &EvidenceSource, phase: &str) -> String {
    match &source.note {
        Some(note) if source.status == "degraded" => {
            format!("The {phase} left no usable result ({note})")
        }
        _ => format!("No {phase} is available for this run"),
    }
}

fn degrade<T>(evidence: &mut (Option<T>, EvidenceSource), unusable: impl Fn(&T) -> Option<String>) {
    if let Some(note) = evidence.0.as_ref().and_then(unusable) {
        evidence.0 = None;
        evidence.1.status = "degraded".to_string();
        evidence.1.note = Some(note);
    }
}

/// The run database of the investigation.
pub fn investigation_database(paths: &RunPaths) -> Result<PathBuf> {
    let dir = paths.investigation();
    let default = dir.join("programs.sqlite");
    if default.is_file() {
        return Ok(default);
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&dir)
        .with_context(|| {
            format!(
                "the investigation directory {} does not exist",
                dir.display()
            )
        })?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "sqlite")
                && path
                    .file_name()
                    .is_some_and(|name| name != "prompts.sqlite")
        })
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .with_context(|| format!("no run database in {}", dir.display()))
}

/// Gather the evidence of a run. Errors when there is nothing to write
/// about or the final outcome policy forbids publication.
pub fn build_paper_data(
    paths: &RunPaths,
    task_name: &str,
    task_goal: &str,
    language: &str,
    policy: &str,
) -> Result<PaperData> {
    let selected = load_selection(paths)?
        .context("no program was selected for the downstream phases, so there is nothing to write a paper about")?;
    let db = ProgramDb::open_read_only(&investigation_database(paths)?)?;
    let programs = own_programs(&db)?;
    let failed_proposals = db
        .attempts()?
        .iter()
        .filter(|attempt| attempt.status == "failed")
        .count();
    let evaluated: Vec<Program> = programs
        .into_iter()
        .filter(|program| !evaluation_failed(program))
        .collect();
    if evaluated.is_empty() {
        bail!("the investigation has no evaluated programs, so there is nothing to write a paper about");
    }
    let investigation = paths.investigation();
    let totals = Totals {
        ideas: fsutil::read_jsonl::<Value>(&investigation.join("ideas_history.jsonl")).len(),
        attempts: selected.selection.attempt_counts.total,
        evaluated_programs: evaluated.len(),
        correct_programs: evaluated.iter().filter(|program| program.correct).count(),
        failed_proposals,
    };

    let mut ablations =
        read_optional::<AblationSummary>(paths, &paths.ablations().join("pipeline_summary.json"));
    degrade(&mut ablations, |summary| {
        (!summary
            .variants
            .iter()
            .any(|variant| variant.summary.is_some()))
        .then(|| "no variant produced a correct replication".to_string())
    });
    let mut final_evaluation = read_optional::<FinalManifest>(
        paths,
        &paths.final_evaluation().join("final_manifest.json"),
    );
    degrade(&mut final_evaluation, |manifest| {
        (manifest.status != "completed")
            .then(|| format!("the final evaluation has status `{}`", manifest.status))
    });
    let mut posthoc =
        read_optional::<PosthocManifest>(paths, &paths.posthoc().join("latest_manifest.json"));
    degrade(&mut posthoc, |manifest| {
        manifest
            .valid_analyses
            .is_empty()
            .then(|| "no valid analysis".to_string())
    });
    publication_allowed(
        &selected.selection.selection_kind,
        final_evaluation.0.as_ref(),
        policy,
    )?;

    let mut grounding = Grounding::default();
    let (best_correct, seed, top_programs, improvement_trace) =
        ground_search(&mut grounding, &evaluated, &totals, &selected.selection);
    if let Some(summary) =
        fsutil::read_json_opt::<Value>(&paths.run_dir.join("metrics").join("cost_summary.json"))
    {
        if let Some(cost) = summary["total_cost_usd"].as_f64() {
            grounding.number(
                "result.total_cost_usd",
                "cost",
                cost,
                "model spend of the run in US dollars, reported plus estimated",
            );
        }
    }
    for (index, value) in constants(task_goal).into_iter().enumerate() {
        grounding.number(
            &format!("task.constant_{:02}", index + 1),
            "task_constant",
            value,
            "a number stated in the task description",
        );
    }
    for (index, value) in constants(&selected.code).into_iter().enumerate() {
        grounding.number(
            &format!("source.selected_constant_{:03}", index + 1),
            "source_constant",
            value,
            "a numeric constant in the studied program's source",
        );
    }
    if selected.selection.selection_kind != "winner" {
        ground_study(&mut grounding, &selected.selection);
    }
    ground_evidence(
        &mut grounding,
        ablations.0.as_ref(),
        final_evaluation.0.as_ref(),
        posthoc.0.as_ref(),
    );

    let mut limitations = Vec::new();
    match selected.selection.selection_kind.as_str() {
        "winner" => {}
        "baseline" => limitations.push(
            "The search found no correct program that scored above the seed; this paper studies the seed program and reports a negative result."
                .to_string(),
        ),
        _ => limitations.push(
            "The search found no program that scored above the seed; this paper studies the best correct candidate and reports a negative result."
                .to_string(),
        ),
    }
    if totals.failed_proposals > 0 {
        limitations.push(format!(
            "{} {} produced no evaluated program. Such proposals are missing evidence, not measured losses.",
            totals.failed_proposals,
            if totals.failed_proposals == 1 { "proposal" } else { "proposals" }
        ));
    }
    if ablations.0.is_none() {
        limitations.push(format!(
            "{}, so the contribution of individual components was not measured.",
            absence(&ablations.1, "ablation study")
        ));
    } else if ablations
        .0
        .as_ref()
        .is_some_and(|summary| !summary.evaluator_is_stochastic)
    {
        limitations.push(
            "Every replication of the control returned the same score, so the evaluator does not vary with the seed and the reported intervals have zero width."
                .to_string(),
        );
    }
    match &final_evaluation.0 {
        None => limitations.push(format!(
            "{}; every score reported here is a search-time score on the evaluator the search optimised against.",
            absence(&final_evaluation.1, "sealed final evaluation")
        )),
        Some(manifest) => {
            if manifest.outcome != "qualified" {
                limitations.push(
                    "The final evaluation did not qualify the studied program. The cause is known only as far as the evaluator reports it; a failed qualification does not by itself show that inferior performance was the reason."
                        .to_string(),
                );
            }
            limitations.extend(manifest.limitations.iter().cloned());
        }
    }
    if posthoc.0.is_none() {
        limitations.push(format!("{}.", absence(&posthoc.1, "post-hoc analysis")));
    }

    let conclusions = std::fs::read_to_string(investigation.join("conclusions.md"))
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty() && text != genesis_scientist::memory::NO_CONCLUSIONS);
    Ok(PaperData {
        generated_at: chrono::Utc::now().to_rfc3339(),
        task_name: task_name.to_string(),
        task_goal: task_goal.to_string(),
        language: language.to_string(),
        selected_code: selected.code.clone(),
        selection: selected.selection,
        best_correct,
        seed,
        totals,
        top_programs,
        improvement_trace,
        ablations: ablations.0,
        final_evaluation: final_evaluation.0,
        posthoc: posthoc.0,
        distilled_conclusions: conclusions,
        grounding_sources: grounding.sources,
        evidence_sources: BTreeMap::from([
            ("ablations".to_string(), ablations.1),
            ("final_evaluation".to_string(), final_evaluation.1),
            ("posthoc".to_string(), posthoc.1),
        ]),
        limitations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_scientist::artifacts::FinalManifest;

    #[test]
    fn ids_are_never_overwritten_and_non_finite_numbers_are_dropped() {
        let mut grounding = Grounding::default();
        grounding.number("a.b", "metric", 1.0, "first");
        grounding.number("a.b", "metric", 2.0, "second");
        grounding.number("a.c", "metric", f64::NAN, "dropped");
        grounding.number("a.d", "metric", f64::INFINITY, "dropped");
        grounding.count("a.b", 3, "third");
        let ids: Vec<&str> = grounding.sources().keys().map(String::as_str).collect();
        assert_eq!(ids, ["a.b", "a.b_2", "a.b_3"]);
        assert_eq!(grounding.inserted(), grounding.sources().len());
        assert_eq!(grounding.sources()["a.b_2"].value, json!(2.0));
    }

    #[test]
    fn constants_are_distinct_and_in_order() {
        assert_eq!(
            constants("pack n=26 circles; 26 again, radius 0.5 and 1e-3"),
            [26.0, 0.5, 0.001]
        );
        assert!(constants("none").is_empty());
    }

    #[test]
    fn the_policy_only_blocks_an_unqualified_winner() {
        let failed = FinalManifest {
            outcome: "not_qualified".into(),
            ..Default::default()
        };
        let passed = FinalManifest {
            outcome: "qualified".into(),
            ..Default::default()
        };
        assert!(publication_allowed("winner", Some(&failed), "qualified_only").is_err());
        assert!(publication_allowed("winner", Some(&failed), "allow_not_qualified").is_ok());
        assert!(publication_allowed("winner", Some(&passed), "qualified_only").is_ok());
        assert!(publication_allowed("winner", None, "qualified_only").is_ok());
        assert!(
            publication_allowed("non_winner_candidate", Some(&failed), "qualified_only").is_ok()
        );
        assert!(publication_allowed("baseline", Some(&failed), "qualified_only").is_ok());
    }
}
