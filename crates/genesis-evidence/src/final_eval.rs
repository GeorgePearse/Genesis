//! The sealed final evaluation: one look at held-out evidence.
//!
//! The selected program and the baseline it came from are each evaluated on
//! the same final seeds: with the task's own final evaluator when it ships
//! one, otherwise with the search evaluator on seeds nothing has used yet.
//! The evaluation happens once. An invocation record is created before
//! anything runs, and a directory that holds one without a finished
//! manifest refuses to run again, because a second look at held-out data is
//! no longer held out. The task and the program are checked against the
//! digests frozen at selection, before and after.
//!
//! The outcome is the evaluator's own verdict: `qualified` when every final
//! episode of the selected program is correct. Whether the improvement over
//! the baseline holds up on the final seeds is reported beside it, as
//! separate evidence.
//!
//! This follows the Agentic Scientist's final evaluation (once-only
//! invocation, disjoint seeds, digest checks, the evaluator's boolean as the
//! outcome) without its container broker.

use crate::common::{self, EvalJob, EvalSettings};
use anyhow::{bail, ensure, Context, Result};
use genesis_core::stats::{self, Comparison};
use genesis_core::{fsutil, EvalOutcome};
use genesis_scientist::artifacts::{AblationSummary, FinalEpisode, FinalManifest};
use genesis_scientist::{PhaseContext, RunPaths};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

pub const QUALIFIED: &str = "qualified";
pub const NOT_QUALIFIED: &str = "not_qualified";
const REFUSE_RERUN: &str = "final evaluation already started; refusing rerun";

/// What a paper must say when the selected program did not qualify.
const NOT_QUALIFIED_LIMITATIONS: [&str; 3] = [
    "The selected program did not qualify in the sealed final evaluation: at least one final episode was not judged \
     correct. This must be disclosed wherever the method's results are reported.",
    "The final evaluation establishes that qualification failed; it does not establish why. Do not infer that the \
     method is inferior, or name a cause, from the failure alone.",
    "Search-time scores and final-evaluation results are separate evidence, obtained with different seeds or a \
     different evaluator. Report them separately and do not substitute one for the other.",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FinalEvalConfig {
    /// Final episodes per program, one per seed.
    pub num_replications: usize,
    /// First final seed. Seeds an ablation study already used are skipped.
    pub seed_base: u64,
    pub max_parallel_evals: usize,
    /// Overrides the task's evaluation time limit.
    pub timeout_secs: Option<u64>,
}

impl Default for FinalEvalConfig {
    fn default() -> Self {
        Self {
            num_replications: 5,
            seed_base: 1_000_003,
            max_parallel_evals: 4,
            timeout_secs: None,
        }
    }
}

/// The held-out seeds: `num_replications` distinct seeds counted up from
/// `seed_base`, leaving out every seed in `used`. They are disjoint from
/// `used` by construction, whatever the configuration.
pub fn final_seeds(config: &FinalEvalConfig, used: &[u64]) -> Vec<u64> {
    let mut seeds = Vec::with_capacity(config.num_replications);
    let mut candidate = config.seed_base;
    while seeds.len() < config.num_replications {
        if !used.contains(&candidate) {
            seeds.push(candidate);
        }
        candidate = candidate.wrapping_add(1);
    }
    seeds
}

fn episode(seed: u64, outcome: &EvalOutcome) -> FinalEpisode {
    FinalEpisode {
        seed,
        score: outcome.combined_score,
        correct: outcome.correct,
        duration_seconds: outcome.runtime_secs,
    }
}

fn correct_scores(episodes: &[FinalEpisode]) -> Vec<f64> {
    episodes
        .iter()
        .filter(|episode| episode.correct && episode.score.is_finite())
        .map(|episode| episode.score)
        .collect()
}

/// Whether the final episodes support an improvement of the candidate over
/// the baseline: the Welch 95% interval of the difference lies above zero,
/// or, when neither side varies, the candidate's mean is strictly higher.
pub fn improvement_supported(
    candidate: &[f64],
    baseline: &[f64],
    comparison: Option<&Comparison>,
) -> bool {
    let Some(comparison) = comparison else {
        return false;
    };
    if let Some(low) = comparison.diff_ci95_low {
        return low > 0.0;
    }
    let constant =
        |scores: &[f64]| !scores.is_empty() && scores.windows(2).all(|pair| pair[0] == pair[1]);
    constant(candidate) && constant(baseline) && comparison.mean_diff > 0.0
}

/// Turn the two sets of final episodes into the manifest's verdict and
/// statistics. `same_program` says the selected program is the baseline.
pub fn assess(manifest: &mut FinalManifest, same_program: bool) {
    let candidate = correct_scores(&manifest.episodes);
    let baseline = correct_scores(&manifest.baseline_episodes);
    manifest.correct =
        !manifest.episodes.is_empty() && manifest.episodes.iter().all(|episode| episode.correct);
    manifest.outcome = if manifest.correct {
        QUALIFIED
    } else {
        NOT_QUALIFIED
    }
    .to_string();
    manifest.candidate = (!candidate.is_empty()).then(|| stats::summarize(&candidate));
    manifest.baseline = (!baseline.is_empty()).then(|| stats::summarize(&baseline));
    manifest.comparison = (!candidate.is_empty() && !baseline.is_empty())
        .then(|| stats::compare(&candidate, &baseline));
    manifest.improvement_supported =
        !same_program && improvement_supported(&candidate, &baseline, manifest.comparison.as_ref());
    manifest.limitations.clear();
    if !manifest.correct {
        manifest.limitations.extend(
            NOT_QUALIFIED_LIMITATIONS
                .iter()
                .map(|sentence| sentence.to_string()),
        );
    }
    if same_program {
        manifest.limitations.push(
            "The selected program is the baseline itself, so the final evaluation compares it with nothing and \
             supports no claim of improvement."
                .to_string(),
        );
    }
}

#[cfg(unix)]
fn seal(root: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            seal(&path);
        } else if kind.is_file() {
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444));
        }
    }
    let _ = std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o555));
}

#[cfg(not(unix))]
fn seal(_root: &Path) {}

/// The finished final evaluation of this run, if there is one.
///
/// The manifest is refused when it is not for the frozen selection's
/// program, or when its verdict contradicts its own episodes.
pub fn load_final_manifest(paths: &RunPaths) -> Result<Option<FinalManifest>> {
    let path = paths.final_evaluation().join("final_manifest.json");
    if !path.is_file() {
        return Ok(None);
    }
    let manifest: FinalManifest = fsutil::read_json(&path)?;
    let selected = common::selected(paths)?;
    ensure!(
        manifest.program_sha256 == selected.selection.program_sha256,
        "{} is for a different program than the frozen selection",
        path.display()
    );
    ensure!(
        manifest.status == "completed",
        "{} does not record a completed evaluation",
        path.display()
    );
    let all_correct =
        !manifest.episodes.is_empty() && manifest.episodes.iter().all(|episode| episode.correct);
    ensure!(
        manifest.correct == all_correct && (manifest.outcome == QUALIFIED) == manifest.correct,
        "{} states an outcome its own episodes do not support",
        path.display()
    );
    Ok(Some(manifest))
}

/// Run the sealed final evaluation of the frozen selection under
/// `<run>/final_evaluation`. A finished evaluation is returned as it is; a
/// started and unfinished one is an error.
pub async fn run_final_evaluation(
    context: &PhaseContext,
    config: &FinalEvalConfig,
) -> Result<FinalManifest> {
    let paths = &context.paths;
    let events = &context.events;
    let dir = paths.final_evaluation();
    if let Some(manifest) = load_final_manifest(paths)? {
        events.info(
            "final_evaluation_reused",
            json!({"outcome": manifest.outcome}),
        );
        return Ok(manifest);
    }
    let invocation_path = dir.join("invocation.json");
    if invocation_path.exists() {
        bail!("{REFUSE_RERUN} ({})", invocation_path.display());
    }
    ensure!(
        config.num_replications >= 1,
        "num_replications must be at least 1"
    );
    if context.cancelled() {
        bail!("the final evaluation was cancelled before it started");
    }

    let selected = common::selected(paths)?;
    let selection = &selected.selection;
    let task = &context.task;
    let fingerprint = task.fingerprint()?;
    ensure!(
        fingerprint == selection.task_fingerprint,
        "the task has changed since the selection was frozen (fingerprint {} against {}); the final evaluation \
         only runs on the task the search used",
        &fingerprint[..12.min(fingerprint.len())],
        &selection.task_fingerprint[..12.min(selection.task_fingerprint.len())]
    );

    let ablation_seeds =
        fsutil::read_json_opt::<AblationSummary>(&paths.ablations().join("pipeline_summary.json"))
            .map(|summary| summary.seeds)
            .unwrap_or_default();
    let seeds = final_seeds(config, &ablation_seeds);
    assert!(
        seeds.iter().all(|seed| !ablation_seeds.contains(seed)),
        "final seeds must be held out"
    );
    let uses_final_evaluator = task.final_eval.is_some();
    let evaluator = if uses_final_evaluator {
        "evaluate_final"
    } else {
        "search_evaluator_heldout_seeds"
    };
    let same_program = selected.code == selected.baseline_code;

    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let started_at = chrono::Utc::now().to_rfc3339();
    let invocation = json!({
        "schema_version": 1,
        "status": "started",
        "started_at": started_at,
        "task_fingerprint": fingerprint,
        "program_sha256": selection.program_sha256,
        "baseline_sha256": selection.baseline_sha256,
        "evaluator": evaluator,
        "seeds": seeds,
    });
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&invocation_path)
    {
        Ok(mut file) => {
            file.write_all(serde_json::to_string_pretty(&invocation)?.as_bytes())?;
            file.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            bail!("{REFUSE_RERUN} ({})", invocation_path.display())
        }
        Err(error) => {
            return Err(error).with_context(|| format!("creating {}", invocation_path.display()))
        }
    }
    events.info(
        "final_evaluation_start",
        json!({"evaluator": evaluator, "seeds": seeds, "selection_kind": selection.selection_kind}),
    );

    let file_name = task.program_file_name();
    let candidate_path = dir.join("program").join(&file_name);
    let baseline_path = dir.join("baseline").join(&file_name);
    fsutil::write_text(&candidate_path, &selected.code)?;
    fsutil::write_text(&baseline_path, &selected.baseline_code)?;
    let mut jobs = Vec::new();
    for seed in &seeds {
        jobs.push(EvalJob {
            key: "candidate".to_string(),
            program_path: candidate_path.clone(),
            results_dir: dir
                .join("results")
                .join("candidate")
                .join(format!("seed_{seed}")),
            seed: *seed,
        });
        if !same_program {
            jobs.push(EvalJob {
                key: "baseline".to_string(),
                program_path: baseline_path.clone(),
                results_dir: dir
                    .join("results")
                    .join("baseline")
                    .join(format!("seed_{seed}")),
                seed: *seed,
            });
        }
    }
    let settings = EvalSettings {
        task,
        parallel: config.max_parallel_evals,
        timeout: config.timeout_secs.map(Duration::from_secs),
        final_evaluator: uses_final_evaluator,
        cancel: &context.cancel,
    };
    let ran = common::run_jobs(&settings, jobs).await;

    let fail = |reason: String| -> anyhow::Error {
        let _ = fsutil::write_json(
            &dir.join("evaluation_failure.json"),
            &json!({"failed_at": chrono::Utc::now().to_rfc3339(), "reason": reason}),
        );
        events.error("final_evaluation_failed", json!({"reason": reason}));
        anyhow::anyhow!(
            "{reason}; the invocation record stays, so this final evaluation cannot be run again"
        )
    };
    let mut by_key: BTreeMap<&str, Vec<FinalEpisode>> = BTreeMap::new();
    for (job, outcome) in &ran {
        match outcome {
            Some(outcome) => by_key
                .entry(job.key.as_str())
                .or_default()
                .push(episode(job.seed, outcome)),
            None => return Err(fail("the final evaluation was interrupted".to_string())),
        }
    }
    if task.fingerprint()? != fingerprint {
        return Err(fail(
            "the task changed while the final evaluation was running".to_string(),
        ));
    }
    if fsutil::sha256_file(&candidate_path)? != selection.program_sha256 {
        return Err(fail(
            "the selected program changed while the final evaluation was running".to_string(),
        ));
    }

    let mut episodes = by_key.remove("candidate").unwrap_or_default();
    episodes.sort_by_key(|episode| episode.seed);
    let mut baseline_episodes = if same_program {
        episodes.clone()
    } else {
        by_key.remove("baseline").unwrap_or_default()
    };
    baseline_episodes.sort_by_key(|episode| episode.seed);
    let mut manifest = FinalManifest {
        schema_version: 1,
        status: "completed".to_string(),
        started_at,
        completed_at: chrono::Utc::now().to_rfc3339(),
        evaluator: evaluator.to_string(),
        seeds,
        episodes,
        baseline_episodes,
        program_sha256: selection.program_sha256.clone(),
        task_fingerprint: fingerprint,
        ..FinalManifest::default()
    };
    assess(&mut manifest, same_program);
    fsutil::write_json(&dir.join("final_manifest.json"), &manifest)?;
    for sealed in ["results", "program", "baseline"] {
        seal(&dir.join(sealed));
    }
    events.info(
        "final_evaluation_end",
        json!({
            "outcome": manifest.outcome,
            "candidate_mean": manifest.candidate.as_ref().map(|summary| summary.mean),
            "baseline_mean": manifest.baseline.as_ref().map(|summary| summary.mean),
            "mean_diff": manifest.comparison.as_ref().map(|comparison| comparison.mean_diff),
            "improvement_supported": manifest.improvement_supported,
        }),
    );
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn episodes(scores: &[(f64, bool)]) -> Vec<FinalEpisode> {
        scores
            .iter()
            .enumerate()
            .map(|(index, (score, correct))| FinalEpisode {
                seed: index as u64,
                score: *score,
                correct: *correct,
                duration_seconds: 0.1,
            })
            .collect()
    }

    #[test]
    fn final_seeds_count_up_and_skip_used_ones() {
        let config = FinalEvalConfig {
            num_replications: 4,
            seed_base: 10,
            ..Default::default()
        };
        assert_eq!(final_seeds(&config, &[]), [10, 11, 12, 13]);
        assert_eq!(final_seeds(&config, &[11, 12, 99]), [10, 13, 14, 15]);
        let at_the_edge = FinalEvalConfig {
            num_replications: 3,
            seed_base: u64::MAX,
            ..Default::default()
        };
        assert_eq!(final_seeds(&at_the_edge, &[0]), [u64::MAX, 1, 2]);
        assert!(final_seeds(
            &FinalEvalConfig {
                num_replications: 0,
                ..Default::default()
            },
            &[]
        )
        .is_empty());
    }

    #[test]
    fn qualification_is_every_episode_correct_and_improvement_is_separate() {
        let mut manifest = FinalManifest {
            episodes: episodes(&[(5.0, true), (5.2, true), (5.1, true)]),
            baseline_episodes: episodes(&[(4.0, true), (4.1, true), (3.9, true)]),
            ..Default::default()
        };
        assess(&mut manifest, false);
        assert_eq!(
            (manifest.correct, manifest.outcome.as_str()),
            (true, QUALIFIED)
        );
        assert!(manifest.improvement_supported && manifest.limitations.is_empty());
        assert!((manifest.comparison.as_ref().unwrap().mean_diff - 1.1).abs() < 1e-9);

        manifest.episodes = episodes(&[(5.0, true), (0.0, false), (5.1, true)]);
        assess(&mut manifest, false);
        assert_eq!(
            (manifest.correct, manifest.outcome.as_str()),
            (false, NOT_QUALIFIED)
        );
        assert_eq!(
            manifest.candidate.as_ref().unwrap().n,
            2,
            "statistics are over correct episodes"
        );
        assert_eq!(manifest.limitations.len(), 3);
        assert!(manifest.limitations[1].contains("does not establish why"));

        // Qualified, but the final seeds do not separate it from the baseline.
        manifest.episodes = episodes(&[(4.0, true), (4.3, true), (3.8, true)]);
        assess(&mut manifest, false);
        assert!(manifest.correct && !manifest.improvement_supported);

        assess(&mut manifest, true);
        assert!(!manifest.improvement_supported);
        assert!(manifest
            .limitations
            .last()
            .unwrap()
            .contains("the baseline itself"));

        let mut empty = FinalManifest::default();
        assess(&mut empty, false);
        assert_eq!(empty.outcome, NOT_QUALIFIED);
        assert!(empty.candidate.is_none() && empty.comparison.is_none());
    }

    #[test]
    fn a_deterministic_evaluator_supports_improvement_by_its_means() {
        let comparison = stats::compare(&[5.0, 5.0], &[4.0, 4.0]);
        assert!(comparison.diff_ci95_low.is_none());
        assert!(improvement_supported(
            &[5.0, 5.0],
            &[4.0, 4.0],
            Some(&comparison)
        ));
        let equal = stats::compare(&[4.0, 4.0], &[4.0, 4.0]);
        assert!(!improvement_supported(
            &[4.0, 4.0],
            &[4.0, 4.0],
            Some(&equal)
        ));
        let single = stats::compare(&[5.0], &[4.0]);
        assert!(improvement_supported(&[5.0], &[4.0], Some(&single)));
        // One side varies and the other has a single episode: no interval, no claim.
        let lopsided = stats::compare(&[5.0, 6.0], &[4.0]);
        assert!(!improvement_supported(&[5.0, 6.0], &[4.0], Some(&lopsided)));
        assert!(!improvement_supported(&[5.0], &[4.0], None));
    }
}
