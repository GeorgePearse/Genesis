//! Which program the downstream phases study.
//!
//! The investigation ends with a winner (a generated program that is
//! correct and beats the seed) or without one. Either way the choice is
//! frozen once, as a copy under `downstream/` plus a manifest with digests,
//! so ablations, the final evaluation, post-hoc analysis and the paper all
//! describe the same bytes, and a resumed run can never quietly study a
//! different program.

use crate::artifacts::{
    AttemptCounts, BestProgramManifest, DownstreamSelection, OUTCOME_NO_WINNER, OUTCOME_WINNER,
};
use crate::paths::RunPaths;
use anyhow::{bail, Context, Result};
use genesis_core::{fsutil, Program, Task};
use genesis_evolve::ProgramDb;
use serde_json::Value;
use std::path::PathBuf;

fn is_copy(program: &Program) -> bool {
    ["_is_island_copy", "_spawned_island"].iter().any(|flag| {
        program
            .metadata
            .get(*flag)
            .and_then(Value::as_bool)
            .unwrap_or(false)
    })
}

/// The programs this run itself evaluated, oldest first.
pub fn own_programs(db: &ProgramDb) -> Result<Vec<Program>> {
    Ok(db
        .all()?
        .into_iter()
        .filter(|program| !is_copy(program))
        .collect())
}

/// What the search established: the best-program manifest and whether there
/// is a winner.
///
/// A winner is a correct generated program that beats the seed's score, or
/// any correct generated program when the seed is itself incorrect.
pub fn judge_investigation(
    paths: &RunPaths,
    task: &Task,
    db: &ProgramDb,
) -> Result<(BestProgramManifest, String)> {
    let programs = own_programs(db)?;
    let seed = programs.iter().find(|program| program.generation == 0);
    let seed_score = seed
        .filter(|seed| seed.correct)
        .map(|seed| seed.combined_score);
    let best_generated = programs
        .iter()
        .filter(|program| program.generation > 0 && program.correct)
        .max_by(|a, b| {
            a.combined_score
                .total_cmp(&b.combined_score)
                .then(b.generation.cmp(&a.generation))
        });
    let winner = best_generated
        .filter(|program| seed_score.is_none_or(|score| program.combined_score > score));

    let investigation = paths.investigation();
    let program_path = |program: &Program| {
        paths.relative(
            &investigation
                .join(format!("gen_{}", program.generation))
                .join(task.program_file_name()),
        )
    };
    let now = chrono::Utc::now().to_rfc3339();
    let manifest = match (winner, seed) {
        (Some(winner), _) => BestProgramManifest {
            program_id: Some(winner.id.clone()),
            generation: Some(winner.generation),
            program_path: Some(program_path(winner)),
            combined_score: Some(winner.combined_score),
            correct: Some(true),
            source: format!("generation_{}", winner.generation),
            patch_name: winner.meta_str("patch_name").map(str::to_string),
            updated_at: now,
        },
        (None, Some(seed)) => BestProgramManifest {
            program_id: Some(seed.id.clone()),
            generation: Some(0),
            program_path: Some(program_path(seed)),
            combined_score: Some(seed.combined_score),
            correct: Some(seed.correct),
            source: "seed".to_string(),
            patch_name: None,
            updated_at: now,
        },
        (None, None) => BestProgramManifest {
            source: "seed".to_string(),
            updated_at: now,
            ..Default::default()
        },
    };
    let outcome = if winner.is_some() {
        OUTCOME_WINNER
    } else {
        OUTCOME_NO_WINNER
    };
    Ok((manifest, outcome.to_string()))
}

/// The frozen selection with its files loaded and verified.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedProgram {
    pub selection: DownstreamSelection,
    pub program_path: PathBuf,
    pub baseline_path: PathBuf,
    pub code: String,
    pub baseline_code: String,
}

/// Load the frozen selection, refusing files that no longer match their
/// recorded digests. `Ok(None)` means nothing was selected.
pub fn load_selection(paths: &RunPaths) -> Result<Option<SelectedProgram>> {
    let manifest_path = paths.downstream_selection();
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let selection: DownstreamSelection = fsutil::read_json(&manifest_path)?;
    let program_path = paths.resolve(&selection.program_path);
    let baseline_path = paths.resolve(&selection.baseline_path);
    let code = std::fs::read_to_string(&program_path)
        .with_context(|| format!("the frozen program {} is missing", program_path.display()))?;
    let baseline_code = std::fs::read_to_string(&baseline_path)
        .with_context(|| format!("the frozen baseline {} is missing", baseline_path.display()))?;
    if fsutil::sha256_hex(code.as_bytes()) != selection.program_sha256 {
        bail!(
            "the frozen program {} no longer matches the digest in {}; the selection is stale and must not be bypassed",
            program_path.display(),
            manifest_path.display()
        );
    }
    if fsutil::sha256_hex(baseline_code.as_bytes()) != selection.baseline_sha256 {
        bail!(
            "the frozen baseline {} no longer matches its recorded digest",
            baseline_path.display()
        );
    }
    Ok(Some(SelectedProgram {
        selection,
        program_path,
        baseline_path,
        code,
        baseline_code,
    }))
}

/// Choose and freeze the program for the downstream phases.
///
/// With a winner, that is the winner. Without one, `best_available` takes
/// the highest-scoring correct generated program, or else the seed if it is
/// correct; `skip` selects nothing. An existing selection is validated and
/// returned unchanged.
pub fn freeze_selection(
    paths: &RunPaths,
    task: &Task,
    db: &ProgramDb,
    no_winner_policy: &str,
) -> Result<Option<SelectedProgram>> {
    if let Some(existing) = load_selection(paths)? {
        return Ok(Some(existing));
    }
    let programs = own_programs(db)?;
    let seed = programs
        .iter()
        .find(|program| program.generation == 0)
        .context("the investigation database has no seed program")?;
    let (manifest, outcome) = judge_investigation(paths, task, db)?;
    let generated: Vec<&Program> = programs
        .iter()
        .filter(|program| program.generation > 0)
        .collect();
    let failed_proposals = db
        .attempts()?
        .iter()
        .filter(|attempt| attempt.status == "failed" && attempt.stage != "evaluation")
        .count();

    let (chosen, kind) = if outcome == OUTCOME_WINNER {
        let id = manifest.program_id.as_deref().unwrap_or_default();
        (programs.iter().find(|program| program.id == id), "winner")
    } else if no_winner_policy == "skip" {
        (None, "")
    } else {
        let candidate = generated
            .iter()
            .filter(|program| program.correct)
            .max_by(|a, b| {
                a.combined_score
                    .total_cmp(&b.combined_score)
                    .then(b.generation.cmp(&a.generation))
            })
            .copied();
        match candidate {
            Some(candidate) => (Some(candidate), "non_winner_candidate"),
            None if seed.correct => (Some(seed), "baseline"),
            None => (None, ""),
        }
    };
    let Some(chosen) = chosen else {
        return Ok(None);
    };

    let file_name = task.program_file_name();
    let program_path = paths.downstream_program().join(&file_name);
    let baseline_path = paths
        .run_dir
        .join("downstream")
        .join("baseline")
        .join(&file_name);
    fsutil::write_text(&program_path, &chosen.code)?;
    fsutil::write_text(&baseline_path, &seed.code)?;
    let selection = DownstreamSelection {
        schema_version: 1,
        investigation_outcome: outcome,
        selection_kind: kind.to_string(),
        program_id: chosen.id.clone(),
        generation: chosen.generation,
        combined_score: chosen.combined_score,
        correct: chosen.correct,
        program_path: paths.relative(&program_path),
        program_sha256: fsutil::sha256_hex(chosen.code.as_bytes()),
        baseline_path: paths.relative(&baseline_path),
        baseline_sha256: fsutil::sha256_hex(seed.code.as_bytes()),
        baseline_score: seed.correct.then_some(seed.combined_score),
        baseline_correct: seed.correct,
        attempt_counts: AttemptCounts {
            total: generated.len() + failed_proposals,
            scored: generated
                .iter()
                .filter(|program| {
                    !program
                        .metadata
                        .get("evaluation_failed")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
                .count(),
            unscored: failed_proposals
                + generated
                    .iter()
                    .filter(|program| {
                        program
                            .metadata
                            .get("evaluation_failed")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                    })
                    .count(),
        },
        task_fingerprint: task.fingerprint()?,
        selected_at: chrono::Utc::now().to_rfc3339(),
    };
    // The manifest is written last: its presence means both files exist.
    fsutil::write_json(&paths.downstream_selection(), &selection)?;
    load_selection(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_evolve::DatabaseConfig;

    fn task(dir: &std::path::Path) -> Task {
        let task_dir = dir.join("task");
        std::fs::create_dir_all(&task_dir).unwrap();
        std::fs::write(task_dir.join("initial.py"), "x = 0\n").unwrap();
        std::fs::write(task_dir.join("evaluate.sh"), "true\n").unwrap();
        Task::load(&task_dir).unwrap()
    }

    fn database(rows: &[(u32, f64, bool)]) -> ProgramDb {
        let db = ProgramDb::open(
            None,
            DatabaseConfig {
                num_islands: 2,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let mut seed_id = None;
        for (generation, score, correct) in rows {
            let mut program = Program::new(format!("x = {generation}\n"), "python");
            program.generation = *generation;
            program.combined_score = *score;
            program.correct = *correct;
            program.parent_id = seed_id.clone();
            db.add(&mut program).unwrap();
            seed_id.get_or_insert(program.id.clone());
        }
        db
    }

    #[test]
    fn a_correct_program_that_beats_the_seed_is_the_winner() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, task) = (RunPaths::new(dir.path().join("run")), task(dir.path()));
        let db = database(&[
            (0, 1.0, true),
            (1, 0.5, true),
            (2, 2.0, true),
            (3, 9.0, false),
            (4, 2.0, true),
        ]);
        let (manifest, outcome) = judge_investigation(&paths, &task, &db).unwrap();
        assert_eq!(outcome, OUTCOME_WINNER);
        // Equal scores: the earlier generation wins.
        assert_eq!(
            (manifest.generation, manifest.combined_score),
            (Some(2), Some(2.0))
        );
        assert_eq!(
            manifest.program_path.as_deref(),
            Some("investigation/gen_2/main.py")
        );

        let selected = freeze_selection(&paths, &task, &db, "best_available")
            .unwrap()
            .unwrap();
        assert_eq!(selected.selection.selection_kind, "winner");
        assert_eq!(
            (selected.code.as_str(), selected.baseline_code.as_str()),
            ("x = 2\n", "x = 0\n")
        );
        assert_eq!(selected.selection.baseline_score, Some(1.0));
        assert_eq!(selected.selection.attempt_counts.total, 4);
        // Frozen: a later, better program does not change the selection.
        let better = database(&[(0, 1.0, true), (1, 50.0, true)]);
        assert_eq!(
            freeze_selection(&paths, &task, &better, "best_available")
                .unwrap()
                .unwrap(),
            selected
        );
    }

    #[test]
    fn without_a_winner_the_policy_decides() {
        let dir = tempfile::tempdir().unwrap();
        let task = task(dir.path());
        let no_improvement = database(&[
            (0, 1.0, true),
            (1, 0.4, true),
            (2, 0.9, true),
            (3, 5.0, false),
        ]);
        let paths = RunPaths::new(dir.path().join("a"));
        let (manifest, outcome) = judge_investigation(&paths, &task, &no_improvement).unwrap();
        assert_eq!(
            (outcome.as_str(), manifest.source.as_str()),
            (OUTCOME_NO_WINNER, "seed")
        );
        let selected = freeze_selection(&paths, &task, &no_improvement, "best_available")
            .unwrap()
            .unwrap();
        assert_eq!(
            (
                selected.selection.selection_kind.as_str(),
                selected.selection.generation
            ),
            ("non_winner_candidate", 2)
        );

        let paths = RunPaths::new(dir.path().join("b"));
        assert!(freeze_selection(&paths, &task, &no_improvement, "skip")
            .unwrap()
            .is_none());

        let only_seed = database(&[(0, 1.0, true), (1, 0.0, false)]);
        let paths = RunPaths::new(dir.path().join("c"));
        let selected = freeze_selection(&paths, &task, &only_seed, "best_available")
            .unwrap()
            .unwrap();
        assert_eq!(selected.selection.selection_kind, "baseline");

        let nothing_correct = database(&[(0, 0.0, false), (1, 0.0, false)]);
        let paths = RunPaths::new(dir.path().join("d"));
        assert!(
            freeze_selection(&paths, &task, &nothing_correct, "best_available")
                .unwrap()
                .is_none()
        );

        // An incorrect seed: any correct generated program wins.
        let fixed = database(&[(0, 0.0, false), (1, 0.1, true)]);
        assert_eq!(
            judge_investigation(&paths, &task, &fixed).unwrap().1,
            OUTCOME_WINNER
        );
    }

    #[test]
    fn a_tampered_frozen_program_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (paths, task) = (RunPaths::new(dir.path().join("run")), task(dir.path()));
        let db = database(&[(0, 1.0, true), (1, 2.0, true)]);
        let selected = freeze_selection(&paths, &task, &db, "best_available")
            .unwrap()
            .unwrap();
        std::fs::write(&selected.program_path, "x = 999\n").unwrap();
        let error = load_selection(&paths).unwrap_err().to_string();
        assert!(error.contains("no longer matches the digest"), "{error}");
        assert!(freeze_selection(&paths, &task, &db, "best_available").is_err());
    }
}
