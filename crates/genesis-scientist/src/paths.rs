//! The run envelope: one directory per run, one subdirectory per phase.
//!
//! ```text
//! results/<task>/<run_id>/
//! ├── run_config.yaml            resolved configuration
//! ├── events.jsonl               every phase's journal
//! ├── metrics/                   cost_events.jsonl, cost_summary.json
//! ├── task/                      immutable run-local task snapshot
//! ├── task_drafting/             drafting log, when the scaffold was drafted
//! ├── investigation/             programs.sqlite, gen_N/, best/, ideas, memory
//! ├── downstream/program/        frozen fallback program when search found no winner
//! ├── ablations/                 controls, baselines and ablations with replications
//! ├── final_evaluation/          sealed final evaluation
//! ├── posthoc/                   post-hoc analyses
//! ├── paper/                     manuscript, evidence audit, manifest, reviews
//! └── revision/                  review-driven revision attempts
//! ```

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct RunPaths {
    pub run_dir: PathBuf,
}

impl RunPaths {
    pub fn new(run_dir: impl Into<PathBuf>) -> Self {
        Self {
            run_dir: run_dir.into(),
        }
    }

    /// `<results_root>/<task>/<run_id>`.
    pub fn for_new_run(results_root: &Path, task_name: &str, run_id: &str) -> Self {
        Self::new(results_root.join(task_name).join(run_id))
    }

    pub fn config(&self) -> PathBuf {
        self.run_dir.join("run_config.yaml")
    }

    pub fn events(&self) -> PathBuf {
        self.run_dir.join("events.jsonl")
    }

    pub fn task(&self) -> PathBuf {
        self.run_dir.join("task")
    }

    pub fn drafting(&self) -> PathBuf {
        self.run_dir.join("task_drafting")
    }

    pub fn investigation(&self) -> PathBuf {
        self.run_dir.join("investigation")
    }

    pub fn ablations(&self) -> PathBuf {
        self.run_dir.join("ablations")
    }

    pub fn final_evaluation(&self) -> PathBuf {
        self.run_dir.join("final_evaluation")
    }

    pub fn posthoc(&self) -> PathBuf {
        self.run_dir.join("posthoc")
    }

    pub fn paper(&self) -> PathBuf {
        self.run_dir.join("paper")
    }

    pub fn revision(&self) -> PathBuf {
        self.run_dir.join("revision")
    }

    /// Receipt written when the investigation starts and again when it ends.
    pub fn completion_receipt(&self) -> PathBuf {
        self.run_dir.join(".investigation_completion.json")
    }

    /// The frozen choice of program every downstream phase studies.
    pub fn downstream_selection(&self) -> PathBuf {
        self.run_dir.join(".downstream_selection.json")
    }

    pub fn downstream_program(&self) -> PathBuf {
        self.run_dir.join("downstream").join("program")
    }

    /// `path` relative to the run directory, for manifests that must stay
    /// valid when the run is moved.
    pub fn relative(&self, path: &Path) -> String {
        path.strip_prefix(&self.run_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }

    pub fn resolve(&self, relative: &str) -> PathBuf {
        let path = Path::new(relative);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.run_dir.join(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_relative_to_the_run_directory() {
        let paths =
            RunPaths::for_new_run(Path::new("results"), "circle_packing", "20261125_101500");
        assert_eq!(
            paths.run_dir,
            Path::new("results/circle_packing/20261125_101500")
        );
        let manifest = paths.paper().join("latest_manifest.json");
        assert_eq!(paths.relative(&manifest), "paper/latest_manifest.json");
        assert_eq!(paths.resolve("paper/latest_manifest.json"), manifest);
        assert_eq!(paths.resolve("/abs/x"), Path::new("/abs/x"));
        assert_eq!(paths.relative(Path::new("/elsewhere/x")), "/elsewhere/x");
        assert!(paths.downstream_program().ends_with("downstream/program"));
    }
}
