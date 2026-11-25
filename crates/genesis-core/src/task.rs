//! The task contract: a directory holding a goal, a seed program and an
//! evaluator.
//!
//! ```text
//! my_task/
//! ├── task.md        # the goal, in prose (optional but recommended)
//! ├── task.toml      # optional manifest; everything below can be inferred
//! ├── initial.rs     # seed program, any supported language
//! └── evaluate.py    # or evaluate.sh, or any command named in task.toml
//! ```
//!
//! Three layouts are recognised without a manifest: the Genesis/ShinkaEvolve
//! one (`initial.<ext>` + `evaluate.py`), the Agentic Scientist one
//! (`program_0/program_spec.toml` naming an entry file under `src/`), and a
//! shell evaluator (`evaluate.sh <program_path> <results_dir>`). Both Python
//! layouts share one evaluator command line:
//! `evaluate.py --program_path <file> --results_dir <dir>`.

use crate::eval::{run_evaluation, EvalOutcome, EvalRequest, EvalSpec};
use crate::fsutil;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// File extensions the engine can evolve, with their canonical language.
pub const LANGUAGE_EXTENSIONS: &[(&str, &str)] = &[
    ("py", "python"),
    ("rs", "rust"),
    ("cpp", "cpp"),
    ("cc", "cpp"),
    ("cu", "cuda"),
    ("go", "go"),
    ("jl", "julia"),
    ("swift", "swift"),
    ("json", "json"),
    ("sv", "verilog"),
    ("f90", "fortran"),
    ("md", "markdown"),
    ("wl", "wolfram"),
    ("js", "javascript"),
    ("ts", "typescript"),
    ("c", "c"),
    ("sh", "bash"),
];

pub fn language_for_extension(ext: &str) -> Option<&'static str> {
    LANGUAGE_EXTENSIONS
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(ext))
        .map(|(_, language)| *language)
}

pub fn extension_for_language(language: &str) -> &'static str {
    let wanted = language.trim().to_ascii_lowercase();
    LANGUAGE_EXTENSIONS
        .iter()
        .find(|(_, candidate)| *candidate == wanted)
        .map(|(ext, _)| *ext)
        .unwrap_or("txt")
}

/// `task.toml`. Every field is optional.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct TaskManifest {
    pub name: Option<String>,
    pub language: Option<String>,
    /// Seed program, relative to the task directory.
    pub init_program: Option<String>,
    /// Evaluator argument vector; see [`EvalSpec::command`] for placeholders.
    pub eval_command: Option<Vec<String>>,
    pub eval_timeout_secs: Option<u64>,
    pub eval_env: BTreeMap<String, String>,
    /// A separate evaluator for the sealed final evaluation, with the same
    /// placeholders. Inferred from `evaluate_final.py` or `evaluate_final.sh`.
    pub final_eval_command: Option<Vec<String>>,
    /// Overrides `task.md` as the task description shown to models.
    pub system_message: Option<String>,
    /// Extra paths left out of run-local snapshots.
    pub ignore: Vec<String>,
    /// True when a larger `combined_score` is better. Scores are always
    /// maximised internally; this only labels reports.
    pub maximize: Option<bool>,
    /// Best published or known score, when there is one, for context.
    pub reference_score: Option<f64>,
}

/// A loaded task, ready to evaluate programs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Task {
    pub name: String,
    pub dir: PathBuf,
    /// The goal in prose, from `task.md` or the manifest.
    pub goal: String,
    pub language: String,
    pub init_program: PathBuf,
    pub eval: EvalSpec,
    /// The held-out evaluator, when the task ships one. It is never run
    /// during the search.
    pub final_eval: Option<EvalSpec>,
    pub manifest: TaskManifest,
}

const DEFAULT_IGNORE: &[&str] = &[
    ".git",
    "__pycache__",
    "*.pyc",
    ".venv",
    "node_modules",
    "target",
    "results",
    ".DS_Store",
];

fn find_initial(dir: &Path) -> Option<PathBuf> {
    for (ext, _) in LANGUAGE_EXTENSIONS {
        let candidate = dir.join(format!("initial.{ext}"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// `program_0/program_spec.toml` names the entry file relative to its
/// `code_root` (`src`); the default is `src/program.<ext>`.
fn find_program_zero(dir: &Path) -> Option<PathBuf> {
    let root = dir.join("program_0");
    if !root.is_dir() {
        return None;
    }
    let spec = root.join("program_spec.toml");
    if let Ok(raw) = std::fs::read_to_string(&spec) {
        if let Ok(value) = raw.parse::<toml::Value>() {
            let code_root = value
                .get("code_root")
                .and_then(toml::Value::as_str)
                .unwrap_or("src");
            let entry = value
                .get("entrypoint")
                .and_then(|entry| entry.get("path"))
                .and_then(toml::Value::as_str);
            if let Some(entry) = entry {
                let candidate = root.join(code_root).join(entry);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    for (ext, _) in LANGUAGE_EXTENSIONS {
        let candidate = root.join(format!("src/program.{ext}"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

impl Task {
    pub fn load(dir: &Path) -> Result<Self> {
        let dir = dir
            .canonicalize()
            .with_context(|| format!("task directory {} does not exist", dir.display()))?;
        let manifest_path = dir.join("task.toml");
        let manifest: TaskManifest = if manifest_path.is_file() {
            let raw = std::fs::read_to_string(&manifest_path)?;
            toml::from_str(&raw).with_context(|| format!("parsing {}", manifest_path.display()))?
        } else {
            TaskManifest::default()
        };

        let agsci_program = find_program_zero(&dir);
        let init_program = match &manifest.init_program {
            Some(rel) => {
                let path = dir.join(rel);
                if !path.is_file() {
                    bail!("task.toml names init_program `{rel}`, which does not exist");
                }
                path
            }
            None => find_initial(&dir)
                .or_else(|| agsci_program.clone())
                .with_context(|| {
                    format!(
                    "no seed program in {}: expected initial.<ext>, program_0/src/program.<ext>, \
                     or `init_program` in task.toml",
                    dir.display()
                )
                })?,
        };

        let language = match &manifest.language {
            Some(language) => language.trim().to_ascii_lowercase(),
            None => {
                let ext = init_program
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                language_for_extension(ext)
                    .with_context(|| {
                        format!(
                            "cannot infer a language from `{}`; set `language` in task.toml",
                            init_program.display()
                        )
                    })?
                    .to_string()
            }
        };

        let command = match &manifest.eval_command {
            Some(command) if !command.is_empty() => command.clone(),
            _ if dir.join("evaluate.py").is_file() => vec![
                "{python}".into(),
                "{task_dir}/evaluate.py".into(),
                "--program_path".into(),
                "{program_path}".into(),
                "--results_dir".into(),
                "{results_dir}".into(),
            ],
            _ if dir.join("evaluate.sh").is_file() => vec![
                "bash".into(),
                "{task_dir}/evaluate.sh".into(),
                "{program_path}".into(),
                "{results_dir}".into(),
            ],
            _ => bail!(
                "no evaluator in {}: expected evaluate.py, evaluate.sh, or `eval_command` in task.toml",
                dir.display()
            ),
        };

        let goal = match &manifest.system_message {
            Some(message) => message.trim().to_string(),
            None => std::fs::read_to_string(dir.join("task.md"))
                .map(|text| text.trim().to_string())
                .unwrap_or_default(),
        };
        let name = manifest.name.clone().unwrap_or_else(|| {
            dir.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "task".to_string())
        });

        let eval = EvalSpec {
            command,
            timeout_secs: Some(manifest.eval_timeout_secs.unwrap_or(600)),
            cwd: None,
            env: manifest.eval_env.clone(),
            ..EvalSpec::default()
        };
        let final_command = match &manifest.final_eval_command {
            Some(command) if !command.is_empty() => Some(command.clone()),
            _ if dir.join("evaluate_final.py").is_file() => Some(vec![
                "{python}".into(),
                "{task_dir}/evaluate_final.py".into(),
                "--program_path".into(),
                "{program_path}".into(),
                "--results_dir".into(),
                "{results_dir}".into(),
            ]),
            _ if dir.join("evaluate_final.sh").is_file() => Some(vec![
                "bash".into(),
                "{task_dir}/evaluate_final.sh".into(),
                "{program_path}".into(),
                "{results_dir}".into(),
            ]),
            _ => None,
        };
        let final_eval = final_command.map(|command| EvalSpec {
            command,
            ..eval.clone()
        });
        Ok(Self {
            name,
            dir,
            goal,
            language,
            init_program,
            eval,
            final_eval,
            manifest,
        })
    }

    pub fn init_code(&self) -> Result<String> {
        std::fs::read_to_string(&self.init_program)
            .with_context(|| format!("reading seed program {}", self.init_program.display()))
    }

    /// File name a candidate must have so the evaluator can find it.
    pub fn program_file_name(&self) -> String {
        format!("main.{}", extension_for_language(&self.language))
    }

    pub fn ignore_patterns(&self) -> Vec<String> {
        let mut patterns: Vec<String> = DEFAULT_IGNORE.iter().map(|p| p.to_string()).collect();
        patterns.extend(self.manifest.ignore.iter().cloned());
        for name in [".genesis_task_ignore", ".agentic_task_ignore"] {
            if let Ok(raw) = std::fs::read_to_string(self.dir.join(name)) {
                patterns.extend(
                    raw.lines()
                        .map(str::trim)
                        .filter(|line| !line.is_empty() && !line.starts_with('#'))
                        .map(str::to_string),
                );
            }
        }
        patterns
    }

    /// Digest of the task as the evaluator sees it. Recorded at run start and
    /// re-checked before evidence is trusted, so an edited evaluator cannot
    /// pass unnoticed.
    pub fn fingerprint(&self) -> Result<String> {
        fsutil::sha256_tree(&self.dir, &self.ignore_patterns())
    }

    /// Copy the task into `dest` and load it from there. Runs work from this
    /// immutable snapshot, never from the source directory.
    pub fn snapshot(&self, dest: &Path) -> Result<Self> {
        if !dest.join(".snapshot_complete").is_file() {
            if dest.exists() {
                std::fs::remove_dir_all(dest)
                    .with_context(|| format!("clearing partial snapshot {}", dest.display()))?;
            }
            fsutil::copy_tree(&self.dir, dest, &self.ignore_patterns())?;
            fsutil::write_text(&dest.join(".snapshot_complete"), &self.fingerprint()?)?;
        }
        let mut snapshot = Self::load(dest)?;
        snapshot.name = self.name.clone();
        Ok(snapshot)
    }

    /// Evaluate one program file with this task's evaluator.
    pub async fn evaluate(
        &self,
        program_path: &Path,
        results_dir: &Path,
        seed: Option<u64>,
        timeout: Option<Duration>,
    ) -> Result<EvalOutcome> {
        let request = EvalRequest {
            task_dir: self.dir.clone(),
            program_path: absolute(program_path)?,
            results_dir: absolute(results_dir)?,
            seed,
            timeout,
        };
        run_evaluation(&self.eval, &request).await
    }

    /// Evaluate one program with the task's final evaluator. Errors when the
    /// task has none.
    pub async fn evaluate_final(
        &self,
        program_path: &Path,
        results_dir: &Path,
        seed: Option<u64>,
        timeout: Option<Duration>,
    ) -> Result<EvalOutcome> {
        let spec = self
            .final_eval
            .as_ref()
            .context("this task has no final evaluator")?;
        let request = EvalRequest {
            task_dir: self.dir.clone(),
            program_path: absolute(program_path)?,
            results_dir: absolute(results_dir)?,
            seed,
            timeout,
        };
        run_evaluation(spec, &request).await
    }
}

fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        let path = dir.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn infers_the_genesis_layout() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "task.md",
            "# Goal\nMaximise the sum of radii.\n",
        );
        write(dir.path(), "initial.rs", "fn main() {}\n");
        write(dir.path(), "evaluate.py", "");
        let task = Task::load(dir.path()).unwrap();
        assert_eq!(task.language, "rust");
        assert_eq!(task.program_file_name(), "main.rs");
        assert!(task.goal.starts_with("# Goal"));
        assert_eq!(task.eval.command[2], "--program_path");
        assert_eq!(task.eval.timeout_secs, Some(600));
    }

    #[test]
    fn infers_the_agentic_scientist_layout() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "program_0/program_spec.toml",
            "schema_version = 1\ncode_root = \"src\"\n[entrypoint]\npath = \"solver.py\"\nlanguage = \"python\"\ntype = \"script\"\n",
        );
        write(dir.path(), "program_0/src/solver.py", "x = 1\n");
        write(dir.path(), "evaluate.py", "");
        let task = Task::load(dir.path()).unwrap();
        assert!(task.init_program.ends_with("program_0/src/solver.py"));
        assert_eq!(task.language, "python");
        assert_eq!(task.eval.command[2], "--program_path");
    }

    #[test]
    fn manifest_overrides_inference_and_rejects_unknown_fields() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "seed/start.go", "package main\n");
        write(
            dir.path(),
            "task.toml",
            "name = \"collatz\"\ninit_program = \"seed/start.go\"\neval_command = [\"./score\", \"{program_path}\"]\neval_timeout_secs = 30\nsystem_message = \"Go faster.\"\n",
        );
        let task = Task::load(dir.path()).unwrap();
        assert_eq!(
            (task.name.as_str(), task.language.as_str()),
            ("collatz", "go")
        );
        assert_eq!(task.goal, "Go faster.");
        assert_eq!(task.eval.timeout_secs, Some(30));

        write(dir.path(), "task.toml", "init_programme = \"x\"\n");
        assert!(Task::load(dir.path()).is_err());
    }

    #[test]
    fn missing_pieces_are_explained() {
        let dir = tempfile::tempdir().unwrap();
        let err = Task::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("no seed program"), "{err}");
        write(dir.path(), "initial.py", "");
        let err = Task::load(dir.path()).unwrap_err().to_string();
        assert!(err.contains("no evaluator"), "{err}");
    }

    #[tokio::test]
    async fn snapshot_is_isolated_and_evaluates() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("task");
        write(&source, "initial.py", "print(1)\n");
        write(&source, "results/old.json", "{}");
        write(&source, ".agentic_task_ignore", "notes\n");
        write(&source, "notes/private.md", "secret");
        write(
            &source,
            "evaluate.sh",
            "echo '{\"combined_score\": 1.5}' > \"$2/metrics.json\"\necho '{\"correct\": true}' > \"$2/correct.json\"\n",
        );
        let task = Task::load(&source).unwrap();
        let before = task.fingerprint().unwrap();
        let snapshot = task.snapshot(&dir.path().join("run/task")).unwrap();
        assert!(!snapshot.dir.join("results").exists());
        assert!(!snapshot.dir.join("notes").exists());
        assert_eq!(snapshot.name, "task");
        assert_eq!(
            std::fs::read_to_string(snapshot.dir.join(".snapshot_complete")).unwrap(),
            before
        );

        let outcome = snapshot
            .evaluate(
                &snapshot.init_program,
                &dir.path().join("run/eval"),
                None,
                None,
            )
            .await
            .unwrap();
        assert!(outcome.correct, "{outcome:?}");
        assert_eq!(outcome.combined_score, 1.5);

        assert!(snapshot.final_eval.is_none());
        assert!(snapshot
            .evaluate_final(
                &snapshot.init_program,
                &dir.path().join("run/final"),
                None,
                None
            )
            .await
            .is_err());
        write(
            &source,
            "evaluate_final.sh",
            "echo \"{\\\"combined_score\\\": $GENESIS_EVAL_SEED}\" > \"$2/metrics.json\"\necho '{\"correct\": true}' > \"$2/correct.json\"\n",
        );
        let with_final = Task::load(&source).unwrap();
        let held_out = with_final
            .evaluate_final(
                &with_final.init_program,
                &dir.path().join("run/final"),
                Some(41),
                None,
            )
            .await
            .unwrap();
        assert_eq!((held_out.correct, held_out.combined_score), (true, 41.0));

        write(&source, "evaluate.sh", "exit 1\n");
        assert_ne!(Task::load(&source).unwrap().fingerprint().unwrap(), before);
    }
}
