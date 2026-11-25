//! The evaluator contract and its local launcher.
//!
//! An evaluator is any command that scores one program and writes two files
//! into the results directory it is given:
//!
//! * `correct.json`: `{"correct": bool, "error": string | null}`. Written
//!   last; it is the completion marker.
//! * `metrics.json`: `{"combined_score": number, "public": {...},
//!   "private": {...}, "text_feedback": string}`.
//!
//! The harness, never the proposing model or agent, runs this command, and it
//! distrusts the output: a claimed success with a non-zero exit status, a
//! non-finite score or malformed metrics is recorded as a failed evaluation.

use crate::fsutil;
use crate::program::JsonMap;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

/// How to launch the evaluator for a task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EvalSpec {
    /// Argument vector with placeholders: `{python}`, `{task_dir}`,
    /// `{program_path}`, `{program_dir}`, `{results_dir}`, `{seed}`.
    pub command: Vec<String>,
    /// Wall-clock limit; the whole process group is killed when it passes.
    pub timeout_secs: Option<u64>,
    /// Working directory. Defaults to the task directory.
    pub cwd: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    /// Longest stdout/stderr excerpt kept on the outcome, in bytes.
    pub max_log_bytes: usize,
}

impl Default for EvalSpec {
    fn default() -> Self {
        Self {
            command: Vec::new(),
            timeout_secs: Some(600),
            cwd: None,
            env: BTreeMap::new(),
            max_log_bytes: 8_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalRequest {
    pub task_dir: PathBuf,
    pub program_path: PathBuf,
    pub results_dir: PathBuf,
    /// Exported as `GENESIS_EVAL_SEED` and substituted for `{seed}`.
    pub seed: Option<u64>,
    pub timeout: Option<Duration>,
}

/// What one evaluator invocation established about one program.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct EvalOutcome {
    pub correct: bool,
    pub combined_score: f64,
    pub public_metrics: JsonMap,
    pub private_metrics: JsonMap,
    pub text_feedback: String,
    /// `correct.json["error"]` (or `"message"`), when the evaluator gave one.
    pub error: Option<String>,
    /// Set when the harness rejected the evaluation itself (missing files,
    /// malformed output, timeout), as opposed to the program being wrong.
    pub evaluation_failed: bool,
    pub diagnostic: Option<String>,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub runtime_secs: f64,
    pub stdout_tail: String,
    pub stderr_tail: String,
    pub results_dir: PathBuf,
    pub seed: Option<u64>,
}

impl EvalOutcome {
    fn failed(diagnostic: impl Into<String>, feedback: &str) -> Self {
        let diagnostic = diagnostic.into();
        let text_feedback = if feedback.trim().is_empty() {
            diagnostic.clone()
        } else {
            format!("{diagnostic}\n\nEvaluator feedback:\n{}", feedback.trim())
        };
        Self {
            correct: false,
            combined_score: 0.0,
            text_feedback,
            evaluation_failed: true,
            diagnostic: Some(diagnostic),
            ..Self::default()
        }
    }
}

pub fn python_executable() -> String {
    std::env::var("GENESIS_PYTHON")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "python3".to_string())
}

/// Substitute placeholders in the command template.
pub fn render_command(spec: &EvalSpec, request: &EvalRequest) -> Vec<String> {
    let program_dir = request
        .program_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let seed = request.seed.map(|s| s.to_string()).unwrap_or_default();
    let substitutions = [
        ("{python}", python_executable()),
        ("{task_dir}", request.task_dir.display().to_string()),
        ("{program_path}", request.program_path.display().to_string()),
        ("{program_dir}", program_dir.display().to_string()),
        ("{results_dir}", request.results_dir.display().to_string()),
        ("{seed}", seed),
    ];
    spec.command
        .iter()
        .map(|token| {
            substitutions
                .iter()
                .fold(token.clone(), |acc, (key, value)| acc.replace(key, value))
        })
        .collect()
}

#[cfg(unix)]
fn kill_process_group(pid: u32) {
    // SAFETY: killpg only signals; a stale or missing group returns ESRCH.
    unsafe {
        libc::killpg(pid as libc::pid_t, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: u32) {}

/// Kills the evaluator's process group if the owning future is dropped, so a
/// cancelled run does not leave evaluators behind.
struct GroupGuard(Option<u32>);

impl Drop for GroupGuard {
    fn drop(&mut self) {
        if let Some(pid) = self.0.take() {
            kill_process_group(pid);
        }
    }
}

/// Run the evaluator once and read back what it established.
///
/// Never returns `Err` for a bad program or a broken evaluator: those become
/// a failed [`EvalOutcome`]. `Err` means the harness could not even prepare
/// the invocation.
pub async fn run_evaluation(spec: &EvalSpec, request: &EvalRequest) -> Result<EvalOutcome> {
    anyhow::ensure!(
        !spec.command.is_empty(),
        "the task has no evaluator command"
    );
    let results_dir = &request.results_dir;
    std::fs::create_dir_all(results_dir)
        .with_context(|| format!("creating {}", results_dir.display()))?;
    // A stale marker from an earlier run must not be read as this run's.
    for name in ["correct.json", "metrics.json"] {
        let _ = std::fs::remove_file(results_dir.join(name));
    }

    let argv = render_command(spec, request);
    let stdout_path = results_dir.join("job_log.out");
    let stderr_path = results_dir.join("job_log.err");
    let stdout = std::fs::File::create(&stdout_path)?;
    let stderr = std::fs::File::create(&stderr_path)?;

    let mut command = tokio::process::Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .current_dir(spec.cwd.as_deref().unwrap_or(&request.task_dir))
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .kill_on_drop(true);
    for (key, value) in &spec.env {
        command.env(key, value);
    }
    if let Some(seed) = request.seed {
        command.env("GENESIS_EVAL_SEED", seed.to_string());
    }
    #[cfg(unix)]
    command.process_group(0);

    let started = Instant::now();
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            let mut outcome = EvalOutcome::failed(
                format!("could not start the evaluator `{}`: {err}", argv[0]),
                "",
            );
            outcome.results_dir = results_dir.clone();
            outcome.seed = request.seed;
            return Ok(outcome);
        }
    };
    let mut guard = GroupGuard(child.id());

    let limit = request
        .timeout
        .or(spec.timeout_secs.map(Duration::from_secs));
    let mut timed_out = false;
    let status = match limit {
        Some(limit) => match tokio::time::timeout(limit, child.wait()).await {
            Ok(status) => status?,
            Err(_) => {
                timed_out = true;
                if let Some(pid) = guard.0 {
                    kill_process_group(pid);
                }
                child.wait().await?
            }
        },
        None => child.wait().await?,
    };
    // Sweep anything the evaluator left running in its group.
    if let Some(pid) = guard.0.take() {
        kill_process_group(pid);
    }
    let runtime_secs = started.elapsed().as_secs_f64();

    let stdout_tail = read_tail(&stdout_path, spec.max_log_bytes);
    let stderr_tail = read_tail(&stderr_path, spec.max_log_bytes);
    let mut outcome = if timed_out {
        let limit_secs = limit.map(|l| l.as_secs_f64()).unwrap_or_default();
        EvalOutcome::failed(
            format!(
                "evaluation timed out after {}",
                fsutil::human_duration(limit_secs)
            ),
            &stderr_tail,
        )
    } else {
        read_results(results_dir, status.code())
    };
    outcome.exit_code = status.code();
    outcome.timed_out = timed_out;
    outcome.runtime_secs = runtime_secs;
    outcome.stdout_tail = stdout_tail;
    outcome.stderr_tail = stderr_tail;
    outcome.results_dir = results_dir.clone();
    outcome.seed = request.seed;
    Ok(outcome)
}

fn read_tail(path: &Path, max_bytes: usize) -> String {
    let bytes = std::fs::read(path).unwrap_or_default();
    fsutil::tail(&String::from_utf8_lossy(&bytes), max_bytes)
}

fn text_feedback_of(metrics: &Value) -> String {
    match metrics.get("text_feedback") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| item.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn object_of(metrics: &Value, key: &str) -> Result<JsonMap, String> {
    match metrics.get(key) {
        None | Some(Value::Null) => Ok(JsonMap::new()),
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(_) => Err(format!("metrics.json `{key}` is not an object")),
    }
}

/// Validate and read `correct.json` and `metrics.json`.
pub fn read_results(results_dir: &Path, exit_code: Option<i32>) -> EvalOutcome {
    let metrics: Option<Value> = fsutil::read_json_opt(&results_dir.join("metrics.json"));
    let feedback = metrics.as_ref().map(text_feedback_of).unwrap_or_default();

    let correct_path = results_dir.join("correct.json");
    let Some(correct_doc) = fsutil::read_json_opt::<Value>(&correct_path) else {
        let reason = if correct_path.is_file() {
            "is not valid JSON"
        } else {
            "was not written"
        };
        return EvalOutcome::failed(
            format!("the evaluator finished (exit status {exit_code:?}) but correct.json {reason}"),
            &feedback,
        );
    };
    let Some(correct) = correct_doc.get("correct").and_then(Value::as_bool) else {
        return EvalOutcome::failed("correct.json has no boolean `correct` field", &feedback);
    };
    let error = correct_doc
        .get("error")
        .or_else(|| correct_doc.get("message"))
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(str::to_string);

    let empty = Value::Object(JsonMap::new());
    let metrics_ref = metrics.as_ref().unwrap_or(&empty);
    let score = metrics_ref.get("combined_score").and_then(Value::as_f64);
    let public = object_of(metrics_ref, "public");
    let private = object_of(metrics_ref, "private");

    if correct {
        if exit_code != Some(0) {
            return EvalOutcome::failed(
                format!("the evaluator reported success but exited with status {exit_code:?}"),
                &feedback,
            );
        }
        if metrics.is_none() {
            return EvalOutcome::failed(
                "the evaluator reported success without metrics.json",
                &feedback,
            );
        }
        let Some(score) = score.filter(|s| s.is_finite()) else {
            return EvalOutcome::failed(
                "the evaluator reported success without a finite numeric `combined_score`",
                &feedback,
            );
        };
        let (public, private) = match (public, private) {
            (Ok(public), Ok(private)) => (public, private),
            (Err(problem), _) | (_, Err(problem)) => {
                return EvalOutcome::failed(problem, &feedback)
            }
        };
        return EvalOutcome {
            correct: true,
            combined_score: score,
            public_metrics: public,
            private_metrics: private,
            text_feedback: feedback,
            error,
            ..EvalOutcome::default()
        };
    }

    // An incorrect program keeps whatever usable metrics it produced.
    let text_feedback = if feedback.trim().is_empty() {
        error.clone().unwrap_or_default()
    } else {
        feedback
    };
    EvalOutcome {
        correct: false,
        combined_score: score.filter(|s| s.is_finite()).unwrap_or(0.0),
        public_metrics: public.unwrap_or_default(),
        private_metrics: private.unwrap_or_default(),
        text_feedback,
        error,
        ..EvalOutcome::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn spec(script: &str, timeout_secs: u64) -> EvalSpec {
        EvalSpec {
            command: vec!["bash".into(), "-c".into(), script.into()],
            timeout_secs: Some(timeout_secs),
            ..EvalSpec::default()
        }
    }

    fn request(dir: &Path) -> EvalRequest {
        EvalRequest {
            task_dir: dir.to_path_buf(),
            program_path: dir.join("main.py"),
            results_dir: dir.join("results"),
            seed: Some(7),
            timeout: None,
        }
    }

    #[tokio::test]
    async fn reads_a_correct_evaluation_and_exports_the_seed() {
        let dir = tempfile::tempdir().unwrap();
        let script = r#"echo "{\"combined_score\": $GENESIS_EVAL_SEED.5, \"public\": {\"n\": 3}, \"text_feedback\": [\"a\", \"b\"]}" > {results_dir}/metrics.json
echo '{"correct": true, "error": null}' > {results_dir}/correct.json"#;
        let outcome = run_evaluation(&spec(script, 20), &request(dir.path()))
            .await
            .unwrap();
        assert!(outcome.correct, "{outcome:?}");
        assert_eq!(outcome.combined_score, 7.5);
        assert_eq!(outcome.public_metrics["n"], json!(3));
        assert_eq!(outcome.text_feedback, "a\nb");
        assert!(!outcome.evaluation_failed);
        assert!(dir.path().join("results/job_log.out").is_file());
    }

    #[tokio::test]
    async fn success_with_nonzero_exit_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let script = r#"echo '{"combined_score": 9}' > {results_dir}/metrics.json
echo '{"correct": true}' > {results_dir}/correct.json; exit 3"#;
        let outcome = run_evaluation(&spec(script, 20), &request(dir.path()))
            .await
            .unwrap();
        assert!(!outcome.correct);
        assert!(outcome.evaluation_failed);
        assert_eq!(outcome.combined_score, 0.0);
        assert!(outcome.text_feedback.contains("exited with status"));
    }

    #[tokio::test]
    async fn missing_marker_and_non_finite_score_are_failures() {
        let dir = tempfile::tempdir().unwrap();
        let outcome = run_evaluation(&spec("echo oops >&2", 20), &request(dir.path()))
            .await
            .unwrap();
        assert!(outcome.evaluation_failed);
        assert!(outcome.stderr_tail.contains("oops"));

        let script = r#"echo '{"combined_score": "high"}' > {results_dir}/metrics.json
echo '{"correct": true}' > {results_dir}/correct.json"#;
        let outcome = run_evaluation(&spec(script, 20), &request(dir.path()))
            .await
            .unwrap();
        assert!(outcome.evaluation_failed);
    }

    #[tokio::test]
    async fn incorrect_program_keeps_its_error_as_feedback() {
        let dir = tempfile::tempdir().unwrap();
        let script = r#"echo '{"combined_score": 0.25}' > {results_dir}/metrics.json
echo '{"correct": false, "error": "circles overlap"}' > {results_dir}/correct.json"#;
        let outcome = run_evaluation(&spec(script, 20), &request(dir.path()))
            .await
            .unwrap();
        assert!(!outcome.correct);
        assert!(!outcome.evaluation_failed);
        assert_eq!(outcome.combined_score, 0.25);
        assert_eq!(outcome.text_feedback, "circles overlap");
    }

    #[tokio::test]
    async fn timeout_kills_the_whole_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("survivor");
        let script = format!("(sleep 3; touch {}) & sleep 30", marker.display());
        let started = Instant::now();
        let outcome = run_evaluation(&spec(&script, 1), &request(dir.path()))
            .await
            .unwrap();
        assert!(outcome.timed_out);
        assert!(outcome.evaluation_failed);
        assert!(started.elapsed() < Duration::from_secs(10));
        tokio::time::sleep(Duration::from_secs(4)).await;
        assert!(!marker.exists(), "a grandchild outlived the timeout");
    }

    #[tokio::test]
    async fn stale_results_are_not_reused() {
        let dir = tempfile::tempdir().unwrap();
        let results = dir.path().join("results");
        std::fs::create_dir_all(&results).unwrap();
        std::fs::write(results.join("correct.json"), r#"{"correct": true}"#).unwrap();
        std::fs::write(results.join("metrics.json"), r#"{"combined_score": 99}"#).unwrap();
        let outcome = run_evaluation(&spec("true", 20), &request(dir.path()))
            .await
            .unwrap();
        assert!(!outcome.correct);
        assert!(outcome.evaluation_failed);
    }
}
