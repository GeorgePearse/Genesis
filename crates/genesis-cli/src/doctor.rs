//! `genesis doctor`: check that a run can start before spending anything.

use crate::pipeline::PipelineConfig;
use genesis_core::Task;
use genesis_evolve::EvolutionConfig;
use genesis_llm::{agent_available, missing_credentials, AgentSpec};
use serde::Serialize;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Check {
    pub name: String,
    /// `ok`, `warn` or `fail`.
    pub status: String,
    pub detail: String,
}

fn check(name: &str, status: &str, detail: impl Into<String>) -> Check {
    Check {
        name: name.to_string(),
        status: status.to_string(),
        detail: detail.into(),
    }
}

fn on_path(program: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(program).is_file()))
}

/// Run every check that applies to `config`. With `evaluate_seed`, the
/// task's evaluator is run once on its seed program.
pub async fn run_checks(config: &PipelineConfig, evaluate_seed: bool) -> Vec<Check> {
    let mut checks = Vec::new();

    let task_dir = Path::new(&config.task.dir);
    let task = match Task::load(task_dir) {
        Ok(task) => {
            checks.push(check(
                "task",
                "ok",
                format!(
                    "{} ({}, seed {})",
                    task.name,
                    task.language,
                    task.init_program.display()
                ),
            ));
            Some(task)
        }
        Err(error) if config.selected("task_drafting") && task_dir.join("task.md").is_file() => {
            checks.push(check(
                "task",
                "warn",
                format!("no scaffold yet; task drafting will write one ({error})"),
            ));
            None
        }
        Err(error) => {
            checks.push(check("task", "fail", format!("{error:#}")));
            None
        }
    };
    if let Some(task) = &task {
        if task.goal.is_empty() {
            checks.push(check(
                "task goal",
                "warn",
                "task.md is missing or empty; models get no task description",
            ));
        }
        let evaluator = &task.eval.command[0];
        let program = if evaluator == "{python}" {
            genesis_core::eval::python_executable()
        } else {
            evaluator.clone()
        };
        if on_path(&program) || Path::new(&program).is_file() {
            checks.push(check("evaluator command", "ok", program));
        } else {
            checks.push(check(
                "evaluator command",
                "fail",
                format!("`{program}` is not on PATH"),
            ));
        }
        if evaluate_seed {
            let scratch =
                std::env::temp_dir().join(format!("genesis-doctor-{}", std::process::id()));
            match task
                .evaluate(
                    &task.init_program,
                    &scratch,
                    Some(0),
                    Some(Duration::from_secs(600)),
                )
                .await
            {
                Ok(outcome) if outcome.evaluation_failed => checks.push(check(
                    "seed evaluation",
                    "fail",
                    outcome
                        .diagnostic
                        .unwrap_or_else(|| "the evaluator produced no valid result".to_string()),
                )),
                Ok(outcome) if !outcome.correct => checks.push(check(
                    "seed evaluation",
                    "warn",
                    format!(
                        "the seed is incorrect ({}); the search starts in fix mode",
                        outcome.text_feedback
                    ),
                )),
                Ok(outcome) => checks.push(check(
                    "seed evaluation",
                    "ok",
                    format!(
                        "combined_score {} in {:.1}s",
                        outcome.combined_score, outcome.runtime_secs
                    ),
                )),
                Err(error) => checks.push(check("seed evaluation", "fail", format!("{error:#}"))),
            }
            let _ = std::fs::remove_dir_all(&scratch);
        }
    }

    let missing = missing_credentials(&config.models_in_use());
    if missing.is_empty() {
        checks.push(check(
            "model credentials",
            "ok",
            format!("{} model(s) usable", config.models_in_use().len()),
        ));
    }
    for problem in missing {
        checks.push(check("model credentials", "fail", problem));
    }

    let mut agents: Vec<String> = config
        .investigation
        .evolution
        .llm_models
        .iter()
        .filter(|arm| EvolutionConfig::is_agent_arm(arm))
        .cloned()
        .collect();
    if let Some(agent) = &config.agent {
        agents.push(format!("agent:{}", agent.trim_start_matches("agent:")));
    }
    for arm in agents {
        match AgentSpec::parse(&arm) {
            Some(spec)
                if config.investigation.agents.contains_key(&spec.name)
                    || agent_available(&spec) =>
            {
                checks.push(check("agent", "ok", arm));
            }
            Some(spec) => checks.push(check(
                "agent",
                "fail",
                format!("the `{}` CLI is not installed", spec.name),
            )),
            None => checks.push(check(
                "agent",
                "fail",
                format!("`{arm}` is not an agent spec"),
            )),
        }
    }

    if config.selected("paper") && config.paper.compile_pdf {
        if on_path("pdflatex") && on_path("bibtex") {
            checks.push(check("tex toolchain", "ok", "pdflatex and bibtex found"));
        } else {
            checks.push(check(
                "tex toolchain",
                "warn",
                "pdflatex or bibtex is missing; the paper will be written as paper.tex and paper.md without a PDF",
            ));
        }
    }

    let results_root = Path::new(&config.paths.results_root);
    match std::fs::create_dir_all(results_root).and_then(|_| tempfile_probe(results_root)) {
        Ok(()) => checks.push(check(
            "results root",
            "ok",
            results_root.display().to_string(),
        )),
        Err(error) => checks.push(check(
            "results root",
            "fail",
            format!("{} is not writable: {error}", results_root.display()),
        )),
    }
    checks
}

fn tempfile_probe(dir: &Path) -> std::io::Result<()> {
    let probe = dir.join(format!(".genesis-doctor-{}", std::process::id()));
    std::fs::write(&probe, b"")?;
    std::fs::remove_file(probe)
}

pub fn render(checks: &[Check]) -> String {
    let width = checks
        .iter()
        .map(|check| check.name.len())
        .max()
        .unwrap_or(0);
    checks
        .iter()
        .map(|check| {
            let mark = match check.status.as_str() {
                "ok" => "ok  ",
                "warn" => "warn",
                _ => "FAIL",
            };
            format!("[{mark}] {:<width$}  {}", check.name, check.detail)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn checks_cover_the_task_the_models_and_the_evaluation() {
        let dir = tempfile::tempdir().unwrap();
        let task = dir.path().join("task");
        std::fs::create_dir_all(&task).unwrap();
        std::fs::write(task.join("task.md"), "Make x large.\n").unwrap();
        std::fs::write(task.join("initial.py"), "x = 1\n").unwrap();
        std::fs::write(
            task.join("evaluate.sh"),
            "echo '{\"combined_score\": 1.5}' > \"$2/metrics.json\"\necho '{\"correct\": true}' > \"$2/correct.json\"\n",
        )
        .unwrap();
        let overrides = vec![
            format!("task.dir={}", task.display()),
            format!(
                "paths.results_root={}",
                dir.path().join("results").display()
            ),
            "models=[mock/perturb, agent:definitely-not-installed]".to_string(),
        ];
        let config = PipelineConfig::load(None, &overrides).unwrap();
        let checks = run_checks(&config, true).await;
        let status = |name: &str| {
            checks
                .iter()
                .find(|check| check.name == name)
                .map(|check| check.status.as_str())
        };
        assert_eq!(status("task"), Some("ok"));
        assert_eq!(status("seed evaluation"), Some("ok"));
        assert_eq!(status("model credentials"), Some("ok"));
        assert_eq!(status("agent"), Some("fail"));
        assert_eq!(status("results root"), Some("ok"));
        let text = render(&checks);
        assert!(
            text.contains("[FAIL] agent") && text.contains("combined_score 1.5"),
            "{text}"
        );

        let missing = PipelineConfig::load(
            None,
            &[format!("task.dir={}", dir.path().join("nowhere").display())],
        )
        .unwrap();
        let checks = run_checks(&missing, false).await;
        assert_eq!(checks[0].status, "fail");
    }
}
