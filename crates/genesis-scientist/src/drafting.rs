//! Task drafting: turn a bare research brief (`task.md`) into a runnable
//! task by writing a seed program and an evaluator, then proving with
//! deterministic gates that the pair is usable before any search starts.
//!
//! A model writes the scaffold; the harness decides whether to accept it.
//! Each gate that fails is reported back to the model for a repair, a
//! bounded number of times. Nothing is published into the run unless every
//! gate passes. Modelled on the Agentic Scientist's task drafting (Sakana
//! AI, Apache-2.0), without its container and contract machinery.

use anyhow::{bail, Context, Result};
use genesis_core::{fsutil, EventLog, Task};
use genesis_llm::{LlmClient, LlmRequest, Message, SamplingKwargs};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DraftingConfig {
    pub models: Vec<String>,
    pub kwargs: SamplingKwargs,
    /// Repair rounds after the first draft.
    pub max_repairs: u32,
    /// Language of the seed program the model is asked to write.
    pub language: String,
    /// Time limit of each gate evaluation.
    pub smoke_timeout_secs: u64,
}

impl Default for DraftingConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            kwargs: SamplingKwargs::default(),
            max_repairs: 4,
            language: "python".to_string(),
            smoke_timeout_secs: 300,
        }
    }
}

/// One gate's verdict, as written to `drafting_report.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GateResult {
    pub gate: String,
    pub passed: bool,
    pub diagnostic: String,
}

/// True when the directory already holds a seed program and an evaluator.
pub fn scaffold_is_complete(task_dir: &Path) -> bool {
    Task::load(task_dir).is_ok()
}

const SYSTEM: &str = "You design the evaluation scaffold for an automated program-optimization campaign. From a task \
brief you write two files: a seed program that a search will improve, and an evaluator that scores any candidate \
program. The harness will run your files through strict gates before using them.

# Seed program
- A complete, runnable, deliberately simple and correct baseline that leaves clear room for improvement.
- The algorithm to be improved sits between a line containing `EVOLVE-BLOCK-START` and a line containing \
`EVOLVE-BLOCK-END`, each written as a comment in the program's language. Everything the evaluator relies on (entry \
point, input and output format) sits outside those markers and will never be changed.

# Evaluator
- Invoked as `<evaluator> --program_path <file> --results_dir <dir>` for a Python evaluator, or \
`<evaluator> <file> <dir>` for a shell evaluator.
- Writes `<dir>/metrics.json`: {\"combined_score\": <finite number, higher is better>, \"public\": {...}, \
\"private\": {...}, \"text_feedback\": \"...\"} and then `<dir>/correct.json`: {\"correct\": <bool>, \"error\": \
<string or null>}. Write correct.json last.
- Runs the candidate in a subprocess with a timeout, and computes every score itself from the candidate's output. \
Never trust a score the candidate reports about itself. Validate the output: a candidate that violates the task's \
constraints is `correct: false`.
- A crashing, hanging or invalid candidate yields `correct: false` with a useful `error`, and the evaluator still \
exits with status 0.
- Deterministic: the same program gets the same score. If the task needs randomness, seed it from the environment \
variable GENESIS_EVAL_SEED (default 0).
- Standard library only, no network, and it writes nothing outside the results directory.

# Output
Respond with exactly one JSON object and nothing else:
{\"language\": \"python\", \"seed_filename\": \"initial.py\", \"seed_program\": \"...\", \
\"evaluator_filename\": \"evaluate.py\" or \"evaluate.sh\", \"evaluator\": \"...\", \"notes\": \"what is measured and why\"}";

fn gate(name: &str, passed: bool, diagnostic: impl Into<String>) -> GateResult {
    GateResult {
        gate: name.to_string(),
        passed,
        diagnostic: diagnostic.into(),
    }
}

fn safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains(['/', '\\'])
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Write the model's files into `staging` (a copy of the task source) and
/// run every gate. An empty list of failures means the draft is usable.
async fn check_draft(
    source: &Path,
    staging: &Path,
    draft: &Value,
    config: &DraftingConfig,
) -> Result<Vec<GateResult>> {
    let mut gates = Vec::new();
    let text = |key: &str| {
        draft
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let (seed_name, evaluator_name) = (text("seed_filename"), text("evaluator_filename"));
    let (seed_program, evaluator) = (text("seed_program"), text("evaluator"));
    let names_ok = safe_file_name(&seed_name)
        && seed_name.starts_with("initial.")
        && matches!(evaluator_name.as_str(), "evaluate.py" | "evaluate.sh");
    gates.push(gate(
        "files_named",
        names_ok && !seed_program.trim().is_empty() && !evaluator.trim().is_empty(),
        "the reply must give a non-empty `seed_program` as `initial.<ext>` and a non-empty `evaluator` as `evaluate.py` or `evaluate.sh`",
    ));
    if !gates[0].passed {
        return Ok(gates);
    }

    if staging.exists() {
        std::fs::remove_dir_all(staging)?;
    }
    fsutil::copy_tree(
        source,
        staging,
        &["results".to_string(), ".git".to_string()],
    )?;
    for stale in ["evaluate.py", "evaluate.sh"] {
        let _ = std::fs::remove_file(staging.join(stale));
    }
    fsutil::write_text(&staging.join(&seed_name), &seed_program)?;
    fsutil::write_text(&staging.join(&evaluator_name), &evaluator)?;

    let brief_intact =
        std::fs::read(source.join("task.md")).ok() == std::fs::read(staging.join("task.md")).ok();
    gates.push(gate(
        "brief_unchanged",
        brief_intact,
        "task.md must stay byte-identical to the brief",
    ));

    let task = match Task::load(staging) {
        Ok(task) => {
            gates.push(gate("task_loads", true, ""));
            task
        }
        Err(error) => {
            gates.push(gate(
                "task_loads",
                false,
                format!("the scaffold does not load as a task: {error:#}"),
            ));
            return Ok(gates);
        }
    };
    let marker_problem = genesis_edit::validate_evolve_markers(&seed_program, &task.language);
    let has_region = !genesis_edit::mutable_ranges(&seed_program).is_empty();
    gates.push(gate(
        "evolve_block",
        marker_problem.is_none() && has_region,
        marker_problem.unwrap_or_else(|| {
            "the seed program needs one EVOLVE-BLOCK-START / EVOLVE-BLOCK-END region".to_string()
        }),
    ));

    let fingerprint_before = task.fingerprint()?;
    let timeout = Some(Duration::from_secs(config.smoke_timeout_secs));
    let scratch = staging.parent().unwrap_or(staging).join("gate_results");
    let _ = std::fs::remove_dir_all(&scratch);
    let first = task
        .evaluate(
            &task.init_program,
            &scratch.join("baseline_1"),
            Some(0),
            timeout,
        )
        .await?;
    let second = task
        .evaluate(
            &task.init_program,
            &scratch.join("baseline_2"),
            Some(0),
            timeout,
        )
        .await?;
    let describe = |outcome: &genesis_core::EvalOutcome| {
        format!(
            "correct={}, combined_score={}, feedback: {}\nstderr: {}",
            outcome.correct,
            outcome.combined_score,
            fsutil::truncate(&outcome.text_feedback, 600),
            fsutil::tail(&outcome.stderr_tail, 1200)
        )
    };
    gates.push(gate(
        "baseline_evaluates",
        !first.evaluation_failed && !second.evaluation_failed,
        format!("the evaluator did not produce valid metrics.json and correct.json for the seed program: {}", describe(&first)),
    ));
    gates.push(gate(
        "baseline_correct",
        first.correct && second.correct,
        format!("the seed program must be a correct baseline, but the evaluator judged it incorrect: {}", describe(&first)),
    ));
    let tolerance = 1e-9 * (1.0 + first.combined_score.abs());
    gates.push(gate(
        "deterministic",
        (first.combined_score - second.combined_score).abs() <= tolerance,
        format!(
            "two evaluations of the same program with the same seed scored {} and {}",
            first.combined_score, second.combined_score
        ),
    ));

    // An evaluator that scores a gutted program as well as the baseline is
    // not measuring the program.
    if has_region {
        let (start, end) = genesis_edit::mutable_ranges(&seed_program)[0];
        let gutted = format!("{}\n{}", &seed_program[..start], &seed_program[end..]);
        let gutted_path = scratch.join("gutted").join(task.program_file_name());
        fsutil::write_text(&gutted_path, &gutted)?;
        let outcome = task
            .evaluate(
                &gutted_path,
                &scratch.join("gutted_results"),
                Some(0),
                timeout,
            )
            .await?;
        let fooled =
            outcome.correct && outcome.combined_score >= first.combined_score && first.correct;
        gates.push(gate(
            "exercises_candidate",
            !fooled,
            "a program with its EVOLVE-BLOCK emptied scored at least as well as the baseline and was judged correct; \
             the evaluator must run the candidate and score what it actually produces",
        ));
    }
    gates.push(gate(
        "evaluator_leaves_task_alone",
        Task::load(staging)?.fingerprint()? == fingerprint_before,
        "evaluating a program changed files in the task directory; write only inside the results directory",
    ));
    Ok(gates)
}

/// Draft the scaffold for `source` and publish the result as the run's task
/// snapshot. Fails, leaving no snapshot, when no draft passes every gate.
pub async fn draft_task(
    source: &Path,
    snapshot: &Path,
    log_dir: &Path,
    llm: &LlmClient,
    config: &DraftingConfig,
    events: &Arc<EventLog>,
) -> Result<()> {
    let brief = std::fs::read_to_string(source.join("task.md"))
        .with_context(|| format!("{} has no task.md to draft from", source.display()))?;
    if brief.trim().is_empty() {
        bail!("{} is empty", source.join("task.md").display());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(0xD4AF7);
    let Some(model) = config.models.choose(&mut rng).cloned() else {
        bail!("no model is configured for task drafting");
    };
    let existing: Vec<String> = std::fs::read_dir(source)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name != "task.md")
        .collect();
    let user = format!(
        "# Task brief (task.md)\n{}\n\n# Requested language for the seed program\n{}\n\n# Other files already in the task directory\n{}\n\n\
         Write the scaffold as the JSON object described in the system message.",
        brief.trim(),
        config.language,
        if existing.is_empty() { "(none)".to_string() } else { existing.join(", ") }
    );
    let mut request = LlmRequest::new(model, SYSTEM, user);
    config.kwargs.apply(&mut request, &mut rng);

    std::fs::create_dir_all(log_dir)?;
    let staging = log_dir.join("staging");
    let mut rounds = Vec::new();
    for round in 0..=config.max_repairs {
        let (draft, response) = llm
            .query_json(
                if round == 0 {
                    "task_draft"
                } else {
                    "task_repair"
                },
                request.clone(),
            )
            .await?;
        let _ = fsutil::write_json(&log_dir.join(format!("draft_{round:02}.json")), &draft);
        let gates = check_draft(source, &staging, &draft, config).await?;
        let failures: Vec<&GateResult> = gates.iter().filter(|gate| !gate.passed).collect();
        events.info(
            "task_draft_checked",
            json!({
                "round": round,
                "passed": gates.iter().filter(|gate| gate.passed).count(),
                "failed": failures.iter().map(|gate| gate.gate.as_str()).collect::<Vec<_>>(),
            }),
        );
        rounds.push(json!({"round": round, "gates": gates, "cost_usd": response.cost_usd}));
        if failures.is_empty() {
            Task::load(&staging)?.snapshot(snapshot)?;
            fsutil::write_json(
                &log_dir.join("drafting_report.json"),
                &json!({"status": "published", "rounds": rounds, "notes": draft.get("notes")}),
            )?;
            return Ok(());
        }
        let report: Vec<String> = failures
            .iter()
            .map(|failure| format!("- gate `{}` failed: {}", failure.gate, failure.diagnostic))
            .collect();
        request.messages.push(Message::assistant(response.content));
        request.messages.push(Message::user(format!(
            "The harness rejected that scaffold:\n{}\n\nFix the root cause and reply with the complete corrected JSON object.",
            report.join("\n")
        )));
    }
    fsutil::write_json(
        &log_dir.join("drafting_report.json"),
        &json!({"status": "failed", "rounds": rounds}),
    )?;
    bail!(
        "no draft passed every gate within {} repair rounds; see {}",
        config.max_repairs,
        log_dir.join("drafting_report.json").display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::ConsoleMode;
    use genesis_llm::MockProvider;

    const SEED: &str = "# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n";
    const GOOD_EVALUATOR: &str = "x=$(sed -n 's/^x = \\([0-9.]*\\)$/\\1/p' \"$1\" | head -n 1)\n\
        if [ -z \"$x\" ]; then echo '{\"combined_score\": 0}' > \"$2/metrics.json\"; echo '{\"correct\": false, \"error\": \"no x\"}' > \"$2/correct.json\"; exit 0; fi\n\
        echo \"{\\\"combined_score\\\": $x}\" > \"$2/metrics.json\"\necho '{\"correct\": true}' > \"$2/correct.json\"\n";
    /// Ignores the program entirely.
    const CONSTANT_EVALUATOR: &str = "echo '{\"combined_score\": 7}' > \"$2/metrics.json\"\necho '{\"correct\": true}' > \"$2/correct.json\"\n";

    fn draft(evaluator: &str) -> String {
        json!({"language": "python", "seed_filename": "initial.py", "seed_program": SEED,
               "evaluator_filename": "evaluate.sh", "evaluator": evaluator, "notes": "x is the score"})
        .to_string()
    }

    fn brief(dir: &Path) -> std::path::PathBuf {
        let source = dir.join("brief");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("task.md"), "# Goal\nMake x large.\n").unwrap();
        source
    }

    fn config() -> DraftingConfig {
        DraftingConfig {
            models: vec!["drafter".into()],
            max_repairs: 2,
            smoke_timeout_secs: 30,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn a_rejected_draft_is_repaired_and_then_published() {
        let dir = tempfile::tempdir().unwrap();
        let source = brief(dir.path());
        assert!(!scaffold_is_complete(&source));
        let provider =
            MockProvider::scripted(vec![draft(CONSTANT_EVALUATOR), draft(GOOD_EVALUATOR)]);
        let llm = LlmClient::mock(provider.clone());
        let snapshot = dir.path().join("run/task");
        let log_dir = dir.path().join("run/task_drafting");
        draft_task(
            &source,
            &snapshot,
            &log_dir,
            &llm,
            &config(),
            &EventLog::console_only(ConsoleMode::Quiet),
        )
        .await
        .unwrap();

        assert!(scaffold_is_complete(&snapshot));
        let task = Task::load(&snapshot).unwrap();
        assert_eq!(
            (task.language.as_str(), task.goal.as_str()),
            ("python", "# Goal\nMake x large.")
        );
        assert!(
            !source.join("initial.py").exists(),
            "the brief directory is never written to"
        );

        let calls = provider.calls();
        assert_eq!(
            (calls[0].stage.as_str(), calls[1].stage.as_str()),
            ("task_draft", "task_repair")
        );
        assert!(calls[1]
            .last_user()
            .contains("gate `exercises_candidate` failed"));
        let report: Value = fsutil::read_json(&log_dir.join("drafting_report.json")).unwrap();
        assert_eq!(report["status"], "published");
        assert_eq!(report["rounds"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn drafts_that_never_pass_publish_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let source = brief(dir.path());
        let incorrect_baseline = "echo '{\"combined_score\": 1}' > \"$2/metrics.json\"\necho '{\"correct\": false, \"error\": \"always wrong\"}' > \"$2/correct.json\"\n";
        let llm = LlmClient::mock(MockProvider::scripted(vec![draft(incorrect_baseline)]));
        let snapshot = dir.path().join("run/task");
        let log_dir = dir.path().join("run/task_drafting");
        let error = draft_task(
            &source,
            &snapshot,
            &log_dir,
            &llm,
            &config(),
            &EventLog::console_only(ConsoleMode::Quiet),
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("no draft passed every gate"),
            "{error}"
        );
        assert!(!snapshot.exists());
        let report: Value = fsutil::read_json(&log_dir.join("drafting_report.json")).unwrap();
        assert_eq!(report["status"], "failed");
        assert_eq!(report["rounds"].as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn each_gate_names_its_own_failure() {
        let dir = tempfile::tempdir().unwrap();
        let source = brief(dir.path());
        let staging = dir.path().join("staging");
        let failed = |gates: Vec<GateResult>| -> Vec<String> {
            gates
                .into_iter()
                .filter(|gate| !gate.passed)
                .map(|gate| gate.gate)
                .collect()
        };
        let check = |value: Value| {
            let (source, staging) = (source.clone(), staging.clone());
            async move {
                failed(
                    check_draft(&source, &staging, &value, &config())
                        .await
                        .unwrap(),
                )
            }
        };
        let base: Value = serde_json::from_str(&draft(GOOD_EVALUATOR)).unwrap();
        assert!(check(base.clone()).await.is_empty());

        let mut bad_name = base.clone();
        bad_name["seed_filename"] = json!("../escape.py");
        assert_eq!(check(bad_name).await, ["files_named"]);

        let mut no_markers = base.clone();
        no_markers["seed_program"] = json!("x = 1\n");
        assert!(check(no_markers)
            .await
            .contains(&"evolve_block".to_string()));

        let mut nondeterministic = base.clone();
        nondeterministic["evaluator"] = json!("echo \"{\\\"combined_score\\\": $RANDOM}\" > \"$2/metrics.json\"\necho '{\"correct\": true}' > \"$2/correct.json\"\n");
        assert!(check(nondeterministic)
            .await
            .contains(&"deterministic".to_string()));

        let mut writes_into_task = base.clone();
        writes_into_task["evaluator"] = json!(format!(
            "touch \"$(dirname \"$0\")/leftover\"\n{GOOD_EVALUATOR}"
        ));
        assert_eq!(
            check(writes_into_task).await,
            ["evaluator_leaves_task_alone"]
        );

        let mut broken = base;
        broken["evaluator"] = json!("exit 3\n");
        let failures = check(broken).await;
        assert!(
            failures.contains(&"baseline_evaluates".to_string())
                && failures.contains(&"baseline_correct".to_string())
        );
    }
}
