//! A finished investigation on disk, for the phase tests: a task with a
//! real shell evaluator, a run database, and a frozen selection.

#![allow(dead_code)]

use genesis_core::{fsutil, ConsoleMode, EventLog, Program, Task};
use genesis_evolve::{DatabaseConfig, ProgramDb};
use genesis_llm::{LlmClient, LlmError, LlmRequest, MockProvider};
use genesis_scientist::selection::freeze_selection;
use genesis_scientist::{PhaseContext, RunPaths};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Scores `10 - ((x - 3)^2 + (y + 1)^2) + bonus`. A program that does not
/// assign `x` and `y` is incorrect. Every call is counted in
/// `<run>/eval_count.log`, outside the task directory.
pub const DETERMINISTIC: &str = r#"
program="$1"; results="$2"
echo "$program seed=$GENESIS_EVAL_SEED" >> "$(dirname "$0")/../eval_count.log"
x=$(sed -n 's/^x = \(-\{0,1\}[0-9.]*\)$/\1/p' "$program" | head -n 1)
y=$(sed -n 's/^y = \(-\{0,1\}[0-9.]*\)$/\1/p' "$program" | head -n 1)
bonus=$(sed -n 's/^bonus = \([0-9.]*\)$/\1/p' "$program" | head -n 1)
if [ -z "$x" ] || [ -z "$y" ]; then
  echo '{"combined_score": 0.0}' > "$results/metrics.json"
  echo '{"correct": false, "error": "x or y is not assigned a number"}' > "$results/correct.json"
  echo "missing assignment" >&2
  exit 0
fi
score=$(awk -v x="$x" -v y="$y" -v b="${bonus:-0}" -v n="NOISE" 'BEGIN { printf "%.6f", 10 - ((x - 3) * (x - 3) + (y + 1) * (y + 1)) + b + n }')
echo "{\"combined_score\": $score, \"public\": {\"x\": $x, \"y\": $y, \"label\": \"text\"}}" > "$results/metrics.json"
echo '{"correct": true}' > "$results/correct.json"
"#;

/// The deterministic evaluator with a seed-dependent offset of up to 0.099.
pub fn stochastic() -> String {
    DETERMINISTIC
        .replace(
            "score=$(awk",
            "noise=$(awk -v s=\"${GENESIS_EVAL_SEED:-0}\" 'BEGIN { printf \"%.3f\", ((s * 7919) % 100) / 1000 }')\nscore=$(awk",
        )
        .replace("-v n=\"NOISE\"", "-v n=\"$noise\"")
}

pub fn deterministic() -> String {
    DETERMINISTIC.replace("-v n=\"NOISE\"", "-v n=\"0\"")
}

pub fn block(body: &str) -> String {
    format!("import math\n# EVOLVE-BLOCK-START\n{body}# EVOLVE-BLOCK-END\nprint(math.pi)\n")
}

/// A model reply that rewrites the whole program with `body` in its block.
pub fn rewrite(body: &str) -> String {
    format!("Changed one thing.\n```python\n{}```\n", block(body))
}

/// One row of the investigation: generation, block body, score, correct.
pub type Row<'a> = (u32, &'a str, f64, bool);

pub struct Run {
    pub dir: tempfile::TempDir,
    pub context: PhaseContext,
    pub db: ProgramDb,
}

impl Run {
    pub fn paths(&self) -> &RunPaths {
        &self.context.paths
    }

    pub fn run_dir(&self) -> PathBuf {
        self.context.paths.run_dir.clone()
    }

    /// How many times the evaluator has been started.
    pub fn evaluations(&self) -> usize {
        std::fs::read_to_string(self.run_dir().join("eval_count.log"))
            .map(|log| log.lines().count())
            .unwrap_or(0)
    }

    pub fn events(&self) -> Vec<String> {
        fsutil::read_jsonl::<Value>(&self.context.paths.events())
            .iter()
            .map(|event| event["event"].as_str().unwrap_or("").to_string())
            .collect()
    }

    pub fn json(&self, relative: &str) -> Value {
        fsutil::read_json(&self.run_dir().join(relative))
            .unwrap_or_else(|error| panic!("{relative}: {error}"))
    }

    pub fn text(&self, relative: &str) -> String {
        std::fs::read_to_string(self.run_dir().join(relative))
            .unwrap_or_else(|error| panic!("{relative}: {error}"))
    }
}

impl Drop for Run {
    /// The final evaluation seals its results read-only; make the tree
    /// writable again so the temporary directory can be removed.
    fn drop(&mut self) {
        unseal(self.dir.path());
    }
}

pub fn unseal(root: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let Ok(metadata) = std::fs::symlink_metadata(root) else {
        return;
    };
    if metadata.is_dir() {
        let _ = std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o755));
        for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
            unseal(&entry.path());
        }
    } else if metadata.is_file() {
        let _ = std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o644));
    }
}

/// Lay out a run whose investigation holds `rows` (the first is the seed,
/// the rest its children) and freeze the downstream selection.
pub fn build(
    evaluator: &str,
    final_evaluator: Option<&str>,
    rows: &[Row<'_>],
    llm: LlmClient,
) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let paths = RunPaths::new(dir.path().join("run"));
    let task_dir = paths.task();
    std::fs::create_dir_all(&task_dir).unwrap();
    std::fs::write(
        task_dir.join("task.md"),
        "Bring x to 3 and y to -1; a bonus adds to the score.\n",
    )
    .unwrap();
    std::fs::write(task_dir.join("initial.py"), block(rows[0].1)).unwrap();
    std::fs::write(task_dir.join("evaluate.sh"), evaluator).unwrap();
    if let Some(script) = final_evaluator {
        std::fs::write(task_dir.join("evaluate_final.sh"), script).unwrap();
    }
    let task = Task::load(&task_dir).unwrap();

    let db_path = paths.investigation().join("programs.sqlite");
    let db = ProgramDb::open(
        Some(&db_path),
        DatabaseConfig {
            num_islands: 1,
            seed: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let mut seed_id = None;
    for (index, (generation, body, score, correct)) in rows.iter().enumerate() {
        let mut program = Program::new(block(body), "python");
        program.generation = *generation;
        program.combined_score = *score;
        program.correct = *correct;
        program.parent_id = seed_id.clone();
        program.timestamp = index as f64;
        if *generation > 0 {
            let patch_type = if index % 2 == 0 { "full" } else { "diff" };
            program
                .metadata
                .insert("patch_type".into(), json!(patch_type));
            program
                .metadata
                .insert("patch_name".into(), json!(format!("edit_{generation}")));
            program.metadata.insert(
                "arm".into(),
                json!(if index % 3 == 0 { "model-b" } else { "model-a" }),
            );
            program
                .metadata
                .insert("api_costs".into(), json!(0.01 * f64::from(*generation)));
            program
                .metadata
                .insert("parent_score".into(), json!(rows[0].2));
        } else {
            program
                .metadata
                .insert("patch_name".into(), json!("initial_program"));
        }
        db.add(&mut program).unwrap();
        seed_id.get_or_insert(program.id.clone());
    }
    freeze_selection(&paths, &task, &db, "best_available").unwrap();

    let events = EventLog::open(&paths.events(), ConsoleMode::Quiet, false);
    let context = PhaseContext {
        paths,
        task,
        llm,
        events,
        agent: None,
        cancel: Arc::new(AtomicBool::new(false)),
        seed: 11,
    };
    Run { dir, context, db }
}

/// A provider answering by pipeline stage; an unknown stage is an error the
/// client does not retry.
pub fn by_stage(
    handler: impl Fn(&str, &LlmRequest) -> Option<String> + Send + Sync + 'static,
) -> Arc<MockProvider> {
    MockProvider::new(move |request: &LlmRequest| {
        handler(request.stage.as_str(), request).ok_or_else(|| LlmError::Rejected {
            provider: "mock".into(),
            status: 400,
            message: format!("no scripted reply for stage `{}`", request.stage),
        })
    })
}

pub fn calls(provider: &MockProvider, stage: &str) -> usize {
    provider
        .calls()
        .iter()
        .filter(|call| call.stage == stage)
        .count()
}
