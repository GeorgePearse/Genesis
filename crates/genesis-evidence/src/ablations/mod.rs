//! Ablations: what each part of the selected program contributes.
//!
//! The study always evaluates the selected program unchanged
//! (`control_best`) and, when it differs, the seed it grew from
//! (`baseline_reference`). A model then proposes further variants, each
//! removing one innovation (an ablation) or replacing the approach with one
//! simpler method (a baseline); a second call reviews them, and code caps
//! and orders what survives. Each accepted variant is written by a model as
//! a full rewrite of its base program and kept only once the evaluator has
//! judged it correct.
//!
//! Every variant is then evaluated `num_replications` times, with the same
//! seeds for all, and compared with the control by Welch's test and a
//! bootstrap interval. A variant that cannot be implemented is recorded as
//! failed and the study goes on; a control that is not correct ends it.
//!
//! The phase follows the stages of the Agentic Scientist's variants
//! pipeline (extract, generate, select, materialize, execute, aggregate,
//! README), with the statistics it left to the paper phase computed here.

mod prompts;
pub mod report;
pub mod variants;

use crate::common::{self, ask_json, EvalJob, EvalSettings, UNTRUSTED};
use crate::csv;
use anyhow::{bail, ensure, Context, Result};
use genesis_core::{fsutil, EvalOutcome};
use genesis_llm::{extract_fenced, LlmRequest, Message, SamplingKwargs};
use genesis_scientist::artifacts::{
    AblationSummary, ReplicationResult, VariantResult, VariantSpec,
};
use genesis_scientist::selection::SelectedProgram;
use genesis_scientist::PhaseContext;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;
use variants::{
    baseline_reference_spec, control_spec, parse_candidates, parse_decisions, select_variants,
    RejectedVariant, Selection, BASELINE_REFERENCE_ID, CONTROL_ID,
};

const MAX_DIFF_CHARS: usize = 50_000;
const MAX_EVALUATOR_CHARS: usize = 20_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AblationConfig {
    /// Models that propose, review, implement and describe variants. With
    /// none, only the control and the reference baseline are evaluated.
    pub models: Vec<String>,
    pub kwargs: SamplingKwargs,
    /// Most generated ablations to run.
    pub max_ablations: usize,
    /// Most generated baselines to run, besides the reference baseline.
    pub max_extra_baselines: usize,
    /// Evaluations of each variant, one per seed.
    pub num_replications: usize,
    /// Tries a model gets to produce a correct implementation of a variant.
    pub max_evals_per_variant: u32,
    pub max_parallel_evals: usize,
    /// Overrides the task's evaluation time limit.
    pub eval_timeout_secs: Option<u64>,
    /// First seed; replication `i` uses `seed_base + i`.
    pub seed_base: u64,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            kwargs: SamplingKwargs::default(),
            max_ablations: 3,
            max_extra_baselines: 3,
            num_replications: 5,
            max_evals_per_variant: 2,
            max_parallel_evals: 4,
            eval_timeout_secs: None,
            seed_base: 1,
        }
    }
}

/// The seeds every variant of a study is evaluated with.
pub fn ablation_seeds(config: &AblationConfig) -> Vec<u64> {
    (0..config.num_replications as u64)
        .map(|offset| config.seed_base.wrapping_add(offset))
        .collect()
}

fn to_replication(seed: u64, outcome: &EvalOutcome) -> ReplicationResult {
    ReplicationResult {
        seed,
        combined_score: outcome.combined_score,
        correct: outcome.correct,
        evaluation_failed: outcome.evaluation_failed,
        runtime_secs: outcome.runtime_secs,
        public_metrics: outcome.public_metrics.clone(),
    }
}

struct Planned {
    spec: VariantSpec,
    /// `completed` once it has a program to evaluate, else `failed`.
    status: String,
    notes: String,
}

struct Study<'a> {
    context: &'a PhaseContext,
    config: &'a AblationConfig,
    selected: SelectedProgram,
    dir: PathBuf,
    seeds: Vec<u64>,
    file_name: String,
}

impl Study<'_> {
    fn variant_dir(&self, id: &str) -> PathBuf {
        self.dir.join("variants").join(id)
    }

    fn program_path(&self, id: &str) -> PathBuf {
        self.variant_dir(id).join("program").join(&self.file_name)
    }

    fn rep_dir(&self, id: &str, seed: u64) -> PathBuf {
        self.variant_dir(id).join(format!("rep_{seed}"))
    }

    fn eval_settings(&self) -> EvalSettings<'_> {
        EvalSettings {
            task: &self.context.task,
            parallel: self.config.max_parallel_evals,
            timeout: self.config.eval_timeout_secs.map(Duration::from_secs),
            final_evaluator: false,
            cancel: &self.context.cancel,
        }
    }

    fn base_code(&self, spec: &VariantSpec) -> &str {
        if spec.base_program == "reference" {
            &self.selected.baseline_code
        } else {
            &self.selected.code
        }
    }

    fn program_digest(&self, id: &str) -> Option<String> {
        fsutil::sha256_file(&self.program_path(id)).ok()
    }

    /// A replication already on disk for the program as it is now.
    fn stored_replication(&self, id: &str, seed: u64) -> Option<ReplicationResult> {
        let record: Value =
            fsutil::read_json_opt(&self.rep_dir(id, seed).join("replication.json"))?;
        if record.get("program_sha256").and_then(Value::as_str)
            != self.program_digest(id).as_deref()
        {
            return None;
        }
        serde_json::from_value(record.get("result")?.clone()).ok()
    }

    fn store_replication(
        &self,
        id: &str,
        seed: u64,
        outcome: &EvalOutcome,
    ) -> Result<ReplicationResult> {
        let result = to_replication(seed, outcome);
        fsutil::write_json(
            &self.rep_dir(id, seed).join("replication.json"),
            &json!({"program_sha256": self.program_digest(id), "result": result}),
        )?;
        Ok(result)
    }

    fn job(&self, id: &str, seed: u64) -> EvalJob {
        EvalJob {
            key: id.to_string(),
            program_path: self.program_path(id),
            results_dir: self.rep_dir(id, seed).join("results"),
            seed,
        }
    }

    async fn replicate(&self, id: &str, seed: u64) -> Result<ReplicationResult> {
        if let Some(stored) = self.stored_replication(id, seed) {
            return Ok(stored);
        }
        let outcome = common::run_job(&self.eval_settings(), &self.job(id, seed)).await;
        self.store_replication(id, seed, &outcome)
    }

    /// Record what is being studied, and refuse a directory that was built
    /// for another program.
    fn extract(&self) -> Result<String> {
        let selection = &self.selected.selection;
        let manifest_path = self.dir.join("extraction_manifest.json");
        if let Some(previous) = fsutil::read_json_opt::<Value>(&manifest_path) {
            ensure!(
                previous.get("program_sha256").and_then(Value::as_str) == Some(selection.program_sha256.as_str()),
                "{} was built for a different selected program; ablations of two programs cannot share a directory",
                self.dir.display()
            );
        }
        let full_diff = genesis_edit::unified_diff(
            &self.selected.baseline_code,
            &self.selected.code,
            &self.context.task.language,
        );
        let truncated = full_diff.chars().count() > MAX_DIFF_CHARS;
        let diff = if truncated {
            let kept: String = full_diff.chars().take(MAX_DIFF_CHARS).collect();
            format!("{kept}\n... diff truncated at {MAX_DIFF_CHARS} characters ...\n")
        } else {
            full_diff
        };
        fsutil::write_text(
            &self.dir.join("generation").join("program_diff.diff"),
            &diff,
        )?;
        fsutil::write_json(
            &manifest_path,
            &json!({
                "extraction_version": 1,
                "extracted_at": chrono::Utc::now().to_rfc3339(),
                "selection_kind": selection.selection_kind,
                "investigation_outcome": selection.investigation_outcome,
                "program_id": selection.program_id,
                "generation": selection.generation,
                "combined_score": selection.combined_score,
                "program_path": selection.program_path,
                "program_sha256": selection.program_sha256,
                "baseline_path": selection.baseline_path,
                "baseline_sha256": selection.baseline_sha256,
                "baseline_score": selection.baseline_score,
                "task_fingerprint": selection.task_fingerprint,
                "diff_truncated": truncated,
            }),
        )?;
        Ok(diff)
    }

    /// The part of every prompt that describes the study.
    fn study_context(&self, diff: &str) -> String {
        let task = &self.context.task;
        let selection = &self.selected.selection;
        let diff_block = if diff.trim().is_empty() {
            "(empty: the best program is the reference program itself, so it has no innovations to ablate)".to_string()
        } else {
            format!("```diff\n{diff}\n```")
        };
        let evaluator = ["evaluate.py", "evaluate.sh"]
            .iter()
            .find_map(|name| {
                std::fs::read_to_string(task.dir.join(name))
                    .ok()
                    .map(|source| (name, source))
            })
            .map(|(name, source)| {
                format!(
                    "\n\nSource of `{name}`:\n```\n{}\n```",
                    fsutil::truncate(&source, MAX_EVALUATOR_CHARS)
                )
            })
            .unwrap_or_default();
        format!(
            "{UNTRUSTED}\n\n## Task Description\n{}\n\n## Selection\n```json\n{}\n```\n\n\
             ## Program Diff (best vs reference)\n{diff_block}\n\n## Best program\n```{}\n{}\n```\n\n\
             ## Evaluator\nPrograms are scored by `{}`; a higher `combined_score` is better and an incorrect \
             program does not count.{evaluator}",
            task.goal,
            json!({
                "selection_kind": selection.selection_kind,
                "combined_score": selection.combined_score,
                "baseline_score": selection.baseline_score,
                "generation": selection.generation,
            }),
            task.language,
            self.selected.code,
            task.eval.command.join(" "),
        )
    }

    async fn generate(&self, diff: &str) -> Result<Vec<VariantSpec>> {
        let events = &self.context.events;
        let bundle_path = self.dir.join("generation").join("variant_bundle.json");
        if let Some(bundle) = fsutil::read_json_opt::<Value>(&bundle_path) {
            return Ok(parse_candidates(&bundle));
        }
        if self.config.models.is_empty() {
            fsutil::write_json(
                &bundle_path,
                &json!({
                    "decision": "no_relevant_variants",
                    "summary": "no model is configured to propose variants",
                    "candidates": [],
                    "num_candidates": 0,
                }),
            )?;
            return Ok(Vec::new());
        }
        let user = format!(
            "{}\n\nPropose the variants as the JSON object described in the system message.",
            self.study_context(diff)
        );
        let reply = ask_json(
            &self.context.llm,
            "variant_generate",
            &self.config.models,
            &self.config.kwargs,
            prompts::GENERATE_SYSTEM,
            user,
            self.context.seed,
        )
        .await;
        match reply {
            Ok((reply, response)) => {
                let candidates = parse_candidates(&reply);
                fsutil::write_json(
                    &bundle_path,
                    &json!({
                        "decision": reply.get("decision").and_then(Value::as_str).unwrap_or("propose_variants"),
                        "summary": reply.get("summary").and_then(Value::as_str).unwrap_or(""),
                        "candidates": candidates,
                        "num_candidates": candidates.len(),
                    }),
                )?;
                events.info(
                    "variants_generated",
                    json!({"candidates": candidates.len(), "cost_usd": response.cost_usd}),
                );
                Ok(candidates)
            }
            Err(error) => {
                events.warn(
                    "variant_generation_failed",
                    json!({"error": format!("{error:#}")}),
                );
                Ok(Vec::new())
            }
        }
    }

    async fn select(
        &self,
        candidates: &[VariantSpec],
        diff: &str,
        control_ids: &[String],
    ) -> Result<Selection> {
        let events = &self.context.events;
        let path = self.dir.join("generation").join("variant_selection.json");
        let candidate_ids: Vec<&str> = candidates
            .iter()
            .map(|candidate| candidate.variant_id.as_str())
            .collect();
        let caps = json!({"max_ablations": self.config.max_ablations, "max_extra_baselines": self.config.max_extra_baselines});
        if let Some(stored) = fsutil::read_json_opt::<Value>(&path) {
            let same = stored.get("caps") == Some(&caps)
                && stored.get("candidate_ids") == Some(&json!(candidate_ids));
            let accepted = stored
                .get("accepted")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok());
            let rejected = stored
                .get("rejected")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok());
            if let (true, Some(accepted), Some(rejected)) = (same, accepted, rejected) {
                return Ok(Selection { accepted, rejected });
            }
        }
        let mut decisions = BTreeMap::new();
        if !candidates.is_empty() && !self.config.models.is_empty() {
            let task = &self.context.task;
            let user = format!(
                "{UNTRUSTED}\n\n## Task Description\n{}\n\n## Program Diff (best vs reference)\n```diff\n{diff}\n```\n\n\
                 ## Proposed Candidates\n```json\n{}\n```\n\nGive one decision per candidate.",
                task.goal,
                serde_json::to_string_pretty(candidates)?
            );
            match ask_json(
                &self.context.llm,
                "variant_select",
                &self.config.models,
                &self.config.kwargs,
                prompts::SELECT_SYSTEM,
                user,
                self.context.seed.wrapping_add(1),
            )
            .await
            {
                Ok((reply, _)) => decisions = parse_decisions(&reply),
                Err(error) => {
                    // Fail open: a reviewer that cannot be reached rejects nothing.
                    events.warn(
                        "variant_selection_failed",
                        json!({"error": format!("{error:#}")}),
                    );
                }
            }
        }
        let selection = select_variants(
            candidates,
            &decisions,
            self.config.max_ablations,
            self.config.max_extra_baselines,
        );
        fsutil::write_json(
            &path,
            &json!({
                "controls": control_ids,
                "accepted": selection.accepted,
                "rejected": selection.rejected,
                "num_accepted": selection.accepted.len(),
                "num_rejected": selection.rejected.len(),
                "caps": caps,
                "candidate_ids": candidate_ids,
            }),
        )?;
        events.info(
            "variants_selected",
            json!({"accepted": selection.accepted.len(), "rejected": selection.rejected.len()}),
        );
        Ok(selection)
    }

    fn record_implementation(&self, id: &str, status: &str, tries: u32, notes: &str) -> Result<()> {
        fsutil::write_json(
            &self.variant_dir(id).join("implementation.json"),
            &json!({"status": status, "tries": tries, "notes": notes, "program_sha256": self.program_digest(id)}),
        )
    }

    /// The variant's code from a model reply, or what to tell the model.
    fn variant_code(&self, reply: &str, base: &str) -> std::result::Result<String, String> {
        let language = &self.context.task.language;
        let code = if genesis_edit::mutable_ranges(base).is_empty() {
            let tags = genesis_edit::fence_tags(language);
            let tags: Vec<&str> = tags.iter().map(String::as_str).collect();
            let mut code = extract_fenced(reply, &tags).ok_or_else(|| {
                format!("no fenced ```{language} code block was found in the reply")
            })?;
            if !code.ends_with('\n') {
                code.push('\n');
            }
            code
        } else {
            genesis_edit::apply_full_patch(reply, base, language)
                .map_err(|error| error.0)?
                .code
        };
        if code == base {
            return Err("the variant is identical to the program it is derived from; make the one change the variant describes".to_string());
        }
        Ok(code)
    }

    /// Have a model write the variant, keeping the first implementation the
    /// evaluator judges correct. Returns whether one was found, and a note.
    async fn implement(&self, spec: &VariantSpec, index: usize) -> Result<(bool, String)> {
        let id = spec.variant_id.as_str();
        let events = &self.context.events;
        if let Some(record) =
            fsutil::read_json_opt::<Value>(&self.variant_dir(id).join("implementation.json"))
        {
            let notes = record
                .get("notes")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            match record.get("status").and_then(Value::as_str) {
                Some("accepted")
                    if record.get("program_sha256").and_then(Value::as_str)
                        == self.program_digest(id).as_deref()
                        && self.program_path(id).is_file() =>
                {
                    return Ok((true, notes));
                }
                Some("failed") => return Ok((false, notes)),
                _ => {}
            }
        }
        if self.config.models.is_empty() {
            let notes = "no model is configured to implement variants";
            self.record_implementation(id, "failed", 0, notes)?;
            return Ok((false, notes.to_string()));
        }

        let task = &self.context.task;
        let base = self.base_code(spec);
        let mut rng =
            ChaCha8Rng::seed_from_u64(self.context.seed.wrapping_add(1000 + index as u64));
        let model =
            &self.config.models[rand::Rng::gen_range(&mut rng, 0..self.config.models.len())];
        let user = format!(
            "{UNTRUSTED}\n\n# Task\n{}\n\n# Variant to implement\n```json\n{}\n```\n\n\
             # Program to derive the variant from (`{}`)\n```{}\n{base}\n```\n\nReply with the complete variant program.",
            task.goal,
            serde_json::to_string_pretty(spec)?,
            spec.base_program,
            task.language,
        );
        let mut request = LlmRequest::new(
            model.clone(),
            prompts::IMPLEMENT_SYSTEM.replace("{language}", &task.language),
            user,
        );
        self.config.kwargs.apply(&mut request, &mut rng);

        let tries = self.config.max_evals_per_variant.max(1);
        let mut problem = String::from("no implementation was attempted");
        for attempt in 1..=tries {
            if self.context.cancelled() {
                bail!("the ablation phase was interrupted while implementing `{id}`");
            }
            let response = match self
                .context
                .llm
                .query("variant_implement", request.clone())
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    problem = format!("the model call failed: {error}");
                    break;
                }
            };
            let try_dir = self
                .variant_dir(id)
                .join("attempts")
                .join(format!("try_{attempt}"));
            fsutil::write_text(&try_dir.join("response.txt"), &response.content)?;
            let feedback = match self.variant_code(&response.content, base) {
                Err(reason) => {
                    problem = format!("try {attempt}: {reason}");
                    format!("The reply could not be used: {reason}\n\nReply again with the complete program in one fenced block.")
                }
                Ok(code) => {
                    let candidate_path = try_dir.join(&self.file_name);
                    fsutil::write_text(&candidate_path, &code)?;
                    let job = EvalJob {
                        key: id.to_string(),
                        program_path: candidate_path,
                        results_dir: try_dir.join("results"),
                        seed: self.seeds[0],
                    };
                    let outcome = common::run_job(&self.eval_settings(), &job).await;
                    if outcome.correct {
                        fsutil::write_text(&self.program_path(id), &code)?;
                        fsutil::copy_tree(
                            &job.results_dir,
                            &self.rep_dir(id, self.seeds[0]).join("results"),
                            &[],
                        )?;
                        self.store_replication(id, self.seeds[0], &outcome)?;
                        let notes = format!(
                            "Implemented by `{}` on try {attempt} of {tries}.",
                            response.model
                        );
                        self.record_implementation(id, "accepted", attempt, &notes)?;
                        events.info("variant_implemented", json!({"variant": id, "tries": attempt, "score": outcome.combined_score}));
                        return Ok((true, notes));
                    }
                    problem = format!(
                        "try {attempt}: the implementation was not correct: {}",
                        fsutil::truncate(outcome.text_feedback.trim(), 400)
                    );
                    format!(
                        "The variant was evaluated and is not correct.\n\nEvaluator feedback:\n{}\n\nstderr (tail):\n{}\n\n\
                         Fix the variant so that it is correct while still making only its one change. Reply with the \
                         complete program again.",
                        outcome.text_feedback.trim(),
                        fsutil::tail(&outcome.stderr_tail, 2000)
                    )
                }
            };
            request.messages.push(Message::assistant(response.content));
            request.messages.push(Message::user(feedback));
        }
        let notes = format!("No correct implementation in {tries} tries; {problem}.");
        self.record_implementation(id, "failed", tries, &notes)?;
        events.warn("variant_failed", json!({"variant": id, "reason": problem}));
        Ok((false, notes))
    }

    /// Descriptions of the variants for the README, from a model when one
    /// is configured. Results are never shown to it.
    async fn describe(&self, planned: &[Planned]) -> (Option<String>, BTreeMap<String, String>) {
        let path = self.dir.join("generation").join("variant_summaries.json");
        let ids: Vec<&str> = planned
            .iter()
            .map(|variant| variant.spec.variant_id.as_str())
            .collect();
        let read = |value: &Value| -> (Option<String>, BTreeMap<String, String>) {
            let task_summary = value
                .get("task_summary")
                .and_then(Value::as_str)
                .map(str::to_string);
            let summaries = value
                .get("variant_summaries")
                .and_then(Value::as_object)
                .map(|map| {
                    map.iter()
                        .filter_map(|(id, text)| {
                            text.as_str().map(|text| (id.clone(), text.to_string()))
                        })
                        .collect()
                })
                .unwrap_or_default();
            (task_summary, summaries)
        };
        if let Some(stored) = fsutil::read_json_opt::<Value>(&path) {
            if stored.get("variant_ids") == Some(&json!(ids)) {
                return read(&stored);
            }
        }
        if self.config.models.is_empty() {
            return (None, BTreeMap::new());
        }
        let specs: Vec<Value> = planned
            .iter()
            .map(|variant| {
                json!({
                    "variant_id": variant.spec.variant_id,
                    "title": variant.spec.title,
                    "kind": variant.spec.kind,
                    "hypothesis": variant.spec.hypothesis,
                    "target_components": variant.spec.target_components,
                    "implementation_plan": variant.spec.implementation_plan,
                })
            })
            .collect();
        let user = format!(
            "{UNTRUSTED}\n\n## Task Description\n{}\n\n## Variant Specifications\n```json\n{}\n```",
            self.context.task.goal,
            serde_json::to_string_pretty(&specs).unwrap_or_default()
        );
        let reply = ask_json(
            &self.context.llm,
            "variant_readme",
            &self.config.models,
            &self.config.kwargs,
            prompts::README_SYSTEM,
            user,
            self.context.seed.wrapping_add(2),
        )
        .await;
        match reply {
            Ok((mut reply, _)) => {
                if let Some(object) = reply.as_object_mut() {
                    object.insert("variant_ids".to_string(), json!(ids));
                }
                let _ = fsutil::write_json(&path, &reply);
                read(&reply)
            }
            Err(error) => {
                self.context.events.warn(
                    "variant_readme_failed",
                    json!({"error": format!("{error:#}")}),
                );
                (None, BTreeMap::new())
            }
        }
    }
}

/// Run the ablation study of the frozen selection under
/// `<run>/ablations`, reusing whatever an earlier call completed.
pub async fn run_ablations(
    context: &PhaseContext,
    config: &AblationConfig,
) -> Result<AblationSummary> {
    let selected = common::selected(&context.paths)?;
    let dir = context.paths.ablations();
    let events = &context.events;
    let receipt_path = dir.join(".ablations_complete.json");
    let summary_path = dir.join("pipeline_summary.json");
    let receipt = json!({
        "program_sha256": selected.selection.program_sha256,
        "baseline_sha256": selected.selection.baseline_sha256,
        "config": config,
    });
    if fsutil::read_json_opt::<Value>(&receipt_path).as_ref() == Some(&receipt) {
        if let Some(summary) = fsutil::read_json_opt::<AblationSummary>(&summary_path) {
            events.info(
                "ablations_reused",
                json!({"variants": summary.total_variants}),
            );
            return Ok(summary);
        }
    }

    if context.cancelled() {
        bail!("the ablation phase was interrupted before it started");
    }
    let seeds = ablation_seeds(config);
    ensure!(!seeds.is_empty(), "num_replications must be at least 1");
    let sealed: Vec<u64> =
        fsutil::read_json_opt::<Value>(&context.paths.final_evaluation().join("invocation.json"))
            .and_then(|invocation| serde_json::from_value(invocation.get("seeds")?.clone()).ok())
            .unwrap_or_default();
    if let Some(shared) = seeds.iter().find(|seed| sealed.contains(seed)) {
        bail!("ablation seed {shared} is one of the sealed final-evaluation seeds; choose another seed_base");
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let study = Study {
        context,
        config,
        file_name: context.task.program_file_name(),
        selected,
        dir: dir.clone(),
        seeds,
    };
    let selection = study.selected.selection.clone();
    events.info(
        "ablations_start",
        json!({
            "selection_kind": selection.selection_kind,
            "search_score": selection.combined_score,
            "replications": config.num_replications,
            "seeds": study.seeds,
        }),
    );
    let diff = study.extract()?;

    let mut planned = vec![Planned {
        spec: control_spec(),
        status: "completed".to_string(),
        notes: String::new(),
    }];
    fsutil::write_text(&study.program_path(CONTROL_ID), &study.selected.code)?;
    if study.selected.code != study.selected.baseline_code {
        planned.push(Planned {
            spec: baseline_reference_spec(),
            status: "completed".to_string(),
            notes: String::new(),
        });
        fsutil::write_text(
            &study.program_path(BASELINE_REFERENCE_ID),
            &study.selected.baseline_code,
        )?;
    }

    let gate = study.replicate(CONTROL_ID, study.seeds[0]).await?;
    if !gate.correct {
        let feedback = fsutil::read_json_opt::<Value>(
            &study
                .rep_dir(CONTROL_ID, study.seeds[0])
                .join("results")
                .join("correct.json"),
        )
        .and_then(|doc| doc.get("error").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_default();
        bail!(
            "the control (the selected program, unchanged) is not correct under seed {}; an ablation study has no \
             anchor without it. {feedback}",
            study.seeds[0]
        );
    }

    let candidates = study.generate(&diff).await?;
    let control_ids: Vec<String> = planned
        .iter()
        .map(|variant| variant.spec.variant_id.clone())
        .collect();
    let chosen = study.select(&candidates, &diff, &control_ids).await?;
    for (index, spec) in chosen.accepted.iter().enumerate() {
        if context.cancelled() {
            bail!("the ablation phase was interrupted; completed work is kept and will be reused");
        }
        fsutil::write_json(
            &study
                .variant_dir(&spec.variant_id)
                .join("variant_spec.json"),
            spec,
        )?;
        let (accepted, notes) = study.implement(spec, index).await?;
        let status = if accepted { "completed" } else { "failed" };
        planned.push(Planned {
            spec: spec.clone(),
            status: status.to_string(),
            notes,
        });
    }
    for variant in planned.iter().take(control_ids.len()) {
        fsutil::write_json(
            &study
                .variant_dir(&variant.spec.variant_id)
                .join("variant_spec.json"),
            &variant.spec,
        )?;
    }

    let jobs: Vec<EvalJob> = planned
        .iter()
        .filter(|variant| variant.status == "completed")
        .flat_map(|variant| {
            let id = variant.spec.variant_id.as_str();
            study
                .seeds
                .iter()
                .filter(|seed| study.stored_replication(id, **seed).is_none())
                .map(|seed| study.job(id, *seed))
        })
        .collect();
    let ran = common::run_jobs(&study.eval_settings(), jobs).await;
    let mut interrupted = false;
    for (job, outcome) in &ran {
        match outcome {
            Some(outcome) => {
                study.store_replication(&job.key, job.seed, outcome)?;
            }
            None => interrupted = true,
        }
    }
    if interrupted {
        bail!("the ablation phase was interrupted; completed replications are kept and will be reused");
    }

    let mut results: Vec<VariantResult> = planned
        .iter()
        .map(|variant| {
            let id = variant.spec.variant_id.as_str();
            let completed = variant.status == "completed";
            VariantResult {
                spec: variant.spec.clone(),
                status: variant.status.clone(),
                program_path: completed.then(|| context.paths.relative(&study.program_path(id))),
                replications: if completed {
                    study
                        .seeds
                        .iter()
                        .filter_map(|seed| study.stored_replication(id, *seed))
                        .collect()
                } else {
                    Vec::new()
                },
                notes: variant.notes.clone(),
                ..VariantResult::default()
            }
        })
        .collect();
    let control_scores = report::correct_scores(&results[0].replications);
    let stochastic = report::is_stochastic(&control_scores);
    for (index, result) in results.iter_mut().enumerate() {
        if result.status == "completed" {
            report::summarise_variant(
                result,
                &control_scores,
                stochastic,
                context.seed.wrapping_add(index as u64),
            );
            events.info(
                "variant_evaluated",
                json!({
                    "variant": result.spec.variant_id,
                    "kind": result.spec.kind,
                    "mean": result.summary.as_ref().map(|summary| summary.mean),
                    "correct_rate": result.correct_rate,
                    "delta_vs_control": result.comparison_to_control.as_ref().map(|comparison| comparison.mean_diff),
                }),
            );
        }
    }

    let summary = AblationSummary {
        schema_version: 1,
        task: context.task.name.clone(),
        primary_metric_name: "combined_score".to_string(),
        anchor_variant_id: CONTROL_ID.to_string(),
        num_replications: config.num_replications,
        seeds: study.seeds.clone(),
        evaluator_is_stochastic: stochastic,
        total_variants: results.len(),
        total_completed: results
            .iter()
            .filter(|result| result.status == "completed")
            .count(),
        total_failed: results
            .iter()
            .filter(|result| result.status == "failed")
            .count(),
        variants: results,
        generated_at: chrono::Utc::now().to_rfc3339(),
    };
    let relative_dir = context.paths.relative(&dir);
    csv::write(
        &dir.join("aggregated").join("run_summary.csv"),
        &report::RUN_SUMMARY_HEADER,
        &report::run_summary_rows(&summary, &relative_dir),
    )?;
    csv::write(
        &dir.join("aggregated").join("metric_values.csv"),
        &report::METRIC_VALUES_HEADER,
        &report::metric_value_rows(&summary),
    )?;
    let (task_summary, variant_summaries) = study.describe(&planned).await;
    let rejected: &[RejectedVariant] = &chosen.rejected;
    fsutil::write_text(
        &dir.join("README.md"),
        &report::readme(&report::ReadmeInputs {
            summary: &summary,
            selection_kind: &selection.selection_kind,
            search_score: selection.combined_score,
            task_summary: task_summary.as_deref(),
            variant_summaries: &variant_summaries,
            rejected,
        }),
    )?;
    fsutil::write_json(&summary_path, &summary)?;
    fsutil::write_json(&receipt_path, &receipt)?;
    events.info(
        "ablations_end",
        json!({
            "variants": summary.total_variants,
            "completed": summary.total_completed,
            "failed": summary.total_failed,
            "evaluator_is_stochastic": summary.evaluator_is_stochastic,
        }),
    );
    Ok(summary)
}
