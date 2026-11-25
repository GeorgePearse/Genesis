//! The evolution loop.
//!
//! Generation 0 is the task's seed program. Every later generation is one
//! proposal: sample a parent and inspirations, let the bandit pick a model
//! or agent, produce a candidate, reject it early if it is not novel,
//! evaluate it with the task's evaluator, and store the result. Up to
//! `max_evaluation_jobs` proposals are in flight at once; results are
//! stored one at a time, in completion order, so the archive, the bandit
//! and the scratchpad always see a consistent history.
//!
//! A run is resumable: pointing a new runner at the same results directory
//! continues with the generations that have neither a program nor a
//! recorded terminal failure.

use crate::config::{DatabaseConfig, EvolutionConfig};
use crate::db::prompts::{PromptDb, PromptDbConfig};
use crate::db::{ProgramDb, Sample};
use crate::hooks::{EvalHook, EvaluatedProposal, IdeaSource, Operator, Proposal, ProposalContext};
use crate::meta::{MetaConfig, MetaSummarizer};
use crate::novelty::{NoveltyJudge, NoveltyVerdict};
use crate::operators::PatchOperator;
use crate::prompt_evolver::{PromptEvolver, PromptEvolverConfig};
use crate::sampler::PromptSampler;
use anyhow::{Context, Result};
use genesis_core::{fsutil, EvalOutcome, EventLog, IdeaBundle, Program, Task};
use genesis_llm::{build_bandit, load_bandit_state, save_bandit_state, Bandit, LlmClient};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

/// What a runner is given besides its configuration.
pub struct RunnerDeps {
    pub task: Task,
    pub results_dir: PathBuf,
    pub llm: LlmClient,
    pub events: Arc<EventLog>,
    /// Serves `agent:` entries of `llm_models`.
    pub agent_operator: Option<Arc<dyn Operator>>,
    pub idea_source: Option<Arc<dyn IdeaSource>>,
    pub eval_hooks: Vec<Arc<dyn EvalHook>>,
    /// Set to stop the run after in-flight work is abandoned.
    pub cancel: Arc<AtomicBool>,
    /// Model spend of earlier sessions of this run, counted against
    /// `max_api_costs`.
    pub prior_cost_usd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunSummary {
    pub best: Option<Program>,
    pub num_generations: u32,
    /// Generations that produced an evaluated program, the seed included.
    pub generations_completed: usize,
    pub failed_proposals: usize,
    pub programs: usize,
    pub correct_programs: usize,
    pub cost_usd: f64,
    /// `completed`, `budget_exhausted`, `llm_endpoint_error`, `interrupted`
    /// or `seed_evaluation_failed`.
    pub stop_reason: String,
    pub elapsed_secs: f64,
    /// The run produced nothing beyond its seed, or stopped on an unusable
    /// model endpoint.
    pub evolution_failed: bool,
    pub results_dir: PathBuf,
}

struct Candidate {
    sample: Sample,
    proposal: Proposal,
    idea: Option<IdeaBundle>,
    system_prompt_id: Option<String>,
    arm: usize,
}

struct Evaluated {
    generation: u32,
    gen_dir: PathBuf,
    candidate: Candidate,
    embedding: Vec<f32>,
    novelty: Option<NoveltyVerdict>,
    outcome: EvalOutcome,
    api_cost: f64,
    novelty_cost: f64,
    posterior: Vec<f64>,
    meta_recommendation: Option<String>,
    proposal_secs: f64,
}

struct FailedProposal {
    generation: u32,
    class: String,
    stage: String,
    message: String,
    cost_usd: f64,
    endpoint_failure: bool,
    budget_exhausted: bool,
    interrupted: bool,
    idea: Option<IdeaBundle>,
}

enum GenerationOutcome {
    Evaluated(Box<Evaluated>),
    Failed(Box<FailedProposal>),
}

struct Shared {
    config: EvolutionConfig,
    task: Arc<Task>,
    results_dir: PathBuf,
    llm: LlmClient,
    events: Arc<EventLog>,
    db: Arc<ProgramDb>,
    sampler: Arc<PromptSampler>,
    patch_operator: Arc<dyn Operator>,
    agent_operator: Option<Arc<dyn Operator>>,
    idea_source: Option<Arc<dyn IdeaSource>>,
    eval_hooks: Vec<Arc<dyn EvalHook>>,
    bandit: Mutex<Box<dyn Bandit>>,
    failed_arms: Mutex<HashSet<usize>>,
    /// Charges made outside the model client (billed coding agents).
    external_spend: Mutex<f64>,
    meta: Option<Arc<MetaSummarizer>>,
    novelty: Option<NoveltyJudge>,
    prompt_evolver: Option<Arc<PromptEvolver>>,
    seed: u64,
    cancel: Arc<AtomicBool>,
}

pub struct EvolutionRunner {
    shared: Arc<Shared>,
    prior_cost_usd: f64,
    spent_at_start: f64,
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn weighted_choice(rng: &mut ChaCha8Rng, weights: &[f64]) -> Option<usize> {
    let total: f64 = weights.iter().filter(|w| w.is_finite() && **w > 0.0).sum();
    if total <= 0.0 {
        return None;
    }
    let mut point = rng.gen::<f64>() * total;
    let mut last = None;
    for (index, weight) in weights.iter().enumerate() {
        if !weight.is_finite() || *weight <= 0.0 {
            continue;
        }
        last = Some(index);
        if point < *weight {
            return Some(index);
        }
        point -= weight;
    }
    last
}

impl Shared {
    /// Draw an arm from the bandit's posterior, skipping arms whose endpoint
    /// has failed in this run.
    fn select_arm(&self, rng: &mut ChaCha8Rng) -> Option<(usize, Vec<f64>)> {
        let mut posterior = lock(&self.bandit).posterior();
        let failed = lock(&self.failed_arms).clone();
        let alive: Vec<usize> = (0..posterior.len())
            .filter(|arm| !failed.contains(arm))
            .collect();
        if alive.is_empty() {
            return None;
        }
        for arm in &failed {
            if let Some(probability) = posterior.get_mut(*arm) {
                *probability = 0.0;
            }
        }
        let arm = weighted_choice(rng, &posterior)
            .unwrap_or_else(|| alive[rng.gen_range(0..alive.len())]);
        Some((arm, posterior))
    }

    fn operator_for(&self, arm_name: &str) -> Arc<dyn Operator> {
        match (
            &self.agent_operator,
            EvolutionConfig::is_agent_arm(arm_name),
        ) {
            (Some(agent), true) => Arc::clone(agent),
            _ => Arc::clone(&self.patch_operator),
        }
    }

    async fn run_generation(self: Arc<Self>, generation: u32) -> GenerationOutcome {
        let started = Instant::now();
        let gen_dir = self.results_dir.join(format!("gen_{generation}"));
        if gen_dir.exists() {
            let aside = self.results_dir.join(format!(
                ".interrupted-gen_{generation}-{}",
                &uuid::Uuid::new_v4().simple().to_string()[..8]
            ));
            let _ = std::fs::rename(&gen_dir, aside);
        }
        let _ = std::fs::create_dir_all(&gen_dir);
        let mut rng = ChaCha8Rng::seed_from_u64(
            self.seed
                .wrapping_add(u64::from(generation).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );

        let meta_recommendation = self
            .meta
            .as_ref()
            .and_then(|meta| meta.recommendation(self.config.sample_single_meta_rec, &mut rng));
        let conclusions = self
            .idea_source
            .as_ref()
            .and_then(|source| source.conclusions());

        let mut failure = FailedProposal {
            generation,
            class: "proposal_generation_failed".to_string(),
            stage: "proposal".to_string(),
            message: "no proposal was produced".to_string(),
            cost_usd: 0.0,
            endpoint_failure: false,
            budget_exhausted: false,
            interrupted: false,
            idea: None,
        };
        let Some((mut arm, posterior)) = self.select_arm(&mut rng) else {
            failure.class = "llm_endpoint_error".to_string();
            failure.message = "every configured model endpoint has failed".to_string();
            failure.endpoint_failure = true;
            return GenerationOutcome::Failed(Box::new(failure));
        };

        let (mut api_cost, mut novelty_cost) = (0.0, 0.0);
        // Per arm: whether it delivered a candidate, and what its failures cost.
        let mut arm_outcomes: Vec<(usize, bool, f64)> = Vec::new();
        let note = |outcomes: &mut Vec<(usize, bool, f64)>,
                    arm: usize,
                    delivered: bool,
                    failed_cost: f64| {
            match outcomes.iter_mut().find(|entry| entry.0 == arm) {
                Some(entry) => {
                    entry.1 |= delivered;
                    entry.2 += failed_cost;
                }
                None => outcomes.push((arm, delivered, failed_cost)),
            }
        };
        let mut accepted: Option<(Candidate, Vec<f32>, Option<NoveltyVerdict>)> = None;

        'novelty: for novelty_attempt in 0..self.config.max_novelty_attempts {
            let mut candidate = None;
            for resample in 0..self.config.max_patch_resamples {
                if self.cancel.load(Ordering::SeqCst) {
                    failure.interrupted = true;
                    break 'novelty;
                }
                let sample = match self.db.sample() {
                    Ok(sample) => sample,
                    Err(error) => {
                        failure.message = format!("sampling a parent failed: {error:#}");
                        break 'novelty;
                    }
                };
                let attempt_dir = gen_dir.join(format!(
                    "attempts/novelty_{novelty_attempt}/resample_{resample}"
                ));
                let system_prompt = if sample.needs_fix {
                    None
                } else {
                    self.prompt_evolver
                        .as_ref()
                        .and_then(|evolver| evolver.sample())
                };
                let arm_name = self.config.llm_models[arm].clone();
                let mut context = ProposalContext {
                    generation,
                    attempt_dir: attempt_dir.clone(),
                    task: Arc::clone(&self.task),
                    parent: sample.parent.clone(),
                    archive_inspirations: sample.archive_inspirations.clone(),
                    top_k_inspirations: sample.top_k_inspirations.clone(),
                    needs_fix: sample.needs_fix,
                    system_prompt: system_prompt
                        .as_ref()
                        .map(|prompt| prompt.prompt_text.clone())
                        .unwrap_or_else(|| self.sampler.task_sys_msg().to_string()),
                    meta_recommendation: if sample.needs_fix {
                        None
                    } else {
                        meta_recommendation.clone()
                    },
                    conclusions: conclusions.clone(),
                    idea: None,
                    arm: arm_name.clone(),
                    best: self.db.best(None).ok().flatten(),
                    rng_seed: rng.gen(),
                };
                if let (Some(source), false) = (&self.idea_source, sample.needs_fix) {
                    match source.generate(&context).await {
                        Ok((idea, cost)) => {
                            api_cost += cost;
                            failure.idea = Some(idea.clone());
                            context.idea = Some(idea);
                        }
                        Err(error) => {
                            self.events.warn(
                                "idea_generation_failed",
                                json!({"generation": generation, "error": format!("{error:#}")}),
                            );
                            if self.llm.budget_exhausted() {
                                failure.class = "budget_exhausted".to_string();
                                failure.message = format!("{error:#}");
                                failure.budget_exhausted = true;
                                break 'novelty;
                            }
                        }
                    }
                }

                lock(&self.bandit).update_submitted(arm);
                match self.operator_for(&arm_name).propose(&context).await {
                    Ok(proposal) => {
                        api_cost += proposal.cost_usd;
                        *lock(&self.external_spend) += proposal.external_billed_usd.max(0.0);
                        lock(&self.bandit).update_cost(arm, proposal.cost_usd);
                        note(&mut arm_outcomes, arm, true, 0.0);
                        candidate = Some(Candidate {
                            sample,
                            proposal,
                            idea: context.idea,
                            system_prompt_id: system_prompt.map(|prompt| prompt.id),
                            arm,
                        });
                        break;
                    }
                    Err(error) => {
                        api_cost += error.cost_usd;
                        *lock(&self.external_spend) += error.external_billed_usd.max(0.0);
                        note(&mut arm_outcomes, arm, false, error.cost_usd);
                        let _ = fsutil::write_json(
                            &attempt_dir.join("failure.json"),
                            &json!({"class": error.class, "message": error.message, "arm": arm_name, "cost_usd": error.cost_usd}),
                        );
                        failure.class = error.class.clone();
                        failure.stage = "proposal".to_string();
                        failure.message = error.message.clone();
                        if error.budget_exhausted {
                            failure.budget_exhausted = true;
                            break 'novelty;
                        }
                        if error.endpoint_failure {
                            lock(&self.failed_arms).insert(arm);
                            self.events.warn(
                                "model_endpoint_failed",
                                json!({"generation": generation, "arm": arm_name, "error": error.message}),
                            );
                            match self.select_arm(&mut rng) {
                                Some((next, _)) => {
                                    arm = next;
                                    continue;
                                }
                                None => {
                                    failure.endpoint_failure = true;
                                    break 'novelty;
                                }
                            }
                        }
                        if !error.retryable {
                            break 'novelty;
                        }
                    }
                }
            }
            let Some(candidate) = candidate else { continue };

            let embedding = match &self.config.embedding_model {
                Some(model) => {
                    let text: String = candidate.proposal.code.chars().take(10_000).collect();
                    self.llm
                        .embed("embedding", model, &text)
                        .await
                        .unwrap_or_default()
                }
                None => Vec::new(),
            };
            let mut verdict = None;
            if let (Some(judge), false) = (&self.novelty, candidate.sample.needs_fix) {
                let assessed = judge
                    .assess(
                        &self.llm,
                        &self.db,
                        &candidate.proposal.code,
                        &embedding,
                        &candidate.sample.parent,
                        &mut rng,
                    )
                    .await;
                novelty_cost += assessed.cost_usd;
                if !assessed.accepted {
                    let _ = fsutil::write_json(
                        &gen_dir.join(format!("attempts/novelty_{novelty_attempt}/novelty.json")),
                        &assessed,
                    );
                    failure.class = "novelty_rejected".to_string();
                    failure.stage = "novelty".to_string();
                    failure.message = fsutil::truncate(&assessed.explanation, 600);
                    failure.idea = candidate.idea.clone();
                    continue 'novelty;
                }
                verdict = Some(assessed);
            }
            accepted = Some((candidate, embedding, verdict));
            break;
        }

        {
            let mut bandit = lock(&self.bandit);
            for (arm, delivered, failed_cost) in &arm_outcomes {
                bandit.report_outcome(*arm, *delivered, *failed_cost);
            }
        }

        let Some((candidate, embedding, novelty)) = accepted else {
            failure.cost_usd = api_cost + novelty_cost;
            let _ = fsutil::write_json(
                &gen_dir.join("failure.json"),
                &json!({
                    "generation": generation,
                    "failure_stage": failure.stage,
                    "failure_class": failure.class,
                    "message": failure.message,
                    "cost_usd": failure.cost_usd,
                }),
            );
            return GenerationOutcome::Failed(Box::new(failure));
        };

        let proposal_secs = started.elapsed().as_secs_f64();
        let outcome = self.evaluate(&gen_dir, &candidate).await;
        GenerationOutcome::Evaluated(Box::new(Evaluated {
            generation,
            gen_dir,
            candidate,
            embedding,
            novelty,
            outcome,
            api_cost,
            novelty_cost,
            posterior,
            meta_recommendation,
            proposal_secs,
        }))
    }

    /// Write the candidate and its provenance into the generation directory
    /// and run the task's evaluator on it.
    async fn evaluate(&self, gen_dir: &Path, candidate: &Candidate) -> EvalOutcome {
        let extension = genesis_core::task::extension_for_language(&self.task.language);
        let main_path = gen_dir.join(self.task.program_file_name());
        let _ = fsutil::write_text(&main_path, &candidate.proposal.code);
        let _ = fsutil::write_text(
            &gen_dir.join(format!("original.{extension}")),
            &candidate.sample.parent.code,
        );
        let _ = fsutil::write_text(&gen_dir.join("edit.diff"), &candidate.proposal.diff);
        let patch_file = if candidate.proposal.patch_type == "diff" {
            "search_replace.txt"
        } else {
            "rewrite.txt"
        };
        let _ = fsutil::write_text(&gen_dir.join(patch_file), &candidate.proposal.patch_text);
        self.evaluate_file(&main_path, &gen_dir.join("results"))
            .await
    }

    async fn evaluate_file(&self, program_path: &Path, results_dir: &Path) -> EvalOutcome {
        let timeout = self.config.eval_timeout_secs.map(Duration::from_secs);
        match self
            .task
            .evaluate(program_path, results_dir, None, timeout)
            .await
        {
            Ok(outcome) => outcome,
            Err(error) => {
                let diagnostic = format!("the evaluation could not be started: {error:#}");
                EvalOutcome {
                    evaluation_failed: true,
                    text_feedback: diagnostic.clone(),
                    diagnostic: Some(diagnostic),
                    results_dir: results_dir.to_path_buf(),
                    ..EvalOutcome::default()
                }
            }
        }
    }

    fn apply_outcome(&self, program: &mut Program, outcome: &EvalOutcome) {
        program.combined_score = outcome.combined_score;
        program.correct = outcome.correct;
        program.public_metrics = outcome.public_metrics.clone();
        program.private_metrics = outcome.private_metrics.clone();
        program.text_feedback = outcome.text_feedback.clone();
        let metadata = &mut program.metadata;
        let limit = self.config.max_log_chars;
        metadata.insert(
            "stdout_log".into(),
            json!(fsutil::tail(&outcome.stdout_tail, limit)),
        );
        metadata.insert(
            "stderr_log".into(),
            json!(fsutil::tail(&outcome.stderr_tail, limit)),
        );
        metadata.insert("evaluation_failed".into(), json!(outcome.evaluation_failed));
        metadata.insert("evaluation_seconds".into(), json!(outcome.runtime_secs));
        metadata.insert("evaluation_timed_out".into(), json!(outcome.timed_out));
        if let Some(error) = &outcome.error {
            metadata.insert("evaluation_error".into(), json!(error));
        }
        let analysis = genesis_edit::analyze_code_metrics(&program.code, &program.language);
        program.complexity = analysis.complexity_score;
        metadata.insert(
            "code_analysis_metrics".into(),
            serde_json::to_value(&analysis).unwrap_or(Value::Null),
        );
    }

    /// Replace `best/` with a copy of the best program's generation
    /// directory, via a staging directory so a reader never sees half of it.
    fn export_best(&self, best: &Program) {
        let source = self.results_dir.join(format!("gen_{}", best.generation));
        if !source.is_dir() {
            return;
        }
        let staging = self.results_dir.join(format!(
            ".best-staging-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        ));
        let ignore = vec!["attempts".to_string()];
        if fsutil::copy_tree(&source, &staging, &ignore).is_err() {
            let _ = std::fs::remove_dir_all(&staging);
            return;
        }
        let _ = fsutil::write_json(
            &staging.join("best_program.json"),
            &json!({
                "program_id": best.id,
                "generation": best.generation,
                "combined_score": best.combined_score,
                "public_metrics": best.public_metrics,
                "patch_name": best.meta_str("patch_name"),
            }),
        );
        let target = self.results_dir.join("best");
        let _ = std::fs::remove_dir_all(&target);
        if std::fs::rename(&staging, &target).is_err() {
            let _ = std::fs::remove_dir_all(&staging);
        }
    }

    fn save_bandit(&self) {
        let bandit = lock(&self.bandit);
        if let Err(error) =
            save_bandit_state(&self.results_dir.join("bandit_state.json"), bandit.as_ref())
        {
            self.events.warn(
                "bandit_state_not_saved",
                json!({"error": format!("{error:#}")}),
            );
        }
    }
}

impl EvolutionRunner {
    pub fn new(
        config: EvolutionConfig,
        db_config: DatabaseConfig,
        deps: RunnerDeps,
    ) -> Result<Self> {
        config.validate()?;
        db_config.validate()?;
        let agent_arms: Vec<&String> = config
            .llm_models
            .iter()
            .filter(|arm| EvolutionConfig::is_agent_arm(arm))
            .collect();
        anyhow::ensure!(
            agent_arms.is_empty() || deps.agent_operator.is_some(),
            "llm_models names {} but no agent operator is available",
            agent_arms
                .iter()
                .map(|arm| arm.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        std::fs::create_dir_all(&deps.results_dir)
            .with_context(|| format!("creating {}", deps.results_dir.display()))?;
        let seed = config.seed.unwrap_or_else(rand::random);
        let db_config = DatabaseConfig {
            seed: db_config.seed.or(Some(seed)),
            ..db_config
        };
        let db_path = deps.results_dir.join(&db_config.db_path);
        let db = Arc::new(ProgramDb::open(Some(&db_path), db_config)?);

        let language = deps.task.language.clone();
        let task_sys_msg = config
            .task_sys_msg
            .clone()
            .or_else(|| Some(deps.task.goal.clone()));
        let sampler = Arc::new(PromptSampler::new(
            task_sys_msg,
            &language,
            config.patch_types.clone(),
            config.patch_type_probs.clone(),
            config.use_text_feedback,
            &config.inspiration_sort_order,
        )?);
        let patch_operator: Arc<dyn Operator> = Arc::new(PatchOperator::new(
            deps.llm.clone(),
            Arc::clone(&sampler),
            config.llm_kwargs.clone(),
            config.max_patch_attempts,
        ));

        let mut bandit_kwargs = config.llm_dynamic_selection_kwargs.clone();
        if matches!(
            config.llm_dynamic_selection.as_deref(),
            Some("ucb") | Some("ucb1")
        ) {
            bandit_kwargs
                .entry("cost_aware_coef".to_string())
                .or_insert(json!(0.5));
        }
        let bandit = build_bandit(
            config.llm_dynamic_selection.as_deref(),
            config.llm_models.clone(),
            &Value::Object(bandit_kwargs),
        )?;

        let meta = match (config.meta_rec_interval, config.meta_llm_models.is_empty()) {
            (Some(interval), false) if interval > 0 => Some(Arc::new(MetaSummarizer::new(
                MetaConfig {
                    models: config.meta_llm_models.clone(),
                    kwargs: config.meta_llm_kwargs.clone(),
                    interval,
                    max_recommendations: config.meta_max_recommendations,
                    language: language.clone(),
                    use_text_feedback: config.use_text_feedback,
                },
                Some(&deps.results_dir),
            ))),
            _ => None,
        };
        let novelty = (!config.novelty_llm_models.is_empty() && config.embedding_model.is_some())
            .then(|| {
                NoveltyJudge::new(
                    config.novelty_llm_models.clone(),
                    config.novelty_llm_kwargs.clone(),
                    config.code_embed_sim_threshold,
                    &language,
                )
            });
        let prompt_evolver = if config.evolve_prompts {
            let prompt_db = Arc::new(PromptDb::open(
                Some(&deps.results_dir.join("prompts.sqlite")),
                PromptDbConfig {
                    archive_size: config.prompt_archive_size,
                    ucb_exploration_constant: config.prompt_ucb_exploration_constant,
                    epsilon: config.prompt_epsilon,
                    seed: Some(seed),
                    ..PromptDbConfig::default()
                },
            )?);
            let models = if config.prompt_llm_models.is_empty() {
                config
                    .llm_models
                    .iter()
                    .filter(|arm| !EvolutionConfig::is_agent_arm(arm))
                    .cloned()
                    .collect()
            } else {
                config.prompt_llm_models.clone()
            };
            Some(Arc::new(PromptEvolver::new(
                PromptEvolverConfig {
                    models,
                    kwargs: config.prompt_llm_kwargs.clone(),
                    patch_types: config.prompt_patch_types.clone(),
                    patch_type_probs: config.prompt_patch_type_probs.clone(),
                    evolution_interval: config.prompt_evolution_interval,
                    recompute_interval: config.prompt_percentile_recompute_interval,
                    top_k_programs: config.prompt_evo_top_k_programs,
                    language: language.clone(),
                    use_text_feedback: config.use_text_feedback,
                },
                prompt_db,
                sampler.task_sys_msg(),
            )?))
        } else {
            None
        };

        let spent_at_start = deps.llm.spent_usd();
        Ok(Self {
            shared: Arc::new(Shared {
                config,
                task: Arc::new(deps.task),
                results_dir: deps.results_dir,
                llm: deps.llm,
                events: deps.events,
                db,
                sampler,
                patch_operator,
                agent_operator: deps.agent_operator,
                idea_source: deps.idea_source,
                eval_hooks: deps.eval_hooks,
                bandit: Mutex::new(bandit),
                failed_arms: Mutex::new(HashSet::new()),
                external_spend: Mutex::new(0.0),
                meta,
                novelty,
                prompt_evolver,
                seed,
                cancel: deps.cancel,
            }),
            prior_cost_usd: deps.prior_cost_usd,
            spent_at_start,
        })
    }

    pub fn db(&self) -> &Arc<ProgramDb> {
        &self.shared.db
    }

    pub fn results_dir(&self) -> &Path {
        &self.shared.results_dir
    }

    /// Model spend of this run so far, earlier sessions included.
    pub fn cost_usd(&self) -> f64 {
        self.prior_cost_usd
            + (self.shared.llm.spent_usd() - self.spent_at_start)
            + *lock(&self.shared.external_spend)
    }

    fn budget_exhausted(&self) -> bool {
        self.shared.llm.budget_exhausted()
            || self
                .shared
                .config
                .max_api_costs
                .is_some_and(|budget| self.cost_usd() >= budget)
    }

    /// Whether another proposal fits in what is left of the budget, holding
    /// back the largest recent proposal cost for each one already in flight.
    fn affordable(&self, in_flight: usize, recent_costs: &VecDeque<f64>) -> bool {
        let Some(budget) = self.shared.config.max_api_costs else {
            return true;
        };
        let remaining = budget - self.cost_usd();
        if remaining <= 0.0 {
            return false;
        }
        let reserve = recent_costs.iter().copied().fold(0.0, f64::max);
        if reserve <= 0.0 {
            return in_flight == 0;
        }
        in_flight < (remaining / reserve * (1.0 + 1e-9)).floor() as usize
    }

    /// Evaluate and store the seed program, or pick up an existing run.
    async fn start(&self) -> Result<()> {
        let shared = &self.shared;
        let seed_program = shared
            .db
            .all()?
            .into_iter()
            .find(|program| program.generation == 0);
        if let Some(seed_program) = seed_program {
            let baseline = if seed_program.correct {
                seed_program.combined_score
            } else {
                0.0
            };
            let mut bandit = lock(&shared.bandit);
            bandit.set_baseline(baseline);
            let restored = load_bandit_state(
                &shared.results_dir.join("bandit_state.json"),
                bandit.as_mut(),
            )
            .unwrap_or(false);
            drop(bandit);
            shared.events.info(
                "run_resumed",
                json!({
                    "programs": shared.db.count()?,
                    "generations": shared.db.generations()?.len(),
                    "bandit_state_restored": restored,
                    "prior_cost_usd": self.prior_cost_usd,
                }),
            );
            return Ok(());
        }

        let code = shared.task.init_code()?;
        let gen_dir = shared.results_dir.join("gen_0");
        let main_path = gen_dir.join(shared.task.program_file_name());
        fsutil::write_text(&main_path, &code)?;
        let outcome = shared
            .evaluate_file(&main_path, &gen_dir.join("results"))
            .await;
        let mut program = Program::new(code, shared.task.language.clone());
        program
            .metadata
            .insert("patch_name".into(), json!("initial_program"));
        program.metadata.insert("patch_type".into(), json!("init"));
        program.metadata.insert(
            "patch_description".into(),
            json!("The task's seed program."),
        );
        shared.apply_outcome(&mut program, &outcome);
        if let Some(model) = &shared.config.embedding_model {
            let text: String = program.code.chars().take(10_000).collect();
            program.embedding = shared
                .llm
                .embed("embedding", model, &text)
                .await
                .unwrap_or_default();
        }
        shared.db.add(&mut program)?;
        lock(&shared.bandit).set_baseline(if program.correct {
            program.combined_score
        } else {
            0.0
        });
        if let Some(meta) = &shared.meta {
            meta.add_evaluated_program(&program);
        }
        if program.correct {
            shared.export_best(&program);
        }
        shared.events.info(
            "seed_evaluated",
            json!({
                "score": program.combined_score,
                "correct": program.correct,
                "evaluation_failed": outcome.evaluation_failed,
                "feedback": (!program.correct).then(|| fsutil::truncate(&program.text_feedback, 300)),
            }),
        );
        Ok(())
    }

    async fn persist(
        &self,
        evaluated: Evaluated,
        meta_task: &mut Option<tokio::task::JoinHandle<()>>,
    ) -> Result<Program> {
        let shared = &self.shared;
        let Evaluated {
            generation,
            gen_dir,
            candidate,
            embedding,
            novelty,
            outcome,
            api_cost,
            novelty_cost,
            posterior,
            meta_recommendation,
            proposal_secs,
        } = evaluated;
        let Candidate {
            sample,
            proposal,
            idea,
            system_prompt_id,
            arm,
        } = candidate;
        let arm_name = shared.config.llm_models[arm].clone();
        let parent = sample.parent;

        let mut program = Program::new(proposal.code.clone(), shared.task.language.clone());
        program.parent_id = Some(parent.id.clone());
        program.archive_inspiration_ids = sample
            .archive_inspirations
            .iter()
            .map(|p| p.id.clone())
            .collect();
        program.top_k_inspiration_ids = sample
            .top_k_inspirations
            .iter()
            .map(|p| p.id.clone())
            .collect();
        program.generation = generation;
        program.code_diff = Some(proposal.diff.clone());
        program.embedding = embedding;
        program.system_prompt_id = system_prompt_id;
        program.metadata = proposal.metadata.clone();
        let metadata = &mut program.metadata;
        metadata.insert("patch_name".into(), json!(proposal.name));
        metadata.insert("patch_description".into(), json!(proposal.description));
        metadata.insert("patch_type".into(), json!(proposal.patch_type));
        metadata.insert("num_applied".into(), json!(proposal.num_applied));
        metadata.insert("llm_calls".into(), json!(proposal.llm_calls));
        metadata.insert("arm".into(), json!(arm_name));
        metadata
            .entry("model_name".to_string())
            .or_insert(json!(arm_name));
        metadata.insert(
            "operator".into(),
            json!(if EvolutionConfig::is_agent_arm(&arm_name) {
                "agent"
            } else {
                "patch"
            }),
        );
        metadata.insert("api_costs".into(), json!(api_cost));
        metadata.insert("novelty_cost".into(), json!(novelty_cost));
        metadata.insert("embed_cost".into(), json!(0.0));
        metadata.insert("meta_cost".into(), json!(0.0));
        metadata.insert("parent_score".into(), json!(parent.combined_score));
        metadata.insert("proposal_seconds".into(), json!(proposal_secs));
        let posteriors: serde_json::Map<String, Value> = shared
            .config
            .llm_models
            .iter()
            .cloned()
            .zip(posterior.iter().map(|p| json!(p)))
            .collect();
        metadata.insert("model_posteriors".into(), Value::Object(posteriors));
        if let Some(recommendation) = &meta_recommendation {
            metadata.insert("meta_recommendations".into(), json!(recommendation));
        }
        if let Some(idea) = &idea {
            metadata.insert("idea".into(), serde_json::to_value(idea)?);
        }
        if let Some(verdict) = &novelty {
            metadata.insert("novelty".into(), serde_json::to_value(verdict)?);
        }
        shared.apply_outcome(&mut program, &outcome);

        let previous_best = shared.db.best(None)?;
        shared.db.add(&mut program)?;
        let is_new_best = shared
            .db
            .best(None)?
            .is_some_and(|best| best.id == program.id);

        let hook_input = EvaluatedProposal {
            generation,
            gen_dir,
            parent: Some(parent.clone()),
            idea,
            outcome: outcome.clone(),
            is_new_best,
        };
        let before_hooks = program.metadata.clone();
        for hook in &shared.eval_hooks {
            if let Err(error) = hook.after_evaluation(&mut program, &hook_input).await {
                shared.events.warn(
                    "eval_hook_failed",
                    json!({"generation": generation, "error": format!("{error:#}")}),
                );
            }
        }
        if program.metadata != before_hooks {
            shared.db.update_metadata(&program.id, &program.metadata)?;
        }

        lock(&shared.bandit).update(
            arm,
            program.correct.then_some(program.combined_score),
            Some(parent.combined_score),
        );

        if let Some(evolver) = &shared.prompt_evolver {
            let scores: Vec<f64> = shared
                .db
                .top(usize::MAX >> 1, true)?
                .iter()
                .map(|p| p.combined_score)
                .collect();
            if evolver.record(&program, &scores)? {
                let top = shared
                    .db
                    .top(shared.config.prompt_evo_top_k_programs.max(1), true)?;
                let scratchpad = shared.meta.as_ref().and_then(|meta| meta.scratchpad());
                let mut rng = ChaCha8Rng::seed_from_u64(shared.seed ^ u64::from(generation));
                match evolver.evolve(&shared.llm, &top, scratchpad.as_deref(), generation, &mut rng).await {
                    Ok((Some(prompt), cost)) => shared.events.info(
                        "prompt_evolved",
                        json!({"prompt_generation": prompt.generation, "name": prompt.name, "cost_usd": cost}),
                    ),
                    Ok((None, _)) => {}
                    Err(error) => shared.events.warn("prompt_evolution_failed", json!({"error": format!("{error:#}")})),
                }
            }
        }

        if let Some(meta) = &shared.meta {
            meta.add_evaluated_program(&program);
            let idle = meta_task.as_ref().is_none_or(|task| task.is_finished());
            if idle && meta.should_update() {
                let (meta, llm, events) = (
                    Arc::clone(meta),
                    shared.llm.clone(),
                    Arc::clone(&shared.events),
                );
                let best = shared.db.best(None)?;
                let seed = shared.seed ^ (u64::from(generation) << 32);
                *meta_task = Some(tokio::spawn(async move {
                    match meta.update(&llm, best.as_ref(), seed).await {
                        Ok(cost) => events.info(
                            "meta_updated",
                            json!({"cost_usd": cost, "pending": meta.pending()}),
                        ),
                        Err(error) => events
                            .warn("meta_update_failed", json!({"error": format!("{error:#}")})),
                    }
                }));
            }
        }

        if is_new_best {
            shared.export_best(&program);
        }
        let best_score = previous_best.as_ref().map(|best| best.combined_score);
        shared.events.info(
            "proposal_evaluated",
            json!({
                "generation": generation,
                "name": proposal.name,
                "patch_type": proposal.patch_type,
                "arm": arm_name,
                "score": program.combined_score,
                "correct": program.correct,
                "parent_score": parent.combined_score,
                "new_best": is_new_best,
                "best_before": best_score,
                "island": program.island_idx,
                "cost_usd": api_cost + novelty_cost,
                "eval_secs": outcome.runtime_secs,
                "feedback": (!program.correct).then(|| fsutil::truncate(&program.text_feedback, 200)),
            }),
        );
        Ok(program)
    }

    async fn record_failure(&self, failed: &FailedProposal) -> Result<()> {
        let shared = &self.shared;
        // An interrupted or budget-stopped proposal may be proposed again on
        // resume; any other failure settles its generation for good.
        let terminal = !failed.interrupted && !failed.budget_exhausted && !failed.endpoint_failure;
        shared.db.log_attempt(
            failed.generation,
            &failed.stage,
            "failed",
            &json!({
                "node_kind": "failed_proposal",
                "failure_class": failed.class,
                "message": fsutil::truncate(&failed.message, 2000),
                "cost_usd": failed.cost_usd,
                "terminal": terminal,
            }),
        )?;
        for hook in &shared.eval_hooks {
            hook.after_failure(
                failed.generation,
                failed.idea.as_ref(),
                &failed.class,
                &failed.message,
            )
            .await;
        }
        shared.events.warn(
            "proposal_failed",
            json!({
                "generation": failed.generation,
                "class": failed.class,
                "stage": failed.stage,
                "message": fsutil::truncate(&failed.message, 300),
                "cost_usd": failed.cost_usd,
            }),
        );
        Ok(())
    }

    /// Generations that will not be proposed (again): those with a program,
    /// and those whose proposal failed terminally.
    fn settled_generations(&self) -> Result<BTreeSet<u32>> {
        let mut settled = self.shared.db.generations()?;
        for attempt in self.shared.db.attempts()? {
            if attempt.status == "failed"
                && attempt.details.get("terminal").and_then(Value::as_bool) == Some(true)
            {
                settled.insert(attempt.generation);
            }
        }
        Ok(settled)
    }

    pub async fn run(&self) -> Result<RunSummary> {
        let started = Instant::now();
        let shared = &self.shared;
        let total = shared.config.num_generations;
        shared.events.info(
            "run_start",
            json!({
                "task": shared.task.name,
                "language": shared.task.language,
                "num_generations": total,
                "arms": shared.config.llm_models,
                "max_evaluation_jobs": shared.config.max_evaluation_jobs,
                "budget_usd": shared.config.max_api_costs,
                "results_dir": shared.results_dir.display().to_string(),
            }),
        );
        self.start().await?;

        let mut settled = self.settled_generations()?;
        let mut failed_proposals = settled.len().saturating_sub(shared.db.generations()?.len());
        let endpoint_failure_limit = if shared.config.llm_models.len() > 1 {
            3
        } else {
            1
        };
        let mut consecutive_endpoint_failures = 0;
        let mut recent_costs: VecDeque<f64> = VecDeque::new();
        let mut in_flight: JoinSet<GenerationOutcome> = JoinSet::new();
        let mut meta_task: Option<tokio::task::JoinHandle<()>> = None;
        // An evaluator that cannot score the seed would fail every proposal
        // the same way, each at the price of a model call.
        let seed_unscored = shared.db.all()?.iter().any(|program| {
            program.generation == 0
                && program.metadata.get("evaluation_failed") == Some(&json!(true))
        });
        let mut stop_reason: Option<&'static str> =
            seed_unscored.then_some("seed_evaluation_failed");
        let mut next_generation = 1u32;
        let mut stored_since_save = 0usize;

        loop {
            while stop_reason.is_none() && in_flight.len() < shared.config.max_evaluation_jobs {
                while next_generation < total && settled.contains(&next_generation) {
                    next_generation += 1;
                }
                if next_generation >= total {
                    break;
                }
                if shared.cancel.load(Ordering::SeqCst) {
                    stop_reason = Some("interrupted");
                    break;
                }
                if self.budget_exhausted() {
                    stop_reason = Some("budget_exhausted");
                    break;
                }
                if !self.affordable(in_flight.len(), &recent_costs) {
                    if in_flight.is_empty() {
                        stop_reason = Some("budget_exhausted");
                    }
                    break;
                }
                let generation = next_generation;
                next_generation += 1;
                settled.insert(generation);
                in_flight.spawn(Arc::clone(shared).run_generation(generation));
            }
            let Some(joined) = in_flight.join_next().await else {
                break;
            };
            let outcome = match joined {
                Ok(outcome) => outcome,
                Err(error) => {
                    shared
                        .events
                        .error("proposal_task_crashed", json!({"error": error.to_string()}));
                    failed_proposals += 1;
                    continue;
                }
            };
            match outcome {
                GenerationOutcome::Evaluated(evaluated) => {
                    recent_costs.push_back(evaluated.api_cost + evaluated.novelty_cost);
                    if recent_costs.len() > 50 {
                        recent_costs.pop_front();
                    }
                    consecutive_endpoint_failures = 0;
                    self.persist(*evaluated, &mut meta_task).await?;
                    stored_since_save += 1;
                    if stored_since_save >= 5 {
                        stored_since_save = 0;
                        shared.save_bandit();
                    }
                }
                GenerationOutcome::Failed(failed) => {
                    failed_proposals += 1;
                    if failed.cost_usd > 0.0 {
                        recent_costs.push_back(failed.cost_usd);
                    }
                    self.record_failure(&failed).await?;
                    if failed.interrupted {
                        stop_reason = stop_reason.or(Some("interrupted"));
                    } else if failed.budget_exhausted {
                        stop_reason = stop_reason.or(Some("budget_exhausted"));
                    } else if failed.endpoint_failure {
                        consecutive_endpoint_failures += 1;
                        if consecutive_endpoint_failures >= endpoint_failure_limit {
                            stop_reason = stop_reason.or(Some("llm_endpoint_error"));
                        }
                    } else {
                        consecutive_endpoint_failures = 0;
                    }
                }
            }
            if shared.cancel.load(Ordering::SeqCst) && stop_reason.is_none() {
                stop_reason = Some("interrupted");
                in_flight.abort_all();
            }
        }

        let stop_reason = stop_reason.unwrap_or("completed");
        if let Some(task) = meta_task.take() {
            if stop_reason == "interrupted" {
                task.abort();
            } else {
                let _ = tokio::time::timeout(Duration::from_secs(600), task).await;
            }
        }
        if let (Some(meta), "completed") = (&shared.meta, stop_reason) {
            if meta.pending() > 0 && !self.budget_exhausted() {
                let best = shared.db.best(None)?;
                if let Err(error) = meta.update(&shared.llm, best.as_ref(), shared.seed).await {
                    shared
                        .events
                        .warn("meta_update_failed", json!({"error": format!("{error:#}")}));
                }
            }
        }
        if let Some(evolver) = &shared.prompt_evolver {
            let scores: Vec<f64> = shared
                .db
                .top(usize::MAX >> 1, true)?
                .iter()
                .map(|p| p.combined_score)
                .collect();
            let _ = evolver.recompute(&scores);
        }
        shared.save_bandit();
        let _ = shared.db.checkpoint();

        let programs = shared.db.all()?;
        let own: Vec<&Program> = programs
            .iter()
            .filter(|program| {
                !["_is_island_copy", "_spawned_island"].iter().any(|flag| {
                    program
                        .metadata
                        .get(*flag)
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                })
            })
            .collect();
        let best = shared.db.best(None)?;
        if let Some(best) = &best {
            shared.export_best(best);
        }
        let evolved = own.iter().any(|program| program.generation > 0);
        let summary = RunSummary {
            num_generations: total,
            generations_completed: own
                .iter()
                .map(|program| program.generation)
                .collect::<BTreeSet<_>>()
                .len(),
            failed_proposals,
            programs: own.len(),
            correct_programs: own.iter().filter(|program| program.correct).count(),
            cost_usd: self.cost_usd(),
            stop_reason: stop_reason.to_string(),
            elapsed_secs: started.elapsed().as_secs_f64(),
            evolution_failed: matches!(
                stop_reason,
                "llm_endpoint_error" | "seed_evaluation_failed"
            ) || (total > 1 && !evolved && stop_reason == "completed"),
            results_dir: shared.results_dir.clone(),
            best,
        };
        fsutil::write_json(&shared.results_dir.join("run_summary.json"), &summary)?;
        shared.events.info(
            "run_end",
            json!({
                "stop_reason": summary.stop_reason,
                "best_score": summary.best.as_ref().map(|best| best.combined_score),
                "best_generation": summary.best.as_ref().map(|best| best.generation),
                "generations_completed": summary.generations_completed,
                "failed_proposals": summary.failed_proposals,
                "cost_usd": summary.cost_usd,
                "elapsed": fsutil::human_duration(summary.elapsed_secs),
            }),
        );
        Ok(summary)
    }
}
