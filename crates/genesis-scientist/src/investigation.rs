//! The investigation phase: the search for a better program.
//!
//! One loop serves three engines. `evolve` is ShinkaEvolve: a population on
//! islands, mutated by model-written patches. `scientist` is the Agentic
//! Scientist: every cycle states an idea with rubrics and starts from the
//! incumbent. `hybrid`, the default, keeps the population and the bandit
//! and adds the idea, the harness-checked rubrics, the memory and the
//! distilled conclusions to every proposal, whichever model or agent
//! implements it.

use crate::agent_operator::AgentOperator;
use crate::artifacts::{BestProgramManifest, InvestigationCompletion};
use crate::context::PhaseContext;
use crate::ideas::{IdeaConfig, LlmIdeaSource};
use crate::memory::{distill_conclusions, Memory, MemoryEvent};
use crate::rubrics::{judge_unassessed, verify_rubrics, RubricJudge};
use crate::selection::judge_investigation;
use anyhow::Result;
use async_trait::async_trait;
use genesis_core::{fsutil, IdeaBundle, Program};
use genesis_evolve::{
    DatabaseConfig, EvalHook, EvaluatedProposal, EvolutionConfig, EvolutionRunner, IdeaSource,
    Operator, RunSummary, RunnerDeps,
};
use genesis_llm::{AgentSpec, LlmClient, SamplingKwargs};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct InvestigationConfig {
    /// `hybrid`, `evolve` or `scientist`.
    pub engine: String,
    pub evolution: EvolutionConfig,
    pub database: DatabaseConfig,
    pub ideas: IdeaConfig,
    /// Models that judge implementation rubrics the harness cannot check
    /// from source patterns; empty leaves those rubrics unassessed.
    pub rubric_judge_models: Vec<String>,
    /// Models that maintain the distilled conclusions; empty falls back to
    /// the idea models, then the proposal models.
    pub conclusions_models: Vec<String>,
    /// Evaluated proposals between conclusion updates; 0 never updates.
    pub conclusions_interval: u32,
    pub judge_kwargs: SamplingKwargs,
    /// Wall-clock limit of one coding-agent attempt.
    pub agent_timeout_secs: u64,
    /// Custom agents by name, usable as `agent:<name>` in
    /// `evolution.llm_models`.
    pub agents: BTreeMap<String, AgentSpec>,
}

impl Default for InvestigationConfig {
    fn default() -> Self {
        Self {
            engine: "hybrid".to_string(),
            evolution: EvolutionConfig::default(),
            database: DatabaseConfig::default(),
            ideas: IdeaConfig::default(),
            rubric_judge_models: Vec::new(),
            conclusions_models: Vec::new(),
            conclusions_interval: 5,
            judge_kwargs: SamplingKwargs::default(),
            agent_timeout_secs: 1800,
            agents: BTreeMap::new(),
        }
    }
}

impl InvestigationConfig {
    /// The configuration with the engine's meaning applied.
    pub fn resolved(&self) -> Result<Self> {
        let mut config = self.clone();
        match self.engine.as_str() {
            "evolve" => {
                config.ideas.enabled = false;
                config.conclusions_interval = 0;
            }
            "hybrid" => config.ideas.enabled = true,
            "scientist" => {
                config.ideas.enabled = true;
                config.database.num_islands = 1;
                config.database.parent_selection_strategy = "hill_climb".to_string();
            }
            other => anyhow::bail!(
                "unknown investigation engine `{other}`; expected hybrid, evolve or scientist"
            ),
        }
        Ok(config)
    }

    fn text_models(&self) -> Vec<String> {
        self.evolution
            .llm_models
            .iter()
            .filter(|arm| !EvolutionConfig::is_agent_arm(arm))
            .cloned()
            .collect()
    }
}

/// Verifies rubrics, keeps the attempt history and the memory, and
/// maintains the distilled conclusions.
struct ScientistHook {
    investigation_dir: PathBuf,
    language: String,
    goal: String,
    memory: Arc<Memory>,
    llm: LlmClient,
    judge_models: Vec<String>,
    conclusions_models: Vec<String>,
    conclusions_interval: u32,
    kwargs: SamplingKwargs,
    evaluated: AtomicU32,
}

impl ScientistHook {
    fn record_attempt(&self, record: serde_json::Value) {
        let _ = fsutil::append_jsonl(
            &self.investigation_dir.join("attempts_history.jsonl"),
            &record,
        );
    }
}

#[async_trait]
impl EvalHook for ScientistHook {
    async fn after_evaluation(
        &self,
        program: &mut Program,
        evaluated: &EvaluatedProposal,
    ) -> Result<()> {
        let generation = evaluated.generation;
        let mut rng = ChaCha8Rng::seed_from_u64(u64::from(generation) ^ 0x5C1E);
        let name = program
            .meta_str("patch_name")
            .unwrap_or("unnamed")
            .to_string();
        let parent_score = evaluated
            .parent
            .as_ref()
            .map(|parent| parent.combined_score);
        let mut rubric_summary = String::new();
        let mut rubric_pass = None;
        let (idea_id, idea_dir_name) = match &evaluated.idea {
            Some(idea) => (idea.idea_id.clone(), Some(idea.dir_name(generation))),
            None => (name.clone(), None),
        };
        let attempt_id = format!("{idea_id}_attempt_01");

        if let Some(idea) = &evaluated.idea {
            let mut status = verify_rubrics(
                idea,
                &attempt_id,
                &program.code,
                &self.language,
                &evaluated.outcome,
            );
            if let Some(model) = self.judge_models.choose(&mut rng) {
                let judge = RubricJudge {
                    llm: &self.llm,
                    model,
                    kwargs: &self.kwargs,
                };
                judge_unassessed(
                    &mut status,
                    idea,
                    &program.code,
                    &self.language,
                    &judge,
                    &mut rng,
                )
                .await;
            }
            rubric_summary = format!(
                "; rubrics {}/{} passed",
                status.passed(),
                status.rubrics.len()
            );
            rubric_pass = Some(status.overall_pass);
            if let Some(dir_name) = &idea_dir_name {
                fsutil::write_json(
                    &self
                        .investigation_dir
                        .join("ideas")
                        .join(dir_name)
                        .join("rubric_status.json"),
                    &status,
                )?;
            }
            program
                .metadata
                .insert("rubric_status".into(), serde_json::to_value(&status)?);
        }

        let failure_reason = if !program.correct {
            Some(if evaluated.outcome.evaluation_failed {
                "process"
            } else {
                "correctness"
            })
        } else if rubric_pass == Some(false) {
            Some("rubrics")
        } else {
            None
        };
        self.record_attempt(json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "cycle_index": generation,
            "idea_id": idea_id,
            "idea_dir_name": idea_dir_name,
            "attempt_id": attempt_id,
            "attempt_number": 1,
            "attempt_status": if failure_reason.is_none() { "pass" } else { "fail" },
            "failure_reason": failure_reason,
            "rubric_overall_pass": rubric_pass,
            "best_updated": evaluated.is_new_best,
            "program_id": program.id,
            "experiment_dir": format!("gen_{generation}"),
            "agent": program.meta_str("arm"),
            "metrics": {"combined_score": program.combined_score, "correct": program.correct},
        }));

        let delta = parent_score
            .map(|parent| {
                format!(
                    " ({:+.6} against its parent)",
                    program.combined_score - parent
                )
            })
            .unwrap_or_default();
        let verdict = if evaluated.outcome.evaluation_failed {
            "the evaluation failed"
        } else if program.correct {
            "correct"
        } else {
            "correctness not established"
        };
        let mut event = MemoryEvent::new(
            "evaluate",
            "attempt",
            format!(
                "{name}: combined_score {:.6}{delta}, {verdict}{rubric_summary}{}. {}",
                program.combined_score,
                if evaluated.is_new_best {
                    "; new best"
                } else {
                    ""
                },
                fsutil::truncate(program.meta_str("patch_description").unwrap_or(""), 300)
            ),
            format!("gen_{generation}/results"),
            if evaluated.is_new_best {
                "build on this program"
            } else {
                "generate a new idea"
            },
        );
        event.idea_id = Some(idea_id);
        event.attempt_id = Some(attempt_id);
        event.program_id = Some(program.id.clone());
        event.metrics =
            Some(json!({"combined_score": program.combined_score, "correct": program.correct}));
        self.memory.append(event)?;

        let count = self.evaluated.fetch_add(1, Ordering::SeqCst) + 1;
        if self.conclusions_interval > 0 && count.is_multiple_of(self.conclusions_interval) {
            if let Some(model) = self.conclusions_models.choose(&mut rng) {
                let _ = distill_conclusions(
                    &self.memory,
                    &self.llm,
                    model,
                    &self.kwargs,
                    &self.goal,
                    generation,
                    &mut rng,
                )
                .await;
            }
        }
        Ok(())
    }

    async fn after_failure(
        &self,
        generation: u32,
        idea: Option<&IdeaBundle>,
        class: &str,
        message: &str,
    ) {
        let idea_id = idea
            .map(|idea| idea.idea_id.clone())
            .unwrap_or_else(|| format!("proposal_{generation}"));
        self.record_attempt(json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "cycle_index": generation,
            "idea_id": idea_id,
            "attempt_id": format!("{idea_id}_attempt_01"),
            "attempt_number": 1,
            "attempt_status": "fail",
            "failure_reason": "process",
            "failure_category": class,
            "best_updated": false,
        }));
        let mut event = MemoryEvent::new(
            "evaluate",
            "attempt",
            format!(
                "no program was evaluated ({class}): {}",
                fsutil::truncate(message, 300)
            ),
            format!("gen_{generation}/failure.json"),
            "generate a new idea; this is missing evidence, not a measured loss",
        );
        event.idea_id = Some(idea_id);
        let _ = self.memory.append(event);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InvestigationOutcome {
    pub summary: RunSummary,
    pub manifest: BestProgramManifest,
    /// `winner` or `completed_no_winner`.
    pub outcome: String,
    /// False when the search was interrupted or its endpoint failed, so the
    /// run can be resumed rather than read as finished.
    pub completed: bool,
}

/// Run, or resume, the search in `<run>/investigation`.
pub async fn run_investigation(
    context: &PhaseContext,
    config: &InvestigationConfig,
) -> Result<InvestigationOutcome> {
    let config = config.resolved()?;
    let dir = context.paths.investigation();
    std::fs::create_dir_all(&dir)?;
    let receipt_path = context.paths.completion_receipt();
    let finished_before = fsutil::read_json_opt::<InvestigationCompletion>(&receipt_path)
        .is_some_and(|receipt| receipt.status == "completed");
    if !finished_before {
        fsutil::write_json(
            &receipt_path,
            &InvestigationCompletion {
                schema_version: 1,
                status: "running".to_string(),
                ..Default::default()
            },
        )?;
    }

    let memory = Arc::new(Memory::open(&dir)?);
    let text_models = config.text_models();
    let idea_source: Option<Arc<dyn IdeaSource>> = config.ideas.enabled.then(|| {
        Arc::new(LlmIdeaSource::new(
            context.llm.clone(),
            config.ideas.clone(),
            text_models.clone(),
            &context.task.goal,
            &context.task.language,
            &dir,
            Arc::clone(&memory),
        )) as Arc<dyn IdeaSource>
    });
    let conclusions_models = [
        &config.conclusions_models,
        &config.ideas.models,
        &text_models,
    ]
    .into_iter()
    .find(|models| !models.is_empty())
    .cloned()
    .unwrap_or_default();
    let hook = Arc::new(ScientistHook {
        investigation_dir: dir.clone(),
        language: context.task.language.clone(),
        goal: context.task.goal.clone(),
        memory,
        llm: context.llm.clone(),
        judge_models: config.rubric_judge_models.clone(),
        conclusions_models,
        conclusions_interval: config.conclusions_interval,
        kwargs: config.judge_kwargs.clone(),
        evaluated: AtomicU32::new(0),
    });
    let ledger = context.llm.ledger();
    let prior_cost_usd = ledger
        .as_ref()
        .and_then(|ledger| {
            ledger
                .summary()
                .by_phase
                .get("investigation")
                .map(|bucket| bucket.cost_usd)
        })
        .unwrap_or(0.0);
    let agent_operator: Arc<dyn Operator> = Arc::new(AgentOperator::new(
        config.agents.clone(),
        config.agent_timeout_secs,
        ledger,
        "investigation",
    ));

    let runner = EvolutionRunner::new(
        config.evolution.clone(),
        config.database.clone(),
        RunnerDeps {
            task: context.task.clone(),
            results_dir: dir.clone(),
            llm: context.llm.clone(),
            events: Arc::clone(&context.events),
            agent_operator: Some(agent_operator),
            idea_source,
            eval_hooks: vec![hook],
            cancel: Arc::clone(&context.cancel),
            prior_cost_usd,
        },
    )?;
    let summary = runner.run().await?;

    let (manifest, outcome) = judge_investigation(&context.paths, &context.task, runner.db())?;
    fsutil::write_json(&dir.join("best_program_manifest.json"), &manifest)?;
    fsutil::write_json(
        &dir.join("outer_state.json"),
        &json!({
            "version": 1,
            "updated_at": chrono::Utc::now().to_rfc3339(),
            "current_cycle": summary.generations_completed.saturating_sub(1) + summary.failed_proposals,
            "max_idea_cycles": summary.num_generations.saturating_sub(1),
            "best_program_dir": manifest.program_path,
            "best_combined_score": manifest.combined_score,
            "best_correct": manifest.correct,
            "engine": config.engine,
            "stop_reason": summary.stop_reason,
        }),
    )?;
    let completed = matches!(
        summary.stop_reason.as_str(),
        "completed" | "budget_exhausted"
    );
    if completed {
        let digest = fsutil::sha256_hex(serde_json::to_string(&manifest)?.as_bytes());
        fsutil::write_json(
            &receipt_path,
            &InvestigationCompletion {
                schema_version: 1,
                status: "completed".to_string(),
                outcome: Some(outcome.clone()),
                manifest_sha256: Some(digest),
                stop_reason: Some(summary.stop_reason.clone()),
            },
        )?;
    }
    context.events.info(
        "investigation_end",
        json!({
            "outcome": outcome,
            "completed": completed,
            "best_score": manifest.combined_score,
            "best_source": manifest.source,
            "stop_reason": summary.stop_reason,
        }),
    );
    Ok(InvestigationOutcome {
        summary,
        manifest,
        outcome,
        completed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engines_set_ideas_and_selection() {
        let hybrid = InvestigationConfig::default().resolved().unwrap();
        assert!(hybrid.ideas.enabled);
        assert_eq!(hybrid.database.num_islands, 2);
        let evolve = InvestigationConfig {
            engine: "evolve".into(),
            ..Default::default()
        }
        .resolved()
        .unwrap();
        assert!(!evolve.ideas.enabled && evolve.conclusions_interval == 0);
        let scientist = InvestigationConfig {
            engine: "scientist".into(),
            ..Default::default()
        }
        .resolved()
        .unwrap();
        assert!(scientist.ideas.enabled);
        assert_eq!(
            (
                scientist.database.num_islands,
                scientist.database.parent_selection_strategy.as_str()
            ),
            (1, "hill_climb")
        );
        assert!(InvestigationConfig {
            engine: "random".into(),
            ..Default::default()
        }
        .resolved()
        .is_err());
        let mixed = InvestigationConfig {
            evolution: EvolutionConfig {
                llm_models: vec!["agent:claude".into(), "m".into()],
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(mixed.text_models(), ["m"]);
    }
}
