//! Co-evolution of the task system prompt: each proposal is written under a
//! prompt drawn from an archive, the prompt is credited with how its
//! programs rank, and every so often a model rewrites a good prompt in the
//! light of the best programs so far.

use crate::db::prompts::{PromptDb, SystemPrompt};
use crate::prompts::{self, fill};
use anyhow::Result;
use genesis_core::Program;
use genesis_llm::{extract_tag, LlmClient, LlmRequest, SamplingKwargs};
use rand::seq::SliceRandom;
use rand::Rng;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PromptEvolverConfig {
    pub models: Vec<String>,
    pub kwargs: SamplingKwargs,
    pub patch_types: Vec<String>,
    pub patch_type_probs: Vec<f64>,
    /// Evaluated programs between mutations; `None` never mutates.
    pub evolution_interval: Option<u32>,
    pub recompute_interval: u32,
    pub top_k_programs: usize,
    pub language: String,
    pub use_text_feedback: bool,
}

/// A rewritten prompt shorter than this is treated as a failed mutation.
const MIN_PROMPT_CHARS: usize = 50;

pub struct PromptEvolver {
    config: PromptEvolverConfig,
    db: Arc<PromptDb>,
    since_evolution: AtomicU32,
    correct_since_recompute: AtomicU32,
}

impl PromptEvolver {
    /// Make sure the archive holds the task's own prompt as generation 0.
    pub fn new(config: PromptEvolverConfig, db: Arc<PromptDb>, task_sys_msg: &str) -> Result<Self> {
        if db.all()?.is_empty() {
            let mut seed = SystemPrompt::new(task_sys_msg, "init");
            seed.name = Some("task_prompt".to_string());
            db.add(&mut seed)?;
        }
        Ok(Self {
            config,
            db,
            since_evolution: AtomicU32::new(0),
            correct_since_recompute: AtomicU32::new(0),
        })
    }

    pub fn db(&self) -> &Arc<PromptDb> {
        &self.db
    }

    /// The prompt to write the next proposal under.
    pub fn sample(&self) -> Option<SystemPrompt> {
        self.db.sample().ok().flatten()
    }

    /// Credit the program's prompt and report whether a mutation is due.
    pub fn record(&self, program: &Program, all_correct_scores: &[f64]) -> Result<bool> {
        let Some(prompt_id) = &program.system_prompt_id else {
            return Ok(false);
        };
        self.db.record_program(
            prompt_id,
            &program.id,
            program.combined_score,
            program.correct,
            all_correct_scores,
        )?;
        if program.correct {
            let correct = self.correct_since_recompute.fetch_add(1, Ordering::SeqCst) + 1;
            if self.config.recompute_interval > 0 && correct >= self.config.recompute_interval {
                self.correct_since_recompute.store(0, Ordering::SeqCst);
                self.db.recompute_fitness(all_correct_scores)?;
            }
        }
        let Some(interval) = self
            .config
            .evolution_interval
            .filter(|interval| *interval > 0)
        else {
            return Ok(false);
        };
        let seen = self.since_evolution.fetch_add(1, Ordering::SeqCst) + 1;
        if seen >= interval {
            self.since_evolution.store(0, Ordering::SeqCst);
            return Ok(true);
        }
        Ok(false)
    }

    pub fn recompute(&self, all_correct_scores: &[f64]) -> Result<()> {
        self.db.recompute_fitness(all_correct_scores)
    }

    /// Ask a model to improve a sampled prompt, given the top programs and
    /// the meta scratchpad. Returns the stored prompt and the call's cost;
    /// `None` when the reply held no usable prompt.
    pub async fn evolve(
        &self,
        llm: &LlmClient,
        top_programs: &[Program],
        scratchpad: Option<&str>,
        program_generation: u32,
        rng: &mut (dyn rand::RngCore + Send),
    ) -> Result<(Option<SystemPrompt>, f64)> {
        let Some(parent) = self.db.sample()? else {
            return Ok((None, 0.0));
        };
        let mut point = rng.gen::<f64>() * self.config.patch_type_probs.iter().sum::<f64>();
        let mut patch_type = self
            .config
            .patch_types
            .last()
            .cloned()
            .unwrap_or_else(|| "diff".to_string());
        for (candidate, probability) in self
            .config
            .patch_types
            .iter()
            .zip(&self.config.patch_type_probs)
        {
            if point < *probability {
                patch_type = candidate.clone();
                break;
            }
            point -= probability;
        }
        let (system, template) = if patch_type == "full" {
            (
                prompts::PROMPT_EVO_FULL_SYSTEM,
                prompts::PROMPT_EVO_FULL_USER,
            )
        } else {
            (
                prompts::PROMPT_EVO_DIFF_SYSTEM,
                prompts::PROMPT_EVO_DIFF_USER,
            )
        };
        let shown: Vec<Program> = top_programs
            .iter()
            .take(self.config.top_k_programs)
            .cloned()
            .collect();
        let user = fill(
            template,
            &[
                ("current_prompt", parent.prompt_text.as_str()),
                (
                    "global_scratchpad_section",
                    prompts::global_scratchpad_section(scratchpad).as_str(),
                ),
                (
                    "top_programs",
                    prompts::top_programs_block(
                        &shown,
                        &self.config.language,
                        self.config.use_text_feedback,
                    )
                    .as_str(),
                ),
            ],
        );
        let model = self
            .config
            .models
            .choose(rng)
            .ok_or_else(|| anyhow::anyhow!("no prompt-evolution model"))?;
        let mut request = LlmRequest::new(model.clone(), system, user);
        self.config.kwargs.apply(&mut request, rng);
        let response = llm.query("prompt_evolution", request).await?;
        let text = extract_tag(&response.content, "PROMPT")
            .unwrap_or_else(|| response.content.trim().to_string());
        if text.chars().count() < MIN_PROMPT_CHARS {
            return Ok((None, response.cost_usd));
        }
        let mut evolved = SystemPrompt::new(text, &patch_type);
        evolved.name = extract_tag(&response.content, "NAME");
        evolved.description = extract_tag(&response.content, "DESCRIPTION");
        evolved.parent_id = Some(parent.id);
        evolved.generation = self.db.next_generation()?;
        evolved.program_generation = program_generation;
        evolved.metadata.insert(
            "llm".into(),
            serde_json::json!({"model": response.model, "cost": response.cost_usd}),
        );
        self.db.add(&mut evolved)?;
        Ok((Some(evolved), response.cost_usd))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::prompts::PromptDbConfig;
    use genesis_llm::MockProvider;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn evolver(interval: Option<u32>, types: &[&str], probs: &[f64]) -> PromptEvolver {
        let db = Arc::new(
            PromptDb::open(
                None,
                PromptDbConfig {
                    seed: Some(1),
                    epsilon: 0.0,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        PromptEvolver::new(
            PromptEvolverConfig {
                models: vec!["prompt-model".into()],
                kwargs: SamplingKwargs::default(),
                patch_types: types.iter().map(|t| t.to_string()).collect(),
                patch_type_probs: probs.to_vec(),
                evolution_interval: interval,
                recompute_interval: 2,
                top_k_programs: 1,
                language: "python".into(),
                use_text_feedback: false,
            },
            db,
            "Pack circles as densely as you can.",
        )
        .unwrap()
    }

    fn program(score: f64, prompt_id: &str) -> Program {
        let mut program = Program::new("code", "python");
        program.combined_score = score;
        program.correct = true;
        program.system_prompt_id = Some(prompt_id.to_string());
        program
    }

    #[tokio::test]
    async fn fitness_is_credited_and_mutations_come_due_on_the_interval() {
        let evolver = evolver(Some(2), &["diff"], &[1.0]);
        let seed = evolver.sample().unwrap();
        assert_eq!((seed.patch_type.as_str(), seed.generation), ("init", 0));
        assert!(!evolver.record(&program(1.0, &seed.id), &[1.0]).unwrap());
        assert!(evolver
            .record(&program(2.0, &seed.id), &[1.0, 2.0])
            .unwrap());
        assert!(!evolver
            .record(&program(3.0, &seed.id), &[1.0, 2.0, 3.0])
            .unwrap());
        let mut unattributed = program(9.0, &seed.id);
        unattributed.system_prompt_id = None;
        assert!(!evolver.record(&unattributed, &[9.0]).unwrap());
        let stored = evolver.db().get(&seed.id).unwrap().unwrap();
        assert_eq!(stored.correct_program_count, 3);
        assert!(stored.fitness > 0.0 && stored.fitness <= 1.0);

        let never = super::tests::evolver(None, &["diff"], &[1.0]);
        let seed = never.sample().unwrap();
        for _ in 0..5 {
            assert!(!never.record(&program(1.0, &seed.id), &[1.0]).unwrap());
        }
    }

    #[tokio::test]
    async fn a_mutation_stores_a_child_prompt_and_short_replies_are_dropped() {
        let evolver = evolver(Some(1), &["full"], &[1.0]);
        let rewritten = "Pack circles using simulated annealing with restarts and a final local refinement step.";
        let provider = MockProvider::scripted(vec![
            format!("<NAME>annealing_focus</NAME>\n<DESCRIPTION>Top programs anneal.</DESCRIPTION>\n<PROMPT>\n{rewritten}\n</PROMPT>"),
            "<PROMPT>too short</PROMPT>".to_string(),
        ]);
        let llm = LlmClient::mock(provider.clone());
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let top = [program(2.6, "x"), program(2.5, "x")];
        let (evolved, _) = evolver
            .evolve(&llm, &top, Some("restarts help"), 12, &mut rng)
            .await
            .unwrap();
        let evolved = evolved.unwrap();
        assert_eq!(evolved.prompt_text, rewritten);
        assert_eq!(evolved.name.as_deref(), Some("annealing_focus"));
        assert_eq!(
            (
                evolved.patch_type.as_str(),
                evolved.generation,
                evolved.program_generation
            ),
            ("full", 1, 12)
        );
        assert!(evolved.parent_id.is_some() && evolved.in_archive);

        let call = &provider.calls()[0];
        assert_eq!(call.stage, "prompt_evolution");
        assert_eq!(call.system, prompts::PROMPT_EVO_FULL_SYSTEM);
        let user = call.last_user();
        assert!(
            user.contains("# Current System Prompt\n```\nPack circles as densely as you can.\n```")
        );
        assert!(
            user.contains("# Global Insights from Meta-Review") && user.contains("restarts help")
        );
        assert!(
            user.contains("## Program 1") && !user.contains("## Program 2"),
            "top_k_programs caps the examples"
        );

        let (dropped, _) = evolver
            .evolve(&llm, &top, None, 13, &mut rng)
            .await
            .unwrap();
        assert!(dropped.is_none());
        assert_eq!(evolver.db().all().unwrap().len(), 2);
    }
}
