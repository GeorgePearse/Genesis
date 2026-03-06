use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

use crate::config::EvolutionConfig;
use crate::core::alma_memory::AlmaMemorySystem;
use crate::core::gepa_optimizer::GepaStyleOptimizer;
use crate::core::novelty_judge::NoveltyJudge;
use crate::core::sampler::PromptSampler;
use crate::core::summarizer::MetaSummarizer;
use crate::database::{build_database, ProgramDatabase};
use crate::launch::{build_scheduler, JobScheduler};
use crate::llm::{build_llm_client, LlmClient};
use crate::types::{PatchRequest, Program};

pub struct EvolutionRunner {
    pub cfg: EvolutionConfig,
    pub db: Box<dyn ProgramDatabase>,
    pub llm: Box<dyn LlmClient>,
    pub scheduler: Box<dyn JobScheduler>,
    pub prompt_sampler: PromptSampler,
    pub meta_summarizer: MetaSummarizer,
    pub novelty_judge: NoveltyJudge,
    pub alma_memory: AlmaMemorySystem,
    pub gepa_optimizer: GepaStyleOptimizer,
}

impl EvolutionRunner {
    pub fn new(cfg: EvolutionConfig) -> Self {
        let prompt_sampler = PromptSampler::new(
            cfg.task_sys_msg.clone(),
            cfg.language.clone(),
            cfg.patch_types.clone(),
            cfg.patch_type_probs.clone(),
            cfg.use_text_feedback,
        );

        let db = build_database(&cfg).unwrap_or_else(|_| build_database(&EvolutionConfig::default()).expect("default db"));
        let llm = build_llm_client(&cfg).unwrap_or_else(|_| build_llm_client(&EvolutionConfig::default()).expect("default llm"));
        let scheduler = build_scheduler(&cfg);

        let mut runner = Self {
            novelty_judge: NoveltyJudge::new(1.0),
            alma_memory: AlmaMemorySystem::new(
                cfg.alma_enabled,
                cfg.alma_max_entries,
                cfg.alma_max_retrievals,
                cfg.alma_min_success_delta,
            ),
            gepa_optimizer: GepaStyleOptimizer::new(
                cfg.gepa_enabled,
                cfg.gepa_num_fewshot_traces,
                cfg.gepa_max_traces,
                cfg.gepa_min_improvement,
                cfg.gepa_exploration_weight,
                cfg.gepa_candidate_instructions.clone(),
            ),
            cfg,
            db,
            llm,
            scheduler,
            prompt_sampler,
            meta_summarizer: MetaSummarizer::default(),
        };

        runner.restore_state();
        runner
    }

    pub fn run(&mut self) -> Result<()> {
        self.run_generation_0()?;

        for generation in 1..self.cfg.num_generations {
            self.run_generation(generation)?;
        }

        self.save_state()?;
        if let Some(best) = self.db.get_best_program()? {
            println!(
                "[rust-backend] best program gen={} score={:.4}",
                best.generation, best.combined_score
            );
        }
        Ok(())
    }

    fn run_generation_0(&mut self) -> Result<()> {
        let (sys, user) = self.prompt_sampler.initial_program_prompt();
        let resp = self.llm.query(&user, &sys)?;
        let eval = self.scheduler.run(&resp.content, 0)?;

        let program = Program {
            id: uuid(0),
            code: resp.content,
            language: self.cfg.language.clone(),
            parent_id: None,
            generation: 0,
            combined_score: eval.combined_score,
            correct: eval.correct,
            public_metrics: HashMap::new(),
            private_metrics: HashMap::new(),
            text_feedback: eval.text_feedback,
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        };

        self.db.add(program.clone())?;
        self.meta_summarizer.add_evaluated_program(program);
        Ok(())
    }

    fn run_generation(&mut self, generation: usize) -> Result<()> {
        let (parent, archive, top_k) = self
            .db
            .sample(generation)?
            .ok_or_else(|| anyhow::anyhow!("no parent available for generation {generation}"))?;

        let meta_recommendations = if self
            .meta_summarizer
            .should_update_meta(Some(self.cfg.max_parallel_jobs.max(1)))
        {
            self.meta_summarizer.update_meta_memory()
        } else {
            self.meta_summarizer.get_current()
        };

        let alma_context = self
            .alma_memory
            .build_prompt_context(generation, &parent.code, &parent.text_feedback);
        let gepa_ctx = self.gepa_optimizer.build_prompt_context();

        let req = PatchRequest {
            parent: parent.clone(),
            archive_inspirations: archive,
            top_k_inspirations: top_k,
            meta_recommendations,
            alma_memory_context: alma_context,
            gepa_instruction: gepa_ctx.candidate_instruction.clone(),
            gepa_fewshot_examples: gepa_ctx.fewshot_examples.clone(),
        };

        let (patch_sys, patch_msg, patch_type) = self.prompt_sampler.sample(&req);
        let resp = self.llm.query(&patch_msg, &patch_sys)?;
        let eval = self.scheduler.run(&resp.content, generation)?;

        if !self.novelty_judge.should_accept(0.0) {
            return Ok(());
        }

        let mut metadata = HashMap::new();
        metadata.insert(
            "patch_type".to_string(),
            serde_json::Value::String(patch_type.clone()),
        );
        if let Some(idx) = gepa_ctx.candidate_id {
            metadata.insert(
                "gepa_candidate_id".to_string(),
                serde_json::Value::Number(idx.into()),
            );
        }

        let program = Program {
            id: uuid(generation),
            code: resp.content,
            language: self.cfg.language.clone(),
            parent_id: Some(parent.id.clone()),
            generation,
            combined_score: eval.combined_score,
            correct: eval.correct,
            public_metrics: HashMap::new(),
            private_metrics: HashMap::new(),
            text_feedback: eval.text_feedback.clone(),
            metadata,
            timestamp: Utc::now(),
        };

        self.db.add(program.clone())?;
        self.meta_summarizer.add_evaluated_program(program.clone());

        let delta = program.combined_score - parent.combined_score;
        self.gepa_optimizer.observe_result(
            generation,
            parent.combined_score,
            program.combined_score,
            &patch_type,
            "rust-backend-patch",
            "Rust backend generated mutation",
            "diff summary unavailable",
            gepa_ctx.candidate_id,
            program.correct,
        );
        self.alma_memory.observe_outcome(
            generation,
            parent.combined_score,
            program.combined_score,
            program.correct,
            &patch_type,
            "rust-backend-patch",
            "Rust backend generated mutation",
            "diff summary unavailable",
            &program.text_feedback,
            "",
        );

        println!(
            "[rust-backend] gen={} parent={:.4} child={:.4} delta={:+.4}",
            generation, parent.combined_score, program.combined_score, delta
        );

        self.save_state()?;
        Ok(())
    }

    fn save_state(&self) -> Result<()> {
        self.alma_memory.save_state("alma_memory.json")?;
        self.gepa_optimizer.save_state("gepa_state.json")?;
        Ok(())
    }

    fn restore_state(&mut self) {
        let _ = self.alma_memory.load_state("alma_memory.json");
        let _ = self.gepa_optimizer.load_state("gepa_state.json");
    }
}

fn uuid(generation: usize) -> String {
    format!(
        "rust-gen-{}-{}",
        generation,
        Utc::now().timestamp_nanos_opt().unwrap_or_default()
    )
}
