//! Configuration of the evolution loop and of the program archive.

use genesis_core::JsonMap;
use genesis_llm::SamplingKwargs;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How proposals are generated, checked and budgeted.
///
/// Field names follow ShinkaEvolve's `EvolutionConfig`. An entry of
/// `llm_models` is either a model name, proposed for with one LLM call per
/// patch, or `agent:<cli>[@model]`, proposed for by a coding agent working in
/// a scratch copy of the parent. One bandit chooses among all of them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EvolutionConfig {
    /// Task description for the proposing model. Defaults to the task's
    /// `task.md`.
    pub task_sys_msg: Option<String>,
    pub patch_types: Vec<String>,
    pub patch_type_probs: Vec<f64>,
    /// Total generations including the seed (generation 0).
    pub num_generations: u32,
    /// Fresh parent-and-prompt draws per novelty attempt.
    pub max_patch_resamples: u32,
    /// Model turns within one draw; a failed patch's error is fed back.
    pub max_patch_attempts: u32,
    /// Times a proposal may be rejected as not novel before it is abandoned.
    pub max_novelty_attempts: u32,
    /// Proposals and evaluations in flight at once.
    pub max_evaluation_jobs: usize,
    pub llm_models: Vec<String>,
    /// `ucb`, `thompson`, `fixed`, or unset for uniform choice.
    pub llm_dynamic_selection: Option<String>,
    /// Bandit settings such as `cost_aware_coef`, `exploration_coef`, `epsilon`.
    pub llm_dynamic_selection_kwargs: JsonMap,
    pub llm_kwargs: SamplingKwargs,
    /// Evaluated programs between meta-scratchpad updates.
    pub meta_rec_interval: Option<u32>,
    /// Models for the meta scratchpad; empty turns it off.
    pub meta_llm_models: Vec<String>,
    pub meta_llm_kwargs: SamplingKwargs,
    pub meta_max_recommendations: u32,
    /// Show one random recommendation per proposal instead of the whole list.
    pub sample_single_meta_rec: bool,
    /// Embedding model for novelty and crowding; `local-hash` needs no key.
    pub embedding_model: Option<String>,
    /// Highest cosine similarity to an island member accepted without asking
    /// the novelty judge.
    pub code_embed_sim_threshold: f64,
    /// Models for the novelty judge; empty turns novelty rejection off.
    pub novelty_llm_models: Vec<String>,
    pub novelty_llm_kwargs: SamplingKwargs,
    /// Put evaluator text feedback into prompts.
    pub use_text_feedback: bool,
    /// Spending cap in USD for this run's model calls.
    pub max_api_costs: Option<f64>,
    /// `ascending`, `chronological` or `none`.
    pub inspiration_sort_order: String,
    pub evolve_prompts: bool,
    pub prompt_patch_types: Vec<String>,
    pub prompt_patch_type_probs: Vec<f64>,
    /// Evaluated programs between prompt mutations.
    pub prompt_evolution_interval: Option<u32>,
    pub prompt_archive_size: usize,
    /// Models that mutate prompts; empty falls back to `llm_models`.
    pub prompt_llm_models: Vec<String>,
    pub prompt_llm_kwargs: SamplingKwargs,
    pub prompt_ucb_exploration_constant: f64,
    pub prompt_epsilon: f64,
    pub prompt_evo_top_k_programs: usize,
    pub prompt_percentile_recompute_interval: u32,
    /// Overrides the task's evaluation time limit.
    pub eval_timeout_secs: Option<u64>,
    /// Characters of evaluator stdout and stderr kept with each program.
    pub max_log_chars: usize,
    pub seed: Option<u64>,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            task_sys_msg: None,
            patch_types: vec!["diff".into(), "full".into(), "cross".into()],
            patch_type_probs: vec![0.6, 0.3, 0.1],
            num_generations: 50,
            max_patch_resamples: 3,
            max_patch_attempts: 1,
            max_novelty_attempts: 3,
            max_evaluation_jobs: 4,
            llm_models: vec!["claude-sonnet-5-5".into(), "claude-haiku-5-5".into()],
            llm_dynamic_selection: Some("ucb".into()),
            llm_dynamic_selection_kwargs: JsonMap::new(),
            llm_kwargs: SamplingKwargs::default(),
            meta_rec_interval: Some(10),
            meta_llm_models: Vec::new(),
            meta_llm_kwargs: SamplingKwargs::default(),
            meta_max_recommendations: 5,
            sample_single_meta_rec: true,
            embedding_model: Some("local-hash".into()),
            code_embed_sim_threshold: 0.99,
            novelty_llm_models: Vec::new(),
            novelty_llm_kwargs: SamplingKwargs::default(),
            use_text_feedback: false,
            max_api_costs: None,
            inspiration_sort_order: "ascending".into(),
            evolve_prompts: false,
            prompt_patch_types: vec!["diff".into(), "full".into()],
            prompt_patch_type_probs: vec![0.7, 0.3],
            prompt_evolution_interval: None,
            prompt_archive_size: 10,
            prompt_llm_models: Vec::new(),
            prompt_llm_kwargs: SamplingKwargs::default(),
            prompt_ucb_exploration_constant: 1.0,
            prompt_epsilon: 0.1,
            prompt_evo_top_k_programs: 3,
            prompt_percentile_recompute_interval: 20,
            eval_timeout_secs: None,
            max_log_chars: 4000,
            seed: None,
        }
    }
}

impl EvolutionConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.num_generations >= 1,
            "num_generations must be at least 1"
        );
        anyhow::ensure!(!self.llm_models.is_empty(), "llm_models is empty");
        anyhow::ensure!(
            self.max_evaluation_jobs >= 1,
            "max_evaluation_jobs must be at least 1"
        );
        anyhow::ensure!(
            self.max_patch_resamples >= 1 && self.max_patch_attempts >= 1 && self.max_novelty_attempts >= 1,
            "max_patch_resamples, max_patch_attempts and max_novelty_attempts must each be at least 1"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.code_embed_sim_threshold),
            "code_embed_sim_threshold must be within [0, 1]"
        );
        if let Some(budget) = self.max_api_costs {
            anyhow::ensure!(budget > 0.0, "max_api_costs must be positive");
        }
        if self.evolve_prompts {
            anyhow::ensure!(
                self.prompt_patch_types.len() == self.prompt_patch_type_probs.len()
                    && !self.prompt_patch_types.is_empty()
                    && self.prompt_patch_types.iter().all(|t| t == "diff" || t == "full"),
                "prompt_patch_types must be a non-empty list of diff/full matching prompt_patch_type_probs"
            );
        }
        Ok(())
    }

    /// True for an `llm_models` entry served by a coding agent.
    pub fn is_agent_arm(arm: &str) -> bool {
        arm.starts_with("agent:")
    }
}

/// How the population is stored, partitioned and sampled.
///
/// Defaults follow current ShinkaEvolve: two islands, a global archive of 40,
/// weighted parent selection, one archive and one top-k inspiration, and
/// migration off until `migration_rate` is raised.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DatabaseConfig {
    /// File name of the run database inside the results directory.
    pub db_path: String,
    /// Seed for every random choice the database makes. `None` draws one.
    pub seed: Option<u64>,
    pub num_islands: usize,
    /// Capacity of the global archive. Zero disables the archive and with it
    /// archive and top-k inspirations.
    pub archive_size: usize,
    /// Share of archive inspirations taken from the island's elite.
    pub elite_selection_ratio: f64,
    pub num_archive_inspirations: usize,
    pub num_top_k_inspirations: usize,
    /// Generations between migrations.
    pub migration_interval: u32,
    /// Fraction of each island that moves per migration; zero turns it off.
    pub migration_rate: f64,
    /// Never migrate an island's best program.
    pub island_elitism: bool,
    /// Draw inspirations only from the parent's island.
    pub enforce_island_separation: bool,
    /// `uniform`, `equal`, `proportional` or `weighted`.
    pub island_selection_strategy: String,
    /// Spawn a new island when the best score stops improving.
    pub enable_dynamic_islands: bool,
    /// Generations without a new best before an island is spawned.
    pub stagnation_threshold: u32,
    /// What seeds a spawned island: `initial`, `best` or `archive_random`.
    pub island_spawn_strategy: String,
    pub island_spawn_subtree_size: usize,
    /// `weighted`, `power_law`, `beam_search`, `tournament`, `best_of_n`,
    /// `sequential`, or `hill_climb` (always the island's best program: the
    /// Agentic Scientist's "every attempt starts from the incumbent").
    pub parent_selection_strategy: String,
    /// Power-law exponent over score ranks: 0 is uniform, larger is greedier.
    pub exploitation_alpha: f64,
    /// Sigmoid sharpness of weighted selection.
    pub parent_selection_lambda: f64,
    /// Children a beam-search parent gets before the beam moves on.
    pub num_beams: usize,
    pub tournament_top_k: usize,
    pub tournament_num_draws: usize,
    /// `fitness` replaces the worst archive member; `crowding` replaces the
    /// most similar one.
    pub archive_selection_strategy: String,
    /// Signed weights over `combined_score`, `loc`, `lloc`, `complexity`,
    /// `maintainability` and `nesting`. More than one entry switches the
    /// archive to rank-normalised comparison.
    pub archive_criteria: BTreeMap<String, f64>,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            db_path: "programs.sqlite".to_string(),
            seed: None,
            num_islands: 2,
            archive_size: 40,
            elite_selection_ratio: 0.3,
            num_archive_inspirations: 1,
            num_top_k_inspirations: 1,
            migration_interval: 10,
            migration_rate: 0.0,
            island_elitism: true,
            enforce_island_separation: true,
            island_selection_strategy: "uniform".to_string(),
            enable_dynamic_islands: false,
            stagnation_threshold: 100,
            island_spawn_strategy: "initial".to_string(),
            island_spawn_subtree_size: 1,
            parent_selection_strategy: "weighted".to_string(),
            exploitation_alpha: 1.0,
            parent_selection_lambda: 10.0,
            num_beams: 5,
            tournament_top_k: 8,
            tournament_num_draws: 3,
            archive_selection_strategy: "fitness".to_string(),
            archive_criteria: BTreeMap::from([("combined_score".to_string(), 1.0)]),
        }
    }
}

impl DatabaseConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.num_islands >= 1, "num_islands must be at least 1");
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.migration_rate),
            "migration_rate must be within [0, 1]"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.elite_selection_ratio),
            "elite_selection_ratio must be within [0, 1]"
        );
        let known = |value: &str, allowed: &[&str], key: &str| -> anyhow::Result<()> {
            anyhow::ensure!(
                allowed.contains(&value),
                "{key} is `{value}`; expected one of: {}",
                allowed.join(", ")
            );
            Ok(())
        };
        known(
            &self.parent_selection_strategy,
            &[
                "weighted",
                "power_law",
                "beam_search",
                "tournament",
                "best_of_n",
                "sequential",
                "hill_climb",
            ],
            "parent_selection_strategy",
        )?;
        known(
            &self.island_selection_strategy,
            &["uniform", "equal", "proportional", "weighted"],
            "island_selection_strategy",
        )?;
        known(
            &self.island_spawn_strategy,
            &["initial", "best", "archive_random"],
            "island_spawn_strategy",
        )?;
        known(
            &self.archive_selection_strategy,
            &["fitness", "crowding"],
            "archive_selection_strategy",
        )?;
        if self.parent_selection_strategy == "tournament" {
            anyhow::ensure!(
                self.tournament_top_k >= 1
                    && self.tournament_num_draws >= 1
                    && self.tournament_num_draws <= self.tournament_top_k,
                "tournament needs 1 <= tournament_num_draws <= tournament_top_k"
            );
        }
        for key in self.archive_criteria.keys() {
            known(
                key,
                &[
                    "combined_score",
                    "loc",
                    "lloc",
                    "complexity",
                    "maintainability",
                    "nesting",
                ],
                "archive_criteria key",
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evolution_defaults_are_valid() {
        EvolutionConfig::default().validate().unwrap();
        assert!(EvolutionConfig {
            llm_models: vec![],
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(EvolutionConfig {
            max_api_costs: Some(0.0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(
            EvolutionConfig::is_agent_arm("agent:claude")
                && !EvolutionConfig::is_agent_arm("claude-sonnet-5-5")
        );
    }

    #[test]
    fn defaults_are_valid_and_bad_values_are_named() {
        DatabaseConfig::default().validate().unwrap();
        let bad = DatabaseConfig {
            parent_selection_strategy: "greedy".into(),
            ..DatabaseConfig::default()
        };
        let message = bad.validate().unwrap_err().to_string();
        assert!(
            message.contains("parent_selection_strategy") && message.contains("power_law"),
            "{message}"
        );
        let bad = DatabaseConfig {
            parent_selection_strategy: "tournament".into(),
            tournament_num_draws: 9,
            ..DatabaseConfig::default()
        };
        assert!(bad.validate().is_err());
        let bad = DatabaseConfig {
            migration_rate: 1.5,
            ..DatabaseConfig::default()
        };
        assert!(bad.validate().is_err());
    }
}
