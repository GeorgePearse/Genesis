use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EvolutionConfig {
    pub task_sys_msg: Option<String>,
    pub patch_types: Vec<String>,
    pub patch_type_probs: Vec<f64>,
    pub num_generations: usize,
    pub max_parallel_jobs: usize,
    pub max_patch_resamples: usize,
    pub max_patch_attempts: usize,
    pub language: String,
    pub use_text_feedback: bool,

    pub db_backend: String,
    pub clickhouse_url: Option<String>,
    pub clickhouse_user: Option<String>,
    pub clickhouse_password: Option<String>,
    pub clickhouse_database: String,

    pub llm_backend: String,
    pub openai_base_url: String,
    pub openai_model: String,
    pub openai_api_key: Option<String>,

    pub scheduler_backend: String,
    pub eval_command: Option<String>,

    pub alma_enabled: bool,
    pub alma_max_entries: usize,
    pub alma_max_retrievals: usize,
    pub alma_min_success_delta: f64,

    pub gepa_enabled: bool,
    pub gepa_num_fewshot_traces: usize,
    pub gepa_max_traces: usize,
    pub gepa_min_improvement: f64,
    pub gepa_exploration_weight: f64,
    pub gepa_candidate_instructions: Option<Vec<String>>,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            task_sys_msg: None,
            patch_types: vec!["diff".to_string(), "full".to_string()],
            patch_type_probs: vec![0.5, 0.5],
            num_generations: 20,
            max_parallel_jobs: 1,
            max_patch_resamples: 3,
            max_patch_attempts: 5,
            language: "python".to_string(),
            use_text_feedback: false,

            db_backend: "in_memory".to_string(),
            clickhouse_url: std::env::var("CLICKHOUSE_URL").ok(),
            clickhouse_user: std::env::var("CLICKHOUSE_USER").ok(),
            clickhouse_password: std::env::var("CLICKHOUSE_PASSWORD").ok(),
            clickhouse_database: std::env::var("CLICKHOUSE_DB")
                .unwrap_or_else(|_| "default".to_string()),

            llm_backend: "mock".to_string(),
            openai_base_url: "https://api.openai.com".to_string(),
            openai_model: "gpt-4.1-mini".to_string(),
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),

            scheduler_backend: "mock".to_string(),
            eval_command: None,

            alma_enabled: true,
            alma_max_entries: 256,
            alma_max_retrievals: 4,
            alma_min_success_delta: 0.0,
            gepa_enabled: true,
            gepa_num_fewshot_traces: 3,
            gepa_max_traces: 64,
            gepa_min_improvement: 0.0,
            gepa_exploration_weight: 1.1,
            gepa_candidate_instructions: None,
        }
    }
}

impl EvolutionConfig {
    pub fn from_yaml_file(path: &str) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("failed reading config file: {path}"))?;

        #[derive(Debug, Deserialize)]
        struct Root {
            evo_config: Option<EvolutionConfig>,
        }

        if let Ok(parsed) = serde_yaml::from_str::<Root>(&raw) {
            if let Some(cfg) = parsed.evo_config {
                return Ok(cfg.with_env_fallbacks());
            }
        }

        let cfg = serde_yaml::from_str::<EvolutionConfig>(&raw)
            .with_context(|| format!("failed parsing YAML config: {path}"))?;
        Ok(cfg.with_env_fallbacks())
    }

    fn with_env_fallbacks(mut self) -> Self {
        if self.openai_api_key.is_none() {
            self.openai_api_key = std::env::var("OPENAI_API_KEY").ok();
        }
        if self.clickhouse_url.is_none() {
            self.clickhouse_url = std::env::var("CLICKHOUSE_URL").ok();
        }
        if self.clickhouse_user.is_none() {
            self.clickhouse_user = std::env::var("CLICKHOUSE_USER").ok();
        }
        if self.clickhouse_password.is_none() {
            self.clickhouse_password = std::env::var("CLICKHOUSE_PASSWORD").ok();
        }
        self
    }
}
