//! The meta scratchpad: every few evaluations, summarise what was tried,
//! distil which patterns worked, and turn that into recommendations that
//! steer later proposals.

use crate::prompts::{self, fill};
use anyhow::Result;
use futures::future::join_all;
use genesis_core::{fsutil, Program};
use genesis_llm::{LlmClient, LlmRequest, SamplingKwargs};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// `meta_memory.json`, with ShinkaEvolve's keys.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct MetaState {
    pub unprocessed_programs: Vec<Program>,
    pub meta_summary: Option<String>,
    pub meta_scratch_pad: Option<String>,
    pub meta_recommendations: Option<String>,
    pub meta_recommendations_history: Vec<String>,
    pub total_programs_meta_processed: u32,
}

#[derive(Debug, Clone)]
pub struct MetaConfig {
    pub models: Vec<String>,
    pub kwargs: SamplingKwargs,
    pub interval: u32,
    pub max_recommendations: u32,
    pub language: String,
    pub use_text_feedback: bool,
}

pub struct MetaSummarizer {
    config: MetaConfig,
    results_dir: Option<PathBuf>,
    state: Mutex<MetaState>,
    update_guard: tokio::sync::Mutex<()>,
}

/// The items of a numbered list, without their numbers.
pub fn split_recommendations(text: &str) -> Vec<String> {
    static ITEM: OnceLock<Regex> = OnceLock::new();
    let item = ITEM.get_or_init(|| Regex::new(r"(?m)^\s*\d+\.\s+").expect("valid regex"));
    let starts: Vec<(usize, usize)> = item
        .find_iter(text)
        .map(|found| (found.start(), found.end()))
        .collect();
    let mut items = Vec::new();
    for (index, (_, end)) in starts.iter().enumerate() {
        let stop = starts.get(index + 1).map_or(text.len(), |next| next.0);
        let body = text[*end..stop].trim();
        if !body.is_empty() {
            items.push(body.to_string());
        }
    }
    items
}

impl MetaSummarizer {
    /// Load `meta_memory.json` from `results_dir` when it exists.
    pub fn new(config: MetaConfig, results_dir: Option<&Path>) -> Self {
        let state = results_dir
            .and_then(|dir| fsutil::read_json_opt::<MetaState>(&dir.join("meta_memory.json")))
            .unwrap_or_default();
        Self {
            config,
            results_dir: results_dir.map(Path::to_path_buf),
            state: Mutex::new(state),
            update_guard: tokio::sync::Mutex::new(()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MetaState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn save(&self, state: &MetaState) {
        if let Some(dir) = &self.results_dir {
            let _ = fsutil::write_json(&dir.join("meta_memory.json"), state);
        }
    }

    pub fn state(&self) -> MetaState {
        self.lock().clone()
    }

    /// Queue an evaluated program for the next update. Embeddings are
    /// dropped: the summariser reads code and metrics only.
    pub fn add_evaluated_program(&self, program: &Program) {
        let mut slim = program.clone();
        slim.embedding = Vec::new();
        let mut state = self.lock();
        state.unprocessed_programs.push(slim);
        self.save(&state);
    }

    pub fn should_update(&self) -> bool {
        self.config.interval > 0
            && self.lock().unprocessed_programs.len() >= self.config.interval as usize
    }

    pub fn pending(&self) -> usize {
        self.lock().unprocessed_programs.len()
    }

    pub fn scratchpad(&self) -> Option<String> {
        self.lock().meta_scratch_pad.clone()
    }

    /// The guidance for one proposal: one random recommendation, or the
    /// whole list.
    pub fn recommendation(&self, single: bool, rng: &mut dyn rand::RngCore) -> Option<String> {
        let text = self.lock().meta_recommendations.clone()?;
        if !single {
            return Some(text);
        }
        let items = split_recommendations(&text);
        items.choose(rng).cloned().or(Some(text))
    }

    async fn ask(
        &self,
        llm: &LlmClient,
        seed: u64,
        system: &str,
        user: String,
    ) -> Result<(String, f64)> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let model = self
            .config
            .models
            .choose(&mut rng)
            .ok_or_else(|| anyhow::anyhow!("no meta model configured"))?;
        let mut request = LlmRequest::new(model.clone(), system, user);
        self.config.kwargs.apply(&mut request, &mut rng);
        let response = llm.query("meta", request).await?;
        Ok((response.content, response.cost_usd))
    }

    /// Run the three steps over everything queued. Returns the model cost.
    /// A program whose summary call fails stays queued for the next update.
    pub async fn update(&self, llm: &LlmClient, best: Option<&Program>, seed: u64) -> Result<f64> {
        let _exclusive = self.update_guard.lock().await;
        let queued = self.lock().unprocessed_programs.clone();
        if queued.is_empty() {
            return Ok(0.0);
        }
        let language = self.config.language.as_str();
        let feedback = self.config.use_text_feedback;
        let mut cost = 0.0;

        let summaries = join_all(queued.iter().enumerate().map(|(index, program)| {
            let user = fill(
                prompts::META_STEP1_USER_MSG,
                &[(
                    "individual_program_msg",
                    prompts::individual_program_msg(program, language, feedback).as_str(),
                )],
            );
            async move {
                self.ask(
                    llm,
                    seed.wrapping_add(index as u64),
                    prompts::META_STEP1_SYSTEM_MSG,
                    user,
                )
                .await
            }
        }))
        .await;
        let mut summarised: Vec<(&Program, String)> = Vec::new();
        for (program, result) in queued.iter().zip(summaries) {
            if let Ok((summary, call_cost)) = result {
                cost += call_cost;
                if !summary.trim().is_empty() {
                    let identified = format!(
                        "{}\n**Program Identifier:** Generation {} - Patch Name {} - Correct Program: {}",
                        summary.trim(),
                        program.generation,
                        program.meta_str("patch_name").unwrap_or("unknown"),
                        if program.correct { "True" } else { "False" }
                    );
                    summarised.push((program, identified));
                }
            }
        }
        if summarised.is_empty() {
            anyhow::bail!("no program could be summarised");
        }
        summarised.sort_by_key(|(program, _)| program.generation);
        let new_summaries: Vec<&str> = summarised
            .iter()
            .map(|(_, summary)| summary.as_str())
            .collect();
        let new_summaries = new_summaries.join("\n\n");

        let previous = self.state();
        let best_info = best
            .map(|program| prompts::individual_program_msg(program, language, feedback))
            .unwrap_or_else(|| "None".to_string());
        let (insights, insights_cost) = self
            .ask(
                llm,
                seed.wrapping_add(10_001),
                prompts::META_STEP2_SYSTEM_MSG,
                fill(
                    prompts::META_STEP2_USER_MSG,
                    &[
                        ("individual_summaries", new_summaries.as_str()),
                        (
                            "previous_insights",
                            previous.meta_scratch_pad.as_deref().unwrap_or("None"),
                        ),
                        ("best_program_info", best_info.as_str()),
                    ],
                ),
            )
            .await?;
        cost += insights_cost;
        let max_recommendations = self.config.max_recommendations.to_string();
        let (recommendations, recommendations_cost) = self
            .ask(
                llm,
                seed.wrapping_add(10_002),
                prompts::META_STEP3_SYSTEM_MSG,
                fill(
                    prompts::META_STEP3_USER_MSG,
                    &[
                        ("global_insights", insights.as_str()),
                        (
                            "previous_recommendations",
                            previous.meta_recommendations.as_deref().unwrap_or("None"),
                        ),
                        ("best_program_info", best_info.as_str()),
                        ("max_recommendations", max_recommendations.as_str()),
                    ],
                ),
            )
            .await?;
        cost += recommendations_cost;

        let done: std::collections::HashSet<&str> = summarised
            .iter()
            .map(|(program, _)| program.id.as_str())
            .collect();
        let snapshot = {
            let mut state = self.lock();
            state
                .unprocessed_programs
                .retain(|program| !done.contains(program.id.as_str()));
            state.meta_summary = Some(match state.meta_summary.take() {
                Some(existing) if !existing.trim().is_empty() => {
                    format!("{existing}\n\n{new_summaries}")
                }
                _ => new_summaries.clone(),
            });
            state.meta_scratch_pad = Some(insights.trim().to_string());
            state.meta_recommendations = Some(recommendations.trim().to_string());
            state
                .meta_recommendations_history
                .push(recommendations.trim().to_string());
            state.total_programs_meta_processed += done.len() as u32;
            self.save(&state);
            state.clone()
        };
        if let Some(dir) = &self.results_dir {
            let report = format!(
                "# INDIVIDUAL PROGRAM SUMMARIES\n\n{}\n\n# GLOBAL INSIGHTS SCRATCHPAD\n\n{}\n\n# META RECOMMENDATIONS\n\n{}\n",
                snapshot.meta_summary.as_deref().unwrap_or(""),
                snapshot.meta_scratch_pad.as_deref().unwrap_or(""),
                snapshot.meta_recommendations.as_deref().unwrap_or("")
            );
            let _ = fsutil::write_text(
                &dir.join("meta").join(format!(
                    "meta_{}.txt",
                    snapshot.total_programs_meta_processed
                )),
                &report,
            );
        }
        Ok(cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_llm::LlmError;
    use genesis_llm::MockProvider;

    fn config(interval: u32) -> MetaConfig {
        MetaConfig {
            models: vec!["meta-model".into()],
            kwargs: SamplingKwargs::default(),
            interval,
            max_recommendations: 3,
            language: "python".into(),
            use_text_feedback: false,
        }
    }

    fn program(generation: u32, name: &str) -> Program {
        let mut program = Program::new(format!("x = {generation}"), "python");
        program.generation = generation;
        program.correct = true;
        program.combined_score = f64::from(generation);
        program.embedding = vec![1.0; 8];
        program
            .metadata
            .insert("patch_name".into(), serde_json::json!(name));
        program
    }

    fn scripted() -> std::sync::Arc<MockProvider> {
        MockProvider::new(|request: &LlmRequest| {
            let system = request.system.as_str();
            if system == prompts::META_STEP1_SYSTEM_MSG {
                if request.last_user().contains("x = 2") {
                    // A refusal is not retried, unlike a transient failure.
                    return Err(LlmError::Rejected {
                        provider: "mock".into(),
                        status: 400,
                        message: "summary failed".into(),
                    });
                }
                Ok("**Program Name: tried something**".to_string())
            } else if system == prompts::META_STEP2_SYSTEM_MSG {
                Ok("## Successful Algorithmic Patterns\n- restarts help".to_string())
            } else {
                Ok("1. Add more restarts.\n2. Tune the cooling schedule\n   over two lines.\n3. Try hex grids.".to_string())
            }
        })
    }

    #[test]
    fn recommendations_split_on_list_numbers() {
        let items =
            split_recommendations("Intro\n1. First item\n   continues.\n2. Second.\n\n10. Tenth");
        assert_eq!(items, ["First item\n   continues.", "Second.", "Tenth"]);
        assert!(split_recommendations("no list here").is_empty());
    }

    #[tokio::test]
    async fn an_update_runs_three_steps_and_keeps_failed_summaries_queued() {
        let dir = tempfile::tempdir().unwrap();
        let meta = MetaSummarizer::new(config(2), Some(dir.path()));
        assert!(!meta.should_update());
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        assert_eq!(meta.recommendation(true, &mut rng), None);
        for (generation, name) in [(3, "third"), (1, "first"), (2, "second")] {
            meta.add_evaluated_program(&program(generation, name));
        }
        assert!(meta.should_update());

        let provider = scripted();
        let llm = LlmClient::mock(provider.clone());
        meta.update(&llm, Some(&program(3, "third")), 7)
            .await
            .unwrap();

        let state = meta.state();
        assert_eq!(state.total_programs_meta_processed, 2);
        assert_eq!(
            state.unprocessed_programs.len(),
            1,
            "the program whose summary failed stays queued"
        );
        assert_eq!(state.unprocessed_programs[0].generation, 2);
        assert!(state.unprocessed_programs[0].embedding.is_empty());
        let summary = state.meta_summary.unwrap();
        let first = summary
            .find("Generation 1 - Patch Name first - Correct Program: True")
            .unwrap();
        let third = summary.find("Generation 3 - Patch Name third").unwrap();
        assert!(first < third, "summaries are ordered by generation");
        assert_eq!(
            state.meta_scratch_pad.as_deref(),
            Some("## Successful Algorithmic Patterns\n- restarts help")
        );
        assert_eq!(state.meta_recommendations_history.len(), 1);

        let single = meta.recommendation(true, &mut rng).unwrap();
        assert!([
            "Add more restarts.",
            "Tune the cooling schedule\n   over two lines.",
            "Try hex grids."
        ]
        .contains(&single.as_str()));
        assert!(meta
            .recommendation(false, &mut rng)
            .unwrap()
            .starts_with("1. Add more restarts."));

        let calls = provider.calls();
        assert_eq!(calls.iter().filter(|c| c.stage == "meta").count(), 5);
        let step3 = calls
            .iter()
            .find(|c| c.system == prompts::META_STEP3_SYSTEM_MSG)
            .unwrap();
        assert!(step3
            .last_user()
            .contains("generate 3 actionable recommendations"));
        assert!(step3.last_user().contains("restarts help"));

        assert!(dir.path().join("meta/meta_2.txt").is_file());
        let reloaded = MetaSummarizer::new(config(2), Some(dir.path()));
        assert_eq!(reloaded.state(), meta.state());
        assert_eq!(
            reloaded.scratchpad().as_deref(),
            Some("## Successful Algorithmic Patterns\n- restarts help")
        );
    }

    #[tokio::test]
    async fn nothing_queued_costs_nothing() {
        let meta = MetaSummarizer::new(config(1), None);
        let llm = LlmClient::mock(scripted());
        assert_eq!(meta.update(&llm, None, 1).await.unwrap(), 0.0);
        assert_eq!(meta.pending(), 0);
    }
}
