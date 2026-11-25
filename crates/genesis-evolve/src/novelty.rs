//! Novelty rejection: a candidate whose code embedding is nearly identical
//! to a program already on the parent's island is shown to a judge model,
//! and discarded before evaluation if the judge sees no meaningful change.

use crate::db::ProgramDb;
use crate::prompts::{self, fill};
use genesis_core::Program;
use genesis_llm::{LlmClient, LlmRequest, SamplingKwargs};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct NoveltyVerdict {
    pub accepted: bool,
    /// Highest cosine similarity to a program on the parent's island.
    pub max_similarity: Option<f64>,
    pub most_similar_id: Option<String>,
    /// Whether the judge model was consulted.
    pub judged: bool,
    pub explanation: String,
    pub cost_usd: f64,
}

pub struct NoveltyJudge {
    models: Vec<String>,
    kwargs: SamplingKwargs,
    threshold: f64,
    language: String,
}

/// The judge's reply is novel when it opens with `NOVEL`, plain or bold.
pub fn reply_says_novel(reply: &str) -> bool {
    let upper = reply.trim_start().to_uppercase();
    upper.starts_with("NOVEL") || upper.starts_with("**NOVEL**")
}

impl NoveltyJudge {
    pub fn new(
        models: Vec<String>,
        kwargs: SamplingKwargs,
        threshold: f64,
        language: &str,
    ) -> Self {
        Self {
            models,
            kwargs,
            threshold,
            language: language.to_string(),
        }
    }

    /// Decide whether `code` is different enough from what the parent's
    /// island already holds. Anything that prevents a judgement accepts.
    pub async fn assess(
        &self,
        llm: &LlmClient,
        db: &ProgramDb,
        code: &str,
        embedding: &[f32],
        parent: &Program,
        rng: &mut (dyn rand::RngCore + Send),
    ) -> NoveltyVerdict {
        let accept = |explanation: &str| NoveltyVerdict {
            accepted: true,
            explanation: explanation.to_string(),
            ..NoveltyVerdict::default()
        };
        let Some(island) = parent.island_idx else {
            return accept("the parent has no island");
        };
        if embedding.is_empty() {
            return accept("no embedding");
        }
        let Ok(Some((similarity, nearest))) = db.most_similar(embedding, island) else {
            return accept("no embedded program on the island");
        };
        let mut verdict = NoveltyVerdict {
            accepted: true,
            max_similarity: Some(similarity),
            most_similar_id: Some(nearest.id.clone()),
            ..NoveltyVerdict::default()
        };
        if similarity <= self.threshold {
            verdict.explanation = format!(
                "similarity {similarity:.4} is within the threshold {:.4}",
                self.threshold
            );
            return verdict;
        }
        let Some(model) = self.models.choose(rng) else {
            verdict.explanation = "no judge model configured".to_string();
            return verdict;
        };
        let user = fill(
            prompts::NOVELTY_USER_MSG,
            &[
                ("language", self.language.as_str()),
                ("existing_code", nearest.code.as_str()),
                ("proposed_code", code),
            ],
        );
        let mut request = LlmRequest::new(model.clone(), prompts::NOVELTY_SYSTEM_MSG, user);
        self.kwargs.apply(&mut request, rng);
        verdict.judged = true;
        match llm.query("novelty", request).await {
            Ok(response) => {
                verdict.cost_usd = response.cost_usd;
                verdict.accepted =
                    response.content.trim().is_empty() || reply_says_novel(&response.content);
                verdict.explanation = response.content;
            }
            Err(error) => {
                verdict.explanation = format!("the judge call failed, accepting: {error}");
            }
        }
        verdict
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatabaseConfig;
    use genesis_llm::MockProvider;

    fn island_with(code: &str, embedding: Vec<f32>) -> (ProgramDb, Program) {
        let db = ProgramDb::open(
            None,
            DatabaseConfig {
                num_islands: 1,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let mut seed = Program::new(code, "python");
        seed.correct = true;
        seed.embedding = embedding;
        db.add(&mut seed).unwrap();
        (db, seed)
    }

    fn judge(models: &[&str]) -> NoveltyJudge {
        NoveltyJudge::new(
            models.iter().map(|m| m.to_string()).collect(),
            SamplingKwargs::default(),
            0.9,
            "python",
        )
    }

    #[test]
    fn the_reply_prefix_decides() {
        assert!(reply_says_novel("NOVEL: a different algorithm"));
        assert!(reply_says_novel("  **NOVEL** because"));
        assert!(reply_says_novel("novel"));
        assert!(!reply_says_novel("NOT_NOVEL: renamed variables"));
        assert!(!reply_says_novel("**NOT_NOVEL**"));
        assert!(!reply_says_novel("The code is NOVEL"));
    }

    #[tokio::test]
    async fn dissimilar_code_is_accepted_without_a_judge_call() {
        let (db, seed) = island_with("x = 1", vec![1.0, 0.0]);
        let provider = MockProvider::scripted(vec!["NOT_NOVEL".into()]);
        let llm = LlmClient::mock(provider.clone());
        let mut rng = rand::rngs::mock::StepRng::new(0, 1);
        let verdict = judge(&["judge-model"])
            .assess(&llm, &db, "y = 2", &[0.0, 1.0], &seed, &mut rng)
            .await;
        assert!(verdict.accepted && !verdict.judged);
        assert_eq!(verdict.max_similarity, Some(0.0));
        assert!(provider.calls().is_empty());
    }

    #[tokio::test]
    async fn near_duplicates_go_to_the_judge_and_its_verdict_stands() {
        let (db, seed) = island_with("x = 1", vec![1.0, 0.0]);
        let mut rng = rand::rngs::mock::StepRng::new(0, 1);

        let rejecting = MockProvider::scripted(vec!["NOT_NOVEL: only a rename".into()]);
        let llm = LlmClient::mock(rejecting.clone());
        let verdict = judge(&["judge-model"])
            .assess(&llm, &db, "z = 1", &[1.0, 0.01], &seed, &mut rng)
            .await;
        assert!(!verdict.accepted && verdict.judged);
        assert_eq!(verdict.most_similar_id.as_deref(), Some(seed.id.as_str()));
        let call = &rejecting.calls()[0];
        assert_eq!(call.stage, "novelty");
        assert!(call
            .last_user()
            .contains("**EXISTING CODE:**\n```python\nx = 1\n```"));
        assert!(call
            .last_user()
            .contains("**PROPOSED CODE:**\n```python\nz = 1\n```"));

        let accepting = LlmClient::mock(MockProvider::scripted(vec!["NOVEL: new search".into()]));
        assert!(
            judge(&["judge-model"])
                .assess(&accepting, &db, "z = 1", &[1.0, 0.01], &seed, &mut rng)
                .await
                .accepted
        );

        // Without a judge model the threshold alone never rejects.
        let verdict = judge(&[])
            .assess(&accepting, &db, "z = 1", &[1.0, 0.01], &seed, &mut rng)
            .await;
        assert!(verdict.accepted && !verdict.judged);
        // No embedding, nothing to compare.
        assert!(
            judge(&["judge-model"])
                .assess(&accepting, &db, "z = 1", &[], &seed, &mut rng)
                .await
                .accepted
        );
    }
}
