//! The patch operator: one model call proposes an edit to the parent as a
//! SEARCH/REPLACE diff, a full rewrite, a crossover or a repair.

use crate::hooks::{Operator, Proposal, ProposalContext, ProposalError};
use crate::sampler::{Guidance, PromptSampler};
use async_trait::async_trait;
use genesis_core::fsutil;
use genesis_llm::{extract_tag, LlmClient, LlmError, LlmRequest, Message, SamplingKwargs};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use std::sync::Arc;

pub struct PatchOperator {
    llm: LlmClient,
    sampler: Arc<PromptSampler>,
    kwargs: SamplingKwargs,
    max_patch_attempts: u32,
}

impl PatchOperator {
    pub fn new(
        llm: LlmClient,
        sampler: Arc<PromptSampler>,
        kwargs: SamplingKwargs,
        max_patch_attempts: u32,
    ) -> Self {
        Self {
            llm,
            sampler,
            kwargs,
            max_patch_attempts: max_patch_attempts.max(1),
        }
    }
}

fn from_llm_error(error: LlmError, spent: f64) -> ProposalError {
    match error {
        LlmError::BudgetExceeded { .. } => {
            let mut failure =
                ProposalError::new("budget_exhausted", error.to_string(), spent).fatal();
            failure.budget_exhausted = true;
            failure
        }
        LlmError::Endpoint { .. } | LlmError::Config(_) => {
            let mut failure =
                ProposalError::new("llm_endpoint_error", error.to_string(), spent).fatal();
            failure.endpoint_failure = true;
            failure
        }
        LlmError::Empty {
            ref incomplete_reason,
            cost_usd,
            ..
        } => {
            let truncated = incomplete_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("max"));
            let failure =
                ProposalError::new("llm_output_invalid", error.to_string(), spent + cost_usd);
            if truncated {
                failure.fatal()
            } else {
                failure
            }
        }
        other => ProposalError::new("proposal_generation_failed", other.to_string(), spent),
    }
}

#[async_trait]
impl Operator for PatchOperator {
    async fn propose(&self, context: &ProposalContext) -> Result<Proposal, ProposalError> {
        let mut rng = ChaCha8Rng::seed_from_u64(context.rng_seed);
        let guidance = Guidance {
            meta_recommendations: context.meta_recommendation.as_deref(),
            idea: context.idea.as_ref(),
            conclusions: context.conclusions.as_deref(),
            system_override: Some(context.system_prompt.as_str()),
        };
        let prompt = if context.needs_fix {
            self.sampler
                .sample_fix(&context.parent, &context.archive_inspirations, &guidance)
        } else {
            self.sampler.sample(
                &mut rng,
                &context.parent,
                &context.archive_inspirations,
                &context.top_k_inspirations,
                &guidance,
            )
        };
        let mut request = LlmRequest::new(
            context.arm.clone(),
            prompt.system.clone(),
            prompt.user.clone(),
        );
        self.kwargs.apply(&mut request, &mut rng);

        let language = self.sampler.language();
        let (mut cost, mut calls) = (0.0, 0u32);
        let mut last_error = String::from("No changes applied");
        for attempt in 0..self.max_patch_attempts {
            let response = match self.llm.query("proposal", request.clone()).await {
                Ok(response) => response,
                Err(LlmError::Empty {
                    incomplete_reason,
                    cost_usd,
                    provider,
                }) => {
                    cost += cost_usd;
                    calls += 1;
                    let error = LlmError::Empty {
                        incomplete_reason,
                        cost_usd: 0.0,
                        provider,
                    };
                    let failure = from_llm_error(error, cost);
                    if !failure.retryable || attempt + 1 == self.max_patch_attempts {
                        return Err(failure);
                    }
                    continue;
                }
                Err(error) => return Err(from_llm_error(error, cost)),
            };
            cost += response.cost_usd;
            calls += response.calls.max(1);

            let turn_dir = context.attempt_dir.join(format!("patch_{attempt}"));
            let _ = fsutil::write_text(&turn_dir.join("llm_response.txt"), &response.content);
            let _ = fsutil::write_json(
                &turn_dir.join("metadata.json"),
                &json!({
                    "model": response.model,
                    "provider": response.provider,
                    "patch_type": prompt.patch_type,
                    "variant": prompt.variant,
                    "cost_usd": response.cost_usd,
                    "usage": response.usage,
                    "temperature": request.temperature,
                    "reasoning_effort": request.reasoning_effort,
                }),
            );

            match genesis_edit::apply_patch(
                &prompt.patch_type,
                &response.content,
                &context.parent.code,
                language,
            ) {
                Ok(outcome) => {
                    let mut metadata = genesis_core::JsonMap::new();
                    metadata.insert("model_name".into(), json!(response.model));
                    metadata.insert("provider".into(), json!(response.provider));
                    metadata.insert("patch_variant".into(), json!(prompt.variant));
                    metadata.insert("temperature".into(), json!(request.temperature));
                    metadata.insert("reasoning_effort".into(), json!(request.reasoning_effort));
                    metadata.insert("patch_attempts".into(), json!(attempt + 1));
                    if !response.thought.is_empty() {
                        metadata.insert(
                            "thought".into(),
                            json!(fsutil::truncate(&response.thought, 4000)),
                        );
                    }
                    return Ok(Proposal {
                        code: outcome.code,
                        patch_type: prompt.patch_type.clone(),
                        name: extract_tag(&response.content, "NAME")
                            .unwrap_or_else(|| "unnamed".to_string()),
                        description: extract_tag(&response.content, "DESCRIPTION")
                            .unwrap_or_default(),
                        diff: outcome.diff,
                        patch_text: outcome.patch_text,
                        num_applied: outcome.num_applied,
                        cost_usd: cost,
                        external_billed_usd: 0.0,
                        llm_calls: calls,
                        metadata,
                    });
                }
                Err(error) => {
                    last_error = error.0;
                    let _ = fsutil::write_text(&turn_dir.join("patch_error.txt"), &last_error);
                    let feedback = if prompt.patch_type == "fix" {
                        format!(
                            "The previous fix attempt failed. Error: {last_error}. Try again with a different approach."
                        )
                    } else {
                        format!("The previous edit was not successful. Error: {last_error}\n\nTry again.")
                    };
                    request.messages.push(Message::assistant(response.content));
                    request.messages.push(Message::user(feedback));
                }
            }
        }
        Err(ProposalError::new("patch_apply_failed", last_error, cost))
    }
}
