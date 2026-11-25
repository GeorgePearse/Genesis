//! The extension points of the evolution loop.
//!
//! The loop itself knows how to sample a parent, evaluate a candidate and
//! keep the archive. What turns a parent into a candidate is an
//! [`Operator`]; what the proposal is trying to show is an idea from an
//! [`IdeaSource`]; what is learnt from the evaluation is recorded by
//! [`EvalHook`]s. The patch operator in this crate and the coding-agent
//! operator in `genesis-scientist` plug in through the same trait, so one
//! bandit can choose between them.

use async_trait::async_trait;
use genesis_core::{EvalOutcome, IdeaBundle, JsonMap, Program, Task};
use std::path::PathBuf;
use std::sync::Arc;

/// Everything an operator may use to produce one candidate.
#[derive(Debug, Clone)]
pub struct ProposalContext {
    pub generation: u32,
    /// Scratch directory for this draw's artifacts (prompts, replies, logs).
    pub attempt_dir: PathBuf,
    pub task: Arc<Task>,
    pub parent: Program,
    pub archive_inspirations: Vec<Program>,
    pub top_k_inspirations: Vec<Program>,
    /// No correct program exists yet; the parent is to be repaired.
    pub needs_fix: bool,
    /// The task description in force: the task's own or an evolved prompt.
    pub system_prompt: String,
    pub meta_recommendation: Option<String>,
    pub conclusions: Option<String>,
    pub idea: Option<IdeaBundle>,
    /// The model, or `agent:<cli>[@model]`, the bandit chose.
    pub arm: String,
    pub best: Option<Program>,
    pub rng_seed: u64,
}

/// A candidate program and how it was produced.
#[derive(Debug, Clone, Default)]
pub struct Proposal {
    pub code: String,
    /// `diff`, `full`, `cross`, `fix` or `agent`.
    pub patch_type: String,
    pub name: String,
    pub description: String,
    /// Unified diff from the parent.
    pub diff: String,
    /// The patch as written by the model, for the run directory.
    pub patch_text: String,
    pub num_applied: usize,
    pub cost_usd: f64,
    /// The part of `cost_usd` that was really charged outside the model
    /// client, such as a coding agent billed to an API key. It counts
    /// against the run's budget; a subscription agent's list-price estimate
    /// does not.
    pub external_billed_usd: f64,
    pub llm_calls: u32,
    /// Provenance merged into the program's metadata.
    pub metadata: JsonMap,
}

/// Why an operator produced no candidate.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{class}: {message}")]
pub struct ProposalError {
    /// `llm_output_invalid`, `patch_apply_failed`, `llm_endpoint_error`,
    /// `budget_exhausted`, `agent_failed` or `proposal_generation_failed`.
    pub class: String,
    pub message: String,
    /// What the failed attempt cost; it is still billed.
    pub cost_usd: f64,
    /// As [`Proposal::external_billed_usd`].
    pub external_billed_usd: f64,
    /// Whether a fresh draw could succeed.
    pub retryable: bool,
    /// The model's endpoint is unusable; another arm should be tried.
    pub endpoint_failure: bool,
    pub budget_exhausted: bool,
}

impl ProposalError {
    pub fn new(class: &str, message: impl Into<String>, cost_usd: f64) -> Self {
        Self {
            class: class.to_string(),
            message: message.into(),
            cost_usd,
            external_billed_usd: 0.0,
            retryable: true,
            endpoint_failure: false,
            budget_exhausted: false,
        }
    }

    pub fn fatal(mut self) -> Self {
        self.retryable = false;
        self
    }
}

#[async_trait]
pub trait Operator: Send + Sync {
    async fn propose(&self, context: &ProposalContext) -> Result<Proposal, ProposalError>;
}

/// Supplies the hypothesis a proposal should implement, and the distilled
/// conclusions of earlier cycles.
#[async_trait]
pub trait IdeaSource: Send + Sync {
    /// The idea for this proposal and what generating it cost.
    async fn generate(&self, context: &ProposalContext) -> anyhow::Result<(IdeaBundle, f64)>;

    fn conclusions(&self) -> Option<String> {
        None
    }
}

/// What the loop established about one proposal, handed to hooks.
#[derive(Debug, Clone)]
pub struct EvaluatedProposal {
    pub generation: u32,
    pub gen_dir: PathBuf,
    pub parent: Option<Program>,
    pub idea: Option<IdeaBundle>,
    pub outcome: EvalOutcome,
    /// True when this program became the best correct program of the run.
    pub is_new_best: bool,
}

/// Runs after a program is evaluated and stored. A hook may add to
/// `program.metadata`; the loop persists the change.
#[async_trait]
pub trait EvalHook: Send + Sync {
    async fn after_evaluation(
        &self,
        program: &mut Program,
        evaluated: &EvaluatedProposal,
    ) -> anyhow::Result<()>;

    /// A proposal that never reached evaluation.
    async fn after_failure(
        &self,
        _generation: u32,
        _idea: Option<&IdeaBundle>,
        _class: &str,
        _message: &str,
    ) {
    }
}
