//! LLM-driven program evolution for Genesis.
//!
//! A Rust port of the ShinkaEvolve engine (Sakana AI, Apache-2.0): an island
//! archive of evaluated programs, parent and inspiration sampling, patch
//! operators driven by an ensemble of models, novelty rejection, a meta
//! scratchpad, and prompt co-evolution.

pub mod config;
pub mod db;
pub mod hooks;
pub mod meta;
pub mod novelty;
pub mod operators;
pub mod prompt_evolver;
pub mod prompts;
pub mod runner;
pub mod sampler;

pub use config::{DatabaseConfig, EvolutionConfig};
pub use db::{ProgramDb, Sample};
pub use hooks::{
    EvalHook, EvaluatedProposal, IdeaSource, Operator, Proposal, ProposalContext, ProposalError,
};
pub use runner::{EvolutionRunner, RunSummary, RunnerDeps};
