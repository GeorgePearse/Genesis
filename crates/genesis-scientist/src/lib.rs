//! The science pipeline around the Genesis evolution loop.
//!
//! A Rust port of Sakana AI's Agentic Scientist (AI-Scientist-v3,
//! Apache-2.0), rebuilt on the ShinkaEvolve archive: ideas with
//! harness-checked rubrics, coding agents as mutation operators, campaign
//! memory with distilled conclusions, and the evidence phases that follow a
//! search: ablations, a sealed final evaluation, post-hoc analysis, a paper
//! with an evidence audit, review and revision.

pub mod agent_operator;
pub mod artifacts;
pub mod context;
pub mod drafting;
pub mod ideas;
pub mod investigation;
pub mod memory;
pub mod paths;
pub mod rubrics;
pub mod selection;

pub use context::{PhaseContext, PhaseStatus};
pub use investigation::{run_investigation, InvestigationConfig, InvestigationOutcome};
pub use paths::RunPaths;
