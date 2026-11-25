//! The evidence phases that follow a search.
//!
//! Once the investigation has frozen the program it stands behind, three
//! phases gather the evidence a paper about it needs:
//!
//! * [`ablations`] evaluates the selected program beside its baseline and
//!   beside variants with one component removed or replaced, each over the
//!   same seeds, and reports the differences with their intervals.
//! * [`final_eval`] evaluates the selected program and the baseline once, on
//!   held-out seeds or the task's own final evaluator, and seals the result.
//! * [`posthoc`] analyses the artifacts the search left behind without
//!   evaluating anything again.
//!
//! This is a Rust port of the downstream phases of Sakana AI's Agentic
//! Scientist (AI-Scientist-v3, Apache-2.0: `src/variants_pipeline`,
//! `src/final_evaluation`, `src/posthoc_pipeline`). Where the original
//! drives coding agents through files, these phases make single model calls
//! that return JSON, and every number they publish is computed by code.
//!
//! Each phase reads the frozen selection written by
//! `genesis_scientist::selection`, writes only under its own directory of
//! the run, and can be called again on a directory it has finished.

pub mod ablations;
mod common;
pub mod csv;
pub mod final_eval;
pub mod posthoc;

pub use ablations::{run_ablations, AblationConfig};
pub use final_eval::{load_final_manifest, run_final_evaluation, FinalEvalConfig};
pub use posthoc::{run_posthoc, PosthocConfig};
