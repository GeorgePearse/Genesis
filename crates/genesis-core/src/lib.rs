//! Shared foundations for the Genesis Rust engine.
//!
//! Everything both engines agree on lives here: the [`Program`] record, the
//! task contract and its evaluator, the run journal, the cost ledger, layered
//! configuration and the statistics used when comparing programs.

pub mod config;
pub mod eval;
pub mod events;
pub mod fsutil;
pub mod idea;
pub mod ledger;
pub mod program;
pub mod stats;
pub mod task;

pub use eval::{run_evaluation, EvalOutcome, EvalRequest, EvalSpec};
pub use events::{ConsoleMode, EventLog};
pub use idea::{IdeaBundle, Rubric, RubricResult, RubricStatus};
pub use ledger::{CostEvent, CostKind, CostLedger};
pub use program::{JsonMap, Program};
pub use task::{Task, TaskManifest};
