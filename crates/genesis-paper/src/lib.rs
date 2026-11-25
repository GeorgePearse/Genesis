//! Paper writing from run evidence.
//!
//! After a search and its evidence phases, this crate writes the paper:
//! it gathers everything the run established into grounding data, has a
//! model plan and write the prose, generates every table and figure from
//! the data, compiles the result, and audits it, so that every number in
//! the manuscript is traced to a run artifact. A review ensemble and a
//! review-driven revision follow.
//!
//! This is a Rust port of the paper, review and revision phases of Sakana
//! AI's Agentic Scientist (AI-Scientist-v3, Apache-2.0). Where the original
//! hands a coding agent a workspace, this crate asks a model for text and
//! keeps typesetting, retrieval, validation and the numeric audit in code.

pub mod audit;
pub mod config;
pub mod data;
pub mod draft;
pub mod numbers;
mod paper;
mod prompts;
pub mod references;
pub mod render;
mod review;
mod revision;
pub mod text;
pub mod validate;

pub use config::{PaperConfig, ReviewConfig, RevisionConfig};
pub use paper::{load_manifest, run_paper};
pub use review::{
    average_scores, parse_review, review_manuscript, run_review, validate_review, REVIEW_KEYS,
    SCORE_RANGES,
};
pub use revision::run_revision;
