//! EVOLVE-BLOCK aware patch application.
//!
//! A program under evolution marks what may change:
//!
//! ```text
//! import numpy as np            # read-only
//! # EVOLVE-BLOCK-START
//! def solve():                  # editable
//!     ...
//! # EVOLVE-BLOCK-END
//! def run_experiment(): ...     # read-only
//! ```
//!
//! A model proposes a change either as SEARCH/REPLACE blocks
//! ([`apply_diff_patch`]) or as a full rewrite in a code fence
//! ([`apply_full_patch`]). Both appliers take the model's raw response and
//! the current program, and return the new program or a [`PatchError`] whose
//! text is written for the model to read on its next attempt. Neither lets a
//! response change text outside the editable regions.
//!
//! [`analyze_code_metrics`] scores how complex a program is, and
//! [`languages`] lists what is supported.
//!
//! This crate is a Rust port of the patch layer of
//! [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (`shinka/edit`,
//! `shinka/utils/languages.py`, `shinka/database/complexity.py`), which is
//! Copyright Sakana AI and licensed under Apache-2.0. The block grammar,
//! matching rules and diagnostics follow it; where this crate differs, the
//! documentation of the item says so.

mod blocks;
pub mod complexity;
mod diagnostics;
mod diff;
mod full;
pub mod languages;
pub mod markers;
mod matching;
mod text;
mod unified;

pub use complexity::{analyze_code_metrics, CodeMetrics};
pub use diff::{apply_diff_patch, apply_search_replace};
pub use full::apply_full_patch;
pub use languages::{extension, fence_tags, language, LanguageSpec};
pub use markers::{
    immutable_parts, mutable_ranges, outermost_mutable_ranges, redact_immutable,
    validate_evolve_markers, wrap_in_evolve_block,
};
pub use unified::{summarize_diff, unified_diff, DiffSummary};

/// The message of the error returned when a response held no applicable
/// SEARCH/REPLACE block.
pub const NO_CHANGES_APPLIED: &str = diagnostics::NO_CHANGES_APPLIED;

/// A patch that applied.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchOutcome {
    /// The program after the patch.
    pub code: String,
    /// SEARCH/REPLACE blocks applied; always 1 for a full rewrite.
    pub num_applied: usize,
    /// Unified diff from the program before the patch to `code`, with the
    /// whole file as context. Empty when nothing changed.
    pub diff: String,
    /// What was applied: the SEARCH/REPLACE text with marker lines removed,
    /// or the code taken from the fence of a full rewrite.
    pub patch_text: String,
}

/// Why a patch was refused. The message is meant to be shown to the model
/// that wrote the patch.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct PatchError(pub String);

/// Apply a response of the given patch type: `diff` holds SEARCH/REPLACE
/// blocks; `full`, `cross` and `fix` hold a full rewrite.
pub fn apply_patch(
    patch_type: &str,
    response: &str,
    original: &str,
    language: &str,
) -> Result<PatchOutcome, PatchError> {
    match patch_type {
        "diff" => apply_diff_patch(response, original, language),
        "full" | "cross" | "fix" => apply_full_patch(response, original, language),
        other => Err(PatchError(format!("Unknown patch type: {other}"))),
    }
}
