//! What every phase is handed, and how it reports back.

use crate::paths::RunPaths;
use genesis_core::{EventLog, Task};
use genesis_llm::{AgentSpec, LlmClient};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// The shared inputs of a phase. `task` is the run-local snapshot, never the
/// source directory.
#[derive(Clone)]
pub struct PhaseContext {
    pub paths: RunPaths,
    pub task: Task,
    pub llm: LlmClient,
    pub events: Arc<EventLog>,
    /// The coding agent for stages that edit files, when one is configured.
    pub agent: Option<AgentSpec>,
    pub cancel: Arc<AtomicBool>,
    pub seed: u64,
}

impl PhaseContext {
    /// The same context with model spend and events attributed to `phase`.
    pub fn for_phase(&self, phase: &str) -> Self {
        Self {
            llm: self.llm.with_phase(phase),
            events: self.events.scoped(json!({"phase": phase})),
            ..self.clone()
        }
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// How a phase ended. Only `Failed` on a gating phase stops the pipeline;
/// `Degraded` means optional evidence is missing and the run goes on
/// without it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", content = "reason", rename_all = "snake_case")]
pub enum PhaseStatus {
    Completed,
    Degraded(String),
    Skipped(String),
    Failed(String),
}

impl PhaseStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Degraded(_) => "degraded",
            Self::Skipped(_) => "skipped",
            Self::Failed(_) => "failed",
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Completed => None,
            Self::Degraded(reason) | Self::Skipped(reason) | Self::Failed(reason) => Some(reason),
        }
    }

    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Completed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_serialise_with_their_reason() {
        let degraded = PhaseStatus::Degraded("no valid analyses".into());
        assert_eq!(
            serde_json::to_value(&degraded).unwrap(),
            json!({"status": "degraded", "reason": "no valid analyses"})
        );
        assert_eq!(
            serde_json::to_value(PhaseStatus::Completed).unwrap(),
            json!({"status": "completed"})
        );
        assert_eq!(
            (degraded.label(), degraded.reason(), degraded.is_usable()),
            ("degraded", Some("no valid analyses"), false)
        );
        assert!(PhaseStatus::Completed.is_usable());
    }
}
