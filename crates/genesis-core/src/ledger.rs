//! The cost ledger: an append-only `metrics/cost_events.jsonl` and an atomic
//! aggregate `metrics/cost_summary.json`.
//!
//! Provider-reported charges, list-price estimates and usage with no known
//! price are kept apart, because a subscription-billed agent self-reports
//! prices nobody is invoiced for. The ledger is run telemetry, not a billing
//! system.

use crate::fsutil;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CostKind {
    /// The provider or agent CLI stated the charge.
    Reported,
    /// Tokens priced from the bundled list-price table.
    #[default]
    Estimated,
    /// Tokens were counted but no price is known. Not `$0`.
    UsageOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct CostEvent {
    pub ts: String,
    /// Pipeline phase: `investigation`, `ablations`, `paper`, ...
    pub phase: String,
    /// Stage within the phase: `proposal`, `novelty`, `meta`, `idea`, ...
    pub stage: String,
    /// `llm_api`, `agent_cli` or `embedding`.
    pub source: String,
    pub provider: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub thinking_tokens: u64,
    pub cost_usd: f64,
    pub cost_kind: CostKind,
    pub calls: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct CostBucket {
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub thinking_tokens: u64,
    pub cost_usd: f64,
}

impl CostBucket {
    fn add(&mut self, event: &CostEvent) {
        self.calls = self.calls.saturating_add(event.calls.max(1));
        self.input_tokens = self.input_tokens.saturating_add(event.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(event.output_tokens);
        self.thinking_tokens = self.thinking_tokens.saturating_add(event.thinking_tokens);
        if event.cost_usd.is_finite() && event.cost_usd > 0.0 {
            self.cost_usd += event.cost_usd;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct CostSummary {
    pub total_cost_usd: f64,
    pub reported_cost_usd: f64,
    pub estimated_cost_usd: f64,
    pub unpriced_calls: u64,
    pub total: CostBucket,
    pub by_phase: BTreeMap<String, CostBucket>,
    pub by_stage: BTreeMap<String, CostBucket>,
    pub by_source: BTreeMap<String, CostBucket>,
    pub by_model: BTreeMap<String, CostBucket>,
}

impl CostSummary {
    fn add(&mut self, event: &CostEvent) {
        let cost = if event.cost_usd.is_finite() && event.cost_usd > 0.0 {
            event.cost_usd
        } else {
            0.0
        };
        match event.cost_kind {
            CostKind::Reported => self.reported_cost_usd += cost,
            CostKind::Estimated => self.estimated_cost_usd += cost,
            CostKind::UsageOnly => {
                self.unpriced_calls = self.unpriced_calls.saturating_add(event.calls.max(1))
            }
        }
        self.total_cost_usd = self.reported_cost_usd + self.estimated_cost_usd;
        self.total.add(event);
        self.by_phase
            .entry(event.phase.clone())
            .or_default()
            .add(event);
        self.by_stage
            .entry(format!("{}/{}", event.phase, event.stage))
            .or_default()
            .add(event);
        self.by_source
            .entry(event.source.clone())
            .or_default()
            .add(event);
        self.by_model
            .entry(event.model.clone())
            .or_default()
            .add(event);
    }
}

/// Thread-safe ledger rooted at a run directory.
#[derive(Debug)]
pub struct CostLedger {
    dir: Option<PathBuf>,
    summary: Mutex<CostSummary>,
}

impl CostLedger {
    /// Open the ledger under `<run_dir>/metrics`, replaying earlier events so
    /// a resumed run keeps its totals.
    pub fn open(run_dir: &Path) -> Result<Self> {
        let dir = run_dir.join("metrics");
        std::fs::create_dir_all(&dir)?;
        let mut summary = CostSummary::default();
        for event in fsutil::read_jsonl::<CostEvent>(&dir.join("cost_events.jsonl")) {
            summary.add(&event);
        }
        Ok(Self {
            dir: Some(dir),
            summary: Mutex::new(summary),
        })
    }

    /// A ledger that only keeps totals in memory.
    pub fn in_memory() -> Self {
        Self {
            dir: None,
            summary: Mutex::new(CostSummary::default()),
        }
    }

    pub fn record(&self, mut event: CostEvent) {
        if event.ts.is_empty() {
            event.ts = chrono::Utc::now().to_rfc3339();
        }
        if event.calls == 0 {
            event.calls = 1;
        }
        let mut summary = self.summary.lock().unwrap_or_else(|p| p.into_inner());
        summary.add(&event);
        if let Some(dir) = &self.dir {
            let _ = fsutil::append_jsonl(&dir.join("cost_events.jsonl"), &event);
            let _ = fsutil::write_json(&dir.join("cost_summary.json"), &*summary);
        }
    }

    pub fn total_usd(&self) -> f64 {
        self.summary
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .total_cost_usd
    }

    pub fn summary(&self) -> CostSummary {
        self.summary
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(phase: &str, model: &str, cost: f64, kind: CostKind) -> CostEvent {
        CostEvent {
            phase: phase.into(),
            stage: "proposal".into(),
            source: "llm_api".into(),
            model: model.into(),
            input_tokens: 100,
            output_tokens: 10,
            cost_usd: cost,
            cost_kind: kind,
            ..CostEvent::default()
        }
    }

    /// A ledger is fed by external tools: an absurd token count or a
    /// negative or non-finite charge must not corrupt the totals.
    #[test]
    fn hostile_numbers_do_not_corrupt_the_totals() {
        let ledger = CostLedger::in_memory();
        ledger.record(event("p", "m", 0.5, CostKind::Reported));
        for cost in [-3.0, f64::NAN, f64::INFINITY] {
            let mut bad = event("p", "m", cost, CostKind::Estimated);
            bad.input_tokens = u64::MAX;
            ledger.record(bad);
        }
        let summary = ledger.summary();
        assert_eq!(summary.total_cost_usd, 0.5);
        assert_eq!(summary.total.input_tokens, u64::MAX);
        assert_eq!(summary.total.calls, 4);
    }

    #[test]
    fn totals_separate_provenance_and_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        {
            let ledger = CostLedger::open(dir.path()).unwrap();
            ledger.record(event("investigation", "a", 0.5, CostKind::Reported));
            ledger.record(event("investigation", "b", 0.25, CostKind::Estimated));
            ledger.record(event("paper", "c", 0.0, CostKind::UsageOnly));
            assert!((ledger.total_usd() - 0.75).abs() < 1e-12);
        }
        let ledger = CostLedger::open(dir.path()).unwrap();
        let summary = ledger.summary();
        assert!((summary.total_cost_usd - 0.75).abs() < 1e-12);
        assert_eq!(summary.unpriced_calls, 1);
        assert_eq!(summary.by_phase["investigation"].calls, 2);
        assert_eq!(summary.by_stage["paper/proposal"].input_tokens, 100);
        assert_eq!(summary.total.calls, 3);
        let on_disk: CostSummary =
            fsutil::read_json(&dir.path().join("metrics/cost_summary.json")).unwrap();
        assert_eq!(on_disk, summary);
    }
}
