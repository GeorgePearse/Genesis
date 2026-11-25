//! Configuration of the paper, review and revision phases.

use genesis_llm::SamplingKwargs;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PaperConfig {
    /// Models that plan, write and repair the paper; one is drawn per call.
    pub models: Vec<String>,
    pub kwargs: SamplingKwargs,
    /// `article`: a self-contained LaTeX article and a Markdown twin.
    pub paper_format: String,
    /// Citations to aim for; fewer is a warning, never an error.
    pub min_refs: usize,
    /// References to retrieve at most.
    pub max_refs: usize,
    /// Repair passes after a failed validation or audit.
    pub max_repair_rounds: u32,
    /// `qualified_only` refuses to write up a winner that failed its final
    /// evaluation; `allow_not_qualified` writes it up with that stated.
    pub final_outcome_policy: String,
    /// Compile a PDF when `pdflatex` and `bibtex` are on `PATH`.
    pub compile_pdf: bool,
    /// Look references up over the network. Off, the paper has no
    /// bibliography.
    pub retrieve_references: bool,
    pub semantic_scholar_url: String,
    pub openalex_url: String,
    /// Models for the second audit layer; empty runs the numeric and claim
    /// checks only.
    pub audit_models: Vec<String>,
}

impl Default for PaperConfig {
    fn default() -> Self {
        Self {
            models: vec!["claude-opus-5-5".to_string()],
            kwargs: SamplingKwargs {
                temperatures: vec![0.7],
                ..SamplingKwargs::default()
            },
            paper_format: "article".to_string(),
            min_refs: 8,
            max_refs: 15,
            max_repair_rounds: 4,
            final_outcome_policy: "qualified_only".to_string(),
            compile_pdf: true,
            retrieve_references: true,
            semantic_scholar_url: "https://api.semanticscholar.org".to_string(),
            openalex_url: "https://api.openalex.org".to_string(),
            audit_models: Vec::new(),
        }
    }
}

impl PaperConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.models.is_empty(), "paper.models is empty");
        anyhow::ensure!(
            self.paper_format == "article",
            "paper.paper_format is `{}`; the only format is `article`",
            self.paper_format
        );
        anyhow::ensure!(
            matches!(
                self.final_outcome_policy.as_str(),
                "qualified_only" | "allow_not_qualified"
            ),
            "paper.final_outcome_policy is `{}`; expected qualified_only or allow_not_qualified",
            self.final_outcome_policy
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ReviewConfig {
    pub model: String,
    /// Independent reviews behind the meta-review.
    pub reviewer_count: usize,
    /// Tries per reviewer, counting the corrective retry.
    pub attempts: u32,
    pub max_tokens: u32,
}

impl Default for ReviewConfig {
    fn default() -> Self {
        Self {
            model: "claude-opus-5-5".to_string(),
            reviewer_count: 5,
            attempts: 2,
            max_tokens: 16_384,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RevisionConfig {
    /// Revision attempts before the original paper is kept.
    pub max_attempts: u32,
}

impl Default for RevisionConfig {
    fn default() -> Self {
        Self { max_attempts: 2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_and_bad_values_are_named() {
        PaperConfig::default().validate().unwrap();
        let no_models = PaperConfig {
            models: vec![],
            ..Default::default()
        };
        assert!(no_models
            .validate()
            .unwrap_err()
            .to_string()
            .contains("paper.models"));
        let format = PaperConfig {
            paper_format: "iclr".into(),
            ..Default::default()
        };
        assert!(format
            .validate()
            .unwrap_err()
            .to_string()
            .contains("article"));
        let policy = PaperConfig {
            final_outcome_policy: "always".into(),
            ..Default::default()
        };
        assert!(policy
            .validate()
            .unwrap_err()
            .to_string()
            .contains("allow_not_qualified"));
        assert_eq!(
            (
                ReviewConfig::default().reviewer_count,
                RevisionConfig::default().max_attempts
            ),
            (5, 2)
        );
    }
}
