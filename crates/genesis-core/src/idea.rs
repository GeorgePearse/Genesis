//! Hypotheses and their rubrics.
//!
//! An idea states what a proposal is trying and how to tell whether it did:
//! a hypothesis, an implementation plan, and target rubrics that the harness
//! checks against the evaluated program. The JSON shapes follow the Agentic
//! Scientist's `idea_bundle.json`, `target_rubrics.json` and
//! `rubric_status.json`.

use crate::fsutil::slugify;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

pub const MAX_RUBRICS: usize = 32;
pub const MAX_PATTERNS: usize = 8;
pub const MAX_PATTERN_CHARS: usize = 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Verification {
    /// Only `source_contains` is defined: every pattern must occur in the
    /// candidate's source, compared on code tokens.
    #[serde(rename = "type")]
    pub kind: String,
    pub patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Rubric {
    pub id: String,
    pub description: String,
    /// `implementation` or `metric`.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification: Option<Verification>,
    /// Structured form of a metric rubric: `metric operator threshold`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct IdeaBundle {
    pub idea_id: String,
    pub title: String,
    pub hypothesis: String,
    pub implementation_plan: Vec<String>,
    pub target_rubrics: Vec<Rubric>,
    pub notes_for_agent: String,
}

fn text_of(raw: &Value, key: &str) -> String {
    raw.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string()
}

fn normalize_rubrics(raw: Option<&Value>) -> Result<Vec<Rubric>> {
    let Some(Value::Array(items)) = raw else {
        return Ok(Vec::new());
    };
    let mut seen = HashSet::new();
    let mut rubrics = Vec::new();
    for item in items {
        let Value::Object(_) = item else { continue };
        let description = text_of(item, "description");
        let mut id = slugify(&text_of(item, "id"));
        if id.is_empty() {
            id = slugify(&description);
        }
        if id.is_empty() {
            continue;
        }
        let base = id.clone();
        let mut suffix = 2;
        while !seen.insert(id.clone()) {
            id = format!("{base}_{suffix}");
            suffix += 1;
        }
        let mut kind = text_of(item, "type").to_ascii_lowercase();
        if kind != "metric" {
            kind = "implementation".to_string();
        }
        let verification = match item.get("verification") {
            Some(v) if kind == "implementation" && v.is_object() => {
                let patterns: Vec<String> = v
                    .get("patterns")
                    .and_then(Value::as_array)
                    .map(|list| {
                        list.iter()
                            .filter_map(Value::as_str)
                            .map(str::trim)
                            .filter(|p| !p.is_empty() && p.chars().count() <= MAX_PATTERN_CHARS)
                            .take(MAX_PATTERNS)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default();
                if text_of(v, "type") == "source_contains" && !patterns.is_empty() {
                    Some(Verification {
                        kind: "source_contains".to_string(),
                        patterns,
                    })
                } else {
                    None
                }
            }
            _ => None,
        };
        let first_text = |keys: &[&str]| {
            keys.iter()
                .map(|key| text_of(item, key))
                .find(|text| !text.is_empty())
        };
        let threshold = ["threshold", "target"]
            .iter()
            .find_map(|key| item.get(*key).and_then(Value::as_f64));
        rubrics.push(Rubric {
            id,
            description,
            kind,
            verification,
            metric: first_text(&["metric", "metric_key", "metric_name"]),
            operator: first_text(&["operator", "comparison"]),
            threshold,
        });
        if rubrics.len() == MAX_RUBRICS {
            break;
        }
    }
    Ok(rubrics)
}

impl IdeaBundle {
    /// Validate and canonicalise a model-written bundle. `cycle` supplies the
    /// fallback id `idea_0007`.
    pub fn normalize(raw: &Value, cycle: u32) -> Result<Self> {
        if !raw.is_object() {
            bail!("the idea bundle is not a JSON object");
        }
        let title = text_of(raw, "title");
        let hypothesis = text_of(raw, "hypothesis");
        if title.is_empty() {
            bail!("the idea bundle has no `title`");
        }
        if hypothesis.is_empty() {
            bail!("the idea bundle has no `hypothesis`");
        }
        let implementation_plan: Vec<String> = match raw.get("implementation_plan") {
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|step| !step.is_empty())
                .map(str::to_string)
                .collect(),
            Some(Value::String(text)) => text
                .lines()
                .map(|line| {
                    line.trim_start_matches(|c: char| c == '-' || c == '*' || c.is_whitespace())
                        .trim_end()
                        .to_string()
                })
                .filter(|line| !line.is_empty())
                .collect(),
            _ => Vec::new(),
        };
        if implementation_plan.is_empty() {
            bail!("the idea bundle has an empty `implementation_plan`");
        }
        let mut idea_id = slugify(&text_of(raw, "idea_id"));
        if idea_id.is_empty() {
            idea_id = format!("idea_{cycle:04}");
        }
        let target_rubrics = normalize_rubrics(raw.get("target_rubrics"))?;
        if target_rubrics.is_empty() {
            bail!("the idea bundle has no usable `target_rubrics`");
        }
        Ok(Self {
            idea_id,
            title,
            hypothesis,
            implementation_plan,
            target_rubrics,
            notes_for_agent: text_of(raw, "notes_for_agent"),
        })
    }

    /// `07_adaptive_reheating`: the directory name of this idea's cycle.
    pub fn dir_name(&self, cycle: u32) -> String {
        let slug = slugify(&self.title);
        let slug = if slug.is_empty() {
            self.idea_id.clone()
        } else {
            slug
        };
        let slug: String = slug.chars().take(60).collect();
        format!("{cycle:02}_{}", slug.trim_end_matches('_'))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct RubricResult {
    pub id: String,
    pub pass: bool,
    pub evidence: String,
    /// `passed`, `failed`, or `not_assessed` when nothing could check it.
    pub assessment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct RubricMetrics {
    pub combined_score: f64,
    pub correct: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct RubricStatus {
    pub attempt_id: String,
    pub idea_id: String,
    /// True only when there is at least one rubric and every one passed.
    pub overall_pass: bool,
    pub rubrics: Vec<RubricResult>,
    pub metrics: RubricMetrics,
    pub next_recommendation: String,
}

impl RubricStatus {
    pub fn new(
        attempt_id: &str,
        idea_id: &str,
        rubrics: Vec<RubricResult>,
        metrics: RubricMetrics,
    ) -> Self {
        let overall_pass = !rubrics.is_empty() && rubrics.iter().all(|r| r.pass);
        Self {
            attempt_id: attempt_id.to_string(),
            idea_id: idea_id.to_string(),
            overall_pass,
            rubrics,
            metrics,
            next_recommendation: "generate a new idea".to_string(),
        }
    }

    pub fn passed(&self) -> usize {
        self.rubrics.iter().filter(|r| r.pass).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalizes_ids_types_and_verification() {
        let raw = json!({
            "idea_id": "Idea CircPack / Reheat 001",
            "title": " Adaptive Reheating + Multi-Restart Search ",
            "hypothesis": "Reheating escapes local optima.",
            "implementation_plan": ["Add reheating", "  ", "Use restarts"],
            "target_rubrics": [
                {"id": "Uses Reheating", "description": "reheats", "type": "Implementation",
                 "verification": {"type": "source_contains", "patterns": ["reheat(", ""]}},
                {"id": "uses_reheating", "description": "dup id", "type": "implementation"},
                {"description": "The sum of radii exceeds 2.1", "type": "metric",
                 "metric": "combined_score", "operator": ">", "threshold": 2.1,
                 "verification": {"type": "source_contains", "patterns": ["x"]}},
                "not an object"
            ],
            "notes_for_agent": "Keep it correct."
        });
        let idea = IdeaBundle::normalize(&raw, 3).unwrap();
        assert_eq!(idea.idea_id, "idea_circpack_reheat_001");
        assert_eq!(idea.implementation_plan.len(), 2);
        let ids: Vec<&str> = idea.target_rubrics.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "uses_reheating",
                "uses_reheating_2",
                "the_sum_of_radii_exceeds_2_1"
            ]
        );
        assert_eq!(
            idea.target_rubrics[0]
                .verification
                .as_ref()
                .unwrap()
                .patterns,
            ["reheat("]
        );
        assert_eq!(idea.target_rubrics[2].kind, "metric");
        assert!(idea.target_rubrics[2].verification.is_none());
        assert_eq!(idea.target_rubrics[2].threshold, Some(2.1));
        assert_eq!(
            idea.dir_name(3),
            "03_adaptive_reheating_multi_restart_search"
        );
        let round_trip: IdeaBundle =
            serde_json::from_value(serde_json::to_value(&idea).unwrap()).unwrap();
        assert_eq!(round_trip, idea);
    }

    #[test]
    fn rejects_incomplete_bundles_and_fills_the_id() {
        assert!(IdeaBundle::normalize(&json!([]), 1).is_err());
        assert!(IdeaBundle::normalize(&json!({"title": "t"}), 1).is_err());
        assert!(IdeaBundle::normalize(
            &json!({"title": "t", "hypothesis": "h", "implementation_plan": []}),
            1
        )
        .is_err());
        let idea = IdeaBundle::normalize(
            &json!({"title": "t", "hypothesis": "h", "implementation_plan": "- one\n- two",
                    "target_rubrics": [{"description": "does a thing"}]}),
            7,
        )
        .unwrap();
        assert_eq!(idea.idea_id, "idea_0007");
        assert_eq!(idea.implementation_plan, ["one", "two"]);
        assert_eq!(idea.target_rubrics[0].id, "does_a_thing");
    }

    /// Found by the `a_normalised_idea_has_unique_slug_rubric_ids` property
    /// (EPHYRA_REPLAY=4109901898230880658:2424): whitespace left behind a
    /// bullet survived the first normalisation and was trimmed by the second.
    #[test]
    fn a_bulleted_plan_step_is_fully_trimmed() {
        let raw = json!({"title": "t", "hypothesis": "h", "implementation_plan": "-\u{b}sv'kGsf\n *  \t two ",
                         "target_rubrics": [{"description": "d"}]});
        let idea = IdeaBundle::normalize(&raw, 0).unwrap();
        assert_eq!(idea.implementation_plan, ["sv'kGsf", "two"]);
        let again = IdeaBundle::normalize(&serde_json::to_value(&idea).unwrap(), 0).unwrap();
        assert_eq!(again, idea);
    }

    #[test]
    fn overall_pass_needs_every_rubric_and_at_least_one() {
        let pass = RubricResult {
            id: "a".into(),
            pass: true,
            ..RubricResult::default()
        };
        let fail = RubricResult {
            id: "b".into(),
            pass: false,
            ..RubricResult::default()
        };
        assert!(
            RubricStatus::new("x", "y", vec![pass.clone()], RubricMetrics::default()).overall_pass
        );
        assert!(
            !RubricStatus::new("x", "y", vec![pass, fail], RubricMetrics::default()).overall_pass
        );
        assert!(!RubricStatus::new("x", "y", vec![], RubricMetrics::default()).overall_pass);
    }
}
