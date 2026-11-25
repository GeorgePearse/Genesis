//! Which variants an ablation study runs: the two fixed ones, the
//! candidates a model proposes, and the rules that decide among them.

use genesis_core::fsutil::slugify;
use genesis_scientist::artifacts::VariantSpec;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

/// The selected program, evaluated unchanged. Everything is compared with it.
pub const CONTROL_ID: &str = "control_best";
/// The seed program, evaluated unchanged.
pub const BASELINE_REFERENCE_ID: &str = "baseline_reference";
/// Ids a generated candidate may not take.
pub const RESERVED_IDS: [&str; 2] = [CONTROL_ID, BASELINE_REFERENCE_ID];

pub fn control_spec() -> VariantSpec {
    VariantSpec {
        variant_id: CONTROL_ID.to_string(),
        title: "Selected program (control)".to_string(),
        kind: "control".to_string(),
        base_program: "best".to_string(),
        hypothesis:
            "The program selected by the search, evaluated unchanged under the study's seeds."
                .to_string(),
        ..VariantSpec::default()
    }
}

pub fn baseline_reference_spec() -> VariantSpec {
    VariantSpec {
        variant_id: BASELINE_REFERENCE_ID.to_string(),
        title: "Reference program (seed)".to_string(),
        kind: "baseline".to_string(),
        base_program: "reference".to_string(),
        hypothesis:
            "The program the search started from, evaluated unchanged under the study's seeds."
                .to_string(),
        ..VariantSpec::default()
    }
}

fn text(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string()
}

fn text_list(value: &Value, key: &str) -> Vec<String> {
    match value.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(str::to_string)
            .collect(),
        Some(Value::String(item)) if !item.trim().is_empty() => vec![item.trim().to_string()],
        _ => Vec::new(),
    }
}

/// The well-formed candidates of a model's variant bundle, in its order.
///
/// A candidate needs an id, a title and a kind of `baseline` or `ablation`;
/// anything else is skipped. Ids are slugified, since they name directories.
pub fn parse_candidates(bundle: &Value) -> Vec<VariantSpec> {
    let Some(items) = bundle.get("candidates").and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter(|item| item.is_object())
        .filter_map(|item| {
            let variant_id = slugify(&text(item, "variant_id"));
            let title = text(item, "title");
            let kind = text(item, "kind").to_ascii_lowercase();
            if variant_id.is_empty()
                || title.is_empty()
                || !matches!(kind.as_str(), "baseline" | "ablation")
            {
                return None;
            }
            let base_program = if text(item, "base_program") == "reference" {
                "reference"
            } else {
                "best"
            };
            Some(VariantSpec {
                variant_id,
                title,
                kind,
                base_program: base_program.to_string(),
                hypothesis: text(item, "hypothesis"),
                relevance_reason: text(item, "relevance_reason"),
                feasibility_reason: text(item, "feasibility_reason"),
                target_components: text_list(item, "target_components"),
                implementation_plan: text_list(item, "implementation_plan"),
                expected_effect: text(item, "expected_effect"),
            })
        })
        .collect()
}

/// A reviewer's verdict on one candidate.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Decision {
    pub accepted: bool,
    pub reason: String,
}

/// The reviewer's decisions by variant id. Entries without an id or without
/// a boolean verdict are ignored, which leaves their candidate accepted.
pub fn parse_decisions(reply: &Value) -> BTreeMap<String, Decision> {
    let mut decisions = BTreeMap::new();
    for item in reply
        .get("decisions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = slugify(&text(item, "variant_id"));
        let Some(accepted) = item.get("accepted").and_then(Value::as_bool) else {
            continue;
        };
        if !id.is_empty() {
            decisions.insert(
                id,
                Decision {
                    accepted,
                    reason: text(item, "reason"),
                },
            );
        }
    }
    decisions
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct RejectedVariant {
    pub variant_id: String,
    pub title: String,
    pub kind: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Selection {
    pub accepted: Vec<VariantSpec>,
    pub rejected: Vec<RejectedVariant>,
}

/// Decide which candidates are implemented.
///
/// Candidates are taken in the generator's order. A reserved id, a repeated
/// id and a candidate the reviewer rejected are turned down; a candidate the
/// reviewer did not mention is kept. Then the caps apply, per kind.
pub fn select_variants(
    candidates: &[VariantSpec],
    decisions: &BTreeMap<String, Decision>,
    max_ablations: usize,
    max_extra_baselines: usize,
) -> Selection {
    let mut selection = Selection::default();
    let mut seen = HashSet::new();
    let (mut ablations, mut baselines) = (0usize, 0usize);
    for candidate in candidates {
        let id = candidate.variant_id.as_str();
        let is_ablation = candidate.kind == "ablation";
        let (taken, cap, cap_name) = if is_ablation {
            (&mut ablations, max_ablations, "max_ablations")
        } else {
            (&mut baselines, max_extra_baselines, "max_extra_baselines")
        };
        let first_time = seen.insert(id.to_string());
        let reason = if RESERVED_IDS.contains(&id) {
            Some("reserved variant id".to_string())
        } else if !first_time {
            Some("duplicate variant id".to_string())
        } else if let Some(decision) = decisions.get(id).filter(|decision| !decision.accepted) {
            Some(if decision.reason.is_empty() {
                "rejected by the reviewer".to_string()
            } else {
                format!("rejected by the reviewer: {}", decision.reason)
            })
        } else if *taken >= cap {
            Some(format!("{cap_name} cap of {cap} reached"))
        } else {
            None
        };
        match reason {
            Some(reason) => selection.rejected.push(RejectedVariant {
                variant_id: candidate.variant_id.clone(),
                title: candidate.title.clone(),
                kind: candidate.kind.clone(),
                reason,
            }),
            None => {
                *taken += 1;
                selection.accepted.push(candidate.clone());
            }
        }
    }
    selection
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn candidate(id: &str, kind: &str) -> VariantSpec {
        VariantSpec {
            variant_id: id.into(),
            title: id.into(),
            kind: kind.into(),
            base_program: "best".into(),
            ..Default::default()
        }
    }

    #[test]
    fn malformed_candidates_are_skipped_and_fields_normalised() {
        let bundle = json!({"decision": "propose_variants", "candidates": [
            {"variant_id": "Ablation No-Restarts", "title": " No restarts ", "kind": "Ablation",
             "target_components": "restart_loop", "implementation_plan": ["remove the loop", " "],
             "base_program": "best", "hypothesis": "restarts matter"},
            {"variant_id": "baseline_greedy", "title": "Greedy", "kind": "baseline", "base_program": "reference"},
            {"variant_id": "no_title", "kind": "ablation"},
            {"variant_id": "bad_kind", "title": "x", "kind": "control"},
            {"variant_id": "???", "title": "x", "kind": "ablation"},
            "not an object"
        ]});
        let parsed = parse_candidates(&bundle);
        assert_eq!(parsed.len(), 2);
        assert_eq!(
            (parsed[0].variant_id.as_str(), parsed[0].kind.as_str()),
            ("ablation_no_restarts", "ablation")
        );
        assert_eq!(parsed[0].title, "No restarts");
        assert_eq!(parsed[0].target_components, ["restart_loop"]);
        assert_eq!(parsed[0].implementation_plan, ["remove the loop"]);
        assert_eq!(parsed[1].base_program, "reference");
        assert!(parse_candidates(&json!({"candidates": "none"})).is_empty());
        assert!(parse_candidates(&json!([])).is_empty());
    }

    #[test]
    fn selection_applies_the_rules_in_order() {
        let candidates = vec![
            candidate("control_best", "ablation"),
            candidate("ablation_a", "ablation"),
            candidate("ablation_a", "ablation"),
            candidate("ablation_b", "ablation"),
            candidate("ablation_c", "ablation"),
            candidate("baseline_x", "baseline"),
            candidate("baseline_y", "baseline"),
        ];
        let decisions = parse_decisions(&json!({"decisions": [
            {"variant_id": "ablation_b", "accepted": false, "reason": "tuning sweep"},
            {"variant_id": "ablation_c", "accepted": true, "reason": "fine"},
            {"variant_id": "baseline_x", "accepted": "yes"},
            {"accepted": false}
        ]}));
        let selection = select_variants(&candidates, &decisions, 1, 5);
        let accepted: Vec<&str> = selection
            .accepted
            .iter()
            .map(|v| v.variant_id.as_str())
            .collect();
        assert_eq!(accepted, ["ablation_a", "baseline_x", "baseline_y"]);
        let reasons: Vec<(&str, &str)> = selection
            .rejected
            .iter()
            .map(|r| (r.variant_id.as_str(), r.reason.as_str()))
            .collect();
        assert_eq!(
            reasons,
            [
                ("control_best", "reserved variant id"),
                ("ablation_a", "duplicate variant id"),
                ("ablation_b", "rejected by the reviewer: tuning sweep"),
                ("ablation_c", "max_ablations cap of 1 reached"),
            ]
        );
        assert!(select_variants(&candidates, &BTreeMap::new(), 0, 0)
            .accepted
            .is_empty());
    }
}
