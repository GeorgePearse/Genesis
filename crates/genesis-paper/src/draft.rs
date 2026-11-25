//! The manuscript as the writer model produces it: a title, prose per
//! section in the light markup of [`crate::text`], and the claims with the
//! grounding sources that support them.

use genesis_llm::{extract_json, extract_tag};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// One section: its key in `sections.json`, its heading, and the tag the
/// model wraps it in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionSpec {
    pub key: &'static str,
    pub heading: &'static str,
    pub tag: &'static str,
}

/// The sections of a paper, in order. All are required.
pub const SECTIONS: &[SectionSpec] = &[
    SectionSpec {
        key: "abstract",
        heading: "Abstract",
        tag: "ABSTRACT",
    },
    SectionSpec {
        key: "introduction",
        heading: "Introduction",
        tag: "INTRODUCTION",
    },
    SectionSpec {
        key: "related_work",
        heading: "Related Work",
        tag: "RELATED_WORK",
    },
    SectionSpec {
        key: "method",
        heading: "Method",
        tag: "METHOD",
    },
    SectionSpec {
        key: "results",
        heading: "Results",
        tag: "RESULTS",
    },
    SectionSpec {
        key: "limitations",
        heading: "Limitations",
        tag: "LIMITATIONS",
    },
    SectionSpec {
        key: "conclusion",
        heading: "Conclusion",
        tag: "CONCLUSION",
    },
    SectionSpec {
        key: "supplementary",
        heading: "Supplementary Information",
        tag: "SUPPLEMENTARY",
    },
];

/// One entry of `claims_grounding.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Claim {
    pub claim: String,
    pub source_ids: Vec<String>,
    pub citation_keys: Vec<String>,
}

/// `paper/sections.json`: the source every rendering is made from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct Draft {
    pub title: String,
    pub sections: BTreeMap<String, String>,
    pub claims: Vec<Claim>,
}

impl Draft {
    pub fn section(&self, key: &str) -> &str {
        self.sections.get(key).map(String::as_str).unwrap_or("")
    }

    /// Keys of required sections that are absent or blank.
    pub fn missing_sections(&self) -> Vec<&'static str> {
        SECTIONS
            .iter()
            .filter(|spec| self.section(spec.key).trim().is_empty())
            .map(|spec| spec.key)
            .collect()
    }

    /// Take over whatever a reply supplies and keep the rest. Returns the
    /// keys that changed.
    pub fn merge(&mut self, reply: &DraftReply) -> Vec<String> {
        let mut changed = Vec::new();
        if let Some(title) = reply
            .title
            .as_ref()
            .filter(|title| !title.trim().is_empty())
        {
            if *title != self.title {
                self.title = title.clone();
                changed.push("title".to_string());
            }
        }
        for (key, text) in &reply.sections {
            if self.sections.get(key) != Some(text) {
                self.sections.insert(key.clone(), text.clone());
                changed.push(key.clone());
            }
        }
        if let Some(claims) = &reply.claims {
            if *claims != self.claims {
                self.claims = claims.clone();
                changed.push("claims".to_string());
            }
        }
        changed
    }
}

/// What one model reply contained. A tag the model left out is `None` or
/// absent, which lets a repair reply send only what it changed.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DraftReply {
    pub title: Option<String>,
    pub sections: BTreeMap<String, String>,
    pub claims: Option<Vec<Claim>>,
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect()
}

/// Read `{"claims": [{claim, source_ids, citation_keys}]}`; entries without
/// claim text are dropped.
pub fn parse_claims(value: &Value) -> Vec<Claim> {
    let items = value
        .get("claims")
        .and_then(Value::as_array)
        .or_else(|| value.as_array());
    items
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let claim = item
                .get("claim")
                .and_then(Value::as_str)?
                .trim()
                .to_string();
            (!claim.is_empty()).then(|| Claim {
                claim,
                source_ids: string_list(item.get("source_ids")),
                citation_keys: string_list(item.get("citation_keys")),
            })
        })
        .collect()
}

/// Read a tagged reply: `<TITLE>`, one tag per section, and `<CLAIMS>` with
/// the claims JSON.
pub fn parse_reply(content: &str) -> DraftReply {
    let mut reply = DraftReply {
        title: extract_tag(content, "TITLE"),
        ..DraftReply::default()
    };
    for spec in SECTIONS {
        if let Some(text) = extract_tag(content, spec.tag).filter(|text| !text.is_empty()) {
            reply.sections.insert(spec.key.to_string(), text);
        }
    }
    reply.claims = extract_tag(content, "CLAIMS")
        .and_then(|body| extract_json(&body))
        .map(|value| parse_claims(&value));
    reply
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tagged_reply_is_read_and_merged() {
        let reply = parse_reply(
            "<TITLE>Restarts help</TITLE>\n<ABSTRACT>\nWe study packing.\n\nIt works.\n</ABSTRACT>\n<RESULTS>Best is 2.6.</RESULTS>\n\
             <CLAIMS>\n```json\n{\"claims\": [{\"claim\": \"Best is 2.6\", \"source_ids\": [\"result.best_correct_score\"], \"citation_keys\": []}, {\"claim\": \" \"}]}\n```\n</CLAIMS>",
        );
        assert_eq!(reply.title.as_deref(), Some("Restarts help"));
        assert_eq!(reply.sections["abstract"], "We study packing.\n\nIt works.");
        assert_eq!(reply.claims.as_ref().unwrap().len(), 1);
        assert_eq!(
            reply.claims.as_ref().unwrap()[0].source_ids,
            ["result.best_correct_score"]
        );

        let mut draft = Draft::default();
        assert_eq!(
            draft.merge(&reply),
            ["title", "abstract", "results", "claims"]
        );
        assert_eq!(
            draft.missing_sections(),
            [
                "introduction",
                "related_work",
                "method",
                "limitations",
                "conclusion",
                "supplementary"
            ]
        );
        let repair = parse_reply("<RESULTS>Best is 2.64.</RESULTS>");
        assert_eq!(draft.merge(&repair), ["results"]);
        assert_eq!(draft.section("abstract"), "We study packing.\n\nIt works.");
        assert_eq!(
            draft.claims.len(),
            1,
            "a reply without claims keeps the claims"
        );
        assert!(draft.merge(&repair).is_empty());
    }

    #[test]
    fn claims_accept_an_object_or_a_bare_list() {
        let bare = serde_json::json!([{"claim": "a", "source_ids": ["x", 3, " "]}]);
        assert_eq!(
            parse_claims(&bare),
            vec![Claim {
                claim: "a".into(),
                source_ids: vec!["x".into()],
                citation_keys: vec![]
            }]
        );
        assert!(parse_claims(&serde_json::json!({"other": 1})).is_empty());
    }
}
