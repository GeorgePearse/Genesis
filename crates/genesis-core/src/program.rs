//! The program record shared by the evolution archive and the science pipeline.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub type JsonMap = Map<String, Value>;

/// One candidate program and everything known about it.
///
/// Field names follow the ShinkaEvolve `Program` dataclass so that run
/// databases stay readable by the existing Genesis WebUI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Program {
    pub id: String,
    pub code: String,
    pub language: String,
    pub parent_id: Option<String>,
    pub archive_inspiration_ids: Vec<String>,
    pub top_k_inspiration_ids: Vec<String>,
    pub island_idx: Option<usize>,
    pub generation: u32,
    /// Unix seconds.
    pub timestamp: f64,
    pub code_diff: Option<String>,
    pub combined_score: f64,
    pub public_metrics: JsonMap,
    pub private_metrics: JsonMap,
    pub text_feedback: String,
    pub complexity: f64,
    pub embedding: Vec<f32>,
    pub correct: bool,
    pub children_count: u32,
    /// Free-form provenance: patch name and description, operator, model,
    /// cost, idea bundle, rubric status.
    pub metadata: JsonMap,
    pub migration_history: Vec<Value>,
    pub in_archive: bool,
    pub system_prompt_id: Option<String>,
}

impl Default for Program {
    fn default() -> Self {
        Self {
            id: String::new(),
            code: String::new(),
            language: "python".to_string(),
            parent_id: None,
            archive_inspiration_ids: Vec::new(),
            top_k_inspiration_ids: Vec::new(),
            island_idx: None,
            generation: 0,
            timestamp: 0.0,
            code_diff: None,
            combined_score: 0.0,
            public_metrics: JsonMap::new(),
            private_metrics: JsonMap::new(),
            text_feedback: String::new(),
            complexity: 0.0,
            embedding: Vec::new(),
            correct: false,
            children_count: 0,
            metadata: JsonMap::new(),
            migration_history: Vec::new(),
            in_archive: false,
            system_prompt_id: None,
        }
    }
}

impl Program {
    /// A fresh program with a new id and the current timestamp.
    pub fn new(code: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            code: code.into(),
            language: language.into(),
            timestamp: crate::fsutil::now_ts(),
            ..Self::default()
        }
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(Value::as_str)
    }

    pub fn meta_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(Value::as_f64)
    }

    /// Short human label: the patch name when there is one, else the id prefix.
    pub fn label(&self) -> String {
        match self.meta_str("patch_name") {
            Some(name) if !name.is_empty() => name.to_string(),
            _ => self.id.chars().take(8).collect(),
        }
    }

    /// `Combined score to maximize: 1.23\nkey: 0.50; other: 3`, the metric
    /// block shown to models.
    pub fn perf_str(&self) -> String {
        perf_str(self.combined_score, &self.public_metrics)
    }
}

pub fn perf_str(combined_score: f64, public_metrics: &JsonMap) -> String {
    let mut out = format!("Combined score to maximize: {combined_score:.2}\n");
    let parts: Vec<String> = public_metrics
        .iter()
        .map(|(key, value)| match value {
            Value::Number(n) if n.is_f64() => {
                format!("{key}: {:.2}", n.as_f64().unwrap_or(0.0))
            }
            Value::String(s) => format!("{key}: {s}"),
            other => format!("{key}: {other}"),
        })
        .collect();
    out.push_str(&parts.join("; "));
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn perf_str_formats_floats_and_other_values() {
        let mut metrics = JsonMap::new();
        metrics.insert("radius_sum".into(), json!(2.123456));
        metrics.insert("n".into(), json!(26));
        metrics.insert("note".into(), json!("ok"));
        assert_eq!(
            perf_str(2.123456, &metrics),
            "Combined score to maximize: 2.12\nradius_sum: 2.12; n: 26; note: ok"
        );
    }

    #[test]
    fn program_round_trips_through_json_with_missing_fields() {
        let program: Program = serde_json::from_value(json!({"id": "a", "code": "x"})).unwrap();
        assert_eq!(program.language, "python");
        assert!(!program.correct);
        assert_eq!(program.label(), "a");
    }
}
