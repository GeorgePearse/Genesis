//! The AI review: several independent reviewers and an area chair.
//!
//! Each reviewer fills in a NeurIPS-style form; the meta-review aggregates
//! them, and its numeric scores are replaced by the rounded mean of the
//! reviewers' scores, so the headline number is never one model call's
//! opinion. The ensemble design follows the Agentic Scientist's
//! `aireview.py` (Sakana AI, Apache-2.0).

use crate::config::ReviewConfig;
use crate::paper::load_manifest;
use crate::prompts;
use anyhow::{bail, Context, Result};
use futures::future::join_all;
use genesis_core::fsutil;
use genesis_llm::{extract_json, LlmClient, LlmRequest, Message};
use genesis_scientist::artifacts::{Review, ReviewRecord};
use genesis_scientist::PhaseContext;
use serde_json::{json, Value};

/// Keys of a review file the pipeline knows.
pub const REVIEW_KEYS: &[&str] = &["ai_review", "before_revision", "after_revision"];
/// Characters of manuscript a reviewer is shown at most.
const MAX_MANUSCRIPT_CHARS: usize = 120_000;

/// The numeric fields of the form with their inclusive ranges.
pub const SCORE_RANGES: &[(&str, f64, f64)] = &[
    ("Originality", 1.0, 4.0),
    ("Quality", 1.0, 4.0),
    ("Clarity", 1.0, 4.0),
    ("Significance", 1.0, 4.0),
    ("Soundness", 1.0, 4.0),
    ("Presentation", 1.0, 4.0),
    ("Contribution", 1.0, 4.0),
    ("Overall", 1.0, 10.0),
    ("Confidence", 1.0, 5.0),
];

/// Check a filled-in form and read it. Every field must be present and
/// every score within its range.
pub fn validate_review(value: &Value) -> Result<Review> {
    if !value.is_object() {
        bail!("the review is not a JSON object");
    }
    if value["Summary"]
        .as_str()
        .is_none_or(|text| text.trim().is_empty())
    {
        bail!("`Summary` is missing or empty");
    }
    for list in ["Strengths", "Weaknesses", "Questions", "Limitations"] {
        let Some(items) = value[list].as_array() else {
            bail!("`{list}` must be a list of strings");
        };
        if items.iter().any(|item| !item.is_string()) {
            bail!("`{list}` must be a list of strings");
        }
    }
    if !value["Ethical Concerns"].is_boolean() {
        bail!("`Ethical Concerns` must be true or false");
    }
    for (field, low, high) in SCORE_RANGES {
        let Some(score) = value[*field].as_f64() else {
            bail!("`{field}` must be a number");
        };
        if !(*low..=*high).contains(&score) {
            bail!("`{field}` is {score}; it must be between {low} and {high}");
        }
    }
    if !matches!(value["Decision"].as_str(), Some("Accept" | "Reject")) {
        bail!("`Decision` must be \"Accept\" or \"Reject\"");
    }
    serde_json::from_value(value.clone()).context("the review does not fit the form")
}

/// The form in a reply: the JSON after `REVIEW JSON:`, or failing that the
/// first JSON document in the reply.
pub fn parse_review(reply: &str) -> Result<Review> {
    let body = reply
        .rsplit_once("REVIEW JSON:")
        .map_or(reply, |(_, after)| after);
    let value = extract_json(body)
        .or_else(|| extract_json(reply))
        .context("the reply holds no JSON review")?;
    validate_review(&value)
}

/// Replace every numeric score of `meta` with the rounded mean of the
/// reviewers' scores.
pub fn average_scores(meta: &mut Review, reviews: &[Review]) {
    if reviews.is_empty() {
        return;
    }
    let mean = |score: fn(&Review) -> f64| {
        (reviews.iter().map(score).sum::<f64>() / reviews.len() as f64).round()
    };
    meta.originality = mean(|review| review.originality);
    meta.quality = mean(|review| review.quality);
    meta.clarity = mean(|review| review.clarity);
    meta.significance = mean(|review| review.significance);
    meta.soundness = mean(|review| review.soundness);
    meta.presentation = mean(|review| review.presentation);
    meta.contribution = mean(|review| review.contribution);
    meta.overall = mean(|review| review.overall);
    meta.confidence = mean(|review| review.confidence);
}

async fn fill_form(
    llm: &LlmClient,
    config: &ReviewConfig,
    stage: &str,
    system: &str,
    user: String,
) -> Result<Review> {
    let mut request = LlmRequest::new(config.model.clone(), system, user)
        .temperature(1.0)
        .max_tokens(config.max_tokens);
    let mut problem = anyhow::anyhow!("no attempt was made");
    for _ in 0..config.attempts.max(1) {
        let response = llm.query(stage, request.clone()).await?;
        match parse_review(&response.content) {
            Ok(review) => return Ok(review),
            Err(error) => {
                request.messages.push(Message::assistant(response.content));
                request.messages.push(Message::user(format!(
                    "That review could not be used: {error:#}. Reply again in the required format, with every field of the form."
                )));
                problem = error;
            }
        }
    }
    Err(problem)
}

/// Review any manuscript text: `reviewer_count` independent reviews, then a
/// meta-review with averaged scores. Errors when no reviewer returned a
/// usable form.
pub async fn review_manuscript(
    llm: &LlmClient,
    config: &ReviewConfig,
    manuscript: &str,
) -> Result<Review> {
    if manuscript.trim().chars().count() < 100 {
        bail!("the manuscript is too short to review");
    }
    let paper = fsutil::truncate(manuscript, MAX_MANUSCRIPT_CHARS);
    let user = format!(
        "{}\n\nHere is the paper you are asked to review:\n```\n{paper}\n```",
        prompts::REVIEW_FORM
    );
    let outcomes = join_all(
        (0..config.reviewer_count.max(1))
            .map(|_| fill_form(llm, config, "review", prompts::REVIEW_SYSTEM, user.clone())),
    )
    .await;
    let mut failures = Vec::new();
    let mut reviews = Vec::new();
    for outcome in outcomes {
        match outcome {
            Ok(review) => reviews.push(review),
            Err(error) => failures.push(format!("{error:#}")),
        }
    }
    if reviews.is_empty() {
        bail!(
            "no reviewer returned a usable review: {}",
            failures.join("; ")
        );
    }

    let listing: Vec<String> = reviews
        .iter()
        .enumerate()
        .map(|(index, review)| {
            format!(
                "Review {}:\n```json\n{}\n```",
                index + 1,
                serde_json::to_string_pretty(review).unwrap_or_default()
            )
        })
        .collect();
    let meta_user = format!(
        "{}\n\nThe paper was reviewed by {} reviewers. Their reviews:\n\n{}\n\nWrite the meta-review in the same format.",
        prompts::REVIEW_FORM,
        reviews.len(),
        listing.join("\n\n")
    );
    let mut meta = match fill_form(
        llm,
        config,
        "meta_review",
        prompts::META_REVIEW_SYSTEM,
        meta_user,
    )
    .await
    {
        Ok(meta) => meta,
        Err(_) => {
            // Without a usable meta-review, the reviewer nearest the median
            // overall score stands in, with the majority decision.
            let mut ordered: Vec<&Review> = reviews.iter().collect();
            ordered.sort_by(|a, b| a.overall.total_cmp(&b.overall));
            let mut stand_in = ordered[ordered.len() / 2].clone();
            let accepts = reviews
                .iter()
                .filter(|review| review.decision == "Accept")
                .count();
            stand_in.decision = if accepts * 2 > reviews.len() {
                "Accept"
            } else {
                "Reject"
            }
            .to_string();
            stand_in
        }
    };
    average_scores(&mut meta, &reviews);
    Ok(meta)
}

/// Review the published paper and record the outcome under `key` in its
/// manifest. A failed review is recorded there too, and returned as `Err`;
/// it never removes the paper.
pub async fn run_review(
    context: &PhaseContext,
    config: &ReviewConfig,
    key: &str,
) -> Result<Review> {
    if !REVIEW_KEYS.contains(&key) {
        bail!(
            "unknown review key `{key}`; expected one of: {}",
            REVIEW_KEYS.join(", ")
        );
    }
    let paths = &context.paths;
    let dir = paths.paper();
    let mut manifest = load_manifest(paths)?;
    let outcome = async {
        let manuscript = std::fs::read_to_string(paths.resolve(&manifest.paper_md))
            .with_context(|| format!("reading {}", manifest.paper_md))?;
        review_manuscript(&context.llm, config, &manuscript).await
    }
    .await;

    let file = dir.join(format!("{key}.json"));
    let record = match &outcome {
        Ok(review) => {
            fsutil::write_json(&file, review)?;
            ReviewRecord {
                status: "completed".to_string(),
                review_json: Some(paths.relative(&file)),
                overall: Some(review.overall),
                decision: Some(review.decision.clone()),
                error: None,
            }
        }
        Err(error) => {
            let _ = std::fs::remove_file(&file);
            ReviewRecord {
                status: "failed".to_string(),
                error: Some(format!("{error:#}")),
                ..ReviewRecord::default()
            }
        }
    };
    context.events.info(
        "review_result",
        json!({"key": key, "status": record.status, "overall": record.overall, "decision": record.decision, "error": record.error}),
    );
    manifest.reviews.insert(key.to_string(), record);
    fsutil::write_json(&dir.join("latest_manifest.json"), &manifest)?;
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(overall: f64, decision: &str) -> Value {
        json!({
            "Summary": "The paper reports a program search.", "Strengths": ["clear"], "Weaknesses": ["one task"],
            "Originality": 2, "Quality": 3, "Clarity": 3, "Significance": 2, "Questions": [], "Limitations": ["scope"],
            "Ethical Concerns": false, "Soundness": 3, "Presentation": 3, "Contribution": 2,
            "Overall": overall, "Confidence": 4, "Decision": decision
        })
    }

    #[test]
    fn the_form_is_validated_field_by_field() {
        assert_eq!(validate_review(&form(4.0, "Reject")).unwrap().overall, 4.0);
        let mut out_of_range = form(11.0, "Reject");
        assert!(validate_review(&out_of_range)
            .unwrap_err()
            .to_string()
            .contains("`Overall` is 11"));
        out_of_range["Overall"] = json!(5);
        out_of_range["Soundness"] = json!(0);
        assert!(validate_review(&out_of_range).is_err());
        let mut missing = form(4.0, "Reject");
        missing.as_object_mut().unwrap().remove("Weaknesses");
        assert!(validate_review(&missing)
            .unwrap_err()
            .to_string()
            .contains("`Weaknesses`"));
        assert!(validate_review(&form(4.0, "Maybe")).is_err());
        let mut text_score = form(4.0, "Accept");
        text_score["Clarity"] = json!("high");
        assert!(validate_review(&text_score).is_err());
        assert!(validate_review(&json!([])).is_err());
    }

    #[test]
    fn a_reply_is_read_after_its_marker() {
        let reply = format!(
            "THOUGHT:\nThe paper {{is}} thin.\n\nREVIEW JSON:\n```json\n{}\n```",
            form(3.0, "Reject")
        );
        assert_eq!(parse_review(&reply).unwrap().decision, "Reject");
        assert_eq!(
            parse_review(&form(6.0, "Accept").to_string())
                .unwrap()
                .overall,
            6.0
        );
        assert!(parse_review("THOUGHT: no json here").is_err());
    }

    #[test]
    fn scores_are_the_rounded_mean_of_the_reviewers() {
        let reviews: Vec<Review> = [3.0, 4.0, 6.0]
            .iter()
            .map(|overall| validate_review(&form(*overall, "Reject")).unwrap())
            .collect();
        let mut meta = validate_review(&form(9.0, "Accept")).unwrap();
        meta.clarity = 1.0;
        average_scores(&mut meta, &reviews);
        assert_eq!(
            (meta.overall, meta.clarity, meta.confidence),
            (4.0, 3.0, 4.0)
        );
        assert_eq!(
            meta.decision, "Accept",
            "the decision is the meta-reviewer's"
        );
        let untouched = meta.clone();
        average_scores(&mut meta, &[]);
        assert_eq!(meta, untouched);
    }
}
