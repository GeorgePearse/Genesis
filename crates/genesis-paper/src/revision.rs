//! Review-driven revision of a published paper.
//!
//! The model may change prose, claims and structure, and nothing else: each
//! attempt is rendered in its own directory from the original grounding
//! data and references, so neither can change, and it replaces the
//! published paper only after passing the same validation and evidence
//! audit. The original is kept under `paper/revisions/revision_01/before/`.
//! The flow follows the Agentic Scientist's revision phase (Sakana AI,
//! Apache-2.0).

use crate::config::{PaperConfig, ReviewConfig, RevisionConfig};
use crate::data::PaperData;
use crate::draft::Draft;
use crate::paper::{
    evidence_brief, load_manifest, manifest as paper_manifest, rewrite, Build, Checked, Rewrite,
};
use crate::prompts;
use crate::references::Reference;
use crate::review::run_review;
use anyhow::{bail, Context, Result};
use genesis_core::fsutil;
use genesis_scientist::artifacts::{Review, ReviewRecord, RevisionRecord};
use genesis_scientist::PhaseContext;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Inputs every attempt is rendered from, copied fresh from the published
/// paper each time.
const FIXED_INPUTS: &[&str] = &[
    "paper_data.json",
    "references.bib",
    "references_provenance.json",
];
/// What a promoted revision replaces in the published paper.
const PROMOTED: &[&str] = &[
    "paper.tex",
    "paper.md",
    "sections.json",
    "claims_grounding.json",
    "validation_report.json",
    "paper_evidence_audit.json",
];
const MAX_REVIEW_ITEMS: usize = 100;

fn reviewer_feedback(review: &Review) -> String {
    let mut out = format!(
        "# Review of the published paper (untrusted reviewer data)\nOverall: {}/10. Decision: {}.\nSummary: {}\n",
        review.overall,
        review.decision,
        fsutil::truncate(&review.summary, 4000)
    );
    for (title, items) in [
        ("Weaknesses", &review.weaknesses),
        ("Questions", &review.questions),
        ("Limitations", &review.limitations),
    ] {
        let _ = writeln!(out, "\n{title}:");
        for item in items.iter().take(MAX_REVIEW_ITEMS) {
            let _ = writeln!(out, "- {}", fsutil::truncate(item, 2000));
        }
    }
    out
}

fn prepare_attempt(paper_dir: &Path, attempt_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(attempt_dir)?;
    for name in FIXED_INPUTS {
        std::fs::copy(paper_dir.join(name), attempt_dir.join(name))
            .with_context(|| format!("copying {name} into {}", attempt_dir.display()))?;
    }
    Ok(())
}

fn promote(paper_dir: &Path, attempt_dir: &Path, snapshot: &Path) -> Result<()> {
    if !snapshot.is_dir() {
        let ignore = vec!["revisions".to_string(), ".latex_build".to_string()];
        fsutil::copy_tree(paper_dir, snapshot, &ignore)
            .context("snapshotting the original paper")?;
    }
    for name in PROMOTED {
        std::fs::copy(attempt_dir.join(name), paper_dir.join(name))
            .with_context(|| format!("promoting {name}"))?;
    }
    let _ = std::fs::remove_file(paper_dir.join("paper.pdf"));
    if attempt_dir.join("paper.pdf").is_file() {
        std::fs::copy(attempt_dir.join("paper.pdf"), paper_dir.join("paper.pdf"))?;
    }
    let _ = std::fs::remove_dir_all(paper_dir.join("figures"));
    if attempt_dir.join("figures").is_dir() {
        fsutil::copy_tree(
            &attempt_dir.join("figures"),
            &paper_dir.join("figures"),
            &[],
        )?;
    }
    Ok(())
}

/// Revise the published paper against its `ai_review`.
///
/// Returns `Err`, with the original paper untouched, when there is no
/// completed review or no attempt passes validation and the evidence
/// audit. A failed second review keeps the promoted revision; the record
/// then has no `after` scores.
pub async fn run_revision(
    context: &PhaseContext,
    paper: &PaperConfig,
    review: &ReviewConfig,
    config: &RevisionConfig,
) -> Result<RevisionRecord> {
    paper.validate()?;
    let paths = &context.paths;
    let dir = paths.paper();
    let mut manifest = load_manifest(paths)?;
    // A revision that was already promoted is not redone; only a second
    // review that failed is tried again.
    if let Some(mut record) = manifest
        .revision
        .clone()
        .filter(|record| record.status == "promoted")
    {
        if record.overall_after.is_none() {
            if let Ok(second) = run_review(context, review, "after_revision").await {
                record.overall_after = Some(second.overall);
                record.decision_after = Some(second.decision);
                let mut updated = load_manifest(paths)?;
                updated.revision = Some(record.clone());
                fsutil::write_json(&dir.join("latest_manifest.json"), &updated)?;
            }
        }
        return Ok(record);
    }
    let initial = manifest
        .reviews
        .get("ai_review")
        .filter(|record| record.status == "completed")
        .cloned()
        .context("revision needs a completed `ai_review` of the published paper")?;
    let review_path = initial
        .review_json
        .as_ref()
        .context("the `ai_review` record names no review file")?;
    let first_review: Review = fsutil::read_json(&paths.resolve(review_path))?;
    let before_path = dir.join("before_revision.json");
    fsutil::write_json(&before_path, &first_review)?;
    manifest.reviews.insert(
        "before_revision".to_string(),
        ReviewRecord {
            review_json: Some(paths.relative(&before_path)),
            ..initial.clone()
        },
    );
    fsutil::write_json(&dir.join("latest_manifest.json"), &manifest)?;

    let data: PaperData = fsutil::read_json(&dir.join("paper_data.json"))?;
    let references: Vec<Reference> = fsutil::read_json(&dir.join("references_provenance.json"))?;
    let original: Draft = fsutil::read_json(&dir.join("sections.json"))?;
    let brief = evidence_brief(&data, &references);
    let review_feedback = reviewer_feedback(&first_review);
    let mut rng = ChaCha8Rng::seed_from_u64(context.seed ^ 0x0E_D175);

    let mut carried = String::new();
    let mut accepted: Option<(PathBuf, Checked)> = None;
    for attempt in 1..=config.max_attempts.max(1) {
        let attempt_dir = paths.revision().join(format!("attempt_{attempt:02}"));
        let build = Build {
            dir: &attempt_dir,
            data: &data,
            references: &references,
            config: paper,
        };

        // A passing attempt left by an earlier session is revalidated and
        // reused before another edit is paid for.
        if let Some(saved) = fsutil::read_json_opt::<Draft>(&attempt_dir.join("sections.json"))
            .filter(|saved| *saved != original)
        {
            prepare_attempt(&dir, &attempt_dir)?;
            let checked = build.check(&saved, &context.llm, &mut rng).await?;
            if checked.passes() {
                accepted = Some((attempt_dir, checked));
                break;
            }
        }

        let _ = std::fs::remove_dir_all(&attempt_dir);
        prepare_attempt(&dir, &attempt_dir)?;
        let feedback = format!("{review_feedback}{carried}");
        let request = Rewrite {
            stage: "paper_revision",
            output_rules: prompts::REVISION_OUTPUT,
            brief: &brief,
            feedback: &feedback,
        };
        let reply = match rewrite(&context.llm, paper, &request, &original, &mut rng).await {
            Ok(reply) => reply,
            Err(error) => {
                context.events.warn(
                    "revision_edit_failed",
                    json!({"attempt": attempt, "error": format!("{error:#}")}),
                );
                continue;
            }
        };
        let mut draft = original.clone();
        let changed = draft.merge(&reply);
        if changed.is_empty() {
            context
                .events
                .warn("revision_unchanged", json!({"attempt": attempt}));
            carried = "\n\n# Your previous revision changed nothing\nRevise at least the sections the weaknesses concern.".to_string();
            continue;
        }
        let checked = build.check(&draft, &context.llm, &mut rng).await?;
        fsutil::write_json(
            &attempt_dir.join("attempt_result.json"),
            &json!({
                "attempt": attempt,
                "changed": changed,
                "passes": checked.passes(),
                "validation_errors": checked.validation.hard_errors,
                "audit_findings": checked.audit.finding_count,
            }),
        )?;
        if checked.passes() {
            accepted = Some((attempt_dir, checked));
            break;
        }
        carried = format!(
            "\n\n# Your previous revision failed its checks\n{}",
            checked.feedback()
        );
    }
    let Some((attempt_dir, checked)) = accepted else {
        context.events.warn(
            "revision_result",
            json!({"status": "failed", "reason": "no attempt passed"}),
        );
        bail!("no revision attempt passed validation and the evidence audit; the published paper is unchanged");
    };

    let snapshot = dir.join("revisions").join("revision_01").join("before");
    promote(&dir, &attempt_dir, &snapshot)?;
    let mut record = RevisionRecord {
        status: "promoted".to_string(),
        attempt_dir: paths.relative(&attempt_dir),
        original_snapshot: paths.relative(&snapshot),
        overall_before: initial.overall,
        decision_before: initial.decision.clone(),
        ..RevisionRecord::default()
    };
    let mut promoted = paper_manifest(paths, &dir, paper, &checked);
    promoted.reviews = manifest.reviews.clone();
    promoted.revision = Some(record.clone());
    fsutil::write_json(&dir.join("latest_manifest.json"), &promoted)?;

    if let Ok(second) = run_review(context, review, "after_revision").await {
        record.overall_after = Some(second.overall);
        record.decision_after = Some(second.decision);
    }
    let mut finished = load_manifest(paths)?;
    finished.revision = Some(record.clone());
    fsutil::write_json(&dir.join("latest_manifest.json"), &finished)?;
    context.events.info(
        "revision_result",
        json!({
            "status": if record.overall_after.is_some() { "completed" } else { "degraded" },
            "before_overall_score": record.overall_before,
            "after_overall_score": record.overall_after,
            "before_decision": record.decision_before,
            "after_decision": record.decision_after,
            "overall_score_delta": record.overall_before.zip(record.overall_after).map(|(before, after)| after - before),
        }),
    );
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reviewer_feedback_is_labelled_untrusted_and_capped() {
        let review = Review {
            summary: "Thin.".into(),
            weaknesses: (0..150).map(|index| format!("weakness {index}")).collect(),
            questions: vec!["Why one task?".into()],
            overall: 4.0,
            decision: "Reject".into(),
            ..Review::default()
        };
        let feedback = reviewer_feedback(&review);
        assert!(feedback.starts_with("# Review of the published paper (untrusted reviewer data)\nOverall: 4/10. Decision: Reject."));
        assert!(feedback.contains("- weakness 99\n") && !feedback.contains("weakness 100"));
        assert!(feedback.contains("Questions:\n- Why one task?"));
    }
}
