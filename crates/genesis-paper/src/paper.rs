//! The paper phase: from a finished run to a manuscript whose every number
//! is traced to an artifact.
//!
//! Gates, in order: preparation, grounding data, reference plan and
//! retrieval, strategy, draft, figures and tables, validation and compile,
//! evidence audit, repair rounds, finalization. The sequence follows the
//! Agentic Scientist's paper stage (Sakana AI, Apache-2.0); here the model
//! returns prose and the harness does everything else.

use crate::audit::{deterministic_audit, model_audit, report as audit_report, AuditReport};
use crate::config::PaperConfig;
use crate::data::{build_paper_data, EvidenceSource, PaperData};
use crate::draft::{parse_reply, Draft, DraftReply, SECTIONS};
use crate::prompts;
use crate::references::{validate_query_plan, write_bibtex, QueryPlan, Reference, Retriever};
use crate::render::{assemble, available_floats, format_number};
use crate::validate::{validate, ValidationReport};
use anyhow::{bail, Context, Result};
use genesis_core::fsutil;
use genesis_llm::{LlmClient, LlmRequest, Message, SamplingKwargs};
use genesis_scientist::artifacts::PaperManifest;
use genesis_scientist::{PhaseContext, RunPaths};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::path::Path;

/// Longest program excerpt shown to the writer, in characters.
const MAX_CODE_CHARS: usize = 12_000;
const MAX_PACKET_CHARS: usize = 30_000;

/// What validation and the audit found about one rendering.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Checked {
    pub validation: ValidationReport,
    pub audit: AuditReport,
}

impl Checked {
    pub fn passes(&self) -> bool {
        self.validation.ok && self.audit.passed()
    }

    /// The findings as the repair model reads them.
    pub fn feedback(&self) -> String {
        let mut out = String::new();
        if !self.validation.hard_errors.is_empty() {
            out.push_str("## Validation errors\n");
            for error in &self.validation.hard_errors {
                let _ = writeln!(out, "- {error}");
            }
            if let Some(log) = &self.validation.latex_log_tail {
                let _ = writeln!(out, "\nEnd of the LaTeX log:\n{}", fsutil::tail(log, 1500));
            }
        }
        if !self.audit.findings.is_empty() {
            out.push_str("\n## Evidence audit findings\n");
            for finding in &self.audit.findings {
                let _ = writeln!(
                    out,
                    "- [{}] {}: \"{}\": {}. Evidence: {}. Fix: {}",
                    finding.id,
                    finding.section,
                    finding.claim,
                    finding.issue,
                    finding.evidence,
                    finding.fix
                );
            }
        }
        out
    }
}

/// One paper directory and the fixed inputs every rendering of it uses.
pub(crate) struct Build<'a> {
    pub dir: &'a Path,
    pub data: &'a PaperData,
    pub references: &'a [Reference],
    pub config: &'a PaperConfig,
}

pub(crate) fn pick<'a>(models: &'a [String], rng: &mut ChaCha8Rng) -> Result<&'a str> {
    models
        .choose(rng)
        .map(String::as_str)
        .context("no model is configured for this stage")
}

impl Build<'_> {
    /// Render `draft` into the directory, then validate and audit it.
    pub async fn check(
        &self,
        draft: &Draft,
        llm: &LlmClient,
        rng: &mut ChaCha8Rng,
    ) -> Result<Checked> {
        let assembled = assemble(draft, self.data, self.references);
        fsutil::write_json(&self.dir.join("sections.json"), draft)?;
        fsutil::write_json(
            &self.dir.join("claims_grounding.json"),
            &json!({"claims": draft.claims}),
        )?;
        fsutil::write_text(&self.dir.join("paper.tex"), &assembled.tex)?;
        fsutil::write_text(&self.dir.join("paper.md"), &assembled.markdown)?;
        let figures = self.dir.join("figures");
        let _ = std::fs::remove_dir_all(&figures);
        for (name, svg) in &assembled.figures {
            fsutil::write_text(&figures.join(name), svg)?;
        }

        let validation = validate(
            self.dir,
            draft,
            self.data,
            self.references,
            &assembled,
            self.config.min_refs,
            self.config.compile_pdf,
        )
        .await?;
        let deterministic = deterministic_audit(draft, self.data, self.references);
        // The model layer is only worth its cost on a draft the cheap
        // checks already accept.
        let model =
            if deterministic.is_empty() && validation.ok && !self.config.audit_models.is_empty() {
                let name = pick(&self.config.audit_models, rng)?.to_string();
                let packet = audit_packet(draft, self.data, &assembled.markdown);
                let outcome = model_audit(llm, &name, &self.config.kwargs, &packet, rng).await;
                Some((name, outcome))
            } else {
                None
            };
        let audit = audit_report(self.dir, deterministic, model);
        fsutil::write_json(&self.dir.join("paper_evidence_audit.json"), &audit)?;
        Ok(Checked { validation, audit })
    }
}

fn value_text(value: &Value) -> String {
    match value {
        Value::Number(number) => number.as_f64().map_or(number.to_string(), format_full),
        Value::String(text) => format!("\"{text}\""),
        other => other.to_string(),
    }
}

/// Full precision for the writer, so that any rounding it chooses is a
/// rounding of the real value.
fn format_full(value: f64) -> String {
    if value == value.trunc() && value.abs() < 1e15 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// Everything the run established, as the writer, the strategist and the
/// auditor read it.
pub(crate) fn evidence_brief(data: &PaperData, references: &[Reference]) -> String {
    let mut out = format!("# Task\n{}\n\n# Studied program\n", data.task_goal.trim());
    let selection = &data.selection;
    let _ = writeln!(
        out,
        "selection_kind: {} (search outcome: {}); generation {}; search-time combined score {}; correct: {}",
        selection.selection_kind,
        selection.investigation_outcome,
        selection.generation,
        format_full(selection.combined_score),
        selection.correct
    );
    out.push_str("\n# Grounding sources (the only numbers you may state)\n");
    let mut constants: Vec<String> = Vec::new();
    for source in data.grounding_sources.values() {
        if source.kind == "source_constant" {
            constants.push(value_text(&source.value));
        } else {
            let _ = writeln!(
                out,
                "- {} = {}: {}",
                source.id,
                value_text(&source.value),
                source.description
            );
        }
    }
    if !constants.is_empty() {
        let _ = writeln!(out, "- numeric constants in the studied program's source (ids source.selected_constant_*): {}", constants.join(", "));
    }
    out.push_str("\n# Evidence status\n");
    for (name, source) in &data.evidence_sources {
        let _ = writeln!(
            out,
            "- {name}: {}{}",
            source.status,
            source
                .note
                .as_ref()
                .map(|note| format!(" ({note})"))
                .unwrap_or_default()
        );
    }
    if let Some(summary) = &data.ablations {
        out.push_str("\n# Ablation variants\n");
        for variant in &summary.variants {
            let _ = writeln!(
                out,
                "- {} ({}): {}. Hypothesis: {}. Status: {}.",
                variant.spec.variant_id,
                variant.spec.kind,
                variant.spec.title,
                variant.spec.hypothesis,
                variant.status
            );
        }
    }
    if let Some(manifest) = &data.posthoc {
        out.push_str("\n# Post-hoc analyses\n");
        for analysis in &manifest.valid_analyses {
            let _ = writeln!(
                out,
                "- {} ({}): {}",
                analysis.spec.id, analysis.spec.title, analysis.claim
            );
        }
    }
    out.push_str("\n# Limitations that must be stated\n");
    if data.limitations.is_empty() {
        out.push_str("(none recorded)\n");
    }
    for limitation in &data.limitations {
        let _ = writeln!(out, "- {limitation}");
    }
    out.push_str("\n# References you may cite\n");
    if references.is_empty() {
        out.push_str("(none: write without citations)\n");
    }
    for reference in references {
        let year = reference
            .year
            .map_or("n.d.".to_string(), |year| year.to_string());
        let _ = writeln!(
            out,
            "- [@{}] {} ({year}{})",
            reference.citation_key,
            reference.title,
            if reference.venue.is_empty() {
                String::new()
            } else {
                format!(", {}", reference.venue)
            }
        );
    }
    out.push_str("\n# Tables and figures you can place\n");
    for float in available_floats(data) {
        let _ = writeln!(
            out,
            "- {{{{{}:{}}}}}: {}",
            float.kind.word().to_uppercase(),
            float.name,
            float.description
        );
    }
    if let Some(conclusions) = &data.distilled_conclusions {
        let _ = write!(
            out,
            "\n# Distilled conclusions of the search\n{conclusions}\n"
        );
    }
    let _ = write!(
        out,
        "\n# Source of the studied program\n```{}\n{}\n```\n",
        data.language,
        fsutil::truncate(&data.selected_code, MAX_CODE_CHARS)
    );
    out
}

fn audit_packet(draft: &Draft, data: &PaperData, markdown: &str) -> String {
    let mut out = String::from("# Grounding sources\n");
    for source in data
        .grounding_sources
        .values()
        .filter(|source| source.kind != "source_constant")
    {
        let _ = writeln!(
            out,
            "- {} = {}: {}",
            source.id,
            value_text(&source.value),
            source.description
        );
    }
    out.push_str("\n# Recorded limitations\n");
    for limitation in &data.limitations {
        let _ = writeln!(out, "- {limitation}");
    }
    let claims = serde_json::to_string_pretty(&json!({"claims": draft.claims})).unwrap_or_default();
    let _ = write!(
        out,
        "\n# claims_grounding.json\n{}\n\n# Manuscript\n{}\n",
        fsutil::truncate(&claims, 14_000),
        fsutil::truncate(markdown, MAX_PACKET_CHARS)
    );
    out
}

fn tagged(draft: &Draft) -> String {
    let mut out = format!("<TITLE>{}</TITLE>\n", draft.title);
    for spec in SECTIONS {
        let _ = writeln!(
            out,
            "<{}>\n{}\n</{}>",
            spec.tag,
            draft.section(spec.key),
            spec.tag
        );
    }
    let claims = serde_json::to_string_pretty(&json!({"claims": draft.claims})).unwrap_or_default();
    let _ = writeln!(out, "<CLAIMS>\n{claims}\n</CLAIMS>");
    out
}

fn writer_system(output: &str) -> String {
    format!(
        "{}\n\n{}\n\n{output}",
        prompts::DRAFT_SYSTEM_HEAD,
        prompts::WRITING_RULES
    )
}

async fn gather_references(
    llm: &LlmClient,
    config: &PaperConfig,
    data: &PaperData,
    rng: &mut ChaCha8Rng,
) -> (Vec<Reference>, EvidenceSource, Option<QueryPlan>) {
    let missing = |note: String| EvidenceSource {
        status: "missing".to_string(),
        path: None,
        note: Some(note),
    };
    if !config.retrieve_references || config.max_refs == 0 {
        return (
            Vec::new(),
            missing("reference retrieval is turned off".to_string()),
            None,
        );
    }
    let plan = async {
        let user = format!(
            "# Task\n{}\n\n# What the run found\nselection: {}; best correct score: {}\n\nPlan the queries.",
            data.task_goal.trim(),
            data.selection.selection_kind,
            data.best_correct.as_ref().map_or("none".to_string(), |best| format_number(best.combined_score))
        );
        let mut request = LlmRequest::new(pick(&config.models, rng)?, prompts::REFERENCE_PLAN_SYSTEM, user);
        config.kwargs.apply(&mut request, rng);
        let (reply, response) = llm.query_json("reference_plan", request.clone()).await?;
        match validate_query_plan(&reply) {
            Ok(plan) => Ok(plan),
            Err(problem) => {
                request.messages.push(Message::assistant(response.content));
                request.messages.push(Message::user(format!("That plan was rejected: {problem}. Reply again with one corrected JSON object.")));
                let (reply, _) = llm.query_json("reference_plan", request).await?;
                validate_query_plan(&reply)
            }
        }
    }
    .await;
    let plan = match plan {
        Ok(plan) => plan,
        Err(error) => {
            return (
                Vec::new(),
                missing(format!("no usable query plan: {error:#}")),
                None,
            )
        }
    };
    let retriever = Retriever::new(&config.semantic_scholar_url, &config.openalex_url);
    let (references, failures) = retriever.retrieve(&plan, config.max_refs).await;
    let source = if references.is_empty() {
        missing(format!(
            "retrieval returned nothing{}",
            if failures.is_empty() {
                String::new()
            } else {
                format!(": {}", failures.join("; "))
            }
        ))
    } else {
        EvidenceSource {
            status: "included".to_string(),
            path: Some("paper/references_provenance.json".to_string()),
            note: Some(format!(
                "{} references from {} queries",
                references.len(),
                plan.queries.len()
            )),
        }
    };
    (references, source, Some(plan))
}

fn default_strategy(data: &PaperData) -> Value {
    let thesis = match (data.is_negative_result(), &data.best_correct) {
        (true, _) => "The search did not find a program that improves on the seed.".to_string(),
        (false, Some(best)) => format!(
            "The search found a correct program that scores above the seed (`{}`).",
            best.name
        ),
        (false, None) => "The search found a correct program.".to_string(),
    };
    json!({
        "recommended_title": format!("Automated program search on {}", data.task_name),
        "central_question": "What did an automated program search find on this task, and how well is it supported?",
        "central_thesis": thesis,
        "narrative_rationale": "default framing: report the search result and the evidence behind it",
        "results_focus": [],
        "notes_for_writer": [],
    })
}

async fn plan_strategy(
    llm: &LlmClient,
    config: &PaperConfig,
    brief: &str,
    rng: &mut ChaCha8Rng,
) -> Result<Value> {
    let mut request = LlmRequest::new(pick(&config.models, rng)?, prompts::STRATEGY_SYSTEM, brief);
    config.kwargs.apply(&mut request, rng);
    let (reply, _) = llm.query_json("paper_strategy", request).await?;
    for key in ["recommended_title", "central_question", "central_thesis"] {
        if reply
            .get(key)
            .and_then(Value::as_str)
            .is_none_or(|text| text.trim().is_empty())
        {
            bail!("the strategy has no `{key}`");
        }
    }
    Ok(reply)
}

async fn ask_writer(
    llm: &LlmClient,
    stage: &str,
    model: &str,
    kwargs: &SamplingKwargs,
    system: String,
    user: String,
    rng: &mut ChaCha8Rng,
) -> Result<(LlmRequest, String)> {
    let mut request = LlmRequest::new(model, system, user);
    kwargs.apply(&mut request, rng);
    let response = llm.query(stage, request.clone()).await?;
    Ok((request, response.content))
}

async fn write_draft(
    llm: &LlmClient,
    config: &PaperConfig,
    brief: &str,
    strategy: &Value,
    rng: &mut ChaCha8Rng,
) -> Result<Draft> {
    let user = format!(
        "{brief}\n# Strategy\n{}\n\nWrite the manuscript.",
        serde_json::to_string_pretty(strategy).unwrap_or_default()
    );
    let model = pick(&config.models, rng)?;
    let (mut request, content) = ask_writer(
        llm,
        "paper_draft",
        model,
        &config.kwargs,
        writer_system(prompts::DRAFT_OUTPUT),
        user,
        rng,
    )
    .await?;
    let mut draft = Draft::default();
    draft.merge(&parse_reply(&content));
    if draft.title.trim().is_empty() {
        draft.title = strategy["recommended_title"]
            .as_str()
            .unwrap_or("")
            .to_string();
    }
    let missing = draft.missing_sections();
    if !missing.is_empty() || draft.claims.is_empty() {
        let mut wanted: Vec<String> = SECTIONS
            .iter()
            .filter(|spec| missing.contains(&spec.key))
            .map(|spec| format!("<{}>", spec.tag))
            .collect();
        if draft.claims.is_empty() {
            wanted.push("<CLAIMS>".to_string());
        }
        request.messages.push(Message::assistant(content));
        request.messages.push(Message::user(format!(
            "That reply is incomplete: {} missing or empty. Reply with those tags only, complete.",
            wanted.join(", ")
        )));
        let response = llm.query("paper_draft", request).await?;
        draft.merge(&parse_reply(&response.content));
    }
    let missing = draft.missing_sections();
    if !missing.is_empty() {
        bail!(
            "the draft is missing sections after a corrective retry: {}",
            missing.join(", ")
        );
    }
    Ok(draft)
}

/// One request to rewrite parts of a manuscript.
pub(crate) struct Rewrite<'a> {
    /// The ledger stage: `paper_repair` or `paper_revision`.
    pub stage: &'a str,
    /// The output rules that end the system prompt.
    pub output_rules: &'a str,
    pub brief: &'a str,
    /// What must change, as the model reads it.
    pub feedback: &'a str,
}

/// Ask for a corrected manuscript. The reply holds only what changed.
pub(crate) async fn rewrite(
    llm: &LlmClient,
    config: &PaperConfig,
    request: &Rewrite<'_>,
    draft: &Draft,
    rng: &mut ChaCha8Rng,
) -> Result<DraftReply> {
    let user = format!(
        "{}\n# Current manuscript\n{}\n# What must change\n{}",
        request.brief,
        tagged(draft),
        request.feedback
    );
    let model = pick(&config.models, rng)?;
    let system = writer_system(request.output_rules);
    let (_, content) =
        ask_writer(llm, request.stage, model, &config.kwargs, system, user, rng).await?;
    Ok(parse_reply(&content))
}

/// The manifest of the paper in `dir`, as validation and the audit left it.
pub(crate) fn manifest(
    paths: &RunPaths,
    dir: &Path,
    config: &PaperConfig,
    checked: &Checked,
) -> PaperManifest {
    let file = |name: &str| paths.relative(&dir.join(name));
    PaperManifest {
        generated_at: chrono::Utc::now().to_rfc3339(),
        paper_format: config.paper_format.clone(),
        paper_dir: paths.relative(dir),
        paper_tex: file("paper.tex"),
        paper_md: file("paper.md"),
        paper_pdf: dir.join("paper.pdf").is_file().then(|| file("paper.pdf")),
        paper_data: file("paper_data.json"),
        references_bib: file("references.bib"),
        claims_grounding: file("claims_grounding.json"),
        evidence_audit: file("paper_evidence_audit.json"),
        evidence_audit_verdict: checked.audit.verdict.clone(),
        validation_report: file("validation_report.json"),
        validation_ok: checked.validation.ok,
        ..PaperManifest::default()
    }
}

/// Write the paper for the run in `context.paths`.
///
/// Errors when there is nothing to write about, when the final outcome
/// policy forbids publication, or when the model cannot produce a complete
/// draft. A paper that still fails validation or the audit after every
/// repair round is published with that recorded in its manifest.
pub async fn run_paper(context: &PhaseContext, config: &PaperConfig) -> Result<PaperManifest> {
    config.validate()?;
    let paths = &context.paths;
    let dir = paths.paper();
    let gate = |name: &str, state: &str, detail: Value| {
        context.events.info(
            "paper_gate",
            json!({"gate": name, "state": state, "detail": detail}),
        );
    };
    let mut rng = ChaCha8Rng::seed_from_u64(context.seed ^ 0x00C0_FFEE);

    gate("preparation", "start", Value::Null);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let _ = std::fs::remove_file(dir.join("latest_manifest.json"));

    let mut data = build_paper_data(
        paths,
        &context.task.name,
        &context.task.goal,
        &context.task.language,
        &config.final_outcome_policy,
    )?;
    gate(
        "grounding",
        "done",
        json!({"grounding_sources": data.grounding_sources.len(), "selection_kind": data.selection.selection_kind}),
    );

    let (references, source, plan) = gather_references(&context.llm, config, &data, &mut rng).await;
    if references.is_empty() {
        data.limitations.push(
            "No references were retrieved for this paper, so it has no bibliography and its related-work discussion cites nothing."
                .to_string(),
        );
    }
    gate(
        "references",
        "done",
        json!({"count": references.len(), "status": source.status, "note": source.note}),
    );
    data.evidence_sources
        .insert("references".to_string(), source);
    fsutil::write_json(&dir.join("paper_data.json"), &data)?;
    fsutil::write_json(&dir.join("reference_query_plan.json"), &plan)?;
    fsutil::write_text(&dir.join("references.bib"), &write_bibtex(&references))?;
    fsutil::write_json(&dir.join("references_provenance.json"), &references)?;

    let brief = evidence_brief(&data, &references);
    let strategy = match plan_strategy(&context.llm, config, &brief, &mut rng).await {
        Ok(strategy) => strategy,
        Err(error) => {
            context.events.warn(
                "paper_strategy_fallback",
                json!({"error": format!("{error:#}")}),
            );
            default_strategy(&data)
        }
    };
    fsutil::write_json(&dir.join("paper_strategy.json"), &strategy)?;
    gate(
        "strategy",
        "done",
        json!({"title": strategy["recommended_title"]}),
    );

    let mut draft = write_draft(&context.llm, config, &brief, &strategy, &mut rng)
        .await
        .context("drafting the paper")?;
    gate("draft", "done", json!({"claims": draft.claims.len()}));

    let build = Build {
        dir: &dir,
        data: &data,
        references: &references,
        config,
    };
    let mut checked = build.check(&draft, &context.llm, &mut rng).await?;
    gate(
        "evidence_audit",
        if checked.passes() { "pass" } else { "fail" },
        json!({"validation_errors": checked.validation.hard_errors.len(), "findings": checked.audit.finding_count}),
    );
    let mut rounds = 0;
    while !checked.passes() && rounds < config.max_repair_rounds {
        rounds += 1;
        let feedback = checked.feedback();
        let request = Rewrite {
            stage: "paper_repair",
            output_rules: prompts::REPAIR_OUTPUT,
            brief: &brief,
            feedback: &feedback,
        };
        let reply = match rewrite(&context.llm, config, &request, &draft, &mut rng).await {
            Ok(reply) => reply,
            Err(error) => {
                context.events.warn(
                    "paper_repair_failed",
                    json!({"round": rounds, "error": format!("{error:#}")}),
                );
                continue;
            }
        };
        if draft.merge(&reply).is_empty() {
            context
                .events
                .warn("paper_repair_unchanged", json!({"round": rounds}));
            break;
        }
        checked = build.check(&draft, &context.llm, &mut rng).await?;
        gate(
            "repair",
            if checked.passes() { "pass" } else { "fail" },
            json!({"round": rounds, "validation_errors": checked.validation.hard_errors.len(), "findings": checked.audit.finding_count}),
        );
    }

    let manifest = manifest(paths, &dir, config, &checked);
    fsutil::write_json(&dir.join("latest_manifest.json"), &manifest)?;
    gate(
        "finalization",
        "done",
        json!({
            "validation_ok": manifest.validation_ok,
            "evidence_audit_verdict": manifest.evidence_audit_verdict,
            "pdf": manifest.paper_pdf,
            "repair_rounds": rounds,
        }),
    );
    Ok(manifest)
}

/// The published paper's manifest, if there is one.
pub fn load_manifest(paths: &RunPaths) -> Result<PaperManifest> {
    let path = paths.paper().join("latest_manifest.json");
    fsutil::read_json(&path).with_context(|| {
        format!(
            "no published paper: {} is missing or unreadable",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditFinding;
    use crate::data::GroundingSource;

    #[test]
    fn the_brief_lists_grounding_floats_and_limitations() {
        let mut data = PaperData {
            task_goal: "Pack circles.".into(),
            language: "python".into(),
            selected_code: "x = 26".into(),
            ..Default::default()
        };
        data.limitations
            .push("No ablation study is available.".into());
        for (id, kind, value) in [
            ("result.best_correct_score", "metric", json!(2.635)),
            (
                "source.selected_constant_001",
                "source_constant",
                json!(26.0),
            ),
            (
                "final.qualification.outcome",
                "final_qualification",
                json!("qualified"),
            ),
        ] {
            data.grounding_sources.insert(
                id.into(),
                GroundingSource {
                    id: id.into(),
                    kind: kind.into(),
                    value,
                    description: "d".into(),
                },
            );
        }
        let references = vec![Reference {
            citation_key: "smith2020".into(),
            title: "Evolving".into(),
            year: Some(2020),
            ..Default::default()
        }];
        let brief = evidence_brief(&data, &references);
        for needle in [
            "# Task\nPack circles.",
            "- result.best_correct_score = 2.635: d",
            "- final.qualification.outcome = \"qualified\": d",
            "numeric constants in the studied program's source (ids source.selected_constant_*): 26",
            "- No ablation study is available.",
            "- [@smith2020] Evolving (2020)",
            "- {{TABLE:run_summary}}: counts of the search",
            "```python\nx = 26\n```",
        ] {
            assert!(brief.contains(needle), "missing `{needle}` in:\n{brief}");
        }
        assert!(evidence_brief(&data, &[]).contains("(none: write without citations)"));
    }

    #[test]
    fn feedback_lists_errors_and_findings() {
        let checked = Checked {
            validation: ValidationReport {
                hard_errors: vec!["the Method section is missing or empty".into()],
                ..Default::default()
            },
            audit: AuditReport {
                verdict: "fail".into(),
                findings: vec![AuditFinding {
                    id: "number-results-1".into(),
                    section: "Results".into(),
                    claim: "2.71".into(),
                    issue: "unsupported".into(),
                    evidence: "paper_data".into(),
                    fix: "remove it".into(),
                }],
                ..Default::default()
            },
        };
        assert!(!checked.passes());
        let feedback = checked.feedback();
        assert!(feedback.contains("## Validation errors\n- the Method section is missing or empty"));
        assert!(feedback.contains("- [number-results-1] Results: \"2.71\": unsupported. Evidence: paper_data. Fix: remove it"));
    }

    #[test]
    fn a_draft_round_trips_through_its_tagged_form() {
        let mut draft = Draft {
            title: "T".into(),
            ..Default::default()
        };
        for spec in SECTIONS {
            draft.sections.insert(
                spec.key.into(),
                format!("{} body\n\nsecond paragraph", spec.heading),
            );
        }
        draft.claims.push(crate::draft::Claim {
            claim: "c".into(),
            source_ids: vec!["a.b".into()],
            citation_keys: vec![],
        });
        let mut copy = Draft::default();
        copy.merge(&parse_reply(&tagged(&draft)));
        assert_eq!(copy, draft);
    }
}
