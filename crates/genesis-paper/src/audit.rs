//! The evidence audit: does the manuscript say only what the run measured?
//!
//! The deterministic layer checks every number in the prose against the
//! grounding data, and every claim against the sources it cites. An
//! optional model layer reads the manuscript for overclaiming that numbers
//! cannot show; its line-oriented reply contract and parser rules are the
//! Agentic Scientist's (Sakana AI, Apache-2.0).

use crate::data::PaperData;
use crate::draft::{Draft, SECTIONS};
use crate::numbers::{extract_numbers, STRUCTURAL};
use crate::prompts;
use crate::references::Reference;
use anyhow::{bail, Result};
use genesis_core::fsutil;
use genesis_llm::{LlmClient, LlmRequest, SamplingKwargs};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

pub const MAX_MODEL_FINDINGS: usize = 20;
const SEPARATOR: &str = " || ";
const FIELDS: [&str; 6] = ["id", "section", "claim", "issue", "evidence", "fix"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct AuditFinding {
    pub id: String,
    pub section: String,
    pub claim: String,
    pub issue: String,
    pub evidence: String,
    pub fix: String,
}

/// `paper/paper_evidence_audit.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct AuditReport {
    /// `pass` or `fail`.
    pub verdict: String,
    pub summary: String,
    pub findings: Vec<AuditFinding>,
    pub finding_count: usize,
    /// The model that ran the second layer, when one did.
    pub model_layer: Option<String>,
    /// Why the model layer, though configured, contributed nothing.
    pub model_layer_error: Option<String>,
    /// Digests of the audited files, by file name.
    pub input_sha256: BTreeMap<String, String>,
}

impl AuditReport {
    pub fn passed(&self) -> bool {
        self.verdict == "pass"
    }
}

/// Check the prose and the claims against the grounding data.
pub fn deterministic_audit(
    draft: &Draft,
    data: &PaperData,
    references: &[Reference],
) -> Vec<AuditFinding> {
    let mut findings = Vec::new();
    let mut grounded = data.numeric_values();
    grounded.extend_from_slice(STRUCTURAL);

    for spec in SECTIONS {
        let mut reported: HashSet<String> = HashSet::new();
        for literal in extract_numbers(draft.section(spec.key)) {
            if literal.matches_any(&grounded) || !reported.insert(literal.text.clone()) {
                continue;
            }
            findings.push(AuditFinding {
                id: format!("number-{}-{}", spec.key, reported.len()),
                section: spec.heading.to_string(),
                claim: literal.text.clone(),
                issue: format!("the number {} does not match any grounding source at the precision shown", literal.text),
                evidence: "paper_data.json grounding_sources".to_string(),
                fix: format!(
                    "Replace {} with the value of a grounding source, rounded correctly, or remove the sentence's number.",
                    literal.text
                ),
            });
        }
    }

    let reference_ids: HashSet<&str> = references
        .iter()
        .map(|reference| reference.id.as_str())
        .collect();
    let citation_keys: HashSet<&str> = references
        .iter()
        .map(|reference| reference.citation_key.as_str())
        .collect();
    for (index, claim) in draft.claims.iter().enumerate() {
        let number = index + 1;
        let unknown: Vec<&str> = claim
            .source_ids
            .iter()
            .map(String::as_str)
            .filter(|id| !data.grounding_sources.contains_key(*id) && !reference_ids.contains(id))
            .collect();
        if !unknown.is_empty() {
            findings.push(AuditFinding {
                id: format!("claim-{number}-unknown-source"),
                section: "claims_grounding.json".to_string(),
                claim: claim.claim.clone(),
                issue: format!("the claim cites grounding ids that do not exist: {}", unknown.join(", ")),
                evidence: "paper_data.json grounding_sources".to_string(),
                fix: "Cite existing grounding ids for this claim, or remove the claim from the manuscript.".to_string(),
            });
        }
        let missing_keys: Vec<&str> = claim
            .citation_keys
            .iter()
            .map(String::as_str)
            .filter(|key| !citation_keys.contains(key))
            .collect();
        if !missing_keys.is_empty() {
            findings.push(AuditFinding {
                id: format!("claim-{number}-unknown-citation"),
                section: "claims_grounding.json".to_string(),
                claim: claim.claim.clone(),
                issue: format!(
                    "the claim cites references that were not retrieved: {}",
                    missing_keys.join(", ")
                ),
                evidence: "references_provenance.json".to_string(),
                fix: "Cite only retrieved references, or drop the citation.".to_string(),
            });
        }
        let mut cited = data.values_of(&claim.source_ids);
        cited.extend_from_slice(STRUCTURAL);
        let unsupported: Vec<String> = extract_numbers(&claim.claim)
            .into_iter()
            .filter(|literal| !literal.matches_any(&cited))
            .map(|literal| literal.text)
            .collect();
        if !unsupported.is_empty() {
            findings.push(AuditFinding {
                id: format!("claim-{number}-number"),
                section: "claims_grounding.json".to_string(),
                claim: claim.claim.clone(),
                issue: format!(
                    "the claim states {} but none of the sources it cites has that value",
                    unsupported.join(", ")
                ),
                evidence: format!(
                    "values of {}",
                    if claim.source_ids.is_empty() {
                        "(no sources cited)".to_string()
                    } else {
                        claim.source_ids.join(", ")
                    }
                ),
                fix: "Cite the grounding source each number comes from, or correct the number."
                    .to_string(),
            });
        }
    }
    findings
}

/// What the model layer returned.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelAudit {
    pub pass: bool,
    pub summary: String,
    pub findings: Vec<AuditFinding>,
}

/// A value the contract can carry: one non-empty trimmed line that cannot
/// be mistaken for, or run into, a field separator.
fn single_line(text: &str) -> bool {
    !text.is_empty()
        && text == text.trim()
        && !text.contains(['\n', '\r'])
        && !format!(" {text} ").contains(SEPARATOR)
}

/// Print an audit in the reply contract. `None` when it cannot be printed:
/// a verdict that disagrees with the findings, too many findings, or a
/// field that is empty, spans lines or contains the separator.
pub fn print_model_audit(audit: &ModelAudit) -> Option<String> {
    if audit.pass != audit.findings.is_empty()
        || audit.findings.len() > MAX_MODEL_FINDINGS
        || !single_line(&audit.summary)
    {
        return None;
    }
    let mut out = format!(
        "AUDIT_VERDICT: {}\nAUDIT_SUMMARY: {}\nAUDIT_FINDING_COUNT: {}\n",
        if audit.pass { "PASS" } else { "FAIL" },
        audit.summary,
        audit.findings.len()
    );
    for finding in &audit.findings {
        let values = [
            &finding.id,
            &finding.section,
            &finding.claim,
            &finding.issue,
            &finding.evidence,
            &finding.fix,
        ];
        if !values.iter().all(|value| single_line(value)) {
            return None;
        }
        let fields: Vec<String> = FIELDS
            .iter()
            .zip(values)
            .map(|(key, value)| format!("{key}={value}"))
            .collect();
        out.push_str(&format!("AUDIT_FINDING: {}\n", fields.join(SEPARATOR)));
    }
    Some(out)
}

/// Parse the reply contract: a verdict, a one-line summary, a declared
/// count, and exactly that many findings of exactly six fields each. `PASS`
/// means zero findings and `FAIL` at least one; twenty at most.
pub fn parse_model_audit(text: &str) -> Result<ModelAudit> {
    let mut verdict = None;
    let mut summary = None;
    let mut declared = None;
    let mut findings = Vec::new();
    for line in text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("```"))
    {
        if let Some(value) = line.strip_prefix("AUDIT_VERDICT:") {
            verdict = Some(match value.trim() {
                "PASS" => true,
                "FAIL" => false,
                other => bail!("AUDIT_VERDICT must be PASS or FAIL, not `{other}`"),
            });
        } else if let Some(value) = line.strip_prefix("AUDIT_SUMMARY:") {
            summary = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("AUDIT_FINDING_COUNT:") {
            declared = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| anyhow::anyhow!("AUDIT_FINDING_COUNT is not an integer"))?,
            );
        } else if let Some(value) = line.strip_prefix("AUDIT_FINDING:") {
            let parts: Vec<&str> = value.trim().split(SEPARATOR).collect();
            if parts.len() != FIELDS.len() {
                bail!(
                    "an AUDIT_FINDING has {} fields; it needs exactly {}",
                    parts.len(),
                    FIELDS.len()
                );
            }
            let mut values = Vec::with_capacity(FIELDS.len());
            for (key, part) in FIELDS.iter().zip(parts) {
                let Some(value) = part.strip_prefix(&format!("{key}=")) else {
                    bail!("an AUDIT_FINDING field is not `{key}=…`");
                };
                if value.trim().is_empty() {
                    bail!("the `{key}` of an AUDIT_FINDING is empty");
                }
                values.push(value.trim().to_string());
            }
            let mut values = values.into_iter();
            let mut next = || values.next().unwrap_or_default();
            findings.push(AuditFinding {
                id: next(),
                section: next(),
                claim: next(),
                issue: next(),
                evidence: next(),
                fix: next(),
            });
        }
    }
    let Some(pass) = verdict else {
        bail!("the audit reply has no AUDIT_VERDICT line")
    };
    let Some(summary) = summary.filter(|summary| !summary.is_empty()) else {
        bail!("the audit reply has no AUDIT_SUMMARY line")
    };
    let Some(declared) = declared else {
        bail!("the audit reply has no AUDIT_FINDING_COUNT line")
    };
    if declared != findings.len() {
        bail!(
            "AUDIT_FINDING_COUNT is {declared} but the reply holds {} findings",
            findings.len()
        );
    }
    if findings.len() > MAX_MODEL_FINDINGS {
        bail!("the audit reply holds more than {MAX_MODEL_FINDINGS} findings");
    }
    if pass != findings.is_empty() {
        bail!("a PASS verdict must have no findings and a FAIL verdict at least one");
    }
    Ok(ModelAudit {
        pass,
        summary,
        findings,
    })
}

/// Run the model layer. An unreadable reply is retried once with the
/// parser's complaint.
pub async fn model_audit(
    llm: &LlmClient,
    model: &str,
    kwargs: &SamplingKwargs,
    packet: &str,
    rng: &mut (dyn rand::RngCore + Send),
) -> Result<ModelAudit> {
    let mut request = LlmRequest::new(model, prompts::AUDIT_SYSTEM, packet);
    kwargs.apply(&mut request, rng);
    request.temperature = Some(0.0);
    let first = llm.query("paper_audit", request.clone()).await?;
    let problem = match parse_model_audit(&first.content) {
        Ok(audit) => return Ok(audit),
        Err(problem) => problem,
    };
    request
        .messages
        .push(genesis_llm::Message::assistant(first.content));
    request.messages.push(genesis_llm::Message::user(format!(
        "That reply broke the output contract: {problem}. Reply again with only the contract lines."
    )));
    parse_model_audit(&llm.query("paper_audit", request).await?.content)
}

/// Digests of the files an audit read.
pub fn input_digests(dir: &Path) -> BTreeMap<String, String> {
    [
        "paper.tex",
        "sections.json",
        "claims_grounding.json",
        "paper_data.json",
        "references.bib",
        "references_provenance.json",
    ]
    .iter()
    .filter_map(|name| {
        fsutil::sha256_file(&dir.join(name))
            .ok()
            .map(|digest| (name.to_string(), digest))
    })
    .collect()
}

/// Combine the layers into the report.
pub fn report(
    dir: &Path,
    deterministic: Vec<AuditFinding>,
    model: Option<(String, Result<ModelAudit>)>,
) -> AuditReport {
    let mut findings = deterministic;
    let deterministic_count = findings.len();
    let mut model_layer = None;
    let mut model_layer_error = None;
    let mut model_summary = None;
    if let Some((name, outcome)) = model {
        match outcome {
            Ok(audit) => {
                model_layer = Some(name);
                model_summary = Some(audit.summary);
                findings.extend(audit.findings);
            }
            Err(error) => model_layer_error = Some(format!("{name}: {error:#}")),
        }
    }
    let summary = if findings.is_empty() {
        match model_summary {
            Some(summary) => format!("Every number and claim is grounded. Model layer: {summary}"),
            None => {
                "Every number in the prose and every claim matches the grounding data.".to_string()
            }
        }
    } else {
        format!(
            "{} finding(s): {} from the numeric and claim checks, {} from the model layer.",
            findings.len(),
            deterministic_count,
            findings.len() - deterministic_count
        )
    };
    AuditReport {
        verdict: if findings.is_empty() { "pass" } else { "fail" }.to_string(),
        summary,
        finding_count: findings.len(),
        findings,
        model_layer,
        model_layer_error,
        input_sha256: input_digests(dir),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GroundingSource;
    use crate::draft::Claim;
    use serde_json::json;

    fn data() -> PaperData {
        let mut data = PaperData::default();
        for (id, value) in [
            ("result.best_correct_score", json!(2.6351)),
            ("result.seed_score", json!(1.9)),
            ("result.total_attempts", json!(12)),
            ("final.qualification.outcome", json!("qualified")),
        ] {
            data.grounding_sources.insert(
                id.to_string(),
                GroundingSource {
                    id: id.to_string(),
                    kind: "metric".into(),
                    value,
                    description: String::new(),
                },
            );
        }
        data
    }

    fn draft(results: &str, claims: Vec<Claim>) -> Draft {
        let mut draft = Draft {
            claims,
            ..Default::default()
        };
        draft.sections.insert("results".into(), results.to_string());
        draft
    }

    #[test]
    fn grounded_numbers_pass_and_invented_ones_are_reported_once() {
        let clean = draft(
            "The best program scores 2.64 against 1.9 for the seed over 12 proposals, with a 95% interval; see Table 3.",
            vec![Claim { claim: "The best program scores 2.64".into(), source_ids: vec!["result.best_correct_score".into()], citation_keys: vec![] }],
        );
        assert!(deterministic_audit(&clean, &data(), &[]).is_empty());

        let invented = draft("It reaches 2.71, a 37% gain, and again 2.71.", vec![]);
        let findings = deterministic_audit(&invented, &data(), &[]);
        assert_eq!(
            findings
                .iter()
                .map(|f| f.claim.as_str())
                .collect::<Vec<_>>(),
            ["2.71", "37%"]
        );
        assert_eq!(
            (findings[0].id.as_str(), findings[0].section.as_str()),
            ("number-results-1", "Results")
        );
    }

    #[test]
    fn claims_must_cite_existing_sources_that_carry_their_numbers() {
        let references = vec![Reference {
            id: "ref.s2.abc".into(),
            citation_key: "smith2020".into(),
            ..Default::default()
        }];
        let claims = vec![
            Claim {
                claim: "The best score is 2.64".into(),
                source_ids: vec!["result.seed_score".into()],
                citation_keys: vec![],
            },
            Claim {
                claim: "Prior work evolves programs".into(),
                source_ids: vec!["ref.s2.abc".into(), "result.nope".into()],
                citation_keys: vec!["smith2020".into(), "ghost".into()],
            },
            Claim {
                claim: "The seed scores 1.9".into(),
                source_ids: vec![],
                citation_keys: vec![],
            },
        ];
        let findings = deterministic_audit(&draft("", claims), &data(), &references);
        let ids: Vec<&str> = findings.iter().map(|f| f.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "claim-1-number",
                "claim-2-unknown-source",
                "claim-2-unknown-citation",
                "claim-3-number"
            ]
        );
        assert!(
            findings[1].issue.contains("result.nope") && !findings[1].issue.contains("ref.s2.abc")
        );
        assert!(findings[3].evidence.contains("no sources cited"));
    }

    #[test]
    fn the_reply_contract_is_parsed_strictly() {
        let pass = parse_model_audit(
            "AUDIT_VERDICT: PASS\nAUDIT_SUMMARY: consistent\nAUDIT_FINDING_COUNT: 0\n",
        )
        .unwrap();
        assert!(pass.pass && pass.findings.is_empty());
        let fail = parse_model_audit(
            "```text\nAUDIT_VERDICT: FAIL\nAUDIT_SUMMARY: one overclaim\nAUDIT_FINDING_COUNT: 1\n\
             AUDIT_FINDING: id=r1 || section=Results || claim=always better || issue=one task only || evidence=paper_data || fix=narrow it\n```",
        )
        .unwrap();
        assert_eq!(
            (fail.findings[0].id.as_str(), fail.findings[0].fix.as_str()),
            ("r1", "narrow it")
        );
        for broken in [
            "AUDIT_SUMMARY: x\nAUDIT_FINDING_COUNT: 0",
            "AUDIT_VERDICT: MAYBE\nAUDIT_SUMMARY: x\nAUDIT_FINDING_COUNT: 0",
            "AUDIT_VERDICT: PASS\nAUDIT_FINDING_COUNT: 0",
            "AUDIT_VERDICT: PASS\nAUDIT_SUMMARY: x\nAUDIT_FINDING_COUNT: 1",
            "AUDIT_VERDICT: FAIL\nAUDIT_SUMMARY: x\nAUDIT_FINDING_COUNT: 0",
            "AUDIT_VERDICT: FAIL\nAUDIT_SUMMARY: x\nAUDIT_FINDING_COUNT: 1\nAUDIT_FINDING: id=a || section=b || claim=c",
            "AUDIT_VERDICT: FAIL\nAUDIT_SUMMARY: x\nAUDIT_FINDING_COUNT: 1\nAUDIT_FINDING: id=a || claim=c || section=b || issue=d || evidence=e || fix=f",
            "AUDIT_VERDICT: PASS\nAUDIT_SUMMARY: x\nAUDIT_FINDING_COUNT: 1\nAUDIT_FINDING: id=a || section=b || claim=c || issue=d || evidence=e || fix=f",
            "AUDIT_VERDICT: FAIL\nAUDIT_SUMMARY: x\nAUDIT_FINDING_COUNT: many",
        ] {
            assert!(parse_model_audit(broken).is_err(), "{broken}");
        }
        let printed = print_model_audit(&fail).unwrap();
        assert_eq!(parse_model_audit(&printed).unwrap(), fail);
        assert!(print_model_audit(&ModelAudit {
            pass: true,
            summary: "x".into(),
            findings: fail.findings.clone()
        })
        .is_none());
    }

    #[test]
    fn the_report_merges_layers_and_records_a_failed_model_layer() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("paper.tex"), "x").unwrap();
        let clean = report(
            dir.path(),
            vec![],
            Some((
                "judge".into(),
                Ok(ModelAudit {
                    pass: true,
                    summary: "fine".into(),
                    findings: vec![],
                }),
            )),
        );
        assert!(clean.passed() && clean.model_layer.as_deref() == Some("judge"));
        assert_eq!(clean.input_sha256.len(), 1);
        let finding = AuditFinding {
            id: "m1".into(),
            ..Default::default()
        };
        let failed = report(
            dir.path(),
            vec![],
            Some((
                "judge".into(),
                Ok(ModelAudit {
                    pass: false,
                    summary: "bad".into(),
                    findings: vec![finding],
                }),
            )),
        );
        assert_eq!((failed.verdict.as_str(), failed.finding_count), ("fail", 1));
        let errored = report(
            dir.path(),
            vec![],
            Some(("judge".into(), Err(anyhow::anyhow!("timeout")))),
        );
        assert!(errored.passed() && errored.model_layer.is_none());
        assert!(errored.model_layer_error.unwrap().contains("timeout"));
    }
}
