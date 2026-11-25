//! Post-hoc analysis: what the search's own artifacts say about how the
//! selected program came to be.
//!
//! An analysis here never evaluates a program. It is one of the built-in
//! families in [`analyses`], computed from the run database, the attempt
//! journal and the cost ledger. A model proposes which analyses a paper
//! would want and why, a second call reviews the proposals, and code
//! applies the hard rules: nothing that is an ablation, nothing that
//! retrains or re-evaluates, one analysis per family, at most
//! `max_selected_analyses`. Without a model the families propose
//! themselves.
//!
//! Each analysis publishes a table, a chart, its key numbers and a claim.
//! A model may word the claim, but every number in it is checked against
//! the key numbers by [`grounding`]; text that fails the check is replaced
//! by the sentence the analysis writes from its own numbers.
//!
//! The stages (generate, select, execute, publish) and the proposal schema
//! follow the Agentic Scientist's post-hoc pipeline (`posthoc_generate.md`,
//! `posthoc_select.md`; Sakana AI, Apache-2.0).

pub mod analyses;
pub mod grounding;
pub mod svg;

use crate::common::{self, ask_json, UNTRUSTED};
use crate::csv;
use analyses::{family, registry, AnalysisOutput, RunData};
use anyhow::{bail, Context, Result};
use genesis_core::fsutil;
use genesis_evolve::{DatabaseConfig, ProgramDb};
use genesis_llm::SamplingKwargs;
use genesis_scientist::artifacts::{
    AnalysisResult, AnalysisSpec, PosthocManifest, RejectedAnalysis,
};
use genesis_scientist::selection::SelectedProgram;
use genesis_scientist::PhaseContext;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PosthocConfig {
    /// Models that propose, review and word analyses. With none, the
    /// built-in families propose themselves and state their own findings.
    pub models: Vec<String>,
    pub kwargs: SamplingKwargs,
    pub max_selected_analyses: usize,
}

impl Default for PosthocConfig {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            kwargs: SamplingKwargs::default(),
            max_selected_analyses: 3,
        }
    }
}

const GENERATE_SYSTEM: &str = "You propose post-hoc analyses of a completed program-search run, for a paper about the \
program it selected. Optimize for paper value, grounding, and diversity across analysis families.

This pipeline is analysis-only:
- do not propose ablations, perturbation sweeps, retraining, fine-tuning, or evaluator reruns
- every candidate must use existing artifacts only: the run database, logs, metrics and saved results
- set `requires_retrain` to false for every candidate
- every `expected_outputs` entry must be a safe relative file path, never a prose description
- `family` must be one of the analysis families listed in the prompt. The family fixes what is computed; your \
hypothesis and expected claim say what the paper should learn from it. Do not propose a family that has no data.
Zero candidates is allowed.

# Output
Respond with exactly one JSON object and nothing else:
{
  \"decision\": \"propose_analyses\" or \"no_relevant_analyses\",
  \"summary\": \"short summary\",
  \"analyses\": [
    {
      \"id\": \"snake_case_identifier\",
      \"title\": \"...\",
      \"family\": \"one of the listed families\",
      \"hypothesis\": \"...\",
      \"relevance_reason\": \"...\",
      \"feasibility_reason\": \"...\",
      \"required_artifacts\": [\"investigation/programs.sqlite\"],
      \"requires_retrain\": false,
      \"implementation_plan\": [\"...\"],
      \"expected_claim\": \"...\",
      \"expected_outputs\": [\"outputs/name.csv\"]
    }
  ]
}";

const SELECT_SYSTEM: &str =
    "You review proposed post-hoc analyses of a completed program-search run.

Selection criteria:
- relevant to the paper story
- feasible with the available artifacts
- likely to produce grounded evidence
- not shallow or duplicative
- diverse across analysis families
- reject anything that is an ablation or otherwise belongs in the separate ablation study
- reject anything that requires retraining, fine-tuning, or any evaluator rerun
- prefer analyses that only inspect existing artifacts

# Output
Respond with exactly one JSON object and nothing else:
{\"decisions\": [{\"id\": \"string\", \"accepted\": true, \"reason\": \"string\"}]}";

const INTERPRET_SYSTEM: &str = "You state the finding of one post-hoc analysis of a program-search run. You are given \
the analysis' hypothesis, its key numbers and the start of its table.

Write:
- `claim`: one sentence saying what the numbers show.
- `interpretation`: two to four sentences on what this means for understanding the method, without speculating \
beyond the numbers. Say so plainly when the numbers do not support the hypothesis.

Every number you write must be one of the key numbers, rounded to the digits you show; write a share as a \
percentage. Do not compute new numbers and do not invent any. Put names of programs, models and patch types in \
backticks.

# Output
Respond with exactly one JSON object and nothing else: {\"claim\": \"...\", \"interpretation\": \"...\"}";

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

/// The well-formed analyses of a model's proposal, in its order. An
/// analysis needs a title, a family and an id that is safe as a directory
/// name; anything else is skipped.
pub fn parse_proposals(bundle: &Value) -> Vec<AnalysisSpec> {
    static ID: OnceLock<Regex> = OnceLock::new();
    let id_pattern =
        ID.get_or_init(|| Regex::new(r"^[a-z0-9][a-z0-9_-]{0,127}$").expect("valid regex"));
    let Some(items) = bundle.get("analyses").and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter(|item| item.is_object())
        .filter_map(|item| {
            let id = Some(text(item, "id"))
                .filter(|id| !id.is_empty())
                .unwrap_or_else(|| text(item, "analysis_id"));
            let (title, family) = (
                text(item, "title"),
                text(item, "family").to_ascii_lowercase(),
            );
            if !id_pattern.is_match(&id) || title.is_empty() || family.is_empty() {
                return None;
            }
            Some(AnalysisSpec {
                id,
                title,
                family,
                hypothesis: text(item, "hypothesis"),
                relevance_reason: text(item, "relevance_reason"),
                feasibility_reason: text(item, "feasibility_reason"),
                required_artifacts: text_list(item, "required_artifacts"),
                requires_retrain: item
                    .get("requires_retrain")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                implementation_plan: text_list(item, "implementation_plan"),
                expected_claim: text(item, "expected_claim"),
                expected_outputs: text_list(item, "expected_outputs"),
            })
        })
        .collect()
}

/// One proposal per built-in family that has data for this run.
pub fn default_proposals(data: &RunData) -> Vec<AnalysisSpec> {
    registry()
        .iter()
        .filter(|family| (family.compute)(data).is_some())
        .map(|family| AnalysisSpec {
            id: family.name.to_string(),
            title: family.title.to_string(),
            family: family.name.to_string(),
            hypothesis: family.hypothesis.to_string(),
            relevance_reason: "A built-in analysis of the search's own record.".to_string(),
            feasibility_reason: "Computed from the run database without evaluating anything."
                .to_string(),
            required_artifacts: vec!["investigation/programs.sqlite".to_string()],
            requires_retrain: false,
            implementation_plan: vec![format!("Compute {}.", family.description)],
            expected_claim: String::new(),
            expected_outputs: vec![
                format!("outputs/{}.csv", family.name),
                format!("outputs/{}.svg", family.name),
            ],
        })
        .collect()
}

/// The reviewer's verdicts by analysis id: accepted or not, and why.
pub fn parse_decisions(reply: &Value) -> BTreeMap<String, (bool, String)> {
    let mut decisions = BTreeMap::new();
    for item in reply
        .get("decisions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = text(item, "id");
        let Some(accepted) = item.get("accepted").and_then(Value::as_bool) else {
            continue;
        };
        if !id.is_empty() {
            decisions.insert(id, (accepted, text(item, "reason")));
        }
    }
    decisions
}

const REEVALUATION_WORDING: [&str; 7] = [
    "retrain",
    "train from scratch",
    "fine-tun",
    "finetun",
    "invoke evaluate.py",
    "rerun evaluation",
    "run evaluation",
];

/// Decide which proposed analyses are carried out.
///
/// Proposals are taken in the proposer's order. The reviewer's rejections
/// apply first; a proposal the reviewer did not mention is kept. Then the
/// hard rules, in this order: no ablations, nothing that needs retraining,
/// no plan that retrains or evaluates again, only known families, no
/// repeated id, one analysis per family, and the cap.
pub fn select_analyses(
    candidates: &[AnalysisSpec],
    decisions: &BTreeMap<String, (bool, String)>,
    max_selected: usize,
) -> (Vec<AnalysisSpec>, Vec<RejectedAnalysis>) {
    let (mut accepted, mut rejected) = (Vec::new(), Vec::new());
    let (mut ids, mut families) = (HashSet::new(), HashSet::new());
    for candidate in candidates {
        let plan = candidate.implementation_plan.join(" ").to_lowercase();
        let first_time = ids.insert(candidate.id.clone());
        let reason = if let Some((_, why)) = decisions
            .get(&candidate.id)
            .filter(|(accepted, _)| !accepted)
        {
            Some(if why.is_empty() {
                "rejected by the reviewer".to_string()
            } else {
                format!("rejected by the reviewer: {why}")
            })
        } else if candidate.family.contains("ablation") {
            Some("ablations belong to the ablation study".to_string())
        } else if candidate.requires_retrain {
            Some("requires retraining".to_string())
        } else if REEVALUATION_WORDING
            .iter()
            .any(|wording| plan.contains(wording))
        {
            Some("the plan retrains or evaluates again; post-hoc analysis uses existing artifacts only".to_string())
        } else if family(&candidate.family).is_none() {
            Some(format!("unknown analysis family `{}`", candidate.family))
        } else if !first_time {
            Some("duplicate analysis id".to_string())
        } else if families.contains(&candidate.family) {
            Some("family already represented".to_string())
        } else if accepted.len() >= max_selected {
            Some("max_selected_analyses cap reached".to_string())
        } else {
            None
        };
        match reason {
            Some(reason) => rejected.push(RejectedAnalysis {
                analysis_id: candidate.id.clone(),
                title: candidate.title.clone(),
                family: candidate.family.clone(),
                reason,
            }),
            None => {
                families.insert(candidate.family.clone());
                accepted.push(candidate.clone());
            }
        }
    }
    (accepted, rejected)
}

fn load_run_data(context: &PhaseContext, selected: &SelectedProgram) -> Result<RunData> {
    let investigation = context.paths.investigation();
    let db_path: PathBuf = investigation.join(DatabaseConfig::default().db_path);
    if !db_path.is_file() {
        bail!(
            "post-hoc analysis needs the run database, and {} does not exist",
            db_path.display()
        );
    }
    let db = ProgramDb::open_read_only(&db_path)?;
    Ok(RunData {
        task_name: context.task.name.clone(),
        programs: db.all()?,
        selected_program_id: Some(selected.selection.program_id.clone()),
        attempts: db.attempts()?,
        cost: fsutil::read_json_opt(
            &context
                .paths
                .run_dir
                .join("metrics")
                .join("cost_summary.json"),
        ),
    })
}

struct Posthoc<'a> {
    context: &'a PhaseContext,
    config: &'a PosthocConfig,
    selected: &'a SelectedProgram,
    data: &'a RunData,
    dir: PathBuf,
}

impl Posthoc<'_> {
    fn run_facts(&self) -> Value {
        let selection = &self.selected.selection;
        json!({
            "task": self.data.task_name,
            "selection_kind": selection.selection_kind,
            "selected_score": selection.combined_score,
            "baseline_score": selection.baseline_score,
            "selected_generation": selection.generation,
            "programs_in_database": self.data.programs.len(),
            "failed_proposals": self.data.attempts.iter().filter(|attempt| attempt.status == "failed").count(),
        })
    }

    /// The candidates: a stored bundle, else a model's proposals, else the
    /// families' own.
    async fn propose(&self) -> Result<Vec<AnalysisSpec>> {
        let events = &self.context.events;
        let path = self.dir.join("generation").join("posthoc_bundle.json");
        if let Some(stored) = fsutil::read_json_opt::<Value>(&path) {
            return Ok(parse_proposals(&stored));
        }
        let store = |decision: &str, summary: &str, source: &str, analyses: &[AnalysisSpec]| {
            fsutil::write_json(
                &path,
                &json!({"decision": decision, "summary": summary, "source": source, "analyses": analyses}),
            )
        };
        let fallback = |why: &str| -> Result<Vec<AnalysisSpec>> {
            let analyses = default_proposals(self.data);
            store("propose_analyses", why, "registry", &analyses)?;
            Ok(analyses)
        };
        if self.config.models.is_empty() {
            return fallback(
                "no model is configured; every built-in family with data proposes itself",
            );
        }
        let families: Vec<String> = registry()
            .iter()
            .map(|family| {
                let available = if (family.compute)(self.data).is_some() {
                    "has data"
                } else {
                    "no data in this run"
                };
                format!("- `{}` ({available}): {}", family.name, family.description)
            })
            .collect();
        let user = format!(
            "{UNTRUSTED}\n\nTask:\n```md\n{}\n```\n\nRun facts:\n```json\n{}\n```\n\nAnalysis families:\n{}\n\n\
             Propose the analyses as the JSON object described in the system message.",
            self.context.task.goal,
            serde_json::to_string_pretty(&self.run_facts())?,
            families.join("\n")
        );
        match ask_json(
            &self.context.llm,
            "posthoc_generate",
            &self.config.models,
            &self.config.kwargs,
            GENERATE_SYSTEM,
            user,
            self.context.seed,
        )
        .await
        {
            Ok((reply, _)) => {
                let analyses = parse_proposals(&reply);
                store(
                    reply
                        .get("decision")
                        .and_then(Value::as_str)
                        .unwrap_or("propose_analyses"),
                    reply.get("summary").and_then(Value::as_str).unwrap_or(""),
                    "model",
                    &analyses,
                )?;
                events.info(
                    "analyses_proposed",
                    json!({"candidates": analyses.len(), "source": "model"}),
                );
                Ok(analyses)
            }
            Err(error) => {
                events.warn(
                    "posthoc_generation_failed",
                    json!({"error": format!("{error:#}")}),
                );
                fallback("the proposing model could not be reached; every built-in family with data proposes itself")
            }
        }
    }

    async fn select(
        &self,
        candidates: &[AnalysisSpec],
    ) -> Result<(Vec<AnalysisSpec>, Vec<RejectedAnalysis>)> {
        let path = self.dir.join("generation").join("posthoc_selection.json");
        let ids: Vec<&str> = candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect();
        if let Some(stored) = fsutil::read_json_opt::<Value>(&path) {
            let same = stored.get("max_selected_analyses")
                == Some(&json!(self.config.max_selected_analyses))
                && stored.get("candidate_ids") == Some(&json!(ids));
            let accepted = stored
                .get("accepted")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok());
            let rejected = stored
                .get("rejected")
                .cloned()
                .and_then(|value| serde_json::from_value(value).ok());
            if let (true, Some(accepted), Some(rejected)) = (same, accepted, rejected) {
                return Ok((accepted, rejected));
            }
        }
        let mut decisions = BTreeMap::new();
        if !candidates.is_empty() && !self.config.models.is_empty() {
            let user = format!(
                "{UNTRUSTED}\n\nTask:\n```md\n{}\n```\n\nRun facts:\n```json\n{}\n```\n\nCandidates:\n```json\n{}\n```",
                self.context.task.goal,
                serde_json::to_string_pretty(&self.run_facts())?,
                serde_json::to_string_pretty(candidates)?
            );
            match ask_json(
                &self.context.llm,
                "posthoc_select",
                &self.config.models,
                &self.config.kwargs,
                SELECT_SYSTEM,
                user,
                self.context.seed.wrapping_add(1),
            )
            .await
            {
                Ok((reply, _)) => decisions = parse_decisions(&reply),
                Err(error) => {
                    // Fail open: a reviewer that cannot be reached rejects nothing.
                    self.context.events.warn(
                        "posthoc_selection_failed",
                        json!({"error": format!("{error:#}")}),
                    );
                }
            }
        }
        let (accepted, rejected) =
            select_analyses(candidates, &decisions, self.config.max_selected_analyses);
        fsutil::write_json(
            &path,
            &json!({
                "schema_version": 1,
                "max_selected_analyses": self.config.max_selected_analyses,
                "candidate_ids": ids,
                "accepted": accepted,
                "rejected": rejected,
            }),
        )?;
        self.context.events.info(
            "analyses_selected",
            json!({"accepted": accepted.len(), "rejected": rejected.len()}),
        );
        Ok((accepted, rejected))
    }

    /// A model's wording of the finding, kept only where every number in it
    /// is one of the analysis' key numbers.
    async fn interpret(
        &self,
        spec: &AnalysisSpec,
        output: &AnalysisOutput,
        index: usize,
    ) -> (Option<String>, Option<String>) {
        if self.config.models.is_empty() {
            return (None, None);
        }
        let head: Vec<&Vec<String>> = output.table.rows.iter().take(20).collect();
        let user = format!(
            "{UNTRUSTED}\n\nAnalysis:\n```json\n{}\n```\n\nKey numbers:\n```json\n{}\n```\n\nTable columns: {}\nFirst rows:\n```json\n{}\n```",
            json!({"title": spec.title, "family": spec.family, "hypothesis": spec.hypothesis, "expected_claim": spec.expected_claim}),
            serde_json::to_string_pretty(&output.key_numbers).unwrap_or_default(),
            output.table.header.join(", "),
            serde_json::to_string(&head).unwrap_or_default()
        );
        let reply = ask_json(
            &self.context.llm,
            "posthoc_interpret",
            &self.config.models,
            &self.config.kwargs,
            INTERPRET_SYSTEM,
            user,
            self.context.seed.wrapping_add(10 + index as u64),
        )
        .await;
        let Ok((reply, _)) = reply else {
            return (None, None);
        };
        let grounded = |key: &str| {
            Some(text(&reply, key)).filter(|sentence| {
                !sentence.is_empty() && grounding::claim_is_grounded(sentence, &output.key_numbers)
            })
        };
        let (claim, interpretation) = (grounded("claim"), grounded("interpretation"));
        if claim.is_none() || interpretation.is_none() {
            self.context.events.warn(
                "posthoc_text_not_grounded",
                json!({
                    "analysis": spec.id,
                    "claim_numbers": grounding::ungrounded_numbers(&text(&reply, "claim"), &output.key_numbers),
                    "interpretation_numbers": grounding::ungrounded_numbers(&text(&reply, "interpretation"), &output.key_numbers),
                }),
            );
        }
        (claim, interpretation)
    }

    async fn execute(
        &self,
        spec: &AnalysisSpec,
        index: usize,
    ) -> Result<std::result::Result<AnalysisResult, String>> {
        let Some(family) = family(&spec.family) else {
            return Ok(Err(format!("unknown analysis family `{}`", spec.family)));
        };
        let Some(output) = (family.compute)(self.data) else {
            return Ok(Err(
                "not_executed: this run holds no data for the family".to_string()
            ));
        };
        let paths = &self.context.paths;
        let analysis_dir = self.dir.join("analyses").join(&spec.id);
        let csv_path = analysis_dir
            .join("outputs")
            .join(format!("{}.csv", family.name));
        let svg_path = analysis_dir
            .join("outputs")
            .join(format!("{}.svg", family.name));
        let header: Vec<&str> = output.table.header.iter().map(String::as_str).collect();
        csv::write(&csv_path, &header, &output.table.rows)?;
        fsutil::write_text(&svg_path, &output.chart)?;

        let (model_claim, interpretation) = self.interpret(spec, &output, index).await;
        let claim_source = if model_claim.is_some() {
            "model"
        } else {
            "template"
        };
        let claim = model_claim.unwrap_or_else(|| output.claim.clone());
        let mut executed = spec.clone();
        executed.expected_outputs = vec![
            format!("outputs/{}.csv", family.name),
            format!("outputs/{}.svg", family.name),
        ];
        fsutil::write_json(&analysis_dir.join("analysis_spec.json"), &executed)?;

        let mut summary = format!(
            "# {}\n\nFamily: `{}`.\n\nHypothesis: {}\n\n## Finding\n\n{claim}\n\n",
            spec.title,
            family.name,
            if spec.hypothesis.is_empty() {
                family.hypothesis
            } else {
                &spec.hypothesis
            }
        );
        if let Some(interpretation) = &interpretation {
            summary.push_str(&format!("{interpretation}\n\n"));
        }
        summary.push_str("## Key numbers\n\n| Name | Value |\n|---|---|\n");
        for (name, value) in &output.key_numbers {
            summary.push_str(&format!("| `{}` | {value} |\n", name.replace('|', "\\|")));
        }
        summary.push_str(&format!(
            "\n## Outputs\n\n- `outputs/{0}.csv`: the table behind the finding ({1} rows)\n- `outputs/{0}.svg`: its chart\n",
            family.name,
            output.table.rows.len()
        ));
        let summary_path = analysis_dir.join("analysis_summary.md");
        fsutil::write_text(&summary_path, &summary)?;
        let outputs = vec![paths.relative(&csv_path), paths.relative(&svg_path)];
        fsutil::write_json(
            &analysis_dir.join("analysis_manifest.json"),
            &json!({
                "analysis_id": spec.id,
                "attempt_id": "attempt_1",
                "family": family.name,
                "summary": claim,
                "claim": claim,
                "claim_source": claim_source,
                "interpretation": interpretation,
                "grounded": true,
                "reran_evaluator": false,
                "outputs": executed.expected_outputs,
                "evidence_paths": ["investigation/programs.sqlite"],
                "key_numbers": output.key_numbers,
            }),
        )?;
        self.context.events.info(
            "analysis_completed",
            json!({"analysis": spec.id, "family": family.name, "claim_source": claim_source, "claim": claim}),
        );
        Ok(Ok(AnalysisResult {
            spec: executed,
            analysis_dir: paths.relative(&analysis_dir),
            summary_path: paths.relative(&summary_path),
            outputs,
            claim,
            grounded: true,
            key_numbers: output.key_numbers,
        }))
    }
}

/// Run the post-hoc analyses of the frozen selection under
/// `<run>/posthoc`. A finished directory is returned as it is.
///
/// A manifest without valid analyses is a legitimate result: the caller
/// decides whether that degrades the run.
pub async fn run_posthoc(
    context: &PhaseContext,
    config: &PosthocConfig,
) -> Result<PosthocManifest> {
    let selected = common::selected(&context.paths)?;
    let dir = context.paths.posthoc();
    let events = &context.events;
    let receipt_path = dir.join(".posthoc_complete.json");
    let binding = json!({"program_sha256": selected.selection.program_sha256, "config": config});
    if let Some(receipt) = fsutil::read_json_opt::<Value>(&receipt_path) {
        let manifest = receipt
            .get("manifest")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok());
        if let (true, Some(manifest)) = (receipt.get("binding") == Some(&binding), manifest) {
            events.info("posthoc_reused", json!({}));
            return Ok(manifest);
        }
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let data = load_run_data(context, &selected)?;
    events.info("posthoc_start", json!({"programs": data.programs.len(), "max_selected_analyses": config.max_selected_analyses}));
    let posthoc = Posthoc {
        context,
        config,
        selected: &selected,
        data: &data,
        dir: dir.clone(),
    };

    let candidates = posthoc.propose().await?;
    let (accepted, mut rejected) = posthoc.select(&candidates).await?;
    let wanted: HashSet<&str> = accepted.iter().map(|spec| spec.id.as_str()).collect();
    if let Ok(entries) = std::fs::read_dir(dir.join("analyses")) {
        for entry in entries.flatten() {
            if !wanted.contains(entry.file_name().to_string_lossy().as_ref()) {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
    let mut valid = Vec::new();
    for (index, spec) in accepted.iter().enumerate() {
        if context.cancelled() {
            bail!("post-hoc analysis was interrupted; finished analyses are kept");
        }
        match posthoc.execute(spec, index).await? {
            Ok(result) => valid.push(result),
            Err(reason) => rejected.push(RejectedAnalysis {
                analysis_id: spec.id.clone(),
                title: spec.title.clone(),
                family: spec.family.clone(),
                reason,
            }),
        }
    }

    let manifest = PosthocManifest {
        manifest_version: 1,
        created_at: chrono::Utc::now().to_rfc3339(),
        valid_analyses: valid,
        rejected_analyses: rejected,
    };
    fsutil::write_json(
        &dir.join("ranking.json"),
        &json!({
            "published_at": manifest.created_at,
            "delegated_to_paper_agent": true,
            "selected_analysis_id": null,
            "valid": manifest.valid_analyses.iter().map(|analysis| json!({
                "analysis_id": analysis.spec.id,
                "title": analysis.spec.title,
                "family": analysis.spec.family,
                "analysis_dir": analysis.analysis_dir,
                "summary_path": analysis.summary_path,
                "claim": analysis.claim,
            })).collect::<Vec<_>>(),
            "rejected": manifest.rejected_analyses,
        }),
    )?;
    let latest = dir.join("latest_manifest.json");
    if manifest.valid_analyses.is_empty() {
        let _ = std::fs::remove_file(&latest);
    } else {
        fsutil::write_json(&latest, &manifest)?;
    }
    let status = if manifest.valid_analyses.is_empty() {
        "no_valid_analyses"
    } else {
        "completed"
    };
    fsutil::write_json(
        &dir.join("pipeline_summary.json"),
        &json!({
            "status": status,
            "total_analyses": candidates.len(),
            "total_valid": manifest.valid_analyses.len(),
            "total_rejected": manifest.rejected_analyses.len(),
            "ranking_path": "ranking.json",
            "latest_manifest_path": (!manifest.valid_analyses.is_empty()).then_some("latest_manifest.json"),
        }),
    )?;
    fsutil::write_json(
        &receipt_path,
        &json!({"binding": binding, "manifest": manifest}),
    )?;
    events.info(
        "posthoc_end",
        json!({"status": status, "valid": manifest.valid_analyses.len(), "rejected": manifest.rejected_analyses.len()}),
    );
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str, family: &str) -> AnalysisSpec {
        AnalysisSpec {
            id: id.into(),
            title: id.into(),
            family: family.into(),
            ..Default::default()
        }
    }

    #[test]
    fn proposals_need_a_safe_id_a_title_and_a_family() {
        let bundle = json!({"analyses": [
            {"id": "trajectory", "title": "T", "family": "Search_Trajectory", "requires_retrain": false,
             "implementation_plan": "read the database", "expected_outputs": ["outputs/a.csv"]},
            {"analysis_id": "legacy-id", "title": "L", "family": "lineage_of_best"},
            {"id": "../escape", "title": "x", "family": "search_trajectory"},
            {"id": "Upper", "title": "x", "family": "search_trajectory"},
            {"id": "no_family", "title": "x"},
            {"id": "no_title", "family": "search_trajectory"},
            42
        ]});
        let parsed = parse_proposals(&bundle);
        assert_eq!(parsed.len(), 2);
        assert_eq!(
            (parsed[0].id.as_str(), parsed[0].family.as_str()),
            ("trajectory", "search_trajectory")
        );
        assert_eq!(parsed[0].implementation_plan, ["read the database"]);
        assert_eq!(parsed[1].id, "legacy-id");
        assert!(parse_proposals(&json!({})).is_empty());
    }

    #[test]
    fn the_hard_rules_apply_in_order() {
        let mut retrain = spec("needs_training", "search_trajectory");
        retrain.requires_retrain = true;
        let mut reruns = spec("reruns", "cost_efficiency");
        reruns.implementation_plan = vec!["Rerun evaluation on five seeds".into()];
        let candidates = vec![
            spec("reviewer_said_no", "search_trajectory"),
            spec("remove_restarts", "component_ablation"),
            retrain,
            reruns,
            spec("clusters", "error_clustering"),
            spec("trajectory", "search_trajectory"),
            spec("trajectory", "lineage_of_best"),
            spec("trajectory_again", "search_trajectory"),
            spec("lineage", "lineage_of_best"),
            spec("operators", "operator_effectiveness"),
        ];
        let decisions = parse_decisions(&json!({"decisions": [
            {"id": "reviewer_said_no", "accepted": false, "reason": "shallow"},
            {"id": "lineage", "accepted": true, "reason": "useful"},
            {"id": "operators", "accepted": "maybe"}
        ]}));
        let (accepted, rejected) = select_analyses(&candidates, &decisions, 2);
        assert_eq!(
            accepted.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
            ["trajectory", "lineage"]
        );
        let reasons: Vec<(&str, &str)> = rejected
            .iter()
            .map(|r| (r.analysis_id.as_str(), r.reason.as_str()))
            .collect();
        assert_eq!(
            reasons,
            [
                ("reviewer_said_no", "rejected by the reviewer: shallow"),
                ("remove_restarts", "ablations belong to the ablation study"),
                ("needs_training", "requires retraining"),
                ("reruns", "the plan retrains or evaluates again; post-hoc analysis uses existing artifacts only"),
                ("clusters", "unknown analysis family `error_clustering`"),
                ("trajectory", "duplicate analysis id"),
                ("trajectory_again", "family already represented"),
                ("operators", "max_selected_analyses cap reached"),
            ]
        );
        assert!(select_analyses(&candidates, &BTreeMap::new(), 0)
            .0
            .is_empty());
    }
}
