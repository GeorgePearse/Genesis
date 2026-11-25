//! End-to-end tests of the paper, review and revision phases: a real run
//! directory with a real program database and frozen selection, evidence
//! artifacts written with the shared schemas, scripted model replies, and a
//! local server standing in for the reference services.

use axum::extract::Query;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::get;
use axum::{Json, Router};
use genesis_core::stats::{compare, summarize};
use genesis_core::{fsutil, EventLog, Program, Task};
use genesis_evolve::{DatabaseConfig, ProgramDb};
use genesis_llm::{LlmClient, LlmRequest, MockProvider};
use genesis_paper::audit::AuditReport;
use genesis_paper::data::PaperData;
use genesis_paper::draft::Draft;
use genesis_paper::references::{PlannedQuery, QueryPlan, Reference, Retriever};
use genesis_paper::validate::{tex_available, ValidationReport};
use genesis_paper::{
    load_manifest, review_manuscript, run_paper, run_review, run_revision, PaperConfig,
    ReviewConfig, RevisionConfig,
};
use genesis_scientist::artifacts::{
    AblationSummary, AnalysisResult, AnalysisSpec, FinalEpisode, FinalManifest, PaperManifest,
    PosthocManifest, Review, VariantResult, VariantSpec,
};
use genesis_scientist::selection::freeze_selection;
use genesis_scientist::{PhaseContext, RunPaths};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const WINNER: &[(u32, f64, bool)] = &[
    (0, 6.0, true),
    (1, 7.5, true),
    (2, 0.0, false),
    (3, 9.84, true),
];
const NO_WINNER: &[(u32, f64, bool)] = &[(0, 6.0, true), (1, 5.5, true), (2, 0.0, false)];

struct Run {
    _dir: tempfile::TempDir,
    paths: RunPaths,
    task: Task,
}

fn write(path: &Path, body: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, body).unwrap();
}

/// A finished investigation: a task snapshot, a database holding `rows`
/// (generation, score, correct) plus one failed proposal, and the frozen
/// selection.
fn new_run(rows: &[(u32, f64, bool)]) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let paths = RunPaths::new(dir.path().join("run"));
    let task_dir = paths.task();
    write(&task_dir.join("task.md"), "Bring x to 3 and y to -1.\n");
    write(&task_dir.join("initial.py"), "x = 1.0\ny = -1.0\n");
    write(&task_dir.join("evaluate.sh"), "true\n");
    let task = Task::load(&task_dir).unwrap();
    let db = ProgramDb::open(
        Some(&paths.investigation().join("programs.sqlite")),
        DatabaseConfig {
            num_islands: 1,
            seed: Some(1),
            ..DatabaseConfig::default()
        },
    )
    .unwrap();
    let mut seed_id = None;
    for (generation, score, correct) in rows {
        let mut program = Program::new(format!("x = {score}\ny = -1.0\n"), "python");
        program.generation = *generation;
        program.combined_score = *score;
        program.correct = *correct;
        program.parent_id = seed_id.clone();
        let name = if *generation == 0 {
            "initial_program".to_string()
        } else {
            format!("step_{generation}")
        };
        program.metadata.insert("patch_name".into(), json!(name));
        db.add(&mut program).unwrap();
        seed_id.get_or_insert(program.id.clone());
    }
    db.log_attempt(
        9,
        "proposal",
        "failed",
        &json!({"terminal": true, "failure_class": "patch_apply_failed"}),
    )
    .unwrap();
    freeze_selection(&paths, &task, &db, "best_available")
        .unwrap()
        .expect("a program is selected");
    Run {
        _dir: dir,
        paths,
        task,
    }
}

fn context(run: &Run, provider: Arc<MockProvider>) -> PhaseContext {
    PhaseContext {
        paths: run.paths.clone(),
        task: run.task.clone(),
        llm: LlmClient::mock(provider),
        events: EventLog::null(),
        agent: None,
        cancel: Arc::new(AtomicBool::new(false)),
        seed: 7,
    }
}

fn offline() -> PaperConfig {
    PaperConfig {
        models: vec!["writer".into()],
        retrieve_references: false,
        compile_pdf: false,
        max_repair_rounds: 2,
        ..PaperConfig::default()
    }
}

const SUPPLEMENT: &str = "### Protocol\nEvery candidate program was scored by the task evaluator, and the harness, not the proposing model, ran it. {{REF:TABLE:run_summary}} lists the counts of the search.";

fn tagged_draft(abstract_text: &str, related: &str, results: &str, claims: Value) -> String {
    format!(
        "<TITLE>Automated search improves a two-parameter program</TITLE>\n<ABSTRACT>{abstract_text}</ABSTRACT>\n\
         <INTRODUCTION>We study whether an automated search can improve a small program without human edits.</INTRODUCTION>\n\
         <RELATED_WORK>{related}</RELATED_WORK>\n\
         <METHOD>The search proposes edits to the program and keeps those the evaluator scores higher.</METHOD>\n\
         <RESULTS>{results}</RESULTS>\n\
         <LIMITATIONS>The study covers a single task.</LIMITATIONS>\n\
         <CONCLUSION>The search improved the program on this task.</CONCLUSION>\n\
         <SUPPLEMENTARY>{SUPPLEMENT}</SUPPLEMENTARY>\n<CLAIMS>\n{claims}\n</CLAIMS>"
    )
}

fn base_claims() -> Value {
    json!({"claims": [{
        "claim": "The search raised the combined score from 6 to 9.84",
        "source_ids": ["result.seed_score", "result.best_correct_score"],
        "citation_keys": []
    }]})
}

const ABSTRACT: &str = "An automated program search raised the combined score from 6 to 9.84, a gain of 64% over the seed.";
const RELATED: &str = "Program search with language models is an active area.";
const RESULTS: &str =
    "The best program reaches 9.84 ({{REF:TABLE:results}}).\n\n{{FIGURE:improvement}}";

fn review_form(overall: f64, decision: &str) -> String {
    let form = json!({
        "Summary": "The paper reports an automated program search on one task.", "Strengths": ["clear protocol"],
        "Weaknesses": ["a single task", "no ablations"], "Originality": 2, "Quality": 3, "Clarity": 3, "Significance": 2,
        "Questions": ["Does it transfer?"], "Limitations": ["scope"], "Ethical Concerns": false, "Soundness": 3,
        "Presentation": 3, "Contribution": 2, "Overall": overall, "Confidence": 4, "Decision": decision
    });
    format!("THOUGHT:\nThe evidence is narrow.\n\nREVIEW JSON:\n```json\n{form}\n```")
}

/// A model that answers every stage sensibly, unless `special` answers first.
fn model(
    special: impl Fn(&LlmRequest) -> Option<String> + Send + Sync + 'static,
) -> Arc<MockProvider> {
    MockProvider::new(move |request: &LlmRequest| {
        if let Some(reply) = special(request) {
            return Ok(reply);
        }
        Ok(match request.stage.as_str() {
            "reference_plan" => json!({"central_topic": "program search", "queries": [
                {"query": "LLM program evolution", "category": "core_method", "rationale": "the method"},
                {"query": "automated scientific discovery agents", "category": "result_driven_theme", "rationale": "context"}
            ]})
            .to_string(),
            "paper_strategy" => json!({
                "recommended_title": "Automated search improves a two-parameter program",
                "central_question": "Does the search improve the program?",
                "central_thesis": "The search found a higher-scoring correct program.",
                "narrative_rationale": "report the measured gain", "results_focus": ["result.best_correct_score"],
                "notes_for_writer": []
            })
            .to_string(),
            "paper_draft" => tagged_draft(ABSTRACT, RELATED, RESULTS, base_claims()),
            "paper_audit" => "AUDIT_VERDICT: PASS\nAUDIT_SUMMARY: consistent with the grounding data\nAUDIT_FINDING_COUNT: 0".to_string(),
            "review" => review_form(4.0, "Reject"),
            "meta_review" => review_form(9.0, "Reject"),
            other => format!("no scripted reply for stage {other}"),
        })
    })
}

fn stage_calls(provider: &MockProvider, stage: &str) -> Vec<LlmRequest> {
    provider
        .calls()
        .into_iter()
        .filter(|request| request.stage == stage)
        .collect()
}

fn read<T: serde::de::DeserializeOwned>(run: &Run, relative: &str) -> T {
    fsutil::read_json(&run.paths.run_dir.join(relative))
        .unwrap_or_else(|error| panic!("{relative}: {error:#}"))
}

fn text(run: &Run, relative: &str) -> String {
    std::fs::read_to_string(run.paths.run_dir.join(relative))
        .unwrap_or_else(|error| panic!("{relative}: {error}"))
}

fn write_ablations(run: &Run) {
    let control = [9.8, 9.9, 9.84, 9.86, 9.8];
    let ablated = [7.1, 7.3, 7.0, 7.2, 7.4];
    let variant = |id: &str, kind: &str, scores: &[f64], against: Option<&[f64]>| VariantResult {
        spec: VariantSpec {
            variant_id: id.into(),
            title: format!("{id} title"),
            kind: kind.into(),
            base_program: "best".into(),
            hypothesis: "restarts matter".into(),
            ..VariantSpec::default()
        },
        status: "completed".into(),
        summary: Some(summarize(scores)),
        correct_rate: 1.0,
        comparison_to_control: against.map(|control| compare(scores, control)),
        ..VariantResult::default()
    };
    let summary = AblationSummary {
        schema_version: 1,
        task: "task".into(),
        primary_metric_name: "combined_score".into(),
        anchor_variant_id: "control_best".into(),
        num_replications: 5,
        seeds: vec![1, 2, 3, 4, 5],
        evaluator_is_stochastic: true,
        total_variants: 2,
        total_completed: 2,
        total_failed: 0,
        variants: vec![
            variant("control_best", "control", &control, None),
            variant("no_restarts", "ablation", &ablated, Some(&control)),
        ],
        generated_at: "2026-11-25T00:00:00Z".into(),
    };
    fsutil::write_json(
        &run.paths.ablations().join("pipeline_summary.json"),
        &summary,
    )
    .unwrap();
}

fn write_final(run: &Run, qualified: bool) {
    let episodes = |scores: &[f64], correct: bool| -> Vec<FinalEpisode> {
        scores
            .iter()
            .enumerate()
            .map(|(index, score)| FinalEpisode {
                seed: 1_000_003 + index as u64,
                score: *score,
                correct,
                duration_seconds: 0.1,
            })
            .collect()
    };
    let (candidate, baseline) = ([9.7, 9.8, 9.9], [6.0, 6.1, 5.9]);
    let manifest = FinalManifest {
        schema_version: 3,
        status: "completed".into(),
        correct: qualified,
        outcome: if qualified {
            "qualified"
        } else {
            "not_qualified"
        }
        .into(),
        evaluator: "search_evaluator_heldout_seeds".into(),
        seeds: vec![1_000_003, 1_000_004, 1_000_005],
        episodes: episodes(&candidate, qualified),
        baseline_episodes: episodes(&baseline, true),
        candidate: Some(summarize(&candidate)),
        baseline: Some(summarize(&baseline)),
        comparison: Some(compare(&candidate, &baseline)),
        improvement_supported: true,
        ..FinalManifest::default()
    };
    fsutil::write_json(
        &run.paths.final_evaluation().join("final_manifest.json"),
        &manifest,
    )
    .unwrap();
}

fn write_posthoc(run: &Run) {
    let manifest = PosthocManifest {
        manifest_version: 1,
        created_at: "2026-11-25T00:00:00Z".into(),
        valid_analyses: vec![AnalysisResult {
            spec: AnalysisSpec {
                id: "operator_effectiveness".into(),
                title: "Operator effectiveness".into(),
                family: "operators".into(),
                ..AnalysisSpec::default()
            },
            claim: "Full rewrites succeeded more often than diffs.".into(),
            grounded: true,
            key_numbers: BTreeMap::from([("success_rate".to_string(), 0.75)]),
            ..AnalysisResult::default()
        }],
        rejected_analyses: vec![],
    };
    fsutil::write_json(&run.paths.posthoc().join("latest_manifest.json"), &manifest).unwrap();
}

#[derive(Clone, Default)]
struct ServerState {
    semantic_scholar_hits: Arc<AtomicUsize>,
    openalex_hits: Arc<AtomicUsize>,
    /// Failures Semantic Scholar returns before it answers; `usize::MAX`
    /// never answers.
    semantic_scholar_failures: Arc<AtomicUsize>,
    failure_status: Arc<AtomicUsize>,
    user_agents: Arc<Mutex<Vec<String>>>,
}

async fn reference_server(state: ServerState) -> String {
    let s2 = state.clone();
    let openalex = state.clone();
    let app = Router::new()
        .route(
            "/graph/v1/paper/search",
            get(move |headers: HeaderMap, Query(query): Query<HashMap<String, String>>| {
                let state = s2.clone();
                async move {
                    let hit = state.semantic_scholar_hits.fetch_add(1, Ordering::SeqCst);
                    let agent = headers.get("user-agent").and_then(|value| value.to_str().ok()).unwrap_or("").to_string();
                    state.user_agents.lock().unwrap().push(agent);
                    if hit < state.semantic_scholar_failures.load(Ordering::SeqCst) {
                        let status = StatusCode::from_u16(state.failure_status.load(Ordering::SeqCst) as u16).unwrap();
                        return (status, Json(json!({"error": "unavailable"})));
                    }
                    assert!(query.contains_key("query") && query["fields"].contains("citationCount"));
                    (
                        StatusCode::OK,
                        Json(json!({"data": [
                            {"paperId": "p1", "title": "Evolving programs with language models", "year": 2021, "venue": "NeurIPS",
                             "citationCount": 120, "authors": [{"name": "Ada Lovelace"}, {"name": "Kurt Gödel"}], "url": "https://example.org/p1"},
                            {"paperId": "p2", "title": "Circle packing in a {unit square", "year": 1830, "venue": "",
                             "citationCount": 40, "authors": [{"name": "Carl Gauss"}], "url": "https://example.org/p2"},
                            {"paperId": "p3", "title": "  ", "year": 2000}
                        ]})),
                    )
                }
            }),
        )
        .route(
            "/works",
            get(move |Query(query): Query<HashMap<String, String>>| {
                let state = openalex.clone();
                async move {
                    state.openalex_hits.fetch_add(1, Ordering::SeqCst);
                    assert!(query.contains_key("search"));
                    Json(json!({"results": [{
                        "id": "https://openalex.org/W42", "display_name": "Open-ended program search", "publication_year": 2024,
                        "cited_by_count": 7, "authorships": [{"author": {"display_name": "Grace Hopper"}}],
                        "primary_location": {"source": {"display_name": "ICML"}}
                    }]}))
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{address}")
}

#[tokio::test]
async fn a_full_paper_uses_every_evidence_phase_and_compiles() {
    let run = new_run(WINNER);
    write_ablations(&run);
    write_final(&run, true);
    write_posthoc(&run);
    write(
        &run.paths.investigation().join("ideas_history.jsonl"),
        "{\"idea_id\": \"a\"}\n{\"idea_id\": \"b\"}\n",
    );
    write(
        &run.paths.investigation().join("conclusions.md"),
        "Raising x helps; lowering y does not.\n",
    );
    write(
        &run.paths.run_dir.join("metrics/cost_summary.json"),
        "{\"total_cost_usd\": 1.25}",
    );
    let state = ServerState::default();
    let server = reference_server(state.clone()).await;

    let results = "The best program reaches 9.84 against 6 for the seed ({{REF:TABLE:results}}), over 4 proposals of which 75% of the rewrites succeeded.\n\n\
                   {{FIGURE:improvement}}\n\nRemoving restarts lowers the mean score by 2.64 ({{REF:TABLE:ablations}}, {{REF:FIGURE:ablations}}).\n\n\
                   {{TABLE:ablations}}\n\nIn the sealed final evaluation the studied program averages 9.8 against 6.0 for the baseline, and it qualified ({{REF:TABLE:final}}). The run cost 1.25 dollars.";
    let related = "Language models can evolve programs [@lovelace2021evolving], and the packing problem is old [@gauss1830circle].";
    let claims = json!({"claims": [
        {"claim": "The best program reaches 9.84 against 6 for the seed", "source_ids": ["result.best_correct_score", "result.seed_score"], "citation_keys": []},
        {"claim": "Removing restarts lowers the mean score by 2.64", "source_ids": ["ablation.compare_control_best_vs_no_restarts.combined_score.mean_delta"], "citation_keys": []},
        {"claim": "The studied program averages 9.8 in the final evaluation", "source_ids": ["final.metric.combined_score", "final.qualification.outcome"], "citation_keys": []},
        {"claim": "Language models can evolve programs", "source_ids": ["ref.s2.p1"], "citation_keys": ["lovelace2021evolving"]}
    ]});
    let reply = tagged_draft(ABSTRACT, related, results, claims);
    let provider = model(move |request| (request.stage == "paper_draft").then(|| reply.clone()));
    let config = PaperConfig {
        retrieve_references: true,
        compile_pdf: true,
        semantic_scholar_url: server.clone(),
        openalex_url: server,
        ..offline()
    };
    let manifest = run_paper(&context(&run, provider.clone()), &config)
        .await
        .unwrap();

    assert!(
        manifest.validation_ok,
        "{:?}",
        read::<ValidationReport>(&run, "paper/validation_report.json").hard_errors
    );
    let audit: AuditReport = read(&run, "paper/paper_evidence_audit.json");
    assert_eq!(
        manifest.evidence_audit_verdict, "pass",
        "{:?}",
        audit.findings
    );
    assert!(
        audit.input_sha256.contains_key("paper.tex")
            && audit.input_sha256.contains_key("paper_data.json")
    );
    assert_eq!(
        (manifest.paper_tex.as_str(), manifest.paper_md.as_str()),
        ("paper/paper.tex", "paper/paper.md")
    );
    assert_eq!(load_manifest(&run.paths).unwrap(), manifest);
    assert!(stage_calls(&provider, "paper_repair").is_empty());

    let data: PaperData = read(&run, "paper/paper_data.json");
    for id in [
        "result.best_correct_score",
        "result.seed_score",
        "result.improvement_rel_percent",
        "result.total_ideas",
        "result.total_failed_proposals",
        "result.top_program_01",
        "result.improvement_002",
        "result.total_cost_usd",
        "task.constant_01",
        "source.selected_constant_001",
        "ablation.variant_no_restarts.combined_score.mean",
        "ablation.variant_control_best.combined_score.ci95_low",
        "ablation.compare_control_best_vs_no_restarts.combined_score.p_value",
        "final.metric.combined_score",
        "final.qualification.outcome",
        "final.comparison.mean_diff",
        "posthoc.operator_effectiveness.success_rate",
    ] {
        assert!(
            data.grounding_sources.contains_key(id),
            "missing grounding id {id}"
        );
    }
    assert_eq!(data.grounding_sources["result.total_ideas"].value, json!(2));
    assert_eq!(
        data.grounding_sources["result.improvement_rel_percent"]
            .value
            .as_f64()
            .unwrap()
            .round(),
        64.0
    );
    assert!(!data
        .grounding_sources
        .keys()
        .any(|id| id.starts_with("study.")));
    for phase in ["ablations", "final_evaluation", "posthoc", "references"] {
        assert_eq!(data.evidence_sources[phase].status, "included", "{phase}");
    }
    assert_eq!(
        data.distilled_conclusions.as_deref(),
        Some("Raising x helps; lowering y does not.")
    );
    assert_eq!(data.limitations, ["1 proposal produced no evaluated program. Such proposals are missing evidence, not measured losses."]);

    let references: Vec<Reference> = read(&run, "paper/references_provenance.json");
    let keys: Vec<&str> = references
        .iter()
        .map(|reference| reference.citation_key.as_str())
        .collect();
    assert_eq!(keys, ["lovelace2021evolving", "gauss1830circle"]);
    assert_eq!(references[0].id, "ref.s2.p1");
    let bibliography = text(&run, "paper/references.bib");
    assert!(
        bibliography.contains("@article{lovelace2021evolving,")
            && bibliography.contains("@misc{gauss1830circle,")
    );
    assert!(bibliography.contains("author = {{Ada Lovelace} and {Kurt Gödel}},"));
    let agents = state.user_agents.lock().unwrap().clone();
    assert!(
        !agents.is_empty()
            && agents
                .iter()
                .all(|agent| agent == "genesis-paper/0.1 (reference retrieval)"),
        "{agents:?}"
    );
    assert_eq!(state.openalex_hits.load(Ordering::SeqCst), 0);

    let tex = text(&run, "paper/paper.tex");
    for needle in [
        "\\citep{lovelace2021evolving}",
        "Table~\\ref{tab:ablations}",
        "\\label{tab:final}",
        "\\label{fig:ablations}",
        "\\label{tab:run_summary}",
        "no\\_restarts",
        "\\bibliography{references}",
        "Generated by Genesis",
        "75\\% of the rewrites",
        "Outcome: qualified.",
    ] {
        assert!(tex.contains(needle), "paper.tex lacks `{needle}`");
    }
    let markdown = text(&run, "paper/paper.md");
    assert!(markdown.contains("| no_restarts | ablation | 5 | 7.2000 |"));
    assert!(markdown.contains("lowers the mean score by 2.64 (Table 1, Figure 2)"));
    assert!(
        markdown.contains("against 6 for the seed (Table 2)")
            && markdown.contains("it qualified (Table 3)")
    );
    assert!(markdown.contains(
        "- **[gauss1830circle]** Carl Gauss (1830). *Circle packing in a {unit square*."
    ));
    assert!(text(&run, "paper/figures/improvement.svg").starts_with("<svg"));
    assert!(run.paths.paper().join("figures/ablations.svg").is_file());
    let sections: Draft = read(&run, "paper/sections.json");
    assert_eq!(sections.claims.len(), 4);
    assert!(read::<Value>(&run, "paper/paper_strategy.json")["central_thesis"].is_string());
    assert_eq!(
        read::<Value>(&run, "paper/reference_query_plan.json")["queries"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    // The writer saw the grounding sources, the floats and the references.
    let draft_prompt = stage_calls(&provider, "paper_draft")[0]
        .last_user()
        .to_string();
    for needle in [
        "- result.best_correct_score = 9.84:",
        "- [@lovelace2021evolving] Evolving programs",
        "{{TABLE:ablations}}",
        "- no_restarts (ablation)",
        "Raising x helps",
        "# Strategy",
    ] {
        assert!(
            draft_prompt.contains(needle),
            "the draft prompt lacks `{needle}`"
        );
    }

    // `GENESIS_PAPER_KEEP=<dir>` keeps this paper for a look at the result.
    if let Some(keep) = std::env::var_os("GENESIS_PAPER_KEEP") {
        fsutil::copy_tree(&run.paths.paper(), Path::new(&keep), &[]).unwrap();
    }
    let report: ValidationReport = read(&run, "paper/validation_report.json");
    if tex_available() {
        assert!(report.pdf_built, "{:?}", report.latex_log_tail);
        assert_eq!(manifest.paper_pdf.as_deref(), Some("paper/paper.pdf"));
        let pdf = std::fs::read(run.paths.paper().join("paper.pdf")).unwrap();
        assert!(
            pdf.starts_with(b"%PDF-") && pdf.len() > 20_000,
            "the PDF is {} bytes",
            pdf.len()
        );
        assert!(!run.paths.paper().join(".latex_build").exists());
    } else {
        assert!(!report.pdf_built && report.pdf_note.is_some() && manifest.paper_pdf.is_none());
    }
}

#[tokio::test]
async fn a_paper_is_written_from_the_investigation_alone() {
    let run = new_run(WINNER);
    let provider = model(|_| None);
    let manifest = run_paper(&context(&run, provider.clone()), &offline())
        .await
        .unwrap();
    assert!(manifest.validation_ok && manifest.evidence_audit_verdict == "pass");
    assert!(manifest.paper_pdf.is_none());
    let report: ValidationReport = read(&run, "paper/validation_report.json");
    assert_eq!(
        report.pdf_note.as_deref(),
        Some("PDF compilation is turned off (paper.compile_pdf=false)")
    );

    let data: PaperData = read(&run, "paper/paper_data.json");
    for phase in ["ablations", "final_evaluation", "posthoc", "references"] {
        assert_eq!(data.evidence_sources[phase].status, "missing", "{phase}");
    }
    let limitations = data.limitations.join("\n");
    for needle in [
        "No ablation study is available",
        "No sealed final evaluation is available",
        "No post-hoc analysis",
        "No references were retrieved",
    ] {
        assert!(limitations.contains(needle), "{needle}");
    }
    let tex = text(&run, "paper/paper.tex");
    assert!(!tex.contains("\\bibliography") && !tex.contains("tab:ablations"));
    assert!(tex.contains("\\item No ablation study is available for this run"));
    assert!(text(&run, "paper/references.bib").is_empty());
    assert!(
        stage_calls(&provider, "reference_plan").is_empty(),
        "no plan is paid for when retrieval is off"
    );
    assert!(
        stage_calls(&provider, "paper_audit").is_empty(),
        "no audit model is configured"
    );
}

#[tokio::test]
async fn evidence_that_ran_but_left_nothing_usable_is_reported_as_such() {
    let run = new_run(WINNER);
    write(
        &run.paths.ablations().join("pipeline_summary.json"),
        "{\"variants\": [",
    );
    let interrupted = FinalManifest {
        status: "running".into(),
        outcome: "not_qualified".into(),
        ..FinalManifest::default()
    };
    fsutil::write_json(
        &run.paths.final_evaluation().join("final_manifest.json"),
        &interrupted,
    )
    .unwrap();
    fsutil::write_json(
        &run.paths.posthoc().join("latest_manifest.json"),
        &PosthocManifest::default(),
    )
    .unwrap();

    // An interrupted final evaluation qualifies nothing and forbids nothing.
    let manifest = run_paper(&context(&run, model(|_| None)), &offline())
        .await
        .unwrap();
    assert!(manifest.validation_ok && manifest.evidence_audit_verdict == "pass");
    let data: PaperData = read(&run, "paper/paper_data.json");
    for phase in ["ablations", "final_evaluation", "posthoc"] {
        assert_eq!(data.evidence_sources[phase].status, "degraded", "{phase}");
        assert!(data.evidence_sources[phase].path.is_some());
    }
    assert!(!data
        .grounding_sources
        .keys()
        .any(|id| id.starts_with("ablation.")
            || id.starts_with("final.")
            || id.starts_with("posthoc.")));
    let limitations = data.limitations.join("\n");
    for needle in [
        "The ablation study left no usable result (the result file is unreadable: EOF while parsing",
        "The sealed final evaluation left no usable result (the final evaluation has status `running`); every score reported here is a search-time score",
        "The post-hoc analysis left no usable result (no valid analysis).",
    ] {
        assert!(limitations.contains(needle), "`{needle}` is not in {limitations}");
    }
    assert!(
        !limitations.contains(run.paths.run_dir.to_str().unwrap()),
        "a local path reached the paper"
    );
    let tex = text(&run, "paper/paper.tex");
    assert!(
        tex.contains("The post-hoc analysis left no usable result (no valid analysis).")
            && !tex.contains("tab:final")
    );
}

#[tokio::test]
async fn a_run_without_a_winner_becomes_a_negative_results_paper() {
    let run = new_run(NO_WINNER);
    write_final(&run, false);
    let reply = tagged_draft(
        "The search did not find a program that scores above the seed, which scores 6; the best candidate reached 5.5 and did not qualify in the final evaluation.",
        RELATED,
        "No candidate scored above the seed ({{REF:TABLE:results}}). Of 3 proposals, 1 produced no evaluated program, which is missing evidence rather than a measured loss.",
        json!({"claims": [{"claim": "The best candidate reached 5.5 against 6 for the seed", "source_ids": ["study.search_score", "study.baseline_score"], "citation_keys": []}]}),
    );
    let provider = model(move |request| (request.stage == "paper_draft").then(|| reply.clone()));
    // The strict policy does not block a negative-results paper.
    let manifest = run_paper(&context(&run, provider.clone()), &offline())
        .await
        .unwrap();
    assert!(manifest.validation_ok && manifest.evidence_audit_verdict == "pass");
    let data: PaperData = read(&run, "paper/paper_data.json");
    assert_eq!(data.selection.selection_kind, "non_winner_candidate");
    assert_eq!(
        data.grounding_sources["study.search_score_delta"].value,
        json!(-0.5)
    );
    assert_eq!(
        data.grounding_sources["study.attempts.unscored"].value,
        json!(1)
    );
    assert_eq!(
        data.grounding_sources["study.attempts.total"].value,
        json!(3)
    );
    assert_eq!(
        data.grounding_sources["final.qualification.outcome"].value,
        json!("not_qualified")
    );
    let limitations = data.limitations.join("\n");
    assert!(
        limitations.contains("reports a negative result")
            && limitations.contains("did not qualify the studied program")
    );
    assert!(text(&run, "paper/paper.md")
        .contains("does not by itself show that inferior performance was the reason"));
    let brief = stage_calls(&provider, "paper_draft")[0]
        .last_user()
        .to_string();
    assert!(brief
        .contains("selection_kind: non_winner_candidate (search outcome: completed_no_winner)"));
}

#[tokio::test]
async fn the_outcome_policy_decides_whether_an_unqualified_winner_is_published() {
    let run = new_run(WINNER);
    write_final(&run, false);
    let error = run_paper(&context(&run, model(|_| None)), &offline())
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("paper.final_outcome_policy=allow_not_qualified"),
        "{error:#}"
    );
    assert!(!run.paths.paper().join("latest_manifest.json").exists());

    let lenient = PaperConfig {
        final_outcome_policy: "allow_not_qualified".into(),
        ..offline()
    };
    let manifest = run_paper(&context(&run, model(|_| None)), &lenient)
        .await
        .unwrap();
    assert!(manifest.validation_ok);
    assert!(text(&run, "paper/paper.tex")
        .contains("The final evaluation did not qualify the studied program."));
}

#[tokio::test]
async fn nothing_to_write_about_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let run = new_run(WINNER);
    let empty = Run {
        _dir: dir,
        paths: RunPaths::new(run.paths.run_dir.join("elsewhere")),
        task: run.task.clone(),
    };
    let error = run_paper(&context(&empty, model(|_| None)), &offline())
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("no program was selected"),
        "{error:#}"
    );

    // A selection whose database holds only failed evaluations.
    std::fs::remove_file(run.paths.investigation().join("programs.sqlite")).unwrap();
    for sidecar in ["programs.sqlite-wal", "programs.sqlite-shm"] {
        let _ = std::fs::remove_file(run.paths.investigation().join(sidecar));
    }
    let db = ProgramDb::open(
        Some(&run.paths.investigation().join("programs.sqlite")),
        DatabaseConfig::default(),
    )
    .unwrap();
    let mut failed = Program::new("x = 1\n", "python");
    failed
        .metadata
        .insert("evaluation_failed".into(), json!(true));
    db.add(&mut failed).unwrap();
    let error = run_paper(&context(&run, model(|_| None)), &offline())
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("no evaluated programs"),
        "{error:#}"
    );

    let no_models = PaperConfig {
        models: vec![],
        ..offline()
    };
    assert!(run_paper(&context(&run, model(|_| None)), &no_models)
        .await
        .is_err());
}

#[tokio::test]
async fn an_invented_number_is_caught_by_the_audit_and_removed_by_a_repair() {
    let run = new_run(WINNER);
    let reply = tagged_draft(
        ABSTRACT,
        RELATED,
        "The best program reaches 9.99 ({{REF:TABLE:results}}).",
        base_claims(),
    );
    let provider = model(move |request| match request.stage.as_str() {
        "paper_draft" => Some(reply.clone()),
        "paper_repair" => Some(
            "<RESULTS>The best program reaches 9.84 ({{REF:TABLE:results}}).</RESULTS>".to_string(),
        ),
        _ => None,
    });
    let manifest = run_paper(&context(&run, provider.clone()), &offline())
        .await
        .unwrap();
    assert!(manifest.validation_ok && manifest.evidence_audit_verdict == "pass");
    let repairs = stage_calls(&provider, "paper_repair");
    assert_eq!(repairs.len(), 1);
    let asked = repairs[0].last_user();
    assert!(
        asked.contains("[number-results-1] Results: \"9.99\""),
        "{asked}"
    );
    assert!(
        asked.contains("<RESULTS>\nThe best program reaches 9.99")
            && asked.contains("# What must change")
    );
    assert!(repairs[0]
        .system
        .contains("Fix every finding by changing the manuscript only"));
    let sections: Draft = read(&run, "paper/sections.json");
    assert!(
        sections.section("results").contains("9.84") && sections.section("abstract") == ABSTRACT
    );
    assert!(!text(&run, "paper/paper.tex").contains("9.99"));
}

#[tokio::test]
async fn a_paper_that_cannot_be_repaired_is_published_as_failing() {
    let run = new_run(WINNER);
    let reply = tagged_draft(
        ABSTRACT,
        RELATED,
        "The best program reaches 9.99.",
        base_claims(),
    );
    let round = AtomicUsize::new(0);
    let provider = model(move |request| match request.stage.as_str() {
        "paper_draft" => Some(reply.clone()),
        "paper_repair" => {
            let round = round.fetch_add(1, Ordering::SeqCst) + 1;
            Some(format!(
                "<RESULTS>The best program still reaches 9.9{round}.</RESULTS>"
            ))
        }
        _ => None,
    });
    let manifest = run_paper(&context(&run, provider.clone()), &offline())
        .await
        .unwrap();
    assert_eq!(
        stage_calls(&provider, "paper_repair").len(),
        2,
        "max_repair_rounds bounds the loop"
    );
    assert_eq!(manifest.evidence_audit_verdict, "fail");
    assert!(
        manifest.validation_ok,
        "the structure is sound; only the evidence audit fails"
    );
    let audit: AuditReport = read(&run, "paper/paper_evidence_audit.json");
    assert_eq!(audit.findings[0].claim, "9.92");

    // A repair that changes nothing ends the loop at once.
    let run = new_run(WINNER);
    let reply = tagged_draft(
        ABSTRACT,
        RELATED,
        "The best program reaches 9.99.",
        base_claims(),
    );
    let provider = model(move |request| match request.stage.as_str() {
        "paper_draft" => Some(reply.clone()),
        "paper_repair" => Some("<RESULTS>The best program reaches 9.99.</RESULTS>".to_string()),
        _ => None,
    });
    let manifest = run_paper(&context(&run, provider.clone()), &offline())
        .await
        .unwrap();
    assert_eq!(stage_calls(&provider, "paper_repair").len(), 1);
    assert_eq!(manifest.evidence_audit_verdict, "fail");
}

#[tokio::test]
async fn an_unknown_citation_and_a_missing_section_fail_validation_until_repaired() {
    let run = new_run(WINNER);
    let reply = tagged_draft(ABSTRACT, "Prior work exists [@ghost2020].", "{{TABLE:ablations}}\n\nThe best program reaches 9.84.", base_claims())
        .replace("<METHOD>The search proposes edits to the program and keeps those the evaluator scores higher.</METHOD>\n", "");
    let drafts = AtomicUsize::new(0);
    let provider = model(move |request| match request.stage.as_str() {
        "paper_draft" => Some(if drafts.fetch_add(1, Ordering::SeqCst) == 0 {
            reply.clone()
        } else {
            assert!(
                request.last_user().contains("<METHOD> missing or empty"),
                "{}",
                request.last_user()
            );
            "<METHOD>The search proposes edits and keeps the better ones.</METHOD>".to_string()
        }),
        "paper_repair" => {
            let asked = request.last_user();
            assert!(
                asked.contains("the paper cites `ghost2020`, which is not a retrieved reference"),
                "{asked}"
            );
            assert!(
                asked.contains(
                    "placeholder `{{TABLE:ablations}}` names no table or figure this run has"
                ),
                "{asked}"
            );
            Some("<RELATED_WORK>Prior work exists.</RELATED_WORK>\n<RESULTS>The best program reaches 9.84.</RESULTS>".to_string())
        }
        _ => None,
    });
    let manifest = run_paper(&context(&run, provider.clone()), &offline())
        .await
        .unwrap();
    assert!(manifest.validation_ok && manifest.evidence_audit_verdict == "pass");
    assert_eq!(
        stage_calls(&provider, "paper_draft").len(),
        2,
        "one corrective retry for the missing section"
    );
    assert_eq!(stage_calls(&provider, "paper_repair").len(), 1);

    // A draft that stays incomplete is an error, not a paper.
    let run = new_run(WINNER);
    let provider = model(|request| {
        (request.stage == "paper_draft").then(|| "<TITLE>Only a title</TITLE>".to_string())
    });
    let error = run_paper(&context(&run, provider), &offline())
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("missing sections after a corrective retry"),
        "{error:#}"
    );
}

#[tokio::test]
async fn the_model_audit_layer_runs_after_the_numeric_checks_and_its_findings_are_repaired() {
    let run = new_run(WINNER);
    let audits = AtomicUsize::new(0);
    let provider = model(move |request| match request.stage.as_str() {
        "paper_audit" => Some(match audits.fetch_add(1, Ordering::SeqCst) {
            0 => "AUDIT_VERDICT: FAIL\nAUDIT_SUMMARY: overclaims".to_string(),
            1 => {
                assert!(request.last_user().contains("broke the output contract"));
                "AUDIT_VERDICT: FAIL\nAUDIT_SUMMARY: the conclusion overclaims\nAUDIT_FINDING_COUNT: 1\n\
                 AUDIT_FINDING: id=conclusion-1 || section=Conclusion || claim=improved the program || issue=one task only || evidence=paper_data.json || fix=say on this task"
                    .to_string()
            }
            _ => return None,
        }),
        "paper_repair" => Some(
            "<CONCLUSION>On this one task, the search improved the program.</CONCLUSION>"
                .to_string(),
        ),
        _ => None,
    });
    let config = PaperConfig {
        audit_models: vec!["auditor".into()],
        ..offline()
    };
    let manifest = run_paper(&context(&run, provider.clone()), &config)
        .await
        .unwrap();
    assert_eq!(manifest.evidence_audit_verdict, "pass");
    let audit_calls = stage_calls(&provider, "paper_audit");
    assert_eq!(audit_calls.len(), 3);
    assert!(audit_calls.iter().all(|call| call.model == "auditor"));
    let packet = audit_calls[0].last_user();
    assert!(
        packet.contains("# Grounding sources\n- ")
            && packet.contains("# claims_grounding.json")
            && packet.contains("# Manuscript")
    );
    let repair = &stage_calls(&provider, "paper_repair")[0];
    assert!(repair
        .last_user()
        .contains("[conclusion-1] Conclusion: \"improved the program\": one task only"));
    let audit: AuditReport = read(&run, "paper/paper_evidence_audit.json");
    assert_eq!(audit.model_layer.as_deref(), Some("auditor"));
    assert!(audit
        .summary
        .contains("Model layer: consistent with the grounding data"));

    // An audit model that never honours the contract is recorded, not obeyed.
    let run = new_run(WINNER);
    let provider =
        model(|request| (request.stage == "paper_audit").then(|| "looks fine to me".to_string()));
    let manifest = run_paper(&context(&run, provider), &config).await.unwrap();
    assert_eq!(manifest.evidence_audit_verdict, "pass");
    let audit: AuditReport = read(&run, "paper/paper_evidence_audit.json");
    assert!(
        audit.model_layer.is_none() && audit.model_layer_error.unwrap().contains("AUDIT_VERDICT")
    );
}

#[tokio::test]
async fn retrieval_retries_falls_back_and_degrades_to_no_references() {
    let plan = QueryPlan {
        central_topic: "t".into(),
        queries: vec![PlannedQuery {
            query: "program evolution".into(),
            category: "core_method".into(),
            rationale: String::new(),
        }],
    };
    // A rate limit is retried.
    let state = ServerState::default();
    state.semantic_scholar_failures.store(1, Ordering::SeqCst);
    state.failure_status.store(429, Ordering::SeqCst);
    let server = reference_server(state.clone()).await;
    let (references, failures) = Retriever::new(&server, &server).retrieve(&plan, 5).await;
    assert_eq!(references.len(), 2);
    assert!(failures.is_empty());
    assert_eq!(state.semantic_scholar_hits.load(Ordering::SeqCst), 2);

    // A dead primary falls back to OpenAlex.
    let state = ServerState::default();
    state
        .semantic_scholar_failures
        .store(usize::MAX, Ordering::SeqCst);
    state.failure_status.store(500, Ordering::SeqCst);
    let server = reference_server(state.clone()).await;
    let (references, failures) = Retriever::new(&server, &server).retrieve(&plan, 5).await;
    assert_eq!(references.len(), 1);
    assert_eq!(
        (
            references[0].id.as_str(),
            references[0].citation_key.as_str(),
            references[0].venue.as_str()
        ),
        ("ref.openalex.W42", "hopper2024openended", "ICML")
    );
    assert_eq!(failures.len(), 1);
    assert_eq!(
        state.semantic_scholar_hits.load(Ordering::SeqCst),
        4,
        "one try and three retries"
    );
    // A refusal that is not a rate limit is not retried.
    let state = ServerState::default();
    state
        .semantic_scholar_failures
        .store(usize::MAX, Ordering::SeqCst);
    state.failure_status.store(403, Ordering::SeqCst);
    let server = reference_server(state.clone()).await;
    Retriever::new(&server, &server).retrieve(&plan, 5).await;
    assert_eq!(state.semantic_scholar_hits.load(Ordering::SeqCst), 1);

    // Nothing reachable: the paper is written without a bibliography.
    let run = new_run(WINNER);
    let dead = "http://127.0.0.1:1".to_string();
    let config = PaperConfig {
        retrieve_references: true,
        semantic_scholar_url: dead.clone(),
        openalex_url: dead,
        ..offline()
    };
    let provider = model(|_| None);
    let manifest = run_paper(&context(&run, provider.clone()), &config)
        .await
        .unwrap();
    assert!(manifest.validation_ok);
    let data: PaperData = read(&run, "paper/paper_data.json");
    assert_eq!(data.evidence_sources["references"].status, "missing");
    assert!(data.evidence_sources["references"]
        .note
        .as_ref()
        .unwrap()
        .contains("retrieval returned nothing"));
    assert!(data
        .limitations
        .iter()
        .any(|limitation| limitation.contains("No references were retrieved")));
    assert_eq!(stage_calls(&provider, "reference_plan").len(), 1);

    // An unusable plan is retried once with the validator's message, then given up.
    let run = new_run(WINNER);
    let provider = model(|request| {
        (request.stage == "reference_plan")
            .then(|| json!({"queries": [{"query": "x", "category": "gossip"}]}).to_string())
    });
    run_paper(&context(&run, provider.clone()), &config)
        .await
        .unwrap();
    let plans = stage_calls(&provider, "reference_plan");
    assert_eq!(plans.len(), 2);
    assert!(plans[1]
        .last_user()
        .contains("query category `gossip` is not one of"));
    let data: PaperData = read(&run, "paper/paper_data.json");
    assert!(data.evidence_sources["references"]
        .note
        .as_ref()
        .unwrap()
        .contains("no usable query plan"));
}

fn reviewer_config() -> ReviewConfig {
    ReviewConfig {
        model: "reviewer".into(),
        reviewer_count: 3,
        ..ReviewConfig::default()
    }
}

#[tokio::test]
async fn the_review_ensemble_averages_scores_and_survives_one_malformed_reviewer() {
    let run = new_run(WINNER);
    let calls = AtomicUsize::new(0);
    let provider = model(move |request| match request.stage.as_str() {
        "review" => Some(match calls.fetch_add(1, Ordering::SeqCst) {
            0 => "THOUGHT: I forgot the form.".to_string(),
            index => review_form([3.0, 4.0, 6.0][index - 1], "Reject"),
        }),
        "meta_review" => Some(review_form(9.0, "Accept")),
        _ => None,
    });
    let ctx = context(&run, provider.clone());
    assert!(
        run_review(&ctx, &reviewer_config(), "ai_review")
            .await
            .is_err(),
        "there is no paper yet"
    );
    run_paper(&ctx, &offline()).await.unwrap();
    assert!(run_review(&ctx, &reviewer_config(), "final")
        .await
        .unwrap_err()
        .to_string()
        .contains("unknown review key"));

    let review = run_review(&ctx, &reviewer_config(), "ai_review")
        .await
        .unwrap();
    assert_eq!(review.overall, 4.0, "round((3 + 4 + 6) / 3)");
    assert_eq!(
        review.decision, "Accept",
        "the decision is the area chair's"
    );
    assert_eq!(read::<Review>(&run, "paper/ai_review.json"), review);
    let manifest: PaperManifest = load_manifest(&run.paths).unwrap();
    let record = &manifest.reviews["ai_review"];
    assert_eq!(
        (
            record.status.as_str(),
            record.overall,
            record.review_json.as_deref()
        ),
        ("completed", Some(4.0), Some("paper/ai_review.json"))
    );

    let reviews = stage_calls(&provider, "review");
    assert_eq!(reviews.len(), 4, "three reviewers and one corrective retry");
    assert!(reviews
        .iter()
        .all(|call| call.model == "reviewer" && call.temperature == Some(1.0)));
    assert!(
        reviews[0]
            .last_user()
            .contains("Here is the paper you are asked to review:")
            && reviews[0]
                .last_user()
                .contains("# Automated search improves")
    );
    let retry = reviews
        .iter()
        .find(|call| call.messages.len() == 3)
        .expect("the corrective turn");
    assert!(retry.last_user().contains("could not be used"));
    let meta = &stage_calls(&provider, "meta_review")[0];
    assert!(
        meta.last_user().contains("reviewed by 3 reviewers")
            && meta.last_user().contains("Review 3:")
    );
    assert!(meta.system.contains("Area Chair"));
}

#[tokio::test]
async fn a_failed_review_is_recorded_and_a_failed_meta_review_has_a_stand_in() {
    let run = new_run(WINNER);
    let provider = model(|request| {
        (request.stage == "review").then(|| "I decline to fill in the form.".to_string())
    });
    let ctx = context(&run, provider);
    let manifest = run_paper(&ctx, &offline()).await.unwrap();
    let error = run_review(&ctx, &reviewer_config(), "ai_review")
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("no reviewer returned a usable review"),
        "{error}"
    );
    let after: PaperManifest = load_manifest(&run.paths).unwrap();
    assert_eq!(after.reviews["ai_review"].status, "failed");
    assert!(after.reviews["ai_review"]
        .error
        .as_ref()
        .unwrap()
        .contains("no JSON review"));
    assert!(!run.paths.paper().join("ai_review.json").exists());
    assert_eq!(
        after.paper_tex, manifest.paper_tex,
        "a failed review never removes the paper"
    );

    let calls = AtomicUsize::new(0);
    let provider = model(move |request| match request.stage.as_str() {
        "review" => Some(review_form(
            [2.0, 5.0, 7.0][calls.fetch_add(1, Ordering::SeqCst) % 3],
            if calls.load(Ordering::SeqCst) == 2 {
                "Reject"
            } else {
                "Accept"
            },
        )),
        "meta_review" => Some("no form".to_string()),
        _ => None,
    });
    let review = review_manuscript(
        &LlmClient::mock(provider),
        &reviewer_config(),
        &"A real manuscript sentence. ".repeat(10),
    )
    .await
    .unwrap();
    assert_eq!(review.overall, 5.0, "round(14 / 3)");
    assert_eq!(review.decision, "Accept", "two of three reviewers accepted");
    assert!(review_manuscript(
        &LlmClient::mock(model(|_| None)),
        &reviewer_config(),
        "too short"
    )
    .await
    .is_err());
}

#[tokio::test]
async fn a_revision_that_passes_the_checks_replaces_the_paper_and_is_reviewed_again() {
    let run = new_run(WINNER);
    let revised = "After revision: on this single task, an automated search raised the combined score from 6 to 9.84.";
    let provider = model(move |request| match request.stage.as_str() {
        "paper_revision" => Some(format!("<ABSTRACT>{revised}</ABSTRACT>")),
        "review" if request.last_user().contains("After revision:") => {
            Some(review_form(6.0, "Accept"))
        }
        "meta_review" if request.last_user().contains("\"Overall\": 6") => {
            Some(review_form(6.0, "Accept"))
        }
        _ => None,
    });
    let ctx = context(&run, provider.clone());
    run_paper(&ctx, &offline()).await.unwrap();
    let error = run_revision(
        &ctx,
        &offline(),
        &reviewer_config(),
        &RevisionConfig::default(),
    )
    .await
    .unwrap_err();
    assert!(
        error.to_string().contains("needs a completed `ai_review`"),
        "{error}"
    );
    run_review(&ctx, &reviewer_config(), "ai_review")
        .await
        .unwrap();
    let original_tex = text(&run, "paper/paper.tex");
    let original_data = text(&run, "paper/paper_data.json");

    let record = run_revision(
        &ctx,
        &offline(),
        &reviewer_config(),
        &RevisionConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(record.status, "promoted");
    assert_eq!(
        (record.overall_before, record.overall_after),
        (Some(4.0), Some(6.0))
    );
    assert_eq!(
        (
            record.decision_before.as_deref(),
            record.decision_after.as_deref()
        ),
        (Some("Reject"), Some("Accept"))
    );
    assert_eq!(
        (
            record.attempt_dir.as_str(),
            record.original_snapshot.as_str()
        ),
        ("revision/attempt_01", "paper/revisions/revision_01/before")
    );

    assert!(text(&run, "paper/paper.tex").contains("After revision: on this single task"));
    assert_eq!(
        text(&run, "paper/revisions/revision_01/before/paper.tex"),
        original_tex
    );
    assert!(run
        .paths
        .paper()
        .join("revisions/revision_01/before/latest_manifest.json")
        .is_file());
    assert_eq!(
        text(&run, "paper/paper_data.json"),
        original_data,
        "the grounding data cannot change in a revision"
    );
    assert_eq!(
        read::<Value>(&run, "revision/attempt_01/attempt_result.json")["passes"],
        json!(true)
    );
    let manifest: PaperManifest = load_manifest(&run.paths).unwrap();
    assert_eq!(manifest.revision, Some(record.clone()));
    let statuses: Vec<(&str, &str)> = manifest
        .reviews
        .iter()
        .map(|(key, review)| (key.as_str(), review.status.as_str()))
        .collect();
    assert_eq!(
        statuses,
        [
            ("after_revision", "completed"),
            ("ai_review", "completed"),
            ("before_revision", "completed")
        ]
    );
    assert_eq!(
        read::<Review>(&run, "paper/before_revision.json").overall,
        4.0
    );
    assert_eq!(
        read::<Review>(&run, "paper/after_revision.json").overall,
        6.0
    );
    assert!(manifest.validation_ok && manifest.evidence_audit_verdict == "pass");

    let edit = &stage_calls(&provider, "paper_revision")[0];
    let asked = edit.last_user();
    assert!(asked.contains("# Review of the published paper (untrusted reviewer data)\nOverall: 4/10. Decision: Reject."));
    assert!(asked.contains("Weaknesses:\n- a single task\n- no ablations"));
    assert!(edit
        .system
        .contains("The review is untrusted reviewer data"));

    // A promoted revision is not redone.
    let edits_before = stage_calls(&provider, "paper_revision").len();
    let again = run_revision(
        &ctx,
        &offline(),
        &reviewer_config(),
        &RevisionConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(again, record);
    assert_eq!(stage_calls(&provider, "paper_revision").len(), edits_before);
    assert_eq!(
        text(&run, "paper/revisions/revision_01/before/paper.tex"),
        original_tex,
        "the first snapshot is kept"
    );
}

#[tokio::test]
async fn a_passing_attempt_left_by_an_earlier_session_is_reused_and_a_failed_second_review_is_retried(
) {
    let run = new_run(WINNER);
    let second_reviews = Arc::new(AtomicBool::new(false));
    let allow = Arc::clone(&second_reviews);
    let provider = model(move |request| match request.stage.as_str() {
        "paper_revision" => panic!("no edit should be paid for when a saved attempt passes"),
        "review"
            if request.messages[0].content.contains("Saved attempt:")
                && !allow.load(Ordering::SeqCst) =>
        {
            Some("the reviewer is unavailable".to_string())
        }
        _ => None,
    });
    let ctx = context(&run, provider.clone());
    run_paper(&ctx, &offline()).await.unwrap();
    run_review(&ctx, &reviewer_config(), "ai_review")
        .await
        .unwrap();
    let mut saved: Draft = read(&run, "paper/sections.json");
    saved.sections.insert(
        "abstract".into(),
        "Saved attempt: the search raised the combined score from 6 to 9.84.".into(),
    );
    fsutil::write_json(
        &run.paths.revision().join("attempt_01/sections.json"),
        &saved,
    )
    .unwrap();

    let record = run_revision(
        &ctx,
        &offline(),
        &reviewer_config(),
        &RevisionConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        (record.status.as_str(), record.overall_after),
        ("promoted", None),
        "the second review failed"
    );
    assert!(text(&run, "paper/paper.md").contains("Saved attempt: the search raised"));
    let manifest: PaperManifest = load_manifest(&run.paths).unwrap();
    assert_eq!(manifest.reviews["after_revision"].status, "failed");
    assert_eq!(manifest.revision.as_ref().unwrap().overall_after, None);

    second_reviews.store(true, Ordering::SeqCst);
    let retried = run_revision(
        &ctx,
        &offline(),
        &reviewer_config(),
        &RevisionConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(retried.overall_after, Some(4.0));
    let manifest: PaperManifest = load_manifest(&run.paths).unwrap();
    assert_eq!(manifest.reviews["after_revision"].status, "completed");
    assert_eq!(manifest.revision, Some(retried));
}

#[tokio::test]
async fn a_revision_the_audit_rejects_leaves_the_published_paper_alone() {
    let run = new_run(WINNER);
    let attempts = AtomicUsize::new(0);
    let provider = model(move |request| match request.stage.as_str() {
        "paper_revision" => {
            let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
            if attempt == 2 {
                assert!(
                    request
                        .last_user()
                        .contains("# Your previous revision failed its checks"),
                    "{}",
                    request.last_user()
                );
                assert!(request.last_user().contains("\"12.5\""));
            }
            let nth = if attempt == 1 { "first" } else { "second" };
            Some(format!("<ABSTRACT>In the {nth} revision the search now reaches 12.5 on every task.</ABSTRACT>"))
        }
        _ => None,
    });
    let ctx = context(&run, provider.clone());
    run_paper(&ctx, &offline()).await.unwrap();
    run_review(&ctx, &reviewer_config(), "ai_review")
        .await
        .unwrap();
    let original_tex = text(&run, "paper/paper.tex");
    let original_sections = text(&run, "paper/sections.json");

    let error = run_revision(
        &ctx,
        &offline(),
        &reviewer_config(),
        &RevisionConfig::default(),
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("the published paper is unchanged"),
        "{error}"
    );
    assert_eq!(text(&run, "paper/paper.tex"), original_tex);
    assert_eq!(text(&run, "paper/sections.json"), original_sections);
    assert!(!run.paths.paper().join("revisions").exists());
    assert_eq!(stage_calls(&provider, "paper_revision").len(), 2);
    for attempt in ["attempt_01", "attempt_02"] {
        let result: Value = read(&run, &format!("revision/{attempt}/attempt_result.json"));
        assert_eq!(
            (result["passes"].clone(), result["audit_findings"].clone()),
            (json!(false), json!(1))
        );
    }
    let manifest: PaperManifest = load_manifest(&run.paths).unwrap();
    assert!(manifest.revision.is_none());
    assert!(
        manifest.reviews.contains_key("before_revision")
            && !manifest.reviews.contains_key("after_revision")
    );
    assert!(
        stage_calls(&provider, "review").len() == 3,
        "no second review of a paper that did not change"
    );
}
