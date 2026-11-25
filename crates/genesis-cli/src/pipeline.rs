//! The end-to-end pipeline: one command from a task directory to a paper.
//!
//! Phases run in a fixed order: `task_drafting`, `investigation`,
//! `ablations`, `final_evaluation`, `posthoc`, `paper`, and the opt-in
//! `review` and `revision`. Task drafting, the investigation and the paper
//! are gating: when one fails the pipeline stops. The others add evidence
//! when they succeed and are recorded as degraded when they do not; a
//! publishable paper is never discarded because optional evidence is
//! missing.
//!
//! Every phase resumes against the run directory: pass `paths.run_dir` and
//! the phases to run.

use anyhow::{bail, Context, Result};
use genesis_core::{config, fsutil, ConsoleMode, CostLedger, EventLog, Task};
use genesis_evidence::{
    load_final_manifest, run_ablations, run_final_evaluation, run_posthoc, AblationConfig,
    FinalEvalConfig, PosthocConfig,
};
use genesis_evolve::{DatabaseConfig, EvolutionConfig, ProgramDb};
use genesis_llm::{missing_credentials, AgentSpec, ClientOptions, LlmClient, Provider};
use genesis_paper::{
    run_paper, run_review, run_revision, PaperConfig, ReviewConfig, RevisionConfig,
};
use genesis_scientist::artifacts::InvestigationCompletion;
use genesis_scientist::drafting::{draft_task, scaffold_is_complete, DraftingConfig};
use genesis_scientist::selection::{freeze_selection, load_selection};
use genesis_scientist::{
    run_investigation, InvestigationConfig, PhaseContext, PhaseStatus, RunPaths,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

pub const PHASES: [&str; 8] = [
    "task_drafting",
    "investigation",
    "ablations",
    "final_evaluation",
    "posthoc",
    "paper",
    "review",
    "revision",
];
const DEFAULT_PHASES: [&str; 6] = [
    "task_drafting",
    "investigation",
    "ablations",
    "final_evaluation",
    "posthoc",
    "paper",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TaskSection {
    /// The task directory. Read-only: the run works on a snapshot.
    pub dir: String,
}

impl Default for TaskSection {
    fn default() -> Self {
        Self {
            dir: "examples/circle_packing_rs".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PathsSection {
    pub results_root: String,
    /// An existing run to resume or to run more phases against.
    pub run_dir: Option<String>,
}

impl Default for PathsSection {
    fn default() -> Self {
        Self {
            results_root: "results".to_string(),
            run_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ExecutionSection {
    pub phases: Vec<String>,
    /// Stop at the first failed phase, gating or not.
    pub fail_fast: bool,
    /// What downstream phases study when the search found no winner:
    /// `best_available` or `skip`.
    pub no_winner_policy: String,
    pub seed: Option<u64>,
}

impl Default for ExecutionSection {
    fn default() -> Self {
        Self {
            phases: DEFAULT_PHASES
                .iter()
                .map(|phase| phase.to_string())
                .collect(),
            fail_fast: false,
            no_winner_policy: "best_available".to_string(),
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LoggingSection {
    /// `human`, `json` or `quiet`.
    pub format: String,
    pub verbose: bool,
}

impl Default for LoggingSection {
    fn default() -> Self {
        Self {
            format: "human".to_string(),
            verbose: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PipelineConfig {
    pub task: TaskSection,
    pub paths: PathsSection,
    pub execution: ExecutionSection,
    pub logging: LoggingSection,
    /// Models used wherever a phase names none. When set, it also replaces
    /// the investigation's proposal models.
    pub models: Vec<String>,
    /// Spending cap in USD for the whole run, every phase included.
    pub budget_usd: Option<f64>,
    /// Coding agent for agentic stages, as `claude`, `codex@model`, ...
    pub agent: Option<String>,
    pub task_drafting: DraftingConfig,
    pub investigation: InvestigationConfig,
    pub ablations: AblationConfig,
    pub final_evaluation: FinalEvalConfig,
    pub posthoc: PosthocConfig,
    pub paper: PaperConfig,
    pub review: ReviewConfig,
    pub revision: RevisionConfig,
}

impl Default for PipelineConfig {
    /// Phase model lists start empty, meaning "not chosen": `load` fills
    /// them from `models` or from the investigation's models. A phase's own
    /// default model would otherwise demand a credential the run never set.
    fn default() -> Self {
        Self {
            task: TaskSection::default(),
            paths: PathsSection::default(),
            execution: ExecutionSection::default(),
            logging: LoggingSection::default(),
            models: Vec::new(),
            budget_usd: None,
            agent: None,
            task_drafting: DraftingConfig::default(),
            investigation: InvestigationConfig::default(),
            ablations: AblationConfig {
                models: Vec::new(),
                ..AblationConfig::default()
            },
            final_evaluation: FinalEvalConfig::default(),
            posthoc: PosthocConfig {
                models: Vec::new(),
                ..PosthocConfig::default()
            },
            paper: PaperConfig {
                models: Vec::new(),
                audit_models: Vec::new(),
                ..PaperConfig::default()
            },
            review: ReviewConfig {
                model: String::new(),
                ..ReviewConfig::default()
            },
            revision: RevisionConfig::default(),
        }
    }
}

impl PipelineConfig {
    pub fn load(file: Option<&Path>, overrides: &[String]) -> Result<Self> {
        let mut config: Self = config::load(file, overrides)?;
        config.apply_model_defaults();
        config.validate()?;
        Ok(config)
    }

    /// Fill every empty model list from `models`, or else from the
    /// investigation's text models. The search, ablations, post-hoc analysis
    /// and drafting use the whole list; the paper and the review use its
    /// first entry, so list the strongest model first.
    fn apply_model_defaults(&mut self) {
        if !self.models.is_empty() {
            self.investigation.evolution.llm_models = self.models.clone();
        }
        let text: Vec<String> = self
            .investigation
            .evolution
            .llm_models
            .iter()
            .filter(|arm| !EvolutionConfig::is_agent_arm(arm))
            .cloned()
            .collect();
        for list in [
            &mut self.task_drafting.models,
            &mut self.ablations.models,
            &mut self.posthoc.models,
        ] {
            if list.is_empty() {
                *list = text.clone();
            }
        }
        if self.paper.models.is_empty() {
            self.paper.models = text.first().cloned().into_iter().collect();
        }
        if self.review.model.is_empty() {
            self.review.model = text.first().cloned().unwrap_or_default();
        }
    }

    pub fn validate(&self) -> Result<()> {
        for phase in &self.execution.phases {
            if !PHASES.contains(&phase.as_str()) {
                bail!("unknown phase `{phase}`; phases are: {}", PHASES.join(", "));
            }
        }
        if !["best_available", "skip"].contains(&self.execution.no_winner_policy.as_str()) {
            bail!("execution.no_winner_policy must be best_available or skip");
        }
        if self.selected("revision") && !self.selected("review") {
            bail!("the revision phase needs the review phase in the same selection");
        }
        if let Some(budget) = self.budget_usd {
            if budget <= 0.0 {
                bail!("budget_usd must be positive");
            }
        }
        self.investigation.resolved()?.evolution.validate()?;
        self.investigation.database.validate()?;
        Ok(())
    }

    pub fn selected(&self, phase: &str) -> bool {
        self.execution
            .phases
            .iter()
            .any(|selected| selected == phase)
    }

    /// Every model the selected phases will call. Coding-agent arms are not
    /// models; `doctor` checks them separately.
    pub fn models_in_use(&self) -> Vec<String> {
        let mut models: Vec<String> = Vec::new();
        let mut add = |list: &[String]| {
            for model in list {
                if !model.is_empty()
                    && !EvolutionConfig::is_agent_arm(model)
                    && !models.contains(model)
                {
                    models.push(model.clone());
                }
            }
        };
        if self.selected("task_drafting") {
            add(&self.task_drafting.models);
        }
        if self.selected("investigation") {
            let evolution = &self.investigation.evolution;
            add(&evolution.llm_models);
            add(&evolution.meta_llm_models);
            add(&evolution.novelty_llm_models);
            add(&evolution.prompt_llm_models);
            add(&self.investigation.ideas.models);
            add(&self.investigation.rubric_judge_models);
            add(&self.investigation.conclusions_models);
        }
        if self.selected("ablations") {
            add(&self.ablations.models);
        }
        if self.selected("posthoc") {
            add(&self.posthoc.models);
        }
        if self.selected("paper") {
            add(&self.paper.models);
            add(&self.paper.audit_models);
        }
        if self.selected("review") || self.selected("revision") {
            add(std::slice::from_ref(&self.review.model));
        }
        models
    }
}

/// Configuration of the standalone evolution command.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EvolveConfig {
    pub evo: EvolutionConfig,
    pub db: DatabaseConfig,
    pub logging: LoggingSection,
    /// Custom agents by name, usable as `agent:<name>` in `evo.llm_models`.
    pub agents: BTreeMap<String, AgentSpec>,
    /// Wall-clock limit of one coding-agent attempt.
    pub agent_timeout_secs: u64,
}

impl Default for EvolveConfig {
    fn default() -> Self {
        Self {
            evo: EvolutionConfig::default(),
            db: DatabaseConfig::default(),
            logging: LoggingSection::default(),
            agents: BTreeMap::new(),
            agent_timeout_secs: 1800,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhaseReport {
    pub phase: String,
    #[serde(flatten)]
    pub status: PhaseStatus,
    pub elapsed_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineReport {
    pub run_dir: PathBuf,
    /// `completed`, `completed_degraded`, `failed` or `interrupted`.
    pub status: String,
    pub phases: Vec<PhaseReport>,
    pub total_cost_usd: f64,
    pub elapsed_secs: f64,
    pub paper: Option<String>,
}

impl PipelineReport {
    /// 0 when the run completed, degraded or not; 2 when a gating phase
    /// failed; 130 when it was interrupted.
    pub fn exit_code(&self) -> i32 {
        match self.status.as_str() {
            "completed" | "completed_degraded" => 0,
            "interrupted" => 130,
            _ => 2,
        }
    }
}

fn is_gating(phase: &str) -> bool {
    matches!(phase, "task_drafting" | "investigation" | "paper")
}

/// The plan `--dry-run` prints: what would run, with which models, where.
pub fn describe_plan(config: &PipelineConfig) -> String {
    let mut out = String::new();
    let run_dir = match &config.paths.run_dir {
        Some(dir) => format!("{dir} (resume)"),
        None => format!("{}/<task>/<run_id> (new)", config.paths.results_root),
    };
    out.push_str(&format!(
        "task:      {}\nrun dir:   {run_dir}\n",
        config.task.dir
    ));
    if let Some(budget) = config.budget_usd {
        out.push_str(&format!("budget:    ${budget:.2} across all phases\n"));
    }
    out.push_str("phases:\n");
    for phase in PHASES {
        if !config.selected(phase) {
            continue;
        }
        let detail = match phase {
            "task_drafting" => format!(
                "draft the evaluator and seed if missing; models {:?}",
                config.task_drafting.models
            ),
            "investigation" => {
                let evolution = &config.investigation.evolution;
                format!(
                    "engine {}, {} generations, {} in flight, arms {:?}",
                    config.investigation.engine,
                    evolution.num_generations,
                    evolution.max_evaluation_jobs,
                    evolution.llm_models
                )
            }
            "ablations" => format!(
                "up to {} ablations and {} baselines, {} replications each; models {:?}",
                config.ablations.max_ablations,
                config.ablations.max_extra_baselines,
                config.ablations.num_replications,
                config.ablations.models
            ),
            "final_evaluation" => format!(
                "{} sealed evaluations from seed {}",
                config.final_evaluation.num_replications, config.final_evaluation.seed_base
            ),
            "posthoc" => format!(
                "up to {} analyses; models {:?}",
                config.posthoc.max_selected_analyses, config.posthoc.models
            ),
            "paper" => format!(
                "format {}, {} to {} references, up to {} repair rounds; models {:?}",
                config.paper.paper_format,
                config.paper.min_refs,
                config.paper.max_refs,
                config.paper.max_repair_rounds,
                config.paper.models
            ),
            "review" => format!(
                "{} reviewers with {}",
                config.review.reviewer_count, config.review.model
            ),
            _ => "revise the paper against the review, then review it again".to_string(),
        };
        let kind = if is_gating(phase) {
            "gating"
        } else {
            "best effort"
        };
        out.push_str(&format!("  - {phase:<17} [{kind}] {detail}\n"));
    }
    let missing = missing_credentials(&config.models_in_use());
    if !missing.is_empty() {
        out.push_str("credentials:\n");
        for problem in missing {
            out.push_str(&format!("  ! {problem}\n"));
        }
    }
    out
}

struct Run {
    config: PipelineConfig,
    context: PhaseContext,
    reports: Vec<PhaseReport>,
    started: Instant,
}

impl Run {
    fn record(&mut self, phase: &str, status: PhaseStatus, started: Instant) {
        let elapsed_secs = started.elapsed().as_secs_f64();
        self.context.events.info(
            "phase_end",
            json!({
                "phase": phase,
                "status": status.label(),
                "reason": status.reason(),
                "elapsed": fsutil::human_duration(elapsed_secs),
                "run_cost_usd": self.context.llm.ledger().map(|ledger| ledger.total_usd()),
            }),
        );
        self.reports.push(PhaseReport {
            phase: phase.to_string(),
            status,
            elapsed_secs,
        });
    }

    fn failed(&self) -> bool {
        self.reports
            .iter()
            .any(|report| matches!(report.status, PhaseStatus::Failed(_)))
    }

    fn investigation_db(&self) -> Result<ProgramDb> {
        let path = self
            .context
            .paths
            .investigation()
            .join(&self.config.investigation.database.db_path);
        if !path.is_file() {
            bail!(
                "{} does not exist: run the investigation phase first",
                path.display()
            );
        }
        ProgramDb::open(Some(&path), self.config.investigation.resolved()?.database)
    }

    /// Make sure the downstream selection is frozen. `Ok(false)` when there
    /// is nothing to study.
    fn ensure_selection(&self) -> Result<bool> {
        if load_selection(&self.context.paths)?.is_some() {
            return Ok(true);
        }
        let receipt: Option<InvestigationCompletion> =
            fsutil::read_json_opt(&self.context.paths.completion_receipt());
        if !receipt.is_some_and(|receipt| receipt.status == "completed") {
            bail!(
                "the investigation has not completed; resume it before running downstream phases"
            );
        }
        let db = self.investigation_db()?;
        let selected = freeze_selection(
            &self.context.paths,
            &self.context.task,
            &db,
            &self.config.execution.no_winner_policy,
        )?;
        if let Some(selected) = &selected {
            self.context.events.info(
                "downstream_selection",
                json!({
                    "selection_kind": selected.selection.selection_kind,
                    "generation": selected.selection.generation,
                    "combined_score": selected.selection.combined_score,
                    "baseline_score": selected.selection.baseline_score,
                }),
            );
        }
        Ok(selected.is_some())
    }

    async fn phase(&mut self, phase: &str) -> PhaseStatus {
        let context = self.context.for_phase(phase);
        match phase {
            "investigation" => {
                match run_investigation(&context, &self.config.investigation).await {
                    Ok(outcome) if outcome.completed => PhaseStatus::Completed,
                    Ok(outcome) if outcome.summary.stop_reason == "seed_evaluation_failed" => {
                        PhaseStatus::Failed(format!(
                            "the evaluator could not score the seed program (see {}); fix the task and start a new run",
                            self.context.paths.investigation().join("gen_0/results").display()
                        ))
                    }
                    Ok(outcome) => PhaseStatus::Failed(format!(
                        "the search stopped early ({}); resume with paths.run_dir={}",
                        outcome.summary.stop_reason,
                        self.context.paths.run_dir.display()
                    )),
                    Err(error) => PhaseStatus::Failed(format!("{error:#}")),
                }
            }
            "ablations" | "final_evaluation" | "posthoc" | "paper" => {
                match self.ensure_selection() {
                    Ok(true) => {}
                    Ok(false) => {
                        return PhaseStatus::Skipped(
                            "the search left no correct program to study".to_string(),
                        )
                    }
                    Err(error) => return PhaseStatus::Failed(format!("{error:#}")),
                }
                let outcome: Result<Option<String>> = match phase {
                    "ablations" => run_ablations(&context, &self.config.ablations).await.map(|summary| {
                        (summary.total_completed <= 1).then(|| "no variant beyond the control completed".to_string())
                    }),
                    "final_evaluation" => {
                        run_final_evaluation(&context, &self.config.final_evaluation).await.map(|_| None)
                    }
                    "posthoc" => run_posthoc(&context, &self.config.posthoc).await.map(|manifest| {
                        manifest.valid_analyses.is_empty().then(|| "no valid analyses".to_string())
                    }),
                    _ => run_paper(&context, &self.config.paper).await.map(|manifest| {
                        (!manifest.validation_ok || manifest.evidence_audit_verdict != "pass").then(|| {
                            format!(
                                "the paper was published with validation_ok={} and evidence audit `{}`",
                                manifest.validation_ok, manifest.evidence_audit_verdict
                            )
                        })
                    }),
                };
                match outcome {
                    Ok(None) => PhaseStatus::Completed,
                    Ok(Some(reason)) => PhaseStatus::Degraded(reason),
                    Err(error) if is_gating(phase) => PhaseStatus::Failed(format!("{error:#}")),
                    Err(error) => PhaseStatus::Degraded(format!("{error:#}")),
                }
            }
            "review" => {
                if !self
                    .context
                    .paths
                    .paper()
                    .join("latest_manifest.json")
                    .is_file()
                {
                    return PhaseStatus::Skipped(
                        "there is no published paper to review".to_string(),
                    );
                }
                match run_review(&context, &self.config.review, "ai_review").await {
                    Ok(review) => {
                        self.context.events.info(
                            "review_result",
                            json!({"overall": review.overall, "decision": review.decision}),
                        );
                        PhaseStatus::Completed
                    }
                    Err(error) => PhaseStatus::Degraded(format!("{error:#}")),
                }
            }
            "revision" => match run_revision(
                &context,
                &self.config.paper,
                &self.config.review,
                &self.config.revision,
            )
            .await
            {
                Ok(record) => {
                    self.context.events.info(
                        "revision_result",
                        json!({
                            "before_overall_score": record.overall_before,
                            "after_overall_score": record.overall_after,
                            "before_decision": record.decision_before,
                            "after_decision": record.decision_after,
                        }),
                    );
                    PhaseStatus::Completed
                }
                Err(error) => PhaseStatus::Degraded(format!("{error:#}")),
            },
            other => PhaseStatus::Failed(format!("unknown phase `{other}`")),
        }
    }
}

fn console_mode(logging: &LoggingSection) -> ConsoleMode {
    ConsoleMode::parse(&logging.format)
}

/// Prepare the run directory and the task snapshot, drafting the scaffold
/// first when asked to and when it is missing.
async fn prepare(
    config: &PipelineConfig,
    cancel: Arc<AtomicBool>,
    provider: Option<Arc<dyn Provider>>,
) -> Result<(PhaseContext, Option<(PhaseStatus, f64)>)> {
    let resuming = config.paths.run_dir.is_some();
    let source_dir = PathBuf::from(&config.task.dir);
    let paths = match &config.paths.run_dir {
        Some(dir) => {
            let paths = RunPaths::new(dir);
            if !paths.run_dir.is_dir() {
                bail!("paths.run_dir {} does not exist", paths.run_dir.display());
            }
            paths
        }
        None => {
            let name = source_dir
                .canonicalize()
                .with_context(|| format!("task directory {} does not exist", source_dir.display()))?
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "task".to_string());
            RunPaths::for_new_run(
                Path::new(&config.paths.results_root),
                &name,
                &fsutil::new_run_id(),
            )
        }
    };
    std::fs::create_dir_all(&paths.run_dir)?;
    let run_dir = paths.run_dir.canonicalize()?;
    let paths = RunPaths::new(run_dir);
    fsutil::write_text(&paths.config(), &config::to_yaml(config))?;

    let events = EventLog::open(
        &paths.events(),
        console_mode(&config.logging),
        config.logging.verbose,
    );
    let ledger = Arc::new(CostLedger::open(&paths.run_dir)?);
    let remaining = config
        .budget_usd
        .map(|budget| (budget - ledger.total_usd()).max(1e-9));
    let llm = LlmClient::new(ClientOptions {
        ledger: Some(Arc::clone(&ledger)),
        budget_usd: remaining,
        phase: "pipeline".to_string(),
        work_dir: Some(paths.run_dir.join(".llm_scratch")),
        mock: provider,
        ..ClientOptions::default()
    });
    events.info(
        "pipeline_start",
        json!({
            "run_dir": paths.run_dir.display().to_string(),
            "task": config.task.dir,
            "phases": config.execution.phases,
            "resumed": resuming,
            "prior_cost_usd": ledger.total_usd(),
            "budget_usd": config.budget_usd,
        }),
    );

    let snapshot_dir = paths.task();
    let mut drafting = None;
    let snapshot_exists = snapshot_dir.join(".snapshot_complete").is_file();
    if !snapshot_exists {
        if !source_dir.is_dir() {
            bail!("task directory {} does not exist", source_dir.display());
        }
        if scaffold_is_complete(&source_dir) {
            Task::load(&source_dir)?.snapshot(&snapshot_dir)?;
            if config.selected("task_drafting") {
                drafting = Some((
                    PhaseStatus::Skipped(
                        "the task already has a seed program and an evaluator".into(),
                    ),
                    0.0,
                ));
            }
        } else if config.selected("task_drafting") {
            let started = Instant::now();
            events.info("phase_start", json!({"phase": "task_drafting"}));
            let scoped = llm.with_phase("task_drafting");
            let status = match draft_task(
                &source_dir,
                &snapshot_dir,
                &paths.drafting(),
                &scoped,
                &config.task_drafting,
                &events,
            )
            .await
            {
                Ok(()) => PhaseStatus::Completed,
                Err(error) => PhaseStatus::Failed(format!("{error:#}")),
            };
            drafting = Some((status, started.elapsed().as_secs_f64()));
        } else {
            bail!(
                "{} has no seed program or evaluator; add them, or include the task_drafting phase to draft them from task.md",
                source_dir.display()
            );
        }
    } else if config.selected("task_drafting") {
        drafting = Some((
            PhaseStatus::Skipped("the run already has a task snapshot".into()),
            0.0,
        ));
    }

    // A failed draft leaves no loadable task; the caller reports the failure.
    let drafting_failed = matches!(drafting, Some((PhaseStatus::Failed(_), _)));
    let task = if drafting_failed {
        Task::load(&source_dir)
            .or_else(|_| Task::load(&snapshot_dir))
            .ok()
    } else {
        Some(Task::load(&snapshot_dir)?)
    };
    let Some(task) = task else {
        let reason = drafting
            .as_ref()
            .and_then(|(status, _)| status.reason().map(str::to_string))
            .unwrap_or_default();
        bail!("task drafting failed: {reason}");
    };
    let agent = match &config.agent {
        Some(spec) => Some(
            AgentSpec::parse(&format!("agent:{}", spec.trim_start_matches("agent:")))
                .with_context(|| format!("`agent` is `{spec}`, which is not an agent spec"))?,
        ),
        None => None,
    };
    let seed = config
        .execution
        .seed
        .or(config.investigation.evolution.seed)
        .unwrap_or(0);
    Ok((
        PhaseContext {
            paths,
            task,
            llm,
            events,
            agent,
            cancel,
            seed,
        },
        drafting,
    ))
}

/// Run the selected phases.
pub async fn run_pipeline(
    config: PipelineConfig,
    cancel: Arc<AtomicBool>,
) -> Result<PipelineReport> {
    run_pipeline_with(config, cancel, None).await
}

/// As [`run_pipeline`], with every model call answered by `provider` when
/// one is given. Tests use it to script a whole run.
pub async fn run_pipeline_with(
    config: PipelineConfig,
    cancel: Arc<AtomicBool>,
    provider: Option<Arc<dyn Provider>>,
) -> Result<PipelineReport> {
    config.validate()?;
    let started = Instant::now();
    let (context, drafting) = prepare(&config, Arc::clone(&cancel), provider).await?;
    let mut run = Run {
        config,
        context,
        reports: Vec::new(),
        started,
    };
    if let Some((status, elapsed_secs)) = drafting {
        run.context.events.info(
            "phase_end",
            json!({"phase": "task_drafting", "status": status.label(), "reason": status.reason()}),
        );
        run.reports.push(PhaseReport {
            phase: "task_drafting".to_string(),
            status,
            elapsed_secs,
        });
    }

    let mut interrupted = false;
    for phase in PHASES.iter().skip(1) {
        if !run.config.selected(phase) {
            continue;
        }
        let gating_failed = run.reports.iter().any(|report| {
            is_gating(&report.phase) && matches!(report.status, PhaseStatus::Failed(_))
        });
        if gating_failed || (run.config.execution.fail_fast && run.failed()) {
            let phase_started = Instant::now();
            run.record(
                phase,
                PhaseStatus::Skipped("an earlier phase failed".to_string()),
                phase_started,
            );
            continue;
        }
        if run.context.cancelled() {
            interrupted = true;
            break;
        }
        let phase_started = Instant::now();
        run.context
            .events
            .info("phase_start", json!({"phase": phase}));
        let status = run.phase(phase).await;
        run.record(phase, status, phase_started);
        if run.context.cancelled() {
            interrupted = true;
            break;
        }
    }

    let status =
        if interrupted {
            "interrupted"
        } else if run.reports.iter().any(|report| {
            is_gating(&report.phase) && matches!(report.status, PhaseStatus::Failed(_))
        }) {
            "failed"
        } else if run.reports.iter().any(|report| {
            matches!(
                report.status,
                PhaseStatus::Degraded(_) | PhaseStatus::Failed(_)
            )
        }) {
            "completed_degraded"
        } else {
            "completed"
        };
    let paths = &run.context.paths;
    let paper = fsutil::read_json_opt::<genesis_scientist::artifacts::PaperManifest>(
        &paths.paper().join("latest_manifest.json"),
    )
    .map(|manifest| {
        paths
            .resolve(manifest.paper_pdf.as_deref().unwrap_or(&manifest.paper_md))
            .display()
            .to_string()
    });
    let final_outcome = load_final_manifest(paths)
        .ok()
        .flatten()
        .map(|manifest| manifest.outcome);
    let report = PipelineReport {
        run_dir: paths.run_dir.clone(),
        status: status.to_string(),
        phases: run.reports.clone(),
        total_cost_usd: run
            .context
            .llm
            .ledger()
            .map(|ledger| ledger.total_usd())
            .unwrap_or(0.0),
        elapsed_secs: run.started.elapsed().as_secs_f64(),
        paper,
    };
    fsutil::write_json(&paths.run_dir.join("pipeline_status.json"), &report)?;
    let phase_statuses: BTreeMap<&str, &str> = report
        .phases
        .iter()
        .map(|phase| (phase.phase.as_str(), phase.status.label()))
        .collect();
    run.context.events.info(
        "pipeline_end",
        json!({
            "status": report.status,
            "phases": phase_statuses,
            "final_outcome": final_outcome,
            "paper": report.paper,
            "total_cost_usd": report.total_cost_usd,
            "elapsed": fsutil::human_duration(report.elapsed_secs),
            "run_dir": report.run_dir.display().to_string(),
        }),
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_defaults_flow_to_every_phase() {
        let config =
            PipelineConfig::load(None, &["models=[model-x, agent:claude]".to_string()]).unwrap();
        assert_eq!(
            config.investigation.evolution.llm_models,
            ["model-x", "agent:claude"]
        );
        assert_eq!(config.paper.models, ["model-x"]);
        assert_eq!(config.ablations.models, ["model-x"]);
        assert!(
            config.paper.audit_models.is_empty(),
            "the model audit stays off unless asked for"
        );
        assert_eq!(config.review.model, "model-x");
        let explicit = PipelineConfig::load(
            None,
            &["models=[model-x]".into(), "paper.models=[writer]".into()],
        )
        .unwrap();
        assert_eq!(explicit.paper.models, ["writer"]);
        let defaults = PipelineConfig::load(None, &[]).unwrap();
        assert_eq!(
            defaults.paper.models,
            defaults.investigation.evolution.llm_models[..1]
        );
        assert_eq!(
            defaults.ablations.models,
            defaults.investigation.evolution.llm_models
        );
        let two = PipelineConfig::load(None, &["models=[strong, cheap]".into()]).unwrap();
        assert_eq!(
            (two.paper.models.as_slice(), two.review.model.as_str()),
            (&["strong".to_string()][..], "strong")
        );
        assert_eq!(defaults.execution.phases.len(), 6);
    }

    #[test]
    fn invalid_selections_are_refused() {
        let unknown =
            PipelineConfig::load(None, &["execution.phases=[investigation, publish]".into()])
                .unwrap_err();
        assert!(
            unknown.to_string().contains("unknown phase `publish`"),
            "{unknown}"
        );
        assert!(
            PipelineConfig::load(None, &["execution.phases=[paper, revision]".into()]).is_err()
        );
        assert!(PipelineConfig::load(None, &["execution.no_winner_policy=guess".into()]).is_err());
        assert!(PipelineConfig::load(None, &["budget_usd=0".into()]).is_err());
        assert!(PipelineConfig::load(None, &["investigation.engine=random".into()]).is_err());
        let typo = PipelineConfig::load(None, &["investigation.evolution.num_generation=5".into()])
            .unwrap_err();
        assert!(
            typo.to_string()
                .contains("did you mean `investigation.evolution.num_generations`"),
            "{typo}"
        );
    }

    #[test]
    fn the_plan_names_each_selected_phase_and_the_models_it_uses() {
        let config = PipelineConfig::load(
            None,
            &[
                "models=[mock/perturb]".into(),
                "execution.phases=[investigation, paper, review]".into(),
                "budget_usd=5".into(),
            ],
        )
        .unwrap();
        let plan = describe_plan(&config);
        assert!(
            plan.contains("investigation     [gating]")
                && plan.contains("review            [best effort]"),
            "{plan}"
        );
        assert!(!plan.contains("ablations"));
        assert!(plan.contains("budget:    $5.00"));
        assert_eq!(config.models_in_use(), ["mock/perturb"]);
    }

    #[test]
    fn exit_codes_follow_the_run_status() {
        let report = |status: &str| PipelineReport {
            run_dir: PathBuf::new(),
            status: status.to_string(),
            phases: Vec::new(),
            total_cost_usd: 0.0,
            elapsed_secs: 0.0,
            paper: None,
        };
        assert_eq!(report("completed").exit_code(), 0);
        assert_eq!(report("completed_degraded").exit_code(), 0);
        assert_eq!(report("failed").exit_code(), 2);
        assert_eq!(report("interrupted").exit_code(), 130);
    }
}
