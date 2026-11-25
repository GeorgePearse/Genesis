use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use genesis_cli::doctor;
use genesis_cli::inspect;
use genesis_cli::pipeline::{describe_plan, run_pipeline, EvolveConfig, PipelineConfig};
use genesis_core::{config, fsutil, ConsoleMode, CostLedger, EventLog, Task};
use genesis_evolve::{EvolutionConfig, EvolutionRunner, Operator, RunnerDeps};
use genesis_llm::{
    agent_available, missing_credentials, resolve_model, AgentSpec, ClientOptions, LlmClient,
};
use genesis_paper::{review_manuscript, ReviewConfig};
use genesis_scientist::agent_operator::AgentOperator;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Parser)]
#[command(
    name = "genesis",
    version,
    about = "LLM-driven program evolution and automated science, in Rust",
    long_about = "Genesis evolves programs with an ensemble of models and coding agents, then studies what it found: \
                  ablations, a sealed final evaluation, post-hoc analysis and a paper with an evidence audit.\n\n\
                  Settings are `key=value` overrides after the task directory, for example\n  \
                  genesis run examples/circle_packing_rs 'models=[claude-sonnet-5-5]' investigation.evolution.num_generations=30"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the science pipeline: search, ablations, final evaluation, post-hoc analysis, paper.
    Run {
        /// Task directory, then `key=value` overrides.
        args: Vec<String>,
        /// YAML configuration file; overrides are applied on top of it.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Validate the configuration and print the plan without running anything.
        #[arg(long)]
        dry_run: bool,
    },
    /// Run only the evolution loop on a task.
    Evolve {
        /// Task directory, then `evo.*`, `db.*` and `logging.*` overrides.
        args: Vec<String>,
        #[arg(long)]
        config: Option<PathBuf>,
        /// Where the run is written; an existing run directory is resumed.
        #[arg(long)]
        results_dir: Option<PathBuf>,
    },
    /// Evaluate one program with a task's evaluator and print the outcome as JSON.
    Eval {
        task_dir: PathBuf,
        /// Program to evaluate; the task's seed program when omitted.
        #[arg(long)]
        program: Option<PathBuf>,
        #[arg(long)]
        results_dir: Option<PathBuf>,
        #[arg(long)]
        seed: Option<u64>,
        /// Use the task's final evaluator.
        #[arg(long = "final")]
        final_evaluator: bool,
    },
    /// Review a manuscript (text, Markdown or LaTeX) with the reviewer ensemble.
    Review {
        file: PathBuf,
        #[arg(long, default_value = "claude-sonnet-5-5")]
        model: String,
        #[arg(long, default_value_t = 5)]
        reviewers: usize,
        /// Write the review JSON here instead of standard output.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Check that a run can start: task, evaluator, credentials, agents, tools.
    Doctor {
        /// Task directory, then `key=value` overrides.
        args: Vec<String>,
        #[arg(long)]
        config: Option<PathBuf>,
        /// Also run the evaluator once on the seed program.
        #[arg(long)]
        evaluate: bool,
        #[arg(long)]
        json: bool,
    },
    /// Browse runs in a local web page.
    Inspect {
        #[arg(long, default_value = "results")]
        results_root: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8765)]
        port: u16,
    },
    /// Print a default configuration as YAML.
    Config {
        #[arg(value_enum, default_value_t = ConfigKind::Pipeline)]
        kind: ConfigKind,
    },
    /// Show how model names resolve and which credentials are missing.
    Models { models: Vec<String> },
}

#[derive(Clone, Copy, ValueEnum)]
enum ConfigKind {
    Pipeline,
    Evolve,
}

/// Split positional arguments into the task directory and the overrides.
fn split_args(args: &[String]) -> Result<(Option<String>, Vec<String>)> {
    let mut task = None;
    let mut overrides = Vec::new();
    for arg in args {
        if arg.contains('=') {
            overrides.push(arg.clone());
        } else if task.is_none() {
            task = Some(arg.clone());
        } else {
            bail!("unexpected argument `{arg}`: only one task directory is taken, and settings are key=value");
        }
    }
    Ok((task, overrides))
}

/// The first Ctrl-C asks the run to stop cleanly; the second exits at once.
fn install_interrupt_handler() -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    tokio::spawn(async move {
        let mut seen = false;
        while tokio::signal::ctrl_c().await.is_ok() {
            if seen {
                eprintln!("interrupted again, exiting now");
                std::process::exit(130);
            }
            seen = true;
            flag.store(true, Ordering::SeqCst);
            eprintln!(
                "interrupt received: stopping after in-flight work (Ctrl-C again to exit now)"
            );
        }
    });
    cancel
}

fn pipeline_config(args: &[String], file: Option<&Path>) -> Result<PipelineConfig> {
    let (task, mut overrides) = split_args(args)?;
    if let Some(task) = task {
        overrides.insert(0, format!("task.dir={task}"));
    }
    PipelineConfig::load(file, &overrides)
}

async fn run(args: Vec<String>, file: Option<PathBuf>, dry_run: bool) -> Result<i32> {
    let config = pipeline_config(&args, file.as_deref())?;
    if dry_run {
        println!("{}", describe_plan(&config));
        println!("# resolved configuration\n{}", config::to_yaml(&config));
        return Ok(0);
    }
    let missing = missing_credentials(&config.models_in_use());
    if !missing.is_empty() {
        bail!("cannot start:\n  {}", missing.join("\n  "));
    }
    let report = run_pipeline(config, install_interrupt_handler()).await?;
    eprintln!();
    for phase in &report.phases {
        let reason = phase
            .status
            .reason()
            .map(|reason| format!("  ({reason})"))
            .unwrap_or_default();
        eprintln!("  {:<17} {}{reason}", phase.phase, phase.status.label());
    }
    eprintln!(
        "Pipeline finished · {} · {} · ${:.4} model spend\n  run: {}",
        report.status,
        fsutil::human_duration(report.elapsed_secs),
        report.total_cost_usd,
        report.run_dir.display()
    );
    if let Some(paper) = &report.paper {
        eprintln!("  paper: {paper}");
    }
    Ok(report.exit_code())
}

async fn evolve(
    args: Vec<String>,
    file: Option<PathBuf>,
    results_dir: Option<PathBuf>,
) -> Result<i32> {
    let (task_dir, overrides) = split_args(&args)?;
    let task_dir =
        task_dir.context("give the task directory: genesis evolve <task_dir> [key=value ...]")?;
    let config: EvolveConfig = config::load(file.as_deref(), &overrides)?;
    config.evo.validate()?;
    let text_models: Vec<String> = config
        .evo
        .llm_models
        .iter()
        .filter(|arm| !EvolutionConfig::is_agent_arm(arm))
        .cloned()
        .collect();
    let missing = missing_credentials(&text_models);
    if !missing.is_empty() {
        bail!("cannot start:\n  {}", missing.join("\n  "));
    }
    let task = Task::load(Path::new(&task_dir))?;
    let results_dir = results_dir.unwrap_or_else(|| {
        PathBuf::from("results")
            .join(&task.name)
            .join(fsutil::new_run_id())
    });
    std::fs::create_dir_all(&results_dir)?;
    fsutil::write_text(
        &results_dir.join("evolve_config.yaml"),
        &config::to_yaml(&config),
    )?;
    let ledger = Arc::new(CostLedger::open(&results_dir)?);
    let prior_cost_usd = ledger.total_usd();
    let llm = LlmClient::new(ClientOptions {
        ledger: Some(Arc::clone(&ledger)),
        budget_usd: config
            .evo
            .max_api_costs
            .map(|budget| (budget - prior_cost_usd).max(1e-9)),
        phase: "evolve".to_string(),
        work_dir: Some(results_dir.join(".llm_scratch")),
        ..ClientOptions::default()
    });
    let events = EventLog::open(
        &results_dir.join("events.jsonl"),
        ConsoleMode::parse(&config.logging.format),
        config.logging.verbose,
    );
    let agent_operator: Arc<dyn Operator> = Arc::new(AgentOperator::new(
        config.agents.clone(),
        config.agent_timeout_secs,
        Some(ledger),
        "evolve",
    ));
    let runner = EvolutionRunner::new(
        config.evo,
        config.db,
        RunnerDeps {
            task,
            results_dir: results_dir.clone(),
            llm,
            events,
            agent_operator: Some(agent_operator),
            idea_source: None,
            eval_hooks: Vec::new(),
            cancel: install_interrupt_handler(),
            prior_cost_usd,
        },
    )?;
    let summary = runner.run().await?;
    match &summary.best {
        Some(best) => eprintln!(
            "Evolution finished · {} · best {:.6} from generation {} · {} programs · ${:.4}\n  run: {}\n  best: {}",
            summary.stop_reason,
            best.combined_score,
            best.generation,
            summary.programs,
            summary.cost_usd,
            results_dir.display(),
            results_dir.join("best").display()
        ),
        None => eprintln!(
            "Evolution finished · {} · no correct program · run: {}",
            summary.stop_reason,
            results_dir.display()
        ),
    }
    if summary.stop_reason == "seed_evaluation_failed" {
        eprintln!(
            "The evaluator could not score the seed program; see {}. Nothing was proposed.",
            results_dir.join("gen_0/results").display()
        );
    }
    Ok(match summary.stop_reason.as_str() {
        "interrupted" => 130,
        _ if summary.evolution_failed => 1,
        _ => 0,
    })
}

async fn eval(
    task_dir: PathBuf,
    program: Option<PathBuf>,
    results_dir: Option<PathBuf>,
    seed: Option<u64>,
    final_evaluator: bool,
) -> Result<i32> {
    let task = Task::load(&task_dir)?;
    let program = program.unwrap_or_else(|| task.init_program.clone());
    let scratch;
    let results_dir = match results_dir {
        Some(dir) => dir,
        None => {
            scratch = std::env::temp_dir().join(format!("genesis-eval-{}", std::process::id()));
            scratch.clone()
        }
    };
    let outcome = if final_evaluator {
        task.evaluate_final(&program, &results_dir, seed, None)
            .await?
    } else {
        task.evaluate(&program, &results_dir, seed, None).await?
    };
    println!("{}", serde_json::to_string_pretty(&outcome)?);
    Ok(if outcome.correct { 0 } else { 1 })
}

async fn review(
    file: PathBuf,
    model: String,
    reviewers: usize,
    output: Option<PathBuf>,
) -> Result<i32> {
    let manuscript = std::fs::read_to_string(&file)
        .with_context(|| format!("reading {} as text", file.display()))?;
    let missing = missing_credentials(std::slice::from_ref(&model));
    if !missing.is_empty() {
        bail!("cannot start:\n  {}", missing.join("\n  "));
    }
    let llm = LlmClient::new(ClientOptions {
        phase: "review".to_string(),
        ..ClientOptions::default()
    });
    let config = ReviewConfig {
        model,
        reviewer_count: reviewers,
        ..ReviewConfig::default()
    };
    let review = review_manuscript(&llm, &config, &manuscript).await?;
    let text = serde_json::to_string_pretty(&review)?;
    match output {
        Some(path) => {
            fsutil::write_text(&path, &format!("{text}\n"))?;
            eprintln!(
                "Review · overall {}/10 · {} · {}",
                review.overall,
                review.decision,
                path.display()
            );
        }
        None => println!("{text}"),
    }
    Ok(0)
}

async fn doctor_command(
    args: Vec<String>,
    file: Option<PathBuf>,
    evaluate: bool,
    as_json: bool,
) -> Result<i32> {
    let config = pipeline_config(&args, file.as_deref())?;
    let checks = doctor::run_checks(&config, evaluate).await;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&checks)?);
    } else {
        println!("{}", doctor::render(&checks));
    }
    Ok(if checks.iter().any(|check| check.status == "fail") {
        1
    } else {
        0
    })
}

fn models(models: Vec<String>) -> Result<i32> {
    let models = if models.is_empty() {
        PipelineConfig::load(None, &[])?.models_in_use()
    } else {
        models
    };
    let mut unusable = 0;
    for model in &models {
        if EvolutionConfig::is_agent_arm(model) {
            match AgentSpec::parse(model) {
                Some(spec) if agent_available(&spec) => {
                    println!("{model:<40} {:<14} ready", "agent")
                }
                _ => {
                    unusable += 1;
                    println!("{model:<40} {:<14} the agent CLI is not installed", "agent");
                }
            }
            continue;
        }
        let route = match resolve_model(model) {
            Ok(route) => route.provider().to_string(),
            Err(error) => {
                unusable += 1;
                println!("{model:<40} unknown: {error}");
                continue;
            }
        };
        match missing_credentials(std::slice::from_ref(model)).first() {
            Some(problem) => {
                unusable += 1;
                println!("{model:<40} {route:<14} {problem}");
            }
            None => println!("{model:<40} {route:<14} ready"),
        }
    }
    Ok(if unusable == 0 { 0 } else { 1 })
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let outcome = match cli.command {
        Command::Run {
            args,
            config,
            dry_run,
        } => run(args, config, dry_run).await,
        Command::Evolve {
            args,
            config,
            results_dir,
        } => evolve(args, config, results_dir).await,
        Command::Eval {
            task_dir,
            program,
            results_dir,
            seed,
            final_evaluator,
        } => eval(task_dir, program, results_dir, seed, final_evaluator).await,
        Command::Review {
            file,
            model,
            reviewers,
            output,
        } => review(file, model, reviewers, output).await,
        Command::Doctor {
            args,
            config,
            evaluate,
            json,
        } => doctor_command(args, config, evaluate, json).await,
        Command::Inspect {
            results_root,
            host,
            port,
        } => inspect::serve(&results_root, &host, port).await.map(|()| 0),
        Command::Config { kind } => {
            match kind {
                ConfigKind::Pipeline => print!("{}", config::to_yaml(&PipelineConfig::default())),
                ConfigKind::Evolve => print!("{}", config::to_yaml(&EvolveConfig::default())),
            }
            Ok(0)
        }
        Command::Models { models: names } => models(names),
    };
    match outcome {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(1);
        }
    }
}
