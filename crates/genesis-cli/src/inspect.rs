//! `genesis inspect`: a local web page for browsing runs.
//!
//! Read-only. It serves what is already on disk under the results root: the
//! program archive, lineage and diffs, the event journal, costs, and the
//! paper. Every path is confined to the results root, and program text is
//! only ever sent as JSON strings that the page inserts as text.

use anyhow::{Context, Result};
use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use genesis_core::fsutil;
use genesis_evolve::ProgramDb;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const PAGE: &str = include_str!("inspect.html");
const MAX_PREVIEW_BYTES: u64 = 1_048_576;

struct AppState {
    root: PathBuf,
}

type Shared = Arc<AppState>;

#[derive(Debug)]
struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
    }
}

fn not_found(what: impl Into<String>) -> ApiError {
    ApiError(StatusCode::NOT_FOUND, what.into())
}

/// Resolve `relative` under `root`, refusing anything that escapes it.
fn confine(root: &Path, relative: &str) -> Result<PathBuf, ApiError> {
    let candidate = root.join(relative);
    let resolved = candidate
        .canonicalize()
        .map_err(|_| not_found(format!("`{relative}` does not exist")))?;
    if !resolved.starts_with(root) {
        return Err(ApiError(
            StatusCode::FORBIDDEN,
            "the path is outside the results root".to_string(),
        ));
    }
    Ok(resolved)
}

/// The directory holding `programs.sqlite` for a run: the run itself for an
/// evolution run, its `investigation/` for a pipeline run.
fn search_dir(run_dir: &Path) -> Option<PathBuf> {
    [run_dir.join("investigation"), run_dir.to_path_buf()]
        .into_iter()
        .find(|dir| dir.join("programs.sqlite").is_file())
}

fn discover(root: &Path) -> Vec<Value> {
    let mut runs = Vec::new();
    let mut pending = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = pending.pop() {
        if let Some(search) = search_dir(&dir) {
            let relative = dir
                .strip_prefix(root)
                .unwrap_or(&dir)
                .to_string_lossy()
                .replace('\\', "/");
            let summary: Option<Value> = fsutil::read_json_opt(&search.join("run_summary.json"));
            let status: Option<Value> = fsutil::read_json_opt(&dir.join("pipeline_status.json"));
            let modified = std::fs::metadata(search.join("programs.sqlite"))
                .and_then(|meta| meta.modified())
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|age| age.as_secs());
            runs.push(json!({
                "id": if relative.is_empty() { ".".to_string() } else { relative },
                "kind": if search == dir { "evolve" } else { "pipeline" },
                "modified": modified,
                "status": status.as_ref().and_then(|s| s["status"].as_str()).or(summary.as_ref().and_then(|s| s["stop_reason"].as_str())),
                "best_score": summary.as_ref().and_then(|s| s["best"]["combined_score"].as_f64()),
                "has_paper": dir.join("paper/latest_manifest.json").is_file(),
            }));
            continue;
        }
        if depth >= 4 {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let hidden = entry.file_name().to_string_lossy().starts_with('.');
            if path.is_dir() && !hidden {
                pending.push((path, depth + 1));
            }
        }
    }
    runs.sort_by(|a, b| b["modified"].as_u64().cmp(&a["modified"].as_u64()));
    runs
}

#[derive(Deserialize)]
struct RunQuery {
    run: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

fn open_db(state: &AppState, run: &str) -> Result<(PathBuf, ProgramDb), ApiError> {
    let run_dir = confine(&state.root, run)?;
    let search = search_dir(&run_dir)
        .ok_or_else(|| not_found(format!("`{run}` holds no program database")))?;
    let db = ProgramDb::open_read_only(&search.join("programs.sqlite"))?;
    Ok((run_dir, db))
}

async fn list_runs(State(state): State<Shared>) -> Json<Value> {
    Json(json!({"root": state.root.display().to_string(), "runs": discover(&state.root)}))
}

async fn programs(
    State(state): State<Shared>,
    Query(query): Query<RunQuery>,
) -> Result<Json<Value>, ApiError> {
    let (_, db) = open_db(&state, &query.run)?;
    let rows: Vec<Value> = db
        .all()?
        .iter()
        .map(|program| {
            let cost = ["api_costs", "novelty_cost"].iter().filter_map(|key| program.meta_f64(key)).sum::<f64>();
            json!({
                "id": program.id,
                "generation": program.generation,
                "combined_score": program.combined_score,
                "correct": program.correct,
                "parent_id": program.parent_id,
                "island": program.island_idx,
                "in_archive": program.in_archive,
                "children": program.children_count,
                "name": program.meta_str("patch_name"),
                "patch_type": program.meta_str("patch_type"),
                "arm": program.meta_str("arm"),
                "copy": program.metadata.contains_key("_is_island_copy") || program.metadata.contains_key("_spawned_island"),
                "cost_usd": cost,
                "rubrics_pass": program.metadata.get("rubric_status").and_then(|status| status["overall_pass"].as_bool()),
            })
        })
        .collect();
    Ok(Json(json!({"programs": rows})))
}

async fn program(
    State(state): State<Shared>,
    Query(query): Query<RunQuery>,
) -> Result<Json<Value>, ApiError> {
    let (_, db) = open_db(&state, &query.run)?;
    let id = query
        .id
        .ok_or_else(|| ApiError(StatusCode::BAD_REQUEST, "missing `id`".to_string()))?;
    let mut found = db
        .get(&id)?
        .ok_or_else(|| not_found(format!("no program `{id}`")))?;
    found.embedding = Vec::new();
    let lineage: Vec<Value> = db
        .ancestry(&id, 200)?
        .iter()
        .map(|ancestor| {
            json!({
                "id": ancestor.id,
                "generation": ancestor.generation,
                "combined_score": ancestor.combined_score,
                "name": ancestor.meta_str("patch_name"),
            })
        })
        .collect();
    Ok(Json(json!({"program": found, "lineage": lineage})))
}

async fn summary(
    State(state): State<Shared>,
    Query(query): Query<RunQuery>,
) -> Result<Json<Value>, ApiError> {
    let (run_dir, db) = open_db(&state, &query.run)?;
    let search = search_dir(&run_dir).unwrap_or_else(|| run_dir.clone());
    let read = |path: PathBuf| fsutil::read_json_opt::<Value>(&path);
    let best = db.best(None)?.map(|mut best| {
        best.embedding = Vec::new();
        best
    });
    Ok(Json(json!({
        "run": query.run,
        "pipeline_status": read(run_dir.join("pipeline_status.json")),
        "run_summary": read(search.join("run_summary.json")),
        "best": best,
        "islands": db.island_stats()?,
        "selection": read(run_dir.join(".downstream_selection.json")),
        "completion": read(run_dir.join(".investigation_completion.json")),
        "final_manifest": read(run_dir.join("final_evaluation/final_manifest.json")),
        "ablations": read(run_dir.join("ablations/pipeline_summary.json")),
        "paper_manifest": read(run_dir.join("paper/latest_manifest.json")),
        "cost": read(run_dir.join("metrics/cost_summary.json")),
        "conclusions": std::fs::read_to_string(search.join("conclusions.md")).ok(),
        "meta": read(search.join("meta_memory.json")).map(|meta| json!({
            "recommendations": meta["meta_recommendations"],
            "scratchpad": meta["meta_scratch_pad"],
        })),
    })))
}

async fn events(
    State(state): State<Shared>,
    Query(query): Query<RunQuery>,
) -> Result<Json<Value>, ApiError> {
    let run_dir = confine(&state.root, &query.run)?;
    let rows: Vec<Value> = fsutil::read_jsonl(&run_dir.join("events.jsonl"));
    let limit = query.limit.unwrap_or(400).min(5000);
    let start = rows.len().saturating_sub(limit);
    Ok(Json(json!({"total": rows.len(), "events": rows[start..]})))
}

async fn file(
    State(state): State<Shared>,
    Query(query): Query<RunQuery>,
) -> Result<Response, ApiError> {
    let run_dir = confine(&state.root, &query.run)?;
    let relative = query
        .path
        .ok_or_else(|| ApiError(StatusCode::BAD_REQUEST, "missing `path`".to_string()))?;
    let path = confine(&run_dir, &relative)?;
    if !path.starts_with(&run_dir) {
        return Err(ApiError(
            StatusCode::FORBIDDEN,
            "the path is outside the run".to_string(),
        ));
    }
    let meta =
        std::fs::metadata(&path).map_err(|_| not_found(format!("`{relative}` does not exist")))?;
    if !meta.is_file() {
        return Err(not_found(format!("`{relative}` is not a file")));
    }
    if path.extension().is_some_and(|ext| ext == "pdf") {
        let bytes = std::fs::read(&path)
            .map_err(|error| ApiError(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
        return Ok(([(header::CONTENT_TYPE, "application/pdf")], bytes).into_response());
    }
    if meta.len() > MAX_PREVIEW_BYTES {
        return Err(ApiError(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("`{relative}` is larger than 1 MiB"),
        ));
    }
    let bytes = std::fs::read(&path)
        .map_err(|error| ApiError(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    Ok((
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        String::from_utf8_lossy(&bytes).into_owned(),
    )
        .into_response())
}

pub fn router(results_root: &Path) -> Result<Router> {
    let root = results_root
        .canonicalize()
        .with_context(|| format!("results root {} does not exist", results_root.display()))?;
    Ok(Router::new()
        .route("/", get(|| async { Html(PAGE) }))
        .route(
            "/api/healthz",
            get(|| async { Json(json!({"status": "ok"})) }),
        )
        .route("/api/runs", get(list_runs))
        .route("/api/run/summary", get(summary))
        .route("/api/run/programs", get(programs))
        .route("/api/run/program", get(program))
        .route("/api/run/events", get(events))
        .route("/api/run/file", get(file))
        .with_state(Arc::new(AppState { root })))
}

pub async fn serve(results_root: &Path, host: &str, port: u16) -> Result<()> {
    let app = router(results_root)?;
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .with_context(|| format!("binding {host}:{port}"))?;
    eprintln!(
        "genesis inspect: http://{}/  (results root {})",
        listener.local_addr()?,
        results_root.display()
    );
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::Program;
    use genesis_evolve::DatabaseConfig;

    async fn spawn(root: &Path) -> String {
        let app = router(root).unwrap();
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{address}")
    }

    fn make_run(root: &Path) -> (String, String) {
        let run = root.join("circle/20261125_120000");
        let investigation = run.join("investigation");
        let db = ProgramDb::open(
            Some(&investigation.join("programs.sqlite")),
            DatabaseConfig {
                num_islands: 1,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let mut seed = Program::new("x = 1  # <script>alert(1)</script>\n", "python");
        seed.correct = true;
        seed.combined_score = 1.0;
        db.add(&mut seed).unwrap();
        let mut child = Program::new("x = 2\n", "python");
        child.correct = true;
        child.combined_score = 2.0;
        child.generation = 1;
        child.parent_id = Some(seed.id.clone());
        child.embedding = vec![0.5; 16];
        child
            .metadata
            .insert("patch_name".into(), json!("double_x"));
        db.add(&mut child).unwrap();
        db.checkpoint().unwrap();
        fsutil::append_jsonl(&run.join("events.jsonl"), &json!({"event": "run_start"})).unwrap();
        fsutil::write_json(
            &run.join("pipeline_status.json"),
            &json!({"status": "completed"}),
        )
        .unwrap();
        fsutil::write_text(&run.join("paper/paper.md"), "# A paper\n").unwrap();
        fsutil::write_text(&root.join("outside.txt"), "not part of any run").unwrap();
        ("circle/20261125_120000".to_string(), child.id)
    }

    #[tokio::test]
    async fn the_api_serves_runs_programs_lineage_and_files() {
        let dir = tempfile::tempdir().unwrap();
        let (run, child_id) = make_run(dir.path());
        let base = spawn(dir.path()).await;
        let get = |path: String| async move { reqwest::get(path).await.unwrap() };

        let page = get(format!("{base}/")).await.text().await.unwrap();
        assert!(page.contains("<title>Genesis runs</title>"));

        let runs: Value = get(format!("{base}/api/runs")).await.json().await.unwrap();
        assert_eq!(runs["runs"][0]["id"], run);
        assert_eq!(
            (
                runs["runs"][0]["kind"].as_str(),
                runs["runs"][0]["status"].as_str()
            ),
            (Some("pipeline"), Some("completed"))
        );

        let programs: Value = get(format!("{base}/api/run/programs?run={run}"))
            .await
            .json()
            .await
            .unwrap();
        let rows = programs["programs"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert!(
            rows.iter().all(|row| row.get("code").is_none()),
            "the table carries no code"
        );

        let detail: Value = get(format!("{base}/api/run/program?run={run}&id={child_id}"))
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(detail["program"]["code"], "x = 2\n");
        assert_eq!(detail["program"]["embedding"].as_array().unwrap().len(), 0);
        assert_eq!(detail["lineage"].as_array().unwrap().len(), 1);

        let summary: Value = get(format!("{base}/api/run/summary?run={run}"))
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(summary["best"]["combined_score"], 2.0);
        assert_eq!(summary["pipeline_status"]["status"], "completed");

        let events: Value = get(format!("{base}/api/run/events?run={run}"))
            .await
            .json()
            .await
            .unwrap();
        assert_eq!(events["total"], 1);
        let paper = get(format!("{base}/api/run/file?run={run}&path=paper/paper.md"))
            .await
            .text()
            .await
            .unwrap();
        assert_eq!(paper, "# A paper\n");
    }

    /// Property (ephyra): whatever path a request names, `confine` either
    /// refuses it or returns a path inside the root. Source: the module
    /// docs, "Every path is confined to the results root". The tree holds a
    /// symlink that points outside, and the generator favours `..`.
    #[test]
    fn confine_never_returns_a_path_outside_the_root() {
        use ephyra::gens;
        use ephyra::prelude::*;

        let dir = tempfile::tempdir().unwrap();
        let outside = dir.path().join("outside");
        std::fs::create_dir_all(outside.join("deep")).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();
        let root = dir.path().join("root");
        std::fs::create_dir_all(root.join("run/paper")).unwrap();
        std::fs::write(root.join("run/paper/paper.md"), "paper").unwrap();
        std::os::unix::fs::symlink(&outside, root.join("run/link")).unwrap();
        let root = root.canonicalize().unwrap();

        let component = gens::frequency(vec![
            (
                4,
                gens::pick_from(vec![
                    "..",
                    ".",
                    "run",
                    "paper",
                    "paper.md",
                    "link",
                    "secret.txt",
                    "deep",
                    "outside",
                    "",
                ])
                .boxed(),
            ),
            (
                1,
                gens::string_len(gens::char(), 0..=6)
                    .map(|text| Box::leak(text.into_boxed_str()) as &str)
                    .boxed(),
            ),
        ]);
        let path =
            (gens::bool_with(0.15), gens::vec(component).len(0..=7)).map(|(absolute, parts)| {
                format!("{}{}", if absolute { "/" } else { "" }, parts.join("/"))
            });
        let checked_root = root.clone();
        let report = check(
            &Config::default().tests(20_000),
            path,
            move |relative: String| match confine(&checked_root, &relative) {
                Ok(resolved) => {
                    ephyra::cover(true, "accepted");
                    assert!(
                        resolved.starts_with(&checked_root),
                        "`{relative}` resolved to {}",
                        resolved.display()
                    );
                    assert!(
                        !resolved.to_string_lossy().contains("secret")
                            && !resolved.starts_with(&outside)
                    );
                }
                Err(ApiError(status, _)) => {
                    assert!(matches!(
                        status,
                        StatusCode::FORBIDDEN | StatusCode::NOT_FOUND
                    ));
                }
            },
        );
        report.assert_passed();
    }

    #[tokio::test]
    async fn paths_cannot_leave_the_results_root_or_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("results");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(dir.path().join("secret.txt"), "secret").unwrap();
        let (run, _) = make_run(&root);
        let base = spawn(&root).await;
        for url in [
            format!("{base}/api/run/file?run={run}&path=../../../secret.txt"),
            format!("{base}/api/run/file?run={run}&path=../../outside.txt"),
            format!("{base}/api/run/file?run=..&path=secret.txt"),
            format!("{base}/api/run/programs?run=../"),
        ] {
            let response = reqwest::get(&url).await.unwrap();
            assert!(
                matches!(response.status().as_u16(), 403 | 404),
                "{url} answered {}",
                response.status()
            );
            assert!(!response.text().await.unwrap().contains("secret"));
        }
        let missing = reqwest::get(format!("{base}/api/run/program?run={run}&id=nope"))
            .await
            .unwrap();
        assert_eq!(missing.status().as_u16(), 404);
    }
}
