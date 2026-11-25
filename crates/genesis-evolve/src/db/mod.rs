//! The run database: every evaluated program, the archive of the best of
//! them, and the island structure that keeps the search diverse.
//!
//! The SQLite schema is ShinkaEvolve's, column for column, so a database
//! written here opens in the existing Genesis WebUI and analysis scripts.
//! All randomness comes from one seeded generator, so sampling is
//! reproducible given `DatabaseConfig::seed`.

mod islands;
pub mod prompts;
mod selection;

pub use selection::{power_law_probabilities, weighted_probabilities, Sample};

use crate::config::DatabaseConfig;
use anyhow::{Context, Result};
use genesis_core::{JsonMap, Program};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rusqlite::{params, Connection, OptionalExtension, Row, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Mutex;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS programs (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    language TEXT NOT NULL,
    parent_id TEXT,
    archive_inspiration_ids TEXT,
    top_k_inspiration_ids TEXT,
    generation INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    code_diff TEXT,
    combined_score REAL,
    public_metrics TEXT,
    private_metrics TEXT,
    text_feedback TEXT,
    complexity REAL,
    embedding TEXT,
    embedding_pca_2d TEXT,
    embedding_pca_3d TEXT,
    embedding_cluster_id INTEGER,
    correct BOOLEAN DEFAULT 0,
    children_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,
    migration_history TEXT,
    island_idx INTEGER,
    system_prompt_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_programs_generation ON programs(generation);
CREATE INDEX IF NOT EXISTS idx_programs_timestamp ON programs(timestamp);
CREATE INDEX IF NOT EXISTS idx_programs_complexity ON programs(complexity);
CREATE INDEX IF NOT EXISTS idx_programs_parent_id ON programs(parent_id);
CREATE INDEX IF NOT EXISTS idx_programs_children_count ON programs(children_count);
CREATE INDEX IF NOT EXISTS idx_programs_island_idx ON programs(island_idx);
CREATE INDEX IF NOT EXISTS idx_programs_system_prompt_id ON programs(system_prompt_id);
CREATE TABLE IF NOT EXISTS archive (
    program_id TEXT PRIMARY KEY,
    FOREIGN KEY (program_id) REFERENCES programs(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS metadata_store (
    key TEXT PRIMARY KEY, value TEXT
);
CREATE TABLE IF NOT EXISTS generation_event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL,
    source_job_id TEXT,
    details TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_generation_event_log_generation ON generation_event_log(generation);
CREATE TABLE IF NOT EXISTS attempt_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    details TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempt_log_generation ON attempt_log(generation);
"#;

const SELECT: &str = "SELECT p.id, p.code, p.language, p.parent_id, p.archive_inspiration_ids, \
    p.top_k_inspiration_ids, p.generation, p.timestamp, p.code_diff, p.combined_score, \
    p.public_metrics, p.private_metrics, p.text_feedback, p.complexity, p.embedding, p.correct, \
    p.children_count, p.metadata, p.migration_history, p.island_idx, p.system_prompt_id, \
    (a.program_id IS NOT NULL) AS in_archive \
    FROM programs p LEFT JOIN archive a ON a.program_id = p.id";

pub(crate) const META_BEST_ID: &str = "best_program_id";
pub(crate) const META_BEST_SCORE: &str = "best_score_ever";
pub(crate) const META_BEST_GENERATION: &str = "best_score_generation";
pub(crate) const META_LAST_ITERATION: &str = "last_iteration";
pub(crate) const META_LAST_MIGRATION: &str = "last_migration_generation";
pub(crate) const META_BEAM_PARENT: &str = "beam_search_parent_id";

/// One row of the attempt journal: a failed proposal, a charge that never
/// reached a program row, or any other per-generation event worth keeping.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttemptRecord {
    pub generation: u32,
    pub stage: String,
    pub status: String,
    pub details: Value,
    pub created_at: f64,
}

/// Aggregates of one island, for telemetry and island sampling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IslandStats {
    pub island_idx: usize,
    pub programs: usize,
    pub correct_programs: usize,
    /// Best score among correct programs.
    pub best_score: Option<f64>,
}

pub(crate) struct Inner {
    pub(crate) conn: Connection,
    pub(crate) config: DatabaseConfig,
    pub(crate) rng: ChaCha8Rng,
}

/// Thread-safe handle on the run database.
pub struct ProgramDb {
    inner: Mutex<Inner>,
}

fn json_text<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn parse_json<T: serde::de::DeserializeOwned + Default>(text: Option<String>) -> T {
    text.and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

fn row_to_program(row: &Row<'_>) -> rusqlite::Result<Program> {
    Ok(Program {
        id: row.get(0)?,
        code: row.get(1)?,
        language: row.get(2)?,
        parent_id: row.get(3)?,
        archive_inspiration_ids: parse_json(row.get(4)?),
        top_k_inspiration_ids: parse_json(row.get(5)?),
        generation: row.get::<_, i64>(6)?.max(0) as u32,
        timestamp: row.get(7)?,
        code_diff: row.get(8)?,
        combined_score: row.get::<_, Option<f64>>(9)?.unwrap_or(0.0),
        public_metrics: parse_json(row.get(10)?),
        private_metrics: parse_json(row.get(11)?),
        text_feedback: row.get::<_, Option<String>>(12)?.unwrap_or_default(),
        complexity: row.get::<_, Option<f64>>(13)?.unwrap_or(0.0),
        embedding: parse_json(row.get(14)?),
        correct: row.get::<_, Option<bool>>(15)?.unwrap_or(false),
        children_count: row.get::<_, i64>(16)?.max(0) as u32,
        metadata: parse_json(row.get(17)?),
        migration_history: parse_json(row.get(18)?),
        island_idx: row
            .get::<_, Option<i64>>(19)?
            .map(|idx| idx.max(0) as usize),
        system_prompt_id: row.get(20)?,
        in_archive: row.get(21)?,
    })
}

/// Programs matching `tail`, a `WHERE … ORDER BY … LIMIT …` suffix over the
/// aliases `p` (programs) and `a` (archive).
pub(crate) fn query(
    conn: &Connection,
    tail: &str,
    args: &[&dyn rusqlite::ToSql],
) -> Result<Vec<Program>> {
    let sql = format!("{SELECT} {tail}");
    let mut statement = conn
        .prepare(&sql)
        .with_context(|| format!("preparing `{tail}`"))?;
    let rows = statement.query_map(args, row_to_program)?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

pub(crate) fn get(conn: &Connection, id: &str) -> Result<Option<Program>> {
    Ok(query(conn, "WHERE p.id = ?1", &[&id])?.into_iter().next())
}

pub(crate) fn meta_get(conn: &Connection, key: &str) -> Result<Option<String>> {
    let value: Option<Option<String>> = conn
        .query_row(
            "SELECT value FROM metadata_store WHERE key = ?1",
            [key],
            |row| row.get(0),
        )
        .optional()?;
    Ok(value.flatten().filter(|text| text != "None"))
}

pub(crate) fn meta_set(conn: &Connection, key: &str, value: &str) -> Result<()> {
    conn.execute(
        "INSERT INTO metadata_store (key, value) VALUES (?1, ?2) \
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

pub(crate) fn insert_row(conn: &Connection, program: &Program) -> Result<()> {
    conn.execute(
        "INSERT INTO programs (id, code, language, parent_id, archive_inspiration_ids, \
         top_k_inspiration_ids, generation, timestamp, code_diff, combined_score, public_metrics, \
         private_metrics, text_feedback, complexity, embedding, correct, children_count, metadata, \
         migration_history, island_idx, system_prompt_id) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
        params![
            program.id,
            program.code,
            program.language,
            program.parent_id,
            json_text(&program.archive_inspiration_ids),
            json_text(&program.top_k_inspiration_ids),
            program.generation,
            program.timestamp,
            program.code_diff,
            program.combined_score,
            json_text(&program.public_metrics),
            json_text(&program.private_metrics),
            program.text_feedback,
            program.complexity,
            json_text(&program.embedding),
            program.correct,
            program.children_count,
            json_text(&program.metadata),
            json_text(&program.migration_history),
            program.island_idx.map(|idx| idx as i64),
            program.system_prompt_id,
        ],
    )
    .with_context(|| format!("inserting program {}", program.id))?;
    Ok(())
}

fn mean_public_metric(program: &Program) -> f64 {
    let values: Vec<f64> = program
        .public_metrics
        .values()
        .filter_map(Value::as_f64)
        .collect();
    if values.is_empty() {
        f64::NEG_INFINITY
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Whether `a` beats `b` on the evidence: correctness first, then score,
/// then the mean public metric. `None` when they are level.
fn outranks(a: &Program, b: &Program) -> Option<bool> {
    if a.correct != b.correct {
        return Some(a.correct);
    }
    if a.combined_score != b.combined_score {
        return Some(a.combined_score > b.combined_score);
    }
    let (mean_a, mean_b) = (mean_public_metric(a), mean_public_metric(b));
    (mean_a != mean_b).then_some(mean_a > mean_b)
}

/// Whether `a` should be kept over `b` in the archive: by evidence, and
/// between level programs the newer one, which keeps the archive fresh.
pub fn is_better(a: &Program, b: &Program) -> bool {
    outranks(a, b).unwrap_or(a.timestamp > b.timestamp)
}

/// Whether `a` displaces `b` as the best program. A program that only
/// matches the best does not: the first to reach a score keeps the title.
pub fn is_strictly_better(a: &Program, b: &Program) -> bool {
    outranks(a, b).unwrap_or(false)
}

pub(crate) fn is_copy(program: &Program) -> bool {
    ["_is_island_copy", "_spawned_island"].iter().any(|flag| {
        program
            .metadata
            .get(*flag)
            .and_then(Value::as_bool)
            .unwrap_or(false)
    })
}

fn criterion_value(program: &Program, criterion: &str) -> f64 {
    let analysis = |key: &str| {
        program
            .metadata
            .get("code_analysis_metrics")
            .and_then(|metrics| metrics.get(key))
            .and_then(Value::as_f64)
    };
    match criterion {
        "combined_score" => program.combined_score,
        "loc" => analysis("lines_of_code").unwrap_or(0.0),
        "lloc" => analysis("logical_lines_of_code").unwrap_or(0.0),
        "complexity" => analysis("complexity_score").unwrap_or(program.complexity),
        "maintainability" => analysis("maintainability_index").unwrap_or(0.0),
        "nesting" => analysis("max_nesting_depth").unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Rank-normalised multi-criteria score of `program` within `pool`.
fn ranked_score(config: &DatabaseConfig, program: &Program, pool: &[&Program]) -> f64 {
    let mut score = 0.0;
    for (criterion, weight) in &config.archive_criteria {
        let own = criterion_value(program, criterion);
        let beaten = pool
            .iter()
            .filter(|other| {
                let value = criterion_value(other, criterion);
                if *weight >= 0.0 {
                    value < own
                } else {
                    value > own
                }
            })
            .count();
        score += weight.abs() * beaten as f64 / pool.len().max(1) as f64;
    }
    score
}

fn single_score_criterion(config: &DatabaseConfig) -> bool {
    config.archive_criteria.len() <= 1
        && config
            .archive_criteria
            .get("combined_score")
            .is_none_or(|weight| *weight > 0.0)
        && config
            .archive_criteria
            .keys()
            .all(|key| key == "combined_score")
}

pub(crate) fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }
    let (mut dot, mut norm_a, mut norm_b) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        norm_a += f64::from(*x).powi(2);
        norm_b += f64::from(*y).powi(2);
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        return None;
    }
    Some(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

/// Admit a correct program to the archive, evicting one member when full.
fn update_archive(
    tx: &Transaction<'_>,
    config: &DatabaseConfig,
    program: &mut Program,
) -> Result<()> {
    if !program.correct || config.archive_size == 0 || program.in_archive {
        return Ok(());
    }
    let members = query(
        tx,
        "WHERE a.program_id IS NOT NULL ORDER BY p.timestamp, p.rowid",
        &[],
    )?;
    if members.len() < config.archive_size {
        tx.execute(
            "INSERT OR IGNORE INTO archive (program_id) VALUES (?1)",
            [&program.id],
        )?;
        program.in_archive = true;
        return Ok(());
    }

    let victim = if config.archive_selection_strategy == "crowding" && !program.embedding.is_empty()
    {
        let most_similar = members
            .iter()
            .filter_map(|member| {
                cosine(&program.embedding, &member.embedding).map(|sim| (sim, member))
            })
            .max_by(|x, y| x.0.total_cmp(&y.0))
            .map(|(_, member)| member);
        match most_similar {
            Some(member) => is_better(program, member).then_some(member),
            None => fitness_victim(config, program, &members),
        }
    } else {
        fitness_victim(config, program, &members)
    };
    if let Some(victim) = victim {
        tx.execute("DELETE FROM archive WHERE program_id = ?1", [&victim.id])?;
        tx.execute(
            "INSERT OR IGNORE INTO archive (program_id) VALUES (?1)",
            [&program.id],
        )?;
        program.in_archive = true;
    }
    Ok(())
}

/// The archive member `program` displaces under fitness replacement, if any.
fn fitness_victim<'a>(
    config: &DatabaseConfig,
    program: &Program,
    members: &'a [Program],
) -> Option<&'a Program> {
    if single_score_criterion(config) {
        let mut worst = members.first()?;
        for member in &members[1..] {
            if is_better(worst, member) {
                worst = member;
            }
        }
        return is_better(program, worst).then_some(worst);
    }
    let mut pool: Vec<&Program> = members.iter().collect();
    pool.push(program);
    let worst = members
        .iter()
        .min_by(|x, y| ranked_score(config, x, &pool).total_cmp(&ranked_score(config, y, &pool)))?;
    (ranked_score(config, program, &pool) > ranked_score(config, worst, &pool)).then_some(worst)
}

fn update_best(tx: &Transaction<'_>, program: &Program) -> Result<()> {
    if !program.correct {
        return Ok(());
    }
    let current = match meta_get(tx, META_BEST_ID)? {
        Some(id) => get(tx, &id)?.filter(|best| best.correct),
        None => None,
    };
    if current
        .as_ref()
        .is_none_or(|best| is_strictly_better(program, best))
    {
        meta_set(tx, META_BEST_ID, &program.id)?;
        let best_ever: Option<f64> =
            meta_get(tx, META_BEST_SCORE)?.and_then(|text| text.parse().ok());
        if best_ever.is_none_or(|score| program.combined_score > score) {
            let previous: u32 = meta_get(tx, META_BEST_GENERATION)?
                .and_then(|text| text.parse().ok())
                .unwrap_or(0);
            meta_set(tx, META_BEST_SCORE, &program.combined_score.to_string())?;
            meta_set(
                tx,
                META_BEST_GENERATION,
                &previous.max(program.generation).to_string(),
            )?;
        }
    }
    Ok(())
}

impl ProgramDb {
    /// Open or create the database. `None` keeps it in memory.
    pub fn open(path: Option<&Path>, config: DatabaseConfig) -> Result<Self> {
        config.validate()?;
        let conn = match path {
            Some(path) => {
                if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
                    std::fs::create_dir_all(dir)?;
                }
                Connection::open(path).with_context(|| format!("opening {}", path.display()))?
            }
            None => Connection::open_in_memory()?,
        };
        conn.busy_timeout(std::time::Duration::from_secs(60))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };
        Ok(Self {
            inner: Mutex::new(Inner { conn, config, rng }),
        })
    }

    /// Open an existing database for reading only, as the inspector does
    /// while a run is still writing to it.
    pub fn open_read_only(path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening {} read-only", path.display()))?;
        conn.busy_timeout(std::time::Duration::from_secs(10))?;
        Ok(Self {
            inner: Mutex::new(Inner {
                conn,
                config: DatabaseConfig::default(),
                rng: ChaCha8Rng::seed_from_u64(0),
            }),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn config(&self) -> DatabaseConfig {
        self.lock().config.clone()
    }

    /// Insert an evaluated program and bring the archive, the best-program
    /// record and the islands up to date, all in one transaction.
    ///
    /// On return `program.island_idx` and `program.in_archive` reflect where
    /// it ended up.
    pub fn add(&self, program: &mut Program) -> Result<()> {
        let mut guard = self.lock();
        let Inner { conn, config, rng } = &mut *guard;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let is_first: bool =
            tx.query_row("SELECT COUNT(*) = 0 FROM programs", [], |row| row.get(0))?;

        islands::assign_island(&tx, config, rng, program, is_first)?;
        if program.id.is_empty() {
            program.id = uuid::Uuid::new_v4().to_string();
        }
        if program.timestamp == 0.0 {
            program.timestamp = genesis_core::fsutil::now_ts();
        }
        program.in_archive = false;
        insert_row(&tx, program)?;
        if let Some(parent_id) = &program.parent_id {
            tx.execute(
                "UPDATE programs SET children_count = children_count + 1 WHERE id = ?1",
                [parent_id],
            )?;
        }
        let last: u32 = meta_get(&tx, META_LAST_ITERATION)?
            .and_then(|text| text.parse().ok())
            .unwrap_or(0);
        if program.generation >= last {
            meta_set(&tx, META_LAST_ITERATION, &program.generation.to_string())?;
        }

        update_archive(&tx, config, program)?;
        update_best(&tx, program)?;
        if is_first {
            islands::copy_seed_to_islands(&tx, config, program)?;
        }
        if program.correct {
            islands::seed_uninitialized_islands(&tx, config, program)?;
        }
        if config.enable_dynamic_islands {
            islands::spawn_island_if_stagnant(&tx, config, rng, program.generation)?;
        }
        if islands::migration_due(&tx, config, program.generation)? {
            islands::perform_migration(&tx, config, rng, program.generation)?;
            meta_set(&tx, META_LAST_MIGRATION, &program.generation.to_string())?;
            // A migrant may be the program just inserted.
            if let Some(stored) = get(&tx, &program.id)? {
                program.island_idx = stored.island_idx;
                program.migration_history = stored.migration_history;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<Option<Program>> {
        get(&self.lock().conn, id)
    }

    /// Every program, oldest generation first.
    pub fn all(&self) -> Result<Vec<Program>> {
        query(
            &self.lock().conn,
            "ORDER BY p.generation, p.timestamp, p.rowid",
            &[],
        )
    }

    pub fn count(&self) -> Result<usize> {
        let count: i64 =
            self.lock()
                .conn
                .query_row("SELECT COUNT(*) FROM programs", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// The best correct program overall, or on one island.
    pub fn best(&self, island: Option<usize>) -> Result<Option<Program>> {
        let guard = self.lock();
        best(&guard.conn, island)
    }

    /// The `n` highest-scoring programs, best first.
    pub fn top(&self, n: usize, correct_only: bool) -> Result<Vec<Program>> {
        let filter = if correct_only {
            "AND p.correct = 1"
        } else {
            ""
        };
        let tail = format!(
            "WHERE p.combined_score IS NOT NULL {filter} ORDER BY p.combined_score DESC, p.timestamp, p.rowid LIMIT ?1"
        );
        query(&self.lock().conn, &tail, &[&(n as i64)])
    }

    /// Choose a parent and its inspirations for the next proposal.
    pub fn sample(&self) -> Result<Sample> {
        let mut guard = self.lock();
        let Inner { conn, config, rng } = &mut *guard;
        selection::sample(conn, config, rng)
    }

    /// Up to `max` ancestors of `id`, oldest first.
    pub fn ancestry(&self, id: &str, max: usize) -> Result<Vec<Program>> {
        let guard = self.lock();
        let mut chain = Vec::new();
        let mut seen = BTreeSet::new();
        let mut cursor = get(&guard.conn, id)?.and_then(|program| program.parent_id);
        while let Some(parent_id) = cursor {
            if chain.len() >= max || !seen.insert(parent_id.clone()) {
                break;
            }
            let Some(parent) = get(&guard.conn, &parent_id)? else {
                break;
            };
            cursor = parent.parent_id.clone();
            chain.push(parent);
        }
        chain.reverse();
        Ok(chain)
    }

    /// The program on `island` whose embedding is closest to `embedding`,
    /// with its cosine similarity.
    pub fn most_similar(&self, embedding: &[f32], island: usize) -> Result<Option<(f64, Program)>> {
        let guard = self.lock();
        let candidates = query(
            &guard.conn,
            "WHERE p.island_idx = ?1 ORDER BY p.timestamp, p.rowid",
            &[&(island as i64)],
        )?;
        Ok(candidates
            .into_iter()
            .filter_map(|program| cosine(embedding, &program.embedding).map(|sim| (sim, program)))
            .max_by(|x, y| x.0.total_cmp(&y.0)))
    }

    /// Generations that have a program row of their own (island copies do
    /// not count).
    pub fn generations(&self) -> Result<BTreeSet<u32>> {
        Ok(self
            .all()?
            .iter()
            .filter(|program| !is_copy(program))
            .map(|program| program.generation)
            .collect())
    }

    pub fn archive_ids(&self) -> Result<Vec<String>> {
        let guard = self.lock();
        let mut statement = guard
            .conn
            .prepare("SELECT program_id FROM archive ORDER BY program_id")?;
        let ids = statement
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<String>>>()?;
        Ok(ids)
    }

    pub fn island_stats(&self) -> Result<Vec<IslandStats>> {
        let guard = self.lock();
        let mut statement = guard.conn.prepare(
            "SELECT island_idx, COUNT(*), SUM(correct), MAX(CASE WHEN correct = 1 THEN combined_score END) \
             FROM programs WHERE island_idx IS NOT NULL GROUP BY island_idx ORDER BY island_idx",
        )?;
        let stats = statement
            .query_map([], |row| {
                Ok(IslandStats {
                    island_idx: row.get::<_, i64>(0)? as usize,
                    programs: row.get::<_, i64>(1)? as usize,
                    correct_programs: row.get::<_, Option<i64>>(2)?.unwrap_or(0) as usize,
                    best_score: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(stats)
    }

    pub fn meta_get(&self, key: &str) -> Result<Option<String>> {
        meta_get(&self.lock().conn, key)
    }

    pub fn meta_set(&self, key: &str, value: &str) -> Result<()> {
        meta_set(&self.lock().conn, key, value)
    }

    /// Replace a stored program's metadata, for provenance added after the
    /// insert (rubric status, meta cost).
    pub fn update_metadata(&self, id: &str, metadata: &JsonMap) -> Result<()> {
        self.lock().conn.execute(
            "UPDATE programs SET metadata = ?1 WHERE id = ?2",
            params![json_text(metadata), id],
        )?;
        Ok(())
    }

    pub fn log_attempt(
        &self,
        generation: u32,
        stage: &str,
        status: &str,
        details: &Value,
    ) -> Result<()> {
        self.lock().conn.execute(
            "INSERT INTO attempt_log (generation, stage, status, details, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![generation, stage, status, details.to_string(), genesis_core::fsutil::now_ts()],
        )?;
        Ok(())
    }

    pub fn attempts(&self) -> Result<Vec<AttemptRecord>> {
        let guard = self.lock();
        let mut statement = guard.conn.prepare(
            "SELECT generation, stage, status, details, created_at FROM attempt_log ORDER BY id",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok(AttemptRecord {
                    generation: row.get::<_, i64>(0)?.max(0) as u32,
                    stage: row.get(1)?,
                    status: row.get(2)?,
                    details: row
                        .get::<_, Option<String>>(3)?
                        .and_then(|text| serde_json::from_str(&text).ok())
                        .unwrap_or(Value::Null),
                    created_at: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Run a migration now, whatever the interval says. Returns how many
    /// programs moved.
    pub fn migrate(&self, generation: u32) -> Result<usize> {
        let mut guard = self.lock();
        let Inner { conn, config, rng } = &mut *guard;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let moved = islands::perform_migration(&tx, config, rng, generation)?;
        tx.commit()?;
        Ok(moved)
    }

    /// Flush the write-ahead log into the main file, so the database can be
    /// copied or read by another tool.
    pub fn checkpoint(&self) -> Result<()> {
        self.lock()
            .conn
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }
}

pub(crate) fn best(conn: &Connection, island: Option<usize>) -> Result<Option<Program>> {
    if island.is_none() {
        if let Some(id) = meta_get(conn, META_BEST_ID)? {
            if let Some(program) = get(conn, &id)?.filter(|program| program.correct) {
                return Ok(Some(program));
            }
        }
    }
    let candidates = match island {
        Some(island) => query(
            conn,
            "WHERE p.correct = 1 AND p.island_idx = ?1 ORDER BY p.rowid",
            &[&(island as i64)],
        )?,
        None => query(conn, "WHERE p.correct = 1 ORDER BY p.rowid", &[])?,
    };
    Ok(candidates.into_iter().reduce(|best, candidate| {
        if is_strictly_better(&candidate, &best) {
            candidate
        } else {
            best
        }
    }))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub fn program(
        score: f64,
        correct: bool,
        generation: u32,
        parent: Option<&Program>,
    ) -> Program {
        let mut program = Program::new(format!("score = {score}\n"), "python");
        program.combined_score = score;
        program.correct = correct;
        program.generation = generation;
        program.parent_id = parent.map(|p| p.id.clone());
        program
    }

    pub fn db(config: DatabaseConfig) -> ProgramDb {
        ProgramDb::open(
            None,
            DatabaseConfig {
                seed: Some(7),
                ..config
            },
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{db, program};
    use super::*;

    #[test]
    fn programs_round_trip_with_all_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run/programs.sqlite");
        let db = ProgramDb::open(
            Some(&path),
            DatabaseConfig {
                num_islands: 1,
                seed: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        let mut seed = program(1.5, true, 0, None);
        seed.public_metrics
            .insert("n".into(), serde_json::json!(26));
        seed.private_metrics
            .insert("secret".into(), serde_json::json!(0.25));
        seed.text_feedback = "fine".into();
        seed.embedding = vec![0.5, -0.5];
        seed.metadata
            .insert("patch_name".into(), serde_json::json!("init"));
        seed.code_diff = Some("diff".into());
        seed.system_prompt_id = Some("prompt-1".into());
        db.add(&mut seed).unwrap();
        assert_eq!(seed.island_idx, Some(0));
        assert!(seed.in_archive);

        let reader = ProgramDb::open_read_only(&path).unwrap();
        assert_eq!(reader.get(&seed.id).unwrap().unwrap(), seed);
        let mut refused = program(9.0, true, 1, None);
        assert!(
            reader.add(&mut refused).is_err(),
            "a read-only handle cannot write"
        );
        drop(reader);

        drop(db);
        let reopened = ProgramDb::open(
            Some(&path),
            DatabaseConfig {
                num_islands: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(reopened.get(&seed.id).unwrap().unwrap(), seed);
        assert_eq!(reopened.count().unwrap(), 1);
        assert_eq!(reopened.best(None).unwrap().unwrap().id, seed.id);
        assert!(reopened.get("missing").unwrap().is_none());
    }

    #[test]
    fn children_count_archive_and_best_follow_inserts() {
        let db = db(DatabaseConfig {
            num_islands: 1,
            archive_size: 2,
            ..Default::default()
        });
        let mut seed = program(1.0, true, 0, None);
        db.add(&mut seed).unwrap();
        let mut better = program(2.0, true, 1, Some(&seed));
        let mut wrong = program(9.0, false, 2, Some(&seed));
        let mut best = program(3.0, true, 3, Some(&better));
        for p in [&mut better, &mut wrong, &mut best] {
            db.add(p).unwrap();
        }
        assert_eq!(db.get(&seed.id).unwrap().unwrap().children_count, 2);
        assert_eq!(db.get(&better.id).unwrap().unwrap().children_count, 1);
        assert_eq!(db.best(None).unwrap().unwrap().id, best.id);
        // The archive holds the two best correct programs; the incorrect one never enters.
        let mut expected = vec![better.id.clone(), best.id.clone()];
        expected.sort();
        assert_eq!(db.archive_ids().unwrap(), expected);
        assert!(!db.get(&wrong.id).unwrap().unwrap().in_archive);
        assert!(!db.get(&seed.id).unwrap().unwrap().in_archive);

        let top: Vec<f64> = db
            .top(3, true)
            .unwrap()
            .iter()
            .map(|p| p.combined_score)
            .collect();
        assert_eq!(top, [3.0, 2.0, 1.0]);
        assert_eq!(db.top(1, false).unwrap()[0].id, wrong.id);
        let chain: Vec<String> = db
            .ancestry(&best.id, 5)
            .unwrap()
            .into_iter()
            .map(|p| p.id)
            .collect();
        assert_eq!(chain, [seed.id.clone(), better.id.clone()]);
        assert_eq!(db.ancestry(&best.id, 1).unwrap().len(), 1);
        assert_eq!(db.generations().unwrap(), BTreeSet::from([0, 1, 2, 3]));
    }

    /// Seen in a live run: a later program that only matched the best score
    /// took over as "best", so `best/` and the run summary named different
    /// programs. The first program to reach a score keeps the title.
    #[test]
    fn matching_the_best_score_does_not_displace_the_best() {
        let db = db(DatabaseConfig {
            num_islands: 1,
            ..Default::default()
        });
        let mut seed = program(1.0, true, 0, None);
        db.add(&mut seed).unwrap();
        let mut first = program(2.5, true, 1, Some(&seed));
        db.add(&mut first).unwrap();
        let mut equal = program(2.5, true, 2, Some(&first));
        db.add(&mut equal).unwrap();
        assert_eq!(db.best(None).unwrap().unwrap().id, first.id);
        assert_eq!(db.best(Some(0)).unwrap().unwrap().id, first.id);
        let mut higher = program(2.6, true, 3, Some(&first));
        db.add(&mut higher).unwrap();
        assert_eq!(db.best(None).unwrap().unwrap().id, higher.id);
        assert!(is_better(&equal, &first) && !is_strictly_better(&equal, &first));
    }

    #[test]
    fn ties_keep_the_newer_program_and_public_metrics_break_score_ties() {
        let mut older = program(1.0, true, 0, None);
        let mut newer = program(1.0, true, 1, None);
        older.timestamp = 10.0;
        newer.timestamp = 20.0;
        assert!(is_better(&newer, &older) && !is_better(&older, &newer));
        older
            .public_metrics
            .insert("speed".into(), serde_json::json!(5.0));
        newer
            .public_metrics
            .insert("speed".into(), serde_json::json!(1.0));
        assert!(is_better(&older, &newer));
        let wrong = program(100.0, false, 2, None);
        assert!(is_better(&older, &wrong));
    }

    #[test]
    fn crowding_replaces_the_most_similar_member_only_when_better() {
        let db = db(DatabaseConfig {
            num_islands: 1,
            archive_size: 2,
            archive_selection_strategy: "crowding".into(),
            ..Default::default()
        });
        let mut a = program(1.0, true, 0, None);
        a.embedding = vec![1.0, 0.0];
        let mut b = program(5.0, true, 1, None);
        b.embedding = vec![0.0, 1.0];
        db.add(&mut a).unwrap();
        db.add(&mut b).unwrap();
        // Closest to `b`, but worse than it: rejected even though it beats `a`.
        let mut near_b = program(2.0, true, 2, None);
        near_b.embedding = vec![0.1, 1.0];
        db.add(&mut near_b).unwrap();
        assert!(!near_b.in_archive);
        // Closest to `a` and better than it: replaces `a`.
        let mut near_a = program(2.0, true, 3, None);
        near_a.embedding = vec![1.0, 0.1];
        db.add(&mut near_a).unwrap();
        assert!(near_a.in_archive);
        assert!(!db.get(&a.id).unwrap().unwrap().in_archive);
    }

    #[test]
    fn multi_criteria_archive_trades_score_against_size() {
        let db = db(DatabaseConfig {
            num_islands: 1,
            archive_size: 2,
            archive_criteria: BTreeMap::from([
                ("combined_score".to_string(), 1.0),
                ("loc".to_string(), -1.0),
            ]),
            ..Default::default()
        });
        let with_loc = |score: f64, loc: f64, generation: u32| {
            let mut p = program(score, true, generation, None);
            p.metadata.insert(
                "code_analysis_metrics".into(),
                serde_json::json!({"lines_of_code": loc}),
            );
            p
        };
        let (mut small, mut large) = (with_loc(1.0, 10.0, 0), with_loc(1.2, 500.0, 1));
        db.add(&mut small).unwrap();
        db.add(&mut large).unwrap();
        // Better score and shorter than `large`: it wins on both criteria.
        let mut both = with_loc(1.3, 20.0, 2);
        db.add(&mut both).unwrap();
        assert!(both.in_archive);
        assert!(!db.get(&large.id).unwrap().unwrap().in_archive);
        assert!(db.get(&small.id).unwrap().unwrap().in_archive);
    }

    use std::collections::BTreeMap;

    #[test]
    fn most_similar_metadata_and_attempt_log() {
        let db = db(DatabaseConfig {
            num_islands: 1,
            ..Default::default()
        });
        let mut a = program(1.0, true, 0, None);
        a.embedding = vec![1.0, 0.0];
        let mut b = program(0.0, false, 1, Some(&a));
        b.embedding = vec![0.6, 0.8];
        db.add(&mut a).unwrap();
        db.add(&mut b).unwrap();
        let (similarity, nearest) = db.most_similar(&[0.6, 0.8], 0).unwrap().unwrap();
        assert_eq!(nearest.id, b.id);
        assert!((similarity - 1.0).abs() < 1e-9);
        assert!(db.most_similar(&[1.0, 0.0, 0.0], 0).unwrap().is_none());
        assert!(db.most_similar(&[1.0, 0.0], 3).unwrap().is_none());

        let mut metadata = b.metadata.clone();
        metadata.insert(
            "rubric_status".into(),
            serde_json::json!({"overall_pass": false}),
        );
        db.update_metadata(&b.id, &metadata).unwrap();
        assert_eq!(db.get(&b.id).unwrap().unwrap().metadata, metadata);

        db.meta_set("k", "v").unwrap();
        assert_eq!(db.meta_get("k").unwrap().as_deref(), Some("v"));
        assert_eq!(db.meta_get("absent").unwrap(), None);
        db.log_attempt(
            4,
            "proposal",
            "failed",
            &serde_json::json!({"failure_class": "patch_apply_failed"}),
        )
        .unwrap();
        let attempts = db.attempts().unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].details["failure_class"], "patch_apply_failed");
        let stats = db.island_stats().unwrap();
        assert_eq!(
            (
                stats[0].programs,
                stats[0].correct_programs,
                stats[0].best_score
            ),
            (2, 1, Some(1.0))
        );
    }
}
