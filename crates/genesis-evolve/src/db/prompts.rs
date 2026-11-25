//! The archive of task system prompts that co-evolves with the programs.
//!
//! A prompt's fitness is the mean percentile, among all correct programs, of
//! the correct programs it produced. Prompts are sampled by an upper
//! confidence bound, so one that has barely been tried gets its chance.

use anyhow::{Context, Result};
use genesis_core::{fsutil, JsonMap};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS system_prompts (
    id TEXT PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    name TEXT,
    description TEXT,
    parent_id TEXT,
    generation INTEGER NOT NULL,
    program_generation INTEGER NOT NULL DEFAULT 0,
    patch_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    program_count INTEGER NOT NULL DEFAULT 0,
    correct_program_count INTEGER NOT NULL DEFAULT 0,
    total_percentile REAL NOT NULL DEFAULT 0.0,
    total_improvement REAL NOT NULL DEFAULT 0.0,
    fitness REAL NOT NULL DEFAULT 0.0,
    program_scores TEXT,
    program_ids TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_prompts_generation ON system_prompts(generation);
CREATE INDEX IF NOT EXISTS idx_prompts_timestamp ON system_prompts(timestamp);
CREATE INDEX IF NOT EXISTS idx_prompts_fitness ON system_prompts(fitness);
CREATE INDEX IF NOT EXISTS idx_prompts_parent_id ON system_prompts(parent_id);
CREATE TABLE IF NOT EXISTS prompt_archive (
    prompt_id TEXT PRIMARY KEY,
    FOREIGN KEY (prompt_id) REFERENCES system_prompts(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS prompt_metadata_store (
    key TEXT PRIMARY KEY, value TEXT
);
"#;

const SELECT: &str = "SELECT s.id, s.prompt_text, s.name, s.description, s.parent_id, s.generation, \
    s.program_generation, s.patch_type, s.timestamp, s.program_count, s.correct_program_count, \
    s.total_percentile, s.total_improvement, s.fitness, s.program_scores, s.program_ids, s.metadata, \
    (a.prompt_id IS NOT NULL) FROM system_prompts s LEFT JOIN prompt_archive a ON a.prompt_id = s.id";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PromptDbConfig {
    pub archive_size: usize,
    pub ucb_exploration_constant: f64,
    /// Correct programs a prompt needs before its fitness is trusted.
    pub min_programs_for_fitness: u32,
    /// Probability of sampling uniformly instead of by confidence bound.
    pub epsilon: f64,
    pub seed: Option<u64>,
}

impl Default for PromptDbConfig {
    fn default() -> Self {
        Self {
            archive_size: 10,
            ucb_exploration_constant: 1.0,
            min_programs_for_fitness: 3,
            epsilon: 0.1,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct SystemPrompt {
    pub id: String,
    pub prompt_text: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub parent_id: Option<String>,
    /// Position in the prompt lineage; the task's own prompt is 0.
    pub generation: u32,
    /// Program generation at which this prompt was created.
    pub program_generation: u32,
    /// `init`, `diff` or `full`.
    pub patch_type: String,
    pub timestamp: f64,
    pub program_count: u32,
    pub correct_program_count: u32,
    pub total_percentile: f64,
    pub total_improvement: f64,
    pub fitness: f64,
    pub program_scores: Vec<f64>,
    pub program_ids: Vec<String>,
    pub metadata: JsonMap,
    pub in_archive: bool,
}

impl SystemPrompt {
    pub fn new(prompt_text: impl Into<String>, patch_type: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            prompt_text: prompt_text.into(),
            patch_type: patch_type.to_string(),
            timestamp: fsutil::now_ts(),
            ..Self::default()
        }
    }
}

struct Inner {
    conn: Connection,
    config: PromptDbConfig,
    rng: ChaCha8Rng,
}

pub struct PromptDb {
    inner: Mutex<Inner>,
}

fn parse<T: serde::de::DeserializeOwned + Default>(text: Option<String>) -> T {
    text.and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

fn row_to_prompt(row: &Row<'_>) -> rusqlite::Result<SystemPrompt> {
    Ok(SystemPrompt {
        id: row.get(0)?,
        prompt_text: row.get(1)?,
        name: row.get(2)?,
        description: row.get(3)?,
        parent_id: row.get(4)?,
        generation: row.get::<_, i64>(5)?.max(0) as u32,
        program_generation: row.get::<_, i64>(6)?.max(0) as u32,
        patch_type: row.get(7)?,
        timestamp: row.get(8)?,
        program_count: row.get::<_, i64>(9)?.max(0) as u32,
        correct_program_count: row.get::<_, i64>(10)?.max(0) as u32,
        total_percentile: row.get(11)?,
        total_improvement: row.get(12)?,
        fitness: row.get(13)?,
        program_scores: parse(row.get(14)?),
        program_ids: parse(row.get(15)?),
        metadata: parse(row.get(16)?),
        in_archive: row.get(17)?,
    })
}

fn query(
    conn: &Connection,
    tail: &str,
    args: &[&dyn rusqlite::ToSql],
) -> Result<Vec<SystemPrompt>> {
    let mut statement = conn.prepare(&format!("{SELECT} {tail}"))?;
    let rows = statement.query_map(args, row_to_prompt)?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Share of `all_scores` that `score` beats, counting ties as half.
pub fn percentile_on_insert(score: f64, all_scores: &[f64]) -> f64 {
    if all_scores.is_empty() {
        return 0.5;
    }
    let beats = all_scores.iter().filter(|other| score > **other).count() as f64;
    let ties = all_scores.iter().filter(|other| score == **other).count() as f64;
    (beats + 0.5 * ties) / all_scores.len() as f64
}

/// Share of the other scores that `score` strictly beats.
pub fn percentile_on_recompute(score: f64, all_scores: &[f64]) -> f64 {
    if all_scores.len() < 2 {
        return 0.5;
    }
    let beats = all_scores.iter().filter(|other| score > **other).count() as f64;
    beats / (all_scores.len() - 1) as f64
}

/// Whether prompt `a` should be kept over `b`: an established fitness beats
/// an unestablished one, then fitness, then recency.
fn is_better(a: &SystemPrompt, b: &SystemPrompt, min_programs: u32) -> bool {
    let (established_a, established_b) = (
        a.correct_program_count >= min_programs,
        b.correct_program_count >= min_programs,
    );
    if established_a != established_b {
        return established_a;
    }
    if a.fitness != b.fitness {
        return a.fitness > b.fitness;
    }
    a.timestamp > b.timestamp
}

impl PromptDb {
    pub fn open(path: Option<&Path>, config: PromptDbConfig) -> Result<Self> {
        let conn = match path {
            Some(path) => {
                Connection::open(path).with_context(|| format!("opening {}", path.display()))?
            }
            None => Connection::open_in_memory()?,
        };
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
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

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Store a prompt. An evolved prompt (one with a parent) always enters
    /// the archive, evicting the worst member when it is full; the task's
    /// own prompt enters only while there is room or it beats the worst.
    pub fn add(&self, prompt: &mut SystemPrompt) -> Result<()> {
        let guard = self.lock();
        let conn = &guard.conn;
        if prompt.id.is_empty() {
            prompt.id = uuid::Uuid::new_v4().to_string();
        }
        conn.execute(
            "INSERT INTO system_prompts (id, prompt_text, name, description, parent_id, generation, \
             program_generation, patch_type, timestamp, program_count, correct_program_count, total_percentile, \
             total_improvement, fitness, program_scores, program_ids, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                prompt.id,
                prompt.prompt_text,
                prompt.name,
                prompt.description,
                prompt.parent_id,
                prompt.generation,
                prompt.program_generation,
                prompt.patch_type,
                prompt.timestamp,
                prompt.program_count,
                prompt.correct_program_count,
                prompt.total_percentile,
                prompt.total_improvement,
                prompt.fitness,
                serde_json::to_string(&prompt.program_scores)?,
                serde_json::to_string(&prompt.program_ids)?,
                serde_json::to_string(&prompt.metadata)?,
            ],
        )?;
        conn.execute(
            "INSERT INTO prompt_metadata_store (key, value) VALUES ('last_generation', ?1) \
             ON CONFLICT(key) DO UPDATE SET value = MAX(CAST(value AS INTEGER), CAST(excluded.value AS INTEGER))",
            [prompt.generation.to_string()],
        )?;

        let members = query(
            conn,
            "WHERE a.prompt_id IS NOT NULL ORDER BY s.timestamp, s.id",
            &[],
        )?;
        let minimum = guard.config.min_programs_for_fitness;
        prompt.in_archive = false;
        if members.len() < guard.config.archive_size {
            conn.execute(
                "INSERT INTO prompt_archive (prompt_id) VALUES (?1)",
                [&prompt.id],
            )?;
            prompt.in_archive = true;
        } else if let Some(worst) = members.iter().reduce(|worst, member| {
            if is_better(worst, member, minimum) {
                member
            } else {
                worst
            }
        }) {
            if prompt.parent_id.is_some() || is_better(prompt, worst, minimum) {
                conn.execute(
                    "DELETE FROM prompt_archive WHERE prompt_id = ?1",
                    [&worst.id],
                )?;
                conn.execute(
                    "INSERT INTO prompt_archive (prompt_id) VALUES (?1)",
                    [&prompt.id],
                )?;
                prompt.in_archive = true;
            }
        }
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<Option<SystemPrompt>> {
        Ok(query(&self.lock().conn, "WHERE s.id = ?1", &[&id])?
            .into_iter()
            .next())
    }

    pub fn all(&self) -> Result<Vec<SystemPrompt>> {
        query(
            &self.lock().conn,
            "ORDER BY s.generation, s.timestamp, s.id",
            &[],
        )
    }

    pub fn archive(&self) -> Result<Vec<SystemPrompt>> {
        query(
            &self.lock().conn,
            "WHERE a.prompt_id IS NOT NULL ORDER BY s.fitness DESC, s.timestamp, s.id",
            &[],
        )
    }

    /// The archived prompt with the highest established fitness.
    pub fn best(&self) -> Result<Option<SystemPrompt>> {
        let guard = self.lock();
        let minimum = guard.config.min_programs_for_fitness;
        let members = query(
            &guard.conn,
            "WHERE a.prompt_id IS NOT NULL ORDER BY s.timestamp, s.id",
            &[],
        )?;
        Ok(members.into_iter().reduce(|best, member| {
            if is_better(&member, &best, minimum) {
                member
            } else {
                best
            }
        }))
    }

    pub fn next_generation(&self) -> Result<u32> {
        let guard = self.lock();
        let last: Option<String> = guard
            .conn
            .query_row(
                "SELECT value FROM prompt_metadata_store WHERE key = 'last_generation'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(last
            .and_then(|text| text.parse::<u32>().ok())
            .map_or(0, |generation| generation + 1))
    }

    /// Pick the prompt for the next proposal: usually the one with the
    /// highest upper confidence bound, sometimes a uniform draw.
    pub fn sample(&self) -> Result<Option<SystemPrompt>> {
        let mut guard = self.lock();
        let Inner { conn, config, rng } = &mut *guard;
        let members = query(
            conn,
            "WHERE a.prompt_id IS NOT NULL ORDER BY s.fitness DESC, s.timestamp, s.id",
            &[],
        )?;
        if members.len() <= 1 {
            return Ok(members.into_iter().next());
        }
        let total_correct: u32 = members
            .iter()
            .map(|prompt| prompt.correct_program_count)
            .sum();
        if total_correct == 0 || rng.gen::<f64>() < config.epsilon {
            return Ok(members.choose(rng).cloned());
        }
        let established_best = members
            .iter()
            .filter(|prompt| prompt.correct_program_count >= config.min_programs_for_fitness)
            .map(|prompt| prompt.fitness)
            .fold(None, |best: Option<f64>, fitness| {
                Some(best.map_or(fitness, |b| b.max(fitness)))
            });
        let bound = |prompt: &SystemPrompt| {
            if prompt.correct_program_count >= config.min_programs_for_fitness {
                prompt.fitness
                    + config.ucb_exploration_constant
                        * (f64::from(total_correct).ln() / f64::from(prompt.correct_program_count))
                            .sqrt()
            } else {
                established_best.unwrap_or(0.5) + config.ucb_exploration_constant
            }
        };
        Ok(members
            .iter()
            .max_by(|a, b| bound(a).total_cmp(&bound(b)))
            .cloned())
    }

    /// Credit `prompt_id` with one evaluated program. `all_correct_scores`
    /// is every correct program's score, including this one when correct.
    pub fn record_program(
        &self,
        prompt_id: &str,
        program_id: &str,
        score: f64,
        correct: bool,
        all_correct_scores: &[f64],
    ) -> Result<()> {
        let guard = self.lock();
        let Some(mut prompt) = query(&guard.conn, "WHERE s.id = ?1", &[&prompt_id])?
            .into_iter()
            .next()
        else {
            return Ok(());
        };
        prompt.program_count += 1;
        prompt.program_ids.push(program_id.to_string());
        if correct {
            prompt.correct_program_count += 1;
            prompt.program_scores.push(score);
            prompt.total_percentile += percentile_on_insert(score, all_correct_scores);
            prompt.fitness = prompt.total_percentile / f64::from(prompt.correct_program_count);
        }
        guard.conn.execute(
            "UPDATE system_prompts SET program_count = ?1, correct_program_count = ?2, total_percentile = ?3, \
             fitness = ?4, program_scores = ?5, program_ids = ?6 WHERE id = ?7",
            params![
                prompt.program_count,
                prompt.correct_program_count,
                prompt.total_percentile,
                prompt.fitness,
                serde_json::to_string(&prompt.program_scores)?,
                serde_json::to_string(&prompt.program_ids)?,
                prompt.id,
            ],
        )?;
        Ok(())
    }

    /// Re-rank every prompt's programs against the current population, so an
    /// early prompt is not flattered by the weak field it started in.
    pub fn recompute_fitness(&self, all_correct_scores: &[f64]) -> Result<()> {
        let guard = self.lock();
        for prompt in query(&guard.conn, "ORDER BY s.id", &[])? {
            if prompt.program_scores.is_empty() {
                continue;
            }
            let total: f64 = prompt
                .program_scores
                .iter()
                .map(|score| percentile_on_recompute(*score, all_correct_scores))
                .sum();
            guard.conn.execute(
                "UPDATE system_prompts SET total_percentile = ?1, fitness = ?2 WHERE id = ?3",
                params![total, total / prompt.program_scores.len() as f64, prompt.id],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db(archive_size: usize) -> PromptDb {
        PromptDb::open(
            None,
            PromptDbConfig {
                archive_size,
                seed: Some(3),
                epsilon: 0.0,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn percentiles_count_ties_as_half_on_insert_and_strictly_on_recompute() {
        assert_eq!(
            percentile_on_insert(2.0, &[1.0, 2.0, 3.0, 2.0]),
            (1.0 + 0.5 * 2.0) / 4.0
        );
        assert_eq!(percentile_on_insert(1.0, &[]), 0.5);
        assert_eq!(percentile_on_recompute(3.0, &[1.0, 2.0, 3.0]), 1.0);
        assert_eq!(percentile_on_recompute(1.0, &[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(percentile_on_recompute(1.0, &[1.0]), 0.5);
    }

    #[test]
    fn fitness_is_the_mean_percentile_of_correct_programs() {
        let db = db(4);
        let mut seed = SystemPrompt::new("You are an expert.", "init");
        db.add(&mut seed).unwrap();
        assert!(seed.in_archive);
        db.record_program(&seed.id, "p1", 1.0, true, &[1.0])
            .unwrap();
        db.record_program(&seed.id, "p2", 3.0, true, &[1.0, 3.0])
            .unwrap();
        db.record_program(&seed.id, "p3", 9.0, false, &[1.0, 3.0])
            .unwrap();
        let stored = db.get(&seed.id).unwrap().unwrap();
        assert_eq!((stored.program_count, stored.correct_program_count), (3, 2));
        // 1.0 against [1.0] ties itself: 0.5. 3.0 against [1.0, 3.0]: (1 + 0.5) / 2 = 0.75.
        assert!((stored.fitness - (0.5 + 0.75) / 2.0).abs() < 1e-12);
        db.recompute_fitness(&[1.0, 3.0, 2.0, 0.5]).unwrap();
        let stored = db.get(&seed.id).unwrap().unwrap();
        // Strict beats over the other three: 1.0 beats 1, 3.0 beats 3.
        assert!((stored.fitness - (1.0 / 3.0 + 1.0) / 2.0).abs() < 1e-12);
        assert_eq!(db.next_generation().unwrap(), 1);
    }

    #[test]
    fn an_evolved_prompt_always_enters_and_evicts_the_worst() {
        let db = db(2);
        let mut seed = SystemPrompt::new("seed", "init");
        db.add(&mut seed).unwrap();
        for index in 0..3 {
            db.record_program(&seed.id, &format!("s{index}"), 5.0, true, &[1.0, 5.0])
                .unwrap();
        }
        let mut weak = SystemPrompt::new("weak", "diff");
        weak.parent_id = Some(seed.id.clone());
        weak.generation = 1;
        db.add(&mut weak).unwrap();
        let mut newer = SystemPrompt::new("newer", "full");
        newer.parent_id = Some(seed.id.clone());
        newer.generation = 2;
        newer.timestamp = weak.timestamp + 1.0;
        db.add(&mut newer).unwrap();
        let archive: Vec<String> = db
            .archive()
            .unwrap()
            .into_iter()
            .map(|p| p.prompt_text)
            .collect();
        assert_eq!(
            archive.len(),
            2,
            "the archive never shrinks or grows past its size"
        );
        assert!(archive.contains(&"seed".to_string()) && archive.contains(&"newer".to_string()));
        assert_eq!(db.best().unwrap().unwrap().id, seed.id);
        assert_eq!(db.all().unwrap().len(), 3);
        assert_eq!(db.next_generation().unwrap(), 3);
    }

    #[test]
    fn sampling_tries_unestablished_prompts_first_then_follows_the_bound() {
        let db = db(3);
        let mut seed = SystemPrompt::new("seed", "init");
        db.add(&mut seed).unwrap();
        assert_eq!(db.sample().unwrap().unwrap().id, seed.id);
        for index in 0..3 {
            db.record_program(&seed.id, &format!("s{index}"), 2.0, true, &[1.0, 2.0, 3.0])
                .unwrap();
        }
        let mut fresh = SystemPrompt::new("fresh", "diff");
        fresh.parent_id = Some(seed.id.clone());
        db.add(&mut fresh).unwrap();
        // The untried prompt gets the optimistic prior: best fitness + c.
        assert_eq!(db.sample().unwrap().unwrap().id, fresh.id);
        for index in 0..60 {
            db.record_program(
                &fresh.id,
                &format!("f{index}"),
                0.0,
                true,
                &[0.0, 1.0, 2.0, 3.0],
            )
            .unwrap();
        }
        // Heavily sampled and poor: the bound now favours the seed.
        assert_eq!(db.sample().unwrap().unwrap().id, seed.id);
        assert!(PromptDb::open(None, PromptDbConfig::default())
            .unwrap()
            .sample()
            .unwrap()
            .is_none());
    }
}
