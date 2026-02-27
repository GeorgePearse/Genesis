use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::EvolutionConfig;
use crate::types::Program;

pub trait ProgramDatabase {
    fn add(&mut self, program: Program) -> Result<()>;
    fn get(&self, id: &str) -> Result<Option<Program>>;
    fn get_best_program(&self) -> Result<Option<Program>>;
    fn get_top_programs(&self, n: usize) -> Result<Vec<Program>>;
    fn sample(&self, target_generation: usize) -> Result<Option<(Program, Vec<Program>, Vec<Program>)>>;
}

pub fn build_database(cfg: &EvolutionConfig) -> Result<Box<dyn ProgramDatabase>> {
    match cfg.db_backend.as_str() {
        "clickhouse" => Ok(Box::new(ClickHouseProgramDatabase::new(cfg)?)),
        _ => Ok(Box::new(InMemoryProgramDatabase::new())),
    }
}

#[derive(Debug, Default, Clone)]
pub struct InMemoryProgramDatabase {
    programs: Vec<Program>,
}

impl InMemoryProgramDatabase {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ProgramDatabase for InMemoryProgramDatabase {
    fn add(&mut self, program: Program) -> Result<()> {
        self.programs.push(program);
        Ok(())
    }

    fn get(&self, id: &str) -> Result<Option<Program>> {
        Ok(self.programs.iter().find(|p| p.id == id).cloned())
    }

    fn get_best_program(&self) -> Result<Option<Program>> {
        Ok(self
            .programs
            .iter()
            .filter(|p| p.correct)
            .max_by(|a, b| a.combined_score.total_cmp(&b.combined_score))
            .cloned())
    }

    fn get_top_programs(&self, n: usize) -> Result<Vec<Program>> {
        let mut ps = self.programs.clone();
        ps.sort_by(|a, b| b.combined_score.total_cmp(&a.combined_score));
        Ok(ps.into_iter().take(n).collect())
    }

    fn sample(&self, _target_generation: usize) -> Result<Option<(Program, Vec<Program>, Vec<Program>)>> {
        let parent = self.get_best_program()?.or_else(|| self.programs.last().cloned());
        let Some(parent) = parent else {
            return Ok(None);
        };
        let mut inspirations = self.get_top_programs(6)?;
        inspirations.retain(|p| p.id != parent.id);
        let archive = inspirations.iter().take(4).cloned().collect::<Vec<_>>();
        let top_k = inspirations.into_iter().take(2).collect::<Vec<_>>();
        Ok(Some((parent, archive, top_k)))
    }
}

#[derive(Debug)]
pub struct ClickHouseProgramDatabase {
    client: Client,
    base_url: String,
    user: Option<String>,
    password: Option<String>,
    database: String,
}

impl ClickHouseProgramDatabase {
    pub fn new(cfg: &EvolutionConfig) -> Result<Self> {
        let base_url = cfg
            .clickhouse_url
            .clone()
            .unwrap_or_else(|| "http://localhost:8123".to_string());

        let db = Self {
            client: Client::new(),
            base_url,
            user: cfg.clickhouse_user.clone(),
            password: cfg.clickhouse_password.clone(),
            database: cfg.clickhouse_database.clone(),
        };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS {}.programs (\
                id String,\
                code String,\
                language String,\
                parent_id Nullable(String),\
                generation UInt32,\
                combined_score Float64,\
                correct UInt8,\
                public_metrics String,\
                private_metrics String,\
                text_feedback String,\
                metadata String,\
                timestamp DateTime64(3)\
            ) ENGINE = ReplacingMergeTree(timestamp) ORDER BY id",
            self.database
        );
        self.exec_sql(&sql)?;
        Ok(())
    }

    fn exec_sql(&self, sql: &str) -> Result<String> {
        let url = format!("{}/", self.base_url.trim_end_matches('/'));
        let mut req = self.client.post(url).query(&[("query", sql)]);
        if let Some(user) = &self.user {
            req = req.basic_auth(user, self.password.clone());
        }
        let resp = req
            .send()
            .with_context(|| "clickhouse request failed")?
            .error_for_status()
            .with_context(|| "clickhouse non-success response")?;
        Ok(resp.text().unwrap_or_default())
    }

    fn select_rows(&self, sql: &str) -> Result<Vec<Program>> {
        let out = self.exec_sql(sql)?;
        let mut programs = Vec::new();
        for line in out.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let row: DbProgramRow = serde_json::from_str(line)
                .with_context(|| "failed to parse clickhouse JSONEachRow line")?;
            programs.push(row.try_into_program()?);
        }
        Ok(programs)
    }
}

impl ProgramDatabase for ClickHouseProgramDatabase {
    fn add(&mut self, program: Program) -> Result<()> {
        let row = DbProgramRow::from_program(&program)?;
        let payload = serde_json::to_string(&row)?;
        let sql = format!(
            "INSERT INTO {}.programs FORMAT JSONEachRow\n{}",
            self.database, payload
        );
        self.exec_sql(&sql)?;
        Ok(())
    }

    fn get(&self, id: &str) -> Result<Option<Program>> {
        let sql = format!(
            "SELECT * FROM {}.programs WHERE id = '{}' ORDER BY timestamp DESC LIMIT 1 FORMAT JSONEachRow",
            self.database,
            id.replace('\'', "''")
        );
        Ok(self.select_rows(&sql)?.into_iter().next())
    }

    fn get_best_program(&self) -> Result<Option<Program>> {
        let sql = format!(
            "SELECT * FROM {}.programs WHERE correct = 1 ORDER BY combined_score DESC LIMIT 1 FORMAT JSONEachRow",
            self.database
        );
        Ok(self.select_rows(&sql)?.into_iter().next())
    }

    fn get_top_programs(&self, n: usize) -> Result<Vec<Program>> {
        let sql = format!(
            "SELECT * FROM {}.programs ORDER BY combined_score DESC LIMIT {} FORMAT JSONEachRow",
            self.database,
            n.max(1)
        );
        self.select_rows(&sql)
    }

    fn sample(&self, _target_generation: usize) -> Result<Option<(Program, Vec<Program>, Vec<Program>)>> {
        let parent = self.get_best_program()?;
        let Some(parent) = parent else {
            return Ok(None);
        };

        let mut inspirations = self.get_top_programs(6)?;
        inspirations.retain(|p| p.id != parent.id);
        let archive = inspirations.iter().take(4).cloned().collect::<Vec<_>>();
        let top_k = inspirations.into_iter().take(2).collect::<Vec<_>>();
        Ok(Some((parent, archive, top_k)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DbProgramRow {
    id: String,
    code: String,
    language: String,
    parent_id: Option<String>,
    generation: u32,
    combined_score: f64,
    correct: u8,
    public_metrics: String,
    private_metrics: String,
    text_feedback: String,
    metadata: String,
    timestamp: String,
}

impl DbProgramRow {
    fn from_program(p: &Program) -> Result<Self> {
        Ok(Self {
            id: p.id.clone(),
            code: p.code.clone(),
            language: p.language.clone(),
            parent_id: p.parent_id.clone(),
            generation: p.generation as u32,
            combined_score: p.combined_score,
            correct: if p.correct { 1 } else { 0 },
            public_metrics: serde_json::to_string(&p.public_metrics)?,
            private_metrics: serde_json::to_string(&p.private_metrics)?,
            text_feedback: p.text_feedback.clone(),
            metadata: serde_json::to_string(&p.metadata)?,
            timestamp: p.timestamp.to_rfc3339(),
        })
    }

    fn try_into_program(self) -> Result<Program> {
        let public_metrics = serde_json::from_str::<HashMap<String, serde_json::Value>>(&self.public_metrics)
            .unwrap_or_default();
        let private_metrics = serde_json::from_str::<HashMap<String, serde_json::Value>>(&self.private_metrics)
            .unwrap_or_default();
        let metadata = serde_json::from_str::<HashMap<String, serde_json::Value>>(&self.metadata)
            .unwrap_or_default();
        let timestamp = DateTime::parse_from_rfc3339(&self.timestamp)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now());

        Ok(Program {
            id: self.id,
            code: self.code,
            language: self.language,
            parent_id: self.parent_id,
            generation: self.generation as usize,
            combined_score: self.combined_score,
            correct: self.correct == 1,
            public_metrics,
            private_metrics,
            text_feedback: self.text_feedback,
            metadata,
            timestamp,
        })
    }
}
