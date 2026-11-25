//! The run journal: every stage appends one JSON object per line to
//! `events.jsonl`, and the same stream drives the terminal view.

use crate::fsutil;
use serde_json::{json, Map, Value};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsoleMode {
    /// One concise line per event.
    #[default]
    Human,
    /// The journal lines themselves, for automation.
    Json,
    Quiet,
}

impl ConsoleMode {
    pub fn parse(text: &str) -> Self {
        match text.trim().to_ascii_lowercase().as_str() {
            "json" => Self::Json,
            "quiet" | "none" => Self::Quiet,
            _ => Self::Human,
        }
    }
}

/// Append-only structured log shared by every component of a run.
#[derive(Debug)]
pub struct EventLog {
    path: Option<PathBuf>,
    console: ConsoleMode,
    verbose: bool,
    started: Instant,
    scope: Map<String, Value>,
    write_lock: Arc<Mutex<()>>,
}

impl EventLog {
    pub fn open(path: &Path, console: ConsoleMode, verbose: bool) -> Arc<Self> {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        Arc::new(Self {
            path: Some(path.to_path_buf()),
            console,
            verbose,
            started: Instant::now(),
            scope: Map::new(),
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    /// A log that only prints; used by tests and one-off commands.
    pub fn console_only(console: ConsoleMode) -> Arc<Self> {
        Arc::new(Self {
            path: None,
            console,
            verbose: false,
            started: Instant::now(),
            scope: Map::new(),
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    pub fn null() -> Arc<Self> {
        Self::console_only(ConsoleMode::Quiet)
    }

    /// The same journal with extra fields stamped on every event, such as
    /// `{"phase": "ablations"}`.
    pub fn scoped(self: &Arc<Self>, fields: Value) -> Arc<Self> {
        let mut scope = self.scope.clone();
        if let Value::Object(extra) = fields {
            scope.extend(extra);
        }
        Arc::new(Self {
            path: self.path.clone(),
            console: self.console,
            verbose: self.verbose,
            started: self.started,
            scope,
            write_lock: Arc::clone(&self.write_lock),
        })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn info(&self, event: &str, fields: Value) {
        self.emit("info", event, fields);
    }

    pub fn warn(&self, event: &str, fields: Value) {
        self.emit("warn", event, fields);
    }

    pub fn error(&self, event: &str, fields: Value) {
        self.emit("error", event, fields);
    }

    /// Journal-only detail, shown on the terminal just in verbose mode.
    pub fn debug(&self, event: &str, fields: Value) {
        self.emit("debug", event, fields);
    }

    fn emit(&self, level: &str, event: &str, fields: Value) {
        let mut record = Map::new();
        record.insert("ts".into(), json!(chrono::Utc::now().to_rfc3339()));
        record.insert(
            "elapsed_s".into(),
            json!((self.started.elapsed().as_secs_f64() * 1000.0).round() / 1000.0),
        );
        record.insert("level".into(), json!(level));
        record.insert("event".into(), json!(event));
        for (key, value) in &self.scope {
            record.insert(key.clone(), value.clone());
        }
        match fields {
            Value::Object(extra) => record.extend(extra),
            Value::Null => {}
            other => {
                record.insert("detail".into(), other);
            }
        }
        let record = Value::Object(record);
        let _held = self.write_lock.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(path) = &self.path {
            let _ = fsutil::append_jsonl(path, &record);
        }
        match self.console {
            ConsoleMode::Quiet => {}
            ConsoleMode::Json => {
                let mut out = std::io::stdout().lock();
                let _ = writeln!(out, "{record}");
            }
            ConsoleMode::Human => {
                if level != "debug" || self.verbose {
                    let mut err = std::io::stderr().lock();
                    let _ = writeln!(err, "{}", render_human(&record, colour_enabled()));
                }
            }
        }
    }
}

fn colour_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none() && std::io::stderr().is_terminal()
}

fn render_value(value: &Value) -> String {
    match value {
        Value::String(text) => fsutil::truncate(&text.replace('\n', " "), 160),
        Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().unwrap_or(0.0);
            if x.abs() >= 1000.0 || x == x.trunc() {
                format!("{x:.1}")
            } else {
                format!("{x:.4}")
            }
        }
        other => fsutil::truncate(&other.to_string(), 160),
    }
}

/// `[  12.3s] proposal_evaluated  generation=4 score=2.1034 correct=true`
pub fn render_human(record: &Value, colour: bool) -> String {
    let level = record["level"].as_str().unwrap_or("info");
    let event = record["event"].as_str().unwrap_or("");
    let elapsed = record["elapsed_s"].as_f64().unwrap_or(0.0);
    let mut parts = Vec::new();
    if let Value::Object(map) = record {
        for (key, value) in map {
            if matches!(key.as_str(), "ts" | "elapsed_s" | "level" | "event") {
                continue;
            }
            if value.is_null() {
                continue;
            }
            parts.push(format!("{key}={}", render_value(value)));
        }
    }
    let tag = match level {
        "warn" => "WARN ",
        "error" => "ERROR ",
        _ => "",
    };
    let (start, end) = match (colour, level) {
        (true, "warn") => ("\x1b[33m", "\x1b[0m"),
        (true, "error") => ("\x1b[31m", "\x1b[0m"),
        (true, "debug") => ("\x1b[2m", "\x1b[0m"),
        _ => ("", ""),
    };
    format!(
        "{start}[{elapsed:>8.1}s] {tag}{event}  {}{end}",
        parts.join(" ")
    )
    .trim_end()
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_are_journaled_with_scope_and_rendered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run/events.jsonl");
        let log = EventLog::open(&path, ConsoleMode::Quiet, false);
        log.info("run_start", json!({"task": "circle"}));
        let scoped = log.scoped(json!({"phase": "ablations"}));
        scoped.warn(
            "variant_failed",
            json!({"variant": "no_restarts", "score": 1.23456}),
        );
        let rows: Vec<Value> = fsutil::read_jsonl(&path);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["event"], "run_start");
        assert_eq!(rows[1]["phase"], "ablations");
        assert_eq!(rows[1]["level"], "warn");
        let line = render_human(&rows[1], false);
        assert!(line.contains("WARN variant_failed"), "{line}");
        assert!(line.contains("score=1.2346"), "{line}");
    }
}
