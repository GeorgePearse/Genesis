//! Filesystem helpers: atomic publication, JSON lines, tree copies, hashing.

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn now_ts() -> f64 {
    let now = chrono::Utc::now();
    now.timestamp() as f64 + f64::from(now.timestamp_subsec_micros()) / 1e6
}

/// `20261125_142301`, the directory name of a run.
pub fn new_run_id() -> String {
    chrono::Local::now().format("%Y%m%d_%H%M%S").to_string()
}

/// Lowercase, every run of non-`[a-z0-9]` becomes `_`, trimmed.
pub fn slugify(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last_sep = true;
    for ch in text.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            last_sep = false;
        } else if !last_sep {
            out.push('_');
            last_sep = true;
        }
    }
    out.trim_matches('_').to_string()
}

/// Write to a sibling temp file, fsync, then rename over the target, so a
/// reader never sees a half-written file.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(dir) = parent {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    let tmp = path.with_file_name(format!(
        ".{file_name}.{}.tmp",
        uuid::Uuid::new_v4().simple()
    ));
    {
        let mut file =
            fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("publishing {}", path.display()))?;
    Ok(())
}

pub fn write_text(path: &Path, text: &str) -> Result<()> {
    atomic_write(path, text.as_bytes())
}

pub fn write_json<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

pub fn read_json_opt<T: DeserializeOwned>(path: &Path) -> Option<T> {
    if !path.is_file() {
        return None;
    }
    read_json(path).ok()
}

pub fn append_jsonl<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    let mut line = serde_json::to_string(value)?;
    line.push('\n');
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.write_all(line.as_bytes())?;
    Ok(())
}

/// Every parseable line of a JSON-lines file; a torn last line is skipped.
pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Vec<T> {
    let Ok(raw) = fs::read_to_string(path) else {
        return Vec::new();
    };
    raw.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

/// True when `rel` (a `/`-separated relative path) is excluded by `patterns`.
///
/// A pattern matches a whole path component (`__pycache__`), a path prefix
/// (`data/raw`), or a suffix glob (`*.pyc`).
pub fn is_ignored(rel: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|pattern| {
        let pattern = pattern.trim().trim_end_matches('/');
        if pattern.is_empty() || pattern.starts_with('#') {
            return false;
        }
        if let Some(suffix) = pattern.strip_prefix('*') {
            return rel.ends_with(suffix);
        }
        rel == pattern
            || rel.starts_with(&format!("{pattern}/"))
            || rel.split('/').any(|component| component == pattern)
    })
}

fn relative_files(root: &Path, ignore: &[String]) -> Result<Vec<(String, PathBuf)>> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .follow_links(false)
        .sort_by_file_name()
    {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(root)?
            .to_string_lossy()
            .replace('\\', "/");
        if is_ignored(&rel, ignore) {
            continue;
        }
        files.push((rel, entry.path().to_path_buf()));
    }
    Ok(files)
}

/// Copy regular files under `src` into `dst`, skipping ignored paths and
/// symlinks. Returns the number of files copied.
pub fn copy_tree(src: &Path, dst: &Path, ignore: &[String]) -> Result<usize> {
    let files = relative_files(src, ignore)?;
    for (rel, path) in &files {
        let target = dst.join(rel);
        if let Some(dir) = target.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::copy(path, &target)
            .with_context(|| format!("copying {} to {}", path.display(), target.display()))?;
    }
    Ok(files.len())
}

/// Digest of every file's relative path and content, in sorted order.
pub fn sha256_tree(root: &Path, ignore: &[String]) -> Result<String> {
    let mut hasher = Sha256::new();
    for (rel, path) in relative_files(root, ignore)? {
        hasher.update(rel.as_bytes());
        hasher.update([0u8]);
        hasher.update(fs::read(&path)?);
        hasher.update([0u8]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// The last `max_bytes` of `text`, cut on a character boundary.
pub fn tail(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut start = text.len() - max_bytes;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &text[start..])
}

/// The first `max_chars` characters of `text`, with an ellipsis when cut.
pub fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let cut: String = text.chars().take(max_chars).collect();
    format!("{cut}…")
}

/// `1h 02m 03s`, `4m 12s`, `7.3s`.
pub fn human_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        return format!("{seconds:.1}s");
    }
    let total = seconds.round() as u64;
    let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
    if h > 0 {
        format!("{h}h {m:02}m {s:02}s")
    } else {
        format!("{m}m {s:02}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_collapses_separators() {
        assert_eq!(
            slugify("  Adaptive Reheating + Multi-Restart!! "),
            "adaptive_reheating_multi_restart"
        );
        assert_eq!(slugify("___"), "");
    }

    #[test]
    fn atomic_write_replaces_and_jsonl_appends() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/out.json");
        write_json(&path, &serde_json::json!({"a": 1})).unwrap();
        write_json(&path, &serde_json::json!({"a": 2})).unwrap();
        let value: serde_json::Value = read_json(&path).unwrap();
        assert_eq!(value["a"], 2);
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);

        let log = dir.path().join("log.jsonl");
        append_jsonl(&log, &serde_json::json!({"n": 1})).unwrap();
        append_jsonl(&log, &serde_json::json!({"n": 2})).unwrap();
        fs::OpenOptions::new()
            .append(true)
            .open(&log)
            .unwrap()
            .write_all(b"{\"n\": 3")
            .unwrap();
        let rows: Vec<serde_json::Value> = read_jsonl(&log);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn copy_tree_honours_ignore_patterns_and_hash_is_stable() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(src.join("__pycache__")).unwrap();
        fs::create_dir_all(src.join("data/raw")).unwrap();
        fs::write(src.join("a.py"), "a").unwrap();
        fs::write(src.join("b.pyc"), "b").unwrap();
        fs::write(src.join("__pycache__/c"), "c").unwrap();
        fs::write(src.join("data/raw/d"), "d").unwrap();
        let ignore = vec![
            "__pycache__".to_string(),
            "*.pyc".to_string(),
            "data/raw".to_string(),
        ];
        let dst = dir.path().join("dst");
        assert_eq!(copy_tree(&src, &dst, &ignore).unwrap(), 1);
        assert!(dst.join("a.py").is_file());
        assert_eq!(
            sha256_tree(&src, &ignore).unwrap(),
            sha256_tree(&dst, &ignore).unwrap()
        );
        assert_ne!(
            sha256_tree(&src, &ignore).unwrap(),
            sha256_tree(&src, &[]).unwrap()
        );
    }

    #[test]
    fn tail_and_duration() {
        assert_eq!(tail("héllo", 3), "…llo");
        assert_eq!(human_duration(7.31), "7.3s");
        assert_eq!(human_duration(252.0), "4m 12s");
        assert_eq!(human_duration(7973.0), "2h 12m 53s");
    }
}
