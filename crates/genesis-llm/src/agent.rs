//! Coding-agent CLIs as workers.
//!
//! An agent is a command-line coding assistant (Claude Code, Codex, Gemini
//! CLI, opencode, Cursor Agent, pi) run non-interactively in a working
//! directory. Each has a native adapter that knows its flags and how to read
//! its output; a `command` template runs anything else.
//!
//! **Running an agent with `allow_edits` gives it full permission to run
//! commands and change files as the current user.** Only do that in a
//! directory, and on a machine, where that is acceptable.

use crate::pricing::catalog;
use crate::types::Usage;
use anyhow::{bail, Context, Result};
use genesis_core::CostKind;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub const PROMPT_FILE_NAME: &str = ".genesis_agent_prompt.md";
const LOG_CAP_BYTES: usize = 16 * 1024 * 1024;
/// Longest prompt passed as a command-line argument; beyond it the agent is
/// pointed at the prompt file instead.
const INLINE_PROMPT_BYTES: usize = 60_000;
const BUILTIN_AGENTS: &[(&str, &str)] = &[
    ("claude", "claude"),
    ("codex", "codex"),
    ("gemini", "gemini"),
    ("opencode", "opencode"),
    ("cursor", "cursor-agent"),
    ("pi", "pi"),
];

/// Which agent to run and how.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AgentSpec {
    /// `claude`, `codex`, `gemini`, `opencode`, `cursor`, `pi`, or any label
    /// when `command` is given.
    pub name: String,
    pub model: Option<String>,
    /// `low` … `max`; passed on where the CLI has such a control.
    pub reasoning_effort: Option<String>,
    /// Replaces the native adapter. Tokens may contain `{prompt_file}`,
    /// `{prompt}`, `{model}` and `{cwd}`.
    pub command: Option<Vec<String>>,
    /// Extra environment for the agent process.
    pub env: BTreeMap<String, String>,
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

impl AgentSpec {
    /// Parse `agent:<name>[@<model>][?effort=<level>]`.
    pub fn parse(arm: &str) -> Option<Self> {
        Self::parse_reference(arm.strip_prefix("agent:")?)
    }

    /// Parse the part after a prefix: `<name>[@<model>][?effort=<level>]`.
    pub(crate) fn parse_reference(reference: &str) -> Option<Self> {
        let (head, query) = match reference.split_once('?') {
            Some((head, query)) => (head, Some(query)),
            None => (reference, None),
        };
        let (name, model) = match head.split_once('@') {
            Some((name, model)) if !model.is_empty() => (name, Some(model.to_string())),
            Some(_) => return None,
            None => (head, None),
        };
        if !valid_name(name) {
            return None;
        }
        let reasoning_effort = match query {
            None => None,
            Some(query) => {
                let effort = query.strip_prefix("effort=")?;
                if effort.is_empty() || effort.contains(['&', '=', '?']) {
                    return None;
                }
                Some(effort.to_string())
            }
        };
        Some(Self {
            name: name.to_string(),
            model,
            reasoning_effort,
            ..Self::default()
        })
    }

    /// The arm name [`AgentSpec::parse`] reads back. `command` and `env` are
    /// configuration, not part of the name.
    pub fn arm_name(&self) -> String {
        let mut name = format!("agent:{}", self.name);
        if let Some(model) = &self.model {
            name.push('@');
            name.push_str(model);
        }
        if let Some(effort) = &self.reasoning_effort {
            name.push_str("?effort=");
            name.push_str(effort);
        }
        name
    }
}

/// One job for an agent.
#[derive(Clone, Debug)]
pub struct AgentTask {
    pub prompt: String,
    /// The agent's working directory. The prompt is also written here, as
    /// [`PROMPT_FILE_NAME`].
    pub cwd: PathBuf,
    pub timeout: Duration,
    /// Full permissions when true; read-only tools when false.
    pub allow_edits: bool,
    /// Receives the agent's stdout and stderr, up to 16 MiB.
    pub log_path: PathBuf,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AgentOutcome {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    /// The agent's last message, or its error text when it failed.
    pub final_message: String,
    pub usage: Usage,
    pub cost_usd: f64,
    pub cost_kind: CostKind,
    pub duration_secs: f64,
    pub model: String,
}

impl AgentOutcome {
    pub fn succeeded(&self) -> bool {
        !self.timed_out && self.exit_code == Some(0)
    }
}

/// What the agent may do.
pub(crate) enum Mode<'a> {
    Edit,
    ReadOnly,
    /// Answer a prompt and nothing else: no tools where the CLI can switch
    /// them off, and `system` as the system prompt where it can be set.
    Completion {
        system: &'a str,
    },
}

fn on_path(binary: &str) -> bool {
    if binary.contains('/') {
        return Path::new(binary).is_file();
    }
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| {
            let candidate = dir.join(binary);
            candidate.is_file()
        })
    })
}

fn binary_of(name: &str) -> Option<&'static str> {
    BUILTIN_AGENTS
        .iter()
        .find(|(agent, _)| *agent == name)
        .map(|(_, binary)| *binary)
}

/// True when the agent's executable can be found.
pub fn agent_available(spec: &AgentSpec) -> bool {
    match &spec.command {
        Some(command) => command.first().is_some_and(|program| on_path(program)),
        None => binary_of(&spec.name).is_some_and(on_path),
    }
}

struct Invocation {
    argv: Vec<String>,
    /// File fed to the agent's stdin.
    stdin: Option<PathBuf>,
    /// File the agent writes its last message to.
    last_message: Option<PathBuf>,
    /// Environment variables withheld from the agent.
    remove_env: Vec<&'static str>,
}

/// True when the agent's CLI takes the system prompt as its own argument in
/// completion mode, so it need not be folded into the prompt text.
pub(crate) fn takes_system_argument(spec: &AgentSpec, system: &str) -> bool {
    spec.command.is_none()
        && spec.name == "claude"
        && !system.is_empty()
        && system.len() <= INLINE_PROMPT_BYTES
}

fn effort_level(spec: &AgentSpec) -> Option<&'static str> {
    crate::kwargs::enabled_effort(spec.reasoning_effort.as_deref())
}

fn pointer_prompt() -> String {
    format!(
        "Read the complete task instructions from the file {PROMPT_FILE_NAME} in the current directory, then follow them exactly."
    )
}

fn inline_or_pointer(prompt: &str) -> String {
    if prompt.len() <= INLINE_PROMPT_BYTES {
        prompt.to_string()
    } else {
        pointer_prompt()
    }
}

fn strs(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| item.to_string()).collect()
}

/// The command line for a built-in agent, as checked against each CLI's
/// `--help` (claude 2.1, codex 0.163, gemini 0.55, opencode 1.18,
/// cursor-agent 2026.10, pi 1.5).
fn native_invocation(
    spec: &AgentSpec,
    mode: &Mode<'_>,
    cwd: &Path,
    prompt_file: &Path,
    prompt: &str,
) -> Result<Invocation> {
    let cwd_text = cwd.display().to_string();
    let model = spec.model.clone();
    let effort = effort_level(spec);
    let mut invocation = Invocation {
        argv: Vec::new(),
        stdin: None,
        last_message: None,
        remove_env: Vec::new(),
    };
    match spec.name.as_str() {
        "claude" => {
            // No user settings, hooks, MCP servers or skills: an automated
            // run should not inherit an interactive setup.
            let mut argv = strs(&[
                "claude",
                "-p",
                "--output-format",
                "json",
                "--no-session-persistence",
                "--strict-mcp-config",
                "--disable-slash-commands",
                "--setting-sources",
                "",
            ]);
            match mode {
                Mode::Edit => argv.push("--dangerously-skip-permissions".into()),
                Mode::ReadOnly => argv.extend(strs(&["--tools", "Read,Glob,Grep"])),
                Mode::Completion { system } => {
                    argv.extend(strs(&["--tools", ""]));
                    if takes_system_argument(spec, system) {
                        argv.extend(["--system-prompt".to_string(), system.to_string()]);
                    }
                }
            }
            if let Some(model) = model {
                argv.extend(["--model".to_string(), model]);
            }
            if let Some(effort) = effort {
                let level = if effort == "minimal" { "low" } else { effort };
                argv.extend(["--effort".to_string(), level.to_string()]);
            }
            invocation.argv = argv;
            invocation.stdin = Some(prompt_file.to_path_buf());
            // Bill the logged-in subscription unless the caller passes a key.
            if !spec.env.contains_key("ANTHROPIC_API_KEY") {
                invocation.remove_env.push("ANTHROPIC_API_KEY");
            }
        }
        "codex" => {
            let last_message = cwd.join(".genesis_agent_last_message.txt");
            let mut argv = strs(&["codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral", "-C", &cwd_text]);
            argv.extend(["-o".to_string(), last_message.display().to_string()]);
            match mode {
                Mode::Edit => argv.push("--dangerously-bypass-approvals-and-sandbox".into()),
                Mode::ReadOnly | Mode::Completion { .. } => argv.extend(strs(&["-s", "read-only"])),
            }
            if let Some(model) = model {
                argv.extend(["-m".to_string(), model]);
            }
            if let Some(effort) = effort {
                let level = if effort == "max" { "xhigh" } else { effort };
                argv.extend(["-c".to_string(), format!("model_reasoning_effort=\"{level}\"")]);
            }
            argv.push("-".into());
            invocation.argv = argv;
            invocation.stdin = Some(prompt_file.to_path_buf());
            invocation.last_message = Some(last_message);
        }
        "gemini" => {
            let mut argv = strs(&["gemini", "-p", "Follow the instructions above.", "-o", "json", "--skip-trust"]);
            let approval = if matches!(mode, Mode::Edit) { "yolo" } else { "plan" };
            argv.extend(strs(&["--approval-mode", approval]));
            if let Some(model) = model {
                argv.extend(["-m".to_string(), model]);
            }
            invocation.argv = argv;
            invocation.stdin = Some(prompt_file.to_path_buf());
        }
        "opencode" => {
            let mut argv = strs(&["opencode", "run", "--format", "json", "--dir", &cwd_text]);
            match mode {
                Mode::Edit => argv.push("--auto".into()),
                Mode::ReadOnly | Mode::Completion { .. } => argv.extend(strs(&["--agent", "plan"])),
            }
            if let Some(model) = model {
                argv.extend(["-m".to_string(), model]);
            }
            if let Some(effort) = effort {
                argv.extend(["--variant".to_string(), effort.to_string()]);
            }
            argv.push(inline_or_pointer(prompt));
            invocation.argv = argv;
        }
        "cursor" => {
            let mut argv = strs(&["cursor-agent", "-p", "--output-format", "json", "--trust", "--workspace", &cwd_text]);
            match mode {
                Mode::Edit => argv.push("--force".into()),
                Mode::ReadOnly | Mode::Completion { .. } => argv.extend(strs(&["--mode", "ask"])),
            }
            if let Some(model) = model {
                argv.extend(["--model".to_string(), model]);
            }
            argv.push(inline_or_pointer(prompt));
            invocation.argv = argv;
        }
        "pi" => {
            let mut argv = strs(&["pi", "-p", "--mode", "json", "--no-session", "--no-extensions"]);
            match mode {
                Mode::Edit => {}
                Mode::ReadOnly => argv.extend(strs(&["--tools", "read,grep,find,ls"])),
                Mode::Completion { .. } => argv.push("--no-tools".into()),
            }
            if let Some(model) = model {
                argv.extend(["--model".to_string(), model]);
            }
            let thinking = match (effort, spec.reasoning_effort.as_deref()) {
                (Some(effort), _) => Some(effort),
                (None, Some(_)) => Some("off"),
                (None, None) => None,
            };
            if let Some(thinking) = thinking {
                argv.extend(["--thinking".to_string(), thinking.to_string()]);
            }
            argv.push(format!("@{}", prompt_file.display()));
            invocation.argv = argv;
        }
        other => bail!(
            "unknown agent `{other}`: the built-in agents are claude, codex, gemini, opencode, cursor and pi; \
             any other needs a `command`"
        ),
    }
    Ok(invocation)
}

fn custom_invocation(
    spec: &AgentSpec,
    command: &[String],
    cwd: &Path,
    prompt_file: &Path,
    prompt: &str,
) -> Result<Invocation> {
    if command.is_empty() {
        bail!("agent `{}` has an empty `command`", spec.name);
    }
    let substitutions = [
        ("{prompt_file}", prompt_file.display().to_string()),
        ("{prompt}", prompt.to_string()),
        ("{model}", spec.model.clone().unwrap_or_default()),
        ("{cwd}", cwd.display().to_string()),
    ];
    let argv = command
        .iter()
        .map(|token| {
            substitutions
                .iter()
                .fold(token.clone(), |acc, (key, value)| acc.replace(key, value))
        })
        .collect();
    Ok(Invocation {
        argv,
        stdin: None,
        last_message: None,
        remove_env: Vec::new(),
    })
}

#[cfg(unix)]
fn kill_process_group(pid: u32) {
    // SAFETY: killpg only signals; a stale or missing group returns ESRCH.
    unsafe {
        libc::killpg(pid as libc::pid_t, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: u32) {}

/// Kills the agent's process group if the owning future is dropped.
struct GroupGuard(Option<u32>);

impl Drop for GroupGuard {
    fn drop(&mut self) {
        if let Some(pid) = self.0.take() {
            kill_process_group(pid);
        }
    }
}

struct Log {
    file: tokio::fs::File,
    remaining: usize,
}

/// Drain `reader` to its end, copying into the shared log and optionally
/// into memory, both bounded. Draining continues past the bound so the agent
/// never blocks on a full pipe.
async fn pump(
    mut reader: impl tokio::io::AsyncRead + Unpin,
    log: Arc<tokio::sync::Mutex<Log>>,
    keep: bool,
) -> Vec<u8> {
    let mut kept = Vec::new();
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let read = match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        if keep && kept.len() < LOG_CAP_BYTES {
            let room = LOG_CAP_BYTES - kept.len();
            kept.extend_from_slice(&chunk[..read.min(room)]);
        }
        let mut log = log.lock().await;
        let allowed = read.min(log.remaining);
        if allowed > 0 {
            let _ = log.file.write_all(&chunk[..allowed]).await;
            log.remaining -= allowed;
        }
    }
    kept
}

/// The output a pump gathered. A process that escaped the agent's group can
/// hold the pipe open after the agent is gone; it is not waited for.
async fn collect(pump: tokio::task::JoinHandle<Vec<u8>>) -> String {
    let abort = pump.abort_handle();
    match tokio::time::timeout(Duration::from_secs(5), pump).await {
        Ok(Ok(bytes)) => String::from_utf8_lossy(&bytes).into_owned(),
        _ => {
            abort.abort();
            String::new()
        }
    }
}

#[derive(Default)]
struct Parsed {
    message: String,
    usage: Usage,
    /// A charge the CLI itself computed.
    cost_usd: Option<f64>,
    model: Option<String>,
    error: Option<String>,
}

fn count(value: &Value, pointer: &str) -> u64 {
    crate::providers::u64_at(value, pointer)
}

fn json_lines(stdout: &str) -> impl Iterator<Item = Value> + '_ {
    stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
}

fn parse_claude(stdout: &str) -> Parsed {
    let Ok(result) = serde_json::from_str::<Value>(stdout.trim()) else {
        return Parsed {
            message: stdout.trim().to_string(),
            ..Parsed::default()
        };
    };
    let fresh = count(&result, "/usage/input_tokens");
    let cache_read = count(&result, "/usage/cache_read_input_tokens");
    let cache_creation = count(&result, "/usage/cache_creation_input_tokens");
    let output = count(&result, "/usage/output_tokens");
    let thinking = count(&result, "/usage/output_tokens_details/thinking_tokens").min(output);
    let message = result
        .get("result")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let failed = result
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Parsed {
        usage: Usage {
            input_tokens: fresh + cache_read + cache_creation,
            output_tokens: output - thinking,
            cached_input_tokens: cache_read,
            thinking_tokens: thinking,
        },
        cost_usd: result.get("total_cost_usd").and_then(Value::as_f64),
        model: result
            .get("modelUsage")
            .and_then(Value::as_object)
            .and_then(|models| models.keys().next().cloned()),
        error: failed.then(|| message.clone()),
        message,
    }
}

fn parse_codex(stdout: &str, last_message: Option<String>) -> Parsed {
    let mut parsed = Parsed::default();
    for event in json_lines(stdout) {
        match event.get("type").and_then(Value::as_str) {
            Some("item.completed")
                if event.pointer("/item/type").and_then(Value::as_str) == Some("agent_message") =>
            {
                parsed.message = event
                    .pointer("/item/text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
            }
            Some("turn.completed") => {
                let output = count(&event, "/usage/output_tokens");
                let thinking = count(&event, "/usage/reasoning_output_tokens").min(output);
                parsed.usage.input_tokens += count(&event, "/usage/input_tokens");
                parsed.usage.cached_input_tokens += count(&event, "/usage/cached_input_tokens");
                parsed.usage.output_tokens += output - thinking;
                parsed.usage.thinking_tokens += thinking;
            }
            Some("turn.failed" | "error") => {
                let message = event
                    .pointer("/error/message")
                    .or_else(|| event.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("the codex turn failed");
                parsed.error = Some(message.to_string());
            }
            _ => {}
        }
    }
    if let Some(text) = last_message.filter(|text| !text.trim().is_empty()) {
        parsed.message = text.trim().to_string();
    }
    parsed
}

fn parse_gemini(stdout: &str) -> Parsed {
    let Ok(result) = serde_json::from_str::<Value>(stdout.trim()) else {
        return Parsed {
            message: stdout.trim().to_string(),
            ..Parsed::default()
        };
    };
    let mut parsed = Parsed {
        message: result
            .get("response")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        error: result
            .pointer("/error/message")
            .and_then(Value::as_str)
            .map(str::to_string),
        ..Parsed::default()
    };
    if let Some(models) = result.pointer("/stats/models").and_then(Value::as_object) {
        for (name, stats) in models {
            parsed.model.get_or_insert_with(|| name.clone());
            parsed.usage.input_tokens += count(stats, "/tokens/prompt");
            parsed.usage.output_tokens += count(stats, "/tokens/candidates");
            parsed.usage.cached_input_tokens += count(stats, "/tokens/cached");
            parsed.usage.thinking_tokens += count(stats, "/tokens/thoughts");
        }
    }
    parsed
}

fn parse_pi(stdout: &str) -> Parsed {
    let mut parsed = Parsed::default();
    let mut cost = 0.0;
    for event in json_lines(stdout) {
        let is_assistant_end = event.get("type").and_then(Value::as_str) == Some("message_end")
            && event.pointer("/message/role").and_then(Value::as_str) == Some("assistant");
        if !is_assistant_end {
            continue;
        }
        let message = &event["message"];
        let text: String = message
            .get("content")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect()
            })
            .unwrap_or_default();
        if !text.trim().is_empty() {
            parsed.message = text;
        }
        let output = count(message, "/usage/output");
        let thinking = count(message, "/usage/reasoning").min(output);
        let cache_read = count(message, "/usage/cacheRead");
        parsed.usage.input_tokens +=
            count(message, "/usage/input") + cache_read + count(message, "/usage/cacheWrite");
        parsed.usage.cached_input_tokens += cache_read;
        parsed.usage.output_tokens += output - thinking;
        parsed.usage.thinking_tokens += thinking;
        cost += message
            .pointer("/usage/cost/total")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        if let Some(model) = message.get("model").and_then(Value::as_str) {
            parsed.model = Some(model.to_string());
        }
        if message.get("stopReason").and_then(Value::as_str) == Some("error") {
            parsed.error = message
                .get("errorMessage")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
    }
    parsed.cost_usd = (cost > 0.0).then_some(cost);
    parsed
}

fn parse_opencode(stdout: &str) -> Parsed {
    let mut parsed = Parsed::default();
    let mut cost = 0.0;
    let mut text = String::new();
    for event in json_lines(stdout) {
        match event.get("type").and_then(Value::as_str) {
            Some("text") => {
                text.push_str(
                    event
                        .pointer("/part/text")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                );
            }
            Some("step_finish") => {
                parsed.usage.input_tokens += count(&event, "/part/tokens/input")
                    + count(&event, "/part/tokens/cache/read")
                    + count(&event, "/part/tokens/cache/write");
                parsed.usage.cached_input_tokens += count(&event, "/part/tokens/cache/read");
                parsed.usage.output_tokens += count(&event, "/part/tokens/output");
                parsed.usage.thinking_tokens += count(&event, "/part/tokens/reasoning");
                cost += event
                    .pointer("/part/cost")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
            }
            Some("error") => {
                let message = event
                    .pointer("/error/data/message")
                    .or_else(|| event.pointer("/error/message"))
                    .or_else(|| event.pointer("/error/name"))
                    .and_then(Value::as_str)
                    .unwrap_or("opencode reported an error");
                parsed.error = Some(message.to_string());
            }
            _ => {}
        }
    }
    parsed.message = text;
    parsed.cost_usd = (cost > 0.0).then_some(cost);
    parsed
}

fn parse_cursor(stdout: &str) -> Parsed {
    let result = json_lines(stdout)
        .filter(|event| event.get("type").and_then(Value::as_str) == Some("result"))
        .last();
    let Some(result) = result else {
        return Parsed {
            message: stdout.trim().to_string(),
            ..Parsed::default()
        };
    };
    let message = result
        .get("result")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let failed = result
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Parsed {
        error: failed.then(|| message.clone()),
        message,
        ..Parsed::default()
    }
}

/// Output of a `command` agent: its stdout, minus an optional final JSON
/// line with a top-level `usage` object (the convention of the `headless`
/// wrapper's `--usage` flag).
fn parse_custom(stdout: &str) -> Parsed {
    let lines: Vec<&str> = stdout.lines().collect();
    let last = lines.iter().rposition(|line| !line.trim().is_empty());
    let trailer = last.and_then(|index| {
        let value: Value = serde_json::from_str(lines[index].trim()).ok()?;
        value
            .get("usage")
            .filter(|usage| usage.is_object())
            .cloned()
            .map(|usage| (index, usage))
    });
    let Some((index, usage)) = trailer else {
        return Parsed {
            message: stdout.trim().to_string(),
            ..Parsed::default()
        };
    };
    let first = |keys: &[&str]| {
        let found = keys
            .iter()
            .find_map(|key| usage.get(*key).and_then(Value::as_u64));
        found
            .unwrap_or(0)
            .min(crate::providers::MAX_PLAUSIBLE_TOKENS)
    };
    let cost = usage
        .get("cost")
        .or_else(|| usage.get("total_cost"))
        .and_then(|cost| {
            cost.as_f64()
                .or_else(|| cost.get("total").and_then(Value::as_f64))
        });
    Parsed {
        message: lines[..index].join("\n").trim().to_string(),
        usage: Usage {
            input_tokens: first(&["input_tokens", "prompt_tokens", "inputTokens"]),
            output_tokens: first(&["output_tokens", "completion_tokens", "outputTokens"]),
            cached_input_tokens: first(&["cached_input_tokens", "cachedInputTokens"]),
            thinking_tokens: first(&[
                "thinking_tokens",
                "reasoning_tokens",
                "reasoningOutputTokens",
            ]),
        },
        cost_usd: cost.filter(|cost| cost.is_finite() && *cost >= 0.0),
        ..Parsed::default()
    }
}

fn parse_output(spec: &AgentSpec, stdout: &str, last_message: Option<String>) -> Parsed {
    if spec.command.is_some() {
        return parse_custom(stdout);
    }
    match spec.name.as_str() {
        "claude" => parse_claude(stdout),
        "codex" => parse_codex(stdout, last_message),
        "gemini" => parse_gemini(stdout),
        "opencode" => parse_opencode(stdout),
        "cursor" => parse_cursor(stdout),
        "pi" => parse_pi(stdout),
        _ => parse_custom(stdout),
    }
}

/// The charge to record. A figure from the CLI is a list-price estimate
/// unless the run was billed to an API key the caller supplied; without one,
/// known models are priced from the catalog.
fn settle_cost(spec: &AgentSpec, parsed: &Parsed, model: &str) -> (f64, CostKind) {
    let billed_to_key = spec.env.keys().any(|key| key.ends_with("_API_KEY"));
    // A charge that is negative or not a number is output gone wrong, not a bill.
    if let Some(cost) = parsed
        .cost_usd
        .filter(|cost| cost.is_finite() && *cost >= 0.0)
    {
        return (
            cost,
            if billed_to_key {
                CostKind::Reported
            } else {
                CostKind::Estimated
            },
        );
    }
    let billed_output = parsed.usage.billed_output_tokens();
    match catalog().cost(model, parsed.usage.input_tokens, billed_output) {
        Some((input, output)) => (input + output, CostKind::Estimated),
        None => (0.0, CostKind::UsageOnly),
    }
}

pub(crate) async fn run_with_mode(
    spec: &AgentSpec,
    task: &AgentTask,
    mode: Mode<'_>,
) -> Result<AgentOutcome> {
    std::fs::create_dir_all(&task.cwd)
        .with_context(|| format!("creating {}", task.cwd.display()))?;
    let cwd = task
        .cwd
        .canonicalize()
        .with_context(|| format!("resolving {}", task.cwd.display()))?;
    let prompt_file = cwd.join(PROMPT_FILE_NAME);
    std::fs::write(&prompt_file, &task.prompt)
        .with_context(|| format!("writing {}", prompt_file.display()))?;
    if let Some(dir) = task
        .log_path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
    {
        std::fs::create_dir_all(dir)?;
    }

    let invocation = match &spec.command {
        Some(command) => custom_invocation(spec, command, &cwd, &prompt_file, &task.prompt)?,
        None => native_invocation(spec, &mode, &cwd, &prompt_file, &task.prompt)?,
    };
    if let Some(stale) = &invocation.last_message {
        let _ = std::fs::remove_file(stale);
    }

    let mut command = tokio::process::Command::new(&invocation.argv[0]);
    command
        .args(&invocation.argv[1..])
        .current_dir(&cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    match &invocation.stdin {
        Some(path) => command.stdin(Stdio::from(std::fs::File::open(path)?)),
        None => command.stdin(Stdio::null()),
    };
    for name in &invocation.remove_env {
        command.env_remove(name);
    }
    for (key, value) in &spec.env {
        command.env(key, value);
    }
    #[cfg(unix)]
    command.process_group(0);

    let started = Instant::now();
    let mut child = command
        .spawn()
        .with_context(|| format!("starting agent `{}` ({})", spec.name, invocation.argv[0]))?;
    let mut guard = GroupGuard(child.id());
    let log = Arc::new(tokio::sync::Mutex::new(Log {
        file: tokio::fs::File::create(&task.log_path)
            .await
            .with_context(|| format!("creating {}", task.log_path.display()))?,
        remaining: LOG_CAP_BYTES,
    }));
    let stdout = tokio::spawn(pump(
        child.stdout.take().expect("stdout is piped"),
        Arc::clone(&log),
        true,
    ));
    let stderr = tokio::spawn(pump(
        child.stderr.take().expect("stderr is piped"),
        Arc::clone(&log),
        true,
    ));

    let mut timed_out = false;
    let status = match tokio::time::timeout(task.timeout, child.wait()).await {
        Ok(status) => status?,
        Err(_) => {
            timed_out = true;
            if let Some(pid) = guard.0 {
                kill_process_group(pid);
            }
            child.wait().await?
        }
    };
    // Anything the agent left running would hold the pipes open.
    if let Some(pid) = guard.0.take() {
        kill_process_group(pid);
    }
    let stdout = collect(stdout).await;
    let stderr = collect(stderr).await;
    let _ = log.lock().await.file.flush().await;

    let last_message = invocation
        .last_message
        .as_ref()
        .and_then(|path| std::fs::read_to_string(path).ok());
    let parsed = parse_output(spec, &stdout, last_message);
    let model = parsed
        .model
        .clone()
        .or_else(|| spec.model.clone())
        .unwrap_or_default();
    let (cost_usd, cost_kind) = settle_cost(spec, &parsed, &model);
    let failed = timed_out || status.code() != Some(0);
    let final_message = match (&parsed.error, failed) {
        (Some(error), _) => error.clone(),
        (None, true) if parsed.message.trim().is_empty() => {
            genesis_core::fsutil::tail(stderr.trim(), 4000)
        }
        _ => parsed.message.trim().to_string(),
    };
    // A CLI that reports its failure in-band while exiting 0 still failed.
    let exit_code = match (status.code(), &parsed.error) {
        (Some(0), Some(_)) => Some(1),
        (code, _) => code,
    };
    Ok(AgentOutcome {
        exit_code,
        timed_out,
        final_message,
        usage: parsed.usage,
        cost_usd,
        cost_kind,
        duration_secs: started.elapsed().as_secs_f64(),
        model,
    })
}

/// Run a coding agent on `task` and report what it said, used and cost.
///
/// `Err` means the agent could not be started at all. An agent that ran and
/// failed, or was killed at the timeout, is an `Ok` outcome with a non-zero
/// `exit_code` or `timed_out` set.
pub async fn run_agent(spec: &AgentSpec, task: &AgentTask) -> Result<AgentOutcome> {
    let mode = if task.allow_edits {
        Mode::Edit
    } else {
        Mode::ReadOnly
    };
    run_with_mode(spec, task, mode).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> AgentSpec {
        AgentSpec {
            name: name.to_string(),
            ..AgentSpec::default()
        }
    }

    fn argv(spec: &AgentSpec, mode: Mode<'_>) -> Vec<String> {
        native_invocation(
            spec,
            &mode,
            Path::new("/work"),
            Path::new("/work/.genesis_agent_prompt.md"),
            "do it",
        )
        .unwrap()
        .argv
    }

    #[test]
    fn arm_names_parse_and_print() {
        let full = AgentSpec::parse("agent:codex@gpt-5.6-terra?effort=high").unwrap();
        assert_eq!(full.name, "codex");
        assert_eq!(full.model.as_deref(), Some("gpt-5.6-terra"));
        assert_eq!(full.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(full.arm_name(), "agent:codex@gpt-5.6-terra?effort=high");
        assert_eq!(
            AgentSpec::parse("agent:claude").unwrap().arm_name(),
            "agent:claude"
        );
        let slashed = AgentSpec::parse("agent:opencode@anthropic/claude-opus-5-5").unwrap();
        assert_eq!(slashed.model.as_deref(), Some("anthropic/claude-opus-5-5"));
        for bad in [
            "claude",
            "agent:",
            "agent:cl aude",
            "agent:claude@",
            "agent:claude?speed=1",
            "agent:claude?effort=",
        ] {
            assert_eq!(AgentSpec::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn claude_modes_differ_only_in_permissions() {
        let edit = argv(&spec("claude"), Mode::Edit);
        assert!(edit.contains(&"--dangerously-skip-permissions".to_string()));
        assert!(edit.windows(2).any(|w| w == ["--setting-sources", ""]));
        let read = argv(&spec("claude"), Mode::ReadOnly);
        assert!(read.windows(2).any(|w| w == ["--tools", "Read,Glob,Grep"]));
        assert!(!read.contains(&"--dangerously-skip-permissions".to_string()));
        let mut with_model = spec("claude");
        with_model.model = Some("haiku".into());
        with_model.reasoning_effort = Some("minimal".into());
        let completion = argv(
            &with_model,
            Mode::Completion {
                system: "Be terse.",
            },
        );
        assert!(completion.windows(2).any(|w| w == ["--tools", ""]));
        assert!(completion
            .windows(2)
            .any(|w| w == ["--system-prompt", "Be terse."]));
        assert!(completion.windows(2).any(|w| w == ["--model", "haiku"]));
        assert!(completion.windows(2).any(|w| w == ["--effort", "low"]));
    }

    #[test]
    fn claude_bills_the_subscription_unless_given_a_key() {
        let plain = native_invocation(
            &spec("claude"),
            &Mode::Edit,
            Path::new("/w"),
            Path::new("/w/p"),
            "x",
        )
        .unwrap();
        assert_eq!(plain.remove_env, ["ANTHROPIC_API_KEY"]);
        let mut keyed = spec("claude");
        keyed
            .env
            .insert("ANTHROPIC_API_KEY".into(), "sk-test".into());
        let keyed = native_invocation(&keyed, &Mode::Edit, Path::new("/w"), Path::new("/w/p"), "x")
            .unwrap();
        assert!(keyed.remove_env.is_empty());
    }

    #[test]
    fn other_agents_get_their_own_flags() {
        let mut codex = spec("codex");
        codex.reasoning_effort = Some("max".into());
        let read = argv(&codex, Mode::ReadOnly);
        assert!(read.windows(2).any(|w| w == ["-s", "read-only"]));
        assert!(read.contains(&"model_reasoning_effort=\"xhigh\"".to_string()));
        assert_eq!(read.last().map(String::as_str), Some("-"));
        assert!(argv(&codex, Mode::Edit)
            .contains(&"--dangerously-bypass-approvals-and-sandbox".to_string()));

        assert!(argv(&spec("gemini"), Mode::Edit)
            .windows(2)
            .any(|w| w == ["--approval-mode", "yolo"]));
        assert!(argv(&spec("gemini"), Mode::ReadOnly)
            .windows(2)
            .any(|w| w == ["--approval-mode", "plan"]));
        assert!(argv(&spec("opencode"), Mode::Edit).contains(&"--auto".to_string()));
        assert!(argv(&spec("opencode"), Mode::ReadOnly)
            .windows(2)
            .any(|w| w == ["--agent", "plan"]));
        assert_eq!(
            argv(&spec("opencode"), Mode::Edit)
                .last()
                .map(String::as_str),
            Some("do it")
        );
        assert!(argv(&spec("cursor"), Mode::Edit).contains(&"--force".to_string()));
        assert!(argv(&spec("cursor"), Mode::ReadOnly)
            .windows(2)
            .any(|w| w == ["--mode", "ask"]));

        let mut pi = spec("pi");
        pi.reasoning_effort = Some("disabled".into());
        let pi_read = argv(&pi, Mode::ReadOnly);
        assert!(pi_read.windows(2).any(|w| w == ["--thinking", "off"]));
        assert!(pi_read
            .windows(2)
            .any(|w| w == ["--tools", "read,grep,find,ls"]));
        assert_eq!(
            pi_read.last().map(String::as_str),
            Some("@/work/.genesis_agent_prompt.md")
        );
        assert!(
            argv(&spec("pi"), Mode::Completion { system: "" }).contains(&"--no-tools".to_string())
        );
        assert!(native_invocation(
            &spec("emacs"),
            &Mode::Edit,
            Path::new("/w"),
            Path::new("/w/p"),
            "x"
        )
        .is_err());
    }

    #[test]
    fn a_long_prompt_is_passed_by_reference() {
        let long = "x".repeat(INLINE_PROMPT_BYTES + 1);
        let invocation = native_invocation(
            &spec("cursor"),
            &Mode::Edit,
            Path::new("/w"),
            Path::new("/w/p"),
            &long,
        )
        .unwrap();
        assert!(invocation.argv.last().unwrap().contains(PROMPT_FILE_NAME));
    }

    #[test]
    fn claude_output_is_read_from_a_real_capture() {
        let stdout = r#"{"type":"result","subtype":"success","is_error":false,"result":"OK","total_cost_usd":0.0001332,"usage":{"input_tokens":2,"cache_creation_input_tokens":655,"cache_read_input_tokens":10,"output_tokens":14,"output_tokens_details":{"thinking_tokens":4}},"modelUsage":{"claude-haiku-5-5":{"inputTokens":2,"costUSD":0.0001332,"costBasis":"list"}}}"#;
        let parsed = parse_claude(stdout);
        assert_eq!(parsed.message, "OK");
        assert_eq!(
            parsed.usage,
            Usage {
                input_tokens: 667,
                output_tokens: 10,
                cached_input_tokens: 10,
                thinking_tokens: 4
            }
        );
        assert_eq!(parsed.cost_usd, Some(0.0001332));
        assert_eq!(parsed.model.as_deref(), Some("claude-haiku-5-5"));
        assert!(parsed.error.is_none());
        let failed = parse_claude(r#"{"type":"result","is_error":true,"result":"Not logged in"}"#);
        assert_eq!(failed.error.as_deref(), Some("Not logged in"));
    }

    #[test]
    fn codex_and_pi_output_is_read_from_real_captures() {
        let codex = concat!(
            "{\"type\":\"thread.started\",\"thread_id\":\"t\"}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"item_0\",\"type\":\"error\",\"message\":\"a deprecation notice\"}}\n",
            "{\"type\":\"turn.started\"}\n",
            "{\"type\":\"item.completed\",\"item\":{\"id\":\"item_1\",\"type\":\"agent_message\",\"text\":\"OK\"}}\n",
            "{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":22929,\"cached_input_tokens\":100,\"cache_write_input_tokens\":0,\"output_tokens\":5,\"reasoning_output_tokens\":2}}\n",
        );
        let parsed = parse_codex(codex, None);
        assert_eq!(parsed.message, "OK");
        assert_eq!(
            parsed.usage,
            Usage {
                input_tokens: 22929,
                output_tokens: 3,
                cached_input_tokens: 100,
                thinking_tokens: 2
            }
        );
        assert!(parsed.error.is_none(), "a notice item is not a failed turn");
        assert_eq!(
            parse_codex(codex, Some("from file\n".into())).message,
            "from file"
        );

        let pi = concat!(
            "{\"type\":\"message_end\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"prompt\"}]}}\n",
            "{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"OK\"}],\"model\":\"gpt-6.1-sol\",\"usage\":{\"input\":504,\"output\":5,\"cacheRead\":20,\"cacheWrite\":0,\"reasoning\":1,\"cost\":{\"total\":0.001058}},\"stopReason\":\"stop\"}}\n",
        );
        let parsed = parse_pi(pi);
        assert_eq!(
            (parsed.message.as_str(), parsed.model.as_deref()),
            ("OK", Some("gpt-6.1-sol"))
        );
        assert_eq!(
            parsed.usage,
            Usage {
                input_tokens: 524,
                output_tokens: 4,
                cached_input_tokens: 20,
                thinking_tokens: 1
            }
        );
        assert_eq!(parsed.cost_usd, Some(0.001058));
    }

    #[test]
    fn gemini_opencode_and_cursor_output_shapes() {
        let gemini = parse_gemini(
            r#"{"response":"hello","stats":{"models":{"gemini-3.5-flash":{"tokens":{"prompt":10,"candidates":4,"cached":1,"thoughts":2}}}}}"#,
        );
        assert_eq!(
            (gemini.message.as_str(), gemini.model.as_deref()),
            ("hello", Some("gemini-3.5-flash"))
        );
        assert_eq!(
            gemini.usage,
            Usage {
                input_tokens: 10,
                output_tokens: 4,
                cached_input_tokens: 1,
                thinking_tokens: 2
            }
        );
        let denied =
            parse_gemini(r#"{"error":{"type":"Error","message":"no credentials","code":41}}"#);
        assert_eq!(denied.error.as_deref(), Some("no credentials"));

        let opencode = parse_opencode(concat!(
            "{\"type\":\"text\",\"part\":{\"text\":\"Hel\"}}\n",
            "{\"type\":\"text\",\"part\":{\"text\":\"lo\"}}\n",
            "{\"type\":\"step_finish\",\"part\":{\"tokens\":{\"input\":7,\"output\":3,\"reasoning\":1,\"cache\":{\"read\":2,\"write\":0}},\"cost\":0.002}}\n",
        ));
        assert_eq!(opencode.message, "Hello");
        assert_eq!(
            opencode.usage,
            Usage {
                input_tokens: 9,
                output_tokens: 3,
                cached_input_tokens: 2,
                thinking_tokens: 1
            }
        );
        assert_eq!(opencode.cost_usd, Some(0.002));
        let expired = parse_opencode(
            r#"{"type":"error","error":{"name":"APIError","data":{"message":"This API key has expired.","statusCode":401}}}"#,
        );
        assert_eq!(expired.error.as_deref(), Some("This API key has expired."));

        let cursor = parse_cursor(
            r#"{"type":"result","subtype":"success","is_error":false,"result":"done"}"#,
        );
        assert_eq!(cursor.message, "done");
    }

    #[test]
    fn a_command_agent_may_append_a_usage_line() {
        let parsed = parse_custom("line one\n\nline two\n{\"usage\":{\"input_tokens\":3,\"completion_tokens\":4,\"reasoning_tokens\":1,\"cost\":{\"total\":0.5}}}\n");
        assert_eq!(parsed.message, "line one\n\nline two");
        assert_eq!(
            parsed.usage,
            Usage {
                input_tokens: 3,
                output_tokens: 4,
                cached_input_tokens: 0,
                thinking_tokens: 1
            }
        );
        assert_eq!(parsed.cost_usd, Some(0.5));
        assert_eq!(
            parse_custom("just text {\"usage\": 1}").message,
            "just text {\"usage\": 1}"
        );
    }

    #[test]
    fn cost_is_reported_only_when_billed_to_a_supplied_key() {
        let parsed = Parsed {
            cost_usd: Some(0.25),
            ..Parsed::default()
        };
        assert_eq!(
            settle_cost(&spec("claude"), &parsed, ""),
            (0.25, CostKind::Estimated)
        );
        let mut keyed = spec("claude");
        keyed.env.insert("ANTHROPIC_API_KEY".into(), "sk".into());
        assert_eq!(settle_cost(&keyed, &parsed, ""), (0.25, CostKind::Reported));
        let usage_only = Parsed {
            usage: Usage {
                input_tokens: 1_000_000,
                ..Usage::default()
            },
            ..Parsed::default()
        };
        assert_eq!(
            settle_cost(&spec("codex"), &usage_only, "unknown-model"),
            (0.0, CostKind::UsageOnly)
        );
        let priced = settle_cost(&spec("codex"), &usage_only, "gpt-4.1-mini");
        assert!((priced.0 - 0.4).abs() < 1e-9 && priced.1 == CostKind::Estimated);
    }

    /// Found by the `agent_output_is_parsed_without_panicking_whatever_it_holds`
    /// property (EPHYRA_REPLAY=1:2): a CLI that reported a negative charge had
    /// it recorded as the cost of the call.
    #[test]
    fn a_nonsensical_reported_charge_is_not_recorded() {
        let parsed = parse_claude(
            r#"{"type":"result","is_error":false,"result":"OK","total_cost_usd":-9223372036854775808,"usage":{"input_tokens":2,"output_tokens":5}}"#,
        );
        assert_eq!(
            settle_cost(&spec("claude"), &parsed, ""),
            (0.0, CostKind::UsageOnly)
        );
        let infinite = Parsed {
            cost_usd: Some(f64::INFINITY),
            ..Parsed::default()
        };
        assert_eq!(
            settle_cost(&spec("pi"), &infinite, ""),
            (0.0, CostKind::UsageOnly)
        );
    }

    /// Found by the same property (EPHYRA_REPLAY=1:152, 2:26 and 3:272):
    /// token counts near u64::MAX overflowed the usage sums of the pi, claude
    /// and codex parsers, a panic in debug builds and a wrapped count in release.
    #[test]
    fn absurd_token_counts_do_not_overflow_usage_sums() {
        let huge = u64::MAX - 1;
        let claude = parse_claude(&format!(
            r#"{{"result":"OK","usage":{{"input_tokens":2,"cache_creation_input_tokens":655,"cache_read_input_tokens":{huge},"output_tokens":14}}}}"#
        ));
        assert!(claude.usage.input_tokens >= claude.usage.cached_input_tokens);
        let turn = format!(
            r#"{{"type":"turn.completed","usage":{{"input_tokens":{huge},"output_tokens":5}}}}"#
        );
        let codex = parse_codex(&format!("{turn}\n{turn}"), None);
        let message = format!(
            r#"{{"type":"message_end","message":{{"role":"assistant","content":[],"usage":{{"input":504,"output":5,"cacheRead":20,"cacheWrite":{huge}}}}}}}"#
        );
        let pi = parse_pi(&format!("{message}\n{message}"));
        for usage in [claude.usage, codex.usage, pi.usage] {
            assert!(usage.input_tokens > 0);
            let (cost, _) = settle_cost(
                &spec("codex"),
                &Parsed {
                    usage,
                    ..Parsed::default()
                },
                "gpt-4.1-mini",
            );
            assert!(cost.is_finite());
        }
    }

    fn captures() -> ephyra::BoxGen<(String, String)> {
        use crate::testgen::{near, near_lines};
        use ephyra::prelude::*;
        use serde_json::json;
        let claude = near(json!({"type": "result", "is_error": false, "result": "OK", "total_cost_usd": 0.01,
            "usage": {"input_tokens": 2, "cache_creation_input_tokens": 655, "cache_read_input_tokens": 10,
                      "output_tokens": 14, "output_tokens_details": {"thinking_tokens": 4}},
            "modelUsage": {"claude-haiku-5-5": {"costUSD": 0.01}}}))
        .map(|value| ("claude".to_string(), value.to_string()));
        let gemini = near(json!({"response": "hello", "error": {"message": "denied"},
            "stats": {"models": {"gemini-3.5-flash": {"tokens": {"prompt": 10, "candidates": 4, "cached": 1, "thoughts": 2}}}}}))
        .map(|value| ("gemini".to_string(), value.to_string()));
        let cursor = near(json!({"type": "result", "is_error": false, "result": "done"}))
            .map(|value| ("cursor".to_string(), value.to_string()));
        let codex = near_lines(vec![
            json!({"type": "item.completed", "item": {"type": "agent_message", "text": "OK"}}),
            json!({"type": "turn.completed", "usage": {"input_tokens": 9, "cached_input_tokens": 1, "output_tokens": 5, "reasoning_output_tokens": 2}}),
            json!({"type": "turn.completed", "usage": {"input_tokens": 9, "cached_input_tokens": 1, "output_tokens": 5, "reasoning_output_tokens": 2}}),
            json!({"type": "turn.failed", "error": {"message": "boom"}}),
        ])
        .map(|lines| ("codex".to_string(), lines));
        let pi_message = json!({"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "OK"}],
            "model": "gpt-6.1-sol", "stopReason": "stop",
            "usage": {"input": 504, "output": 5, "cacheRead": 20, "cacheWrite": 3, "reasoning": 1, "cost": {"total": 0.001}}}});
        let pi =
            near_lines(vec![pi_message.clone(), pi_message]).map(|lines| ("pi".to_string(), lines));
        let step = json!({"type": "step_finish", "part": {"tokens": {"input": 7, "output": 3, "reasoning": 1, "cache": {"read": 2, "write": 1}}, "cost": 0.002}});
        let opencode = near_lines(vec![
            json!({"type": "text", "part": {"text": "Hello"}}),
            step.clone(),
            step,
            json!({"type": "error", "error": {"name": "APIError", "data": {"message": "expired"}}}),
        ])
        .map(|lines| ("opencode".to_string(), lines));
        let custom = near(json!({"usage": {"input_tokens": 3, "completion_tokens": 4, "cached_input_tokens": 1, "reasoning_tokens": 1, "cost": {"total": 0.5}}}))
            .map(|value| ("fake".to_string(), format!("some output\n{value}\n")));
        ephyra::gens::one_of(vec![
            claude.boxed(),
            gemini.boxed(),
            cursor.boxed(),
            codex.boxed(),
            pi.boxed(),
            opencode.boxed(),
            custom.boxed(),
        ])
        .boxed()
    }

    /// Agent output is untrusted: a CLI that prints nonsense, or counts that
    /// cannot be true, must yield an outcome, never a panic.
    #[ephyra::test(gen = captures())]
    fn agent_output_is_parsed_without_panicking_whatever_it_holds(capture: (String, String)) {
        let (agent, stdout) = capture;
        let mut agent_spec = spec(&agent);
        if agent == "fake" {
            agent_spec.command = Some(vec!["true".into()]);
        }
        for text in [
            stdout.clone(),
            stdout.replace('\n', "\n\n"),
            stdout.chars().take(stdout.len() / 2).collect(),
        ] {
            let parsed = parse_output(&agent_spec, &text, Some(text.clone()));
            let (cost, _) =
                settle_cost(&agent_spec, &parsed, parsed.model.as_deref().unwrap_or(""));
            assert!(cost.is_finite() && cost >= 0.0, "cost {cost} from {text}");
        }
    }

    fn fake(script: &str) -> AgentSpec {
        AgentSpec {
            name: "fake".into(),
            model: Some("m1".into()),
            command: Some(vec![
                "bash".into(),
                "-c".into(),
                script.into(),
                "agent".into(),
                "{prompt_file}".into(),
                "{model}".into(),
            ]),
            ..AgentSpec::default()
        }
    }

    fn task(dir: &Path, timeout_secs: u64) -> AgentTask {
        AgentTask {
            prompt: "append a line".into(),
            cwd: dir.join("work"),
            timeout: Duration::from_secs(timeout_secs),
            allow_edits: true,
            log_path: dir.join("logs/agent.log"),
        }
    }

    #[tokio::test]
    async fn a_command_agent_runs_in_the_workspace_and_is_logged() {
        let dir = tempfile::tempdir().unwrap();
        let script = r#"cat "$1" > seen.txt; echo "model=$2" >> seen.txt; echo "warning" >&2; echo "edited the program"; echo '{"usage":{"input_tokens":11,"output_tokens":7}}'"#;
        let outcome = run_agent(&fake(script), &task(dir.path(), 30))
            .await
            .unwrap();
        assert!(outcome.succeeded(), "{outcome:?}");
        assert_eq!(outcome.final_message, "edited the program");
        assert_eq!(
            (outcome.usage.input_tokens, outcome.usage.output_tokens),
            (11, 7)
        );
        assert_eq!(outcome.model, "m1");
        let seen = std::fs::read_to_string(dir.path().join("work/seen.txt")).unwrap();
        assert_eq!(seen, "append a linemodel=m1\n");
        let log = std::fs::read_to_string(dir.path().join("logs/agent.log")).unwrap();
        assert!(log.contains("edited the program") && log.contains("warning"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("work").join(PROMPT_FILE_NAME)).unwrap(),
            "append a line"
        );
    }

    #[tokio::test]
    async fn a_failing_agent_reports_its_stderr_and_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        let outcome = run_agent(
            &fake("echo 'cannot authenticate' >&2; exit 3"),
            &task(dir.path(), 30),
        )
        .await
        .unwrap();
        assert_eq!(outcome.exit_code, Some(3));
        assert!(!outcome.succeeded());
        assert_eq!(outcome.final_message, "cannot authenticate");
    }

    #[tokio::test]
    async fn a_timed_out_agent_is_killed_with_its_children() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("survivor");
        let script = format!("(sleep 3; touch {}) & sleep 30", marker.display());
        let started = Instant::now();
        let outcome = run_agent(&fake(&script), &task(dir.path(), 1))
            .await
            .unwrap();
        assert!(outcome.timed_out && !outcome.succeeded());
        assert!(started.elapsed() < Duration::from_secs(10));
        tokio::time::sleep(Duration::from_secs(4)).await;
        assert!(!marker.exists(), "a grandchild outlived the timeout");
    }

    #[tokio::test]
    async fn an_agent_that_cannot_start_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = AgentSpec {
            name: "ghost".into(),
            command: Some(vec!["/no/such/agent".into()]),
            ..AgentSpec::default()
        };
        assert!(run_agent(&missing, &task(dir.path(), 5)).await.is_err());
        assert!(!agent_available(&missing));
        assert!(agent_available(&fake("true")));
        assert!(!agent_available(&spec("emacs")));
    }
}
