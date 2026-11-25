//! Coding-agent CLIs as plain completion backends: `cli/claude`,
//! `cli/codex@gpt-5.6-terra?effort=high`, and ShinkaEvolve's spelling
//! `headless/<agent>`. The agent answers one prompt with no permission to
//! change anything, which lets a run draw on a CLI's own login instead of an
//! API key.

use super::{Backend, Billed, Call, Failure};
use crate::agent::{run_with_mode, takes_system_argument, AgentSpec, AgentTask, Mode};
use crate::types::{LlmError, LlmResponse, Role};
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;

pub(crate) struct CliBackend {
    pub spec: AgentSpec,
    /// Each call gets its own directory under here.
    pub work_dir: PathBuf,
    /// Calls are kept when the caller chose the directory, removed otherwise.
    pub keep_files: bool,
    /// One CLI process at a time, whatever the caller's concurrency.
    pub gate: Arc<tokio::sync::Mutex<()>>,
}

fn conversation(call: &Call<'_>, with_system: bool) -> String {
    let request = call.request;
    let (history, last) = match request.messages.split_last() {
        Some((last, history)) if last.role == Role::User => (history, last.content.as_str()),
        _ => (request.messages.as_slice(), ""),
    };
    if !with_system && history.is_empty() {
        return last.to_string();
    }
    let mut prompt = String::new();
    if with_system && !request.system.is_empty() {
        prompt.push_str(&format!("# System Instructions\n\n{}\n\n", request.system));
    }
    if !history.is_empty() {
        let turns = serde_json::to_string_pretty(history).unwrap_or_default();
        prompt.push_str(&format!("# Previous Messages\n\n{turns}\n\n"));
    }
    prompt.push_str(&format!("# User Request\n\n{last}\n"));
    prompt
}

/// Failure texts that mean the CLI cannot work until someone fixes its login.
fn looks_like_auth_failure(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "not logged in",
        "login",
        "authenticat",
        "api key",
        "credit balance",
        "unauthorized",
        "credentials",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

#[async_trait]
impl Backend for CliBackend {
    fn source(&self) -> &'static str {
        "agent_cli"
    }

    async fn attempt(&self, call: &Call<'_>) -> Result<LlmResponse, Failure> {
        let provider = format!("cli/{}", self.spec.name);
        let system_as_argument = takes_system_argument(&self.spec, &call.request.system);
        let dir = self
            .work_dir
            .join("headless_prompts")
            .join(uuid::Uuid::new_v4().simple().to_string());
        let task = AgentTask {
            prompt: conversation(call, !system_as_argument),
            cwd: dir.clone(),
            timeout: call.timeout,
            allow_edits: false,
            log_path: dir.join("agent.log"),
        };
        let system = if system_as_argument {
            call.request.system.as_str()
        } else {
            ""
        };
        let outcome = {
            let _one_at_a_time = self.gate.lock().await;
            run_with_mode(&self.spec, &task, Mode::Completion { system }).await
        };
        if !self.keep_files {
            let _ = std::fs::remove_dir_all(&dir);
        }
        let outcome = outcome.map_err(|err| {
            Failure::fatal(LlmError::Config(format!(
                "{provider} could not be started: {err:#}"
            )))
        })?;
        let billed = Box::new(Billed {
            usage: outcome.usage,
            cost_usd: outcome.cost_usd,
            cost_kind: outcome.cost_kind,
        });
        if outcome.timed_out {
            return Err(Failure {
                error: LlmError::Transient(format!(
                    "{provider} timed out after {:.0}s",
                    outcome.duration_secs
                )),
                retryable: true,
                retry_after: None,
                billed: Some(billed),
            });
        }
        if !outcome.succeeded() {
            let message = genesis_core::fsutil::truncate(&outcome.final_message, 1000);
            let error = if looks_like_auth_failure(&message) {
                LlmError::Endpoint {
                    provider,
                    status: 0,
                    message,
                }
            } else {
                LlmError::Transient(format!(
                    "{provider} exited with {:?}: {message}",
                    outcome.exit_code
                ))
            };
            let retryable = matches!(error, LlmError::Transient(_));
            return Err(Failure {
                error,
                retryable,
                retry_after: None,
                billed: Some(billed),
            });
        }
        if outcome.final_message.trim().is_empty() {
            return Err(Failure {
                error: LlmError::Empty {
                    provider,
                    incomplete_reason: None,
                    cost_usd: outcome.cost_usd,
                },
                retryable: true,
                retry_after: None,
                billed: Some(billed),
            });
        }
        Ok(LlmResponse {
            content: outcome.final_message,
            thought: String::new(),
            model: call.request.model.clone(),
            provider,
            usage: outcome.usage,
            cost_usd: outcome.cost_usd,
            cost_kind: outcome.cost_kind,
            latency_secs: outcome.duration_secs,
            incomplete_reason: None,
            calls: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LlmRequest, Message};
    use std::time::Duration;

    fn backend(dir: &std::path::Path, script: &str) -> CliBackend {
        CliBackend {
            spec: AgentSpec {
                name: "fake".into(),
                command: Some(vec![
                    "bash".into(),
                    "-c".into(),
                    script.into(),
                    "agent".into(),
                    "{prompt_file}".into(),
                ]),
                ..AgentSpec::default()
            },
            work_dir: dir.to_path_buf(),
            keep_files: true,
            gate: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    async fn attempt(backend: &CliBackend, request: &LlmRequest) -> Result<LlmResponse, Failure> {
        backend
            .attempt(&Call {
                request,
                api_model: "fake",
                timeout: Duration::from_secs(20),
            })
            .await
    }

    #[tokio::test]
    async fn the_prompt_file_carries_system_history_and_request() {
        let dir = tempfile::tempdir().unwrap();
        let backend = backend(
            dir.path(),
            r#"cat "$1"; echo; echo '{"usage":{"input_tokens":5,"output_tokens":2,"cost":0.01}}'"#,
        );
        let mut request = LlmRequest::new("cli/fake", "Be brief.", "first");
        request.messages.push(Message::assistant("reply"));
        request.messages.push(Message::user("second"));
        let response = attempt(&backend, &request).await.unwrap();
        assert!(response
            .content
            .starts_with("# System Instructions\n\nBe brief.\n\n# Previous Messages"));
        assert!(response.content.contains("\"content\": \"reply\""));
        assert!(response.content.ends_with("# User Request\n\nsecond"));
        assert_eq!(response.provider, "cli/fake");
        assert_eq!((response.usage.input_tokens, response.cost_usd), (5, 0.01));
        assert_eq!(
            std::fs::read_dir(dir.path().join("headless_prompts"))
                .unwrap()
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn failures_are_classified_and_scratch_directories_removed() {
        let dir = tempfile::tempdir().unwrap();
        let request = LlmRequest::new("cli/fake", "", "hi");
        let mut auth = backend(
            dir.path(),
            "echo 'Not logged in. Please run /login' >&2; exit 1",
        );
        auth.keep_files = false;
        let failure = attempt(&auth, &request).await.unwrap_err();
        assert!(matches!(failure.error, LlmError::Endpoint { .. }) && !failure.retryable);
        assert_eq!(
            std::fs::read_dir(dir.path().join("headless_prompts"))
                .unwrap()
                .count(),
            0
        );

        let crash = attempt(&backend(dir.path(), "echo boom >&2; exit 2"), &request)
            .await
            .unwrap_err();
        assert!(matches!(crash.error, LlmError::Transient(_)) && crash.retryable);
        let silent = attempt(&backend(dir.path(), "true"), &request)
            .await
            .unwrap_err();
        assert!(matches!(silent.error, LlmError::Empty { .. }));
    }
}
