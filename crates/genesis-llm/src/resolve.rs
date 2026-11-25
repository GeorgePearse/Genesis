//! From a model name to the backend that serves it.
//!
//! | Spelling | Backend |
//! |---|---|
//! | `mock/perturb`, `mock/echo` | built-in offline models |
//! | `cli/<agent>[@model][?effort=..]`, `headless/<agent>…` | a coding-agent CLI answering one prompt |
//! | `openrouter/<id>` | OpenRouter |
//! | `local/<model>@http(s)://host/v1[?api_key_env=NAME]` | any OpenAI-compatible server |
//! | `azure-<deployment>` | Azure OpenAI |
//! | `claude-*`, `anthropic/<model>` | Anthropic |
//! | `gpt-*`, `o<digit>*`, `chatgpt-*`, `openai/<model>` | OpenAI |
//! | `gemini-*`, `google/<model>` | Gemini API |
//! | `deepseek-*` | DeepSeek |

use crate::agent::{agent_available, AgentSpec};
use crate::types::LlmError;

/// Where a model name leads.
#[derive(Debug, Clone, PartialEq)]
pub enum Route {
    Mock {
        behaviour: String,
    },
    Cli {
        spec: AgentSpec,
    },
    OpenAi {
        model: String,
    },
    Azure {
        deployment: String,
    },
    OpenRouter {
        model: String,
    },
    DeepSeek {
        model: String,
    },
    Local {
        model: String,
        base_url: String,
        api_key_env: Option<String>,
    },
    Anthropic {
        model: String,
    },
    Gemini {
        model: String,
    },
}

impl Route {
    /// Name of the provider, as it appears in errors and the ledger.
    pub fn provider(&self) -> &'static str {
        match self {
            Self::Mock { .. } => "mock",
            Self::Cli { .. } => "cli",
            Self::OpenAi { .. } => "openai",
            Self::Azure { .. } => "azure_openai",
            Self::OpenRouter { .. } => "openrouter",
            Self::DeepSeek { .. } => "deepseek",
            Self::Local { .. } => "local_openai",
            Self::Anthropic { .. } => "anthropic",
            Self::Gemini { .. } => "google",
        }
    }

    /// The model name the backend's API expects.
    pub fn api_model(&self) -> &str {
        match self {
            Self::Mock { behaviour } => behaviour,
            Self::Cli { spec } => spec.model.as_deref().unwrap_or(""),
            Self::OpenAi { model }
            | Self::OpenRouter { model }
            | Self::DeepSeek { model }
            | Self::Local { model, .. }
            | Self::Anthropic { model }
            | Self::Gemini { model } => model,
            Self::Azure { deployment } => deployment,
        }
    }
}

const ACCEPTED: &str = "accepted model names: mock/<behaviour>, cli/<agent>[@model][?effort=level], \
openrouter/<id>, local/<model>@http(s)://host/v1[?api_key_env=NAME], azure-<deployment>, \
claude-* or anthropic/<model>, gpt-*, o<digit>* or openai/<model>, gemini-* or google/<model>, deepseek-*";

fn valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn local(rest: &str, name: &str) -> Result<Route, LlmError> {
    let malformed = || {
        LlmError::Config(format!(
            "`{name}` is not of the form local/<model>@http(s)://host/v1[?api_key_env=NAME]"
        ))
    };
    let (model, url) = rest.split_once('@').ok_or_else(malformed)?;
    if model.is_empty() || !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(malformed());
    }
    let (base_url, api_key_env) = match url.split_once("?api_key_env=") {
        Some((base, env)) if valid_env_name(env) => (base, Some(env.to_string())),
        Some(_) => return Err(malformed()),
        None => (url, None),
    };
    Ok(Route::Local {
        model: model.to_string(),
        base_url: base_url.trim_end_matches('/').to_string(),
        api_key_env,
    })
}

fn is_o_series(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next() == Some('o') && chars.next().is_some_and(|c| c.is_ascii_digit())
}

/// Decide which backend serves `name`.
pub fn resolve_model(name: &str) -> Result<Route, LlmError> {
    let name = name.trim();
    let nonempty = |model: &str, route: fn(String) -> Route| {
        if model.is_empty() {
            Err(LlmError::Config(format!(
                "model name `{name}` is missing the model part; {ACCEPTED}"
            )))
        } else {
            Ok(route(model.to_string()))
        }
    };
    if let Some(behaviour) = name.strip_prefix("mock/") {
        return nonempty(behaviour, |behaviour| Route::Mock { behaviour });
    }
    if let Some(reference) = name
        .strip_prefix("cli/")
        .or_else(|| name.strip_prefix("headless/"))
    {
        return AgentSpec::parse_reference(reference)
            .map(|spec| Route::Cli { spec })
            .ok_or_else(|| {
                LlmError::Config(format!(
                    "`{name}` is not of the form cli/<agent>[@model][?effort=level]"
                ))
            });
    }
    if let Some(model) = name.strip_prefix("openrouter/") {
        return nonempty(model, |model| Route::OpenRouter { model });
    }
    if let Some(rest) = name.strip_prefix("local/") {
        return local(rest, name);
    }
    if let Some(deployment) = name.strip_prefix("azure-") {
        return nonempty(deployment, |deployment| Route::Azure { deployment });
    }
    if let Some(model) = name.strip_prefix("anthropic/") {
        return nonempty(model, |model| Route::Anthropic { model });
    }
    if let Some(model) = name.strip_prefix("openai/") {
        return nonempty(model, |model| Route::OpenAi { model });
    }
    if let Some(model) = name.strip_prefix("google/") {
        return nonempty(model, |model| Route::Gemini { model });
    }
    if name.starts_with("claude-") {
        return Ok(Route::Anthropic {
            model: name.to_string(),
        });
    }
    if name.starts_with("gpt-") || name.starts_with("chatgpt-") || is_o_series(name) {
        return Ok(Route::OpenAi {
            model: name.to_string(),
        });
    }
    if name.starts_with("gemini-") {
        return Ok(Route::Gemini {
            model: name.to_string(),
        });
    }
    if name.starts_with("deepseek-") {
        return Ok(Route::DeepSeek {
            model: name.to_string(),
        });
    }
    Err(LlmError::Config(format!(
        "unknown model `{name}`; {ACCEPTED}"
    )))
}

fn env_set(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| !value.trim().is_empty())
}

/// Sets of environment variables, any one of which makes the route usable.
fn credential_sets(route: &Route) -> Vec<Vec<String>> {
    let sets: &[&[&str]] = match route {
        Route::Mock { .. } | Route::Cli { .. } => &[],
        Route::OpenAi { .. } => &[&["OPENAI_API_KEY"]],
        Route::Azure { .. } => &[&["AZURE_OPENAI_API_KEY", "AZURE_API_ENDPOINT"]],
        Route::OpenRouter { .. } => &[&["OPENROUTER_API_KEY"]],
        Route::DeepSeek { .. } => &[&["DEEPSEEK_API_KEY"]],
        Route::Anthropic { .. } => &[&["ANTHROPIC_API_KEY"], &["ANTHROPIC_AUTH_TOKEN"]],
        Route::Gemini { .. } => &[&["GEMINI_API_KEY"], &["GOOGLE_API_KEY"]],
        Route::Local { api_key_env, .. } => {
            return api_key_env.iter().map(|name| vec![name.clone()]).collect();
        }
    };
    sets.iter()
        .map(|set| set.iter().map(|name| name.to_string()).collect())
        .collect()
}

/// One sentence per model that cannot be used as the environment stands:
/// an unknown name, a missing API key, or an agent CLI that is not installed.
/// Empty when every model is usable. Nothing is sent over the network.
pub fn missing_credentials(models: &[String]) -> Vec<String> {
    let mut problems = Vec::new();
    for model in models {
        let route = match resolve_model(model) {
            Ok(route) => route,
            Err(err) => {
                problems.push(err.to_string());
                continue;
            }
        };
        if let Route::Cli { spec } = &route {
            if !agent_available(spec) {
                problems.push(format!(
                    "`{model}` needs the `{}` command-line agent, which is not on PATH",
                    spec.name
                ));
            }
            continue;
        }
        let sets = credential_sets(&route);
        if sets.is_empty() || sets.iter().any(|set| set.iter().all(|name| env_set(name))) {
            continue;
        }
        let alternatives: Vec<String> = sets.iter().map(|set| set.join(" and ")).collect();
        problems.push(format!(
            "`{model}` needs {} to be set",
            alternatives.join(", or ")
        ));
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_documented_spelling_resolves() {
        let cases = [
            ("mock/perturb", "mock", "perturb"),
            (
                "openrouter/qwen/qwen3-coder",
                "openrouter",
                "qwen/qwen3-coder",
            ),
            ("azure-gpt-4.1", "azure_openai", "gpt-4.1"),
            ("claude-opus-5-5", "anthropic", "claude-opus-5-5"),
            (
                "anthropic/claude-haiku-5-5",
                "anthropic",
                "claude-haiku-5-5",
            ),
            ("gpt-4.1-mini", "openai", "gpt-4.1-mini"),
            ("o4-mini", "openai", "o4-mini"),
            ("openai/gpt-6-luna", "openai", "gpt-6-luna"),
            ("gemini-3.5-flash", "google", "gemini-3.5-flash"),
            ("google/gemini-4-argon", "google", "gemini-4-argon"),
            ("deepseek-v4-pro", "deepseek", "deepseek-v4-pro"),
            ("  gpt-5  ", "openai", "gpt-5"),
        ];
        for (name, provider, api_model) in cases {
            let route = resolve_model(name).unwrap();
            assert_eq!(
                (route.provider(), route.api_model()),
                (provider, api_model),
                "{name}"
            );
        }
    }

    #[test]
    fn cli_and_headless_spellings_carry_the_agent_spec() {
        let route = resolve_model("cli/codex@gpt-5.6-terra?effort=high").unwrap();
        let Route::Cli { spec } = &route else {
            panic!("{route:?}")
        };
        assert_eq!(
            (
                spec.name.as_str(),
                spec.model.as_deref(),
                spec.reasoning_effort.as_deref()
            ),
            ("codex", Some("gpt-5.6-terra"), Some("high"))
        );
        assert_eq!(
            resolve_model("headless/claude").unwrap(),
            Route::Cli {
                spec: AgentSpec {
                    name: "claude".into(),
                    ..AgentSpec::default()
                }
            }
        );
        assert!(resolve_model("cli/").is_err() && resolve_model("cli/claude?speed=2").is_err());
    }

    #[test]
    fn local_endpoints_parse_their_url_and_key_variable() {
        assert_eq!(
            resolve_model("local/qwen3@http://localhost:8000/v1/").unwrap(),
            Route::Local {
                model: "qwen3".into(),
                base_url: "http://localhost:8000/v1".into(),
                api_key_env: None
            }
        );
        assert_eq!(
            resolve_model("local/m@https://host/v1?api_key_env=MY_KEY").unwrap(),
            Route::Local {
                model: "m".into(),
                base_url: "https://host/v1".into(),
                api_key_env: Some("MY_KEY".into())
            }
        );
        for bad in [
            "local/m",
            "local/@http://x",
            "local/m@ftp://x",
            "local/m@http://x?api_key_env=9bad",
        ] {
            assert!(resolve_model(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn unknown_names_list_the_accepted_spellings() {
        for bad in ["llama3", "", "openrouter/", "mock/", "azure-"] {
            let err = resolve_model(bad).unwrap_err();
            assert!(err.is_endpoint(), "{bad}");
            assert!(err.to_string().contains("accepted model names"), "{err}");
        }
    }

    #[test]
    fn missing_credentials_names_the_variables_without_reading_secrets() {
        let unusable = missing_credentials(&[
            "mock/perturb".to_string(),
            "local/m@http://localhost:1/v1".to_string(),
            "local/m@http://localhost:1/v1?api_key_env=GENESIS_TEST_ABSENT_KEY".to_string(),
            "nonsense".to_string(),
            "cli/emacs".to_string(),
        ]);
        assert_eq!(unusable.len(), 3, "{unusable:?}");
        assert!(unusable[0].contains("GENESIS_TEST_ABSENT_KEY"));
        assert!(unusable[1].contains("unknown model"));
        assert!(unusable[2].contains("not on PATH"));
    }
}
