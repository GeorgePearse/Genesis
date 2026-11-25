//! Per-call sampling settings: each request draws its temperature and
//! reasoning effort from the configured choices, as ShinkaEvolve's
//! `llm_kwargs` does.

use crate::types::LlmRequest;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SamplingKwargs {
    /// One is drawn per call. Empty leaves the temperature to the backend.
    pub temperatures: Vec<f64>,
    pub max_tokens: u32,
    /// One is drawn per call; see [`LlmRequest::reasoning_effort`].
    pub reasoning_efforts: Vec<String>,
}

impl Default for SamplingKwargs {
    fn default() -> Self {
        Self {
            temperatures: vec![0.0, 0.5, 1.0],
            max_tokens: 16_384,
            reasoning_efforts: vec!["disabled".to_string()],
        }
    }
}

impl SamplingKwargs {
    /// Set `temperature`, `max_tokens` and `reasoning_effort` on `request`.
    pub fn apply(&self, request: &mut LlmRequest, rng: &mut dyn rand::RngCore) {
        request.temperature = match self.temperatures.len() {
            0 => None,
            n => Some(self.temperatures[rng.gen_range(0..n)]),
        };
        request.max_tokens = Some(self.max_tokens);
        request.reasoning_effort = match self.reasoning_efforts.len() {
            0 => None,
            n => Some(self.reasoning_efforts[rng.gen_range(0..n)].clone()),
        };
    }
}

/// The canonical spelling of a reasoning effort, or `None` when the caller
/// asked for no reasoning (`disabled`, `none`, `auto`, empty, or unset).
pub(crate) fn enabled_effort(effort: Option<&str>) -> Option<&'static str> {
    match effort.map(|e| e.trim().to_ascii_lowercase()).as_deref() {
        Some("min") | Some("minimal") => Some("minimal"),
        Some("low") => Some("low"),
        Some("medium") => Some("medium"),
        Some("high") => Some("high"),
        Some("xhigh") => Some("xhigh"),
        Some("max") => Some("max"),
        _ => None,
    }
}

/// Thinking-token budgets for backends that take a number rather than a
/// level (ShinkaEvolve's table).
pub(crate) fn thinking_budget(effort: &str) -> u32 {
    match effort {
        "minimal" => 1024,
        "low" => 2048,
        "medium" => 4096,
        "high" => 8192,
        "xhigh" => 12_288,
        _ => 16_384,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn apply_draws_from_the_configured_choices() {
        let kwargs = SamplingKwargs {
            temperatures: vec![0.2, 0.9],
            max_tokens: 777,
            reasoning_efforts: vec!["low".into(), "high".into()],
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut seen_temperatures = std::collections::BTreeSet::new();
        for _ in 0..50 {
            let mut request = LlmRequest::new("m", "s", "u");
            kwargs.apply(&mut request, &mut rng);
            assert_eq!(request.max_tokens, Some(777));
            let temperature = request.temperature.unwrap();
            assert!(kwargs.temperatures.contains(&temperature));
            assert!(kwargs
                .reasoning_efforts
                .contains(&request.reasoning_effort.unwrap()));
            seen_temperatures.insert(temperature.to_bits());
        }
        assert_eq!(seen_temperatures.len(), 2);
    }

    #[test]
    fn empty_choices_leave_the_backend_default() {
        let kwargs = SamplingKwargs {
            temperatures: vec![],
            max_tokens: 5,
            reasoning_efforts: vec![],
        };
        let mut request = LlmRequest::new("m", "s", "u")
            .temperature(0.3)
            .reasoning_effort("high");
        kwargs.apply(&mut request, &mut rand::rngs::StdRng::seed_from_u64(0));
        assert_eq!(
            (request.temperature, request.reasoning_effort),
            (None, None)
        );
    }

    #[test]
    fn effort_spellings_normalise() {
        assert_eq!(enabled_effort(Some(" MIN ")), Some("minimal"));
        assert_eq!(enabled_effort(Some("xhigh")), Some("xhigh"));
        for off in [
            None,
            Some("disabled"),
            Some("none"),
            Some("auto"),
            Some(""),
            Some("bogus"),
        ] {
            assert_eq!(enabled_effort(off), None);
        }
        assert!(thinking_budget("low") < thinking_budget("max"));
    }
}
