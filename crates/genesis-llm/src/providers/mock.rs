//! Backends that need no network: a scriptable test double, and the built-in
//! `mock/*` models that let a whole evolution run offline.

use super::{Backend, Call, Failure};
use crate::extract::fences;
use crate::types::{LlmError, LlmRequest, LlmResponse, Provider, Usage};
use async_trait::async_trait;
use genesis_core::CostKind;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};

type Handler = dyn Fn(&LlmRequest) -> Result<String, LlmError> + Send + Sync;

/// A [`Provider`] driven by a closure, recording every request it sees.
///
/// ```
/// use genesis_llm::{LlmClient, LlmRequest, MockProvider};
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let mock = MockProvider::new(|request| Ok(format!("stage={}", request.stage)));
/// let client = LlmClient::mock(mock.clone());
/// let reply = client.query("idea", LlmRequest::new("any-model", "", "hi")).await.unwrap();
/// assert_eq!(reply.content, "stage=idea");
/// assert_eq!(mock.calls().len(), 1);
/// # });
/// ```
pub struct MockProvider {
    handler: Box<Handler>,
    calls: Mutex<Vec<LlmRequest>>,
    cost_per_call: f64,
}

impl MockProvider {
    pub fn new(
        handler: impl Fn(&LlmRequest) -> Result<String, LlmError> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Self::costing(0.0, handler)
    }

    /// Like [`MockProvider::new`], but every reply is billed `cost_per_call`
    /// dollars, for exercising budgets.
    pub fn costing(
        cost_per_call: f64,
        handler: impl Fn(&LlmRequest) -> Result<String, LlmError> + Send + Sync + 'static,
    ) -> Arc<Self> {
        Arc::new(Self {
            handler: Box::new(handler),
            calls: Mutex::new(Vec::new()),
            cost_per_call,
        })
    }

    /// Replies with `responses` in order, starting over when they run out.
    pub fn scripted(responses: Vec<String>) -> Arc<Self> {
        let next = Mutex::new(0usize);
        Self::new(move |_| {
            if responses.is_empty() {
                return Ok(String::new());
            }
            let mut index = next.lock().unwrap_or_else(|p| p.into_inner());
            let reply = responses[*index % responses.len()].clone();
            *index += 1;
            Ok(reply)
        })
    }

    /// Every request received so far, in order.
    pub fn calls(&self) -> Vec<LlmRequest> {
        self.calls.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

fn estimated_tokens(chars: usize) -> u64 {
    chars.div_ceil(4) as u64
}

fn estimated_usage(request: &LlmRequest, reply: &str) -> Usage {
    let prompt_chars = request.system.len()
        + request
            .messages
            .iter()
            .map(|m| m.content.len())
            .sum::<usize>();
    Usage {
        input_tokens: estimated_tokens(prompt_chars),
        output_tokens: estimated_tokens(reply.len()),
        ..Usage::default()
    }
}

fn offline_response(request: &LlmRequest, content: String, cost_usd: f64) -> LlmResponse {
    LlmResponse {
        usage: estimated_usage(request, &content),
        content,
        thought: String::new(),
        model: request.model.clone(),
        provider: "mock".to_string(),
        cost_usd,
        cost_kind: if cost_usd > 0.0 {
            CostKind::Estimated
        } else {
            CostKind::UsageOnly
        },
        latency_secs: 0.0,
        incomplete_reason: None,
        calls: 1,
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    async fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        self.calls
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(request.clone());
        let content = (self.handler)(request)?;
        if content.trim().is_empty() {
            return Err(LlmError::Empty {
                provider: "mock".to_string(),
                incomplete_reason: None,
                cost_usd: self.cost_per_call,
            });
        }
        Ok(offline_response(request, content, self.cost_per_call))
    }
}

/// The `mock/<behaviour>` model family.
pub(crate) struct BuiltinMock {
    pub behaviour: String,
}

#[async_trait]
impl Backend for BuiltinMock {
    async fn attempt(&self, call: &Call<'_>) -> Result<LlmResponse, Failure> {
        let content = match self.behaviour.as_str() {
            "echo" => call.request.last_user().to_string(),
            "perturb" => perturb_reply(call.request),
            other => {
                return Err(Failure::fatal(LlmError::Config(format!(
                    "unknown mock model `mock/{other}`; the built-in ones are mock/perturb and mock/echo"
                ))))
            }
        };
        if content.trim().is_empty() {
            return Err(Failure::fatal(LlmError::Empty {
                provider: "mock".to_string(),
                incomplete_reason: None,
                cost_usd: 0.0,
            }));
        }
        Ok(offline_response(call.request, content, 0.0))
    }
}

struct Literal {
    start: usize,
    end: usize,
    is_float: bool,
}

/// Plain decimal literals on a line: `12`, `0.35`, `1e-3`. Anything glued to
/// an identifier or carrying a suffix (`x1`, `0x1F`, `10usize`, `1f32`) is
/// left alone, as is a number that only looks fractional (`0..n`, `1.max(2)`).
fn numeric_literals(line: &str) -> Vec<Literal> {
    let bytes = line.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let glued =
            i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || matches!(bytes[i - 1], b'_' | b'.'));
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let mut is_float = false;
        if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
            is_float = true;
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < bytes.len() && matches!(bytes[i], b'e' | b'E') {
            let mut j = i + 1;
            if j < bytes.len() && matches!(bytes[j], b'+' | b'-') {
                j += 1;
            }
            if j < bytes.len() && bytes[j].is_ascii_digit() {
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                is_float = true;
                i = j;
            }
        }
        let suffixed = i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_');
        if suffixed {
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            continue;
        }
        if !glued {
            found.push(Literal {
                start,
                end: i,
                is_float,
            });
        }
    }
    found
}

fn perturbed(literal: &str, is_float: bool, rng: &mut impl Rng) -> String {
    let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
    if !is_float {
        let value: i64 = literal.parse().unwrap_or(0);
        let step = (value.abs() / 10).max(1) * if sign > 0.0 { 1 } else { -1 };
        return (value + step).max(0).to_string();
    }
    let value: f64 = literal.parse().unwrap_or(0.0);
    let relative = sign * rng.gen_range(0.05..0.3);
    let nudged = if value == 0.0 {
        relative
    } else {
        value * (1.0 + relative)
    };
    if literal.contains(['e', 'E']) {
        return format!("{nudged:e}");
    }
    let decimals = literal.split('.').nth(1).map_or(1, str::len).max(3);
    let text = format!("{nudged:.decimals$}");
    let trimmed = text.trim_end_matches('0');
    if trimmed.ends_with('.') {
        format!("{trimmed}0")
    } else {
        trimmed.to_string()
    }
}

fn is_comment(line: &str) -> bool {
    let trimmed = line.trim_start();
    ["#", "//", "!", "<!--", "(*", "--", "%"]
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
}

/// The program to mutate: the fenced block under "# Current program", else
/// the first one holding an evolve block, else the first fenced block.
fn current_program(prompt: &str) -> Option<(String, String)> {
    let blocks = fences(prompt);
    let heading = prompt.find("# Current program");
    let chosen = heading
        .and_then(|at| blocks.iter().find(|block| block.start > at))
        .or_else(|| {
            blocks
                .iter()
                .find(|block| block.body.contains("EVOLVE-BLOCK-START"))
        })
        .or_else(|| blocks.first())?;
    let code = chosen.body.strip_suffix('\n').unwrap_or(chosen.body);
    Some((chosen.info.to_string(), code.to_string()))
}

/// Line edits `(index, old, new)` nudging one or two constants in the evolve
/// block, chosen by `rng`.
fn mutate(code: &str, rng: &mut impl Rng) -> Vec<(usize, String, String)> {
    let lines: Vec<&str> = code.split('\n').collect();
    let has_markers = code.contains("EVOLVE-BLOCK-START");
    let mut mutable = !has_markers;
    let mut candidates: Vec<(usize, Literal)> = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if line.contains("EVOLVE-BLOCK-START") {
            mutable = true;
            continue;
        }
        if line.contains("EVOLVE-BLOCK-END") {
            mutable = false;
            continue;
        }
        if mutable && !is_comment(line) {
            candidates.extend(
                numeric_literals(line)
                    .into_iter()
                    .map(|literal| (index, literal)),
            );
        }
    }
    let floats = candidates
        .iter()
        .filter(|(_, literal)| literal.is_float)
        .count();
    if floats > 0 && rng.gen_bool(0.8) {
        candidates.retain(|(_, literal)| literal.is_float);
    }
    if candidates.is_empty() {
        return Vec::new();
    }
    let wanted = rng.gen_range(1..=2usize).min(candidates.len());
    let mut picked: Vec<(usize, Literal)> = Vec::new();
    while picked.len() < wanted {
        let (index, literal) = candidates.swap_remove(rng.gen_range(0..candidates.len()));
        // Two edits on one line would have to be merged; one per line is enough.
        if picked.iter().all(|(taken, _)| *taken != index) {
            picked.push((index, literal));
        }
        if candidates.is_empty() {
            break;
        }
    }
    picked
        .into_iter()
        .filter_map(|(index, literal)| {
            let line = lines[index];
            let replacement = perturbed(&line[literal.start..literal.end], literal.is_float, rng);
            let new_line = format!(
                "{}{}{}",
                &line[..literal.start],
                replacement,
                &line[literal.end..]
            );
            (new_line != line).then(|| (index, line.to_string(), new_line))
        })
        .collect()
}

fn seed_of(request: &LlmRequest) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(request.system.as_bytes());
    for message in &request.messages {
        hasher.update([0u8]);
        hasher.update(message.content.as_bytes());
    }
    hasher.update(request.temperature.unwrap_or(0.0).to_bits().to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("a sha256 digest has 32 bytes"),
    )
}

/// A deterministic stand-in for a mutation model: nudge numeric constants in
/// the evolve block of the program the prompt shows, and answer in whichever
/// patch format the system prompt asks for.
fn perturb_reply(request: &LlmRequest) -> String {
    if request.system.contains("NOT_NOVEL") {
        return "NOVEL: the offline mock judge accepts every proposal.".to_string();
    }
    let Some((language, code)) = current_program(request.last_user()) else {
        return "mock/perturb found no fenced program in the prompt.".to_string();
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed_of(request));
    let edits = mutate(&code, &mut rng);
    let header = format!(
        "<NAME>\nperturb_constants\n</NAME>\n\n<DESCRIPTION>\nOffline mock mutation: nudged {} numeric constant(s) inside the evolve block.\n</DESCRIPTION>\n\n",
        edits.len()
    );
    if request.system.contains("<<<<<<< SEARCH") {
        let blocks: String = edits
            .iter()
            .map(|(_, old, new)| {
                format!("<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n\n")
            })
            .collect();
        return format!("{header}<DIFF>\n{blocks}</DIFF>\n");
    }
    let mut lines: Vec<String> = code.split('\n').map(str::to_string).collect();
    for (index, _, new) in &edits {
        lines[*index] = new.clone();
    }
    let rewritten = lines.join("\n");
    if request.system.contains("<CODE>") {
        format!("{header}<CODE>\n```{language}\n{rewritten}\n```\n</CODE>\n")
    } else {
        format!("```{language}\n{rewritten}\n```\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{extract_fenced, extract_tag};

    const PROGRAM: &str = "import math\nSCALE = 2\n# EVOLVE-BLOCK-START\ndef radius(i):\n    # tuned by hand 0.99\n    step = 0.35\n    count = 26\n    return step * math.sqrt(i + 1) / count\n# EVOLVE-BLOCK-END\nprint(radius(3) * 1.5)";

    fn request(system: &str, temperature: f64) -> LlmRequest {
        let user =
            format!("# Current program\n\nHere it is:\n\n```python\n{PROGRAM}\n```\n\nImprove it.");
        LlmRequest::new("mock/perturb", system, user).temperature(temperature)
    }

    fn literals(line: &str) -> Vec<(&str, bool)> {
        numeric_literals(line)
            .into_iter()
            .map(|l| (&line[l.start..l.end], l.is_float))
            .collect()
    }

    #[test]
    fn literal_scanner_skips_identifiers_suffixes_and_ranges() {
        assert_eq!(
            literals("x1 = 12 + 0.35 * 1e-3"),
            [("12", false), ("0.35", true), ("1e-3", true)]
        );
        assert_eq!(
            literals("for i in 0..n { a[i] = 1.max(2) }"),
            [("0", false), ("1", false), ("2", false)]
        );
        assert_eq!(literals("let k = 10usize + 0x1F + 2f32 + v.0;"), []);
        assert_eq!(literals("t = 3.0e2"), [("3.0e2", true)]);
    }

    #[test]
    fn full_rewrite_changes_only_constants_inside_the_evolve_block() {
        let reply = perturb_reply(&request(
            "Respond with <NAME>, <DESCRIPTION> and <CODE>.",
            0.5,
        ));
        assert_eq!(
            extract_tag(&reply, "NAME").as_deref(),
            Some("perturb_constants")
        );
        let code = extract_fenced(&extract_tag(&reply, "CODE").unwrap(), &["python"]).unwrap();
        assert_ne!(code, PROGRAM, "the mock must change something");
        let (old, new): (Vec<&str>, Vec<&str>) =
            (PROGRAM.lines().collect(), code.lines().collect());
        assert_eq!(old.len(), new.len());
        for (index, (before, after)) in old.iter().zip(&new).enumerate() {
            if before != after {
                assert!(
                    (5..=7).contains(&index),
                    "line {index} is outside the evolve block or a comment"
                );
            }
        }
        assert!(
            code.contains("SCALE = 2")
                && code.contains("# tuned by hand 0.99")
                && code.ends_with("* 1.5)")
        );
    }

    #[test]
    fn diff_format_quotes_original_lines_verbatim() {
        let reply = perturb_reply(&request("Use <<<<<<< SEARCH blocks inside <DIFF>.", 0.0));
        let diff = extract_tag(&reply, "DIFF").unwrap();
        let search = diff
            .split("<<<<<<< SEARCH\n")
            .nth(1)
            .unwrap()
            .split("\n=======")
            .next()
            .unwrap();
        assert!(
            PROGRAM.lines().any(|line| line == search),
            "SEARCH must be an original line: {search:?}"
        );
        assert!(diff.contains(">>>>>>> REPLACE"));
    }

    #[test]
    fn replies_are_deterministic_and_vary_with_the_prompt() {
        let system = "Reply with <CODE>.";
        assert_eq!(
            perturb_reply(&request(system, 0.5)),
            perturb_reply(&request(system, 0.5))
        );
        let variants: std::collections::BTreeSet<String> = [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .map(|t| perturb_reply(&request(system, *t)))
            .collect();
        assert!(variants.len() > 1);
    }

    #[test]
    fn the_novelty_judge_prompt_and_programless_prompts_get_plain_answers() {
        let judge = LlmRequest::new(
            "mock/perturb",
            "Respond with NOVEL or NOT_NOVEL.",
            "compare",
        );
        assert!(perturb_reply(&judge).starts_with("NOVEL"));
        let empty = LlmRequest::new("mock/perturb", "<CODE>", "no code here");
        assert!(perturb_reply(&empty).contains("no fenced program"));
    }

    #[test]
    fn perturbed_numbers_stay_parseable() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for literal in ["0.35", "2.0", "0.0", "1e-3", "100.125"] {
            for _ in 0..50 {
                let out = perturbed(literal, true, &mut rng);
                assert!(out.parse::<f64>().is_ok(), "{literal} -> {out}");
                assert_ne!(out, literal);
            }
        }
        for literal in ["0", "7", "260"] {
            for _ in 0..50 {
                let out: i64 = perturbed(literal, false, &mut rng).parse().unwrap();
                assert!(out >= 0);
            }
        }
    }

    #[tokio::test]
    async fn scripted_mock_cycles_and_records_calls() {
        let mock = MockProvider::scripted(vec!["a".into(), "b".into()]);
        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.push(
                mock.complete(&LlmRequest::new("m", "s", "u"))
                    .await
                    .unwrap()
                    .content,
            );
        }
        assert_eq!(seen, ["a", "b", "a"]);
        assert_eq!(mock.calls().len(), 3);
        let reply = mock
            .complete(&LlmRequest::new("m", "sys!", "12345678"))
            .await
            .unwrap();
        assert_eq!(
            (reply.usage.input_tokens, reply.usage.output_tokens),
            (3, 1)
        );
        assert_eq!(
            (reply.cost_usd, reply.cost_kind),
            (0.0, CostKind::UsageOnly)
        );
        let empty = MockProvider::new(|_| Ok("  ".into()));
        assert!(matches!(
            empty.complete(&LlmRequest::default()).await,
            Err(LlmError::Empty { .. })
        ));
    }
}
