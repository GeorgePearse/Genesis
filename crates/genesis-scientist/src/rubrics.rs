//! Rubric verification by the harness.
//!
//! The model that proposed an idea does not get to say whether its rubrics
//! were met. Metric rubrics are compared with the evaluator's numbers;
//! implementation rubrics that carry `source_contains` patterns are checked
//! against the candidate's code tokens; anything else is reported as not
//! assessed unless a judge model is configured. When the evaluation did not
//! establish correctness, no rubric is assessed at all.

use genesis_core::idea::{IdeaBundle, Rubric, RubricMetrics, RubricResult, RubricStatus};
use genesis_core::EvalOutcome;
use genesis_llm::{LlmClient, LlmRequest, SamplingKwargs};
use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

fn line_comment_prefix(language: &str) -> &'static str {
    match language {
        "python" | "julia" | "bash" | "ruby" | "r" => "#",
        "fortran" => "!",
        _ => "//",
    }
}

/// Code tokens with comments and whitespace removed. A string literal is one
/// token, quotes included.
pub fn tokenize(code: &str, language: &str) -> Vec<String> {
    let prefix: Vec<char> = line_comment_prefix(language).chars().collect();
    let block_comments = prefix == ['/', '/'];
    let chars: Vec<char> = code.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if chars[i..].starts_with(&prefix) {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if block_comments && chars[i..].starts_with(&['/', '*']) {
            i += 2;
            while i < chars.len() && !chars[i..].starts_with(&['*', '/']) {
                i += 1;
            }
            i = (i + 2).min(chars.len());
        } else if c == '"' || c == '\'' {
            let triple = chars[i..].starts_with(&[c, c, c]);
            let quote_len = if triple { 3 } else { 1 };
            let start = i;
            i += quote_len;
            loop {
                if i >= chars.len() {
                    break;
                }
                if chars[i] == '\\' {
                    i += 2;
                    continue;
                }
                let closes = if triple {
                    chars[i..].starts_with(&[c, c, c])
                } else {
                    chars[i] == c
                };
                if closes {
                    i += quote_len;
                    break;
                }
                if !triple && chars[i] == '\n' {
                    break;
                }
                i += 1;
            }
            let end = i.min(chars.len());
            tokens.push(chars[start..end].iter().collect());
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
        } else if c.is_ascii_digit() {
            let start = i;
            while i < chars.len()
                && (chars[i].is_alphanumeric() || chars[i] == '.' || chars[i] == '_')
            {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
        } else {
            tokens.push(c.to_string());
            i += 1;
        }
    }
    tokens
}

/// True when the tokens of `pattern` occur, in order and adjacent, among the
/// tokens of `code`. Comments and layout never count.
pub fn source_contains(code: &str, language: &str, pattern: &str) -> bool {
    let needle = tokenize(pattern, language);
    if needle.is_empty() {
        return false;
    }
    let haystack = tokenize(code, language);
    haystack
        .windows(needle.len())
        .any(|window| window == needle.as_slice())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Comparison {
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
}

impl Comparison {
    fn parse(text: &str) -> Option<Self> {
        Some(match text.trim().to_ascii_lowercase().as_str() {
            ">" | "gt" | "greater_than" => Self::Greater,
            ">=" | "ge" | "gte" | "at_least" => Self::GreaterEqual,
            "<" | "lt" | "less_than" => Self::Less,
            "<=" | "le" | "lte" | "at_most" => Self::LessEqual,
            "==" | "=" | "eq" => Self::Equal,
            _ => return None,
        })
    }

    fn holds(self, value: f64, threshold: f64) -> bool {
        match self {
            Self::Greater => value > threshold,
            Self::GreaterEqual => value >= threshold,
            Self::Less => value < threshold,
            Self::LessEqual => value <= threshold,
            Self::Equal => (value - threshold).abs() <= 1e-9 * (1.0 + threshold.abs()),
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Equal => "==",
        }
    }
}

const NUMBER: &str = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?";

/// What a metric rubric asks for: `metric comparison threshold`. Structured
/// fields win; otherwise the description is read for `name >= 2.1` or for a
/// phrase such as "exceeds 2.1", which applies to `combined_score`.
pub fn parse_metric_rubric(rubric: &Rubric) -> Option<(String, Comparison, f64)> {
    if let (Some(threshold), Some(comparison)) = (
        rubric.threshold,
        rubric.operator.as_deref().and_then(Comparison::parse),
    ) {
        let metric = rubric
            .metric
            .clone()
            .unwrap_or_else(|| "combined_score".to_string());
        return Some((metric, comparison, threshold));
    }
    static SYMBOLIC: OnceLock<Regex> = OnceLock::new();
    static PHRASE: OnceLock<Regex> = OnceLock::new();
    let symbolic = SYMBOLIC.get_or_init(|| {
        Regex::new(&format!(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|>|<)\s*({NUMBER})"
        ))
        .expect("valid regex")
    });
    if let Some(found) = symbolic.captures(&rubric.description) {
        let threshold: f64 = found[3].parse().ok()?;
        return Some((
            found[1].to_string(),
            Comparison::parse(&found[2])?,
            threshold,
        ));
    }
    let phrase = PHRASE.get_or_init(|| {
        Regex::new(&format!(
            r"(?i)\b(exceeds?|greater than|more than|above|at least|no less than|less than|below|under|at most|no more than)\s+({NUMBER})"
        ))
        .expect("valid regex")
    });
    let found = phrase.captures(&rubric.description)?;
    let threshold: f64 = found[2].parse().ok()?;
    let comparison = match found[1].to_ascii_lowercase().as_str() {
        "at least" | "no less than" => Comparison::GreaterEqual,
        "less than" | "below" | "under" => Comparison::Less,
        "at most" | "no more than" => Comparison::LessEqual,
        _ => Comparison::Greater,
    };
    Some((
        rubric
            .metric
            .clone()
            .unwrap_or_else(|| "combined_score".to_string()),
        comparison,
        threshold,
    ))
}

fn metric_value(outcome: &EvalOutcome, name: &str) -> Option<f64> {
    if name == "combined_score" {
        return Some(outcome.combined_score);
    }
    outcome.public_metrics.get(name).and_then(Value::as_f64)
}

fn result(rubric: &Rubric, pass: Option<bool>, evidence: String) -> RubricResult {
    RubricResult {
        id: rubric.id.clone(),
        pass: pass.unwrap_or(false),
        evidence,
        assessment: match pass {
            Some(true) => "passed",
            Some(false) => "failed",
            None => "not_assessed",
        }
        .to_string(),
    }
}

/// Check every rubric that can be checked without a model.
pub fn verify_rubrics(
    idea: &IdeaBundle,
    attempt_id: &str,
    code: &str,
    language: &str,
    outcome: &EvalOutcome,
) -> RubricStatus {
    let metrics = RubricMetrics {
        combined_score: outcome.combined_score,
        correct: outcome.correct,
    };
    let unassessable = outcome.evaluation_failed || !outcome.correct;
    let results = idea
        .target_rubrics
        .iter()
        .map(|rubric| {
            if unassessable {
                return result(
                    rubric,
                    None,
                    "the evaluation did not establish correctness; criterion not assessed"
                        .to_string(),
                );
            }
            if rubric.kind == "metric" {
                let Some((metric, comparison, threshold)) = parse_metric_rubric(rubric) else {
                    return result(
                        rubric,
                        None,
                        "no machine-checkable threshold in this rubric".to_string(),
                    );
                };
                return match metric_value(outcome, &metric) {
                    Some(value) => result(
                        rubric,
                        Some(comparison.holds(value, threshold)),
                        format!(
                            "{metric} = {value} against {} {threshold}",
                            comparison.symbol()
                        ),
                    ),
                    None => result(
                        rubric,
                        None,
                        format!("the evaluator reported no metric named `{metric}`"),
                    ),
                };
            }
            match &rubric.verification {
                Some(verification) => {
                    let missing: Vec<&str> = verification
                        .patterns
                        .iter()
                        .filter(|pattern| !source_contains(code, language, pattern))
                        .map(String::as_str)
                        .collect();
                    if missing.is_empty() {
                        result(
                            rubric,
                            Some(true),
                            format!("all {} source patterns found", verification.patterns.len()),
                        )
                    } else {
                        result(
                            rubric,
                            Some(false),
                            format!("source patterns not found: {}", missing.join(" | ")),
                        )
                    }
                }
                None => result(
                    rubric,
                    None,
                    "no source_contains verification; criterion not assessed".to_string(),
                ),
            }
        })
        .collect();
    RubricStatus::new(attempt_id, &idea.idea_id, results, metrics)
}

const JUDGE_SYSTEM: &str = "You verify implementation rubrics against a program. For each rubric decide from the code \
alone whether the program does what the rubric describes. Quote the line that shows it, or say what is missing. Be \
strict: a comment or a name that mentions a technique is not an implementation of it.";

/// The model that judges rubrics the harness cannot check from source.
pub struct RubricJudge<'a> {
    pub llm: &'a LlmClient,
    pub model: &'a str,
    pub kwargs: &'a SamplingKwargs,
}

/// Ask a judge model about the implementation rubrics the harness could not
/// check itself. Leaves the status untouched when the call fails. Returns
/// what the call cost.
pub async fn judge_unassessed(
    status: &mut RubricStatus,
    idea: &IdeaBundle,
    code: &str,
    language: &str,
    judge: &RubricJudge<'_>,
    rng: &mut (dyn rand::RngCore + Send),
) -> f64 {
    let pending: Vec<&Rubric> = idea
        .target_rubrics
        .iter()
        .filter(|rubric| {
            rubric.kind == "implementation"
                && rubric.verification.is_none()
                && status
                    .rubrics
                    .iter()
                    .any(|r| r.id == rubric.id && r.assessment == "not_assessed")
        })
        .collect();
    if pending.is_empty() || !status.metrics.correct {
        return 0.0;
    }
    let listing: Vec<String> = pending
        .iter()
        .map(|rubric| format!("- {}: {}", rubric.id, rubric.description))
        .collect();
    let user = format!(
        "# Rubrics\n{}\n\n# Program\n```{language}\n{code}\n```\n\nRespond with one JSON object: \
         {{\"rubrics\": [{{\"id\": \"rubric_id\", \"pass\": true or false, \"evidence\": \"brief evidence\"}}]}}",
        listing.join("\n")
    );
    let mut request = LlmRequest::new(judge.model, JUDGE_SYSTEM, user);
    judge.kwargs.apply(&mut request, rng);
    let Ok((reply, response)) = judge.llm.query_json("rubric_judge", request).await else {
        return 0.0;
    };
    for verdict in reply["rubrics"].as_array().into_iter().flatten() {
        let (Some(id), Some(pass)) = (verdict["id"].as_str(), verdict["pass"].as_bool()) else {
            continue;
        };
        if !pending.iter().any(|rubric| rubric.id == id) {
            continue;
        }
        if let Some(entry) = status.rubrics.iter_mut().find(|entry| entry.id == id) {
            entry.pass = pass;
            entry.assessment = if pass { "passed" } else { "failed" }.to_string();
            entry.evidence = format!(
                "judge model: {}",
                verdict["evidence"].as_str().unwrap_or("")
            );
        }
    }
    status.overall_pass = !status.rubrics.is_empty() && status.rubrics.iter().all(|r| r.pass);
    response.cost_usd
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_core::idea::Verification;
    use genesis_llm::MockProvider;
    use serde_json::json;

    fn rubric(id: &str, kind: &str, description: &str) -> Rubric {
        Rubric {
            id: id.into(),
            kind: kind.into(),
            description: description.into(),
            ..Rubric::default()
        }
    }

    fn correct(score: f64) -> EvalOutcome {
        let mut outcome = EvalOutcome {
            correct: true,
            combined_score: score,
            ..EvalOutcome::default()
        };
        outcome
            .public_metrics
            .insert("runtime_ms".into(), json!(120.0));
        outcome
    }

    #[test]
    fn tokens_ignore_comments_and_layout_but_keep_strings_whole() {
        let python =
            "x = reheat( temp )  # reheat(fake)\ns = \"a # not a comment\"\n'''doc\nstring'''\n";
        assert_eq!(
            tokenize(python, "python"),
            [
                "x",
                "=",
                "reheat",
                "(",
                "temp",
                ")",
                "s",
                "=",
                "\"a # not a comment\"",
                "'''doc\nstring'''"
            ]
        );
        let rust = "let v = a/b; // divide\n/* block\n comment */ let w = 1.5e3;";
        assert_eq!(
            tokenize(rust, "rust"),
            ["let", "v", "=", "a", "/", "b", ";", "let", "w", "=", "1.5e3", ";"]
        );
        assert!(tokenize("\"unterminated", "python").len() == 1);
    }

    #[test]
    fn source_patterns_match_on_tokens_not_text() {
        let code = "def solve():\n    # we do not call anneal(x) here\n    return restart_search( seeds = 8 )\n";
        assert!(source_contains(code, "python", "restart_search(seeds=8)"));
        assert!(source_contains(code, "python", "restart_search ("));
        assert!(
            !source_contains(code, "python", "anneal(x)"),
            "a comment is not an implementation"
        );
        assert!(!source_contains(code, "python", "restart_search(seeds=80)"));
        assert!(
            !source_contains(code, "python", "start_search"),
            "a pattern never matches inside an identifier"
        );
        assert!(!source_contains(code, "python", "  # only a comment"));
    }

    #[test]
    fn metric_rubrics_parse_structured_symbolic_and_phrased_forms() {
        let mut structured = rubric("a", "metric", "improves");
        structured.metric = Some("runtime_ms".into());
        structured.operator = Some("<=".into());
        structured.threshold = Some(150.0);
        assert_eq!(
            parse_metric_rubric(&structured),
            Some(("runtime_ms".into(), Comparison::LessEqual, 150.0))
        );
        assert_eq!(
            parse_metric_rubric(&rubric("b", "metric", "combined_score >= 2.1")),
            Some(("combined_score".into(), Comparison::GreaterEqual, 2.1))
        );
        assert_eq!(
            parse_metric_rubric(&rubric(
                "c",
                "metric",
                "The sum of radii for 26 circles exceeds 2.1"
            )),
            Some(("combined_score".into(), Comparison::Greater, 2.1))
        );
        assert_eq!(
            parse_metric_rubric(&rubric("d", "metric", "error stays below 1e-3")),
            Some(("combined_score".into(), Comparison::Less, 1e-3))
        );
        assert_eq!(
            parse_metric_rubric(&rubric("e", "metric", "the result is better")),
            None
        );
    }

    #[test]
    fn verification_is_deterministic_and_overall_pass_needs_everything() {
        let mut with_patterns = rubric("uses_restarts", "implementation", "uses restarts");
        with_patterns.verification = Some(Verification {
            kind: "source_contains".into(),
            patterns: vec!["restart(".into(), "for seed in".into()],
        });
        let idea = IdeaBundle {
            idea_id: "i".into(),
            target_rubrics: vec![
                with_patterns,
                rubric("score_gate", "metric", "combined_score > 2.0"),
                rubric("fast", "metric", "runtime_ms < 100"),
                rubric("no_metric", "metric", "throughput > 5"),
                rubric("vague", "implementation", "is elegant"),
            ],
            ..IdeaBundle::default()
        };
        let code = "for seed in range(4):\n    restart(seed)\n";
        let status = verify_rubrics(&idea, "i_attempt_01", code, "python", &correct(2.5));
        let by_id = |id: &str| status.rubrics.iter().find(|r| r.id == id).unwrap();
        assert_eq!(
            (
                by_id("uses_restarts").pass,
                by_id("uses_restarts").assessment.as_str()
            ),
            (true, "passed")
        );
        assert_eq!(
            (
                by_id("score_gate").pass,
                by_id("score_gate").evidence.as_str()
            ),
            (true, "combined_score = 2.5 against > 2")
        );
        assert_eq!(by_id("fast").assessment, "failed");
        assert_eq!(by_id("no_metric").assessment, "not_assessed");
        assert_eq!(by_id("vague").assessment, "not_assessed");
        assert!(!status.overall_pass);
        assert_eq!((status.passed(), status.metrics.correct), (2, true));

        let missing = verify_rubrics(&idea, "a", "restart(1)", "python", &correct(2.5));
        assert!(missing.rubrics[0].evidence.contains("for seed in"));

        // An incorrect or failed evaluation assesses nothing.
        let incorrect = EvalOutcome {
            correct: false,
            combined_score: 9.0,
            ..EvalOutcome::default()
        };
        let status = verify_rubrics(&idea, "a", code, "python", &incorrect);
        assert!(status
            .rubrics
            .iter()
            .all(|r| !r.pass && r.assessment == "not_assessed"));
        assert!(!status.overall_pass);
    }

    #[tokio::test]
    async fn the_judge_only_fills_in_rubrics_the_harness_could_not_check() {
        let idea = IdeaBundle {
            idea_id: "i".into(),
            target_rubrics: vec![
                rubric("score_gate", "metric", "combined_score > 2.0"),
                rubric("vague", "implementation", "uses annealing"),
            ],
            ..IdeaBundle::default()
        };
        let mut status = verify_rubrics(&idea, "a", "anneal()", "python", &correct(2.5));
        assert!(!status.overall_pass);
        let provider = MockProvider::scripted(vec![json!({"rubrics": [
            {"id": "vague", "pass": true, "evidence": "anneal() on line 1"},
            {"id": "score_gate", "pass": false, "evidence": "ignored: not pending"}
        ]})
        .to_string()]);
        let llm = LlmClient::mock(provider.clone());
        let mut rng = rand::rngs::mock::StepRng::new(1, 1);
        let kwargs = SamplingKwargs::default();
        let judge = RubricJudge {
            llm: &llm,
            model: "judge",
            kwargs: &kwargs,
        };
        judge_unassessed(&mut status, &idea, "anneal()", "python", &judge, &mut rng).await;
        assert!(status.overall_pass);
        assert_eq!(
            status.rubrics[1].evidence,
            "judge model: anneal() on line 1"
        );
        assert!(
            status.rubrics[0].pass,
            "a deterministic verdict is never overruled by the judge"
        );
        assert!(provider.calls()[0]
            .last_user()
            .contains("- vague: uses annealing"));
        assert!(!provider.calls()[0].last_user().contains("score_gate"));
    }
}
