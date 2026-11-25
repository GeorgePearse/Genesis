//! Property tests for genesis-core, run with ephyra
//! (<https://github.com/datadog-labs/ephyra>).
//!
//! What "correct" means here, and where each expectation comes from:
//!
//! | Property | Oracle | Source |
//! |---|---|---|
//! | An evaluation is only ever `correct` with exit status 0 and a finite score | invariant | `eval.rs` module docs: "a claimed success with a non-zero exit status, a non-finite score or malformed metrics is recorded as a failed evaluation" |
//! | `read_results` and `IdeaBundle::normalize` accept any JSON without panicking | crash | both read model- or evaluator-written files |
//! | Normalised rubric ids are unique, non-empty slugs, at most 32 of them | invariant | `idea.rs`: ids are slugified and de-duplicated; `MAX_RUBRICS` |
//! | A `key=value` override is what `load` returns; an unknown key is an error | round trip | `config.rs` module docs: "Unknown keys fail at load" |
//! | Summary statistics are ordered; Welch's test is antisymmetric with p in [0, 1] | invariant, metamorphic | definitions of the statistics |
//! | `percentile` agrees with sorting | reference model | definition |
//! | `slugify` is idempotent over `[a-z0-9_]`; `tail` is a suffix on a char boundary | invariant | `fsutil.rs` docs |

use ephyra::prelude::*;
use ephyra::{gens, BoxGen};
use genesis_core::eval::read_results;
use genesis_core::idea::{IdeaBundle, MAX_RUBRICS};
use genesis_core::{config, fsutil, stats};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

fn key() -> BoxGen<String> {
    gens::string_len(gens::alphanumeric(), 1..=6).boxed()
}

fn json_leaf() -> BoxGen<Value> {
    gens::one_of(vec![
        gens::constant(Value::Null).boxed(),
        gens::bool().map(Value::Bool).boxed(),
        gens::int(-1000i64..=1000).map(|n| json!(n)).boxed(),
        gens::f64_range(-1e6..=1e6).map(|x| json!(x)).boxed(),
        gens::string_len(gens::char(), 0..=12)
            .map(Value::String)
            .boxed(),
    ])
    .boxed()
}

/// Any JSON document up to three levels deep.
fn json_value() -> BoxGen<Value> {
    gens::recursive(3, json_leaf(), |inner| {
        gens::frequency(vec![
            (2, json_leaf()),
            (
                1,
                gens::vec(inner.clone())
                    .len(0..=4)
                    .map(Value::Array)
                    .boxed(),
            ),
            (
                1,
                gens::vec((key(), inner))
                    .len(0..=4)
                    .map(|pairs| Value::Object(pairs.into_iter().collect()))
                    .boxed(),
            ),
        ])
        .boxed()
    })
    .boxed()
}

/// A field that is usually well-formed and sometimes arbitrary JSON or absent.
fn field(valid: BoxGen<Value>) -> BoxGen<Option<Value>> {
    gens::frequency(vec![
        (6, valid.map(Some).boxed()),
        (2, json_value().map(Some).boxed()),
        (1, gens::constant(None).boxed()),
    ])
    .boxed()
}

fn object(fields: Vec<(&'static str, Option<Value>)>) -> Value {
    Value::Object(
        fields
            .into_iter()
            .filter_map(|(name, value)| value.map(|v| (name.to_string(), v)))
            .collect(),
    )
}

#[derive(Debug, Clone)]
struct EvaluatorOutput {
    /// `None` leaves the file unwritten; text that is not JSON is written as is.
    metrics: Option<String>,
    correct: Option<String>,
    exit_code: Option<i32>,
}

fn file_text(doc: BoxGen<Value>) -> BoxGen<Option<String>> {
    gens::frequency(vec![
        (8, doc.map(|v| Some(v.to_string())).boxed()),
        (1, gens::constant(None).boxed()),
        (
            1,
            gens::pick_from(vec![
                Some("{".to_string()),
                Some(String::new()),
                Some("NaN".to_string()),
            ])
            .boxed(),
        ),
    ])
    .boxed()
}

fn evaluator_output() -> BoxGen<EvaluatorOutput> {
    let score = gens::one_of(vec![
        gens::f64_range(-1e9..=1e9).map(|x| json!(x)).boxed(),
        gens::int(-5i64..=5).map(|n| json!(n)).boxed(),
    ])
    .boxed();
    let metrics = (
        field(score),
        field(json_value()),
        field(json_value()),
        field(
            gens::string_len(gens::char(), 0..=20)
                .map(Value::String)
                .boxed(),
        ),
    )
        .map(|(score, public, private, feedback)| {
            object(vec![
                ("combined_score", score),
                ("public", public),
                ("private", private),
                ("text_feedback", feedback),
            ])
        })
        .boxed();
    let correct = (
        field(gens::bool_with(0.7).map(Value::Bool).boxed()),
        field(
            gens::string_len(gens::char(), 0..=20)
                .map(Value::String)
                .boxed(),
        ),
    )
        .map(|(correct, error)| object(vec![("correct", correct), ("error", error)]))
        .boxed();
    let exit = gens::frequency(vec![
        (6, gens::constant(Some(0)).boxed()),
        (2, gens::int(1i32..=255).map(Some).boxed()),
        (1, gens::constant(None).boxed()),
    ]);
    (file_text(metrics), file_text(correct), exit)
        .map(|(metrics, correct, exit_code)| EvaluatorOutput {
            metrics,
            correct,
            exit_code,
        })
        .boxed()
}

#[ephyra::test(gen = evaluator_output(), tests = 3000)]
fn an_evaluation_is_only_correct_with_exit_zero_and_a_finite_score(output: EvaluatorOutput) {
    let dir = tempfile::tempdir().unwrap();
    if let Some(text) = &output.metrics {
        std::fs::write(dir.path().join("metrics.json"), text).unwrap();
    }
    if let Some(text) = &output.correct {
        std::fs::write(dir.path().join("correct.json"), text).unwrap();
    }
    let outcome = read_results(dir.path(), output.exit_code);

    ephyra::cover(outcome.correct, "accepted as correct");
    ephyra::cover(outcome.evaluation_failed, "rejected by the harness");
    ephyra::cover(
        !outcome.correct && !outcome.evaluation_failed,
        "incorrect program",
    );
    assert!(
        outcome.combined_score.is_finite(),
        "a non-finite score was stored"
    );
    if outcome.correct {
        assert_eq!(
            output.exit_code,
            Some(0),
            "correct despite a failing exit status"
        );
        assert!(!outcome.evaluation_failed);
        let metrics: Value = serde_json::from_str(output.metrics.as_deref().unwrap()).unwrap();
        let claimed = metrics["combined_score"]
            .as_f64()
            .expect("correct without a numeric score");
        assert_eq!(outcome.combined_score, claimed);
        let correct: Value = serde_json::from_str(output.correct.as_deref().unwrap()).unwrap();
        assert_eq!(correct["correct"], json!(true));
        assert!(metrics
            .get("public")
            .is_none_or(|p| p.is_object() || p.is_null()));
        assert!(metrics
            .get("private")
            .is_none_or(|p| p.is_object() || p.is_null()));
    }
    if outcome.evaluation_failed {
        assert!(!outcome.correct);
        assert_eq!(outcome.combined_score, 0.0);
        assert!(outcome.diagnostic.is_some());
        assert!(!outcome.text_feedback.is_empty());
    }
}

fn idea_document() -> BoxGen<Value> {
    let text = || {
        gens::string_len(gens::char(), 0..=24)
            .map(Value::String)
            .boxed()
    };
    let rubric = (
        field(text()),
        field(text()),
        field(
            gens::pick_from(vec![
                json!("metric"),
                json!("implementation"),
                json!("Metric"),
                json!("other"),
            ])
            .boxed(),
        ),
        field(
            gens::vec(gens::string_len(gens::char(), 0..=6))
                .len(0..=10)
                .map(|patterns| json!({"type": "source_contains", "patterns": patterns}))
                .boxed(),
        ),
        field(gens::f64_range(-10.0..=10.0).map(|x| json!(x)).boxed()),
    )
        .map(|(id, description, kind, verification, threshold)| {
            object(vec![
                ("id", id),
                ("description", description),
                ("type", kind),
                ("verification", verification),
                ("threshold", threshold),
            ])
        })
        .boxed();
    let rubrics = gens::frequency(vec![
        (
            5,
            gens::vec(rubric.clone())
                .len(0..=6)
                .map(Value::Array)
                .boxed(),
        ),
        (1, gens::vec(rubric).len(30..=40).map(Value::Array).boxed()),
    ])
    .boxed();
    let plan = gens::vec(gens::string_len(gens::char(), 0..=12))
        .len(0..=4)
        .map(|steps| json!(steps))
        .boxed();
    gens::frequency(vec![
        (
            8,
            (
                field(text()),
                field(text()),
                field(text()),
                field(plan),
                field(rubrics),
                field(text()),
            )
                .map(|(idea_id, title, hypothesis, plan, rubrics, notes)| {
                    object(vec![
                        ("idea_id", idea_id),
                        ("title", title),
                        ("hypothesis", hypothesis),
                        ("implementation_plan", plan),
                        ("target_rubrics", rubrics),
                        ("notes_for_agent", notes),
                    ])
                })
                .boxed(),
        ),
        (1, json_value()),
    ])
    .boxed()
}

fn is_slug(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && !text.starts_with('_')
        && !text.ends_with('_')
}

#[ephyra::test(gen = (idea_document(), gens::int(0u32..=9999)))]
fn a_normalised_idea_has_unique_slug_rubric_ids(input: (Value, u32)) {
    let (document, cycle) = input;
    let normalised = IdeaBundle::normalize(&document, cycle);
    ephyra::cover(normalised.is_ok(), "accepted");
    let Ok(idea) = normalised else { return };
    assert!(is_slug(&idea.idea_id), "idea id {:?}", idea.idea_id);
    assert!(!idea.title.is_empty() && !idea.hypothesis.is_empty());
    assert!(!idea.implementation_plan.is_empty());
    assert!(idea
        .implementation_plan
        .iter()
        .all(|step| !step.trim().is_empty()));
    assert!(!idea.target_rubrics.is_empty() && idea.target_rubrics.len() <= MAX_RUBRICS);
    let ids: HashSet<&str> = idea.target_rubrics.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids.len(), idea.target_rubrics.len(), "duplicate rubric ids");
    for rubric in &idea.target_rubrics {
        assert!(is_slug(&rubric.id), "rubric id {:?}", rubric.id);
        assert!(rubric.kind == "metric" || rubric.kind == "implementation");
        if let Some(verification) = &rubric.verification {
            assert_eq!(rubric.kind, "implementation");
            assert!(!verification.patterns.is_empty() && verification.patterns.len() <= 8);
            assert!(verification.patterns.iter().all(|p| !p.trim().is_empty()));
        }
    }
    assert!(idea.dir_name(cycle).starts_with(&format!("{cycle:02}_")));
    // Normalising what was just written is a fixed point.
    let again = IdeaBundle::normalize(&serde_json::to_value(&idea).unwrap(), cycle).unwrap();
    assert_eq!(again, idea);
}

#[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct Limits {
    max_cycles: u32,
    budget: Option<f64>,
    resume: bool,
}

#[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct Settings {
    name: String,
    limits: Limits,
    env: BTreeMap<String, String>,
}

#[ephyra::test(gen = (gens::int(0u32..=1_000_000), gens::f64_range(0.0..=1e6), gens::bool(), key(), key()))]
fn an_override_is_what_load_returns(input: (u32, f64, bool, String, String)) {
    let (cycles, budget, resume, env_key, env_value) = input;
    let settings: Settings = config::load(
        None,
        &[
            format!("limits.max_cycles={cycles}"),
            format!("limits.budget={budget:?}"),
            format!("limits.resume={resume}"),
            format!("env.{env_key}='{env_value}'"),
        ],
    )
    .unwrap();
    assert_eq!(settings.limits.max_cycles, cycles);
    assert_eq!(settings.limits.budget, Some(budget));
    assert_eq!(settings.limits.resume, resume);
    assert_eq!(settings.env.get(&env_key), Some(&env_value));
    assert_eq!(settings.name, "");
}

#[ephyra::test(gen = (key(), key(), json_value()))]
fn an_unknown_key_is_always_an_error(input: (String, String, Value)) {
    // No key of `Settings` starts with `q`, so these are unknown by construction.
    let (first, second, value) = (format!("q{}", input.0), format!("q{}", input.1), input.2);
    let value = serde_json::to_string(&value).unwrap();
    assert!(config::load::<Settings>(None, &[format!("{first}={value}")]).is_err());
    assert!(config::load::<Settings>(None, &[format!("limits.{second}={value}")]).is_err());
    assert!(config::load::<Settings>(None, &[format!("{first}.{second}={value}")]).is_err());
}

fn sample() -> BoxGen<Vec<f64>> {
    gens::frequency(vec![
        (
            4,
            gens::vec(gens::f64_range(-1e6..=1e6)).len(1..=40).boxed(),
        ),
        (
            1,
            gens::vec(gens::int(-3i64..=3).map(|n| n as f64))
                .len(1..=12)
                .boxed(),
        ),
    ])
    .boxed()
}

#[ephyra::test(gen = sample())]
fn summary_statistics_are_ordered(values: Vec<f64>) {
    let s = stats::summarize(&values);
    let slack = 1e-9 * (1.0 + s.max.abs().max(s.min.abs()));
    assert_eq!(s.n, values.len());
    assert!(s.min <= s.median && s.median <= s.max);
    assert!(s.min - slack <= s.mean && s.mean <= s.max + slack);
    assert!(s.std >= 0.0 && s.sem >= 0.0 && s.sem <= s.std + slack);
    assert!(s.ci95_low <= s.mean && s.mean <= s.ci95_high);
    if values.len() == 1 {
        assert_eq!((s.std, s.ci95_low, s.ci95_high), (0.0, s.mean, s.mean));
    }
}

#[ephyra::test(gen = (sample(), sample()))]
fn welch_comparison_is_antisymmetric(pair: (Vec<f64>, Vec<f64>)) {
    let (a, b) = pair;
    let forward = stats::compare(&a, &b);
    let backward = stats::compare(&b, &a);
    let scale = 1.0 + forward.mean_a.abs() + forward.mean_b.abs();
    assert!((forward.mean_diff + backward.mean_diff).abs() <= 1e-9 * scale);
    assert_eq!(forward.t.is_some(), backward.t.is_some());
    assert_eq!(forward.p_value.is_some(), forward.t.is_some());
    ephyra::cover(forward.t.is_some(), "test statistic defined");
    if let (Some(t), Some(back_t), Some(p), Some(back_p)) =
        (forward.t, backward.t, forward.p_value, backward.p_value)
    {
        assert!(t.is_finite() && (t + back_t).abs() <= 1e-6 * (1.0 + t.abs()));
        assert!((0.0..=1.0).contains(&p));
        assert!((p - back_p).abs() <= 1e-9);
        let (low, high) = (
            forward.diff_ci95_low.unwrap(),
            forward.diff_ci95_high.unwrap(),
        );
        assert!(low <= forward.mean_diff && forward.mean_diff <= high);
        // The interval excludes zero exactly when the test rejects at 5%.
        if (p - 0.05).abs() > 1e-6 {
            assert_eq!(
                low > 0.0 || high < 0.0,
                p < 0.05,
                "p={p} ci=[{low}, {high}]"
            );
        }
    } else {
        assert!(
            a.len() < 2
                || b.len() < 2
                || (stats::sample_std(&a) == 0.0 && stats::sample_std(&b) == 0.0)
        );
    }
}

#[ephyra::test(gen = (gens::f64_range(-50.0..=50.0), gens::f64_range(-50.0..=50.0), gens::f64_range(1.0..=200.0)))]
fn student_t_cdf_is_a_symmetric_monotone_distribution(input: (f64, f64, f64)) {
    let (x, y, df) = input;
    let (low, high) = if x <= y { (x, y) } else { (y, x) };
    let (cdf_low, cdf_high) = (
        stats::student_t_cdf(low, df),
        stats::student_t_cdf(high, df),
    );
    assert!((0.0..=1.0).contains(&cdf_low) && (0.0..=1.0).contains(&cdf_high));
    assert!(cdf_low <= cdf_high + 1e-12);
    assert!((stats::student_t_cdf(x, df) + stats::student_t_cdf(-x, df) - 1.0).abs() <= 1e-9);
    let critical = stats::t_critical_975(df);
    assert!(critical > 1.959 && (stats::student_t_cdf(critical, df) - 0.975).abs() <= 1e-9);
}

#[ephyra::test(gen = (sample(), gens::f64_range(0.0..=100.0), gens::f64_range(0.0..=100.0)))]
fn percentile_agrees_with_sorting(input: (Vec<f64>, f64, f64)) {
    let (values, p, q) = input;
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    assert_eq!(stats::percentile(&values, 0.0), sorted[0]);
    assert_eq!(stats::percentile(&values, 100.0), sorted[sorted.len() - 1]);
    let (low, high) = if p <= q { (p, q) } else { (q, p) };
    let (at_low, at_high) = (
        stats::percentile(&values, low),
        stats::percentile(&values, high),
    );
    let slack = 1e-9 * (1.0 + at_low.abs().max(at_high.abs()));
    assert!(at_low <= at_high + slack);
    assert!(sorted[0] - slack <= at_low && at_high <= sorted[sorted.len() - 1] + slack);
    assert!((stats::percentile(&values, 50.0) - stats::median(&values)).abs() <= slack);
}

#[ephyra::test]
fn slugify_is_idempotent_over_a_fixed_alphabet(text: String) {
    let slug = fsutil::slugify(&text);
    assert!(slug.is_empty() || is_slug(&slug), "{slug:?}");
    assert!(!slug.contains("__"));
    assert_eq!(fsutil::slugify(&slug), slug);
}

#[ephyra::test(gen = (gens::string_len(gens::char(), 0..=40), gens::int(0usize..=48)))]
fn tail_and_truncate_cut_on_character_boundaries(input: (String, usize)) {
    let (text, limit) = input;
    let tail = fsutil::tail(&text, limit);
    if text.len() <= limit {
        assert_eq!(tail, text);
    } else {
        let kept = tail.strip_prefix('…').expect("a cut tail is marked");
        assert!(text.ends_with(kept) && kept.len() <= limit);
    }
    let head = fsutil::truncate(&text, limit);
    if text.chars().count() <= limit {
        assert_eq!(head, text);
    } else {
        let kept = head.strip_suffix('…').expect("a cut head is marked");
        assert!(text.starts_with(kept) && kept.chars().count() == limit);
    }
}

#[ephyra::test(gen = (gens::vec(key()).len(1..=4), gens::int(0usize..=3)))]
fn a_path_component_pattern_ignores_everything_beneath_it(input: (Vec<String>, usize)) {
    let (components, pick) = input;
    let path = components.join("/");
    let chosen = components[pick % components.len()].clone();
    assert!(fsutil::is_ignored(&path, &[chosen]));
    assert!(fsutil::is_ignored(&path, &[components[0].clone() + "/"]));
    assert!(!fsutil::is_ignored(&path, &[]));
    assert!(!fsutil::is_ignored(
        &path,
        &["# a comment".to_string(), String::new()]
    ));
    let suffix = format!("*{}", &path[path.len() - 1..]);
    assert!(fsutil::is_ignored(&path, &[suffix]));
}
