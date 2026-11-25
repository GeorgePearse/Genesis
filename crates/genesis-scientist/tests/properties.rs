//! Property tests for genesis-scientist, run with ephyra
//! (<https://github.com/datadog-labs/ephyra>).
//!
//! | Property | Oracle | Source |
//! |---|---|---|
//! | A source pattern cut from a program's own tokens is found; comments and layout added to the program never change the verdict; a pattern holding a token the program lacks is not found | metamorphic | `rubrics.rs`: patterns are "compared on code tokens, so comments and whitespace do not count" |
//! | Tokenising never panics, and re-tokenising the tokens joined by spaces gives the same tokens | crash, round trip | definition of a token |
//! | A metric rubric written as `name op threshold` parses back to the same triple, and passes exactly when the comparison holds | round trip, reference model | `parse_metric_rubric` docs |
//! | `overall_pass` holds exactly when there is a rubric and all pass; nothing is assessed without an established-correct evaluation; verdicts keep rubric order | invariant | `rubrics.rs` module docs |
//! | Conclusions are accepted exactly when they are one non-empty paragraph of at most 300 words | reference model | `memory.rs` module docs |
//! | Memory events come back in the order they were appended | round trip | the journal is append-only |
//! | An idea's directory name is a single safe path component whatever a model put in the title | invariant | model-written titles become directory names |
//! | The investigation has a winner exactly when a correct generated program beats the seed (or the seed is incorrect); the frozen selection is a correct program of maximal score among those eligible, and freezing twice changes nothing | reference model, invariant | `selection.rs` module docs |

use ephyra::prelude::*;
use ephyra::{gens, BoxGen};
use genesis_core::idea::{IdeaBundle, Rubric, Verification};
use genesis_core::{EvalOutcome, Program, Task};
use genesis_evolve::{DatabaseConfig, ProgramDb};
use genesis_scientist::artifacts::{OUTCOME_NO_WINNER, OUTCOME_WINNER};
use genesis_scientist::ideas::incumbent_gate;
use genesis_scientist::memory::{validate_conclusions, Memory, MemoryEvent, MAX_CONCLUSION_WORDS};
use genesis_scientist::rubrics::{
    parse_metric_rubric, source_contains, tokenize, verify_rubrics, Comparison,
};
use genesis_scientist::selection::{freeze_selection, judge_investigation, load_selection};
use genesis_scientist::RunPaths;
use serde_json::json;

/// One code token that survives a round trip through the tokeniser.
fn token() -> BoxGen<String> {
    gens::frequency(vec![
        (
            4,
            gens::string_len(gens::pick_from(vec!['a', 'b', 'x', 'y', 'z', '_']), 1..=5).boxed(),
        ),
        (2, gens::int(0u32..=999).map(|n| n.to_string()).boxed()),
        (
            3,
            gens::pick_from(vec![
                "(", ")", "=", "+", "*", ",", ":", "[", "]", "<", ".", ";",
            ])
            .map(str::to_string)
            .boxed(),
        ),
        (
            1,
            gens::string_len(gens::pick_from(vec!['a', 'b', ' ', '#']), 0..=6)
                .map(|body| format!("\"{body}\""))
                .boxed(),
        ),
    ])
    .boxed()
}

fn tokens() -> BoxGen<Vec<String>> {
    gens::vec(token()).len(1..=30).boxed()
}

/// Join tokens so that neighbours cannot merge: identifiers and numbers are
/// always separated by a space.
fn join(tokens: &[String]) -> String {
    tokens.join(" ")
}

#[ephyra::test(gen = tokens())]
fn tokens_round_trip_through_the_tokeniser(tokens: Vec<String>) {
    for language in ["python", "rust"] {
        assert_eq!(tokenize(&join(&tokens), language), tokens, "{language}");
    }
}

#[ephyra::test(gen = (gens::string_len(gens::char(), 0..=60), gens::pick_from(vec!["python", "rust", "fortran", "unknown"])))]
fn tokenising_arbitrary_text_never_panics_or_yields_empty_tokens(input: (String, &'static str)) {
    let (text, language) = input;
    let tokens = tokenize(&text, language);
    assert!(tokens.iter().all(|token| !token.is_empty()));
    let _ = source_contains(&text, language, &text);
}

#[ephyra::test(gen = (tokens(), gens::int(0usize..=29), gens::int(1usize..=6), gens::vec(gens::int(0usize..=30)).len(0..=5)))]
fn a_pattern_cut_from_the_program_is_found_whatever_comments_are_added(
    input: (Vec<String>, usize, usize, Vec<usize>),
) {
    let (tokens, start, length, comment_positions) = input;
    let start = start % tokens.len();
    let end = (start + length).min(tokens.len());
    let pattern = join(&tokens[start..end]);
    let code = join(&tokens);
    assert!(
        source_contains(&code, "python", &pattern),
        "pattern `{pattern}` not found in `{code}`"
    );

    // Interleave comment lines and extra layout between tokens.
    let mut noisy = String::new();
    for (index, token) in tokens.iter().enumerate() {
        if comment_positions.contains(&index) {
            noisy.push_str("  # reheat( restart( anneal(\n");
        }
        noisy.push_str(token);
        noisy.push_str(if index % 3 == 0 { "\n\t" } else { "  " });
    }
    assert_eq!(
        tokenize(&noisy, "python"),
        tokens,
        "comments and layout changed the tokens"
    );
    assert!(source_contains(&noisy, "python", &pattern));

    // A token the program does not contain is never found, even though the
    // comments mention it.
    assert!(!source_contains(&noisy, "python", "reheat("));
    let absent = format!("{pattern} never_in_program_q");
    assert!(!source_contains(&noisy, "python", &absent));
}

fn comparison() -> BoxGen<(&'static str, Comparison)> {
    gens::pick_from(vec![
        (">", Comparison::Greater),
        (">=", Comparison::GreaterEqual),
        ("<", Comparison::Less),
        ("<=", Comparison::LessEqual),
    ])
    .boxed()
}

fn metric_name() -> BoxGen<String> {
    gens::pick_from(vec![
        "combined_score",
        "runtime_ms",
        "sum_of_radii",
        "accuracy",
    ])
    .map(str::to_string)
    .boxed()
}

#[ephyra::test(gen = (metric_name(), comparison(), gens::int(-5000i32..=5000), gens::int(-5000i32..=5000), gens::bool()))]
fn a_metric_rubric_parses_back_and_passes_exactly_when_it_holds(
    input: (String, (&'static str, Comparison), i32, i32, bool),
) {
    let (metric, (symbol, comparison), threshold_milli, value_milli, structured) = input;
    let (threshold, value) = (
        f64::from(threshold_milli) / 1000.0,
        f64::from(value_milli) / 1000.0,
    );
    let rubric = if structured {
        Rubric {
            id: "gate".into(),
            kind: "metric".into(),
            description: "the metric improves".into(),
            metric: Some(metric.clone()),
            operator: Some(symbol.into()),
            threshold: Some(threshold),
            ..Rubric::default()
        }
    } else {
        Rubric {
            id: "gate".into(),
            kind: "metric".into(),
            description: format!("{metric} {symbol} {threshold:?}"),
            ..Rubric::default()
        }
    };
    assert_eq!(
        parse_metric_rubric(&rubric),
        Some((metric.clone(), comparison, threshold))
    );

    let mut outcome = EvalOutcome {
        correct: true,
        combined_score: value,
        ..EvalOutcome::default()
    };
    outcome.public_metrics.insert(metric.clone(), json!(value));
    let idea = IdeaBundle {
        idea_id: "i".into(),
        target_rubrics: vec![rubric],
        ..IdeaBundle::default()
    };
    let status = verify_rubrics(&idea, "i_attempt_01", "x = 1", "python", &outcome);
    let expected = match comparison {
        Comparison::Greater => value > threshold,
        Comparison::GreaterEqual => value >= threshold,
        Comparison::Less => value < threshold,
        Comparison::LessEqual => value <= threshold,
        Comparison::Equal => unreachable!("not generated"),
    };
    assert_eq!(
        status.rubrics[0].pass, expected,
        "{value} {symbol} {threshold}"
    );
    assert_eq!(status.overall_pass, expected);
}

#[derive(Debug, Clone)]
struct RubricCase {
    rubrics: Vec<Rubric>,
    code: Vec<String>,
    correct: bool,
    evaluation_failed: bool,
    score: f64,
}

fn rubric_case() -> BoxGen<RubricCase> {
    let rubric =
        (gens::int(0u8..=3), tokens(), gens::int(-3i32..=3)).map(|(kind, pattern, threshold)| {
            match kind {
                0 => Rubric {
                    kind: "implementation".into(),
                    verification: Some(Verification {
                        kind: "source_contains".into(),
                        patterns: vec![join(&pattern[..pattern.len().min(3)])],
                    }),
                    ..Rubric::default()
                },
                1 => Rubric {
                    kind: "implementation".into(),
                    description: "is elegant".into(),
                    ..Rubric::default()
                },
                2 => Rubric {
                    kind: "metric".into(),
                    description: format!("combined_score > {threshold}"),
                    ..Rubric::default()
                },
                _ => Rubric {
                    kind: "metric".into(),
                    description: "is better".into(),
                    ..Rubric::default()
                },
            }
        });
    (
        gens::vec(rubric).len(0..=6),
        tokens(),
        gens::bool_with(0.7),
        gens::bool_with(0.15),
        gens::int(-4i32..=4),
    )
        .map(|(mut rubrics, code, correct, evaluation_failed, score)| {
            for (index, rubric) in rubrics.iter_mut().enumerate() {
                rubric.id = format!("rubric_{index}");
            }
            RubricCase {
                rubrics,
                code,
                correct,
                evaluation_failed,
                score: f64::from(score),
            }
        })
        .boxed()
}

#[ephyra::test(gen = rubric_case())]
fn rubric_verdicts_are_consistent(case: RubricCase) {
    let idea = IdeaBundle {
        idea_id: "idea".into(),
        target_rubrics: case.rubrics.clone(),
        ..IdeaBundle::default()
    };
    let outcome = EvalOutcome {
        correct: case.correct && !case.evaluation_failed,
        evaluation_failed: case.evaluation_failed,
        combined_score: case.score,
        ..EvalOutcome::default()
    };
    let status = verify_rubrics(
        &idea,
        "idea_attempt_01",
        &join(&case.code),
        "python",
        &outcome,
    );
    assert_eq!(
        (status.attempt_id.as_str(), status.idea_id.as_str()),
        ("idea_attempt_01", "idea")
    );
    let ids: Vec<&str> = status.rubrics.iter().map(|r| r.id.as_str()).collect();
    let expected: Vec<&str> = case.rubrics.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, expected, "verdicts keep the rubric order");
    assert_eq!(
        status.overall_pass,
        !status.rubrics.is_empty() && status.rubrics.iter().all(|r| r.pass)
    );
    assert_eq!(
        status.passed(),
        status.rubrics.iter().filter(|r| r.pass).count()
    );
    ephyra::cover(status.overall_pass, "all rubrics pass");
    for verdict in &status.rubrics {
        assert!(!verdict.evidence.is_empty());
        match verdict.assessment.as_str() {
            "passed" => assert!(verdict.pass),
            "failed" | "not_assessed" => assert!(!verdict.pass),
            other => panic!("unknown assessment {other}"),
        }
    }
    if !outcome.correct {
        assert!(
            status
                .rubrics
                .iter()
                .all(|r| r.assessment == "not_assessed"),
            "assessed without an established-correct evaluation"
        );
    }
}

#[ephyra::test(gen = (gens::int(-2000i32..=2000), gens::int(-2000i32..=2000)))]
fn the_incumbent_gate_passes_exactly_on_a_strict_improvement(input: (i32, i32)) {
    let (parent, child) = (f64::from(input.0) / 100.0, f64::from(input.1) / 100.0);
    let idea = IdeaBundle {
        idea_id: "i".into(),
        target_rubrics: vec![incumbent_gate(parent)],
        ..IdeaBundle::default()
    };
    let outcome = EvalOutcome {
        correct: true,
        combined_score: child,
        ..EvalOutcome::default()
    };
    let status = verify_rubrics(&idea, "a", "", "python", &outcome);
    assert_eq!(
        status.rubrics[0].pass,
        child > parent,
        "child {child} against parent {parent}"
    );
}

fn paragraph_text() -> BoxGen<String> {
    let word = gens::string_len(gens::alphanumeric(), 1..=6).boxed();
    let any_separator = gens::frequency(vec![
        (10, gens::constant(" ".to_string()).boxed()),
        (1, gens::constant("\n".to_string()).boxed()),
        (1, gens::constant("\n\n".to_string()).boxed()),
        (1, gens::constant("  \t ".to_string()).boxed()),
    ])
    .boxed();
    // Around the word limit a blank line would decide the verdict first, so
    // these texts have none: only the count can reject them.
    let plain_separator =
        gens::pick_from(vec![" ".to_string(), "\n".to_string(), "  ".to_string()]).boxed();
    let build = |pieces: Vec<(String, String)>| {
        pieces
            .into_iter()
            .map(|(word, separator)| format!("{word}{separator}"))
            .collect::<String>()
    };
    gens::frequency(vec![
        (
            5,
            gens::vec((word.clone(), any_separator))
                .len(0..=40)
                .map(build)
                .boxed(),
        ),
        (
            3,
            gens::vec((word, plain_separator))
                .len(297..=303)
                .map(build)
                .boxed(),
        ),
    ])
    .boxed()
}

#[ephyra::test(gen = paragraph_text(), tests = 3000)]
fn conclusions_are_accepted_exactly_as_one_short_paragraph(text: String) {
    let trimmed = text.trim();
    let words = trimmed.split_whitespace().count();
    let expected =
        !trimmed.is_empty() && !trimmed.contains("\n\n") && words <= MAX_CONCLUSION_WORDS;
    ephyra::cover(expected, "accepted");
    ephyra::cover(words > MAX_CONCLUSION_WORDS, "too long");
    ephyra::cover(
        words == MAX_CONCLUSION_WORDS && expected,
        "exactly at the limit",
    );
    match validate_conclusions(&text) {
        Ok(accepted) => {
            assert!(
                expected,
                "accepted {words} words, blank line: {}",
                trimmed.contains("\n\n")
            );
            assert_eq!(accepted, trimmed);
        }
        Err(_) => assert!(!expected, "rejected a valid paragraph of {words} words"),
    }
}

#[ephyra::test(gen = gens::vec((gens::pick_from(vec!["idea", "evaluate", "verify"]), gens::string_len(gens::char(), 1..=30))).len(0..=12), tests = 400)]
fn memory_events_come_back_in_order(events: Vec<(&'static str, String)>) {
    let dir = tempfile::tempdir().unwrap();
    let memory = Memory::open(dir.path()).unwrap();
    let mut written = Vec::new();
    for (step, summary) in &events {
        let event = MemoryEvent::new(step, "attempt", summary.clone(), "evidence", "next");
        match memory.append(event) {
            Ok(()) => written.push((step.to_string(), summary.clone())),
            Err(_) => assert!(summary.trim().is_empty(), "a non-blank summary was refused"),
        }
    }
    let read: Vec<(String, String)> = memory
        .events()
        .into_iter()
        .map(|event| (event.step, event.summary))
        .collect();
    assert_eq!(read, written);
    assert_eq!(
        memory.recent_summary(5).lines().count(),
        written.len().clamp(1, 5)
    );
}

#[ephyra::test(gen = (gens::string_len(gens::char(), 0..=120), gens::string_len(gens::char(), 0..=20), gens::int(0u32..=200_000)))]
fn an_idea_directory_name_is_one_safe_path_component(input: (String, String, u32)) {
    let (title, idea_id, cycle) = input;
    let raw = json!({
        "idea_id": idea_id, "title": title, "hypothesis": "h", "implementation_plan": ["p"],
        "target_rubrics": [{"description": "d"}]
    });
    let Ok(idea) = IdeaBundle::normalize(&raw, cycle) else {
        return;
    };
    let name = idea.dir_name(cycle);
    assert!(!name.is_empty() && name.len() <= 80, "{name:?}");
    assert!(
        name.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
        "{name:?}"
    );
    assert!(name != "." && name != ".." && !name.contains('/') && !name.contains('\\'));
    assert!(std::path::Path::new(&name).components().count() == 1);
}

fn run_rows() -> BoxGen<Vec<(i32, bool)>> {
    gens::vec((gens::int(-5i32..=5), gens::bool_with(0.6)))
        .len(1..=10)
        .boxed()
}

#[ephyra::test(gen = (run_rows(), gens::pick_from(vec!["best_available", "skip"])), tests = 300)]
fn the_winner_and_the_frozen_selection_follow_the_stated_rules(
    input: (Vec<(i32, bool)>, &'static str),
) {
    let (rows, policy) = input;
    let dir = tempfile::tempdir().unwrap();
    let task_dir = dir.path().join("task");
    std::fs::create_dir_all(&task_dir).unwrap();
    std::fs::write(task_dir.join("initial.py"), "x = 0\n").unwrap();
    std::fs::write(task_dir.join("evaluate.sh"), "true\n").unwrap();
    let task = Task::load(&task_dir).unwrap();
    let paths = RunPaths::new(dir.path().join("run"));
    let db = ProgramDb::open(
        None,
        DatabaseConfig {
            num_islands: 2,
            seed: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let mut seed_id = None;
    for (generation, (score, correct)) in rows.iter().enumerate() {
        let mut program = Program::new(format!("x = {generation}\n"), "python");
        program.generation = generation as u32;
        program.combined_score = f64::from(*score);
        program.correct = *correct;
        program.parent_id = seed_id.clone();
        db.add(&mut program).unwrap();
        seed_id.get_or_insert(program.id.clone());
    }

    let (seed_score, seed_correct) = (f64::from(rows[0].0), rows[0].1);
    let generated: Vec<f64> = rows[1..]
        .iter()
        .filter(|row| row.1)
        .map(|row| f64::from(row.0))
        .collect();
    let best_generated = generated.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let has_winner = !generated.is_empty() && (!seed_correct || best_generated > seed_score);

    let (manifest, outcome) = judge_investigation(&paths, &task, &db).unwrap();
    assert_eq!(
        outcome,
        if has_winner {
            OUTCOME_WINNER
        } else {
            OUTCOME_NO_WINNER
        }
    );
    ephyra::cover(has_winner, "winner");
    if has_winner {
        assert_eq!(
            (manifest.combined_score, manifest.correct),
            (Some(best_generated), Some(true))
        );
        assert!(manifest.generation.unwrap() > 0);
    } else {
        assert_eq!(
            (manifest.source.as_str(), manifest.generation),
            ("seed", Some(0))
        );
    }

    let selected = freeze_selection(&paths, &task, &db, policy).unwrap();
    let expected_kind = if has_winner {
        Some("winner")
    } else if policy == "skip" {
        None
    } else if !generated.is_empty() {
        Some("non_winner_candidate")
    } else if seed_correct {
        Some("baseline")
    } else {
        None
    };
    assert_eq!(
        selected
            .as_ref()
            .map(|s| s.selection.selection_kind.as_str()),
        expected_kind
    );
    if let Some(selected) = &selected {
        assert!(
            selected.selection.correct,
            "an incorrect program was selected"
        );
        let expected_score = if expected_kind == Some("baseline") {
            seed_score
        } else {
            best_generated
        };
        assert_eq!(selected.selection.combined_score, expected_score);
        assert_eq!(selected.baseline_code, "x = 0\n");
        assert_eq!(selected.selection.baseline_correct, seed_correct);
        assert_eq!(selected.selection.attempt_counts.total, rows.len() - 1);
        // Freezing again, even under the other policy, changes nothing.
        let again = freeze_selection(&paths, &task, &db, "best_available")
            .unwrap()
            .unwrap();
        assert_eq!(&again, selected);
        assert_eq!(load_selection(&paths).unwrap().as_ref(), Some(selected));
    } else {
        assert!(load_selection(&paths).unwrap().is_none());
    }
}
