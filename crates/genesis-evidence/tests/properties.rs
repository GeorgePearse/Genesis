//! Property tests for the evidence phases, run with ephyra
//! (<https://github.com/datadog-labs/ephyra>).
//!
//! | Property | Oracle | Source |
//! |---|---|---|
//! | Variant selection matches a rule-by-rule reference: it never exceeds a cap, never accepts a reserved or repeated id, and keeps the generator's order | reference model | `ablations/variants.rs`: "Candidates are taken in the generator's order. A reserved id, a repeated id and a candidate the reviewer rejected are turned down ... Then the caps apply, per kind." |
//! | Final seeds are distinct, as many as asked for, and share nothing with the ablation seeds, for any configuration | invariant | `final_eval.rs`: "disjoint from `used` by construction, whatever the configuration" |
//! | A sample compared with itself differs by zero; swapping variant and control negates the difference and keeps the p-value | metamorphic | definition of a difference of means and of Welch's test |
//! | A variant's statistics are over exactly its correct replications; the anchor carries no comparison; a difference is never called "no effect" | invariant | `artifacts.rs`: "Statistics of `combined_score` over correct replications"; `report.rs` on `difference_note` |
//! | The run-summary table has one row per replication and survives the CSV round trip; any cell does | round trip | `csv.rs`: "any string survives a round trip" |
//! | The grounding check accepts a number exactly when a key number (or, for a percentage, a key number times 100) is within half a unit of its last digit | reference model in integer arithmetic | `posthoc/grounding.rs` module docs |
//! | Every built-in analysis is total on any run database: finite key numbers, a claim grounded in them, a rectangular table, a well-formed chart | crash, invariant | `posthoc/analyses.rs`: "Every number the analysis reports ... All finite", "One sentence, every number of which is a key number" |
//! | Post-hoc selection keeps one analysis per family, within the cap, in order | invariant | `posthoc/mod.rs` on `select_analyses` |
//! | A final evaluation qualifies exactly when every episode is correct, and never supports an improvement of a program over itself | invariant | `final_eval.rs` module docs |
//! | Model replies of any JSON shape parse without panicking, into candidates whose ids are safe directory names and whose kinds are the two allowed | crash, invariant | `variants.rs` and `posthoc/mod.rs`: the parsers read model-written JSON |
//! | Charts and the number scanner accept any input: no panic, and no non-finite coordinate in a chart | crash, invariant | `svg.rs`: charts are written into papers; `grounding.rs` reads model-written text |

use ephyra::prelude::*;
use ephyra::{gens, BoxGen};
use genesis_core::Program;
use genesis_evidence::ablations::report::{
    compare_to_control, correct_scores, difference_note, is_stochastic, metric_value_rows,
    required_replications, run_summary_rows, summarise_variant, RUN_SUMMARY_HEADER,
};
use genesis_evidence::ablations::variants::{parse_candidates, parse_decisions};
use genesis_evidence::ablations::variants::{select_variants, Decision, CONTROL_ID, RESERVED_IDS};
use genesis_evidence::ablations::{ablation_seeds, AblationConfig};
use genesis_evidence::csv;
use genesis_evidence::final_eval::{assess, final_seeds, FinalEvalConfig, QUALIFIED};
use genesis_evidence::posthoc::analyses::{registry, RunData};
use genesis_evidence::posthoc::grounding::numbers_in;
use genesis_evidence::posthoc::grounding::{claim_is_grounded, ungrounded_numbers};
use genesis_evidence::posthoc::parse_proposals;
use genesis_evidence::posthoc::svg::{bar_chart, xy_chart, Series};
use genesis_evidence::posthoc::{default_proposals, select_analyses};
use genesis_evolve::db::AttemptRecord;
use genesis_evolve::{DatabaseConfig, ProgramDb};
use genesis_scientist::artifacts::{
    AblationSummary, AnalysisSpec, FinalEpisode, FinalManifest, ReplicationResult, VariantResult,
    VariantSpec,
};
use serde_json::json;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

fn variant_candidates() -> BoxGen<Vec<VariantSpec>> {
    let ids = vec![
        "control_best",
        "baseline_reference",
        "ablation_a",
        "ablation_b",
        "ablation_c",
        "baseline_x",
        "baseline_y",
        "other",
    ];
    let id = gens::pick_from(ids.into_iter().map(str::to_string).collect());
    gens::vec(
        (id, gens::bool()).map(|(variant_id, is_ablation)| VariantSpec {
            title: format!("title of {variant_id}"),
            kind: if is_ablation { "ablation" } else { "baseline" }.to_string(),
            base_program: "best".to_string(),
            variant_id,
            ..VariantSpec::default()
        }),
    )
    .len(0..=14)
    .boxed()
}

fn reviewer_decisions() -> BoxGen<BTreeMap<String, Decision>> {
    let ids = vec![
        "control_best",
        "ablation_a",
        "ablation_b",
        "ablation_c",
        "baseline_x",
        "baseline_y",
        "other",
        "unknown",
    ];
    let id = gens::pick_from(ids.into_iter().map(str::to_string).collect());
    gens::vec((id, gens::bool()))
        .len(0..=6)
        .map(|entries| {
            entries
                .into_iter()
                .map(|(id, accepted)| {
                    (
                        id,
                        Decision {
                            accepted,
                            reason: "because".to_string(),
                        },
                    )
                })
                .collect()
        })
        .boxed()
}

#[ephyra::test(gen = (variant_candidates(), reviewer_decisions(), gens::int(0usize..=4), gens::int(0usize..=4)))]
fn variant_selection_follows_the_rules_in_order(
    input: (Vec<VariantSpec>, BTreeMap<String, Decision>, usize, usize),
) {
    let (candidates, decisions, max_ablations, max_baselines) = input;
    let selection = select_variants(&candidates, &decisions, max_ablations, max_baselines);

    // The reference: one pass, one rule at a time.
    let (mut expected, mut seen) = (Vec::new(), HashSet::new());
    let (mut ablations, mut baselines) = (0, 0);
    for candidate in &candidates {
        let id = candidate.variant_id.as_str();
        let repeated = !seen.insert(id);
        let refused = decisions.get(id).is_some_and(|decision| !decision.accepted);
        let counter = if candidate.kind == "ablation" {
            &mut ablations
        } else {
            &mut baselines
        };
        let cap = if candidate.kind == "ablation" {
            max_ablations
        } else {
            max_baselines
        };
        if RESERVED_IDS.contains(&id) || repeated || refused || *counter >= cap {
            continue;
        }
        *counter += 1;
        expected.push(id);
    }
    let accepted: Vec<&str> = selection
        .accepted
        .iter()
        .map(|variant| variant.variant_id.as_str())
        .collect();
    assert_eq!(accepted, expected);

    assert_eq!(
        selection.accepted.len() + selection.rejected.len(),
        candidates.len(),
        "every candidate has one fate"
    );
    assert!(
        selection
            .accepted
            .iter()
            .filter(|v| v.kind == "ablation")
            .count()
            <= max_ablations
    );
    assert!(
        selection
            .accepted
            .iter()
            .filter(|v| v.kind == "baseline")
            .count()
            <= max_baselines
    );
    let unique: HashSet<&str> = accepted.iter().copied().collect();
    assert_eq!(unique.len(), accepted.len(), "an id was accepted twice");
    assert!(
        accepted.iter().all(|id| !RESERVED_IDS.contains(id)),
        "a reserved id was accepted"
    );
    assert!(selection
        .rejected
        .iter()
        .all(|rejected| !rejected.reason.is_empty()));
    ephyra::cover(
        !selection.accepted.is_empty() && !selection.rejected.is_empty(),
        "some accepted, some rejected",
    );
}

fn seed_base() -> BoxGen<u64> {
    gens::frequency(vec![
        (4, gens::int(0u64..=40).boxed()),
        (2, gens::int(0u64..=u64::MAX).boxed()),
        (1, gens::int(u64::MAX - 40..=u64::MAX).boxed()),
    ])
    .boxed()
}

#[ephyra::test(gen = (seed_base(), gens::int(0usize..=40), seed_base(), gens::int(0usize..=40), gens::int(-45i64..=45), gens::bool()))]
fn final_seeds_are_held_out_for_any_configuration(input: (u64, usize, u64, usize, i64, bool)) {
    let (ablation_base, ablation_count, final_base, final_count, offset, nearby) = input;
    let ablation = AblationConfig {
        seed_base: ablation_base,
        num_replications: ablation_count,
        ..AblationConfig::default()
    };
    // Half the time the final seeds start inside or next to the ablation seeds.
    let final_base = if nearby {
        ablation_base.wrapping_add_signed(offset)
    } else {
        final_base
    };
    let config = FinalEvalConfig {
        seed_base: final_base,
        num_replications: final_count,
        ..FinalEvalConfig::default()
    };

    let used = ablation_seeds(&ablation);
    assert_eq!(used.len(), ablation_count);
    assert_eq!(
        used.iter().collect::<HashSet<_>>().len(),
        ablation_count,
        "ablation seeds repeat"
    );
    let held_out = final_seeds(&config, &used);
    assert_eq!(held_out.len(), final_count);
    assert_eq!(
        held_out.iter().collect::<HashSet<_>>().len(),
        final_count,
        "final seeds repeat"
    );
    let overlap: Vec<&u64> = held_out.iter().filter(|seed| used.contains(seed)).collect();
    assert!(
        overlap.is_empty(),
        "final seeds {overlap:?} were used by the ablation study"
    );
    ephyra::cover(
        final_seeds(&config, &[]) != held_out,
        "an ablation seed had to be skipped",
    );
    if final_seeds(&config, &[])
        .iter()
        .all(|seed| !used.contains(seed))
    {
        assert_eq!(
            held_out,
            final_seeds(&config, &[]),
            "nothing to skip, nothing skipped"
        );
    }
}

fn scores() -> BoxGen<Vec<f64>> {
    gens::frequency(vec![
        (
            4,
            gens::vec(gens::f64_range(-1e4..=1e4)).len(1..=12).boxed(),
        ),
        (
            2,
            gens::vec(gens::int(-3i32..=3).map(f64::from))
                .len(1..=8)
                .boxed(),
        ),
        (1, gens::vec(gens::constant(2.5)).len(1..=6).boxed()),
    ])
    .boxed()
}

// Each case bootstraps three comparisons, so the count is kept moderate.
#[ephyra::test(gen = (scores(), scores(), gens::int(0u64..=1000)), tests = 400)]
fn a_difference_is_zero_against_itself_and_negates_when_swapped(input: (Vec<f64>, Vec<f64>, u64)) {
    let (a, b, seed) = input;
    let (same, interval) = compare_to_control(&a, &a, seed);
    let same = same.unwrap();
    assert_eq!(same.mean_diff, 0.0);
    assert!(same.t.is_none_or(|t| t == 0.0) && same.p_value.is_none_or(|p| (p - 1.0).abs() < 1e-9));
    let (low, high) = interval.unwrap();
    assert!(
        low <= 0.0 && 0.0 <= high,
        "the bootstrap interval of a sample against itself is [{low}, {high}]"
    );

    let (forward, forward_interval) = compare_to_control(&a, &b, seed);
    let (backward, _) = compare_to_control(&b, &a, seed);
    let (forward, backward) = (forward.unwrap(), backward.unwrap());
    let scale = 1.0 + forward.mean_a.abs() + forward.mean_b.abs();
    assert!((forward.mean_diff + backward.mean_diff).abs() <= 1e-9 * scale);
    assert_eq!(forward.p_value.is_some(), backward.p_value.is_some());
    if let (Some(p), Some(q)) = (forward.p_value, backward.p_value) {
        assert!((p - q).abs() <= 1e-9 && (0.0..=1.0).contains(&p));
    }
    let (low, high) = forward_interval.unwrap();
    assert!(low <= high);
    assert_eq!(compare_to_control(&a, &[], seed), (None, None));
    assert_eq!(compare_to_control(&[], &b, seed), (None, None));
    if let Some(needed) = required_replications(&forward) {
        assert!(needed >= 2);
        let mut stronger = forward.clone();
        stronger.cohens_d = forward.cohens_d.map(|d| d * 2.0);
        assert!(
            required_replications(&stronger).unwrap() <= needed,
            "a larger effect needs no more replications"
        );
    }
}

fn replications() -> BoxGen<Vec<ReplicationResult>> {
    let score = gens::frequency(vec![
        (5, gens::f64_range(-100.0..=100.0).boxed()),
        (2, gens::int(0i32..=2).map(f64::from).boxed()),
        (1, gens::pick_from(vec![f64::NAN, f64::INFINITY]).boxed()),
    ]);
    gens::vec((score, gens::bool_with(0.75), gens::int(0u8..=2)))
        .len(0..=8)
        .map(|rows| {
            rows.into_iter()
                .enumerate()
                .map(|(index, (combined_score, correct, metrics))| {
                    let mut replication = ReplicationResult {
                        seed: index as u64 + 1,
                        combined_score,
                        correct,
                        ..Default::default()
                    };
                    if metrics > 0 {
                        replication
                            .public_metrics
                            .insert("runtime, ms".into(), json!(12.5));
                    }
                    if metrics > 1 {
                        replication
                            .public_metrics
                            .insert("note".into(), json!("a \"quoted\"\nnote"));
                    }
                    replication
                })
                .collect()
        })
        .boxed()
}

fn variant_result(id: &str, kind: &str, replications: Vec<ReplicationResult>) -> VariantResult {
    VariantResult {
        spec: VariantSpec {
            variant_id: id.into(),
            title: id.into(),
            kind: kind.into(),
            base_program: "best".into(),
            ..Default::default()
        },
        status: "completed".into(),
        replications,
        ..Default::default()
    }
}

#[ephyra::test(gen = (replications(), replications(), replications(), gens::int(0u64..=99)), tests = 600)]
fn a_study_is_summarised_over_correct_replications_and_tabulated_in_full(
    input: (
        Vec<ReplicationResult>,
        Vec<ReplicationResult>,
        Vec<ReplicationResult>,
        u64,
    ),
) {
    let (control_reps, first_reps, second_reps, seed) = input;
    let control_scores = correct_scores(&control_reps);
    let stochastic = is_stochastic(&control_scores);
    let mut variants = vec![
        variant_result(CONTROL_ID, "control", control_reps),
        variant_result("ablation_one", "ablation", first_reps),
        variant_result("baseline, \"two\"", "baseline", second_reps),
    ];
    for variant in &mut variants {
        summarise_variant(variant, &control_scores, stochastic, seed);
        let usable = variant
            .replications
            .iter()
            .filter(|r| r.correct && r.combined_score.is_finite())
            .count();
        match &variant.summary {
            Some(summary) => {
                assert_eq!(summary.n, usable);
                assert!(summary.min <= summary.mean + 1e-9 && summary.mean <= summary.max + 1e-9);
                assert!(summary.std >= 0.0 && summary.ci95_low <= summary.ci95_high);
            }
            None => assert_eq!(usable, 0),
        }
        assert!((0.0..=1.0).contains(&variant.correct_rate));
        let correct = variant.replications.iter().filter(|r| r.correct).count();
        assert_eq!(
            variant.correct_rate * variant.replications.len() as f64,
            correct as f64
        );
        if variant.spec.variant_id == CONTROL_ID {
            assert!(variant.comparison_to_control.is_none() && variant.bootstrap_ci95.is_none());
        } else {
            assert_eq!(
                variant.comparison_to_control.is_some(),
                usable > 0 && !control_scores.is_empty()
            );
            assert!(
                stochastic || variant.bootstrap_ci95.is_none(),
                "an interval over a deterministic evaluator"
            );
            if let Some(comparison) = &variant.comparison_to_control {
                assert!(comparison.mean_diff.is_finite());
                assert!(
                    variant.notes.starts_with("Difference from the control: "),
                    "{}",
                    variant.notes
                );
            }
        }
        let note = difference_note(variant, stochastic).to_lowercase();
        assert!(
            !note.contains("no effect") && !note.contains("no difference"),
            "{note}"
        );
    }
    ephyra::cover(stochastic, "stochastic control");

    let total: usize = variants
        .iter()
        .map(|variant| variant.replications.len())
        .sum();
    let summary = AblationSummary {
        primary_metric_name: "combined_score".into(),
        anchor_variant_id: CONTROL_ID.into(),
        evaluator_is_stochastic: stochastic,
        variants,
        ..Default::default()
    };
    let rows = run_summary_rows(&summary, "ablations");
    assert_eq!(rows.len(), total, "one row per replication");
    assert!(rows.iter().all(|row| row.len() == RUN_SUMMARY_HEADER.len()));
    let text = csv::encode(&RUN_SUMMARY_HEADER, &rows);
    let parsed = csv::parse(&text);
    assert_eq!(parsed.len(), total + 1);
    assert_eq!(&parsed[1..], rows.as_slice());
    let metric_rows = metric_value_rows(&summary);
    assert!(metric_rows.len() >= total && metric_rows.iter().all(|row| row.len() == 5));
    assert_eq!(
        metric_rows.iter().filter(|row| row[4] == "true").count(),
        total,
        "one primary metric per replication"
    );
    assert!(
        metric_rows.iter().all(|row| row[2] != "note"),
        "a text metric was tabulated as a number"
    );
}

fn cell() -> BoxGen<String> {
    let character = gens::frequency(vec![
        (3, gens::char().boxed()),
        (
            2,
            gens::pick_from(vec![',', '"', '\n', '\r', ' ', '\t']).boxed(),
        ),
    ]);
    gens::string_len(character, 0..=10).boxed()
}

#[ephyra::test(gen = gens::vec(gens::vec(cell()).len(1..=5)).len(0..=6))]
fn any_cell_survives_the_csv_round_trip(rows: Vec<Vec<String>>) {
    let text: String = rows.iter().map(|row| csv::encode_row(row)).collect();
    assert_eq!(csv::parse(&text), rows);
    ephyra::cover(text.contains('"'), "a quoted cell");
}

/// `k` units of `10^-decimals`, written without floating point.
fn decimal(k: i64, decimals: u32) -> String {
    let (sign, magnitude) = (if k < 0 { "-" } else { "" }, k.unsigned_abs());
    if decimals == 0 {
        return format!("{sign}{magnitude}");
    }
    let scale = 10u64.pow(decimals);
    format!(
        "{sign}{}.{:0width$}",
        magnitude / scale,
        magnitude % scale,
        width = decimals as usize
    )
}

#[ephyra::test(
    gen = (gens::vec(gens::int(-1_000_000i64..=1_000_000)).len(1..=5), gens::int(0usize..=4), gens::int(0u32..=4),
           gens::int(-3i64..=3), gens::bool_with(0.3), gens::bool_with(0.2), gens::int(-2_000_000i64..=2_000_000)),
    tests = 20000
)]
fn a_number_is_grounded_exactly_when_a_key_number_rounds_to_it(
    input: (Vec<i64>, usize, u32, i64, bool, bool, i64),
) {
    // Key numbers are thousandths; the token is `k` units of 10^-decimals,
    // placed a few units from a key, or anywhere.
    let (thousandths, pick, decimals, offset, percent, anywhere, free) = input;
    let near = thousandths[pick % thousandths.len()] * if percent { 100 } else { 1 };
    let unit = 10i64.pow(decimals);
    let k = if anywhere {
        free * unit / 1000
    } else {
        near * unit / 1000 + offset
    };
    let token = decimal(k, decimals);
    let keys: BTreeMap<String, f64> = thousandths
        .iter()
        .enumerate()
        .map(|(index, value)| (format!("key_{index}"), *value as f64 / 1000.0))
        .collect();

    // The reference, in millionths: half a unit of the last digit shown.
    let token_millionths = k * 10i64.pow(6 - decimals);
    let half_unit = 5 * 10i64.pow(5 - decimals);
    let expected = thousandths.iter().any(|value| {
        (value * 1000 - token_millionths).abs() <= half_unit
            || (percent && (value * 100_000 - token_millionths).abs() <= half_unit)
    });

    let sentence = format!(
        "For `gen_7` (model-3x) the measured value is {token}{} overall.",
        if percent { "%" } else { "" }
    );
    assert_eq!(
        claim_is_grounded(&sentence, &keys),
        expected,
        "token {token} against thousandths {thousandths:?} (percent: {percent})"
    );
    if !expected {
        assert_eq!(ungrounded_numbers(&sentence, &keys), [token]);
    }
    ephyra::cover(expected, "grounded");
    ephyra::cover(!expected, "not grounded");
}

#[derive(Debug, Clone, ephyra::Arbitrary)]
struct Insert {
    /// Which earlier program is the parent.
    #[ephyra(gen = gens::int(0usize..=63))]
    parent: usize,
    #[ephyra(gen = gens::int(-30i32..=30))]
    tenths: i32,
    #[ephyra(gen = gens::bool_with(0.7))]
    correct: bool,
    #[ephyra(gen = gens::bool_with(0.1))]
    orphan: bool,
    #[ephyra(gen = gens::int(0usize..=5))]
    patch: usize,
    #[ephyra(gen = gens::int(0usize..=4))]
    arm: usize,
    #[ephyra(gen = gens::int(0u32..=50))]
    cost_cents: u32,
    /// 0: no rubrics; 1: all pass; 2: one fails; 3: unassessed; 4: malformed.
    #[ephyra(gen = gens::int(0u8..=4))]
    rubric: u8,
    /// 0 to 3 replace the stored score with NaN, infinity or an extreme.
    #[ephyra(gen = gens::int(0u8..=15))]
    oddity: u8,
}

const PATCH_TYPES: [&str; 5] = ["diff", "full", "cross", "fix", "we`ird 7\nname"];
const ARMS: [&str; 4] = ["model-a", "gpt-5.6-luna", "agent:claude", ""];

fn run_data(inserts: &[Insert], islands: usize, migrate: bool, selected: usize) -> RunData {
    let db = ProgramDb::open(
        None,
        DatabaseConfig {
            num_islands: islands,
            migration_rate: if migrate { 0.5 } else { 0.0 },
            migration_interval: 2,
            seed: Some(3),
            ..DatabaseConfig::default()
        },
    )
    .unwrap();
    let mut ids: Vec<String> = Vec::new();
    let mut attempts = Vec::new();
    for (generation, insert) in inserts.iter().enumerate() {
        let mut program = Program::new(format!("x = {}\n", insert.tenths), "python");
        program.generation = generation as u32;
        program.combined_score = f64::from(insert.tenths) / 10.0;
        program.correct = insert.correct;
        program.timestamp = generation as f64;
        if !ids.is_empty() && !insert.orphan {
            program.parent_id = Some(ids[insert.parent % ids.len()].clone());
        }
        if let Some(patch_type) = PATCH_TYPES.get(insert.patch) {
            program
                .metadata
                .insert("patch_type".into(), json!(patch_type));
            program
                .metadata
                .insert("patch_name".into(), json!(format!("edit {generation} `x`")));
        }
        if let Some(arm) = ARMS.get(insert.arm) {
            program.metadata.insert("arm".into(), json!(arm));
        }
        program.metadata.insert(
            "api_costs".into(),
            json!(f64::from(insert.cost_cents) / 100.0),
        );
        match insert.rubric {
            1 => program.metadata.insert("rubric_status".into(), json!({"overall_pass": true, "rubrics": [{"pass": true, "assessment": "passed"}]})),
            2 => program.metadata.insert(
                "rubric_status".into(),
                json!({"overall_pass": false, "rubrics": [{"pass": true, "assessment": "passed"}, {"pass": false, "assessment": "failed"}]}),
            ),
            3 => program.metadata.insert("rubric_status".into(), json!({"rubrics": [{"pass": false, "assessment": "not_assessed"}]})),
            4 => program.metadata.insert("rubric_status".into(), json!({"overall_pass": "yes", "rubrics": "none"})),
            _ => None,
        };
        db.add(&mut program).unwrap();
        ids.push(program.id.clone());
        if insert.cost_cents % 7 == 0 {
            attempts.push(AttemptRecord {
                generation: generation as u32,
                stage: "proposal".into(),
                status: "failed".into(),
                details: json!({"cost_usd": f64::from(insert.cost_cents) / 50.0}),
                created_at: 0.0,
            });
        }
    }
    let mut programs = db.all().unwrap();
    // What a database cannot hold can still reach an analysis through code.
    for (program, insert) in programs.iter_mut().zip(inserts.iter().cycle()) {
        match insert.oddity {
            0 => program.combined_score = f64::NAN,
            1 => program.combined_score = f64::INFINITY,
            2 => program.combined_score = 1e308,
            3 => program.combined_score = -1e308,
            _ => {}
        }
    }
    let selected_program_id = match selected % 4 {
        0 => None,
        1 => Some("not-in-the-database".to_string()),
        _ => ids.get(selected % ids.len().max(1)).cloned(),
    };
    RunData {
        task_name: "generated".into(),
        programs,
        selected_program_id,
        attempts,
        cost: None,
    }
}

#[ephyra::test(
    gen = (gens::vec(ephyra::any::<Insert>()).len(0..=24), gens::int(1usize..=3), gens::bool(), gens::int(0usize..=40), gens::int(0usize..=7)),
    tests = 1500
)]
fn every_built_in_analysis_is_total_and_grounded_on_any_run(
    input: (Vec<Insert>, usize, bool, usize, usize),
) {
    let (inserts, islands, migrate, selected, cap) = input;
    let data = run_data(&inserts, islands, migrate, selected);
    let mut with_data = 0;
    for family in registry() {
        let Some(output) = (family.compute)(&data) else {
            assert!(
                family.name == "rubric_outcomes"
                    || family.name == "lineage_of_best"
                    || data.programs.iter().all(|p| p.generation == 0),
                "{} had nothing to say about a run with proposals",
                family.name
            );
            continue;
        };
        with_data += 1;
        for (name, value) in &output.key_numbers {
            assert!(
                value.is_finite(),
                "{}: key number {name} is {value}",
                family.name
            );
        }
        assert_eq!(
            ungrounded_numbers(&output.claim, &output.key_numbers),
            Vec::<String>::new(),
            "{}: `{}` cites numbers the analysis did not compute",
            family.name,
            output.claim
        );
        assert!(
            !output.claim.is_empty()
                && !output.claim.contains("NaN")
                && !output.claim.contains("inf")
        );
        assert!(
            output
                .table
                .rows
                .iter()
                .all(|row| row.len() == output.table.header.len()),
            "{}: ragged table",
            family.name
        );
        let header: Vec<&str> = output.table.header.iter().map(String::as_str).collect();
        let parsed = csv::parse(&csv::encode(&header, &output.table.rows));
        assert_eq!(
            &parsed[1..],
            output.table.rows.as_slice(),
            "{}: the table does not survive CSV",
            family.name
        );
        assert!(output.chart.starts_with("<svg") && output.chart.trim_end().ends_with("</svg>"));
        assert!(
            !output.chart.contains("NaN") && !output.chart.contains("inf"),
            "{}: a non-finite coordinate",
            family.name
        );
    }
    ephyra::cover(with_data >= 4, "most families have data");

    let proposals = default_proposals(&data);
    assert_eq!(
        proposals.len(),
        with_data,
        "exactly the families with data propose themselves"
    );
    let (accepted, rejected) = select_analyses(&proposals, &BTreeMap::new(), cap);
    assert_eq!(accepted.len(), proposals.len().min(cap));
    assert_eq!(accepted.len() + rejected.len(), proposals.len());
    assert_eq!(
        accepted.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
        proposals
            .iter()
            .take(cap)
            .map(|p| p.id.as_str())
            .collect::<Vec<_>>()
    );
}

fn analysis_candidates() -> BoxGen<Vec<AnalysisSpec>> {
    let families = vec![
        "search_trajectory",
        "operator_effectiveness",
        "lineage_of_best",
        "cost_efficiency",
        "component_ablation",
        "made_up",
    ];
    let family = gens::pick_from(families.into_iter().map(str::to_string).collect());
    gens::vec((
        gens::int(0u8..=6),
        family,
        gens::bool_with(0.15),
        gens::bool_with(0.15),
    ))
    .len(0..=12)
    .map(|rows| {
        rows.into_iter()
            .map(|(id, family, requires_retrain, reruns)| AnalysisSpec {
                id: format!("analysis_{id}"),
                title: "t".into(),
                family,
                requires_retrain,
                implementation_plan: if reruns {
                    vec!["Fine-tune and rerun evaluation".into()]
                } else {
                    vec!["read the database".into()]
                },
                ..AnalysisSpec::default()
            })
            .collect()
    })
    .boxed()
}

#[ephyra::test(gen = (analysis_candidates(), gens::vec((gens::int(0u8..=6), gens::bool())).len(0..=5), gens::int(0usize..=5)))]
fn posthoc_selection_keeps_one_analysis_per_family_within_the_cap(
    input: (Vec<AnalysisSpec>, Vec<(u8, bool)>, usize),
) {
    let (candidates, verdicts, cap) = input;
    let decisions: BTreeMap<String, (bool, String)> = verdicts
        .into_iter()
        .map(|(id, accepted)| (format!("analysis_{id}"), (accepted, String::new())))
        .collect();
    let (accepted, rejected) = select_analyses(&candidates, &decisions, cap);
    assert_eq!(accepted.len() + rejected.len(), candidates.len());
    assert!(accepted.len() <= cap);
    let families: HashSet<&str> = accepted.iter().map(|a| a.family.as_str()).collect();
    let ids: HashSet<&str> = accepted.iter().map(|a| a.id.as_str()).collect();
    assert_eq!(
        (families.len(), ids.len()),
        (accepted.len(), accepted.len()),
        "a family or an id is repeated"
    );
    for analysis in &accepted {
        assert!(
            !analysis.family.contains("ablation")
                && !analysis.requires_retrain
                && analysis.family != "made_up"
        );
        assert!(!analysis
            .implementation_plan
            .iter()
            .any(|step| step.to_lowercase().contains("rerun evaluation")));
        assert!(
            decisions.get(&analysis.id).is_none_or(|(ok, _)| *ok),
            "the reviewer rejected {}",
            analysis.id
        );
    }
    // Order: the accepted analyses appear in the order they were proposed.
    let mut cursor = 0;
    for analysis in &accepted {
        let at = candidates[cursor..]
            .iter()
            .position(|candidate| candidate == analysis)
            .expect("accepted out of order");
        cursor += at + 1;
    }
    ephyra::cover(!accepted.is_empty(), "something accepted");
}

fn episodes(
    scores: std::ops::RangeInclusive<f64>,
    correct: f64,
    count: std::ops::RangeInclusive<usize>,
) -> BoxGen<Vec<FinalEpisode>> {
    gens::vec((gens::f64_range(scores), gens::bool_with(correct)))
        .len(count)
        .map(|rows| {
            rows.into_iter()
                .enumerate()
                .map(|(index, (score, correct))| FinalEpisode {
                    seed: index as u64,
                    score,
                    correct,
                    duration_seconds: 0.0,
                })
                .collect()
        })
        .boxed()
}

/// The candidate's and the baseline's final episodes: unrelated, or a
/// candidate that is correct throughout and scores well above the baseline.
fn final_episodes() -> BoxGen<(Vec<FinalEpisode>, Vec<FinalEpisode>)> {
    gens::frequency(vec![
        (
            3,
            (
                episodes(-10.0..=10.0, 0.8, 0..=6),
                episodes(-10.0..=10.0, 0.8, 0..=6),
            )
                .boxed(),
        ),
        (
            1,
            (
                episodes(3.0..=4.0, 1.0, 3..=6),
                episodes(0.0..=1.0, 0.9, 3..=6),
            )
                .boxed(),
        ),
    ])
    .boxed()
}

#[ephyra::test(gen = (final_episodes(), gens::bool_with(0.3), gens::bool()))]
fn qualification_is_every_episode_correct(
    input: ((Vec<FinalEpisode>, Vec<FinalEpisode>), bool, bool),
) {
    // A selection that is the baseline normally shares its episodes; the
    // verdict must not depend on that.
    let ((candidate, baseline), same_program, shared_episodes) = input;
    let baseline = if same_program && shared_episodes {
        candidate.clone()
    } else {
        baseline
    };
    let mut manifest = FinalManifest {
        episodes: candidate.clone(),
        baseline_episodes: baseline,
        ..Default::default()
    };
    assess(&mut manifest, same_program);
    let all_correct = !candidate.is_empty() && candidate.iter().all(|episode| episode.correct);
    assert_eq!(manifest.correct, all_correct);
    assert_eq!(manifest.outcome == QUALIFIED, all_correct);
    assert_eq!(
        manifest.candidate.as_ref().map_or(0, |summary| summary.n),
        candidate.iter().filter(|e| e.correct).count()
    );
    assert_eq!(
        manifest.limitations.len(),
        if all_correct { 0 } else { 3 } + usize::from(same_program)
    );
    if same_program {
        assert!(
            !manifest.improvement_supported,
            "a program was found to improve on itself"
        );
    }
    if manifest.improvement_supported {
        let comparison = manifest
            .comparison
            .as_ref()
            .expect("an improvement without a comparison");
        assert!(comparison.mean_diff > 0.0);
    }
    ephyra::cover(manifest.improvement_supported, "improvement supported");
    // Assessing twice changes nothing.
    let once = manifest.clone();
    assess(&mut manifest, same_program);
    assert_eq!(manifest, once);
}

fn json_leaf() -> BoxGen<Value> {
    gens::one_of(vec![
        gens::constant(Value::Null).boxed(),
        gens::bool().map(Value::Bool).boxed(),
        gens::int(-5i64..=5).map(|n| json!(n)).boxed(),
        gens::string_len(gens::char(), 0..=10)
            .map(Value::String)
            .boxed(),
        gens::pick_from(vec![
            json!("ablation"),
            json!("baseline"),
            json!("Ablation"),
            json!("control_best"),
            json!("search_trajectory"),
            json!("../up"),
        ])
        .boxed(),
    ])
    .boxed()
}

fn json_value() -> BoxGen<Value> {
    gens::recursive(2, json_leaf(), |inner| {
        gens::frequency(vec![
            (2, json_leaf()),
            (
                1,
                gens::vec(inner.clone())
                    .len(0..=3)
                    .map(Value::Array)
                    .boxed(),
            ),
        ])
        .boxed()
    })
    .boxed()
}

/// A model reply shaped like a bundle, with fields that are usually of the
/// right type and sometimes anything at all.
fn model_bundle() -> BoxGen<Value> {
    let keys = [
        "variant_id",
        "id",
        "analysis_id",
        "title",
        "kind",
        "family",
        "base_program",
        "accepted",
        "reason",
        "requires_retrain",
        "implementation_plan",
        "target_components",
        "expected_outputs",
    ];
    let entry = gens::vec((gens::int(0usize..=12), json_value()))
        .len(0..=8)
        .map(move |fields| {
            Value::Object(
                fields
                    .into_iter()
                    .map(|(key, value)| (keys[key].to_string(), value))
                    .collect(),
            )
        });
    let list = gens::frequency(vec![
        (5, gens::vec(entry).len(0..=6).map(Value::Array).boxed()),
        (1, json_value()),
    ])
    .boxed();
    (list.clone(), list.clone(), list)
        .map(|(candidates, analyses, decisions)| json!({"candidates": candidates, "analyses": analyses, "decisions": decisions}))
        .boxed()
}

#[ephyra::test(gen = gens::frequency(vec![(6, model_bundle()), (1, json_value())]))]
fn model_replies_of_any_shape_parse_into_safe_candidates(reply: Value) {
    for candidate in parse_candidates(&reply) {
        let id = &candidate.variant_id;
        assert!(
            !id.is_empty()
                && id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
            "{id:?}"
        );
        assert!(candidate.kind == "ablation" || candidate.kind == "baseline");
        assert!(candidate.base_program == "best" || candidate.base_program == "reference");
        assert!(!candidate.title.trim().is_empty());
        assert!(candidate
            .implementation_plan
            .iter()
            .all(|step| !step.trim().is_empty()));
        ephyra::cover(true, "a candidate parsed");
    }
    for proposal in parse_proposals(&reply) {
        let id = &proposal.id;
        assert!(
            !id.is_empty() && id.len() <= 128 && !id.contains(['/', '.', ' ']),
            "{id:?}"
        );
        assert!(id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-'));
        assert!(!proposal.title.is_empty() && !proposal.family.is_empty());
        assert_eq!(proposal.family, proposal.family.to_ascii_lowercase());
    }
    for (id, decision) in parse_decisions(&reply) {
        assert!(!id.is_empty());
        let _ = decision.accepted;
    }
    assert!(genesis_evidence::posthoc::parse_decisions(&reply)
        .keys()
        .all(|id| !id.is_empty()));
}

fn any_float() -> BoxGen<f64> {
    gens::frequency(vec![
        (4, gens::f64_range(-1e6..=1e6).boxed()),
        (3, gens::f64_bits().boxed()),
        (
            2,
            gens::pick_from(vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::MAX,
                f64::MIN,
                0.0,
                -0.0,
                f64::MIN_POSITIVE,
            ])
            .boxed(),
        ),
    ])
    .boxed()
}

/// The points of one series, and whether they are joined by a line.
type Line = (Vec<(f64, f64)>, bool);

/// Text that needs escaping and can never spell `NaN` or `inf` itself.
fn label() -> BoxGen<String> {
    gens::string_len(
        gens::pick_from(vec![
            '<', '>', '&', '"', '\'', 'x', ' ', '\u{1}', '\n', 'é', '7',
        ]),
        0..=10,
    )
    .boxed()
}

#[ephyra::test(gen = (gens::vec((gens::vec((any_float(), any_float())).len(0..=6), gens::bool())).len(0..=3),
                      gens::vec((label(), any_float())).len(0..=5),
                      label()))]
fn charts_draw_any_numbers_without_a_non_finite_coordinate(
    input: (Vec<Line>, Vec<(String, f64)>, String),
) {
    let (lines, bars, label) = input;
    let series: Vec<Series> = lines
        .into_iter()
        .enumerate()
        .map(|(index, (points, line))| Series {
            name: format!("{label} {index}"),
            points,
            line,
        })
        .collect();
    for chart in [
        xy_chart(&label, &label, "y", &series),
        bar_chart(&label, &label, &bars),
    ] {
        assert!(chart.starts_with("<svg") && chart.trim_end().ends_with("</svg>"));
        assert!(
            !chart.contains("NaN") && !chart.contains("inf"),
            "a non-finite number reached the chart"
        );
        assert_eq!(
            chart.matches("<svg").count(),
            1,
            "a label broke out of its element"
        );
        assert!(
            !chart.contains('\u{1}'),
            "a control character reached the chart"
        );
        assert_eq!(
            chart.matches('<').count(),
            chart.matches('>').count(),
            "unbalanced markup"
        );
    }
}

/// Text made of arbitrary characters, the characters numbers are written
/// with, and exponents beyond what a float holds.
fn scanned_text() -> BoxGen<String> {
    let fragment = gens::frequency(vec![
        (3, gens::char().map(String::from).boxed()),
        (
            3,
            gens::pick_from(vec![
                '0', '1', '9', '.', ',', '-', '+', '%', 'e', 'E', '`', ' ', '\u{2212}', '_', '/',
            ])
            .map(String::from)
            .boxed(),
        ),
        (
            1,
            gens::pick_from(vec![
                " 8e-987",
                " 1e999",
                "e-400",
                "e400",
                "e99999999999",
                "E-99999999999",
            ])
            .map(str::to_string)
            .boxed(),
        ),
    ]);
    gens::vec(fragment)
        .len(0..=30)
        .map(|fragments| fragments.concat())
        .boxed()
}

#[ephyra::test(gen = scanned_text())]
fn the_number_scanner_reads_any_text(text: String) {
    let tokens = numbers_in(&text);
    for token in &tokens {
        assert!(
            text.contains(&token.text),
            "a token that is not in the text"
        );
        assert!(!token.value.is_nan() && !token.tolerance.is_nan());
        ephyra::cover(true, "a number was found");
        ephyra::cover(!token.representable(), "a number a float cannot hold");
    }
    // A text is grounded by its own numbers, except those a float cannot
    // hold: nothing grounds those, not even their own reading.
    let own: BTreeMap<String, f64> = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| (index.to_string(), token.value))
        .collect();
    let unheld: Vec<String> = tokens
        .iter()
        .filter(|token| !token.representable())
        .map(|token| token.text.clone())
        .collect();
    assert_eq!(ungrounded_numbers(&text, &own), unheld);
    let everything: BTreeMap<String, f64> = [0.0, 0.3, 1.0, -1.0, f64::MAX, f64::MIN_POSITIVE]
        .iter()
        .map(|value| (value.to_string(), *value))
        .collect();
    assert!(unheld
        .iter()
        .all(|text| ungrounded_numbers(text, &everything).len() == numbers_in(text).len()));
}
