//! Turning replications into the study's numbers and tables. Everything
//! here is computed by code; no model writes a number.

use super::variants::{RejectedVariant, CONTROL_ID};
use genesis_core::stats::{self, Comparison};
use genesis_scientist::artifacts::{AblationSummary, ReplicationResult, VariantResult};
use serde_json::Value;
use std::collections::BTreeMap;

const BOOTSTRAP_RESAMPLES: usize = 2000;
/// (z for a two-sided 5% test + z for 80% power) squared, times two: the
/// per-group sample size for a standardised effect `d` is this over `d²`.
const POWER_FACTOR: f64 = 2.0 * (1.959_964 + 0.841_621) * (1.959_964 + 0.841_621);

/// Scores of the replications the evaluator judged correct.
pub fn correct_scores(replications: &[ReplicationResult]) -> Vec<f64> {
    replications
        .iter()
        .filter(|replication| replication.correct && replication.combined_score.is_finite())
        .map(|replication| replication.combined_score)
        .collect()
}

/// False when every correct score of the control is the same: the evaluator
/// then ignores the seed, and an interval over replications means nothing.
pub fn is_stochastic(control_scores: &[f64]) -> bool {
    control_scores.windows(2).any(|pair| pair[0] != pair[1])
}

/// A variant against the control: Welch's test and a percentile-bootstrap
/// interval of `mean(variant) - mean(control)`. Both are `None` when either
/// side has no correct score.
pub fn compare_to_control(
    variant: &[f64],
    control: &[f64],
    bootstrap_seed: u64,
) -> (Option<Comparison>, Option<(f64, f64)>) {
    if variant.is_empty() || control.is_empty() {
        return (None, None);
    }
    (
        Some(stats::compare(variant, control)),
        stats::bootstrap_diff_ci(variant, control, BOOTSTRAP_RESAMPLES, bootstrap_seed),
    )
}

/// Replications per variant a two-sided 5% test would need to detect the
/// observed standardised difference with 80% power. `None` when the effect
/// size is unknown or zero.
pub fn required_replications(comparison: &Comparison) -> Option<u64> {
    let d = comparison.cohens_d.filter(|d| d.is_finite() && *d != 0.0)?;
    let needed = (POWER_FACTOR / (d * d)).ceil();
    (needed.is_finite() && needed < 1e12).then_some(needed.max(2.0) as u64)
}

/// Six decimals in the ordinary range; scientific notation for a value so
/// small or so large that six decimals would hide it.
fn num(value: f64) -> String {
    let magnitude = value.abs();
    if value == 0.0 || (1e-3..1e9).contains(&magnitude) {
        format!("{value:.6}")
    } else {
        format!("{value:.5e}")
    }
}

fn signed(value: f64) -> String {
    let text = num(value);
    if value >= 0.0 && !text.starts_with('+') {
        format!("+{text}")
    } else {
        text
    }
}

/// One sentence on how a variant differs from the control. A difference
/// whose interval includes zero is reported as it is, with the sample size
/// it would take to resolve, never as "no effect".
pub fn difference_note(result: &VariantResult, stochastic: bool) -> String {
    let Some(comparison) = &result.comparison_to_control else {
        return if result.summary.is_none() && !result.replications.is_empty() {
            "No replication of this variant was correct, so no score is compared with the control."
                .to_string()
        } else {
            String::new()
        };
    };
    let delta = signed(comparison.mean_diff);
    if !stochastic {
        return format!(
            "Difference from the control: {delta} (mean {} against {}). The evaluator returned the same score for \
             every seed, so this compares means and carries no interval.",
            num(comparison.mean_a),
            num(comparison.mean_b)
        );
    }
    let (Some(low), Some(high), Some(p)) = (
        comparison.diff_ci95_low,
        comparison.diff_ci95_high,
        comparison.p_value,
    ) else {
        return format!(
            "Difference from the control: {delta} (mean {} against {}); too few distinct correct replications for \
             an interval.",
            num(comparison.mean_a),
            num(comparison.mean_b)
        );
    };
    let mut note = format!(
        "Difference from the control: {delta} (95% CI {} to {}, Welch p = {p:.4}).",
        num(low),
        num(high)
    );
    if low <= 0.0 && high >= 0.0 {
        note.push_str(" The interval includes zero.");
        if let (Some(needed), Some(d)) = (required_replications(comparison), comparison.cohens_d) {
            let have = comparison.n_a.min(comparison.n_b) as u64;
            if needed > have {
                note.push_str(&format!(
                    " A difference of this size (Cohen's d = {d:.2}) would need about {needed} replications per \
                     variant to resolve at 80% power; this study ran {have}."
                ));
            }
        }
    }
    note
}

/// Fill in a variant's statistics from its replications.
pub fn summarise_variant(
    result: &mut VariantResult,
    control_scores: &[f64],
    stochastic: bool,
    bootstrap_seed: u64,
) {
    let scores = correct_scores(&result.replications);
    result.summary = (!scores.is_empty()).then(|| stats::summarize(&scores));
    result.correct_rate = if result.replications.is_empty() {
        0.0
    } else {
        result
            .replications
            .iter()
            .filter(|replication| replication.correct)
            .count() as f64
            / result.replications.len() as f64
    };
    if result.spec.variant_id == CONTROL_ID {
        result.comparison_to_control = None;
        result.bootstrap_ci95 = None;
        return;
    }
    let (comparison, bootstrap) = compare_to_control(&scores, control_scores, bootstrap_seed);
    result.comparison_to_control = comparison;
    result.bootstrap_ci95 = if stochastic { bootstrap } else { None };
    let note = difference_note(result, stochastic);
    if !note.is_empty() {
        result.notes = if result.notes.is_empty() {
            note
        } else {
            format!("{} {note}", result.notes)
        };
    }
}

pub fn evaluation_id(variant_id: &str, seed: u64) -> String {
    format!("{variant_id}_seed_{seed}")
}

pub const RUN_SUMMARY_HEADER: [&str; 11] = [
    "evaluation_id",
    "variant_id",
    "variant_kind",
    "base_program",
    "seed",
    "correct",
    "combined_score",
    "primary_metric_name",
    "primary_metric_value",
    "program_dir",
    "results_dir",
];

pub const METRIC_VALUES_HEADER: [&str; 5] = [
    "evaluation_id",
    "variant_id",
    "metric_name",
    "metric_value",
    "is_primary_metric",
];

/// One row per replication of every variant that ran.
pub fn run_summary_rows(summary: &AblationSummary, ablations_dir: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for variant in &summary.variants {
        let id = &variant.spec.variant_id;
        for replication in &variant.replications {
            rows.push(vec![
                evaluation_id(id, replication.seed),
                id.clone(),
                variant.spec.kind.clone(),
                variant.spec.base_program.clone(),
                replication.seed.to_string(),
                replication.correct.to_string(),
                replication.combined_score.to_string(),
                summary.primary_metric_name.clone(),
                replication.combined_score.to_string(),
                format!("{ablations_dir}/variants/{id}/program"),
                format!(
                    "{ablations_dir}/variants/{id}/rep_{}/results",
                    replication.seed
                ),
            ]);
        }
    }
    rows
}

/// One row per numeric metric of every replication, the primary one first.
pub fn metric_value_rows(summary: &AblationSummary) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for variant in &summary.variants {
        let id = &variant.spec.variant_id;
        for replication in &variant.replications {
            let evaluation = evaluation_id(id, replication.seed);
            rows.push(vec![
                evaluation.clone(),
                id.clone(),
                summary.primary_metric_name.clone(),
                replication.combined_score.to_string(),
                "true".to_string(),
            ]);
            for (name, value) in &replication.public_metrics {
                if let Some(number) = value.as_f64().filter(|_| !matches!(value, Value::Bool(_))) {
                    rows.push(vec![
                        evaluation.clone(),
                        id.clone(),
                        name.clone(),
                        number.to_string(),
                        "false".to_string(),
                    ]);
                }
            }
        }
    }
    rows
}

fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

/// The results table, in Markdown.
pub fn results_table(summary: &AblationSummary) -> String {
    let mut table = String::from(
        "| Variant | Kind | Correct | n | Mean | Std | 95% CI | Δ vs control | 95% CI of Δ | Welch p |\n\
         |---|---|---|---|---|---|---|---|---|---|\n",
    );
    for variant in &summary.variants {
        let id = cell(&variant.spec.variant_id);
        let kind = cell(&variant.spec.kind);
        if variant.status != "completed" {
            table.push_str(&format!(
                "| `{id}` | {kind} | {} | | | | | | | |\n",
                cell(&variant.status)
            ));
            continue;
        }
        let correct = format!(
            "{}/{}",
            variant
                .replications
                .iter()
                .filter(|replication| replication.correct)
                .count(),
            variant.replications.len()
        );
        let (n, mean, std, interval) = match &variant.summary {
            Some(s) if summary.evaluator_is_stochastic && s.n > 1 => (
                s.n.to_string(),
                num(s.mean),
                num(s.std),
                format!("{} to {}", num(s.ci95_low), num(s.ci95_high)),
            ),
            Some(s) => (s.n.to_string(), num(s.mean), num(s.std), "n/a".to_string()),
            None => ("0".to_string(), String::new(), String::new(), String::new()),
        };
        let (delta, delta_interval, p) = match &variant.comparison_to_control {
            Some(c) => (
                signed(c.mean_diff),
                match (c.diff_ci95_low, c.diff_ci95_high) {
                    (Some(low), Some(high)) if summary.evaluator_is_stochastic => {
                        format!("{} to {}", num(low), num(high))
                    }
                    _ => "n/a".to_string(),
                },
                match c.p_value {
                    Some(p) if summary.evaluator_is_stochastic => format!("{p:.4}"),
                    _ => "n/a".to_string(),
                },
            ),
            None if variant.spec.variant_id == summary.anchor_variant_id => {
                ("(anchor)".to_string(), String::new(), String::new())
            }
            None => (String::new(), String::new(), String::new()),
        };
        table.push_str(&format!(
            "| `{id}` | {kind} | {correct} | {n} | {mean} | {std} | {interval} | {delta} | {delta_interval} | {p} |\n"
        ));
    }
    table
}

/// What the README needs besides the summary itself.
pub struct ReadmeInputs<'a> {
    pub summary: &'a AblationSummary,
    pub selection_kind: &'a str,
    pub search_score: f64,
    pub task_summary: Option<&'a str>,
    /// Model-written description per variant id; the hypothesis stands in
    /// for a missing one.
    pub variant_summaries: &'a BTreeMap<String, String>,
    pub rejected: &'a [RejectedVariant],
}

pub fn readme(inputs: &ReadmeInputs<'_>) -> String {
    let summary = inputs.summary;
    let mut out = format!("# Ablation study: {}\n\n", summary.task);
    if let Some(task_summary) = inputs
        .task_summary
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        out.push_str(&format!("{task_summary}\n\n"));
    }
    let seeds = match (summary.seeds.first(), summary.seeds.last()) {
        (Some(first), Some(last)) if first != last => format!("seeds {first} to {last}"),
        (Some(first), _) => format!("seed {first}"),
        _ => "no seeds".to_string(),
    };
    out.push_str(&format!(
        "The program under study is the search's `{}` selection (search score {}). {} variants were planned and {} \
         completed, each evaluated {} times with the same {seeds}. Scores are `{}`; higher is better.\n\n",
        inputs.selection_kind,
        num(inputs.search_score),
        summary.total_variants,
        summary.total_completed,
        summary.num_replications,
        summary.primary_metric_name,
    ));
    if !summary.evaluator_is_stochastic {
        out.push_str(
            "The evaluator returned the same score for the control on every seed: it is deterministic, or it ignores \
             `GENESIS_EVAL_SEED`. Intervals and p-values over replications would be meaningless and are not \
             reported; variants are compared by their means only.\n\n",
        );
    }
    out.push_str("## Results\n\n");
    out.push_str(&results_table(summary));
    out.push_str(
        "\n`Correct` counts replications the evaluator judged correct; `n`, the mean, the deviation and the intervals \
         are over those. Δ is the variant's mean minus the control's, tested with Welch's unequal-variance test.\n\n\
         ## Variants\n\n",
    );
    for variant in &summary.variants {
        let spec = &variant.spec;
        out.push_str(&format!(
            "### `{}`: {} ({})\n\n",
            spec.variant_id, spec.title, spec.kind
        ));
        let described = inputs
            .variant_summaries
            .get(&spec.variant_id)
            .map(|text| text.trim())
            .filter(|text| !text.is_empty())
            .unwrap_or(spec.hypothesis.trim());
        if !described.is_empty() {
            out.push_str(&format!("{described}\n\n"));
        }
        if !variant.notes.trim().is_empty() {
            out.push_str(&format!("{}\n\n", variant.notes.trim()));
        }
    }
    if !inputs.rejected.is_empty() {
        out.push_str("## Candidates that were not run\n\n");
        for rejected in inputs.rejected {
            out.push_str(&format!(
                "- `{}` ({}): {}\n",
                rejected.variant_id, rejected.kind, rejected.reason
            ));
        }
        out.push('\n');
    }
    out.push_str(
        "## Files\n\n\
         - `pipeline_summary.json`: every number above, per variant and per replication\n\
         - `aggregated/run_summary.csv`: one row per replication\n\
         - `aggregated/metric_values.csv`: every numeric metric of every replication\n\
         - `variants/<id>/program/`: the program that was evaluated\n\
         - `variants/<id>/rep_<seed>/results/`: the evaluator's output for one replication\n\
         - `generation/`: the proposed candidates, the selection and the program diff\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use genesis_scientist::artifacts::VariantSpec;
    use serde_json::json;

    fn replication(seed: u64, score: f64, correct: bool) -> ReplicationResult {
        ReplicationResult {
            seed,
            combined_score: score,
            correct,
            ..Default::default()
        }
    }

    fn variant(id: &str, kind: &str, scores: &[(f64, bool)]) -> VariantResult {
        VariantResult {
            spec: VariantSpec {
                variant_id: id.into(),
                title: id.into(),
                kind: kind.into(),
                base_program: "best".into(),
                ..Default::default()
            },
            status: "completed".into(),
            replications: scores
                .iter()
                .enumerate()
                .map(|(i, (score, correct))| replication(i as u64 + 1, *score, *correct))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn a_variant_is_summarised_over_its_correct_replications() {
        let control = [5.0, 5.2, 4.9, 5.1];
        let mut ablated = variant(
            "ablation_no_x",
            "ablation",
            &[(4.0, true), (4.3, true), (99.0, false), (4.1, true)],
        );
        summarise_variant(&mut ablated, &control, true, 7);
        let summary = ablated.summary.clone().unwrap();
        assert_eq!(summary.n, 3);
        assert!((summary.mean - (4.0 + 4.3 + 4.1) / 3.0).abs() < 1e-12);
        assert_eq!(ablated.correct_rate, 0.75);
        let comparison = ablated.comparison_to_control.clone().unwrap();
        assert!(comparison.mean_diff < -0.8 && comparison.p_value.unwrap() < 0.05);
        let (low, high) = ablated.bootstrap_ci95.unwrap();
        assert!(low < comparison.mean_diff && comparison.mean_diff < high && high < 0.0);
        assert!(ablated
            .notes
            .starts_with("Difference from the control: -0.9"));
        assert!(!ablated.notes.contains("includes zero"));

        let mut anchor = variant(CONTROL_ID, "control", &[(5.0, true), (5.2, true)]);
        summarise_variant(&mut anchor, &control, true, 7);
        assert!(anchor.comparison_to_control.is_none() && anchor.notes.is_empty());

        let mut broken = variant("ablation_broken", "ablation", &[(0.0, false), (0.0, false)]);
        summarise_variant(&mut broken, &control, true, 7);
        assert!(broken.summary.is_none() && broken.comparison_to_control.is_none());
        assert!(broken
            .notes
            .starts_with("No replication of this variant was correct"));
    }

    #[test]
    fn a_small_difference_is_reported_with_the_sample_it_would_need() {
        let control = [5.0, 5.4, 4.6, 5.2, 4.8];
        let mut close = variant(
            "ablation_small",
            "ablation",
            &[
                (5.1, true),
                (5.3, true),
                (4.7, true),
                (5.2, true),
                (4.9, true),
            ],
        );
        summarise_variant(&mut close, &control, true, 3);
        assert!(
            close.notes.contains("The interval includes zero."),
            "{}",
            close.notes
        );
        assert!(
            close
                .notes
                .contains("replications per variant to resolve at 80% power; this study ran 5"),
            "{}",
            close.notes
        );
        assert!(!close.notes.to_lowercase().contains("no effect"));
        let needed = required_replications(close.comparison_to_control.as_ref().unwrap()).unwrap();
        assert!(needed > 5);
        // d = 1: 2 * (1.96 + 0.84)^2 = 15.7, so sixteen per group.
        let unit = Comparison {
            cohens_d: Some(1.0),
            ..Default::default()
        };
        assert_eq!(required_replications(&unit), Some(16));
        assert_eq!(
            required_replications(&Comparison {
                cohens_d: Some(0.0),
                ..Default::default()
            }),
            None
        );
        assert_eq!(required_replications(&Comparison::default()), None);
    }

    #[test]
    fn numbers_keep_their_digits_at_any_magnitude() {
        assert_eq!(
            (num(5.1), signed(5.1), signed(-0.5), signed(0.0)),
            (
                "5.100000".into(),
                "+5.100000".into(),
                "-0.500000".into(),
                "+0.000000".into()
            )
        );
        assert_eq!(
            (num(0.000_012_3), signed(0.000_012_3), signed(-4.2e11)),
            (
                "1.23000e-5".into(),
                "+1.23000e-5".into(),
                "-4.20000e11".into()
            )
        );
    }

    #[test]
    fn a_deterministic_evaluator_gets_means_without_intervals() {
        let control = [3.0, 3.0, 3.0];
        assert!(
            !is_stochastic(&control) && !is_stochastic(&[1.0]) && is_stochastic(&[1.0, 1.0, 1.5])
        );
        let mut ablated = variant(
            "ablation_no_x",
            "ablation",
            &[(2.5, true), (2.5, true), (2.5, true)],
        );
        summarise_variant(&mut ablated, &control, false, 1);
        assert!(ablated.bootstrap_ci95.is_none());
        assert_eq!(
            ablated.comparison_to_control.as_ref().unwrap().mean_diff,
            -0.5
        );
        assert!(ablated.notes.contains("carries no interval"));
    }

    #[test]
    fn tables_and_csv_rows_cover_every_replication() {
        let mut control = variant(CONTROL_ID, "control", &[(5.0, true), (5.2, true)]);
        control.replications[0]
            .public_metrics
            .insert("x".into(), json!(3.0));
        control.replications[0]
            .public_metrics
            .insert("label".into(), json!("text"));
        control.replications[0]
            .public_metrics
            .insert("flag".into(), json!(true));
        let mut failed = variant("ablation_failed", "ablation", &[]);
        failed.status = "failed".into();
        let mut ablated = variant("ablation_no_x", "ablation", &[(4.0, true), (4.4, true)]);
        summarise_variant(&mut control, &[5.0, 5.2], true, 1);
        summarise_variant(&mut ablated, &[5.0, 5.2], true, 1);
        let summary = AblationSummary {
            task: "pack".into(),
            primary_metric_name: "combined_score".into(),
            anchor_variant_id: CONTROL_ID.into(),
            num_replications: 2,
            seeds: vec![1, 2],
            evaluator_is_stochastic: true,
            total_variants: 3,
            total_completed: 2,
            total_failed: 1,
            variants: vec![control, ablated, failed],
            ..Default::default()
        };
        let rows = run_summary_rows(&summary, "ablations");
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|row| row.len() == RUN_SUMMARY_HEADER.len()));
        assert_eq!(rows[2][0], "ablation_no_x_seed_1");
        assert_eq!(
            rows[2][10],
            "ablations/variants/ablation_no_x/rep_1/results"
        );
        let metrics = metric_value_rows(&summary);
        assert_eq!(
            metrics.len(),
            5,
            "four primary rows and one numeric public metric"
        );
        assert_eq!(
            metrics[1],
            ["control_best_seed_1", "control_best", "x", "3", "false"]
        );

        let table = results_table(&summary);
        assert!(table.contains("| `control_best` | control | 2/2 | 2 | 5.100000 |"));
        assert!(table.contains("(anchor)"));
        assert!(table.contains("| `ablation_failed` | ablation | failed |"));
        let text = readme(&ReadmeInputs {
            summary: &summary,
            selection_kind: "winner",
            search_score: 5.1,
            task_summary: Some("Packs circles."),
            variant_summaries: &BTreeMap::from([(
                "ablation_no_x".to_string(),
                "Removes x.".to_string(),
            )]),
            rejected: &[RejectedVariant {
                variant_id: "baseline_y".into(),
                kind: "baseline".into(),
                reason: "cap".into(),
                ..Default::default()
            }],
        });
        assert!(text.contains("# Ablation study: pack\n\nPacks circles."));
        assert!(text.contains("Removes x.") && text.contains("- `baseline_y` (baseline): cap"));
        assert!(text.contains("seeds 1 to 2") && !text.contains("deterministic"));
    }
}
