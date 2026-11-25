//! The built-in post-hoc analyses.
//!
//! Each family is a pure function of what the search left behind: the
//! program database, the attempt journal and the cost ledger. It never
//! evaluates a program. A family returns a table, a chart, its key numbers
//! and a one-sentence claim written from those numbers alone, or `None`
//! when the run holds nothing for it to analyse.

use super::svg::{bar_chart, xy_chart, Series};
use genesis_core::ledger::CostSummary;
use genesis_core::{stats, Program};
use genesis_evolve::db::AttemptRecord;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};

/// What a search left behind, as the analyses read it.
#[derive(Debug, Clone, Default)]
pub struct RunData {
    pub task_name: String,
    /// Every row of the run database, island copies included.
    pub programs: Vec<Program>,
    /// The program the downstream phases study.
    pub selected_program_id: Option<String>,
    pub attempts: Vec<AttemptRecord>,
    pub cost: Option<CostSummary>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct AnalysisOutput {
    pub table: Table,
    /// An SVG document.
    pub chart: String,
    /// Every number the analysis reports, by name. All finite.
    pub key_numbers: BTreeMap<String, f64>,
    /// One sentence, every number of which is a key number.
    pub claim: String,
}

pub struct Family {
    pub name: &'static str,
    pub title: &'static str,
    /// What the family computes, as told to the proposing model.
    pub description: &'static str,
    /// What a paper would want to learn from it.
    pub hypothesis: &'static str,
    pub compute: fn(&RunData) -> Option<AnalysisOutput>,
}

const FAMILIES: [Family; 6] = [
    Family {
        name: "search_trajectory",
        title: "How the best score developed over the search",
        description: "the best correct score after each generation, and the generations at which it improved",
        hypothesis: "Improvements arrive in a few steps rather than gradually, and the search had not plateaued long before it ended.",
        compute: search_trajectory,
    },
    Family {
        name: "operator_effectiveness",
        title: "Which proposal operators produced improvements",
        description: "per patch type and per model or agent: proposals, share correct, share improving on their parent, mean score change",
        hypothesis: "Operators differ in how often their proposals are correct and how often they improve on the parent.",
        compute: operator_effectiveness,
    },
    Family {
        name: "lineage_of_best",
        title: "The chain of edits behind the selected program",
        description: "the ancestry of the selected program with the score, patch name and operator of every step",
        hypothesis: "The selected program's gain comes from a small number of decisive edits along its lineage.",
        compute: lineage_of_best,
    },
    Family {
        name: "cost_efficiency",
        title: "What improvements cost in model calls",
        description: "model cost per evaluated program and per improvement, by model or agent, from program metadata and the cost ledger",
        hypothesis: "The cost of an improvement differs between models, and failed proposals are a material share of the spend.",
        compute: cost_efficiency,
    },
    Family {
        name: "rubric_outcomes",
        title: "Whether ideas that met their rubrics scored better",
        description: "rubric pass rates of evaluated ideas against the score change of their programs (only when ideas were enabled)",
        hypothesis: "Programs that implement their idea as specified improve on their parent more often than those that do not.",
        compute: rubric_outcomes,
    },
    Family {
        name: "island_dynamics",
        title: "How the islands diverged",
        description: "per island: programs, best correct score over time, and migrations between islands",
        hypothesis: "Islands reach different best scores, and migration spreads the best solutions.",
        compute: island_dynamics,
    },
];

pub fn registry() -> &'static [Family] {
    &FAMILIES
}

pub fn family(name: &str) -> Option<&'static Family> {
    FAMILIES.iter().find(|family| family.name == name)
}

/// Key numbers under construction. Only finite values are kept, and a value
/// is handed back only when it was kept, so a claim can only quote what the
/// analysis actually reports.
#[derive(Default)]
struct Numbers(BTreeMap<String, f64>);

impl Numbers {
    fn put(&mut self, name: impl Into<String>, value: f64) -> Option<f64> {
        value.is_finite().then(|| {
            self.0.insert(name.into(), value);
            value
        })
    }

    fn count(&mut self, name: impl Into<String>, value: usize) -> usize {
        self.0.insert(name.into(), value as f64);
        value
    }
}

fn is_copy(program: &Program) -> bool {
    ["_is_island_copy", "_spawned_island"].iter().any(|flag| {
        program
            .metadata
            .get(*flag)
            .and_then(Value::as_bool)
            .unwrap_or(false)
    })
}

/// The programs the run itself proposed and evaluated, oldest first.
fn own(data: &RunData) -> Vec<&Program> {
    let mut programs: Vec<&Program> = data
        .programs
        .iter()
        .filter(|program| !is_copy(program))
        .collect();
    programs.sort_by(|a, b| {
        a.generation
            .cmp(&b.generation)
            .then(a.timestamp.total_cmp(&b.timestamp))
            .then(a.id.cmp(&b.id))
    });
    programs
}

fn score(program: &Program) -> Option<f64> {
    Some(program.combined_score).filter(|score| score.is_finite())
}

fn is_correct(program: &Program) -> bool {
    program.correct && program.combined_score.is_finite()
}

fn index(data: &RunData) -> HashMap<&str, &Program> {
    data.programs
        .iter()
        .map(|program| (program.id.as_str(), program))
        .collect()
}

fn parent_score(program: &Program, by_id: &HashMap<&str, &Program>) -> Option<f64> {
    program
        .parent_id
        .as_deref()
        .and_then(|id| by_id.get(id))
        .and_then(|parent| score(parent))
        .or_else(|| {
            program
                .meta_f64("parent_score")
                .filter(|score| score.is_finite())
        })
}

/// Score change against the parent, for a correct program with a known
/// parent score.
fn delta(program: &Program, by_id: &HashMap<&str, &Program>) -> Option<f64> {
    if !is_correct(program) {
        return None;
    }
    Some(program.combined_score - parent_score(program, by_id)?).filter(|delta| delta.is_finite())
}

/// Whether a program is correct and better than what it was derived from. A
/// correct child of an incorrect parent counts: it repaired it.
fn improved(program: &Program, by_id: &HashMap<&str, &Program>) -> bool {
    if !is_correct(program) {
        return false;
    }
    match program.parent_id.as_deref().and_then(|id| by_id.get(id)) {
        Some(parent) if is_correct(parent) => program.combined_score > parent.combined_score,
        Some(_) => true,
        None => program
            .meta_f64("parent_score")
            .is_some_and(|parent| program.combined_score > parent),
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty())
        .then(|| stats::mean(values))
        .filter(|mean| mean.is_finite())
}

fn share(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// A name from run metadata, made safe to sit inside backticks in a claim
/// and inside a key-number name.
fn name_of(text: &str) -> String {
    let cleaned: String = text
        .trim()
        .chars()
        .map(|c| if c == '`' || c.is_control() { '\'' } else { c })
        .collect();
    if cleaned.is_empty() {
        "unknown".to_string()
    } else {
        genesis_core::fsutil::truncate(&cleaned, 60)
    }
}

/// A score, a gain or a cost as a claim shows it: four decimals in the
/// ordinary range, four significant digits in scientific notation outside
/// it, so a tiny or a huge value is not printed as `0.0000` or as three
/// hundred digits.
fn shown(value: f64) -> String {
    let magnitude = value.abs();
    if value == 0.0 || (1e-3..1e9).contains(&magnitude) {
        format!("{value:.4}")
    } else {
        format!("{value:.3e}")
    }
}

fn cell(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn table(header: &[&str], rows: Vec<Vec<String>>) -> Table {
    Table {
        header: header.iter().map(|name| name.to_string()).collect(),
        rows,
    }
}

fn arm_of(program: &Program) -> String {
    name_of(
        program
            .meta_str("arm")
            .or_else(|| program.meta_str("model_name"))
            .unwrap_or("unknown"),
    )
}

fn search_trajectory(data: &RunData) -> Option<AnalysisOutput> {
    let programs = own(data);
    if programs.is_empty() {
        return None;
    }
    let mut numbers = Numbers::default();
    let seed = programs
        .iter()
        .find(|program| program.generation == 0)
        .filter(|program| is_correct(program));
    let mut best: Option<f64> = None;
    let (mut steps, mut generation_of_best, mut first_improvement) = (0usize, None, None);
    let (mut rows, mut best_line, mut dots) = (Vec::new(), Vec::new(), Vec::new());
    for program in &programs {
        let correct = is_correct(program);
        let is_step = correct && best.is_none_or(|best| program.combined_score > best);
        if is_step {
            best = Some(program.combined_score);
            generation_of_best = Some(program.generation);
            if program.generation > 0 {
                steps += 1;
                first_improvement.get_or_insert(program.generation);
            }
        }
        let generation = f64::from(program.generation);
        if correct {
            dots.push((generation, program.combined_score));
        }
        if let Some(best) = best {
            best_line.push((generation, best));
        }
        rows.push(vec![
            program.generation.to_string(),
            cell(score(program)),
            correct.to_string(),
            cell(best),
            is_step.to_string(),
        ]);
    }
    let evaluated = numbers.count("programs_evaluated", programs.len());
    let correct = numbers.count(
        "correct_programs",
        programs
            .iter()
            .filter(|program| is_correct(program))
            .count(),
    );
    numbers.put("share_correct", share(correct, evaluated));
    let steps = numbers.count("improvement_steps", steps);
    let seed_score = seed.and_then(|seed| numbers.put("seed_score", seed.combined_score));
    let best_score = best.and_then(|best| numbers.put("final_best_score", best));
    let last = generation_of_best
        .and_then(|generation| numbers.put("generation_of_best", f64::from(generation)));
    if let Some(first) = first_improvement {
        numbers.put("first_improvement_generation", f64::from(first));
    }
    if let (Some(seed), Some(best)) = (seed_score, best_score) {
        numbers.put("total_improvement", best - seed);
    }
    let claim = match (seed_score, best_score, last) {
        (Some(seed), Some(best), Some(last)) if steps > 0 => format!(
            "The best correct score rose from {} to {} over {evaluated} evaluated programs, in {steps} improvement \
             steps; the last improvement came at generation {last:.0}.",
            shown(seed),
            shown(best)
        ),
        (Some(seed), Some(_), _) => {
            format!("None of the {evaluated} evaluated programs improved on the seed's score of {}.", shown(seed))
        }
        (None, Some(best), Some(last)) => format!(
            "The seed was not correct; the best correct score among {evaluated} evaluated programs is {}, reached at \
             generation {last:.0}.",
            shown(best)
        ),
        _ if best.is_none() => format!("None of the {evaluated} evaluated programs was correct."),
        _ => "The search trajectory is tabulated; its scores are too large to summarise.".to_string(),
    };
    Some(AnalysisOutput {
        table: table(
            &[
                "generation",
                "combined_score",
                "correct",
                "best_so_far",
                "improved_best",
            ],
            rows,
        ),
        chart: xy_chart(
            "Best correct score by generation",
            "generation",
            "combined_score",
            &[
                Series {
                    name: "best so far".into(),
                    points: best_line,
                    line: true,
                },
                Series {
                    name: "correct programs".into(),
                    points: dots,
                    line: false,
                },
            ],
        ),
        key_numbers: numbers.0,
        claim,
    })
}

#[derive(Default)]
struct Group {
    proposals: usize,
    correct: usize,
    improving: usize,
    deltas: Vec<f64>,
    cost: f64,
}

fn operator_effectiveness(data: &RunData) -> Option<AnalysisOutput> {
    let by_id = index(data);
    let proposals: Vec<&Program> = own(data)
        .into_iter()
        .filter(|program| program.generation > 0)
        .collect();
    if proposals.is_empty() {
        return None;
    }
    let mut groups: BTreeMap<(&str, String), Group> = BTreeMap::new();
    let (mut correct, mut improving) = (0usize, 0usize);
    for program in &proposals {
        let (is_correct, is_better, change) = (
            is_correct(program),
            improved(program, &by_id),
            delta(program, &by_id),
        );
        correct += usize::from(is_correct);
        improving += usize::from(is_better);
        let patch_type = name_of(program.meta_str("patch_type").unwrap_or("unknown"));
        for key in [("patch_type", patch_type), ("arm", arm_of(program))] {
            let group = groups.entry(key).or_default();
            group.proposals += 1;
            group.correct += usize::from(is_correct);
            group.improving += usize::from(is_better);
            group.deltas.extend(change);
        }
    }
    let mut numbers = Numbers::default();
    let total = numbers.count("total_proposals", proposals.len());
    let share_correct = numbers.put("share_correct", share(correct, total));
    let share_improving = numbers.put("share_improving", share(improving, total));
    let (mut rows, mut bars) = (Vec::new(), Vec::new());
    let mut leader: Option<(f64, usize, String)> = None;
    for ((kind, name), group) in &groups {
        let prefix = format!("{kind}.{name}");
        let count = numbers.count(format!("{prefix}.proposals"), group.proposals);
        numbers.put(
            format!("{prefix}.share_correct"),
            share(group.correct, count),
        );
        let improving_share = share(group.improving, count);
        numbers.put(format!("{prefix}.share_improving"), improving_share);
        let mean_delta =
            mean(&group.deltas).and_then(|mean| numbers.put(format!("{prefix}.mean_delta"), mean));
        rows.push(vec![
            kind.to_string(),
            name.clone(),
            count.to_string(),
            share(group.correct, count).to_string(),
            improving_share.to_string(),
            cell(mean_delta),
        ]);
        bars.push((format!("{kind}: {name}"), improving_share));
        let better = leader.as_ref().is_none_or(|(best_share, best_count, _)| {
            (improving_share, count) > (*best_share, *best_count)
        });
        if *kind == "patch_type" && better {
            leader = Some((improving_share, count, name.clone()));
        }
    }
    let claim = match (share_correct, share_improving, leader) {
        (Some(share_correct), Some(share_improving), Some((leader_share, leader_count, leader_name))) => format!(
            "Of {total} evaluated proposals, {:.1}% were correct and {:.1}% improved on their parent; the patch type \
             with the highest share of improvements was `{leader_name}` ({:.1}% of {leader_count}).",
            share_correct * 100.0,
            share_improving * 100.0,
            leader_share * 100.0
        ),
        _ => format!("The run evaluated {total} proposals; their outcomes by operator are tabulated."),
    };
    Some(AnalysisOutput {
        table: table(
            &[
                "group_kind",
                "group",
                "proposals",
                "share_correct",
                "share_improving",
                "mean_delta",
            ],
            rows,
        ),
        chart: bar_chart(
            "Share of proposals improving on their parent",
            "share improving",
            &bars,
        ),
        key_numbers: numbers.0,
        claim,
    })
}

fn lineage_of_best(data: &RunData) -> Option<AnalysisOutput> {
    let by_id = index(data);
    let target = data
        .selected_program_id
        .as_deref()
        .and_then(|id| by_id.get(id).copied())
        .or_else(|| {
            own(data)
                .into_iter()
                .filter(|program| is_correct(program))
                .max_by(|a, b| {
                    a.combined_score
                        .total_cmp(&b.combined_score)
                        .then(b.generation.cmp(&a.generation))
                })
        })?;
    let mut chain = vec![target];
    let mut seen = HashSet::from([target.id.as_str()]);
    let mut cursor = target.parent_id.as_deref();
    while let Some(parent) = cursor.and_then(|id| by_id.get(id).copied()) {
        if !seen.insert(parent.id.as_str()) {
            break;
        }
        chain.push(parent);
        cursor = parent.parent_id.as_deref();
    }
    chain.reverse();

    let mut numbers = Numbers::default();
    let (mut rows, mut points) = (Vec::new(), Vec::new());
    let mut largest: Option<(f64, u32, String)> = None;
    for (step, program) in chain.iter().enumerate() {
        let gain = step.checked_sub(1).and_then(|previous| {
            Some(score(program)? - score(chain[previous])?).filter(|gain| gain.is_finite())
        });
        let patch_name = name_of(program.meta_str("patch_name").unwrap_or("unnamed"));
        if let Some(gain) = gain {
            if largest.as_ref().is_none_or(|(best, _, _)| gain > *best) {
                largest = Some((gain, program.generation, patch_name.clone()));
            }
        }
        if let Some(score) = score(program) {
            points.push((step as f64, score));
        }
        rows.push(vec![
            step.to_string(),
            program.generation.to_string(),
            cell(score(program)),
            program.correct.to_string(),
            patch_name,
            name_of(program.meta_str("patch_type").unwrap_or("unknown")),
            arm_of(program),
            cell(gain),
        ]);
    }
    numbers.count("lineage_length", chain.len());
    let edits = numbers.count("edits", chain.len() - 1);
    let root_score = score(chain[0]).and_then(|score| numbers.put("root_score", score));
    let final_score = score(target).and_then(|score| numbers.put("final_score", score));
    if let (Some(root), Some(last)) = (root_score, final_score) {
        numbers.put("total_gain", last - root);
    }
    let largest = largest.and_then(|(gain, generation, patch)| {
        let gain = numbers.put("largest_step_gain", gain)?;
        let generation = numbers.put("largest_step_generation", f64::from(generation))?;
        Some((gain, generation, patch))
    });
    let claim = match (final_score, root_score, largest) {
        (Some(last), _, _) if edits == 0 => {
            format!("The selected program (score {}) has no recorded ancestor: no edit lies behind it.", shown(last))
        }
        (Some(last), Some(root), Some((gain, generation, patch))) => format!(
            "The selected program (score {}) descends from a root scoring {} through {edits} edits; the largest \
             single gain was {}, at generation {generation:.0} (`{patch}`).",
            shown(last),
            shown(root),
            shown(gain)
        ),
        _ => format!("The selected program descends from its root through {edits} edits, tabulated step by step."),
    };
    Some(AnalysisOutput {
        table: table(
            &[
                "step",
                "generation",
                "combined_score",
                "correct",
                "patch_name",
                "patch_type",
                "arm",
                "gain_from_previous",
            ],
            rows,
        ),
        chart: xy_chart(
            "Score along the lineage of the selected program",
            "edit",
            "combined_score",
            &[Series {
                name: "lineage".into(),
                points,
                line: true,
            }],
        ),
        key_numbers: numbers.0,
        claim,
    })
}

fn program_cost(program: &Program) -> f64 {
    ["api_costs", "novelty_cost", "embed_cost", "meta_cost"]
        .iter()
        .filter_map(|key| program.meta_f64(key))
        .filter(|cost| cost.is_finite() && *cost > 0.0)
        .sum()
}

fn cost_efficiency(data: &RunData) -> Option<AnalysisOutput> {
    let by_id = index(data);
    let proposals: Vec<&Program> = own(data)
        .into_iter()
        .filter(|program| program.generation > 0)
        .collect();
    if proposals.is_empty() {
        return None;
    }
    let mut arms: BTreeMap<String, Group> = BTreeMap::new();
    for program in &proposals {
        let arm = arms.entry(arm_of(program)).or_default();
        arm.proposals += 1;
        arm.improving += usize::from(improved(program, &by_id));
        arm.cost += program_cost(program);
    }
    let failed: Vec<&AttemptRecord> = data
        .attempts
        .iter()
        .filter(|attempt| attempt.status == "failed")
        .collect();
    let failed_cost: f64 = failed
        .iter()
        .filter_map(|attempt| attempt.details.get("cost_usd").and_then(Value::as_f64))
        .filter(|cost| cost.is_finite() && *cost > 0.0)
        .sum();

    let mut numbers = Numbers::default();
    let evaluated = numbers.count("programs_evaluated", proposals.len());
    let improvements = numbers.count(
        "total_improvements",
        arms.values().map(|arm| arm.improving).sum(),
    );
    let total_cost = numbers.put(
        "total_program_cost",
        arms.values().map(|arm| arm.cost).sum(),
    );
    let per_program =
        total_cost.and_then(|cost| numbers.put("cost_per_program", cost / evaluated as f64));
    let per_improvement = total_cost
        .filter(|_| improvements > 0)
        .and_then(|cost| numbers.put("cost_per_improvement", cost / improvements as f64));
    numbers.count("failed_proposals", failed.len());
    numbers.put("failed_proposal_cost", failed_cost);
    if let Some(ledger) = &data.cost {
        numbers.put("ledger_total_cost_usd", ledger.total_cost_usd);
    }
    let (mut rows, mut bars) = (Vec::new(), Vec::new());
    for (name, arm) in &arms {
        numbers.count(format!("arm.{name}.programs"), arm.proposals);
        numbers.count(format!("arm.{name}.improvements"), arm.improving);
        numbers.put(format!("arm.{name}.cost"), arm.cost);
        let each = numbers.put(
            format!("arm.{name}.cost_per_program"),
            arm.cost / arm.proposals as f64,
        );
        let per_gain = (arm.improving > 0)
            .then(|| {
                numbers.put(
                    format!("arm.{name}.cost_per_improvement"),
                    arm.cost / arm.improving as f64,
                )
            })
            .flatten();
        rows.push(vec![
            name.clone(),
            arm.proposals.to_string(),
            arm.improving.to_string(),
            arm.cost.to_string(),
            cell(each),
            cell(per_gain),
        ]);
        bars.push((name.clone(), arm.cost));
    }
    let claim = match (total_cost, per_program, per_improvement) {
        (Some(0.0), _, _) => {
            format!("No model cost was recorded for the {evaluated} evaluated proposals (offline or unpriced models).")
        }
        (Some(total), Some(each), Some(per_gain)) => format!(
            "The {evaluated} evaluated proposals cost {} USD in model calls ({} USD each); each of the {improvements} \
             improvements over a parent cost {} USD on average.",
            shown(total),
            shown(each),
            shown(per_gain)
        ),
        (Some(total), Some(each), None) => format!(
            "The {evaluated} evaluated proposals cost {} USD in model calls ({} USD each); none of them improved on \
             its parent.",
            shown(total),
            shown(each)
        ),
        _ => format!("Model costs of the {evaluated} evaluated proposals are tabulated by model."),
    };
    Some(AnalysisOutput {
        table: table(
            &[
                "arm",
                "programs",
                "improvements",
                "cost_usd",
                "cost_per_program",
                "cost_per_improvement",
            ],
            rows,
        ),
        chart: bar_chart("Model cost of evaluated proposals", "USD", &bars),
        key_numbers: numbers.0,
        claim,
    })
}

fn rubric_outcomes(data: &RunData) -> Option<AnalysisOutput> {
    let by_id = index(data);
    let mut numbers = Numbers::default();
    let (mut rows, mut points) = (Vec::new(), Vec::new());
    let (mut programs, mut all_passed, mut rubrics, mut passed, mut assessed) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut when_pass, mut when_fail, mut fractions, mut changes) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for program in own(data) {
        let Some(status) = program
            .metadata
            .get("rubric_status")
            .filter(|status| status.is_object())
        else {
            continue;
        };
        let entries: Vec<&Value> = status
            .get("rubrics")
            .and_then(Value::as_array)
            .map(|list| list.iter().collect())
            .unwrap_or_default();
        let total = entries.len();
        let ok = entries
            .iter()
            .filter(|entry| entry.get("pass").and_then(Value::as_bool) == Some(true))
            .count();
        let checked = entries
            .iter()
            .filter(|entry| entry.get("assessment").and_then(Value::as_str) != Some("not_assessed"))
            .count();
        let overall = status
            .get("overall_pass")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let change = delta(program, &by_id);
        programs += 1;
        all_passed += usize::from(overall);
        rubrics += total;
        passed += ok;
        assessed += checked;
        if let Some(change) = change {
            if overall {
                &mut when_pass
            } else {
                &mut when_fail
            }
            .push(change);
            if total > 0 {
                fractions.push(share(ok, total));
                changes.push(change);
                points.push((share(ok, total), change));
            }
        }
        rows.push(vec![
            program.generation.to_string(),
            name_of(
                program
                    .metadata
                    .get("idea")
                    .and_then(|idea| idea.get("title"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            ),
            ok.to_string(),
            total.to_string(),
            overall.to_string(),
            cell(score(program)),
            cell(change),
            program.correct.to_string(),
        ]);
    }
    if programs == 0 {
        return None;
    }
    let programs = numbers.count("programs_with_rubrics", programs);
    let share_pass = numbers.put("share_overall_pass", share(all_passed, programs));
    numbers.count("rubrics_total", rubrics);
    numbers.put("share_rubrics_passed", share(passed, rubrics));
    let share_assessed = numbers.put("share_rubrics_assessed", share(assessed, rubrics));
    let mean_pass = mean(&when_pass).and_then(|mean| numbers.put("mean_delta_when_pass", mean));
    let mean_fail = mean(&when_fail).and_then(|mean| numbers.put("mean_delta_when_fail", mean));
    if let Some(rho) = stats::spearman(&fractions, &changes) {
        numbers.put("pass_fraction_delta_spearman", rho);
    }
    let mut claim = match (share_pass, share_assessed) {
        (Some(pass), Some(assessed)) => format!(
            "{programs} evaluated programs carried rubrics; {:.1}% passed all of theirs, and {:.1}% of individual \
             rubrics could be assessed.",
            pass * 100.0,
            assessed * 100.0
        ),
        _ => format!("{programs} evaluated programs carried rubrics; their outcomes are tabulated."),
    };
    if let (Some(pass), Some(fail)) = (mean_pass, mean_fail) {
        claim.push_str(&format!(
            " Programs that passed every rubric changed the score by {} on average against {} for the rest.",
            shown(pass),
            shown(fail)
        ));
    }
    Some(AnalysisOutput {
        table: table(
            &[
                "generation",
                "idea",
                "rubrics_passed",
                "rubrics_total",
                "overall_pass",
                "combined_score",
                "delta_vs_parent",
                "correct",
            ],
            rows,
        ),
        chart: xy_chart(
            "Rubric pass fraction against score change",
            "fraction of rubrics passed",
            "score change against parent",
            &[Series {
                name: "evaluated ideas".into(),
                points,
                line: false,
            }],
        ),
        key_numbers: numbers.0,
        claim,
    })
}

fn island_dynamics(data: &RunData) -> Option<AnalysisOutput> {
    let mut islands: BTreeMap<usize, Vec<&Program>> = BTreeMap::new();
    for program in &data.programs {
        if let Some(island) = program.island_idx {
            islands.entry(island).or_default().push(program);
        }
    }
    if islands.is_empty() {
        return None;
    }
    let mut numbers = Numbers::default();
    let (mut rows, mut series) = (Vec::new(), Vec::new());
    let mut leader: Option<(f64, usize)> = None;
    let mut total = 0usize;
    for (island, programs) in &mut islands {
        programs.sort_by(|a, b| {
            a.generation
                .cmp(&b.generation)
                .then(a.timestamp.total_cmp(&b.timestamp))
        });
        let mut best: Option<f64> = None;
        let mut line = Vec::new();
        for program in programs.iter() {
            if is_correct(program) && best.is_none_or(|best| program.combined_score > best) {
                best = Some(program.combined_score);
            }
            if let Some(best) = best {
                line.push((f64::from(program.generation), best));
                rows.push(vec![
                    island.to_string(),
                    program.generation.to_string(),
                    best.to_string(),
                ]);
            }
        }
        total += programs.len();
        numbers.count(format!("island.{island}.programs"), programs.len());
        numbers.count(
            format!("island.{island}.correct_programs"),
            programs
                .iter()
                .filter(|program| is_correct(program))
                .count(),
        );
        if let Some(best) =
            best.and_then(|best| numbers.put(format!("island.{island}.best_score"), best))
        {
            if leader.is_none_or(|(top, _)| best > top) {
                leader = Some((best, *island));
            }
        }
        series.push(Series {
            name: format!("island {island}"),
            points: line,
            line: true,
        });
    }
    let count = numbers.count("num_islands", islands.len());
    let total = numbers.count("programs_total", total);
    let migrations = numbers.count(
        "migrations",
        data.programs
            .iter()
            .map(|program| program.migration_history.len())
            .sum(),
    );
    let claim = match leader {
        Some((best, island)) => {
            numbers.count("best_island", island);
            format!(
                "{count} islands held {total} programs; island {island} reached the highest correct score ({}), and \
                 {migrations} migrations moved programs between islands.",
                shown(best)
            )
        }
        None => format!("{count} islands held {total} programs, none of them correct."),
    };
    Some(AnalysisOutput {
        table: table(&["island", "generation", "best_so_far"], rows),
        chart: xy_chart(
            "Best correct score per island",
            "generation",
            "combined_score",
            &series,
        ),
        key_numbers: numbers.0,
        claim,
    })
}

#[cfg(test)]
mod tests {
    use super::super::grounding::ungrounded_numbers;
    use super::*;
    use serde_json::json;

    fn program(
        id: &str,
        generation: u32,
        score: f64,
        correct: bool,
        parent: Option<&str>,
    ) -> Program {
        let mut program = Program::new("code", "python");
        program.id = id.to_string();
        program.generation = generation;
        program.combined_score = score;
        program.correct = correct;
        program.parent_id = parent.map(str::to_string);
        program.island_idx = Some(0);
        program.timestamp = f64::from(generation);
        program
    }

    fn with(mut program: Program, entries: &[(&str, Value)]) -> Program {
        for (key, value) in entries {
            program.metadata.insert((*key).to_string(), value.clone());
        }
        program
    }

    /// Seed 1.0; a diff by model-a to 2.0; a failed full rewrite; a diff by
    /// model-b from the 2.0 program to 2.5; a full rewrite that got worse.
    fn run() -> RunData {
        let patch = |kind: &str, arm: &str, name: &str, cost: f64| {
            vec![
                ("patch_type", json!(kind)),
                ("arm", json!(arm)),
                ("patch_name", json!(name)),
                ("api_costs", json!(cost)),
            ]
        };
        RunData {
            task_name: "pack".into(),
            programs: vec![
                with(
                    program("seed", 0, 1.0, true, None),
                    &[("patch_name", json!("initial_program"))],
                ),
                with(
                    program("a", 1, 2.0, true, Some("seed")),
                    &patch("diff", "model-a", "add_restarts", 0.10),
                ),
                with(
                    program("b", 2, 0.0, false, Some("seed")),
                    &patch("full", "model-a", "broken", 0.20),
                ),
                with(
                    program("c", 3, 2.5, true, Some("a")),
                    &patch("diff", "model-b", "tune_cooling", 0.30),
                ),
                with(
                    program("d", 4, 1.5, true, Some("a")),
                    &patch("full", "model-b", "worse", 0.40),
                ),
            ],
            selected_program_id: Some("c".into()),
            attempts: vec![AttemptRecord {
                generation: 5,
                stage: "proposal".into(),
                status: "failed".into(),
                details: json!({"cost_usd": 0.05}),
                created_at: 0.0,
            }],
            cost: None,
        }
    }

    fn output(name: &str, data: &RunData) -> AnalysisOutput {
        let output = (family(name).unwrap().compute)(data)
            .unwrap_or_else(|| panic!("{name} produced nothing"));
        assert!(output.key_numbers.values().all(|value| value.is_finite()));
        assert_eq!(
            ungrounded_numbers(&output.claim, &output.key_numbers),
            Vec::<String>::new(),
            "{}",
            output.claim
        );
        assert!(output.chart.starts_with("<svg"));
        assert!(output
            .table
            .rows
            .iter()
            .all(|row| row.len() == output.table.header.len()));
        output
    }

    #[test]
    fn the_trajectory_counts_improvement_steps() {
        let out = output("search_trajectory", &run());
        assert_eq!(out.key_numbers["improvement_steps"], 2.0);
        assert_eq!(out.key_numbers["final_best_score"], 2.5);
        assert_eq!(out.key_numbers["total_improvement"], 1.5);
        assert_eq!(out.key_numbers["generation_of_best"], 3.0);
        assert_eq!(out.key_numbers["share_correct"], 0.8);
        assert_eq!(
            out.claim,
            "The best correct score rose from 1.0000 to 2.5000 over 5 evaluated programs, in 2 improvement steps; the \
             last improvement came at generation 3."
        );
        assert_eq!(out.table.rows[2], ["2", "0", "false", "2", "false"]);
    }

    #[test]
    fn operators_are_compared_by_patch_type_and_arm() {
        let out = output("operator_effectiveness", &run());
        assert_eq!(out.key_numbers["total_proposals"], 4.0);
        assert_eq!(out.key_numbers["patch_type.diff.share_improving"], 1.0);
        assert_eq!(out.key_numbers["patch_type.full.share_improving"], 0.0);
        assert_eq!(out.key_numbers["patch_type.full.share_correct"], 0.5);
        assert_eq!(
            out.key_numbers["arm.model-b.mean_delta"], 0.0,
            "+0.5 and -0.5"
        );
        assert_eq!(out.key_numbers["patch_type.diff.mean_delta"], 0.75);
        assert!(out.claim.contains(
            "the patch type with the highest share of improvements was `diff` (100.0% of 2)"
        ));
    }

    #[test]
    fn the_lineage_walks_back_to_the_seed() {
        let out = output("lineage_of_best", &run());
        assert_eq!(out.key_numbers["edits"], 2.0);
        assert_eq!(out.key_numbers["total_gain"], 1.5);
        assert_eq!(out.key_numbers["largest_step_gain"], 1.0);
        assert_eq!(out.key_numbers["largest_step_generation"], 1.0);
        assert!(out.claim.contains("(`add_restarts`)"));
        assert_eq!(out.table.rows.len(), 3);

        let mut seed_only = run();
        seed_only.selected_program_id = Some("seed".into());
        assert!(output("lineage_of_best", &seed_only)
            .claim
            .contains("no recorded ancestor"));
        let mut cyclic = run();
        cyclic.programs[0].parent_id = Some("c".into());
        assert_eq!(
            output("lineage_of_best", &cyclic).key_numbers["lineage_length"],
            3.0,
            "a cycle ends the walk"
        );
    }

    #[test]
    fn costs_are_split_by_arm_and_include_failed_proposals() {
        let out = output("cost_efficiency", &run());
        assert!((out.key_numbers["total_program_cost"] - 1.0).abs() < 1e-12);
        assert!((out.key_numbers["cost_per_improvement"] - 0.5).abs() < 1e-12);
        assert!((out.key_numbers["arm.model-a.cost_per_improvement"] - 0.30).abs() < 1e-12);
        assert_eq!(out.key_numbers["failed_proposal_cost"], 0.05);
        assert!(out
            .claim
            .starts_with("The 4 evaluated proposals cost 1.0000 USD"));

        let mut free = run();
        for program in &mut free.programs {
            program.metadata.remove("api_costs");
        }
        assert!(output("cost_efficiency", &free)
            .claim
            .starts_with("No model cost was recorded"));
    }

    #[test]
    fn rubric_outcomes_need_rubric_data() {
        assert!((family("rubric_outcomes").unwrap().compute)(&run()).is_none());
        let mut data = run();
        let status = |pass: bool| {
            json!({"overall_pass": pass, "rubrics": [
            {"id": "a", "pass": pass, "assessment": if pass { "passed" } else { "failed" }},
            {"id": "b", "pass": false, "assessment": "not_assessed"}]})
        };
        data.programs[1]
            .metadata
            .insert("rubric_status".into(), status(true));
        data.programs[3]
            .metadata
            .insert("rubric_status".into(), status(false));
        data.programs[4]
            .metadata
            .insert("rubric_status".into(), status(false));
        let out = output("rubric_outcomes", &data);
        assert_eq!(out.key_numbers["programs_with_rubrics"], 3.0);
        assert_eq!(out.key_numbers["share_rubrics_assessed"], 0.5);
        assert_eq!(out.key_numbers["mean_delta_when_pass"], 1.0);
        assert_eq!(out.key_numbers["mean_delta_when_fail"], 0.0);
        assert!(out.claim.contains("33.3% passed all of theirs"));
    }

    #[test]
    fn islands_report_their_best_and_migrations() {
        let mut data = run();
        data.programs[3].island_idx = Some(1);
        data.programs[3].migration_history = vec![json!({"from": 0, "to": 1})];
        let out = output("island_dynamics", &data);
        assert_eq!(out.key_numbers["num_islands"], 2.0);
        assert_eq!(out.key_numbers["best_island"], 1.0);
        assert_eq!(out.key_numbers["island.0.best_score"], 2.0);
        assert!(out
            .claim
            .contains("island 1 reached the highest correct score (2.5000), and 1 migrations"));
    }

    #[test]
    fn tiny_and_huge_values_are_shown_with_their_digits() {
        assert_eq!(shown(2.635), "2.6350");
        assert_eq!(shown(0.0), "0.0000");
        assert_eq!(shown(-0.25), "-0.2500");
        assert_eq!(shown(0.000_012_34), "1.234e-5");
        assert_eq!(shown(-3.2e10), "-3.200e10");
        let mut data = run();
        for program in &mut data.programs {
            program.combined_score *= 1e-6;
        }
        let out = output("search_trajectory", &data);
        assert!(
            out.claim.contains("rose from 1.000e-6 to 2.500e-6"),
            "{}",
            out.claim
        );
    }

    #[test]
    fn an_empty_run_has_nothing_to_analyse_and_odd_names_stay_harmless() {
        for family in registry() {
            assert!(
                (family.compute)(&RunData::default()).is_none(),
                "{}",
                family.name
            );
        }
        let mut data = run();
        data.programs[1]
            .metadata
            .insert("patch_type".into(), json!("di`ff 99\nx"));
        data.programs[1]
            .metadata
            .insert("patch_name".into(), json!("name with ` and 123"));
        for family in registry() {
            if (family.compute)(&data).is_some() {
                output(family.name, &data);
            }
        }
        assert_eq!(name_of("  "), "unknown");
    }
}
