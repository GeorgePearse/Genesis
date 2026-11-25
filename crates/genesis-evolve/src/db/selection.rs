//! Choosing what to mutate next: an island, a parent on it, and the
//! inspiration programs shown alongside the parent.

use super::islands::initialized_islands;
use super::{best, get, meta_get, meta_set, query, META_BEAM_PARENT, META_BEST_ID};
use crate::config::DatabaseConfig;
use anyhow::{Context, Result};
use genesis_core::{stats, Program};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};

/// The context of one proposal.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub parent: Program,
    pub archive_inspirations: Vec<Program>,
    pub top_k_inspirations: Vec<Program>,
    /// No correct program exists yet: the parent is an incorrect program to
    /// repair, and the "inspirations" are its ancestors, oldest first.
    pub needs_fix: bool,
}

const ARCHIVED_ON_ISLAND: &str =
    "WHERE a.program_id IS NOT NULL AND p.correct = 1 AND p.island_idx = ?1";

/// An index drawn in proportion to `weights`; uniform when they do not sum
/// to a positive finite number.
pub(crate) fn weighted_index(rng: &mut ChaCha8Rng, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    if weights.is_empty() {
        return 0;
    }
    if !total.is_finite() || total <= 0.0 || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return rng.gen_range(0..weights.len());
    }
    let mut point = rng.gen::<f64>() * total;
    for (index, weight) in weights.iter().enumerate() {
        if point < *weight {
            return index;
        }
        point -= weight;
    }
    weights.len() - 1
}

fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Selection probability of each candidate under weighted sampling:
/// a sigmoid of the score's distance from the median in MAD units, damped by
/// how many children the program already has.
pub fn weighted_probabilities(scores: &[f64], children: &[u32], lambda: f64) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let median = stats::median(scores);
    let deviations: Vec<f64> = scores.iter().map(|score| (score - median).abs()).collect();
    let scale = stats::median(&deviations).max(1e-6);
    let weights: Vec<f64> = scores
        .iter()
        .zip(children)
        .map(|(score, n)| stable_sigmoid(lambda * (score - median) / scale) / (1.0 + f64::from(*n)))
        .collect();
    let total: f64 = weights.iter().sum();
    if total > 0.0 && total.is_finite() {
        weights.iter().map(|w| w / total).collect()
    } else {
        vec![1.0 / scores.len() as f64; scores.len()]
    }
}

/// Selection probability by score rank: `(rank + 1)^(-alpha)`, normalised.
pub fn power_law_probabilities(count: usize, alpha: f64) -> Vec<f64> {
    let raw: Vec<f64> = (0..count)
        .map(|rank| ((rank + 1) as f64).powf(-alpha))
        .collect();
    let total: f64 = raw.iter().sum();
    if total > 0.0 && total.is_finite() {
        raw.iter().map(|w| w / total).collect()
    } else {
        vec![1.0 / count.max(1) as f64; count]
    }
}

fn sample_island(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
    initialized: &[usize],
) -> Result<usize> {
    if initialized.len() == 1 || config.island_selection_strategy == "uniform" {
        return Ok(*initialized
            .choose(rng)
            .expect("at least one initialised island"));
    }
    let mut statement = conn.prepare(
        "SELECT island_idx, COUNT(*), MAX(combined_score) FROM programs \
         WHERE correct = 1 AND island_idx IS NOT NULL GROUP BY island_idx",
    )?;
    let rows: HashMap<usize, (f64, Option<f64>)> = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)? as usize,
                (row.get::<_, i64>(1)? as f64, row.get(2)?),
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let counts: Vec<f64> = initialized
        .iter()
        .map(|island| rows.get(island).map_or(0.0, |r| r.0))
        .collect();
    let raw: Vec<Option<f64>> = initialized
        .iter()
        .map(|island| rows.get(island).and_then(|r| r.1).filter(|f| f.is_finite()))
        .collect();
    let worst = raw.iter().flatten().copied().fold(f64::INFINITY, f64::min);
    let worst = if worst.is_finite() { worst } else { 0.0 };
    let fitness: Vec<f64> = raw.iter().map(|value| value.unwrap_or(worst)).collect();

    let index = match config.island_selection_strategy.as_str() {
        "equal" => {
            let fewest = counts.iter().copied().fold(f64::INFINITY, f64::min);
            let minima: Vec<usize> = (0..counts.len()).filter(|i| counts[*i] == fewest).collect();
            *minima.choose(rng).expect("at least one island")
        }
        "proportional" => {
            let peak = fitness.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = fitness.iter().map(|f| (f - peak).exp()).collect();
            weighted_index(rng, &weights)
        }
        _ => {
            let best = fitness.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let span = best - worst;
            let shifted: Vec<f64> = if worst < 0.0 {
                fitness
                    .iter()
                    .map(|f| f - worst + if span > 0.0 { span } else { 1.0 })
                    .collect()
            } else {
                fitness.clone()
            };
            let weights: Vec<f64> = shifted
                .iter()
                .zip(&counts)
                .map(|(f, count)| f / count.max(1.0))
                .collect();
            weighted_index(rng, &weights)
        }
    };
    Ok(initialized[index])
}

fn power_law(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
    island: usize,
) -> Result<Option<Program>> {
    let key = island as i64;
    let mut candidates = query(
        conn,
        &format!("{ARCHIVED_ON_ISLAND} ORDER BY p.combined_score DESC, p.rowid"),
        &[&key],
    )?;
    if candidates.is_empty() {
        candidates = query(
            conn,
            "WHERE p.correct = 1 AND p.island_idx = ?1 ORDER BY p.combined_score DESC, p.rowid",
            &[&key],
        )?;
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    let probabilities = power_law_probabilities(candidates.len(), config.exploitation_alpha);
    Ok(Some(
        candidates.swap_remove(weighted_index(rng, &probabilities)),
    ))
}

fn weighted(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
    island: usize,
) -> Result<Option<Program>> {
    let mut candidates = query(
        conn,
        &format!("{ARCHIVED_ON_ISLAND} ORDER BY p.rowid"),
        &[&(island as i64)],
    )?;
    if candidates.is_empty() {
        return Ok(None);
    }
    let scores: Vec<f64> = candidates
        .iter()
        .map(|program| program.combined_score)
        .collect();
    let children: Vec<u32> = candidates
        .iter()
        .map(|program| program.children_count)
        .collect();
    let probabilities = weighted_probabilities(&scores, &children, config.parent_selection_lambda);
    Ok(Some(
        candidates.swap_remove(weighted_index(rng, &probabilities)),
    ))
}

/// Keep expanding one parent until it has `num_beams` children, then move
/// the beam to the island's current best.
fn beam_search(
    conn: &Connection,
    config: &DatabaseConfig,
    island: usize,
) -> Result<Option<Program>> {
    let current = match meta_get(conn, META_BEAM_PARENT)? {
        Some(id) => get(conn, &id)?.filter(|program| program.island_idx == Some(island)),
        None => None,
    };
    let current = match current {
        Some(program) => program,
        None => {
            let Some(start) = best(conn, Some(island))? else {
                return Ok(None);
            };
            meta_set(conn, META_BEAM_PARENT, &start.id)?;
            start
        }
    };
    if (current.children_count as usize) < config.num_beams {
        return Ok(Some(current));
    }
    let Some(next) = best(conn, Some(island))? else {
        return Ok(Some(current));
    };
    meta_set(conn, META_BEAM_PARENT, &next.id)?;
    Ok(Some(next))
}

fn tournament(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
    island: usize,
) -> Result<Option<Program>> {
    let key = island as i64;
    let limit = config.tournament_top_k as i64;
    let mut pool = query(
        conn,
        &format!("{ARCHIVED_ON_ISLAND} AND p.combined_score IS NOT NULL ORDER BY p.combined_score DESC, p.rowid LIMIT ?2"),
        &[&key, &limit],
    )?;
    if pool.is_empty() {
        pool = query(
            conn,
            "WHERE p.correct = 1 AND p.island_idx = ?1 AND p.combined_score IS NOT NULL \
             ORDER BY p.combined_score DESC, p.rowid LIMIT ?2",
            &[&key, &limit],
        )?;
    }
    if pool.is_empty() {
        return Ok(None);
    }
    let draws = config.tournament_num_draws.min(pool.len());
    let entrants: Vec<&Program> = pool.choose_multiple(rng, draws).collect();
    let top = entrants
        .iter()
        .map(|program| program.combined_score)
        .fold(f64::NEG_INFINITY, f64::max);
    let winners: Vec<&&Program> = entrants
        .iter()
        .filter(|program| program.combined_score == top)
        .collect();
    Ok(winners.choose(rng).map(|winner| (**winner).clone()))
}

fn select_parent(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
    island: usize,
) -> Result<Program> {
    let key = island as i64;
    let chosen = match config.parent_selection_strategy.as_str() {
        "power_law" => power_law(conn, config, rng, island)?,
        "beam_search" => beam_search(conn, config, island)?,
        "hill_climb" => best(conn, Some(island))?,
        "tournament" => tournament(conn, config, rng, island)?,
        "best_of_n" => query(
            conn,
            "WHERE p.correct = 1 AND p.generation = 0 AND p.island_idx = ?1 ORDER BY p.rowid LIMIT 1",
            &[&key],
        )?
        .pop(),
        "sequential" => query(
            conn,
            "WHERE p.correct = 1 AND p.island_idx = ?1 ORDER BY p.generation DESC, p.rowid DESC LIMIT 1",
            &[&key],
        )?
        .pop(),
        _ => weighted(conn, config, rng, island)?,
    };
    if let Some(parent) = chosen {
        return Ok(parent);
    }
    if let Some(id) = meta_get(conn, META_BEST_ID)? {
        if let Some(tracked) = get(conn, &id)?.filter(|p| p.correct && p.island_idx == Some(island))
        {
            return Ok(tracked);
        }
    }
    let correct = query(
        conn,
        "WHERE p.correct = 1 AND p.island_idx = ?1 ORDER BY p.rowid",
        &[&key],
    )?;
    if let Some(parent) = correct.choose(rng) {
        return Ok(parent.clone());
    }
    let anywhere = query(conn, "WHERE p.correct = 1 ORDER BY p.rowid", &[])?;
    anywhere
        .choose(rng)
        .cloned()
        .context("no correct program to select as a parent")
}

/// Archive inspirations: the best program when it shares the parent's
/// island, then island elites, then random archive members.
fn archive_inspirations(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
    parent: &Program,
) -> Result<Vec<Program>> {
    let wanted = config.num_archive_inspirations;
    if wanted == 0 || config.archive_size == 0 {
        return Ok(Vec::new());
    }
    let mut chosen: Vec<Program> = Vec::new();
    let mut taken: HashSet<String> = HashSet::from([parent.id.clone()]);
    let island = parent.island_idx;
    let on_island =
        |program: &Program| !config.enforce_island_separation || program.island_idx == island;

    if let Some(id) = meta_get(conn, META_BEST_ID)? {
        if let Some(tracked) = get(conn, &id)?.filter(|p| p.correct && p.in_archive && on_island(p))
        {
            if taken.insert(tracked.id.clone()) {
                chosen.push(tracked);
            }
        }
    }

    let island_members = match island {
        Some(island) => query(
            conn,
            &format!("{ARCHIVED_ON_ISLAND} ORDER BY p.combined_score DESC, p.rowid"),
            &[&(island as i64)],
        )?,
        None => Vec::new(),
    };
    let elites = (wanted as f64 * config.elite_selection_ratio).floor() as usize;
    if elites > 0 {
        for member in island_members.iter().take(elites + taken.len()) {
            if chosen.len() >= wanted {
                break;
            }
            if taken.insert(member.id.clone()) {
                chosen.push(member.clone());
            }
        }
    }

    let mut fill = |pool: Vec<Program>, chosen: &mut Vec<Program>, taken: &mut HashSet<String>| {
        let mut rest: Vec<Program> = pool
            .into_iter()
            .filter(|p| !taken.contains(&p.id))
            .collect();
        rest.shuffle(rng);
        for program in rest {
            if chosen.len() >= wanted {
                break;
            }
            taken.insert(program.id.clone());
            chosen.push(program);
        }
    };
    if chosen.len() < wanted {
        fill(island_members, &mut chosen, &mut taken);
    }
    if chosen.len() < wanted && !config.enforce_island_separation {
        let everywhere = query(
            conn,
            "WHERE a.program_id IS NOT NULL AND p.correct = 1 ORDER BY p.rowid",
            &[],
        )?;
        fill(everywhere, &mut chosen, &mut taken);
    }
    chosen.truncate(wanted);
    Ok(chosen)
}

/// The highest-scoring archived programs not already shown.
fn top_k_inspirations(
    conn: &Connection,
    config: &DatabaseConfig,
    parent: &Program,
    already: &[Program],
) -> Result<Vec<Program>> {
    let wanted = config.num_top_k_inspirations;
    if wanted == 0 || config.archive_size == 0 {
        return Ok(Vec::new());
    }
    let candidates = match (config.enforce_island_separation, parent.island_idx) {
        (true, None) => return Ok(Vec::new()),
        (true, Some(island)) => query(
            conn,
            &format!("{ARCHIVED_ON_ISLAND} ORDER BY p.combined_score DESC, p.rowid"),
            &[&(island as i64)],
        )?,
        (false, _) => query(
            conn,
            "WHERE a.program_id IS NOT NULL AND p.correct = 1 ORDER BY p.combined_score DESC, p.rowid",
            &[],
        )?,
    };
    let excluded: HashSet<&str> = std::iter::once(parent.id.as_str())
        .chain(already.iter().map(|p| p.id.as_str()))
        .collect();
    Ok(candidates
        .into_iter()
        .filter(|p| !excluded.contains(p.id.as_str()))
        .take(wanted)
        .collect())
}

fn ancestors(conn: &Connection, id: &str, max: usize) -> Result<Vec<Program>> {
    let mut chain = Vec::new();
    let mut seen = HashSet::new();
    let mut cursor = get(conn, id)?.and_then(|program| program.parent_id);
    while let Some(parent_id) = cursor {
        if chain.len() >= max || !seen.insert(parent_id.clone()) {
            break;
        }
        let Some(parent) = get(conn, &parent_id)? else {
            break;
        };
        cursor = parent.parent_id.clone();
        chain.push(parent);
    }
    chain.reverse();
    Ok(chain)
}

pub(crate) fn sample(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
) -> Result<Sample> {
    let initialized = initialized_islands(conn)?;
    if initialized.is_empty() {
        let incorrect = query(
            conn,
            "WHERE p.correct = 0 ORDER BY p.timestamp, p.rowid",
            &[],
        )?;
        let parent = incorrect
            .choose(rng)
            .cloned()
            .context("the database has no program to sample")?;
        let ancestry = ancestors(
            conn,
            &parent.id,
            config.num_archive_inspirations + config.num_top_k_inspirations,
        )?;
        return Ok(Sample {
            parent,
            archive_inspirations: ancestry,
            top_k_inspirations: Vec::new(),
            needs_fix: true,
        });
    }
    let island = sample_island(conn, config, rng, &initialized)?;
    let parent = select_parent(conn, config, rng, island)?;
    let archive_inspirations = archive_inspirations(conn, config, rng, &parent)?;
    let top_k_inspirations = top_k_inspirations(conn, config, &parent, &archive_inspirations)?;
    Ok(Sample {
        parent,
        archive_inspirations,
        top_k_inspirations,
        needs_fix: false,
    })
}

#[cfg(test)]
mod tests {
    use super::super::testing::{db, program};
    use super::*;
    use crate::db::ProgramDb;

    fn seeded(config: DatabaseConfig, scores: &[f64]) -> (ProgramDb, Vec<Program>) {
        let db = db(config);
        let mut seed = program(scores[0], true, 0, None);
        db.add(&mut seed).unwrap();
        let mut all = vec![seed.clone()];
        for (offset, score) in scores[1..].iter().enumerate() {
            let mut child = program(*score, true, offset as u32 + 1, Some(&seed));
            db.add(&mut child).unwrap();
            all.push(child);
        }
        (db, all)
    }

    fn pick_counts(db: &ProgramDb, draws: usize) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for _ in 0..draws {
            *counts.entry(db.sample().unwrap().parent.id).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn weighted_probabilities_favour_high_scores_and_few_children() {
        let p = weighted_probabilities(&[1.0, 2.0, 3.0], &[0, 0, 0], 10.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p[2] > p[1] && p[1] > p[0]);
        // The median program has sigmoid(0) = 0.5; ten children divide its weight by eleven.
        let damped = weighted_probabilities(&[1.0, 2.0, 3.0], &[0, 10, 0], 10.0);
        assert!(damped[1] < p[1] / 5.0);
        // Identical scores: MAD is zero, the scale floor applies, and selection is by novelty alone.
        let flat = weighted_probabilities(&[2.0, 2.0], &[0, 1], 10.0);
        assert!((flat[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!(weighted_probabilities(&[], &[], 10.0).is_empty());
    }

    #[test]
    fn power_law_probabilities_follow_rank() {
        let p = power_law_probabilities(3, 1.0);
        let total = 1.0 + 0.5 + 1.0 / 3.0;
        assert!((p[0] - 1.0 / total).abs() < 1e-12 && (p[2] - (1.0 / 3.0) / total).abs() < 1e-12);
        assert_eq!(power_law_probabilities(4, 0.0), vec![0.25; 4]);
    }

    #[test]
    fn fix_mode_when_nothing_is_correct() {
        let db = db(DatabaseConfig {
            num_islands: 1,
            num_archive_inspirations: 2,
            ..Default::default()
        });
        let mut seed = program(0.0, false, 0, None);
        db.add(&mut seed).unwrap();
        let mut child = program(0.0, false, 1, Some(&seed));
        db.add(&mut child).unwrap();
        let mut grandchild = program(0.0, false, 2, Some(&child));
        db.add(&mut grandchild).unwrap();
        let mut saw_ancestry = false;
        for _ in 0..40 {
            let sample = db.sample().unwrap();
            assert!(
                sample.needs_fix && !sample.parent.correct && sample.top_k_inspirations.is_empty()
            );
            if sample.parent.id == grandchild.id {
                let ids: Vec<&str> = sample
                    .archive_inspirations
                    .iter()
                    .map(|p| p.id.as_str())
                    .collect();
                assert_eq!(ids, [seed.id.as_str(), child.id.as_str()]);
                saw_ancestry = true;
            }
        }
        assert!(saw_ancestry);
        assert!(ProgramDb::open(None, DatabaseConfig::default())
            .unwrap()
            .sample()
            .is_err());
    }

    #[test]
    fn every_strategy_returns_a_correct_parent_on_an_initialised_island() {
        for strategy in [
            "weighted",
            "power_law",
            "beam_search",
            "tournament",
            "best_of_n",
            "sequential",
            "hill_climb",
        ] {
            let (db, all) = seeded(
                DatabaseConfig {
                    num_islands: 2,
                    parent_selection_strategy: strategy.into(),
                    ..Default::default()
                },
                &[1.0, 2.0, 3.0, 0.5],
            );
            let mut wrong = program(50.0, false, 9, Some(&all[0]));
            db.add(&mut wrong).unwrap();
            for _ in 0..30 {
                let sample = db.sample().unwrap();
                assert!(sample.parent.correct && !sample.needs_fix, "{strategy}");
                assert!(sample.parent.id != wrong.id, "{strategy}");
            }
        }
    }

    #[test]
    fn weighted_and_power_law_prefer_better_programs() {
        for strategy in ["weighted", "power_law"] {
            let (db, all) = seeded(
                DatabaseConfig {
                    num_islands: 1,
                    parent_selection_strategy: strategy.into(),
                    ..Default::default()
                },
                &[1.0, 1.1, 5.0],
            );
            let counts = pick_counts(&db, 600);
            let best = counts.get(&all[2].id).copied().unwrap_or(0);
            let worst = counts.get(&all[0].id).copied().unwrap_or(0);
            assert!(best > worst, "{strategy}: {counts:?}");
        }
    }

    #[test]
    fn beam_search_sticks_to_a_parent_until_it_has_enough_children() {
        let (db, all) = seeded(
            DatabaseConfig {
                num_islands: 1,
                parent_selection_strategy: "beam_search".into(),
                num_beams: 2,
                ..Default::default()
            },
            &[1.0, 3.0],
        );
        // The beam starts at the island's best.
        let first = db.sample().unwrap().parent;
        assert_eq!(first.id, all[1].id);
        let mut child = program(2.0, true, 5, Some(&first));
        db.add(&mut child).unwrap();
        assert_eq!(db.sample().unwrap().parent.id, first.id);
        let mut improved = program(4.0, true, 6, Some(&first));
        db.add(&mut improved).unwrap();
        // Two children reached: the beam moves to the new best.
        assert_eq!(db.sample().unwrap().parent.id, improved.id);
    }

    #[test]
    fn hill_climb_always_takes_the_island_best() {
        let (db, all) = seeded(
            DatabaseConfig {
                num_islands: 1,
                parent_selection_strategy: "hill_climb".into(),
                ..Default::default()
            },
            &[1.0, 3.0, 2.0],
        );
        for _ in 0..10 {
            assert_eq!(db.sample().unwrap().parent.id, all[1].id);
        }
        let mut better = program(4.0, true, 9, Some(&all[1]));
        db.add(&mut better).unwrap();
        assert_eq!(db.sample().unwrap().parent.id, better.id);
    }

    #[test]
    fn sequential_and_best_of_n_are_deterministic() {
        let (db, all) = seeded(
            DatabaseConfig {
                num_islands: 1,
                parent_selection_strategy: "sequential".into(),
                ..Default::default()
            },
            &[1.0, 2.0, 0.5],
        );
        assert_eq!(db.sample().unwrap().parent.id, all[2].id);
        let (db, all) = seeded(
            DatabaseConfig {
                num_islands: 1,
                parent_selection_strategy: "best_of_n".into(),
                ..Default::default()
            },
            &[1.0, 2.0, 0.5],
        );
        assert_eq!(db.sample().unwrap().parent.id, all[0].id);
    }

    #[test]
    fn inspirations_exclude_the_parent_and_respect_islands() {
        let (db, _) = seeded(
            DatabaseConfig {
                num_islands: 2,
                num_archive_inspirations: 2,
                num_top_k_inspirations: 2,
                elite_selection_ratio: 0.5,
                ..Default::default()
            },
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        );
        for _ in 0..80 {
            let sample = db.sample().unwrap();
            let mut ids = HashSet::from([sample.parent.id.clone()]);
            for inspiration in sample
                .archive_inspirations
                .iter()
                .chain(&sample.top_k_inspirations)
            {
                assert!(
                    ids.insert(inspiration.id.clone()),
                    "an inspiration repeats or is the parent"
                );
                assert!(inspiration.correct && inspiration.in_archive);
                assert_eq!(inspiration.island_idx, sample.parent.island_idx);
            }
            assert!(sample.archive_inspirations.len() <= 2 && sample.top_k_inspirations.len() <= 2);
            if sample.parent.island_idx == Some(0) {
                // Island 0 holds five archived programs: all four slots can be filled.
                assert_eq!(
                    sample.archive_inspirations.len() + sample.top_k_inspirations.len(),
                    4
                );
                let shown: Vec<f64> = sample
                    .top_k_inspirations
                    .iter()
                    .map(|p| p.combined_score)
                    .collect();
                assert!(
                    shown.windows(2).all(|pair| pair[0] >= pair[1]),
                    "top-k is best first"
                );
            }
        }
    }

    #[test]
    fn without_separation_inspirations_may_cross_islands() {
        let (db, _) = seeded(
            DatabaseConfig {
                num_islands: 2,
                num_archive_inspirations: 3,
                num_top_k_inspirations: 0,
                enforce_island_separation: false,
                ..Default::default()
            },
            &[1.0, 2.0],
        );
        let crossed = (0..60).any(|_| {
            let sample = db.sample().unwrap();
            sample
                .archive_inspirations
                .iter()
                .any(|p| p.island_idx != sample.parent.island_idx)
        });
        assert!(crossed);
    }

    #[test]
    fn island_strategies_only_pick_initialised_islands() {
        for strategy in ["uniform", "equal", "proportional", "weighted"] {
            let (db, all) = seeded(
                DatabaseConfig {
                    num_islands: 3,
                    island_selection_strategy: strategy.into(),
                    ..Default::default()
                },
                &[-2.0, -1.0, -0.5],
            );
            let _ = all;
            let mut seen = HashSet::new();
            for _ in 0..200 {
                seen.insert(db.sample().unwrap().parent.island_idx.unwrap());
            }
            assert!(seen.iter().all(|island| *island < 3), "{strategy}");
            if strategy == "equal" {
                // Island 0 has three correct programs, islands 1 and 2 one each.
                assert_eq!(seen, HashSet::from([1, 2]), "{strategy}");
            } else {
                assert_eq!(seen.len(), 3, "{strategy} never visited some island");
            }
        }
    }

    /// Program ids are random, so every candidate list is ordered by
    /// insertion (rowid), never by id: the same seed and the same inserts
    /// must give the same samples.
    #[test]
    fn sampling_is_reproducible_for_a_seed() {
        for strategy in ["weighted", "power_law", "tournament"] {
            let run = || {
                let (db, _) = seeded(
                    DatabaseConfig {
                        num_islands: 2,
                        parent_selection_strategy: strategy.into(),
                        num_archive_inspirations: 2,
                        ..Default::default()
                    },
                    &[1.0, 2.0, 3.0, 2.5, 0.5, 2.0],
                );
                (0..40)
                    .map(|_| {
                        let sample = db.sample().unwrap();
                        let shown: Vec<u32> = sample
                            .archive_inspirations
                            .iter()
                            .map(|p| p.generation)
                            .collect();
                        (sample.parent.generation, sample.parent.island_idx, shown)
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(run(), run(), "{strategy}");
        }
    }
}
