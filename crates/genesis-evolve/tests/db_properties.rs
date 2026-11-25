//! Stateful property tests for the run database, run with ephyra
//! (<https://github.com/datadog-labs/ephyra>).
//!
//! Each case builds a database from a generated configuration, applies a
//! generated sequence of inserts, samples and migrations, and checks the
//! invariants below after every step.
//!
//! | Property | Oracle | Source |
//! |---|---|---|
//! | No inserted program is ever lost or altered | reference model | a run database is the record of the run |
//! | Only correct programs are archived, and the archive holds at most `archive_size` programs plus one per correct island copy | invariant | ShinkaEvolve archive rule: "Incorrect programs never enter"; copies enter without a capacity check and evictions are one-for-one |
//! | `children_count` equals the number of proposals made from a program; island and spawn copies never count | invariant | it is the novelty term of weighted selection and the beam-search budget |
//! | The tracked best is correct and no correct program scores higher | reference model | `best` drives prompts, exports and beam search |
//! | `top(n)` agrees with sorting | reference model | definition |
//! | With a correct program anywhere, a sample has a correct parent; inspirations are distinct, correct, archived, never the parent, and stay on the parent's island under separation | invariant | sampling contract in `db/selection.rs` |
//! | With no correct program, sampling asks for a fix of an incorrect one | invariant | fix mode |
//! | Migration moves programs, never copies or drops them; seeds and incorrect programs stay; an island's best stays under elitism | invariant | `perform_migration` docs |
//! | Islands stay within `num_islands` unless dynamic islands are on; once anything is correct every island has a correct program (migration off or elitist) | invariant | island seeding |
//! | Selection probabilities are a distribution, monotone in score and in children | invariant, metamorphic | weighted and power-law formulas |
//! | The same seed and the same operations give the same samples, migrations and final islands | metamorphic | `db/mod.rs` docs: "sampling is reproducible given `DatabaseConfig::seed`" |

use ephyra::prelude::*;
use ephyra::{gens, BoxGen};
use genesis_core::Program;
use genesis_evolve::db::prompts::{
    percentile_on_insert, percentile_on_recompute, PromptDb, PromptDbConfig, SystemPrompt,
};
use genesis_evolve::db::{power_law_probabilities, weighted_probabilities};
use genesis_evolve::{DatabaseConfig, ProgramDb};
use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Debug, Clone, ephyra::Arbitrary)]
enum Op {
    Sample,
    Add(
        /// Which earlier program is the parent, as an index into the inserted list.
        #[ephyra(gen = gens::int(0usize..=63))]
        usize,
        /// Score in tenths, so ties are common.
        #[ephyra(gen = gens::int(-20i32..=20))]
        i32,
        #[ephyra(gen = gens::bool_with(0.6))] bool,
        /// Insert without a parent.
        #[ephyra(gen = gens::bool_with(0.1))]
        bool,
    ),
    Migrate,
}

fn pick(values: &[&str]) -> BoxGen<String> {
    gens::pick_from(values.iter().map(|v| v.to_string()).collect()).boxed()
}

fn database_config() -> BoxGen<DatabaseConfig> {
    (
        (
            gens::int(1usize..=4),
            gens::int(0usize..=6),
            pick(&[
                "weighted",
                "power_law",
                "beam_search",
                "tournament",
                "best_of_n",
                "sequential",
                "hill_climb",
            ]),
            pick(&["uniform", "equal", "proportional", "weighted"]),
            gens::bool(),
        ),
        (
            gens::pick_from(vec![0.0, 0.3, 1.0]),
            gens::int(1u32..=5),
            gens::bool(),
            gens::int(0usize..=3),
            gens::int(0usize..=3),
        ),
        (
            gens::pick_from(vec![0.0, 0.5, 1.0]),
            gens::bool_with(0.2),
            gens::int(1u32..=6),
            pick(&["initial", "best", "archive_random"]),
            pick(&["fitness", "crowding"]),
            gens::int(0u64..=1_000_000),
        ),
    )
        .map(|(a, b, c)| DatabaseConfig {
            num_islands: a.0,
            archive_size: a.1,
            parent_selection_strategy: a.2,
            island_selection_strategy: a.3,
            enforce_island_separation: a.4,
            migration_rate: b.0,
            migration_interval: b.1,
            island_elitism: b.2,
            num_archive_inspirations: b.3,
            num_top_k_inspirations: b.4,
            elite_selection_ratio: c.0,
            enable_dynamic_islands: c.1,
            stagnation_threshold: c.2,
            island_spawn_strategy: c.3,
            archive_selection_strategy: c.4,
            seed: Some(c.5),
            island_spawn_subtree_size: 2,
            num_beams: 2,
            tournament_top_k: 4,
            tournament_num_draws: 2,
            ..DatabaseConfig::default()
        })
        .boxed()
}

fn flag(program: &Program, name: &str) -> bool {
    program
        .metadata
        .get(name)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn is_copy(program: &Program) -> bool {
    flag(program, "_is_island_copy") || flag(program, "_spawned_island")
}

fn check_state(db: &ProgramDb, config: &DatabaseConfig, inserted: &[Program]) {
    let rows = db.all().unwrap();
    let by_id: HashMap<&str, &Program> = rows.iter().map(|p| (p.id.as_str(), p)).collect();
    assert_eq!(db.count().unwrap(), rows.len());

    // Nothing inserted is lost or altered.
    for original in inserted {
        let stored = by_id
            .get(original.id.as_str())
            .expect("an inserted program disappeared");
        assert_eq!(stored.code, original.code);
        assert_eq!(stored.combined_score, original.combined_score);
        assert_eq!(stored.correct, original.correct);
        assert_eq!(stored.parent_id, original.parent_id);
        assert_eq!(stored.generation, original.generation);
        assert!(!is_copy(stored));
    }
    assert_eq!(
        rows.iter().filter(|p| !is_copy(p)).count(),
        inserted.len(),
        "a non-copy row appeared from nowhere"
    );
    let generations: BTreeSet<u32> = inserted.iter().map(|p| p.generation).collect();
    assert_eq!(db.generations().unwrap(), generations);

    // Archive.
    let archived: Vec<&Program> = rows.iter().filter(|p| p.in_archive).collect();
    assert_eq!(db.archive_ids().unwrap().len(), archived.len());
    assert!(
        archived.iter().all(|p| p.correct),
        "an incorrect program is archived"
    );
    // Every correct copy entered without a capacity check, and a later
    // eviction is one-for-one, so each of them may have added a slot.
    let bypassed = rows.iter().filter(|p| is_copy(p) && p.correct).count();
    assert!(
        archived.len() <= config.archive_size + bypassed,
        "archive holds {} programs with archive_size {} and {} copies that bypassed the check",
        archived.len(),
        config.archive_size,
        bypassed
    );
    if config.archive_size == 0 {
        assert!(archived.is_empty(), "the archive is disabled but not empty");
    }

    // children_count is the number of proposals made from a program.
    for row in &rows {
        let children = rows
            .iter()
            .filter(|child| child.parent_id.as_deref() == Some(row.id.as_str()) && !is_copy(child))
            .count();
        assert_eq!(
            row.children_count as usize, children,
            "children_count of {}",
            row.id
        );
    }

    // Best and top-n against a sort.
    let mut correct_scores: Vec<f64> = rows
        .iter()
        .filter(|p| p.correct)
        .map(|p| p.combined_score)
        .collect();
    correct_scores.sort_by(|a, b| b.total_cmp(a));
    match db.best(None).unwrap() {
        Some(best) => {
            assert!(best.correct);
            assert_eq!(
                best.combined_score, correct_scores[0],
                "the tracked best is not the highest correct score"
            );
        }
        None => assert!(
            correct_scores.is_empty(),
            "correct programs exist but no best is tracked"
        ),
    }
    let top: Vec<f64> = db
        .top(5, true)
        .unwrap()
        .iter()
        .map(|p| p.combined_score)
        .collect();
    assert_eq!(
        top,
        correct_scores.iter().copied().take(5).collect::<Vec<_>>()
    );

    // Islands.
    assert!(rows.iter().all(|p| p.island_idx.is_some()));
    if !config.enable_dynamic_islands {
        assert!(rows
            .iter()
            .all(|p| p.island_idx.unwrap() < config.num_islands));
    }
    let spawned: HashSet<usize> = rows
        .iter()
        .filter(|p| flag(p, "_spawned_island"))
        .map(|p| p.island_idx.unwrap())
        .collect();
    assert!(
        spawned.iter().all(|island| *island >= config.num_islands),
        "a spawned island reuses a configured index"
    );
    let migration_keeps_islands_seeded = config.migration_rate == 0.0 || config.island_elitism;
    if !correct_scores.is_empty() && migration_keeps_islands_seeded {
        for island in 0..config.num_islands {
            assert!(
                rows.iter()
                    .any(|p| p.correct && p.island_idx == Some(island)),
                "island {island} has no correct program although one exists elsewhere"
            );
        }
    }
    let stats = db.island_stats().unwrap();
    assert_eq!(stats.iter().map(|s| s.programs).sum::<usize>(), rows.len());
}

fn check_sample(db: &ProgramDb, config: &DatabaseConfig) {
    let rows = db.all().unwrap();
    let Ok(sample) = db.sample() else {
        assert!(rows.is_empty(), "sampling failed on a non-empty database");
        return;
    };
    let any_correct = rows.iter().any(|p| p.correct);
    assert_eq!(sample.needs_fix, !any_correct);
    assert_eq!(sample.parent.correct, any_correct);
    if sample.needs_fix {
        assert!(sample.top_k_inspirations.is_empty());
        return;
    }
    let mut seen = HashSet::from([sample.parent.id.clone()]);
    for inspiration in sample
        .archive_inspirations
        .iter()
        .chain(&sample.top_k_inspirations)
    {
        assert!(
            seen.insert(inspiration.id.clone()),
            "an inspiration repeats or is the parent"
        );
        assert!(inspiration.correct && inspiration.in_archive);
        if config.enforce_island_separation {
            assert_eq!(
                inspiration.island_idx, sample.parent.island_idx,
                "an inspiration crossed islands"
            );
        }
    }
    assert!(sample.archive_inspirations.len() <= config.num_archive_inspirations);
    assert!(sample.top_k_inspirations.len() <= config.num_top_k_inspirations);
    let scores: Vec<f64> = sample
        .top_k_inspirations
        .iter()
        .map(|p| p.combined_score)
        .collect();
    assert!(
        scores.windows(2).all(|pair| pair[0] >= pair[1]),
        "top-k inspirations are not best first"
    );
}

fn check_migration(db: &ProgramDb, config: &DatabaseConfig, generation: u32) {
    let before = db.all().unwrap();
    let elites: Vec<Option<String>> = (0..config.num_islands)
        .map(|island| db.best(Some(island)).unwrap().map(|p| p.id))
        .collect();
    let moved = db.migrate(generation).unwrap();
    let after = db.all().unwrap();
    assert_eq!(
        after.len(),
        before.len(),
        "migration changed the number of programs"
    );
    let after_by_id: HashMap<&str, &Program> = after.iter().map(|p| (p.id.as_str(), p)).collect();
    let mut changed = 0;
    for program in &before {
        let now = after_by_id[program.id.as_str()];
        if now.island_idx != program.island_idx {
            changed += 1;
            assert!(
                program.correct && program.generation > 0,
                "a seed or incorrect program migrated"
            );
            assert_eq!(
                now.migration_history.len(),
                program.migration_history.len() + 1
            );
            assert!(now.island_idx.unwrap() < config.num_islands);
        } else {
            assert_eq!(now.migration_history.len(), program.migration_history.len());
        }
        assert_eq!(
            now.in_archive, program.in_archive,
            "migration changed archive membership"
        );
    }
    assert_eq!(changed, moved);
    if config.migration_rate == 0.0 || config.num_islands < 2 {
        assert_eq!(moved, 0);
    }
    if config.island_elitism {
        for (island, elite) in elites.iter().enumerate() {
            if let Some(id) = elite {
                assert_eq!(
                    after_by_id[id.as_str()].island_idx,
                    Some(island),
                    "an island's best program migrated"
                );
            }
        }
    }
}

#[ephyra::test(gen = (database_config(), gens::vec(ephyra::any::<Op>()).len(0..=40)), tests = 1500)]
fn the_database_keeps_its_invariants_under_any_sequence_of_operations(
    input: (DatabaseConfig, Vec<Op>),
) {
    let (config, ops) = input;
    let db = ProgramDb::open(None, config.clone()).unwrap();
    let mut inserted: Vec<Program> = Vec::new();
    let mut generation = 0u32;
    for op in ops {
        match op {
            Op::Sample => check_sample(&db, &config),
            Op::Migrate => check_migration(&db, &config, generation),
            Op::Add(parent_index, tenths, correct, orphan) => {
                let mut program = Program::new(format!("x = {tenths}\n"), "python");
                program.combined_score = f64::from(tenths) / 10.0;
                program.correct = correct;
                program.generation = generation;
                program.embedding = vec![f32::from(tenths as i8), 1.0, (inserted.len() % 7) as f32];
                if !inserted.is_empty() && !orphan {
                    program.parent_id = Some(inserted[parent_index % inserted.len()].id.clone());
                }
                db.add(&mut program).unwrap();
                assert!(program.island_idx.is_some());
                if program.parent_id.is_some() && generation > 0 {
                    ephyra::cover(true, "child insert");
                }
                inserted.push(program);
                generation += 1;
            }
        }
        check_state(&db, &config, &inserted);
    }
    check_sample(&db, &config);
}

/// What one run of an operation sequence did, in terms that do not depend on
/// the random program ids.
fn trace(config: &DatabaseConfig, ops: &[Op]) -> Vec<String> {
    let db = ProgramDb::open(None, config.clone()).unwrap();
    let mut inserted: Vec<Program> = Vec::new();
    let mut out = Vec::new();
    let describe = |p: &Program| {
        format!(
            "g{}s{}i{:?}c{}",
            p.generation,
            p.combined_score,
            p.island_idx,
            is_copy(p)
        )
    };
    for op in ops {
        match op {
            Op::Sample => {
                if let Ok(sample) = db.sample() {
                    let shown: Vec<String> = sample
                        .archive_inspirations
                        .iter()
                        .chain(&sample.top_k_inspirations)
                        .map(&describe)
                        .collect();
                    out.push(format!(
                        "sample {} [{}]",
                        describe(&sample.parent),
                        shown.join(",")
                    ));
                }
            }
            Op::Migrate => out.push(format!(
                "migrate {}",
                db.migrate(inserted.len() as u32).unwrap()
            )),
            Op::Add(parent_index, tenths, correct, orphan) => {
                let mut program = Program::new(format!("x = {tenths}\n"), "python");
                program.combined_score = f64::from(*tenths) / 10.0;
                program.correct = *correct;
                program.generation = inserted.len() as u32;
                program.timestamp = inserted.len() as f64;
                if !inserted.is_empty() && !orphan {
                    program.parent_id = Some(inserted[parent_index % inserted.len()].id.clone());
                }
                db.add(&mut program).unwrap();
                out.push(format!(
                    "add {} archived={}",
                    describe(&program),
                    program.in_archive
                ));
                inserted.push(program);
            }
        }
    }
    let mut islands: Vec<String> = db.all().unwrap().iter().map(&describe).collect();
    islands.sort();
    out.push(islands.join(" "));
    out
}

#[ephyra::test(gen = (database_config(), gens::vec(ephyra::any::<Op>()).len(0..=30)), tests = 600)]
fn the_same_seed_and_operations_give_the_same_run(input: (DatabaseConfig, Vec<Op>)) {
    let (config, ops) = input;
    assert_eq!(trace(&config, &ops), trace(&config, &ops));
}

fn score_list() -> BoxGen<Vec<f64>> {
    gens::frequency(vec![
        (
            3,
            gens::vec(gens::f64_range(-100.0..=100.0))
                .len(1..=12)
                .boxed(),
        ),
        (
            1,
            gens::vec(gens::int(-3i32..=3).map(f64::from))
                .len(1..=12)
                .boxed(),
        ),
    ])
    .boxed()
}

fn is_distribution(probabilities: &[f64]) -> bool {
    probabilities.iter().all(|p| p.is_finite() && *p >= 0.0)
        && (probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9
}

#[ephyra::test(gen = (score_list(), gens::vec(gens::int(0u32..=50)).len(12..=12), gens::f64_range(0.0..=50.0)))]
fn weighted_selection_is_a_distribution_monotone_in_score_and_children(
    input: (Vec<f64>, Vec<u32>, f64),
) {
    let (scores, children, lambda) = input;
    let children = &children[..scores.len()];
    let probabilities = weighted_probabilities(&scores, children, lambda);
    assert!(is_distribution(&probabilities), "{probabilities:?}");
    for i in 0..scores.len() {
        for j in 0..scores.len() {
            if scores[i] >= scores[j] && children[i] <= children[j] {
                assert!(
                    probabilities[i] >= probabilities[j] - 1e-12,
                    "a program with a higher score and no more children is less likely"
                );
            }
        }
    }
    // More children never raise a program's own chance.
    let mut busier = children.to_vec();
    busier[0] += 5;
    let damped = weighted_probabilities(&scores, &busier, lambda);
    assert!(damped[0] <= probabilities[0] + 1e-12);
}

#[ephyra::test(gen = (gens::int(1usize..=60), gens::f64_range(0.0..=6.0)))]
fn power_law_selection_is_a_distribution_decreasing_in_rank(input: (usize, f64)) {
    let (count, alpha) = input;
    let probabilities = power_law_probabilities(count, alpha);
    assert_eq!(probabilities.len(), count);
    assert!(is_distribution(&probabilities));
    assert!(probabilities.windows(2).all(|pair| pair[0] >= pair[1]));
    if alpha == 0.0 {
        assert!(probabilities
            .iter()
            .all(|p| (p - 1.0 / count as f64).abs() < 1e-12));
    }
}

#[derive(Debug, Clone, ephyra::Arbitrary)]
enum PromptOp {
    Sample,
    AddEvolved,
    Record(
        #[ephyra(gen = gens::int(0usize..=15))] usize,
        #[ephyra(gen = gens::int(-5i32..=5))] i32,
        #[ephyra(gen = gens::bool_with(0.7))] bool,
    ),
    Recompute,
}

#[ephyra::test(
    gen = (gens::int(1usize..=4), gens::pick_from(vec![0.0, 0.1, 1.0]), gens::vec(ephyra::any::<PromptOp>()).len(0..=40)),
    tests = 1500
)]
fn the_prompt_archive_stays_bounded_with_fitness_in_the_unit_interval(
    input: (usize, f64, Vec<PromptOp>),
) {
    let (archive_size, epsilon, ops) = input;
    let db = PromptDb::open(
        None,
        PromptDbConfig {
            archive_size,
            epsilon,
            seed: Some(5),
            ..Default::default()
        },
    )
    .unwrap();
    let mut seed = SystemPrompt::new("task prompt", "init");
    db.add(&mut seed).unwrap();
    let mut prompts = vec![seed.id.clone()];
    let mut correct_scores: Vec<f64> = Vec::new();
    for op in ops {
        match op {
            PromptOp::Sample => {
                let sampled = db
                    .sample()
                    .unwrap()
                    .expect("a non-empty archive always yields a prompt");
                assert!(sampled.in_archive);
            }
            PromptOp::AddEvolved => {
                let mut evolved = SystemPrompt::new(format!("evolved {}", prompts.len()), "diff");
                evolved.parent_id = Some(prompts[prompts.len() - 1].clone());
                evolved.generation = db.next_generation().unwrap();
                db.add(&mut evolved).unwrap();
                assert!(
                    evolved.in_archive,
                    "an evolved prompt was kept out of the archive"
                );
                prompts.push(evolved.id);
            }
            PromptOp::Record(which, score, correct) => {
                let score = f64::from(score);
                if correct {
                    correct_scores.push(score);
                }
                db.record_program(
                    &prompts[which % prompts.len()],
                    "program",
                    score,
                    correct,
                    &correct_scores,
                )
                .unwrap();
            }
            PromptOp::Recompute => db.recompute_fitness(&correct_scores).unwrap(),
        }
        let all = db.all().unwrap();
        assert_eq!(all.len(), prompts.len());
        let archived = db.archive().unwrap();
        assert_eq!(
            archived.len(),
            archive_size.min(prompts.len()),
            "the prompt archive shrank or overflowed"
        );
        for prompt in &all {
            assert!(
                (0.0..=1.0).contains(&prompt.fitness),
                "fitness {} out of range",
                prompt.fitness
            );
            assert!(prompt.correct_program_count <= prompt.program_count);
            assert_eq!(
                prompt.program_scores.len() as u32,
                prompt.correct_program_count
            );
        }
    }
}

#[ephyra::test(gen = (gens::int(-5i32..=5).map(f64::from), score_list()))]
fn prompt_percentiles_are_shares_and_order_preserving(input: (f64, Vec<f64>)) {
    let (score, mut population) = input;
    population.push(score);
    for percentile in [
        percentile_on_insert(score, &population),
        percentile_on_recompute(score, &population),
    ] {
        assert!((0.0..=1.0).contains(&percentile));
    }
    let higher = score + 1.0;
    population.push(higher);
    assert!(percentile_on_insert(higher, &population) >= percentile_on_insert(score, &population));
    assert!(
        percentile_on_recompute(higher, &population) >= percentile_on_recompute(score, &population)
    );
}
