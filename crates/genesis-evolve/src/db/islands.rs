//! Islands: where a new program lives, how the seed reaches every island,
//! migration between islands, and spawning a fresh island on stagnation.

use super::{
    best, get, insert_row, meta_get, meta_set, query, META_BEST_GENERATION, META_LAST_MIGRATION,
};
use crate::config::DatabaseConfig;
use anyhow::Result;
use genesis_core::{fsutil, Program};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// Islands that hold at least one correct program, ascending.
pub(crate) fn initialized_islands(conn: &Connection) -> Result<Vec<usize>> {
    let mut statement = conn.prepare(
        "SELECT DISTINCT island_idx FROM programs WHERE correct = 1 AND island_idx IS NOT NULL ORDER BY island_idx",
    )?;
    let islands = statement
        .query_map([], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<i64>>>()?;
    Ok(islands.into_iter().map(|idx| idx.max(0) as usize).collect())
}

/// The first program founds island 0; a child lives with its parent; an
/// orphan goes to the lowest island that has no correct program yet, or to a
/// random one once every island is initialised.
pub(crate) fn assign_island(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
    program: &mut Program,
    is_first: bool,
) -> Result<()> {
    if is_first {
        program.island_idx = Some(0);
        return Ok(());
    }
    if let Some(parent_id) = &program.parent_id {
        if let Some(island) = get(conn, parent_id)?.and_then(|parent| parent.island_idx) {
            program.island_idx = Some(island);
            return Ok(());
        }
    }
    let initialized: BTreeSet<usize> = initialized_islands(conn)?.into_iter().collect();
    let island = (0..config.num_islands)
        .find(|island| !initialized.contains(island))
        .unwrap_or_else(|| rng.gen_range(0..config.num_islands));
    program.island_idx = Some(island);
    Ok(())
}

/// Insert a copy of `original` on `island`. Copies carry no costs, so run
/// totals are not counted twice, and they are not proposals, so they start
/// with no children and never add to a parent's `children_count`: that count
/// is the novelty term of parent selection and means "proposals made from
/// this program" everywhere. While the archive is enabled, correct copies
/// join it without a capacity check: every island needs an archived program
/// to sample from.
fn insert_copy(
    conn: &Connection,
    config: &DatabaseConfig,
    original: &Program,
    island: usize,
    flags: &[(&str, Value)],
) -> Result<Program> {
    let mut copy = original.clone();
    copy.id = uuid::Uuid::new_v4().to_string();
    copy.island_idx = Some(island);
    copy.system_prompt_id = None;
    copy.children_count = 0;
    copy.in_archive = false;
    copy.migration_history = Vec::new();
    for key in ["api_costs", "embed_cost", "novelty_cost", "meta_cost"] {
        if copy.metadata.contains_key(key) {
            copy.metadata.insert(key.to_string(), json!(0.0));
        }
    }
    for (key, value) in flags {
        copy.metadata.insert((*key).to_string(), value.clone());
    }
    insert_row(conn, &copy)?;
    if copy.correct && config.archive_size > 0 {
        conn.execute(
            "INSERT OR IGNORE INTO archive (program_id) VALUES (?1)",
            [&copy.id],
        )?;
        copy.in_archive = true;
    }
    Ok(copy)
}

fn island_copy_flags(original: &Program) -> Vec<(&'static str, Value)> {
    vec![
        ("_is_island_copy", json!(true)),
        ("_original_program_id", json!(original.id)),
    ]
}

/// Give every other island its own copy of the seed program.
pub(crate) fn copy_seed_to_islands(
    conn: &Connection,
    config: &DatabaseConfig,
    seed: &Program,
) -> Result<()> {
    for island in 1..config.num_islands {
        insert_copy(conn, config, seed, island, &island_copy_flags(seed))?;
    }
    Ok(())
}

/// Copy a correct program into every configured island that still has no
/// correct program, which is how islands recover from an incorrect seed.
pub(crate) fn seed_uninitialized_islands(
    conn: &Connection,
    config: &DatabaseConfig,
    program: &Program,
) -> Result<()> {
    let initialized: BTreeSet<usize> = initialized_islands(conn)?.into_iter().collect();
    for island in (0..config.num_islands).filter(|island| !initialized.contains(island)) {
        insert_copy(conn, config, program, island, &island_copy_flags(program))?;
    }
    Ok(())
}

/// True once per `migration_interval` generations, on the first program that
/// crosses into a new interval; out-of-order and failed generations neither
/// skip a migration nor trigger two.
pub(crate) fn migration_due(
    conn: &Connection,
    config: &DatabaseConfig,
    generation: u32,
) -> Result<bool> {
    if config.num_islands < 2
        || config.migration_rate <= 0.0
        || config.migration_interval == 0
        || generation == 0
    {
        return Ok(false);
    }
    let last: u32 = meta_get(conn, META_LAST_MIGRATION)?
        .and_then(|text| text.parse().ok())
        .unwrap_or(0);
    Ok(generation / config.migration_interval > last / config.migration_interval)
}

/// Move a share of each island's correct, non-seed programs to another
/// island. Every move is planned on the pre-migration population.
pub(crate) fn perform_migration(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
    generation: u32,
) -> Result<usize> {
    if config.num_islands < 2 || config.migration_rate <= 0.0 {
        return Ok(0);
    }
    let mut planned: Vec<(Program, usize)> = Vec::new();
    let mut seen = HashSet::new();
    for source in 0..config.num_islands {
        let size: i64 = conn.query_row(
            "SELECT COUNT(*) FROM programs WHERE island_idx = ?1",
            [source as i64],
            |row| row.get(0),
        )?;
        if size <= 1 {
            continue;
        }
        let migrants = ((size as f64 * config.migration_rate).floor() as usize).max(1);
        let mut eligible = query(
            conn,
            "WHERE p.island_idx = ?1 AND p.generation > 0 AND p.correct = 1 ORDER BY p.rowid",
            &[&(source as i64)],
        )?;
        if config.island_elitism {
            if let Some(elite) = best(conn, Some(source))? {
                eligible.retain(|program| program.id != elite.id);
            }
        }
        eligible.shuffle(rng);
        for program in eligible.into_iter().take(migrants) {
            if seen.insert(program.id.clone()) {
                planned.push((program, source));
            }
        }
    }

    let moved = planned.len();
    for (mut program, source) in planned {
        let destinations: Vec<usize> = (0..config.num_islands)
            .filter(|island| *island != source)
            .collect();
        let destination = *destinations.choose(rng).expect("at least two islands");
        program.migration_history.push(json!({
            "generation": generation,
            "from": source,
            "to": destination,
            "timestamp": fsutil::now_ts(),
        }));
        conn.execute(
            "UPDATE programs SET island_idx = ?1, migration_history = ?2 WHERE id = ?3",
            params![
                destination as i64,
                serde_json::to_string(&program.migration_history)?,
                program.id
            ],
        )?;
    }
    Ok(moved)
}

/// When the best score has not improved for `stagnation_threshold`
/// generations, copy a small subtree into a brand-new island so the search
/// can restart from somewhere else without discarding anything.
pub(crate) fn spawn_island_if_stagnant(
    conn: &Connection,
    config: &DatabaseConfig,
    rng: &mut ChaCha8Rng,
    generation: u32,
) -> Result<Option<usize>> {
    let best_generation: u32 = meta_get(conn, META_BEST_GENERATION)?
        .and_then(|text| text.parse().ok())
        .unwrap_or(0);
    if generation.saturating_sub(best_generation) < config.stagnation_threshold {
        return Ok(None);
    }
    let source = match config.island_spawn_strategy.as_str() {
        "best" => best(conn, None)?,
        "archive_random" => {
            let members = query(conn, "WHERE a.program_id IS NOT NULL ORDER BY p.rowid", &[])?;
            members.choose(rng).cloned()
        }
        _ => query(
            conn,
            "WHERE p.generation = 0 AND p.parent_id IS NULL ORDER BY p.timestamp, p.rowid LIMIT 1",
            &[],
        )?
        .into_iter()
        .next(),
    };
    let Some(source) = source else {
        return Ok(None);
    };

    let highest: Option<i64> =
        conn.query_row("SELECT MAX(island_idx) FROM programs", [], |row| row.get(0))?;
    let new_island = (highest.map(|idx| idx as usize + 1).unwrap_or(0)).max(config.num_islands);

    // Breadth-first: the source, then its correct children, best first.
    let mut subtree = Vec::new();
    let mut queue = VecDeque::from([source.clone()]);
    while let Some(program) = queue.pop_front() {
        if subtree.len() >= config.island_spawn_subtree_size.max(1) {
            break;
        }
        let children = query(
            conn,
            "WHERE p.parent_id = ?1 AND p.correct = 1 ORDER BY p.combined_score DESC, p.rowid",
            &[&program.id],
        )?;
        queue.extend(children);
        subtree.push(program);
    }

    let mut new_ids: HashMap<String, String> = HashMap::new();
    for (position, original) in subtree.iter().enumerate() {
        let mut flags = vec![
            ("_spawned_island", json!(true)),
            ("_spawned_from_program_id", json!(original.id)),
            ("_spawn_island_idx", json!(new_island)),
            ("_spawn_strategy", json!(config.island_spawn_strategy)),
        ];
        if position > 0 {
            flags.push(("_spawned_as_child", json!(true)));
        }
        let mut template = original.clone();
        template.parent_id = original
            .parent_id
            .as_ref()
            .and_then(|parent| new_ids.get(parent).cloned());
        template.timestamp = fsutil::now_ts();
        template.code_diff = None;
        let copy = insert_copy(conn, config, &template, new_island, &flags)?;
        new_ids.insert(original.id.clone(), copy.id);
    }
    meta_set(
        conn,
        META_BEST_GENERATION,
        &best_generation.max(generation).to_string(),
    )?;
    Ok(Some(new_island))
}

#[cfg(test)]
mod tests {
    use super::super::is_copy;
    use super::super::testing::{db, program};
    use crate::config::DatabaseConfig;

    #[test]
    fn the_seed_is_copied_to_every_island_and_children_stay_with_parents() {
        let db = db(DatabaseConfig {
            num_islands: 3,
            ..Default::default()
        });
        let mut seed = program(1.0, true, 0, None);
        seed.metadata
            .insert("api_costs".into(), serde_json::json!(0.25));
        db.add(&mut seed).unwrap();
        let all = db.all().unwrap();
        assert_eq!(all.len(), 3);
        let mut islands: Vec<usize> = all.iter().map(|p| p.island_idx.unwrap()).collect();
        islands.sort();
        assert_eq!(islands, [0, 1, 2]);
        let copies: Vec<_> = all.iter().filter(|p| is_copy(p)).collect();
        assert_eq!(copies.len(), 2);
        for copy in &copies {
            assert_eq!(
                copy.metadata["_original_program_id"],
                serde_json::json!(seed.id)
            );
            assert_eq!(copy.metadata["api_costs"], serde_json::json!(0.0));
            assert!(copy.in_archive && copy.correct);
        }
        assert_eq!(db.generations().unwrap().len(), 1);

        let on_island_two = copies.iter().find(|p| p.island_idx == Some(2)).unwrap();
        let mut child = program(2.0, true, 1, Some(on_island_two));
        db.add(&mut child).unwrap();
        assert_eq!(child.island_idx, Some(2));
    }

    #[test]
    fn islands_recover_from_an_incorrect_seed() {
        let db = db(DatabaseConfig {
            num_islands: 3,
            ..Default::default()
        });
        let mut seed = program(0.0, false, 0, None);
        db.add(&mut seed).unwrap();
        assert!(db.archive_ids().unwrap().is_empty());
        let mut fixed = program(1.0, true, 1, Some(&seed));
        db.add(&mut fixed).unwrap();
        // The fix lives on island 0 and is copied to the two islands that had nothing correct.
        let stats = db.island_stats().unwrap();
        assert!(
            stats.iter().all(|island| island.correct_programs == 1),
            "{stats:?}"
        );
        assert_eq!(db.count().unwrap(), 6);
        assert_eq!(db.archive_ids().unwrap().len(), 3);
    }

    #[test]
    fn migration_moves_correct_non_elite_programs_and_records_history() {
        let db = db(DatabaseConfig {
            num_islands: 2,
            migration_rate: 0.5,
            migration_interval: 3,
            ..Default::default()
        });
        let mut seed = program(1.0, true, 0, None);
        db.add(&mut seed).unwrap();
        let mut children = Vec::new();
        for generation in 1..=2 {
            let mut child = program(1.0 + f64::from(generation), true, generation, Some(&seed));
            db.add(&mut child).unwrap();
            children.push(child);
        }
        assert!(db
            .all()
            .unwrap()
            .iter()
            .all(|p| p.migration_history.is_empty()));

        // Generation 3 crosses the interval: island 0 holds seed + 3 children.
        let mut third = program(0.5, true, 3, Some(&seed));
        db.add(&mut third).unwrap();
        let all = db.all().unwrap();
        let migrated: Vec<_> = all
            .iter()
            .filter(|p| !p.migration_history.is_empty())
            .collect();
        assert_eq!(
            migrated.len(),
            2,
            "floor(4 * 0.5) migrants from island 0, none from the single-program island"
        );
        for program in &migrated {
            assert!(program.generation > 0 && program.correct);
            assert_eq!(program.island_idx, Some(1));
            assert_eq!(program.migration_history[0]["from"], 0);
            assert_eq!(program.migration_history[0]["generation"], 3);
        }
        // The island's best program (generation 2, score 3.0) never leaves.
        let elite = all.iter().find(|p| p.id == children[1].id).unwrap();
        assert_eq!(elite.island_idx, Some(0));
        assert_eq!(
            all.len(),
            5,
            "migration moves programs; it never copies or drops them"
        );

        // Not due again until generation 6.
        let mut fourth = program(0.1, true, 4, Some(&seed));
        db.add(&mut fourth).unwrap();
        assert_eq!(
            db.all()
                .unwrap()
                .iter()
                .filter(|p| !p.migration_history.is_empty())
                .count(),
            2
        );
    }

    #[test]
    fn a_skipped_generation_does_not_skip_the_migration() {
        let db = db(DatabaseConfig {
            num_islands: 2,
            migration_rate: 1.0,
            migration_interval: 5,
            ..Default::default()
        });
        let mut seed = program(1.0, true, 0, None);
        db.add(&mut seed).unwrap();
        for generation in [1, 2, 3, 4, 6] {
            // Generation 5 failed and has no row.
            let mut child = program(0.5, true, generation, Some(&seed));
            db.add(&mut child).unwrap();
        }
        assert!(db
            .all()
            .unwrap()
            .iter()
            .any(|p| !p.migration_history.is_empty()));
    }

    /// Found by `the_database_keeps_its_invariants_under_any_sequence_of_operations`
    /// (ephyra, seed 882211 configuration with dynamic islands): a spawned
    /// subtree incremented its copied root's `children_count`, while island
    /// copies of a child never did, so the count meant two different things.
    #[test]
    fn copies_never_count_as_children() {
        let db = db(DatabaseConfig {
            num_islands: 3,
            migration_rate: 1.0,
            migration_interval: 1,
            island_elitism: false,
            enable_dynamic_islands: true,
            stagnation_threshold: 6,
            island_spawn_strategy: "best".into(),
            island_spawn_subtree_size: 2,
            ..Default::default()
        });
        let mut inserted = Vec::new();
        let plan: [(usize, f64, bool); 8] = [
            (0, 0.0, false),
            (0, 0.2, true),
            (1, 0.0, true),
            (0, 0.0, true),
            (0, 0.0, true),
            (0, 0.0, true),
            (0, 0.0, false),
            (0, 0.0, true),
        ];
        for (generation, (parent, score, correct)) in plan.into_iter().enumerate() {
            let parent = inserted.get(parent).cloned();
            let mut next = program(score, correct, generation as u32, parent.as_ref());
            db.add(&mut next).unwrap();
            inserted.push(next);
        }
        let rows = db.all().unwrap();
        assert!(
            rows.iter().any(|p| p.island_idx == Some(3)),
            "the stagnant run spawned an island"
        );
        for row in &rows {
            let proposals = rows
                .iter()
                .filter(|child| {
                    child.parent_id.as_deref() == Some(row.id.as_str()) && !is_copy(child)
                })
                .count();
            assert_eq!(
                row.children_count as usize, proposals,
                "children_count of generation {}",
                row.generation
            );
        }
    }

    #[test]
    fn stagnation_spawns_a_new_island_once() {
        let db = db(DatabaseConfig {
            num_islands: 2,
            enable_dynamic_islands: true,
            stagnation_threshold: 3,
            island_spawn_strategy: "best".into(),
            island_spawn_subtree_size: 2,
            ..Default::default()
        });
        let mut seed = program(1.0, true, 0, None);
        db.add(&mut seed).unwrap();
        let mut best = program(5.0, true, 1, Some(&seed));
        db.add(&mut best).unwrap();
        let mut child_of_best = program(2.0, true, 2, Some(&best));
        db.add(&mut child_of_best).unwrap();
        assert!(db.island_stats().unwrap().len() == 2);
        // Generations 3 and 4 do not improve; at generation 4 the best is 3 generations old.
        for generation in [3, 4] {
            let mut worse = program(0.5, true, generation, Some(&seed));
            db.add(&mut worse).unwrap();
        }
        let stats = db.island_stats().unwrap();
        assert_eq!(stats.len(), 3, "{stats:?}");
        assert_eq!(stats[2].island_idx, 2);
        assert_eq!(stats[2].programs, 2);
        let spawned: Vec<_> = db
            .all()
            .unwrap()
            .into_iter()
            .filter(|p| p.island_idx == Some(2))
            .collect();
        let root = spawned.iter().find(|p| p.parent_id.is_none()).unwrap();
        let leaf = spawned.iter().find(|p| p.parent_id.is_some()).unwrap();
        assert_eq!(root.combined_score, 5.0);
        assert_eq!(leaf.parent_id.as_deref(), Some(root.id.as_str()));
        assert_eq!(
            root.children_count, 0,
            "a copied child is not a proposal made from the copied root"
        );
        assert_eq!(leaf.metadata["_spawned_as_child"], serde_json::json!(true));
        // The stagnation clock was reset, so the next generation spawns nothing.
        let mut next = program(0.5, true, 5, Some(&seed));
        db.add(&mut next).unwrap();
        assert_eq!(db.island_stats().unwrap().len(), 3);
    }
}
