# ClickHouse Setup - Summary

✅ **All tables successfully created and verified!**

## What Was Added

### 1. Database Schema (7 Tables)
All tables automatically created via `genesis/utils/clickhouse_logger.py`:

| Table | Rows | Purpose |
|-------|------|---------|
| `evolution_runs` | 0 | Track each experiment run |
| `generations` | 0 | Per-generation statistics |
| `individuals` | 0 | Every code variant evaluated |
| `pareto_fronts` | 0 | Pareto frontier snapshots |
| `code_lineages` | 0 | Parent-child relationships |
| `llm_logs` | 0 | All LLM API calls (auto-logged) |
| `agent_actions` | 0 | System events (auto-logged) |

### 2. Helper Methods
Added to `genesis/utils/clickhouse_logger.py`:
- `log_evolution_run()` - Start of experiment
- `log_generation()` - Generation stats
- `log_individual()` - Individual evaluation
- `log_pareto_front()` - Pareto frontier
- `log_lineage()` - Parent-child relationship
- `update_evolution_run()` - Mark run complete

### 3. Documentation
- **`docs/clickhouse.md`** (889 lines) - Complete guide with:
  - Detailed explanation of every table and column
  - Example queries for common analyses
  - Usage examples in code
  - Troubleshooting guide
  - Data volume estimates
  - Visualization setup

- **`docs/clickhouse_schema_reference.md`** - Quick reference:
  - All table schemas in one place
  - SQL DDL for recreation
  - Entity relationship diagram
  - Index optimization tips

### 4. Testing Script
- **`scripts/test_clickhouse.py`** - Verify connection and show schema

## Quick Start

### Test Connection
```bash
python scripts/test_clickhouse.py
```

### View Tables
All tables are created automatically. Current state:
- ✅ 7 tables exist
- ✅ Schemas validated
- ⏳ Waiting for data (need to integrate logging in runner.py)

## Next Steps

### To Start Logging Data

The tables exist but are empty. To populate them:

1. **Update `genesis/core/runner.py`** - Add logging calls:
```python
from genesis.utils.clickhouse_logger import ch_logger

# At start of evolution
ch_logger.log_evolution_run(
    run_id=run_id,
    task_name=cfg.task_name,
    config=OmegaConf.to_container(cfg),
    population_size=cfg.evolution.pop_size,
    cluster_type=cfg.cluster.type,
    database_path=db_path,
)

# After each generation
ch_logger.log_generation(
    run_id=run_id,
    generation=gen,
    num_individuals=len(population),
    best_score=max(scores),
    avg_score=np.mean(scores),
    pareto_size=len(pareto_front),
    total_cost=sum(costs),
)

# After each individual evaluation
ch_logger.log_individual(
    run_id=run_id,
    individual_id=ind.id,
    generation=gen,
    parent_id=ind.parent_id,
    mutation_type=ind.mutation_type,
    fitness_score=ind.fitness,
    combined_score=ind.combined_score,
    metrics=ind.metrics,
    is_pareto=ind.is_pareto,
    api_cost=ind.api_cost,
    embed_cost=ind.embed_cost,
    novelty_cost=ind.novelty_cost,
    code_hash=hash(ind.code),
    code_size=len(ind.code),
)
```

2. **Run an experiment** - Data will flow automatically:
```bash
genesis_launch variant=circle_packing_example
```

3. **Query the data**:
```bash
python scripts/test_clickhouse.py  # Will show row counts
```

## Documentation

- **Full Guide**: `docs/clickhouse.md`
- **Schema Reference**: `docs/clickhouse_schema_reference.md`
- **Test Script**: `scripts/test_clickhouse.py`

## Verify Setup

```bash
# Check environment
echo $CLICKHOUSE_URL

# Test connection
python scripts/test_clickhouse.py

# Should output:
# ✅ ClickHouse connection successful!
# 📊 7 tables created
```

---

**Status**: ✅ Setup complete, ready for integration!
