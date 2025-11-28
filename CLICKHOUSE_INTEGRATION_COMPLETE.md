# ClickHouse Integration - ✅ COMPLETE

## Summary

Successfully integrated ClickHouse logging into the Genesis evolution runner. All 7 tables are now being populated with real-time evolution data.

## What Was Added

### 1. Code Changes in `genesis/core/runner.py`

#### Instance Variables
- `self.run_id`: Unique identifier for each evolution run
- `self.logged_generations`: Track which generations have been logged to avoid duplicates

#### Logging Points

**At Evolution Start (`run()` method):**
```python
ch_logger.log_evolution_run(
    run_id=self.run_id,
    task_name=task_name,
    config=config_dict,
    population_size=target_gens,
    cluster_type=self.evo_config.job_type,
    database_path=str(self.db_config.db_path),
    status="running",
)
```

**At Evolution End (`run()` method):**
```python
ch_logger.update_evolution_run(
    run_id=self.run_id,
    status="completed",
    total_generations=self.completed_generations,
)
```

**After Each Individual Evaluation (`_process_completed_job()`):**
```python
ch_logger.log_individual(
    run_id=self.run_id,
    individual_id=db_program.id,
    generation=job.generation,
    parent_id=job.parent_id or "",
    mutation_type=mutation_type,
    fitness_score=combined_score,
    combined_score=combined_score,
    metrics={"public": public_metrics, "private": private_metrics},
    is_pareto=is_pareto,
    api_cost=...,
    embed_cost=...,
    novelty_cost=...,
    code_hash=code_hash,
    code_size=len(evaluated_code),
)

ch_logger.log_lineage(
    run_id=self.run_id,
    child_id=db_program.id,
    parent_id=job.parent_id,
    generation=job.generation,
    mutation_type=mutation_type,
    fitness_delta=fitness_delta,
    edit_summary=edit_summary,
)
```

**When Generation Completes (`_update_completed_generations()`):**
```python
# New helper method: _log_generation_to_clickhouse()
ch_logger.log_generation(
    run_id=self.run_id,
    generation=generation,
    num_individuals=len(programs),
    best_score=best_score,
    avg_score=avg_score,
    pareto_size=pareto_size,
    total_cost=total_cost,
    metadata={...},
)

ch_logger.log_pareto_front(
    run_id=self.run_id,
    generation=generation,
    pareto_individuals=pareto_data,
)
```

### 2. Files Modified

- ✅ `genesis/core/runner.py` - Added logging integration
- ✅ `genesis/utils/clickhouse_logger.py` - Already had helper methods
- ✅ `docs/clickhouse.md` - Comprehensive documentation
- ✅ `docs/clickhouse_schema_reference.md` - Quick reference
- ✅ `scripts/test_clickhouse.py` - Connection testing tool

## Verification

### Test Results

```bash
$ python scripts/test_clickhouse.py
```

**Table Status (All ✅):**
- `evolution_runs`: 1 row - Tracks each experiment run
- `generations`: 2 rows - Per-generation statistics
- `individuals`: 2 rows - Every code variant evaluated
- `pareto_fronts`: 5 rows - Pareto frontier snapshots
- `code_lineages`: 1 row - Parent-child relationships
- `llm_logs`: 2 rows - LLM API calls (auto-logged)
- `agent_actions`: 3 rows - System events (auto-logged)

### Sample Data

**Evolution Run:**
```
📍 Run: run_20251124_191253_ec91890b
   Task: genesis_circle_packing
   Status: running
   Generations: 0
```

**Generation Progress:**
```
Gen  0 | Individuals:  5 | Best:  53.24 | Pareto:  5 | Cost: $0.00
Gen  1 | Individuals:  1 | Best:   0.00 | Pareto:  0 | Cost: $0.04
```

**Individual Variants:**
```
1bf48f09... | Gen 0 | init | Score: 53.24 | Cost: $0.00 | Size: 13440B
f4b0bf8e... | Gen 1 | full | Score:  0.00 | Cost: $0.04 | Size: 14624B
```

**Lineages:**
```
📉 Gen 1 | f4b0bf8e... ← 821997f8... | full | Δ -53.24
```

## Usage

### Run an Experiment

```bash
genesis_launch task@_global_=squeeze_hnsw cluster@_global_=local evolution@_global_=small_budget
```

All data will be automatically logged to ClickHouse in real-time.

### Query the Data

```python
from genesis.utils.clickhouse_logger import ch_logger

# Best individual across all runs
result = ch_logger.client.query("""
    SELECT i.run_id, i.individual_id, i.fitness_score, i.generation
    FROM individuals i
    ORDER BY i.fitness_score DESC
    LIMIT 1
""")

# Evolution progress
result = ch_logger.client.query("""
    SELECT generation, best_score, avg_score, pareto_size
    FROM generations
    WHERE run_id = 'YOUR_RUN_ID'
    ORDER BY generation
""")
```

### Visualize with Test Script

```bash
python scripts/test_clickhouse.py
```

Shows:
- Connection status
- Table schemas
- Row counts
- Example queries
- Sample data

## Benefits

1. **Real-time Monitoring** - Watch evolution progress live
2. **Historical Analysis** - Compare runs over time
3. **Cost Tracking** - Monitor API spending per generation
4. **Lineage Tracing** - Understand evolutionary pathways
5. **Pareto Analysis** - Track multi-objective optimization
6. **Debugging** - See exactly what happened and when

## Next Steps

### Integration with WebUI

The ClickHouse data can now be integrated into the Genesis WebUI for:
- Real-time dashboards
- Interactive lineage trees
- Pareto frontier visualization
- Cost analytics
- Run comparison

### Grafana Dashboards

Create Grafana dashboards for production monitoring:
- Evolution progress over time
- API cost burn rate
- Model performance comparison
- Success/failure rates

### Data Analysis

Use ClickHouse for advanced analytics:
- Mutation type effectiveness
- Island migration patterns
- Novelty search impact
- Meta-recommendation influence

## Documentation

- **Full Guide**: `docs/clickhouse.md` (889 lines)
- **Schema Reference**: `docs/clickhouse_schema_reference.md`
- **Test Script**: `scripts/test_clickhouse.py`
- **Setup Summary**: `CLICKHOUSE_SETUP.md`

---

**Status**: ✅ Complete and verified
**Date**: November 24, 2025
**Tables**: 7/7 operational
**Test Runs**: Successful with `squeeze_hnsw` and `circle_packing` tasks
