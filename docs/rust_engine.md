# Genesis in Rust

The `genesis` binary evolves a program against an evaluator you supply, then studies what it found. It
is a Rust port of two Sakana AI projects, combined into one engine:

- [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) supplies the search: a population of programs
  on islands, mutated by an ensemble of models that a bandit chooses between.
- [The Agentic Scientist](https://github.com/SakanaAI/AI-Scientist-v3) (AI-Scientist-v3) supplies the
  science: ideas with rubrics, coding agents that implement them, ablations, a sealed final evaluation,
  post-hoc analysis, and a paper whose numbers are audited against the run's artifacts.

The Python package under `genesis/` is unchanged. The Rust engine reads the same task layout and writes
the same run database schema. The repository's Python examples run under it as they are: their
`evaluate.py` scripts take the same two flags. Those scripts import the Python package, so they need it
installed in the interpreter named by `GENESIS_PYTHON` (default `python3`).

## Install

```bash
cargo install --path crates/genesis-cli    # installs the `genesis` binary
```

Rust 1.91 or later. Evaluators run as ordinary commands, so a task brings its own requirements: the
bundled example needs `rustc` and `python3`.

## Quick start

Run the search with no credentials. The built-in `mock/perturb` mutator nudges numeric literals, which is
enough to produce a full run directory:

```bash
genesis evolve examples/circle_packing_rs --config configs/rust/offline_demo.yaml
genesis inspect            # http://127.0.0.1:8765
```

Run the search with real models. Set the provider's key, then name the models:

```bash
export ANTHROPIC_API_KEY=...
genesis evolve examples/circle_packing_rs \
  'evo.llm_models=[claude-sonnet-5-5,claude-haiku-5-5]' evo.num_generations=40 evo.max_api_costs=3
```

Run the whole pipeline, from the search to a paper:

```bash
genesis doctor examples/circle_packing_rs --evaluate     # task, evaluator, credentials, tools
genesis run examples/circle_packing_rs --dry-run         # print the plan, run nothing
genesis run examples/circle_packing_rs --config configs/rust/pipeline_quickstart.yaml
```

Every setting is a `key=value` override after the task directory. Quote a list value, because shells
read brackets as a pattern. `genesis config` prints the defaults as YAML, and an unknown key fails at
load with the nearest valid name.

| Command | What it does |
|---|---|
| `genesis run <task>` | The pipeline, or the phases named in `execution.phases` |
| `genesis evolve <task>` | The search alone |
| `genesis eval <task>` | Evaluate one program with the task's evaluator and print the outcome as JSON |
| `genesis review <file>` | Review a manuscript with the reviewer ensemble |
| `genesis doctor <task>` | Check the task, the evaluator, credentials, agents, and tools before a run |
| `genesis inspect` | Browse runs in a local web page |
| `genesis config [pipeline\|evolve]` | Print a default configuration as YAML |
| `genesis models <name>...` | Show how model names resolve and which credentials are missing |

## A task

A task is a directory with a goal, a seed program, and an evaluator.

```text
my_task/
├── task.md        the goal, in prose; models read it as the task description
├── initial.rs     the seed program, in any supported language
├── evaluate.py    the evaluator (or evaluate.sh, or a command named in task.toml)
└── task.toml      optional: name, language, eval_command, eval_timeout_secs, final_eval_command
```

The Agentic Scientist layout (`program_0/program_spec.toml` with the entry file under `src/`) loads too.

**The seed program** marks what may change. Models and agents may edit only the text between a line
containing `EVOLVE-BLOCK-START` and the next line containing `EVOLVE-BLOCK-END`. The engine rejects any
edit that touches text outside those markers.

**The evaluator** is any command. Genesis runs it as

```text
python3 evaluate.py --program_path <file> --results_dir <dir>
```

and reads two files from the results directory:

| File | Content |
|---|---|
| `metrics.json` | `{"combined_score": <number, higher is better>, "public": {...}, "private": {...}, "text_feedback": "..."}` |
| `correct.json` | `{"correct": <bool>, "error": <string or null>}`, written last |

The harness runs the evaluator itself and distrusts the result. A claimed success with a non-zero exit
status, a non-finite score, or malformed metrics is recorded as a failed evaluation. On a timeout the
whole process group is killed. If the evaluator cannot score the seed program, the run stops before any
model is called; `genesis doctor <task> --evaluate` makes the same check for free.

A stochastic evaluator reads its seed from `GENESIS_EVAL_SEED`. Ablations and the final evaluation
replicate by varying that seed. When an evaluator ignores it, every replication returns the same score,
and the reports say that intervals are not meaningful.

A task may ship a second evaluator, `evaluate_final.py`, for the sealed final evaluation. The search
never runs it.

With only a `task.md`, add the `task_drafting` phase. A model writes the seed and the evaluator, and the
harness accepts them only if every gate passes: the task loads, the seed has a valid EVOLVE-BLOCK, the
evaluator scores the seed twice with the same result and judges it correct, a program with its block
emptied scores worse, and evaluating changes nothing in the task directory.

## The search

Generation 0 is the seed. Each later generation is one proposal:

1. **Sample.** Pick an island, a parent on it, and inspiration programs from the archive.
2. **Choose an arm.** A bandit picks one entry of `llm_models`. An entry is a model name or
   `agent:<cli>[@model]`.
3. **Propose.** A model writes a SEARCH/REPLACE diff, a full rewrite, or a crossover with an inspiration.
   A coding agent edits a scratch copy of the parent instead.
4. **Reject duplicates.** If the candidate's embedding is nearly identical to a program on the same
   island, a judge model decides whether the change is meaningful.
5. **Evaluate.** The harness runs the evaluator and stores the program, correct or not.
6. **Learn.** The bandit is rewarded by the improvement over the parent. The archive, the best program,
   and the islands are updated.

Up to `max_evaluation_jobs` proposals are in flight at once. Results are stored one at a time, in
completion order.

| Mechanism | Setting | What it does |
|---|---|---|
| Islands | `db.num_islands`, `db.migration_rate` | Separate populations that exchange programs every `migration_interval` generations |
| Parent selection | `db.parent_selection_strategy` | `weighted` (default), `power_law`, `beam_search`, `tournament`, `best_of_n`, `sequential`, `hill_climb` |
| Model bandit | `evo.llm_dynamic_selection` | `ucb` (asymmetric, cost-aware), `thompson`, `fixed` |
| Novelty | `evo.novelty_llm_models`, `evo.code_embed_sim_threshold` | Rejects near-duplicates before evaluation |
| Meta scratchpad | `evo.meta_llm_models`, `evo.meta_rec_interval` | Summarises what was tried and recommends what to try next |
| Prompt evolution | `evo.evolve_prompts` | Evolves the task prompt alongside the programs |
| Fix mode | automatic | While no correct program exists, proposals repair an incorrect one using its error output |
| Budget | `evo.max_api_costs` | Stops new proposals when the next one may not fit in what is left |

### Engines

`investigation.engine` selects how proposals are framed.

| Engine | Behaviour |
|---|---|
| `evolve` | ShinkaEvolve: the loop above. |
| `scientist` | The Agentic Scientist: every proposal starts from the best program and carries an idea. |
| `hybrid` (default for `genesis run`) | The population and the bandit of `evolve`, with an idea, rubrics, memory, and conclusions on every proposal. |

With ideas on, one model call writes an **idea** before each proposal: a hypothesis, a plan, and target
rubrics. Whichever model or agent implements the proposal sees that idea. After the evaluation the
harness, not the proposer, checks the rubrics:

- a **metric** rubric is compared with the evaluator's numbers;
- an **implementation** rubric with `source_contains` patterns is checked against the program's code
  tokens, so a comment that mentions a technique does not pass;
- a rubric the harness cannot check is reported as not assessed, unless `rubric_judge_models` is set;
- when the evaluation did not establish correctness, no rubric is assessed.

Every evaluated proposal is appended to `memory.jsonl`. Every few proposals a model revises
`conclusions.md`, one paragraph on which directions looked promising and which did not. The paragraph
changes only when a cycle contradicts it, opens a new category, flips a do or don't, or refines a
sentence. Later ideas and proposals read it.

### Coding agents as arms

`agent:claude`, `agent:codex`, `agent:gemini`, `agent:opencode`, `agent:cursor`, and `agent:pi` drive the
installed CLI. The agent receives a scratch directory with the parent program, the task, the idea, and
the inspirations. It edits one file and exits. It never runs the evaluator and never reports a score:
the harness evaluates what it left, and discards an edit that changes text outside the EVOLVE-BLOCK.

An agent arm runs the CLI with its permission prompts off (`--dangerously-skip-permissions` for Claude Code,
and the equivalent flag of each other CLI), because nobody is there to answer them. The agent can run any
command your user can. Run tasks you trust, or run Genesis in a container. A `cli/<agent>` model, by
contrast, runs the same CLI read-only with no tools.

The Claude adapter removes `ANTHROPIC_API_KEY` from the agent's environment, so the agent bills your
Claude login, and runs without your user settings, hooks, or MCP servers. To bill an API key instead, set
it in the agent's `env` in the configuration.

Verified by real runs on the development machine: Claude Code in edit, read-only, and plain-model modes;
Codex read-only; Pi as a plain model. The Gemini, OpenCode, and Cursor adapters follow each CLI's `--help`
and have not been run against a logged-in CLI.

Because agents and models are arms of the same bandit, the search spends on whichever improves the
score more per dollar.

## The pipeline

`genesis run` chains eight phases. Six run by default. `review` and `revision` are opt-in.

| Phase | Kind | What it produces |
|---|---|---|
| `task_drafting` | gating | A seed and an evaluator, when the task has only a brief |
| `investigation` | gating | The search, and the frozen choice of program to study |
| `ablations` | best effort | A control, the seed, and model-proposed ablations and baselines, each replicated, with intervals and Welch tests |
| `final_evaluation` | best effort | One sealed evaluation of the chosen program and the seed, on seeds neither the search nor the ablations used |
| `posthoc` | best effort | Analyses of the run's own artifacts: trajectory, operators, lineage, costs, islands, rubrics |
| `paper` | gating | `paper.tex`, `paper.md`, a PDF when a TeX toolchain is present, and the evidence audit |
| `review` | best effort | An ensemble review with scores and a decision |
| `revision` | best effort | A revision against the review, promoted only if it still passes the audit |

A gating phase that fails stops the run. A best-effort phase that fails is recorded as degraded, and the
paper is written without its evidence.

```text
results/<task>/<run_id>/
├── run_config.yaml          the resolved configuration
├── events.jsonl             every phase's journal
├── pipeline_status.json     status and reason per phase
├── metrics/                 cost_events.jsonl, cost_summary.json
├── task/                    the run's own snapshot of the task
├── investigation/           programs.sqlite, gen_N/, best/, ideas/, memory, conclusions
├── downstream/              the frozen program and baseline every later phase studies
├── ablations/               pipeline_summary.json, aggregated/*.csv, README.md, variants/
├── final_evaluation/        invocation.json, final_manifest.json
├── posthoc/                 latest_manifest.json, analyses/
├── paper/                   paper.tex, paper.md, paper.pdf, claims, audit, latest_manifest.json
└── revision/                revision attempts
```

**Resume.** Pass `paths.run_dir=<run>` and the phases to run. The search continues with the generations
that have neither a program nor a recorded failure. Ablations and post-hoc analysis reuse finished work.
The final evaluation never runs twice: a finished one is returned as it stands, and an interrupted one
refuses to start again.

**The frozen selection.** When the search ends, Genesis copies the program to study into `downstream/`
with its digest. Every later phase reads that copy and refuses it if the bytes have changed. A winner is
a correct generated program that beats the seed. Without one, `execution.no_winner_policy=best_available`
studies the best correct candidate, or the seed, and the paper reports a negative result. `skip` stops
after the search.

**The paper.** Code generates every table and figure from the run's data. A model writes the prose, and
must list each claim with the evidence it rests on. The audit then extracts every number from the prose
and requires each to match a value in the run's evidence at the precision shown. Findings go back to the
model for repair. A paper that still fails is published with `evidence_audit_verdict: "fail"` in its
manifest.

**The budget.** `budget_usd` caps model spend across the whole run. The ledger separates charges a
provider reported, list-price estimates, and usage with no known price. A coding-agent arm counts
against the cap only when its CLI reports a charge that was really billed. The list price a
subscription agent reports about itself is recorded as an estimate.

## Models

| Name | Backend | Credential |
|---|---|---|
| `claude-*` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `gpt-*`, `o<n>*` | OpenAI chat completions | `OPENAI_API_KEY` |
| `gemini-*` | Gemini `generateContent` | `GEMINI_API_KEY` |
| `deepseek-*` | DeepSeek | `DEEPSEEK_API_KEY` |
| `openrouter/<id>` | OpenRouter | `OPENROUTER_API_KEY` |
| `azure-<name>` | Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_API_ENDPOINT` |
| `local/<model>@http://host:port/v1` | Any OpenAI-compatible server | optional |
| `cli/<agent>[@model]` | A coding-agent CLI answering as a plain model, on its own login | the CLI's login |
| `mock/perturb` | The offline mutator | none |

`genesis models <name>...` shows how each name resolves and which credential is missing. Prices come from
a bundled table. A model the table does not know is recorded as usage with no price, never as free.
`GENESIS_PRICING_FILE` replaces the table.

Only the OpenAI backend and the Claude, Codex, and Pi CLIs have been called for real. The Anthropic,
Gemini, OpenRouter, DeepSeek, and Azure backends are tested against a local server that checks the
request each one sends.

`embedding_model: local-hash` needs no key. It hashes character n-grams, which is enough for the
near-duplicate check.

## Crates

| Crate | Contents | From |
|---|---|---|
| `genesis-core` | Program record, task contract, evaluation launcher, event journal, cost ledger, configuration, statistics | both |
| `genesis-edit` | SEARCH/REPLACE and full-rewrite patches with EVOLVE-BLOCK regions, languages, code metrics | ShinkaEvolve `shinka/edit`, `complexity.py` |
| `genesis-llm` | Model backends, pricing, bandits, embeddings, coding-agent CLIs | ShinkaEvolve `shinka/llm`, `shinka/embed`; the agent side replaces the Headless CLI |
| `genesis-evolve` | Run database, islands, selection, prompts, novelty, meta scratchpad, prompt evolution, the runner | ShinkaEvolve `shinka/core`, `shinka/database`, `shinka/prompts` |
| `genesis-scientist` | Ideas, rubrics, memory, the agent operator, task drafting, the investigation phase | Agentic Scientist `src/investigation`, `src/task_drafting` |
| `genesis-evidence` | Ablations, final evaluation, post-hoc analysis | Agentic Scientist `src/variants_pipeline`, `src/final_evaluation`, `src/posthoc_pipeline` |
| `genesis-paper` | Paper data, manuscript, references, evidence audit, review, revision | Agentic Scientist `src/orchestrator`, `src/aireview.py` |
| `genesis-cli` | The `genesis` binary, the pipeline, `doctor`, the inspector | Agentic Scientist `src/pipeline`, `apps/run_inspector` |

`genesis_rust_backend/` is the API server deployed to Cloud Run. It is a separate crate, outside this
workspace.

## Differences from the originals

Changes made on purpose:

- **One loop.** The Agentic Scientist's cycle (idea, attempt, verify, remember) runs inside the
  ShinkaEvolve loop, so ideas, agents, and patches share one archive and one bandit.
- **The harness always evaluates.** The Agentic Scientist lets the agent run the evaluator and report
  rubrics unless harness mode is on. Here harness mode is the only mode.
- **Model calls replace agent sessions** for ideas, variant proposals, analysis proposals, and paper
  prose. A coding agent is used where files are edited: implementing a proposal.
- **Statistics.** The original ablation tables give means and a difference. These give sample standard
  deviations, 95% intervals, Welch's test, and a bootstrap interval, over harness-driven replications.
- **Deterministic sampling.** Every random choice in the database comes from one seeded generator, and
  candidates are ordered by insertion. The same seed and the same sequence of results give the same
  samples.
- **Best program.** A program that only matches the best score does not replace it.
- **Children count** means proposals made from a program. Island and spawn copies never count.

Not ported:

- Container isolation for agents and evaluators, the candidate broker, and task contracts. The sealed
  final evaluation keeps its once-only invocation, held-out seeds, and digest checks, without the
  sandbox.
- Slurm and E2B launchers. Evaluations run as local processes.
- Bedrock and Vertex backends, Weights & Biases logging, and the plotting scripts.
- PCA and cluster columns of the run database. The WebUI's embedding view stays empty for Rust runs.
- Concept figures and venue style files. The paper uses a self-contained `article` template.
- Parallel workers with separate memories. One loop with `max_evaluation_jobs` replaces them.

## Testing

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The suite has three layers: unit tests, end-to-end tests that run real evaluator processes against
scripted model replies, and property tests written with
[ephyra](https://github.com/datadog-labs/ephyra). Each property file opens with a table of what it
checks and where the expectation comes from.

A property test draws a fresh seed on every run. A failure prints the shrunk counterexample and a line
such as `EPHYRA_REPLAY=4109901898230880658:2424`, which reruns exactly that case. `EPHYRA_TESTS`,
`EPHYRA_SEED`, and `EPHYRA_BUDGET_MS` change the search without recompiling. When a property finds a real
bug, the fix comes with a plain regression test that names the replay line.

Live tests call real models and a real coding agent. They are ignored by default:

```bash
GENESIS_LIVE_MODEL=gpt-6-luna cargo test -p genesis-scientist --test live -- --ignored --nocapture
```
