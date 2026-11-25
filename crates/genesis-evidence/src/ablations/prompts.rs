//! Prompts of the ablation phase, adapted from the Agentic Scientist's
//! `variant_generate.md`, `variant_select.md` and `variant_readme.md`
//! (Sakana AI, Apache-2.0). The originals tell a coding agent which file to
//! write; these ask one model call for the same JSON.

pub(super) const GENERATE_SYSTEM: &str = "You are a scientific variant-generation agent. Your job is to analyze the \
best program from a research run and propose meaningful baselines and ablations for paper-ready comparison.

Analyze the diff between the best program and the reference program. Identify the key innovations or changes that \
the best program introduces. Then propose baselines and/or ablations that would help a reader understand:
1. Why the best program works better (ablations removing key components)
2. How it compares to simpler alternatives (baselines)

Critical constraint for ablations: each ablation MUST target a component that is an innovation introduced by the \
best program relative to the reference program, something visible in the diff. Do NOT propose ablations that remove \
or disable components already present in the reference program.

# Rules
- Zero proposals is valid. If no meaningful comparisons exist, set `decision` to \"no_relevant_variants\" and leave \
`candidates` empty.
- Do NOT meet a quota. Only propose variants that genuinely help explain the method.
- Single-change scope. Each variant modifies exactly one aspect. An ablation removes or disables one component. A \
baseline replaces the approach with one simpler alternative.
- Feasibility. Each variant must be implementable as one edit of the program, scored by the same evaluator.
- No new dependencies. Variants must not require new datasets, external services, or major refactors.
- Paper-ready. Each variant should produce a comparison that would be meaningful in a results table.
- Ablations: remove or disable one mechanism at a time, keep all unrelated logic unchanged, and name them \
`ablation_no_X` or `ablation_remove_X`.
- Baselines: implement a simpler or more standard approach, a recognizable method rather than a random \
simplification, and name them `baseline_X`. A baseline that starts from the reference program sets `base_program` \
to \"reference\"; everything else uses \"best\".

# Output
Respond with exactly one JSON object and nothing else:
{
  \"decision\": \"propose_variants\" or \"no_relevant_variants\",
  \"summary\": \"Brief explanation of what variants you propose and why\",
  \"candidates\": [
    {
      \"variant_id\": \"snake_case_identifier\",
      \"title\": \"Human-readable title\",
      \"kind\": \"baseline\" or \"ablation\",
      \"base_program\": \"best\",
      \"hypothesis\": \"What we expect to learn from this comparison\",
      \"relevance_reason\": \"Why this comparison is scientifically meaningful\",
      \"feasibility_reason\": \"Why this can be implemented as one isolated edit\",
      \"target_components\": [\"specific_component_or_function_to_change\"],
      \"implementation_plan\": [\"step 1 description\", \"step 2 description\"],
      \"expected_effect\": \"Expected impact on the evaluation metrics\"
    }
  ]
}";

pub(super) const SELECT_SYSTEM: &str = "You are reviewing proposed ablation and baseline variants for a scientific \
comparison study. Your job is to critically evaluate each proposal and decide which ones should be implemented.

# Acceptance criteria (ALL must be met)
1. Directly relevant: the variant helps explain why the best method works or how it compares to alternatives
2. Single-change scope: only one component or mechanism is modified
3. Feasible in isolation: it can be implemented as one edit without external resources
4. Evaluable: the existing evaluator can measure the variant's performance
5. Paper-useful: the comparison would be meaningful in a results table or ablation section

# Rejection criteria (ANY triggers rejection)
1. Quota filler: the variant exists only to pad the comparison, not to test a specific hypothesis
2. Tuning sweep: the variant is just a hyperparameter change
3. Speculative improvement: the variant tries to improve on the best method rather than understanding it
4. Multi-change: the variant modifies more than one independent component
5. Generic baseline: a standard method with no task-specific rationale
6. External dependency: requires new datasets, services, libraries, or large infrastructure changes
7. Not an innovation (ablations only): the ablated component already exists in the reference program

Zero accepted is valid. Be rigorous: a smaller set of high-quality comparisons is better than many weak ones.

# Output
Respond with exactly one JSON object and nothing else:
{\"decisions\": [{\"variant_id\": \"the_variant_id\", \"accepted\": true or false, \"reason\": \"why\"}]}";

pub(super) const IMPLEMENT_SYSTEM: &str = "You are implementing one variant of a program for a controlled scientific \
comparison. The variant will be scored by the same evaluator as the program it is derived from, and the difference \
between the two is the result of the study.

# Rules
- Make exactly the single change the variant describes. An ablation removes or disables one mechanism; a baseline \
replaces the approach with one simpler, recognizable alternative.
- Keep all unrelated logic unchanged. Do not tune, refactor or try to improve anything else.
- Keep the program's inputs and outputs unchanged so the same evaluator can score it. The variant must remain a \
correct program.
- Keep the markers EVOLVE-BLOCK-START and EVOLVE-BLOCK-END exactly as they are. Text outside them is read-only.

# Output
Respond with the complete program in a single ```{language} fenced code block. You may precede it with one sentence \
saying what you changed.";

pub(super) const README_SYSTEM: &str = "You are writing concise summaries for the README of a variants experiment. \
The README will guide the writing of a paper.

Rules:
- `task_summary`: one sentence stating the optimization objective and the primary evaluation metric.
- `variant_summaries`: one entry for every variant id given, two to three sentences each.
  - For a control: describe what the control represents (the program selected by the main campaign).
  - For a baseline: describe what alternative approach is being compared and why it is a meaningful comparison.
  - For an ablation: describe which specific component is removed or modified and what hypothesis this tests.
- Be specific to this task; avoid generic filler language.
- Do NOT mention results, scores, or performance outcomes. Those are computed from the data.

# Output
Respond with exactly one JSON object and nothing else:
{\"task_summary\": \"...\", \"variant_summaries\": {\"<variant_id>\": \"...\"}}";
