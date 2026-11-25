//! Property tests for genesis-llm, run with ephyra
//! (<https://github.com/datadog-labs/ephyra>).
//!
//! What "correct" means here, and where each expectation comes from:
//!
//! | Property | Oracle | Source |
//! |---|---|---|
//! | A bandit's posterior is a finite probability vector after any sequence of operations, however wild the scores and costs | invariant | `Bandit::posterior`: "Probability of each arm being chosen" |
//! | `state` then `load_state` reproduces the posterior and the state, in memory and through JSON text | round trip | `Bandit::state` / `load_state` docs |
//! | An arm nothing has touched is never less likely than one already pulled, and can always be drawn | invariant | ShinkaEvolve `AsymmetricUCB.posterior` / `ThompsonSampler.posterior`: unseen arms first |
//! | Reporting deliveries only leaves the selection untouched | metamorphic | ShinkaEvolve `test_no_failure_leaves_the_selection_untouched` |
//! | A fixed sampler never gives probability to an arm with none, and failing never raises an arm's share above its prior | invariant | ShinkaEvolve `test_fixed_sampler_never_draws_a_model_it_gives_no_probability`, `test_failing_never_raises_the_share…` |
//! | `select` returns an arm the posterior allows | invariant | `Bandit::select` |
//! | `extract_json` finds a document in a fence or in prose exactly as a JSON parser reads it alone; a trailing comma is repaired | reference model | `extract_json` docs |
//! | `extract_tag` and `extract_fenced` return what was wrapped | round trip | their docs |
//! | Cost never falls when tokens are added | invariant | list prices are per token and tier 2 is the dearer tier |
//! | `SamplingKwargs::apply` only yields configured values | invariant | `SamplingKwargs` docs |
//! | `AgentSpec::parse(spec.arm_name()) == spec` | round trip | `AgentSpec::arm_name` docs |
//! | `resolve_model` accepts any string without panicking and never yields an empty API model for a hosted backend | crash, invariant | model names come from user configuration |
//! | `local_hash_embedding` is deterministic, unit-length or zero, and blind to the length of whitespace runs | invariant, metamorphic | `local_hash_embedding` docs |
//! | `mock/perturb` only ever rewrites lines inside the evolve block, in either patch format | invariant | the EVOLVE-BLOCK contract: everything outside the markers is read-only |

use ephyra::prelude::*;
use ephyra::{gens, BoxGen};
use genesis_llm::bandit::{
    AsymmetricUcb, Bandit, FixedSampler, ThompsonOptions, ThompsonSampler, UcbOptions,
};
use genesis_llm::pricing::bundled_catalog;
use genesis_llm::{
    cosine_similarity, extract_fenced, extract_json, extract_tag, local_hash_embedding,
    resolve_model, AgentSpec, ClientOptions, LlmClient, LlmRequest, Route, SamplingKwargs,
};
use rand::SeedableRng;
use serde_json::{json, Value};

/// Scores and costs as a careless evaluator might report them.
fn wild() -> BoxGen<f64> {
    gens::frequency(vec![
        (10, gens::f64_range(-10.0..=10.0).boxed()),
        (2, gens::f64_range(0.0..=0.05).boxed()),
        (1, gens::constant(f64::NAN).boxed()),
        (1, gens::constant(f64::INFINITY).boxed()),
        (1, gens::constant(f64::NEG_INFINITY).boxed()),
        (
            2,
            gens::pick_from(vec![
                0.0,
                1e300,
                -1e300,
                f64::MAX,
                f64::MIN,
                700.0,
                1e-300,
                -0.0,
            ])
            .boxed(),
        ),
    ])
    .boxed()
}

#[derive(Debug, Clone)]
enum Op {
    Submit(usize),
    Update(usize, Option<f64>, Option<f64>),
    Cost(usize, f64),
    Outcome(usize, bool, f64),
    Baseline(f64),
}

fn op() -> BoxGen<Op> {
    let arm = || gens::int(0usize..=4);
    gens::frequency(vec![
        (3, arm().map(Op::Submit).boxed()),
        (
            4,
            (arm(), gens::option(wild()), gens::option(wild()))
                .map(|(a, r, b)| Op::Update(a, r, b))
                .boxed(),
        ),
        (2, (arm(), wild()).map(|(a, c)| Op::Cost(a, c)).boxed()),
        (
            3,
            (arm(), gens::bool_with(0.6), wild())
                .map(|(a, d, c)| Op::Outcome(a, d, c))
                .boxed(),
        ),
        (1, wild().map(Op::Baseline).boxed()),
    ])
    .boxed()
}

#[derive(Debug, Clone)]
enum Kind {
    Ucb {
        cost_aware_coef: f64,
        epsilon: f64,
        decay: bool,
    },
    Thompson {
        epsilon: f64,
        decay: bool,
    },
    Fixed {
        weights: Vec<u8>,
        failure_memory: f64,
    },
}

fn kind() -> BoxGen<Kind> {
    let unit = || gens::pick_from(vec![0.0, 0.1, 0.2, 0.5, 1.0]);
    gens::one_of(vec![
        (unit(), unit(), gens::bool())
            .map(|(cost_aware_coef, epsilon, decay)| Kind::Ucb {
                cost_aware_coef,
                epsilon,
                decay,
            })
            .boxed(),
        (unit(), gens::bool())
            .map(|(epsilon, decay)| Kind::Thompson { epsilon, decay })
            .boxed(),
        (
            gens::vec(gens::int(0u8..=3)).len(5..=5),
            gens::pick_from(vec![0.0, 1.0, 10.0]),
        )
            .map(|(weights, failure_memory)| Kind::Fixed {
                weights,
                failure_memory,
            })
            .boxed(),
    ])
    .boxed()
}

#[derive(Debug, Clone)]
struct Scenario {
    arms: usize,
    kind: Kind,
    ops: Vec<Op>,
}

fn scenario() -> BoxGen<Scenario> {
    (gens::int(1usize..=5), kind(), gens::vec(op()).len(0..=40))
        .map(|(arms, kind, ops)| Scenario { arms, kind, ops })
        .boxed()
}

fn names(arms: usize) -> Vec<String> {
    (0..arms).map(|index| format!("arm-{index}")).collect()
}

fn fixed_prior(weights: &[u8], arms: usize) -> Vec<f64> {
    let mut prior: Vec<f64> = weights[..arms].iter().map(|w| f64::from(*w)).collect();
    if prior.iter().sum::<f64>() == 0.0 {
        prior[0] = 1.0;
    }
    prior
}

fn build(kind: &Kind, arms: usize) -> Box<dyn Bandit> {
    let decay = |on: bool| on.then_some(0.95);
    match kind {
        Kind::Ucb {
            cost_aware_coef,
            epsilon,
            decay: on,
        } => Box::new(
            AsymmetricUcb::new(
                names(arms),
                UcbOptions {
                    cost_aware_coef: *cost_aware_coef,
                    epsilon: *epsilon,
                    auto_decay: decay(*on),
                    ..UcbOptions::default()
                },
            )
            .unwrap(),
        ),
        Kind::Thompson { epsilon, decay: on } => Box::new(
            ThompsonSampler::new(
                names(arms),
                ThompsonOptions {
                    epsilon: *epsilon,
                    auto_decay: decay(*on),
                    ..ThompsonOptions::default()
                },
            )
            .unwrap(),
        ),
        Kind::Fixed {
            weights,
            failure_memory,
        } => Box::new(
            FixedSampler::new(
                names(arms),
                Some(fixed_prior(weights, arms)),
                *failure_memory,
            )
            .unwrap(),
        ),
    }
}

fn apply(bandit: &mut dyn Bandit, op: &Op, arms: usize) {
    match op {
        Op::Submit(arm) => bandit.update_submitted(arm % arms),
        Op::Update(arm, reward, baseline) => bandit.update(arm % arms, *reward, *baseline),
        Op::Cost(arm, cost) => bandit.update_cost(arm % arms, *cost),
        Op::Outcome(arm, delivered, cost) => bandit.report_outcome(arm % arms, *delivered, *cost),
        Op::Baseline(score) => bandit.set_baseline(*score),
    }
}

fn assert_distribution(posterior: &[f64], arms: usize, context: &str) {
    assert_eq!(posterior.len(), arms, "{context}");
    assert!(
        posterior.iter().all(|p| p.is_finite() && *p >= 0.0),
        "{context}: {posterior:?}"
    );
    let total: f64 = posterior.iter().sum();
    assert!(
        (total - 1.0).abs() <= 1e-9,
        "{context}: sums to {total}: {posterior:?}"
    );
}

#[ephyra::test(gen = scenario())]
fn a_posterior_is_always_a_probability_distribution(scenario: Scenario) {
    let mut bandit = build(&scenario.kind, scenario.arms);
    assert_distribution(&bandit.posterior(), scenario.arms, "fresh");
    for (step, op) in scenario.ops.iter().enumerate() {
        apply(bandit.as_mut(), op, scenario.arms);
        assert_distribution(
            &bandit.posterior(),
            scenario.arms,
            &format!("after step {step}: {op:?}"),
        );
    }
}

#[ephyra::test(gen = (scenario(), gens::int(0u64..=1_000_000)))]
fn select_returns_an_arm_the_posterior_allows(input: (Scenario, u64)) {
    let (scenario, seed) = input;
    let mut bandit = build(&scenario.kind, scenario.arms);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    for op in &scenario.ops {
        apply(bandit.as_mut(), op, scenario.arms);
        let posterior = bandit.posterior();
        let chosen = bandit.select(&mut rng);
        assert!(chosen < scenario.arms);
        // Thompson draws its own scores for each selection, so its reported
        // posterior (from the means) does not bound what it may choose.
        if !matches!(scenario.kind, Kind::Thompson { .. }) {
            assert!(
                posterior[chosen] > 0.0,
                "chose arm {chosen} with posterior {posterior:?}"
            );
        }
    }
}

#[ephyra::test(gen = scenario())]
fn state_round_trips_in_memory_and_through_json_text(scenario: Scenario) {
    let mut bandit = build(&scenario.kind, scenario.arms);
    for op in &scenario.ops {
        apply(bandit.as_mut(), op, scenario.arms);
    }
    let state = bandit.state();

    let mut in_memory = build(&scenario.kind, scenario.arms);
    in_memory.load_state(&state).unwrap();
    assert_eq!(in_memory.posterior(), bandit.posterior());
    assert_eq!(in_memory.state(), state);

    let text = serde_json::to_string(&state).expect("a state is strict JSON");
    let mut from_text = build(&scenario.kind, scenario.arms);
    from_text
        .load_state(&serde_json::from_str(&text).unwrap())
        .unwrap();
    for (restored, original) in from_text.posterior().iter().zip(bandit.posterior()) {
        assert!(
            (restored - original).abs() <= 1e-9,
            "{restored} vs {original}"
        );
    }
}

#[ephyra::test(gen = scenario())]
fn an_untouched_arm_is_never_behind_a_pulled_one(scenario: Scenario) {
    if matches!(scenario.kind, Kind::Fixed { .. }) {
        return;
    }
    let mut bandit = build(&scenario.kind, scenario.arms);
    let mut touched = vec![false; scenario.arms];
    let mut pulled = vec![false; scenario.arms];
    for op in &scenario.ops {
        apply(bandit.as_mut(), op, scenario.arms);
        match op {
            Op::Submit(arm) | Op::Update(arm, ..) => {
                touched[arm % scenario.arms] = true;
                pulled[arm % scenario.arms] = true;
            }
            Op::Cost(arm, _) | Op::Outcome(arm, ..) => touched[arm % scenario.arms] = true,
            Op::Baseline(_) => {}
        }
        let posterior = bandit.posterior();
        for fresh in (0..scenario.arms).filter(|arm| !touched[*arm]) {
            ephyra::cover(
                pulled.iter().any(|p| *p),
                "an untouched arm beside a pulled one",
            );
            assert!(
                posterior[fresh] > 0.0,
                "untouched arm {fresh} cannot be drawn: {posterior:?}"
            );
            for used in (0..scenario.arms).filter(|arm| pulled[*arm]) {
                assert!(
                    posterior[fresh] >= posterior[used],
                    "arm {fresh} is behind pulled arm {used}: {posterior:?}"
                );
            }
        }
    }
}

#[ephyra::test(gen = scenario())]
fn reporting_only_deliveries_leaves_the_selection_untouched(scenario: Scenario) {
    let mut reported = build(&scenario.kind, scenario.arms);
    let mut silent = build(&scenario.kind, scenario.arms);
    for op in scenario
        .ops
        .iter()
        .filter(|op| !matches!(op, Op::Outcome(..)))
    {
        apply(reported.as_mut(), op, scenario.arms);
        apply(silent.as_mut(), op, scenario.arms);
        if let Op::Submit(arm) | Op::Update(arm, ..) = op {
            reported.report_outcome(arm % scenario.arms, true, 0.0);
        }
        assert_eq!(reported.posterior(), silent.posterior());
    }
}

#[ephyra::test(gen = (gens::vec(gens::int(0u8..=3)).len(5..=5), gens::int(2usize..=5), gens::vec((gens::int(0usize..=4), gens::bool())).len(0..=60)))]
fn a_fixed_sampler_keeps_zero_arms_at_zero_and_failure_never_pays(
    input: (Vec<u8>, usize, Vec<(usize, bool)>),
) {
    let (weights, arms, outcomes) = input;
    let prior = fixed_prior(&weights, arms);
    let total: f64 = prior.iter().sum();
    let mut bandit = FixedSampler::new(names(arms), Some(prior.clone()), 10.0).unwrap();
    let mut others_failed = vec![false; arms];
    for (arm, delivered) in outcomes {
        let arm = arm % arms;
        bandit.report_outcome(arm, delivered, 0.0);
        if !delivered {
            for other in (0..arms).filter(|other| *other != arm) {
                others_failed[other] = true;
            }
        }
        let posterior = bandit.posterior();
        let reliability = bandit.reliability();
        for arm in 0..arms {
            assert!((0.0..=1.0).contains(&reliability[arm]));
            if prior[arm] == 0.0 {
                assert_eq!(
                    posterior[arm], 0.0,
                    "an arm with no probability was given some"
                );
            }
            // With every other arm flawless, an arm cannot gain from its own record.
            if !others_failed[arm] {
                assert!(
                    posterior[arm] <= prior[arm] / total + 1e-12,
                    "arm {arm}: {posterior:?} from prior {prior:?}"
                );
            }
        }
    }
}

fn key() -> BoxGen<String> {
    gens::string_len(gens::alphanumeric(), 1..=6).boxed()
}

fn json_leaf() -> BoxGen<Value> {
    gens::one_of(vec![
        gens::constant(Value::Null).boxed(),
        gens::bool().map(Value::Bool).boxed(),
        gens::int(-1000i64..=1000).map(|n| json!(n)).boxed(),
        gens::f64_range(-1e6..=1e6).map(|x| json!(x)).boxed(),
        gens::string_len(gens::char(), 0..=12)
            .map(Value::String)
            .boxed(),
        gens::pick_from(vec![
            json!("```"),
            json!("{[}]\","),
            json!("a\nb"),
            json!("NaN"),
        ])
        .boxed(),
    ])
    .boxed()
}

fn json_value() -> BoxGen<Value> {
    gens::recursive(3, json_leaf(), |inner| {
        gens::frequency(vec![
            (2, json_leaf()),
            (
                1,
                gens::vec(inner.clone())
                    .len(0..=4)
                    .map(Value::Array)
                    .boxed(),
            ),
            (
                1,
                gens::vec((key(), inner))
                    .len(0..=4)
                    .map(|pairs| Value::Object(pairs.into_iter().collect()))
                    .boxed(),
            ),
        ])
        .boxed()
    })
    .boxed()
}

fn json_container() -> BoxGen<Value> {
    gens::one_of(vec![
        gens::vec(json_value()).len(0..=4).map(Value::Array).boxed(),
        gens::vec((key(), json_value()))
            .len(0..=4)
            .map(|pairs| Value::Object(pairs.into_iter().collect()))
            .boxed(),
    ])
    .boxed()
}

/// Prose with no bracket or backtick, so it cannot be mistaken for a document.
fn prose() -> BoxGen<String> {
    gens::string_len(gens::char(), 0..=30)
        .map(|text| {
            text.chars()
                .filter(|c| !matches!(c, '{' | '[' | '`'))
                .collect()
        })
        .boxed()
}

#[ephyra::test(gen = (json_value(), prose(), prose(), gens::bool()))]
fn extract_json_reads_a_fenced_document_as_a_parser_would(input: (Value, String, String, bool)) {
    let (value, before, after, pretty) = input;
    let document = if pretty {
        serde_json::to_string_pretty(&value)
    } else {
        serde_json::to_string(&value)
    }
    .unwrap();
    let expected: Value = serde_json::from_str(&document).unwrap();
    let reply = format!("{before}\n```json\n{document}\n```\n{after}");
    assert_eq!(extract_json(&reply), Some(expected));
}

#[ephyra::test(gen = (json_container(), prose(), prose()))]
fn extract_json_finds_a_document_in_prose_and_repairs_a_trailing_comma(
    input: (Value, String, String),
) {
    let (value, before, after) = input;
    let document = serde_json::to_string(&value).unwrap();
    let expected: Value = serde_json::from_str(&document).unwrap();
    assert_eq!(
        extract_json(&format!("{before} {document} {after}")),
        Some(expected.clone())
    );

    let empty = value.as_array().is_some_and(Vec::is_empty)
        || value.as_object().is_some_and(|o| o.is_empty());
    ephyra::cover(!empty, "non-empty container");
    if !empty {
        let (body, close) = document.split_at(document.len() - 1);
        assert_eq!(
            extract_json(&format!("{before}{body}, {close}{after}")),
            Some(expected)
        );
    }
}

#[ephyra::test(gen = (gens::string_len(gens::char(), 0..=40), gens::string_len(gens::char(), 0..=20)))]
fn extraction_never_panics_and_only_returns_what_parses(input: (String, String)) {
    let (text, more) = input;
    for candidate in [
        text.clone(),
        format!("```json\n{text}\n```{more}"),
        format!("{{\"k\": \"{text}\"}}{more}"),
        format!("[{text}], {{{more}}}"),
    ] {
        let _ = extract_json(&candidate);
        let _ = extract_fenced(&candidate, &["json", "python"]);
        let _ = extract_tag(&candidate, "NAME");
    }
}

#[ephyra::test(gen = (gens::string_len(gens::pick_from(('A'..='Z').collect()), 1..=8), prose(), gens::string_len(gens::char(), 0..=40), prose()))]
fn extract_tag_returns_the_wrapped_text(input: (String, String, String, String)) {
    let (tag, before, body, after) = input;
    let body: String = body.chars().filter(|c| *c != '<').collect();
    let before: String = before.chars().filter(|c| *c != '<').collect();
    let reply = format!("{before}<{tag}>{body}</{tag}>{after}");
    assert_eq!(extract_tag(&reply, &tag), Some(body.trim().to_string()));
    assert_eq!(extract_tag(&reply, &format!("{tag}X")), None);
}

#[ephyra::test(gen = (gens::vec(gens::string_len(gens::char(), 0..=20)).len(1..=6), prose(), gens::pick_from(vec!["python", "rust", "cpp", "json"])))]
fn extract_fenced_returns_the_wrapped_code(input: (Vec<String>, String, &'static str)) {
    let (lines, before, language) = input;
    let code: String = lines
        .iter()
        .map(|line| {
            line.chars()
                .filter(|c| !matches!(c, '`' | '\n' | '\r'))
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let reply = format!("{before}\n```{language}\n{code}\n```\ntrailing prose");
    let expected = code.trim_start_matches(['\n', '\r']).trim_end().to_string();
    assert_eq!(extract_fenced(&reply, &[language]), Some(expected.clone()));
    assert_eq!(
        extract_fenced(&reply, &["no-such-language"]),
        Some(expected)
    );
}

#[ephyra::test(gen = (gens::int(0usize..=200), gens::int(0u64..=2_000_000), gens::int(0u64..=2_000_000), gens::int(0u64..=2_000_000)))]
fn cost_never_falls_when_tokens_are_added(input: (usize, u64, u64, u64)) {
    let (row, base, extra, output) = input;
    let catalog = bundled_catalog();
    let model = &catalog.models()[row % catalog.models().len()];
    let Some((input_cost, output_cost)) = catalog.cost(&model.model_name, base, output) else {
        assert!(model.input_price.is_none() || model.output_price.is_none());
        return;
    };
    assert!(input_cost >= 0.0 && output_cost >= 0.0 && (input_cost + output_cost).is_finite());
    let (more_input, more_output) = catalog
        .cost(&model.model_name, base + extra, output)
        .unwrap();
    assert!(
        more_input + more_output >= input_cost + output_cost,
        "{}: cheaper with more input",
        model.model_name
    );
    let (_, larger_output) = catalog
        .cost(&model.model_name, base, output + extra)
        .unwrap();
    assert!(larger_output >= output_cost);
    assert_eq!(catalog.lookup(&model.model_name), Some(model));
}

#[ephyra::test(gen = (gens::vec(gens::f64_range(0.0..=2.0)).len(0..=4), gens::int(1u32..=200_000), gens::vec(gens::pick_from(vec!["disabled", "low", "medium", "high", "max"])).len(0..=3), gens::int(0u64..=1_000_000)))]
fn sampling_only_yields_configured_values(input: (Vec<f64>, u32, Vec<&'static str>, u64)) {
    let (temperatures, max_tokens, efforts, seed) = input;
    let kwargs = SamplingKwargs {
        temperatures: temperatures.clone(),
        max_tokens,
        reasoning_efforts: efforts.iter().map(|e| e.to_string()).collect(),
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut request = LlmRequest::new("m", "s", "u")
        .temperature(99.0)
        .reasoning_effort("stale");
    kwargs.apply(&mut request, &mut rng);
    assert_eq!(request.max_tokens, Some(max_tokens));
    match request.temperature {
        Some(temperature) => assert!(temperatures
            .iter()
            .any(|t| t.to_bits() == temperature.to_bits())),
        None => assert!(temperatures.is_empty()),
    }
    match &request.reasoning_effort {
        Some(effort) => assert!(efforts.contains(&effort.as_str())),
        None => assert!(efforts.is_empty()),
    }
}

fn agent_spec() -> BoxGen<AgentSpec> {
    let name = gens::string_len(
        gens::pick_from("abcdefghijklmnopqrstuvwxyz0123456789_-".chars().collect()),
        1..=12,
    );
    let model = gens::string_len(gens::char(), 1..=16)
        .map(|text| text.chars().filter(|c| *c != '?').collect::<String>());
    let effort = gens::string_len(gens::char(), 1..=8).map(|text| {
        text.chars()
            .filter(|c| !matches!(c, '?' | '&' | '='))
            .collect::<String>()
    });
    (name, gens::option(model), gens::option(effort))
        .map(|(name, model, effort)| AgentSpec {
            name,
            model: model.filter(|m| !m.is_empty()),
            reasoning_effort: effort.filter(|e| !e.is_empty()),
            ..AgentSpec::default()
        })
        .boxed()
}

#[ephyra::test(gen = agent_spec())]
fn an_agent_spec_survives_its_arm_name(spec: AgentSpec) {
    let arm = spec.arm_name();
    assert_eq!(AgentSpec::parse(&arm), Some(spec.clone()), "{arm:?}");
    // The same reference names a completion backend; a model name as a whole
    // is trimmed, so only the trimmed reference can be expected back.
    let reference = arm.strip_prefix("agent:").unwrap();
    let trimmed = AgentSpec::parse(&format!("agent:{}", reference.trim_end()));
    assert_eq!(
        resolve_model(&format!("cli/{reference}")).ok(),
        trimmed.map(|spec| Route::Cli { spec })
    );
}

#[ephyra::test(gen = (gens::pick_from(vec!["", "mock/", "cli/", "headless/", "openrouter/", "local/", "azure-", "anthropic/", "openai/", "google/", "claude-", "gpt-", "o", "gemini-", "deepseek-", "agent:"]), gens::string_len(gens::char(), 0..=24)))]
fn any_model_name_resolves_or_is_refused_without_panicking(input: (&'static str, String)) {
    let (prefix, rest) = input;
    let name = format!("{prefix}{rest}");
    match resolve_model(&name) {
        Ok(route) => {
            ephyra::cover(true, "resolved");
            assert!(!route.provider().is_empty());
            if !matches!(route, Route::Cli { .. }) {
                assert!(
                    !route.api_model().is_empty(),
                    "{name:?} resolved to an empty API model: {route:?}"
                );
            }
        }
        Err(error) => assert!(error.is_endpoint() && !error.to_string().is_empty()),
    }
    let _ = AgentSpec::parse(&name);
}

#[ephyra::test(gen = (gens::string_len(gens::char(), 0..=80), gens::int(1usize..=300)))]
fn the_offline_embedding_is_stable_and_whitespace_blind(input: (String, usize)) {
    let (text, dim) = input;
    let vector = local_hash_embedding(&text, dim);
    assert_eq!(vector.len(), dim);
    assert!(vector.iter().all(|x| x.is_finite()));
    assert_eq!(vector, local_hash_embedding(&text, dim));
    let norm: f64 = vector
        .iter()
        .map(|x| f64::from(*x).powi(2))
        .sum::<f64>()
        .sqrt();
    let is_zero = vector.iter().all(|x| *x == 0.0);
    ephyra::cover(!is_zero, "non-zero embedding");
    assert!(is_zero || (norm - 1.0).abs() <= 1e-4, "norm {norm}");
    if !is_zero {
        assert!(cosine_similarity(&vector, &vector) > 0.999);
    }
    let stretched: String = text
        .chars()
        .flat_map(|c| {
            if c.is_whitespace() {
                vec![' ', '\t', '\n']
            } else {
                vec![c]
            }
        })
        .collect();
    assert_eq!(
        local_hash_embedding(&stretched, dim),
        vector,
        "whitespace runs changed the embedding"
    );
}

fn code_line() -> BoxGen<String> {
    let token = gens::frequency(vec![
        (
            3,
            gens::pick_from(vec![
                "x", "radius", "let", "=", "+", "*", "(", ")", ";", "fn", "0..n", "v.0", "a[1]",
                "é", "x1", "0x1F",
            ])
            .map(str::to_string)
            .boxed(),
        ),
        (2, gens::int(0u32..=500).map(|n| n.to_string()).boxed()),
        (
            2,
            gens::f64_range(0.0..=50.0)
                .map(|x| format!("{x:.3}"))
                .boxed(),
        ),
        (
            1,
            gens::pick_from(vec![
                "1e-3", "2.5e2", "10usize", "3f32", "1.max(2)", "# 0.5", "// 7",
            ])
            .map(str::to_string)
            .boxed(),
        ),
    ]);
    gens::vec(token)
        .len(0..=6)
        .map(|tokens| tokens.join(" "))
        .boxed()
}

#[ephyra::test(gen = (gens::vec(code_line()).len(0..=4), gens::vec(code_line()).len(0..=6), gens::vec(code_line()).len(0..=4), gens::bool(), gens::pick_from(vec![0.0, 0.5, 1.0])), tests = 2000)]
fn the_mock_mutator_only_rewrites_the_evolve_block(
    input: (Vec<String>, Vec<String>, Vec<String>, bool, f64),
) {
    let (head, block, tail, as_diff, temperature) = input;
    let mut lines = head.clone();
    lines.push("# EVOLVE-BLOCK-START".to_string());
    lines.extend(block.iter().cloned());
    lines.push("# EVOLVE-BLOCK-END".to_string());
    lines.extend(tail.iter().cloned());
    let program = lines.join("\n");
    let system = if as_diff {
        "Reply with <<<<<<< SEARCH blocks in <DIFF>."
    } else {
        "Reply with <NAME>, <DESCRIPTION>, <CODE>."
    };
    let user = format!("# Current program\n\n```python\n{program}\n```\n\nImprove it.");
    let request = LlmRequest::new("mock/perturb", system, user).temperature(temperature);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let client = LlmClient::new(ClientOptions::default());
    let reply = runtime
        .block_on(client.query("proposal", request))
        .unwrap()
        .content;

    let mutable = head.len() + 1..head.len() + 1 + block.len();
    if as_diff {
        let diff = extract_tag(&reply, "DIFF").expect("a diff reply has a <DIFF> section");
        for chunk in diff.split("<<<<<<< SEARCH\n").skip(1) {
            let search = chunk.split("\n=======\n").next().unwrap();
            let replace = chunk
                .split("\n=======\n")
                .nth(1)
                .unwrap()
                .split("\n>>>>>>> REPLACE")
                .next()
                .unwrap();
            assert!(
                mutable.clone().any(|index| lines[index] == search),
                "SEARCH {search:?} is not a line of the evolve block"
            );
            assert_ne!(search, replace);
        }
    } else {
        let code = extract_fenced(
            &extract_tag(&reply, "CODE").expect("a full reply has a <CODE> section"),
            &["python"],
        )
        .unwrap();
        let rewritten: Vec<&str> = code.split('\n').collect();
        let expected = program.trim_start_matches(['\n', '\r']).trim_end();
        let original: Vec<&str> = expected.split('\n').collect();
        assert_eq!(
            rewritten.len(),
            original.len(),
            "the rewrite changed the number of lines"
        );
        let offset = lines.iter().take_while(|line| line.is_empty()).count();
        for (index, (new, old)) in rewritten.iter().zip(&original).enumerate() {
            if new != old {
                assert!(
                    mutable.contains(&(index + offset)),
                    "line {} changed outside the evolve block: {old:?} -> {new:?}",
                    index + offset
                );
            }
        }
    }
}
