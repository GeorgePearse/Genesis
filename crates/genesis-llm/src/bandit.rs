//! Bandits that decide which model (or agent) gets the next proposal.
//!
//! A port of ShinkaEvolve's `shinka/llm/prioritization.py` in its default
//! configuration: rewards are improvements over the parent, summed in log
//! space as `exp(gain) - 1` so one large gain outweighs many small ones;
//! statistics decay after every update; selection is epsilon-greedy over UCB
//! or Thompson scores, optionally mixed with a cost term; and each arm keeps
//! only the share of its probability that its recent reliability earns, so a
//! model that is down stops drawing proposals but is still probed.

use anyhow::{bail, Context, Result};
use genesis_core::stats::percentile;
use rand::Rng;
use rand_distr::Distribution;
use serde_json::{json, Map, Value};
use std::path::Path;

/// Share of each arm's remembered failures kept when a saved state is loaded:
/// a run that stopped because a model was down usually resumes after it is fixed.
const FAILURES_KEPT_ON_LOAD: f64 = 0.5;
const STATE_FORMAT: &str = "genesis-bandit-state";
const STATE_VERSION: u64 = 1;

/// A policy over a fixed set of named arms.
pub trait Bandit: Send {
    fn arm_names(&self) -> &[String];

    /// The score of the seed program: no reward is measured from below it.
    fn set_baseline(&mut self, score: f64);

    /// Probability of each arm being chosen for the next pull.
    fn posterior(&self) -> Vec<f64>;

    /// Choose the arm for the next pull.
    fn select(&mut self, rng: &mut dyn rand::RngCore) -> usize;

    /// A proposal was started on `arm`.
    fn update_submitted(&mut self, arm: usize);

    /// A proposal from `arm` was evaluated. `reward` is its score, or `None`
    /// when the program was incorrect; `baseline` is the parent's score.
    fn update(&mut self, arm: usize, reward: Option<f64>, baseline: Option<f64>);

    /// What a delivered proposal from `arm` cost.
    fn update_cost(&mut self, arm: usize, cost: f64);

    /// Whether a proposal from `arm` produced a usable patch. A failed one
    /// reports what it was still billed; that is not a cost observation,
    /// because it says nothing about what a patch from this arm costs.
    fn report_outcome(&mut self, arm: usize, delivered: bool, failed_cost: f64);

    /// The full state as JSON. Non-finite numbers are written as the strings
    /// `"Infinity"`, `"-Infinity"` and `"NaN"`.
    fn state(&self) -> Value;

    /// Restore a state written by [`Bandit::state`]. Arrays are matched to
    /// arms by name, so the arm set may have changed in between.
    fn load_state(&mut self, state: &Value) -> Result<()>;

    /// Forget part of each arm's remembered failures; applied when a state is
    /// loaded from disk.
    fn forgive_failures(&mut self) {}

    /// `ucb`, `thompson` or `fixed`.
    fn kind(&self) -> &'static str;
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn log_add(a: f64, b: f64) -> f64 {
    let high = a.max(b);
    if high == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    high + ((a - high).exp() + (b - high).exp()).ln()
}

/// `ln(exp(z) - 1)`, the log of a gain's contribution.
fn log_expm1(z: f64) -> f64 {
    if z > 50.0 {
        z
    } else {
        z.exp_m1().ln()
    }
}

/// Shrink a log-space sum `s` as its linear value `S` would shrink under
/// `S' = f·S / (1 + (1 - f)·S)`.
///
/// For `s > 0` the textbook form `s + ln f - (s + ln((1 - f) + e^-s))` loses
/// everything to cancellation once `s` is large, so the `s` terms are
/// cancelled by hand.
fn decay_log(s: f64, factor: f64) -> f64 {
    let rest = 1.0 - factor;
    if rest <= 0.0 {
        return s;
    }
    if s > 0.0 {
        factor.ln() - (rest + (-s).exp()).ln()
    } else {
        s + factor.ln() - (rest * s.exp()).ln_1p()
    }
}

fn sample_index(probabilities: &[f64], rng: &mut dyn rand::RngCore) -> usize {
    let total: f64 = probabilities.iter().sum();
    let mut remaining = rng.gen::<f64>() * total;
    let mut last_positive = 0;
    for (index, probability) in probabilities.iter().enumerate() {
        if *probability > 0.0 {
            last_positive = index;
            if remaining < *probability {
                return index;
            }
            remaining -= probability;
        }
    }
    last_positive
}

fn encode(value: f64) -> Value {
    if value.is_nan() {
        json!("NaN")
    } else if value == f64::INFINITY {
        json!("Infinity")
    } else if value == f64::NEG_INFINITY {
        json!("-Infinity")
    } else {
        json!(value)
    }
}

fn encode_all(values: &[f64]) -> Value {
    Value::Array(values.iter().copied().map(encode).collect())
}

fn decode(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => match text.as_str() {
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            "NaN" => Some(f64::NAN),
            _ => None,
        },
        _ => None,
    }
}

fn scalar(state: &Value, key: &str) -> Option<f64> {
    state.get(key).and_then(decode)
}

/// The saved array `key`, re-laid onto `names`. Arms are matched by name when
/// the state has names, by position otherwise; an arm the state does not
/// know keeps its value from `default`.
fn aligned(state: &Value, key: &str, names: &[String], default: &[f64]) -> Result<Vec<f64>> {
    let Some(saved) = state.get(key) else {
        return Ok(default.to_vec());
    };
    let saved: Vec<f64> = saved
        .as_array()
        .with_context(|| format!("bandit state field `{key}` is not an array"))?
        .iter()
        .map(|item| {
            decode(item).with_context(|| format!("bandit state field `{key}` holds a non-number"))
        })
        .collect::<Result<_>>()?;
    let mut out = default.to_vec();
    match state.get("arm_names").and_then(Value::as_array) {
        Some(saved_names) => {
            for (index, name) in names.iter().enumerate() {
                let position = saved_names
                    .iter()
                    .position(|saved| saved.as_str() == Some(name));
                if let Some(value) = position.and_then(|p| saved.get(p)) {
                    out[index] = *value;
                }
            }
        }
        None => {
            for (slot, value) in out.iter_mut().zip(&saved) {
                *slot = *value;
            }
        }
    }
    Ok(out)
}

/// True when the state was saved for exactly this arm layout, so values that
/// summarise all arms at once (the observed range, the cost range) carry over.
fn same_layout(state: &Value, names: &[String]) -> bool {
    match state.get("arm_names").and_then(Value::as_array) {
        Some(saved) => {
            saved.len() == names.len()
                && saved.iter().zip(names).all(|(a, b)| a.as_str() == Some(b))
        }
        None => state
            .get("s")
            .and_then(Value::as_array)
            .is_none_or(|s| s.len() == names.len()),
    }
}

/// Recency-weighted record of which proposals delivered a usable patch.
#[derive(Debug, Clone)]
struct Outcomes {
    /// How many of an arm's latest outcomes its reliability is judged over.
    /// 0 switches reliability weighting off.
    failure_memory: f64,
    recent_delivered: Vec<f64>,
    recent_failed: Vec<f64>,
    /// Every failed proposal of the run, counted whatever the memory.
    n_failed: Vec<f64>,
}

impl Outcomes {
    fn new(arms: usize, failure_memory: f64) -> Result<Self> {
        if !(failure_memory == 0.0 || failure_memory >= 1.0) || !failure_memory.is_finite() {
            bail!("failure_memory must be 0 or at least 1");
        }
        Ok(Self {
            failure_memory,
            recent_delivered: vec![0.0; arms],
            recent_failed: vec![0.0; arms],
            n_failed: vec![0.0; arms],
        })
    }

    fn record(&mut self, arm: usize, delivered: bool) {
        if !delivered {
            self.n_failed[arm] += 1.0;
        }
        if self.failure_memory == 0.0 {
            return;
        }
        // Forget on the arm's own outcomes, not with time: an arm that is not
        // drawn stays as unreliable as its last proposals showed it to be.
        let keep = 1.0 - 1.0 / self.failure_memory;
        self.recent_delivered[arm] *= keep;
        self.recent_failed[arm] *= keep;
        if delivered {
            self.recent_delivered[arm] += 1.0;
        } else {
            self.recent_failed[arm] += 1.0;
        }
    }

    /// Share of each arm's recent proposals that delivered, in `[0, 1]`. A
    /// new arm is assumed to have delivered once; the assumption fades as
    /// real outcomes fill the memory.
    fn reliability(&self) -> Vec<f64> {
        if self.failure_memory == 0.0 {
            return vec![1.0; self.recent_delivered.len()];
        }
        self.recent_delivered
            .iter()
            .zip(&self.recent_failed)
            .map(|(delivered, failed)| {
                let assumed = (1.0 - (delivered + failed) / self.failure_memory).clamp(0.0, 1.0);
                let delivered = assumed + delivered;
                delivered / (delivered + failed)
            })
            .collect()
    }

    /// Let each arm keep only the reliable part of its probability.
    ///
    /// Arms are weighed against the most reliable eligible arm, so a failure
    /// rate the whole pool shares moves nothing. What an arm loses goes to
    /// the others in proportion to what they keep. Every eligible arm that
    /// has failed is also probed with a small probability, so a model that
    /// was down can show that it works again.
    fn weigh(&self, probabilities: &[f64], eligible: &[bool]) -> Vec<f64> {
        let reliability = self.reliability();
        let n_eligible = eligible.iter().filter(|e| **e).count();
        if n_eligible == 0 || reliability.iter().all(|r| *r == 1.0) {
            return probabilities.to_vec();
        }
        let best = reliability
            .iter()
            .zip(eligible)
            .filter(|(_, eligible)| **eligible)
            .map(|(r, _)| *r)
            .fold(f64::NEG_INFINITY, f64::max);
        let relative: Vec<f64> = reliability
            .iter()
            .map(|r| if best > 0.0 { (r / best).min(1.0) } else { 1.0 })
            .collect();
        let kept: Vec<f64> = probabilities
            .iter()
            .zip(&relative)
            .map(|(p, r)| p * r)
            .collect();
        let mut weighted = kept.clone();
        for arm in 0..kept.len() {
            let lost = probabilities[arm] - kept[arm];
            if lost <= 0.0 {
                continue;
            }
            let mut others = kept.clone();
            others[arm] = 0.0;
            if others.iter().sum::<f64>() <= 0.0 {
                others = (0..kept.len())
                    .map(|other| {
                        if eligible[other] && other != arm {
                            relative[other]
                        } else {
                            0.0
                        }
                    })
                    .collect();
            }
            let total: f64 = others.iter().sum();
            if total > 0.0 {
                for (slot, other) in weighted.iter_mut().zip(&others) {
                    *slot += lost * other / total;
                }
            } else {
                weighted[arm] += lost;
            }
        }
        let even_probe = 1.0 / (n_eligible as f64 * (1.0 + self.failure_memory));
        let probe: Vec<f64> = (0..kept.len())
            .map(|arm| {
                if !eligible[arm] {
                    return 0.0;
                }
                let cap = if probabilities[arm] > 0.0 {
                    probabilities[arm]
                } else {
                    f64::INFINITY
                };
                cap.min(even_probe) * (1.0 - reliability[arm])
            })
            .collect();
        let probed: f64 = probe.iter().sum();
        weighted
            .iter()
            .zip(&probe)
            .map(|(w, p)| w * (1.0 - probed) + p)
            .collect()
    }

    /// The best score takes `1 - epsilon`, the others share `epsilon`; ties
    /// for best share alike. Reliability then scales the result.
    fn epsilon_greedy(&self, scores: &[f64], epsilon: f64) -> Vec<f64> {
        let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let winners = scores.iter().filter(|score| **score == best).count();
        let arms = scores.len();
        let probabilities: Vec<f64> = if winners == arms || winners == 0 {
            vec![1.0 / arms as f64; arms]
        } else {
            scores
                .iter()
                .map(|score| {
                    if *score == best {
                        (1.0 - epsilon) / winners as f64
                    } else {
                        epsilon / (arms - winners) as f64
                    }
                })
                .collect()
        };
        self.weigh(&probabilities, &vec![true; arms])
    }

    fn forgive(&mut self) {
        for failed in &mut self.recent_failed {
            *failed *= FAILURES_KEPT_ON_LOAD;
        }
    }

    fn write(&self, state: &mut Map<String, Value>) {
        state.insert(
            "recent_delivered".into(),
            encode_all(&self.recent_delivered),
        );
        state.insert("recent_failed".into(), encode_all(&self.recent_failed));
        state.insert("n_failed".into(), encode_all(&self.n_failed));
    }

    /// Restore the counts; invalid ones are reset rather than refused, and
    /// counts saved with a longer memory are scaled down to this one.
    fn restore(&mut self, state: &Value, names: &[String]) {
        let zeros = vec![0.0; names.len()];
        let read = |key: &str| aligned(state, key, names, &zeros).ok();
        let valid = |counts: &Vec<f64>| counts.iter().all(|c| c.is_finite() && *c >= 0.0);
        let (delivered, failed, n_failed) = match (
            read("recent_delivered"),
            read("recent_failed"),
            read("n_failed"),
        ) {
            (Some(d), Some(f), Some(n)) if valid(&d) && valid(&f) && valid(&n) => (d, f, n),
            _ => (zeros.clone(), zeros.clone(), zeros),
        };
        self.n_failed = n_failed;
        let half_limit = self.failure_memory / 2.0;
        self.recent_delivered.clear();
        self.recent_failed.clear();
        for (delivered, failed) in delivered.iter().zip(&failed) {
            let half_total = delivered / 2.0 + failed / 2.0;
            let scale = if half_total > half_limit * (1.0 + 1e-9) {
                half_limit / half_total.max(0.5)
            } else {
                1.0
            };
            self.recent_delivered.push(delivered * scale);
            self.recent_failed.push(failed * scale);
        }
    }
}

/// Reward statistics shared by the UCB and Thompson bandits.
#[derive(Debug, Clone)]
struct Rewards {
    baseline: f64,
    auto_decay: Option<f64>,
    adaptive_scale: bool,
    n_submitted: Vec<f64>,
    n_completed: Vec<f64>,
    /// Log of each arm's summed reward contributions; `-inf` for none.
    s: Vec<f64>,
    /// Decayed count of each arm's evaluated proposals.
    divs: Vec<f64>,
    /// Log of the largest single contribution seen, the scale of the means.
    obs_max: f64,
}

impl Rewards {
    fn new(arms: usize, auto_decay: Option<f64>, adaptive_scale: bool) -> Result<Self> {
        if let Some(factor) = auto_decay {
            if !(factor > 0.0 && factor <= 1.0) {
                bail!("auto_decay must be in (0, 1]");
            }
        }
        Ok(Self {
            baseline: 0.0,
            auto_decay,
            adaptive_scale,
            n_submitted: vec![0.0; arms],
            n_completed: vec![0.0; arms],
            s: vec![f64::NEG_INFINITY; arms],
            divs: vec![0.0; arms],
            obs_max: f64::NEG_INFINITY,
        })
    }

    fn pulls(&self) -> Vec<f64> {
        self.n_submitted
            .iter()
            .zip(&self.n_completed)
            .map(|(a, b)| a.max(*b))
            .collect()
    }

    fn log_mean(&self, arm: usize) -> f64 {
        self.s[arm] - self.divs[arm].max(1e-7).ln()
    }

    /// Mean contribution of each arm relative to the largest one seen.
    fn normalised_means(&self) -> Vec<f64> {
        let scaled = self.adaptive_scale && self.obs_max.is_finite();
        (0..self.s.len())
            .map(|arm| {
                if scaled {
                    (self.log_mean(arm) - self.obs_max).exp()
                } else {
                    self.log_mean(arm).exp()
                }
            })
            .collect()
    }

    /// Fold one evaluated proposal in and return the log of its contribution
    /// with whether the reward was a real score.
    fn add(&mut self, arm: usize, reward: Option<f64>, parent: Option<f64>) -> (f64, bool) {
        let reward = reward.and_then(finite);
        let baseline = match parent {
            None => self.baseline,
            Some(parent) => {
                finite(parent).map_or(self.baseline, |parent| parent.max(self.baseline))
            }
        };
        let is_real = reward.is_some();
        // The difference of two finite scores can overflow; an infinite gain
        // would turn the arm's sum, and then its mean, into NaN.
        let gain = (reward.unwrap_or(f64::NEG_INFINITY) - baseline).clamp(0.0, f64::MAX);
        self.divs[arm] += 1.0;
        self.n_completed[arm] += 1.0;
        let contribution = log_expm1(gain);
        self.s[arm] = log_add(self.s[arm], contribution);
        if self.adaptive_scale && is_real && contribution > self.obs_max {
            self.obs_max = contribution;
        }
        (contribution, is_real)
    }

    fn decay(&mut self, factor: f64) {
        for divs in &mut self.divs {
            *divs *= factor;
        }
        for s in &mut self.s {
            *s = decay_log(*s, factor);
        }
        if self.adaptive_scale && self.obs_max.is_finite() {
            let largest_mean = (0..self.s.len())
                .map(|arm| self.log_mean(arm))
                .fold(f64::NEG_INFINITY, f64::max);
            self.obs_max = decay_log(self.obs_max, factor).max(largest_mean);
        }
    }

    fn write(&self, state: &mut Map<String, Value>) {
        state.insert("n_submitted".into(), encode_all(&self.n_submitted));
        state.insert("n_completed".into(), encode_all(&self.n_completed));
        state.insert("s".into(), encode_all(&self.s));
        state.insert("divs".into(), encode_all(&self.divs));
        state.insert("baseline".into(), encode(self.baseline));
        state.insert("obs_max".into(), encode(self.obs_max));
    }

    fn reset(&mut self) {
        let arms = self.s.len();
        self.n_submitted = vec![0.0; arms];
        self.n_completed = vec![0.0; arms];
        self.divs = vec![0.0; arms];
        self.s = vec![f64::NEG_INFINITY; arms];
        self.obs_max = f64::NEG_INFINITY;
    }

    /// Returns false when the saved statistics were corrupt and were reset.
    fn restore(&mut self, state: &Value, names: &[String]) -> Result<bool> {
        self.n_submitted = aligned(state, "n_submitted", names, &self.n_submitted)?;
        self.n_completed = aligned(state, "n_completed", names, &self.n_completed)?;
        self.s = aligned(state, "s", names, &self.s)?;
        self.divs = aligned(state, "divs", names, &self.divs)?;
        if let Some(baseline) = scalar(state, "baseline").and_then(finite) {
            self.baseline = baseline;
        }
        // Only -inf is a legitimate non-finite value: the sum of an unseen arm.
        let corrupt = self.s.iter().any(|s| s.is_nan() || *s == f64::INFINITY)
            || [&self.divs, &self.n_submitted, &self.n_completed]
                .iter()
                .any(|values| values.iter().any(|v| !v.is_finite() || *v < 0.0));
        if corrupt {
            self.reset();
            return Ok(false);
        }
        let saved_max =
            scalar(state, "obs_max").filter(|max| !max.is_nan() && *max != f64::INFINITY);
        self.obs_max = match saved_max {
            Some(max) if same_layout(state, names) => max,
            _ => (0..self.s.len())
                .filter(|arm| self.divs[*arm] > 0.0)
                .map(|arm| self.log_mean(arm))
                .filter(|mean| mean.is_finite())
                .fold(f64::NEG_INFINITY, f64::max),
        };
        Ok(true)
    }
}

/// Cost bookkeeping: what delivered proposals cost, and what failed ones burned.
#[derive(Debug, Clone)]
struct Costs {
    n_costs: Vec<f64>,
    total_costs: Vec<f64>,
    min_cost: f64,
    max_cost: f64,
}

impl Costs {
    fn new(arms: usize) -> Self {
        Self {
            n_costs: vec![0.0; arms],
            total_costs: vec![0.0; arms],
            min_cost: f64::INFINITY,
            max_cost: f64::NEG_INFINITY,
        }
    }

    fn valid(cost: f64) -> Option<f64> {
        (cost.is_finite() && cost >= 0.0).then_some(cost)
    }

    /// Add `cost` to the arm's total unless the sum would stop being finite.
    fn accumulate(&mut self, arm: usize, cost: f64) -> bool {
        let total = self.total_costs[arm] + cost;
        if total.is_finite() {
            self.total_costs[arm] = total;
        }
        total.is_finite()
    }

    fn observe(&mut self, arm: usize, cost: f64) {
        let Some(cost) = Self::valid(cost) else {
            return;
        };
        if !self.accumulate(arm, cost) {
            return;
        }
        self.n_costs[arm] += 1.0;
        self.max_cost = self.max_cost.max(cost);
        self.min_cost = self.min_cost.min(cost);
    }

    fn add_failed(&mut self, arm: usize, cost: f64) {
        if let Some(cost) = Self::valid(cost) {
            self.accumulate(arm, cost);
        }
    }

    fn write(&self, state: &mut Map<String, Value>) {
        state.insert("n_costs".into(), encode_all(&self.n_costs));
        state.insert("total_costs".into(), encode_all(&self.total_costs));
        state.insert("min_cost_observed".into(), encode(self.min_cost));
        state.insert("max_cost_observed".into(), encode(self.max_cost));
    }

    fn restore(&mut self, state: &Value, names: &[String]) -> Result<()> {
        let n_costs = aligned(state, "n_costs", names, &self.n_costs)?;
        let total_costs = aligned(state, "total_costs", names, &self.total_costs)?;
        if n_costs
            .iter()
            .chain(&total_costs)
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            *self = Self::new(names.len());
            return Ok(());
        }
        self.n_costs = n_costs;
        self.total_costs = total_costs;
        let saved = (
            scalar(state, "min_cost_observed"),
            scalar(state, "max_cost_observed"),
        );
        (self.min_cost, self.max_cost) = match saved {
            (Some(min), Some(max))
                if same_layout(state, names) && !min.is_nan() && !max.is_nan() =>
            {
                (min, max)
            }
            _ => {
                let means: Vec<f64> = (0..names.len())
                    .filter(|arm| self.n_costs[*arm] > 0.0)
                    .map(|arm| self.total_costs[arm] / self.n_costs[arm])
                    .collect();
                (
                    means.iter().copied().fold(f64::INFINITY, f64::min),
                    means.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            }
        };
        Ok(())
    }
}

fn check_arms(arms: &[String]) -> Result<()> {
    if arms.is_empty() {
        bail!("a bandit needs at least one arm");
    }
    for (index, name) in arms.iter().enumerate() {
        if arms[..index].contains(name) {
            bail!("bandit arm `{name}` is listed twice");
        }
    }
    Ok(())
}

fn state_object(kind: &str, names: &[String]) -> Map<String, Value> {
    let mut state = Map::new();
    state.insert("kind".into(), json!(kind));
    state.insert("arm_names".into(), json!(names));
    state
}

/// Settings of [`AsymmetricUcb`]; the defaults are ShinkaEvolve's.
#[derive(Debug, Clone, PartialEq)]
pub struct UcbOptions {
    pub exploration_coef: f64,
    pub epsilon: f64,
    /// Decay applied after every update; `None` for no decay.
    pub auto_decay: Option<f64>,
    pub adaptive_scale: bool,
    /// Weight of cheapness in the score, in `[0, 1]`. 0 ignores cost.
    pub cost_aware_coef: f64,
    pub cost_exploration_coef: f64,
    pub cost_power: f64,
    pub cost_ref_percentile: f64,
    pub failure_memory: f64,
}

impl Default for UcbOptions {
    fn default() -> Self {
        Self {
            exploration_coef: 1.0,
            epsilon: 0.2,
            auto_decay: Some(0.95),
            adaptive_scale: true,
            cost_aware_coef: 0.0,
            cost_exploration_coef: 0.1,
            cost_power: 1.0,
            cost_ref_percentile: 50.0,
            failure_memory: 10.0,
        }
    }
}

/// UCB over asymmetric rewards: only improvements over the parent count.
#[derive(Debug, Clone)]
pub struct AsymmetricUcb {
    names: Vec<String>,
    options: UcbOptions,
    rewards: Rewards,
    costs: Costs,
    outcomes: Outcomes,
}

impl AsymmetricUcb {
    pub fn new(arms: Vec<String>, options: UcbOptions) -> Result<Self> {
        check_arms(&arms)?;
        if !(0.0..=1.0).contains(&options.epsilon) {
            bail!("epsilon must be in [0, 1]");
        }
        if !(0.0..=1.0).contains(&options.cost_aware_coef) {
            bail!("cost_aware_coef must be in [0, 1]");
        }
        if options.cost_power <= 0.0 || options.cost_power.is_nan() {
            bail!("cost_power must be greater than 0");
        }
        if !(0.0..=100.0).contains(&options.cost_ref_percentile) {
            bail!("cost_ref_percentile must be in [0, 100]");
        }
        if !options.exploration_coef.is_finite() || !options.cost_exploration_coef.is_finite() {
            bail!("exploration coefficients must be finite");
        }
        Ok(Self {
            rewards: Rewards::new(arms.len(), options.auto_decay, options.adaptive_scale)?,
            costs: Costs::new(arms.len()),
            outcomes: Outcomes::new(arms.len(), options.failure_memory)?,
            names: arms,
            options,
        })
    }

    /// Share of each arm's recent proposals that delivered a patch.
    pub fn reliability(&self) -> Vec<f64> {
        self.outcomes.reliability()
    }

    /// How cheap each arm is against the pool's reference cost: above 1 for
    /// cheaper than typical. `None` until a cost has been observed.
    fn cost_ratio(&self, num: f64) -> Option<Vec<f64>> {
        let costs = &self.costs;
        if !(costs.max_cost.is_finite() && costs.min_cost.is_finite())
            || costs.max_cost < costs.min_cost
        {
            return None;
        }
        const FLOOR: f64 = 1e-7;
        let range = costs.max_cost - costs.min_cost;
        let n_cost: Vec<f64> = costs.n_costs.iter().map(|n| n.max(1.0)).collect();
        let means: Vec<f64> = costs
            .total_costs
            .iter()
            .zip(&n_cost)
            .map(|(total, n)| total / n)
            .collect();
        let observed: Vec<f64> = (0..means.len())
            .filter(|arm| costs.n_costs[*arm] > 0.0)
            .map(|arm| means[arm])
            .collect();
        let reference = if observed.is_empty() {
            FLOOR
        } else {
            percentile(&observed, self.options.cost_ref_percentile).max(FLOOR)
        };
        Some(
            (0..means.len())
                .map(|arm| {
                    let denominator = if costs.n_costs[arm] > 0.0 {
                        let bonus =
                            self.options.cost_exploration_coef * range * (num / n_cost[arm]).sqrt();
                        (means[arm] - bonus).max(FLOOR)
                    } else {
                        // An arm with no billed delivery is not treated as free.
                        means[arm].max(reference)
                    };
                    reference / denominator
                })
                .collect(),
        )
    }
}

impl Bandit for AsymmetricUcb {
    fn arm_names(&self) -> &[String] {
        &self.names
    }

    fn set_baseline(&mut self, score: f64) {
        if let Some(score) = finite(score) {
            self.rewards.baseline = score;
        }
    }

    fn posterior(&self) -> Vec<f64> {
        let pulls = self.rewards.pulls();
        let arms = pulls.len();
        let unseen: Vec<bool> = pulls.iter().map(|n| *n <= 0.0).collect();
        let n_unseen = unseen.iter().filter(|u| **u).count();
        if n_unseen > 0 {
            return unseen
                .iter()
                .map(|u| if *u { 1.0 / n_unseen as f64 } else { 0.0 })
                .collect();
        }
        let total: f64 = pulls.iter().sum();
        let num = 2.0 * total.max(2.0).ln();
        let means = self.rewards.normalised_means();
        let mut scores: Vec<f64> = (0..arms)
            .map(|arm| means[arm] + self.options.exploration_coef * (num / pulls[arm]).sqrt())
            .collect();
        let weight = self.options.cost_aware_coef;
        if weight > 0.0 {
            if let Some(ratio) = self.cost_ratio(num) {
                let largest = ratio
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(1e-9);
                for (score, ratio) in scores.iter_mut().zip(&ratio) {
                    *score = (1.0 - weight) * *score
                        + weight * (ratio / largest).powf(self.options.cost_power);
                }
            }
        }
        self.outcomes.epsilon_greedy(&scores, self.options.epsilon)
    }

    fn select(&mut self, rng: &mut dyn rand::RngCore) -> usize {
        sample_index(&self.posterior(), rng)
    }

    fn update_submitted(&mut self, arm: usize) {
        self.rewards.n_submitted[arm] += 1.0;
    }

    fn update(&mut self, arm: usize, reward: Option<f64>, baseline: Option<f64>) {
        self.rewards.add(arm, reward, baseline);
        if let Some(factor) = self.rewards.auto_decay {
            self.rewards.decay(factor);
        }
    }

    fn update_cost(&mut self, arm: usize, cost: f64) {
        self.costs.observe(arm, cost);
    }

    fn report_outcome(&mut self, arm: usize, delivered: bool, failed_cost: f64) {
        self.outcomes.record(arm, delivered);
        if !delivered {
            self.costs.add_failed(arm, failed_cost);
        }
    }

    fn state(&self) -> Value {
        let mut state = state_object(self.kind(), &self.names);
        self.rewards.write(&mut state);
        self.costs.write(&mut state);
        self.outcomes.write(&mut state);
        Value::Object(state)
    }

    fn load_state(&mut self, state: &Value) -> Result<()> {
        self.rewards.restore(state, &self.names)?;
        self.costs.restore(state, &self.names)?;
        self.outcomes.restore(state, &self.names);
        Ok(())
    }

    fn forgive_failures(&mut self) {
        self.outcomes.forgive();
    }

    fn kind(&self) -> &'static str {
        "ucb"
    }
}

/// Settings of [`ThompsonSampler`]; the defaults are ShinkaEvolve's.
#[derive(Debug, Clone, PartialEq)]
pub struct ThompsonOptions {
    pub epsilon: f64,
    pub prior_alpha: f64,
    pub prior_beta: f64,
    pub auto_decay: Option<f64>,
    pub adaptive_scale: bool,
    pub failure_memory: f64,
}

impl Default for ThompsonOptions {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            prior_alpha: 1.0,
            prior_beta: 1.0,
            auto_decay: Some(0.95),
            adaptive_scale: true,
            failure_memory: 10.0,
        }
    }
}

/// Thompson sampling over a Beta posterior per arm, fed the same asymmetric
/// rewards as [`AsymmetricUcb`].
///
/// [`Bandit::select`] draws each arm's success rate from its posterior, as
/// Thompson sampling does. [`Bandit::posterior`] cannot draw, so it reports
/// the probabilities that the posterior *means* would give: a deterministic
/// summary for logs, not the distribution `select` samples from.
#[derive(Debug, Clone)]
pub struct ThompsonSampler {
    names: Vec<String>,
    options: ThompsonOptions,
    rewards: Rewards,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    outcomes: Outcomes,
}

impl ThompsonSampler {
    pub fn new(arms: Vec<String>, options: ThompsonOptions) -> Result<Self> {
        check_arms(&arms)?;
        if !(0.0..=1.0).contains(&options.epsilon) {
            bail!("epsilon must be in [0, 1]");
        }
        if !(options.prior_alpha > 0.0 && options.prior_beta > 0.0)
            || !options.prior_alpha.is_finite()
            || !options.prior_beta.is_finite()
        {
            bail!("prior_alpha and prior_beta must be positive");
        }
        Ok(Self {
            rewards: Rewards::new(arms.len(), options.auto_decay, options.adaptive_scale)?,
            alpha: vec![options.prior_alpha; arms.len()],
            beta: vec![options.prior_beta; arms.len()],
            outcomes: Outcomes::new(arms.len(), options.failure_memory)?,
            names: arms,
            options,
        })
    }

    pub fn reliability(&self) -> Vec<f64> {
        self.outcomes.reliability()
    }

    /// Arms that must be tried before any sampling: never evaluated and
    /// never failed. `None` when every arm has been tried.
    fn untried(&self) -> Option<Vec<f64>> {
        let arms = self.names.len();
        let untried: Vec<bool> = (0..arms)
            .map(|arm| self.rewards.n_completed[arm] <= 0.0 && self.outcomes.n_failed[arm] <= 0.0)
            .collect();
        let count = untried.iter().filter(|u| **u).count();
        if count == 0 {
            return None;
        }
        if self.rewards.pulls().iter().all(|n| *n <= 0.0) {
            return Some(vec![1.0 / arms as f64; arms]);
        }
        Some(
            untried
                .iter()
                .map(|u| if *u { 1.0 / count as f64 } else { 0.0 })
                .collect(),
        )
    }

    fn probabilities(&self, theta: &[f64]) -> Vec<f64> {
        self.outcomes.epsilon_greedy(theta, self.options.epsilon)
    }
}

impl Bandit for ThompsonSampler {
    fn arm_names(&self) -> &[String] {
        &self.names
    }

    fn set_baseline(&mut self, score: f64) {
        if let Some(score) = finite(score) {
            self.rewards.baseline = score;
        }
    }

    fn posterior(&self) -> Vec<f64> {
        if let Some(untried) = self.untried() {
            return untried;
        }
        let means: Vec<f64> = self
            .alpha
            .iter()
            .zip(&self.beta)
            .map(|(a, b)| a / (a + b))
            .collect();
        self.probabilities(&means)
    }

    fn select(&mut self, rng: &mut dyn rand::RngCore) -> usize {
        if let Some(untried) = self.untried() {
            return sample_index(&untried, rng);
        }
        let theta: Vec<f64> = self
            .alpha
            .iter()
            .zip(&self.beta)
            .map(|(alpha, beta)| match rand_distr::Beta::new(*alpha, *beta) {
                Ok(distribution) => distribution.sample(rng),
                Err(_) => alpha / (alpha + beta),
            })
            .collect();
        sample_index(&self.probabilities(&theta), rng)
    }

    fn update_submitted(&mut self, arm: usize) {
        self.rewards.n_submitted[arm] += 1.0;
    }

    fn update(&mut self, arm: usize, reward: Option<f64>, baseline: Option<f64>) {
        let (contribution, is_real) = self.rewards.add(arm, reward, baseline);
        // How good this proposal was, in [0, 1], against the best one seen.
        let success = if !is_real {
            0.0
        } else if self.rewards.obs_max.is_finite() {
            (contribution - self.rewards.obs_max).exp().clamp(0.0, 1.0)
        } else {
            0.5
        };
        self.alpha[arm] += success;
        self.beta[arm] += 1.0 - success;
        if let Some(factor) = self.rewards.auto_decay {
            self.rewards.decay(factor);
            for alpha in &mut self.alpha {
                *alpha = self.options.prior_alpha + factor * (*alpha - self.options.prior_alpha);
            }
            for beta in &mut self.beta {
                *beta = self.options.prior_beta + factor * (*beta - self.options.prior_beta);
            }
        }
    }

    fn update_cost(&mut self, _arm: usize, _cost: f64) {}

    fn report_outcome(&mut self, arm: usize, delivered: bool, _failed_cost: f64) {
        self.outcomes.record(arm, delivered);
    }

    fn state(&self) -> Value {
        let mut state = state_object(self.kind(), &self.names);
        self.rewards.write(&mut state);
        state.insert("alpha".into(), encode_all(&self.alpha));
        state.insert("beta".into(), encode_all(&self.beta));
        self.outcomes.write(&mut state);
        Value::Object(state)
    }

    fn load_state(&mut self, state: &Value) -> Result<()> {
        let alpha = aligned(state, "alpha", &self.names, &self.alpha)?;
        let beta = aligned(state, "beta", &self.names, &self.beta)?;
        let usable = |values: &[f64]| values.iter().all(|v| v.is_finite() && *v > 0.0);
        if self.rewards.restore(state, &self.names)? && usable(&alpha) && usable(&beta) {
            self.alpha = alpha;
            self.beta = beta;
        } else {
            self.rewards.reset();
            self.alpha = vec![self.options.prior_alpha; self.names.len()];
            self.beta = vec![self.options.prior_beta; self.names.len()];
        }
        self.outcomes.restore(state, &self.names);
        Ok(())
    }

    fn forgive_failures(&mut self) {
        self.outcomes.forgive();
    }

    fn kind(&self) -> &'static str {
        "thompson"
    }
}

/// Fixed selection probabilities: what a comparison of models is run with.
/// It weighs by reliability only when given a `failure_memory`.
#[derive(Debug, Clone)]
pub struct FixedSampler {
    names: Vec<String>,
    baseline: f64,
    probabilities: Vec<f64>,
    n_pulls: Vec<f64>,
    costs: Costs,
    outcomes: Outcomes,
}

fn normalised(probabilities: &[f64]) -> Result<Vec<f64>> {
    if probabilities.iter().any(|p| !p.is_finite() || *p < 0.0) {
        bail!("fixed sampler probabilities must be finite and non-negative");
    }
    let total: f64 = probabilities.iter().sum();
    if !(total > 0.0 && total.is_finite()) {
        bail!("fixed sampler probabilities must sum to more than 0");
    }
    Ok(probabilities.iter().map(|p| p / total).collect())
}

impl FixedSampler {
    /// `probabilities` defaults to uniform; it is normalised to sum to 1.
    pub fn new(
        arms: Vec<String>,
        probabilities: Option<Vec<f64>>,
        failure_memory: f64,
    ) -> Result<Self> {
        check_arms(&arms)?;
        let probabilities = match probabilities {
            Some(given) if given.len() != arms.len() => {
                bail!(
                    "{} probabilities were given for {} arms",
                    given.len(),
                    arms.len()
                )
            }
            Some(given) => normalised(&given)?,
            None => vec![1.0 / arms.len() as f64; arms.len()],
        };
        Ok(Self {
            baseline: 0.0,
            probabilities,
            n_pulls: vec![0.0; arms.len()],
            costs: Costs::new(arms.len()),
            outcomes: Outcomes::new(arms.len(), failure_memory)?,
            names: arms,
        })
    }

    pub fn reliability(&self) -> Vec<f64> {
        self.outcomes.reliability()
    }
}

impl Bandit for FixedSampler {
    fn arm_names(&self) -> &[String] {
        &self.names
    }

    fn set_baseline(&mut self, score: f64) {
        if let Some(score) = finite(score) {
            self.baseline = score;
        }
    }

    fn posterior(&self) -> Vec<f64> {
        let eligible: Vec<bool> = self.probabilities.iter().map(|p| *p > 0.0).collect();
        self.outcomes.weigh(&self.probabilities, &eligible)
    }

    fn select(&mut self, rng: &mut dyn rand::RngCore) -> usize {
        sample_index(&self.posterior(), rng)
    }

    fn update_submitted(&mut self, _arm: usize) {}

    fn update(&mut self, arm: usize, _reward: Option<f64>, _baseline: Option<f64>) {
        self.n_pulls[arm] += 1.0;
    }

    fn update_cost(&mut self, arm: usize, cost: f64) {
        self.costs.observe(arm, cost);
    }

    fn report_outcome(&mut self, arm: usize, delivered: bool, failed_cost: f64) {
        self.outcomes.record(arm, delivered);
        if !delivered {
            self.costs.add_failed(arm, failed_cost);
        }
    }

    fn state(&self) -> Value {
        let mut state = state_object(self.kind(), &self.names);
        state.insert("baseline".into(), encode(self.baseline));
        state.insert("p".into(), encode_all(&self.probabilities));
        state.insert("n_pulls".into(), encode_all(&self.n_pulls));
        self.costs.write(&mut state);
        self.outcomes.write(&mut state);
        Value::Object(state)
    }

    fn load_state(&mut self, state: &Value) -> Result<()> {
        let probabilities = aligned(state, "p", &self.names, &self.probabilities)?;
        let rescaled = normalised(&probabilities).context("loaded fixed sampler state")?;
        // Probabilities that already sum to 1 are kept bit for bit: dividing
        // them again would move each by an ulp on every save and load.
        let already_normalised = (probabilities.iter().sum::<f64>() - 1.0).abs() <= 1e-9;
        self.probabilities = if already_normalised {
            probabilities
        } else {
            rescaled
        };
        if let Some(baseline) = scalar(state, "baseline").and_then(finite) {
            self.baseline = baseline;
        }
        let pulls = aligned(state, "n_pulls", &self.names, &self.n_pulls)?;
        if pulls.iter().all(|n| n.is_finite() && *n >= 0.0) {
            self.n_pulls = pulls;
        }
        self.costs.restore(state, &self.names)?;
        self.outcomes.restore(state, &self.names);
        Ok(())
    }

    fn forgive_failures(&mut self) {
        self.outcomes.forgive();
    }

    fn kind(&self) -> &'static str {
        "fixed"
    }
}

fn number(kwargs: &Map<String, Value>, key: &str, default: f64) -> Result<f64> {
    match kwargs.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .with_context(|| format!("bandit option `{key}` must be a number")),
    }
}

fn optional_number(
    kwargs: &Map<String, Value>,
    key: &str,
    default: Option<f64>,
) -> Result<Option<f64>> {
    match kwargs.get(key) {
        None => Ok(default),
        Some(Value::Null) => Ok(None),
        Some(value) => Ok(Some(value.as_f64().with_context(|| {
            format!("bandit option `{key}` must be a number or null")
        })?)),
    }
}

fn flag(kwargs: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    match kwargs.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .with_context(|| format!("bandit option `{key}` must be true or false")),
    }
}

fn reject_unknown(kwargs: &Map<String, Value>, known: &[&str], kind: &str) -> Result<()> {
    match kwargs.keys().find(|key| !known.contains(&key.as_str())) {
        Some(key) => bail!(
            "unknown option `{key}` for the {kind} bandit; it takes: {}",
            known.join(", ")
        ),
        None => Ok(()),
    }
}

const UCB_OPTIONS: &[&str] = &[
    "exploration_coef",
    "epsilon",
    "auto_decay",
    "adaptive_scale",
    "cost_aware_coef",
    "cost_exploration_coef",
    "cost_power",
    "cost_ref_percentile",
    "failure_memory",
];
const UCB_COST_OPTIONS: &[&str] = &[
    "cost_aware_coef",
    "cost_exploration_coef",
    "cost_power",
    "cost_ref_percentile",
];
const THOMPSON_OPTIONS: &[&str] = &[
    "epsilon",
    "prior_alpha",
    "prior_beta",
    "auto_decay",
    "adaptive_scale",
    "failure_memory",
];

/// Build a bandit by name.
///
/// `kind` is `None` or `"fixed"` (fixed probabilities: uniform, or
/// `kwargs.probs`), `"ucb"`/`"ucb1"` ([`AsymmetricUcb`]) or `"thompson"`
/// ([`ThompsonSampler`]). `kwargs` is an object of that bandit's options; an
/// option it does not have is an error, except that the Thompson sampler
/// ignores the UCB cost options and the fixed sampler ignores everything but
/// `probs` and `failure_memory`.
pub fn build_bandit(
    kind: Option<&str>,
    arms: Vec<String>,
    kwargs: &Value,
) -> Result<Box<dyn Bandit>> {
    let empty = Map::new();
    let kwargs = match kwargs {
        Value::Null => &empty,
        Value::Object(map) => map,
        _ => bail!("bandit options must be an object"),
    };
    match kind.map(|k| k.trim().to_ascii_lowercase()).as_deref() {
        None | Some("fixed") => {
            let probabilities = match kwargs.get("probs").or_else(|| kwargs.get("prior_probs")) {
                None | Some(Value::Null) => None,
                Some(Value::Array(items)) => Some(
                    items
                        .iter()
                        .map(|item| item.as_f64().context("`probs` must be an array of numbers"))
                        .collect::<Result<Vec<f64>>>()?,
                ),
                Some(_) => bail!("`probs` must be an array of numbers"),
            };
            Ok(Box::new(FixedSampler::new(
                arms,
                probabilities,
                number(kwargs, "failure_memory", 0.0)?,
            )?))
        }
        Some("ucb" | "ucb1") => {
            reject_unknown(kwargs, UCB_OPTIONS, "ucb")?;
            let defaults = UcbOptions::default();
            let options = UcbOptions {
                exploration_coef: number(kwargs, "exploration_coef", defaults.exploration_coef)?,
                epsilon: number(kwargs, "epsilon", defaults.epsilon)?,
                auto_decay: optional_number(kwargs, "auto_decay", defaults.auto_decay)?,
                adaptive_scale: flag(kwargs, "adaptive_scale", defaults.adaptive_scale)?,
                cost_aware_coef: number(kwargs, "cost_aware_coef", defaults.cost_aware_coef)?,
                cost_exploration_coef: number(
                    kwargs,
                    "cost_exploration_coef",
                    defaults.cost_exploration_coef,
                )?,
                cost_power: number(kwargs, "cost_power", defaults.cost_power)?,
                cost_ref_percentile: number(
                    kwargs,
                    "cost_ref_percentile",
                    defaults.cost_ref_percentile,
                )?,
                failure_memory: number(kwargs, "failure_memory", defaults.failure_memory)?,
            };
            Ok(Box::new(AsymmetricUcb::new(arms, options)?))
        }
        Some("thompson") => {
            let known: Vec<&str> = THOMPSON_OPTIONS
                .iter()
                .chain(UCB_COST_OPTIONS)
                .copied()
                .collect();
            reject_unknown(kwargs, &known, "thompson")?;
            let defaults = ThompsonOptions::default();
            let options = ThompsonOptions {
                epsilon: number(kwargs, "epsilon", defaults.epsilon)?,
                prior_alpha: number(kwargs, "prior_alpha", defaults.prior_alpha)?,
                prior_beta: number(kwargs, "prior_beta", defaults.prior_beta)?,
                auto_decay: optional_number(kwargs, "auto_decay", defaults.auto_decay)?,
                adaptive_scale: flag(kwargs, "adaptive_scale", defaults.adaptive_scale)?,
                failure_memory: number(kwargs, "failure_memory", defaults.failure_memory)?,
            };
            Ok(Box::new(ThompsonSampler::new(arms, options)?))
        }
        Some(other) => bail!("unknown bandit `{other}`; choose fixed, ucb or thompson"),
    }
}

/// Atomically write `bandit`'s state to `path` as strict JSON.
pub fn save_bandit_state(path: &Path, bandit: &dyn Bandit) -> Result<()> {
    let document = json!({
        "format": STATE_FORMAT,
        "version": STATE_VERSION,
        "kind": bandit.kind(),
        "state": bandit.state(),
    });
    genesis_core::fsutil::write_json(path, &document)
}

/// Load a state saved by [`save_bandit_state`] into `bandit`, forgiving half
/// of each arm's remembered failures. Returns `false` when `path` does not
/// exist; a file of another format or bandit kind is an error.
pub fn load_bandit_state(path: &Path, bandit: &mut dyn Bandit) -> Result<bool> {
    if !path.is_file() {
        return Ok(false);
    }
    let document: Value = genesis_core::fsutil::read_json(path)?;
    let format_ok = document.get("format").and_then(Value::as_str) == Some(STATE_FORMAT)
        && document.get("version").and_then(Value::as_u64) == Some(STATE_VERSION);
    let Some(state) = document
        .get("state")
        .filter(|state| format_ok && state.is_object())
    else {
        bail!(
            "{} is not a {STATE_FORMAT} version {STATE_VERSION} file",
            path.display()
        );
    };
    let saved_kind = document.get("kind").and_then(Value::as_str).unwrap_or("");
    if saved_kind != bandit.kind() {
        bail!(
            "{} holds the state of a `{saved_kind}` bandit and cannot be loaded into a `{}` bandit",
            path.display(),
            bandit.kind()
        );
    }
    bandit.load_state(state)?;
    bandit.forgive_failures();
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    const BASELINE: f64 = 1.0;
    const PATCH_COST: f64 = 0.02;
    const OK: usize = 0;
    const DEAD: usize = 1;

    fn arms(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    fn ucb(names: &[&str], cost_aware_coef: f64) -> AsymmetricUcb {
        AsymmetricUcb::new(
            arms(names),
            UcbOptions {
                cost_aware_coef,
                ..UcbOptions::default()
            },
        )
        .unwrap()
    }

    fn pool(kind: &str) -> Box<dyn Bandit> {
        let names = arms(&["ok", "dead"]);
        match kind {
            "ucb" => Box::new(AsymmetricUcb::new(names, UcbOptions::default()).unwrap()),
            "ucb-cost" => Box::new(ucb(&["ok", "dead"], 0.5)),
            "thompson" => {
                Box::new(ThompsonSampler::new(names, ThompsonOptions::default()).unwrap())
            }
            "fixed" => Box::new(FixedSampler::new(names, None, 10.0).unwrap()),
            other => panic!("{other}"),
        }
    }

    const KINDS: &[&str] = &["ucb", "ucb-cost", "thompson", "fixed"];

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    /// One proposal, reported the way the runner reports it.
    fn propose(
        bandit: &mut dyn Bandit,
        world: &mut rand::rngs::StdRng,
        fails: bool,
        failed_cost: f64,
    ) -> usize {
        let arm = bandit.select(world);
        bandit.update_submitted(arm);
        if arm == DEAD && fails {
            bandit.report_outcome(arm, false, failed_cost);
            return arm;
        }
        bandit.update_cost(arm, PATCH_COST * world.gen_range(0.5..1.5));
        bandit.report_outcome(arm, true, 0.0);
        let roll: f64 = world.gen();
        let reward = if roll < 0.3 {
            Some(BASELINE + world.gen_range(0.05..0.5))
        } else if roll < 0.5 {
            None
        } else {
            Some(BASELINE - world.gen_range(0.0..0.3))
        };
        bandit.update(arm, reward, Some(BASELINE));
        arm
    }

    fn draws(
        kind: &str,
        seed: u64,
        proposals: usize,
        fails: impl Fn(usize) -> bool,
        failed_cost: f64,
    ) -> Vec<usize> {
        let mut bandit = pool(kind);
        bandit.set_baseline(BASELINE);
        let mut world = rand::rngs::StdRng::seed_from_u64(1000 + seed);
        (0..proposals)
            .map(|step| propose(bandit.as_mut(), &mut world, fails(step), failed_cost))
            .collect()
    }

    #[test]
    fn a_model_that_always_fails_gets_few_of_fifty_proposals() {
        for kind in KINDS {
            for failed_cost in [0.0, PATCH_COST] {
                let dead: Vec<usize> = (0..20)
                    .map(|seed| draws(kind, seed, 50, |_| true, failed_cost).iter().sum())
                    .collect();
                let mean = dead.iter().sum::<usize>() as f64 / dead.len() as f64;
                assert!(mean < 11.0, "{kind}: mean {mean}");
                assert!(*dead.iter().max().unwrap() < 17, "{kind}: {dead:?}");
            }
        }
    }

    #[test]
    fn a_model_that_works_again_wins_its_share_back() {
        for kind in KINDS {
            let runs: Vec<Vec<usize>> = (0..20)
                .map(|seed| draws(kind, seed, 150, |step| step < 30, 0.0))
                .collect();
            let share = |from: usize, to: usize| {
                runs.iter().flat_map(|run| &run[from..to]).sum::<usize>() as f64
                    / (runs.len() * (to - from)) as f64
            };
            let (while_failing, long_after) = (share(0, 30), share(100, 150));
            assert!(while_failing < 0.3, "{kind}: {while_failing}");
            assert!(
                long_after > 1.4 * while_failing,
                "{kind}: {while_failing} then {long_after}"
            );
            if matches!(*kind, "thompson" | "fixed") {
                assert!(
                    long_after > 0.35 && long_after < 0.65,
                    "{kind}: {long_after}"
                );
            }
        }
    }

    #[test]
    fn reliability_is_the_share_of_recent_proposals_that_delivered() {
        let mut bandit = FixedSampler::new(arms(&["ok", "dead"]), None, 10.0).unwrap();
        assert_eq!(bandit.reliability(), [1.0, 1.0]);
        for step in 0..80 {
            bandit.report_outcome(DEAD, step % 4 != 0, 0.0);
        }
        let reliability = bandit.reliability();
        assert_eq!(reliability[OK], 1.0);
        assert!(close(reliability[DEAD], 0.75, 0.08), "{reliability:?}");
    }

    #[test]
    fn a_dead_model_loses_its_reliability_but_is_still_probed() {
        for kind in ["ucb", "ucb-cost"] {
            let mut bandit = ucb(&["ok", "dead"], if kind == "ucb" { 0.0 } else { 0.5 });
            bandit.set_baseline(BASELINE);
            for _ in 0..10 {
                bandit.update_submitted(OK);
                bandit.update_cost(OK, PATCH_COST);
                bandit.report_outcome(OK, true, 0.0);
                bandit.update(OK, Some(BASELINE + 0.2), Some(BASELINE));
            }
            let mut seen = Vec::new();
            for _ in 0..200 {
                bandit.update_submitted(DEAD);
                bandit.report_outcome(DEAD, false, 0.0);
                seen.push(bandit.reliability()[DEAD]);
            }
            // The delivery assumed for a new arm fades: 0.9 / 1.9, 0.81 / 2.71, ...
            assert!(close(seen[0], 0.9 / 1.9, 1e-12) && close(seen[1], 0.81 / 2.71, 1e-12));
            assert!(seen.windows(2).all(|pair| pair[1] < pair[0]));
            assert!(*seen.last().unwrap() < 1e-9);
            // All that is left is the probe share: 1 / (1 + failure_memory) of an even share.
            let posterior = bandit.posterior();
            assert!(
                close(posterior[DEAD], 1.0 / 22.0, 1e-9),
                "{kind}: {posterior:?}"
            );
            assert!(close(posterior[OK], 21.0 / 22.0, 1e-9));
        }
    }

    #[test]
    fn reliability_recovers_with_delivered_proposals() {
        let mut bandit = ucb(&["ok", "dead"], 0.0);
        for _ in 0..100 {
            bandit.report_outcome(DEAD, false, 0.0);
        }
        let recovery: Vec<f64> = (0..30)
            .map(|_| {
                bandit.report_outcome(DEAD, true, 0.0);
                bandit.reliability()[DEAD]
            })
            .collect();
        assert!(recovery[0] > 0.05 && recovery[0] < 0.15);
        assert!(recovery[9] > 0.6 && recovery[29] > 0.95);
    }

    #[test]
    fn a_fixed_sampler_is_strictly_fixed_unless_asked_to_weigh() {
        let mut strict = FixedSampler::new(arms(&["a", "b"]), Some(vec![3.0, 1.0]), 0.0).unwrap();
        for _ in 0..50 {
            strict.report_outcome(0, false, 0.0);
        }
        assert_eq!(strict.posterior(), [0.75, 0.25]);
        assert_eq!(strict.reliability(), [1.0, 1.0]);
    }

    #[test]
    fn models_that_fail_alike_keep_their_shares() {
        let mut bandit =
            FixedSampler::new(arms(&["a", "b", "c"]), Some(vec![0.8, 0.1, 0.1]), 10.0).unwrap();
        for arm in 0..3 {
            for _ in 0..3 {
                bandit.report_outcome(arm, false, 0.0);
            }
        }
        let reliability = bandit.reliability();
        assert!(reliability.iter().all(|r| close(*r, 0.729 / 3.439, 1e-12)));
        // Nothing moves between the arms; each is probed for having failed.
        let probe = (1.0 / 33.0) * (1.0 - reliability[0]);
        let posterior = bandit.posterior();
        for (got, prior) in posterior.iter().zip([0.8, 0.1, 0.1]) {
            assert!(
                close(*got, prior * (1.0 - 3.0 * probe) + probe, 1e-12),
                "{posterior:?}"
            );
        }
    }

    #[test]
    fn the_most_reliable_model_keeps_its_whole_share() {
        let mut bandit =
            FixedSampler::new(arms(&["flaky", "dead"]), Some(vec![0.5, 0.5]), 10.0).unwrap();
        for _ in 0..20 {
            bandit.report_outcome(0, true, 0.0);
        }
        bandit.report_outcome(0, false, 0.0);
        for _ in 0..100 {
            bandit.report_outcome(1, false, 0.0);
        }
        assert!(close(bandit.reliability()[0], 0.89, 0.01));
        let posterior = bandit.posterior();
        // `dead` is left with its probe share, `flaky` with a much smaller one.
        assert!(close(posterior[1], 1.0 / 22.0, 1e-3), "{posterior:?}");
        assert!(close(posterior[0], 1.0 - posterior[1], 1e-12));
    }

    #[test]
    fn a_pool_that_fails_alike_keeps_its_best_model_in_front() {
        let mut bandit = ucb(&["best", "second", "third"], 0.0);
        bandit.set_baseline(BASELINE);
        for _ in 0..20 {
            for (arm, gain) in [(0, 0.4), (1, 0.05), (2, 0.05)] {
                bandit.update_submitted(arm);
                bandit.report_outcome(arm, true, 0.0);
                bandit.update(arm, Some(BASELINE + gain), Some(BASELINE));
            }
        }
        let before = bandit.posterior();
        for arm in 0..3 {
            for _ in 0..5 {
                bandit.update_submitted(arm);
                bandit.report_outcome(arm, false, 0.0);
            }
        }
        let after = bandit.posterior();
        assert!(bandit.reliability().iter().all(|r| *r < 0.6));
        assert!(
            before[0] > 0.7 && after[0] > before[0] - 0.08,
            "{before:?} then {after:?}"
        );
    }

    #[test]
    fn an_arm_without_a_cost_report_is_not_treated_as_free() {
        let mut bandit = ucb(&["billed", "unknown"], 1.0);
        for _ in 0..3 {
            bandit.update_submitted(0);
            bandit.update_cost(0, PATCH_COST);
            bandit.update_submitted(1);
        }
        let posterior = bandit.posterior();
        assert!(posterior[1] <= posterior[0], "{posterior:?}");
    }

    #[test]
    fn the_bill_of_a_failed_proposal_is_not_a_cost_observation() {
        let mut steady = ucb(&["a", "b"], 1.0);
        let mut flaky = ucb(&["a", "b"], 1.0);
        for bandit in [&mut steady, &mut flaky] {
            for (arm, cost) in [(0, 0.02), (1, 0.04), (0, 0.02), (1, 0.04)] {
                bandit.update_submitted(arm);
                bandit.update_cost(arm, cost);
                bandit.report_outcome(arm, true, 0.0);
            }
        }
        for _ in 0..4 {
            flaky.update_submitted(1);
            flaky.report_outcome(1, false, 0.0);
        }
        assert_eq!(flaky.costs.n_costs, steady.costs.n_costs);
        assert_eq!(flaky.costs.total_costs, steady.costs.total_costs);
        flaky.report_outcome(1, false, 0.5);
        assert!(close(
            flaky.costs.total_costs[1],
            steady.costs.total_costs[1] + 0.5,
            1e-12
        ));
        assert_eq!(flaky.costs.n_costs, steady.costs.n_costs);
        assert!(flaky.cost_ratio(1.0).unwrap()[1] < steady.cost_ratio(1.0).unwrap()[1]);
    }

    #[test]
    fn unseen_arms_are_pulled_before_any_arm_is_pulled_twice() {
        let mut bandit = ucb(&["a", "b", "c"], 0.0);
        assert_eq!(bandit.posterior(), [1.0 / 3.0; 3]);
        bandit.update_submitted(1);
        assert_eq!(bandit.posterior(), [0.5, 0.0, 0.5]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        for _ in 0..20 {
            assert_ne!(bandit.select(&mut rng), 1);
        }
        let mut thompson =
            ThompsonSampler::new(arms(&["a", "b"]), ThompsonOptions::default()).unwrap();
        thompson.update_submitted(0);
        thompson.update(0, Some(2.0), Some(1.0));
        assert_eq!(thompson.posterior(), [0.0, 1.0]);
        assert_eq!(thompson.select(&mut rng), 1);
    }

    #[test]
    fn only_improvements_over_the_parent_earn_reward() {
        let mut bandit = AsymmetricUcb::new(
            arms(&["good", "flat"]),
            UcbOptions {
                auto_decay: None,
                ..UcbOptions::default()
            },
        )
        .unwrap();
        bandit.set_baseline(1.0);
        for arm in 0..2 {
            bandit.update_submitted(arm);
        }
        bandit.update(0, Some(1.5), Some(1.2));
        bandit.update(1, Some(1.1), Some(1.2));
        assert!(close(bandit.rewards.s[0], (0.3f64.exp() - 1.0).ln(), 1e-12));
        assert_eq!(bandit.rewards.s[1], f64::NEG_INFINITY);
        assert_eq!(bandit.rewards.divs, [1.0, 1.0]);
        // The seed baseline is a floor under a weaker parent.
        bandit.update(1, Some(1.3), Some(0.2));
        assert!(close(bandit.rewards.s[1], (0.3f64.exp() - 1.0).ln(), 1e-12));
        // An incorrect program and a non-finite reward add a pull and nothing else.
        let before = bandit.rewards.s.clone();
        bandit.update(0, None, Some(1.0));
        bandit.update(0, Some(f64::NAN), None);
        assert_eq!(bandit.rewards.s, before);
        assert_eq!(bandit.rewards.divs[0], 3.0);
        let posterior = bandit.posterior();
        assert!(close(posterior.iter().sum::<f64>(), 1.0, 1e-12));
    }

    #[test]
    fn decay_shrinks_sums_as_documented() {
        let mut bandit = ucb(&["a"], 0.0);
        bandit.set_baseline(0.0);
        bandit.update_submitted(0);
        bandit.update(0, Some(2.0), None);
        // S = e^2 - 1, then S' = f S / (1 + (1 - f) S) with f = 0.95.
        let s = 2.0f64.exp() - 1.0;
        let expected = 0.95 * s / (1.0 + 0.05 * s);
        assert!(close(bandit.rewards.s[0].exp(), expected, 1e-9));
        assert!(close(bandit.rewards.divs[0], 0.95, 1e-12));
        assert!(bandit.rewards.obs_max >= bandit.rewards.log_mean(0));
        assert_eq!(decay_log(f64::NEG_INFINITY, 0.95), f64::NEG_INFINITY);
        assert!(close(decay_log(3.0, 1.0), 3.0, 1e-12));
    }

    #[test]
    fn cost_awareness_prefers_the_cheaper_of_equal_arms() {
        let mut bandit = ucb(&["cheap", "dear"], 0.5);
        bandit.set_baseline(BASELINE);
        for _ in 0..10 {
            for (arm, cost) in [(0, 0.01), (1, 0.10)] {
                bandit.update_submitted(arm);
                bandit.update_cost(arm, cost);
                bandit.report_outcome(arm, true, 0.0);
                bandit.update(arm, Some(BASELINE + 0.2), Some(BASELINE));
            }
        }
        let posterior = bandit.posterior();
        assert!(
            close(posterior[0], 0.8, 1e-9) && close(posterior[1], 0.2, 1e-9),
            "{posterior:?}"
        );
        bandit.update_cost(0, f64::NAN);
        bandit.update_cost(0, -1.0);
        assert_eq!(bandit.costs.n_costs[0], 10.0);
    }

    #[test]
    fn state_round_trips_and_realigns_by_arm_name() {
        for kind in KINDS {
            let mut bandit = pool(kind);
            bandit.set_baseline(BASELINE);
            let mut world = rand::rngs::StdRng::seed_from_u64(9);
            for step in 0..40 {
                propose(bandit.as_mut(), &mut world, step % 3 == 0, 0.01);
            }
            let state = bandit.state();
            assert!(serde_json::to_string(&state).is_ok());
            let mut copy = pool(kind);
            copy.load_state(&state).unwrap();
            assert_eq!(copy.posterior(), bandit.posterior(), "{kind}");
            assert_eq!(copy.state(), state, "{kind}");
        }
        let mut source = ucb(&["a", "b"], 0.0);
        source.update_submitted(1);
        source.update(1, Some(3.0), Some(1.0));
        let mut reordered = ucb(&["b", "c", "a"], 0.0);
        reordered.load_state(&source.state()).unwrap();
        assert_eq!(reordered.rewards.n_completed, [1.0, 0.0, 0.0]);
        assert_eq!(reordered.rewards.s[1], f64::NEG_INFINITY);
        assert!(
            reordered.rewards.obs_max.is_finite(),
            "the range is recomputed for a new layout"
        );
    }

    #[test]
    fn corrupt_or_foreign_states_are_reset_or_refused() {
        let mut bandit = ucb(&["a", "b"], 0.0);
        bandit.update_submitted(0);
        bandit.update(0, Some(2.0), Some(1.0));
        let mut state = bandit.state();
        state["s"] = json!(["NaN", "-Infinity"]);
        state["recent_failed"] = json!([-3.0, 0.0]);
        let mut loaded = ucb(&["a", "b"], 0.0);
        loaded.load_state(&state).unwrap();
        assert_eq!(loaded.rewards.n_completed, [0.0, 0.0]);
        assert_eq!(loaded.posterior(), [0.5, 0.5]);
        assert!(loaded.load_state(&json!({"s": "oops"})).is_err());
        assert!(loaded.load_state(&json!({"s": [1.0, "one"]})).is_err());
    }

    #[test]
    fn states_persist_as_strict_json_and_forgive_failures_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/bandit_state.json");
        let mut bandit = ucb(&["ok", "dead"], 0.5);
        for _ in 0..6 {
            bandit.update_submitted(DEAD);
            bandit.report_outcome(DEAD, false, 0.0);
        }
        save_bandit_state(&path, &bandit).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"-Infinity\"") && !text.contains("NaN"));
        let document: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(document["format"], json!("genesis-bandit-state"));
        assert_eq!(document["version"], json!(1));

        let mut resumed = ucb(&["ok", "dead"], 0.5);
        assert!(load_bandit_state(&path, &mut resumed).unwrap());
        assert!(close(
            resumed.outcomes.recent_failed[DEAD],
            bandit.outcomes.recent_failed[DEAD] / 2.0,
            1e-12
        ));
        assert!(resumed.reliability()[DEAD] > bandit.reliability()[DEAD]);
        assert_eq!(resumed.outcomes.n_failed[DEAD], 6.0);

        assert!(!load_bandit_state(&dir.path().join("absent.json"), &mut resumed).unwrap());
        let mut other_kind =
            ThompsonSampler::new(arms(&["ok", "dead"]), ThompsonOptions::default()).unwrap();
        assert!(load_bandit_state(&path, &mut other_kind).is_err());
        std::fs::write(&path, "{\"format\": \"something-else\"}").unwrap();
        assert!(load_bandit_state(&path, &mut resumed).is_err());
    }

    /// Found by the `state_round_trips_in_memory_and_through_json_text`
    /// property (EPHYRA_REPLAY=1:11, fixed sampler with weights 3, 1, 1, 2):
    /// loading a fixed sampler's own state renormalised probabilities that
    /// already summed to 1, moving each by one ulp on every save and load.
    #[test]
    fn loading_its_own_state_does_not_move_a_fixed_samplers_probabilities() {
        let names = arms(&["a", "b", "c", "d"]);
        let bandit = FixedSampler::new(names.clone(), Some(vec![3.0, 1.0, 1.0, 2.0]), 0.0).unwrap();
        let mut loaded = FixedSampler::new(names, Some(vec![3.0, 1.0, 1.0, 2.0]), 0.0).unwrap();
        loaded.load_state(&bandit.state()).unwrap();
        assert_eq!(loaded.posterior(), bandit.posterior());
        assert_eq!(loaded.state(), bandit.state());
    }

    /// Found by the `state_round_trips_in_memory_and_through_json_text`
    /// property (EPHYRA_REPLAY=1:186): two absurd but finite bills, 1e300 and
    /// f64::MAX, overflowed an arm's total cost to infinity; a reload then
    /// discarded every cost statistic as corrupt.
    #[test]
    fn absurd_bills_cannot_overflow_an_arms_total_cost() {
        let mut bandit = ucb(&["a", "b"], 0.5);
        bandit.report_outcome(0, false, 1e300);
        bandit.report_outcome(0, false, f64::MAX);
        bandit.update_cost(1, f64::MAX);
        bandit.update_cost(1, f64::MAX);
        assert!(
            bandit.costs.total_costs.iter().all(|cost| cost.is_finite()),
            "{:?}",
            bandit.costs.total_costs
        );
        let mut loaded = ucb(&["a", "b"], 0.5);
        loaded.load_state(&bandit.state()).unwrap();
        assert_eq!(loaded.state(), bandit.state());
    }

    /// Found by the `state_round_trips_in_memory_and_through_json_text`
    /// property (EPHYRA_REPLAY=1:1320): with the baseline at -f64::MAX, a
    /// reward of f64::MAX is a gain that overflows to infinity. The arm's sum
    /// became +inf, its mean NaN, and a reload discarded the statistics.
    #[test]
    fn an_overflowing_gain_stays_finite() {
        let names = arms(&["a", "b"]);
        let mut ucb = AsymmetricUcb::new(names.clone(), UcbOptions::default()).unwrap();
        let mut thompson = ThompsonSampler::new(names.clone(), ThompsonOptions::default()).unwrap();
        let bandits: [&mut dyn Bandit; 2] = [&mut ucb, &mut thompson];
        for bandit in bandits {
            bandit.set_baseline(-f64::MAX);
            for arm in 0..2 {
                bandit.update_submitted(arm);
            }
            bandit.update(0, Some(1.0), Some(0.0));
            bandit.update(1, Some(f64::MAX), None);
            bandit.update(1, Some(f64::MAX), None);
            let posterior = bandit.posterior();
            assert!(
                posterior.iter().all(|p| p.is_finite()),
                "{}: {posterior:?}",
                bandit.kind()
            );
            assert!(close(posterior.iter().sum::<f64>(), 1.0, 1e-9));
            let state = bandit.state();
            let text = serde_json::to_string(&state).unwrap();
            assert!(!text.contains("NaN"), "{text}");
            assert_ne!(state["obs_max"], json!("Infinity"), "{text}");
            assert!(
                state["s"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|sum| sum != &json!("Infinity")),
                "{text}"
            );
        }
        let mut reloaded = AsymmetricUcb::new(names, UcbOptions::default()).unwrap();
        reloaded.load_state(&ucb.state()).unwrap();
        assert_eq!(reloaded.posterior(), ucb.posterior());
        assert_eq!(reloaded.state(), ucb.state());
    }

    #[test]
    fn build_bandit_reads_options_and_rejects_typos() {
        let names = arms(&["a", "b"]);
        assert_eq!(
            build_bandit(None, names.clone(), &Value::Null)
                .unwrap()
                .kind(),
            "fixed"
        );
        let fixed = build_bandit(Some("fixed"), names.clone(), &json!({"probs": [1, 3]})).unwrap();
        assert_eq!(fixed.posterior(), [0.25, 0.75]);
        assert_eq!(
            build_bandit(
                Some("UCB1"),
                names.clone(),
                &json!({"cost_aware_coef": 0.5})
            )
            .unwrap()
            .kind(),
            "ucb"
        );
        let thompson = build_bandit(
            Some("thompson"),
            names.clone(),
            &json!({"cost_aware_coef": 0.5, "epsilon": 0.0}),
        );
        assert_eq!(thompson.unwrap().kind(), "thompson");
        for (kind, kwargs) in [
            (Some("ucb"), json!({"exploration": 2})),
            (Some("ucb"), json!({"epsilon": 2.0})),
            (Some("ucb"), json!({"epsilon": "high"})),
            (Some("thompson"), json!({"prior_alpha": 0.0})),
            (Some("fixed"), json!({"probs": [1.0]})),
            (Some("fixed"), json!({"probs": [0.0, 0.0]})),
            (Some("greedy"), json!({})),
            (Some("ucb"), json!([1])),
        ] {
            assert!(
                build_bandit(kind, names.clone(), &kwargs).is_err(),
                "{kind:?} {kwargs}"
            );
        }
        assert!(build_bandit(Some("ucb"), vec![], &Value::Null).is_err());
        assert!(build_bandit(Some("ucb"), arms(&["a", "a"]), &Value::Null).is_err());
        assert!(build_bandit(Some("ucb"), names, &json!({"auto_decay": null})).is_ok());
    }
}
