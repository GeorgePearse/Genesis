//! Statistics for comparing programs over replicated evaluations.
//!
//! Everything is computed here rather than by a model, so the numbers a paper
//! cites can be traced to code.

use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    /// Sample standard deviation (n - 1); 0 for a single value.
    pub std: f64,
    pub sem: f64,
    pub ci95_low: f64,
    pub ci95_high: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let ss: f64 = values.iter().map(|v| (v - m).powi(2)).sum();
    (ss / (values.len() - 1) as f64).sqrt()
}

pub fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Linear-interpolation percentile, `q` in `[0, 100]`.
pub fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (q.clamp(0.0, 100.0) / 100.0) * (sorted.len() - 1) as f64;
    let (low, high) = (rank.floor() as usize, rank.ceil() as usize);
    sorted[low] + (sorted[high] - sorted[low]) * (rank - low as f64)
}

/// Mean with a two-sided 95% Student-t confidence interval.
pub fn summarize(values: &[f64]) -> Summary {
    let n = values.len();
    if n == 0 {
        return Summary {
            mean: f64::NAN,
            ..Summary::default()
        };
    }
    let m = mean(values);
    let std = sample_std(values);
    let sem = if n > 1 { std / (n as f64).sqrt() } else { 0.0 };
    let half = if n > 1 {
        t_critical_975((n - 1) as f64) * sem
    } else {
        0.0
    };
    Summary {
        n,
        mean: m,
        std,
        sem,
        ci95_low: m - half,
        ci95_high: m + half,
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        median: median(values),
    }
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation, g = 7, n = 9.
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEF[0];
    let t = x + 7.5;
    for (i, c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized incomplete beta function I_x(a, b), by Lentz's continued
/// fraction.
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - incomplete_beta(b, a, 1.0 - x);
    }
    const TINY: f64 = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let even = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + even * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + even / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let odd = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + odd * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + odd / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-14 {
            break;
        }
    }
    (ln_front.exp() * h / a).clamp(0.0, 1.0)
}

/// CDF of Student's t with `df` degrees of freedom.
pub fn student_t_cdf(t: f64, df: f64) -> f64 {
    if !t.is_finite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    let tail = 0.5 * incomplete_beta(df / 2.0, 0.5, df / (df + t * t));
    if t >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// The 97.5th percentile of Student's t: the multiplier of a two-sided 95%
/// interval. Found by bisection on the CDF.
pub fn t_critical_975(df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let (mut low, mut high) = (0.0_f64, 1000.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if student_t_cdf(mid, df) < 0.975 {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Comparison {
    pub n_a: usize,
    pub n_b: usize,
    pub mean_a: f64,
    pub mean_b: f64,
    /// `mean_a - mean_b`.
    pub mean_diff: f64,
    /// Welch's t statistic; `None` when either side has under two values or
    /// both are constant.
    pub t: Option<f64>,
    pub df: Option<f64>,
    /// Two-sided p-value of Welch's test.
    pub p_value: Option<f64>,
    /// Cohen's d with pooled standard deviation.
    pub cohens_d: Option<f64>,
    pub diff_ci95_low: Option<f64>,
    pub diff_ci95_high: Option<f64>,
}

/// Welch's unequal-variance t-test of `a` against `b`.
pub fn compare(a: &[f64], b: &[f64]) -> Comparison {
    let (mean_a, mean_b) = (mean(a), mean(b));
    let mut out = Comparison {
        n_a: a.len(),
        n_b: b.len(),
        mean_a,
        mean_b,
        mean_diff: mean_a - mean_b,
        ..Comparison::default()
    };
    if a.len() < 2 || b.len() < 2 {
        return out;
    }
    let (var_a, var_b) = (sample_std(a).powi(2), sample_std(b).powi(2));
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let se2 = var_a / na + var_b / nb;
    let pooled = (((na - 1.0) * var_a + (nb - 1.0) * var_b) / (na + nb - 2.0)).sqrt();
    if pooled > 0.0 {
        out.cohens_d = Some(out.mean_diff / pooled);
    }
    if se2 <= 0.0 {
        return out;
    }
    let se = se2.sqrt();
    let t = out.mean_diff / se;
    let df = se2.powi(2) / ((var_a / na).powi(2) / (na - 1.0) + (var_b / nb).powi(2) / (nb - 1.0));
    let p = 2.0 * (1.0 - student_t_cdf(t.abs(), df));
    let half = t_critical_975(df) * se;
    out.t = Some(t);
    out.df = Some(df);
    out.p_value = Some(p.clamp(0.0, 1.0));
    out.diff_ci95_low = Some(out.mean_diff - half);
    out.diff_ci95_high = Some(out.mean_diff + half);
    out
}

/// Percentile-bootstrap 95% interval for `mean(a) - mean(b)`, resampling each
/// side independently. Deterministic for a given seed.
pub fn bootstrap_diff_ci(a: &[f64], b: &[f64], resamples: usize, seed: u64) -> Option<(f64, f64)> {
    if a.is_empty() || b.is_empty() || resamples == 0 {
        return None;
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut diffs = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let draw = |values: &[f64], rng: &mut rand::rngs::StdRng| -> f64 {
            (0..values.len())
                .map(|_| values[rng.gen_range(0..values.len())])
                .sum::<f64>()
                / values.len() as f64
        };
        let ma = draw(a, &mut rng);
        let mb = draw(b, &mut rng);
        diffs.push(ma - mb);
    }
    Some((percentile(&diffs, 2.5), percentile(&diffs, 97.5)))
}

/// Spearman rank correlation, with average ranks for ties.
pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    fn ranks(values: &[f64]) -> Vec<f64> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
        let mut out = vec![0.0; values.len()];
        let mut i = 0;
        while i < order.len() {
            let mut j = i;
            while j + 1 < order.len() && values[order[j + 1]] == values[order[i]] {
                j += 1;
            }
            let rank = (i + j) as f64 / 2.0 + 1.0;
            for k in i..=j {
                out[order[k]] = rank;
            }
            i = j + 1;
        }
        out
    }
    let (rx, ry) = (ranks(x), ranks(y));
    let (mx, my) = (mean(&rx), mean(&ry));
    let cov: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
    let vx: f64 = rx.iter().map(|a| (a - mx).powi(2)).sum();
    let vy: f64 = ry.iter().map(|b| (b - my).powi(2)).sum();
    if vx <= 0.0 || vy <= 0.0 {
        return None;
    }
    Some(cov / (vx * vy).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn t_distribution_matches_reference_values() {
        // scipy.stats.t.ppf(0.975, [1, 4, 7, 9, 30])
        assert!(close(t_critical_975(1.0), 12.706_204_74, 1e-6));
        assert!(close(t_critical_975(4.0), 2.776_445_11, 1e-7));
        assert!(close(t_critical_975(7.0), 2.364_624_25, 1e-7));
        assert!(close(t_critical_975(9.0), 2.262_157_16, 1e-7));
        assert!(close(t_critical_975(30.0), 2.042_272_46, 1e-7));
        assert!(close(t_critical_975(1e6), 1.960, 1e-3));
        assert!(close(student_t_cdf(0.0, 5.0), 0.5, 1e-12));
        assert!(close(student_t_cdf(2.015, 5.0), 0.95, 1e-3));
        assert!(close(student_t_cdf(-2.015, 5.0), 0.05, 1e-3));
    }

    #[test]
    fn summary_reports_mean_and_interval() {
        let s = summarize(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert_eq!(s.n, 8);
        assert!(close(s.mean, 5.0, 1e-12));
        assert!(close(s.std, 2.138_089_935, 1e-8));
        assert!(close(s.median, 4.5, 1e-12));
        assert!(close(s.ci95_high - s.mean, 2.364_624 * s.sem, 1e-4));
        let single = summarize(&[3.0]);
        assert_eq!(
            (single.std, single.ci95_low, single.ci95_high),
            (0.0, 3.0, 3.0)
        );
        assert!(summarize(&[]).mean.is_nan());
    }

    #[test]
    fn welch_test_matches_a_worked_example() {
        // scipy.stats.ttest_ind(a, b, equal_var=False)
        //   -> statistic -2.07401463, pvalue 0.06428000, df 10.20918537
        let a = [19.8, 20.4, 19.6, 17.8, 18.5, 18.9, 18.3, 18.9, 19.5, 22.0];
        let b = [28.2, 26.6, 20.1, 23.3, 25.2, 22.1, 17.7, 27.6, 20.6, 13.7];
        let c = compare(&a, &b);
        assert!(close(c.t.unwrap(), -2.074_014_63, 1e-7), "{c:?}");
        assert!(close(c.df.unwrap(), 10.209_185_37, 1e-7), "{c:?}");
        assert!(close(c.p_value.unwrap(), 0.064_280_00, 1e-7), "{c:?}");
        assert!(c.diff_ci95_low.unwrap() < c.mean_diff && c.mean_diff < c.diff_ci95_high.unwrap());
        assert!(c.cohens_d.unwrap() < 0.0);
    }

    #[test]
    fn degenerate_comparisons_do_not_invent_a_p_value() {
        let c = compare(&[1.0], &[2.0, 3.0]);
        assert!(c.t.is_none() && c.p_value.is_none());
        let c = compare(&[2.0, 2.0], &[2.0, 2.0]);
        assert!(c.p_value.is_none());
        assert_eq!(c.mean_diff, 0.0);
    }

    #[test]
    fn bootstrap_is_deterministic_and_brackets_the_difference() {
        let a = [5.1, 5.3, 4.9, 5.2, 5.0];
        let b = [4.1, 4.3, 3.9, 4.2, 4.0];
        let first = bootstrap_diff_ci(&a, &b, 2000, 11).unwrap();
        assert_eq!(first, bootstrap_diff_ci(&a, &b, 2000, 11).unwrap());
        assert!(first.0 > 0.5 && first.1 < 1.5, "{first:?}");
        assert!(bootstrap_diff_ci(&[], &b, 10, 1).is_none());
    }

    #[test]
    fn spearman_handles_ties_and_monotone_data() {
        assert!(close(
            spearman(&[1.0, 2.0, 3.0, 4.0], &[10.0, 20.0, 30.0, 45.0]).unwrap(),
            1.0,
            1e-12
        ));
        assert!(close(
            spearman(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]).unwrap(),
            -1.0,
            1e-12
        ));
        assert!(spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
        assert!(close(percentile(&[1.0, 2.0, 3.0, 4.0], 50.0), 2.5, 1e-12));
    }
}
