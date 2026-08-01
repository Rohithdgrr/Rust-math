//! Statistical inference: bootstrap, effect sizes, multiple comparison correction, power analysis.

use crate::descriptive::{mean, std_dev_sample};
use crate::distributions::{normal_ppf, student_t_cdf};

/// Bootstrap confidence interval for a statistic.
/// Returns (lower, upper) bounds of the CI.
pub fn bootstrap_ci<F>(data: &[f64], stat_fn: F, n_boot: usize, alpha: f64) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let mut stats = Vec::with_capacity(n_boot);
    let n = data.len();
    for _ in 0..n_boot {
        let sample: Vec<f64> = (0..n)
            .map(|_| data[rand_index(n)])
            .collect();
        stats.push(stat_fn(&sample));
    }
    stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_idx = ((alpha / 2.0) * n_boot as f64) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_boot as f64) as usize;
    (stats[lo_idx.min(n_boot - 1)], stats[hi_idx.min(n_boot - 1)])
}

/// Simple pseudo-random index (xorshift32, no external dependency).
fn rand_index(n: usize) -> usize {
    use core::cell::Cell;
    thread_local! { static STATE: Cell<u32> = const { Cell::new(0x1234_5678) }; }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s.set(x);
        (x as usize) % n
    })
}

/// Cohen's d: standardized mean difference.
/// `pooled_sd` is the pooled standard deviation.
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let ma = mean(a);
    let mb = mean(b);
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let sa = std_dev_sample(a);
    let sb = std_dev_sample(b);
    let pooled = (((na - 1.0) * sa * sa + (nb - 1.0) * sb * sb) / (na + nb - 2.0)).sqrt();
    if pooled == 0.0 { return 0.0; }
    (ma - mb) / pooled
}

/// Hedges' g: bias-corrected Cohen's d.
pub fn hedges_g(a: &[f64], b: &[f64]) -> f64 {
    let d = cohens_d(a, b);
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0);
    d * correction
}

/// Eta-squared: proportion of variance explained (one-way ANOVA effect size).
pub fn eta_squared(groups: &[&[f64]]) -> f64 {
    let all: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let grand = mean(&all);
    let mut ssb = 0.0;
    let mut sst = 0.0;
    for g in groups {
        let mg = mean(g);
        ssb += g.len() as f64 * (mg - grand).powi(2);
        sst += g.iter().map(|x| (x - grand).powi(2)).sum::<f64>();
    }
    if sst == 0.0 { return 0.0; }
    ssb / sst
}

/// Omega-squared: less biased effect size for one-way ANOVA.
pub fn omega_squared(groups: &[&[f64]]) -> f64 {
    let all: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let grand = mean(&all);
    let k = groups.len() as f64;
    let n = all.len() as f64;
    let mut ssb = 0.0;
    let mut ssw = 0.0;
    for g in groups {
        let mg = mean(g);
        ssb += g.len() as f64 * (mg - grand).powi(2);
        ssw += g.iter().map(|x| (x - mg).powi(2)).sum::<f64>();
    }
    if ssb + ssw == 0.0 { return 0.0; }
    (ssb - (k - 1.0) * (ssw / (n - k))) / (ssb + ssw + ssw / (n - k))
}

/// Bonferroni correction: adjust significance level for `m` comparisons.
pub fn bonferroni(alpha: f64, m: usize) -> f64 {
    alpha / m as f64
}

/// Šidák correction: adjust significance level for `m` independent comparisons.
pub fn sidak(alpha: f64, m: usize) -> f64 {
    1.0 - (1.0 - alpha).powf(1.0 / m as f64)
}

/// Holm-Bonferroni (step-down) correction.
/// Returns adjusted p-values.
pub fn holm_bonferroni(pvalues: &[f64]) -> Vec<f64> {
    let m = pvalues.len();
    let mut indexed: Vec<(usize, f64)> = pvalues.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut adjusted = vec![0.0; m];
    for (rank, &(_, p)) in indexed.iter().enumerate() {
        let adj = p * (m - rank) as f64;
        adjusted[rank] = adj.min(1.0);
    }
    // Enforce monotonicity (step-up)
    for i in (1..m).rev() {
        if adjusted[i] < adjusted[i - 1] {
            adjusted[i - 1] = adjusted[i];
        }
    }
    // Put back in original order
    let mut result = vec![0.0; m];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        result[orig_idx] = adjusted[rank];
    }
    result
}

/// Benjamini-Hochberg FDR correction.
/// Returns adjusted p-values.
pub fn benjamini_hochberg(pvalues: &[f64]) -> Vec<f64> {
    let m = pvalues.len();
    let mut indexed: Vec<(usize, f64)> = pvalues.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut adjusted = vec![0.0; m];
    for (rank, &(_, p)) in indexed.iter().enumerate() {
        let adj = p * m as f64 / (rank + 1) as f64;
        adjusted[rank] = adj.min(1.0);
    }
    // Enforce monotonicity (step-down from the end)
    for i in (1..m).rev() {
        if adjusted[i] < adjusted[i - 1] {
            adjusted[i - 1] = adjusted[i];
        }
    }
    let mut result = vec![0.0; m];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        result[orig_idx] = adjusted[rank];
    }
    result
}

/// Power of a two-sample t-test.
/// `d` = Cohen's d, `n` = per-group sample size, `alpha` = significance level.
pub fn power_two_sample(d: f64, n: usize, alpha: f64) -> f64 {
    let nf = n as f64;
    let df = 2.0 * nf - 2.0;
    let ncp = d * (nf / 2.0).sqrt();
    let t_crit = normal_ppf(1.0 - alpha / 2.0);
    1.0 - student_t_cdf(t_crit - ncp, df) + student_t_cdf(-t_crit - ncp, df)
}

/// Required sample size per group for a two-sample t-test.
pub fn sample_size_two_sample(d: f64, alpha: f64, power: f64) -> usize {
    let z_alpha = normal_ppf(1.0 - alpha / 2.0);
    let z_beta = normal_ppf(power);
    let n = 2.0 * ((z_alpha + z_beta) / d).powi(2);
    n.ceil() as usize
}

/// One-sample t-test power.
pub fn power_one_sample(d: f64, n: usize, alpha: f64) -> f64 {
    let nf = n as f64;
    let df = nf - 1.0;
    let ncp = d * nf.sqrt();
    let t_crit = normal_ppf(1.0 - alpha / 2.0);
    1.0 - student_t_cdf(t_crit - ncp, df) + student_t_cdf(-t_crit - ncp, df)
}

/// Effect size from t-statistic and sample sizes (for one-sample/paired).
pub fn effect_size_from_t(t: f64, n: usize) -> f64 {
    t / (n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cohens_d_test() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let d = cohens_d(&a, &b);
        assert!(d < -2.0); // large effect
    }

    #[test]
    fn eta_squared_test() {
        let g1 = [1.0, 2.0, 3.0];
        let g2 = [4.0, 5.0, 6.0];
        let g3 = [7.0, 8.0, 9.0];
        let e = eta_squared(&[&g1, &g2, &g3]);
        assert!(e >= 0.9 - 1e-10); // large effect
    }

    #[test]
    fn bonferroni_test() {
        assert!((bonferroni(0.05, 4) - 0.0125).abs() < 1e-12);
    }

    #[test]
    fn holm_test() {
        let pvals = vec![0.01, 0.04, 0.03, 0.20];
        let adj = holm_bonferroni(&pvals);
        assert_eq!(adj.len(), 4);
        assert!(adj[0] >= pvals[0]);
    }

    #[test]
    fn bh_test() {
        let pvals = vec![0.01, 0.04, 0.03, 0.20];
        let adj = benjamini_hochberg(&pvals);
        assert_eq!(adj.len(), 4);
        assert!(adj[0] >= pvals[0]);
    }

    #[test]
    fn power_test() {
        let p = power_two_sample(0.8, 30, 0.05);
        assert!(p > 0.5); // should have decent power
    }

    #[test]
    fn sample_size_test() {
        let n = sample_size_two_sample(0.8, 0.05, 0.8);
        assert!(n >= 10 && n <= 200);
    }

    #[test]
    fn bootstrap_ci_test() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (lo, hi) = bootstrap_ci(&data, |d| mean(d), 1000, 0.05);
        assert!(lo < 3.0 && hi > 3.0);
    }
}
