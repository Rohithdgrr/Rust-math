//! MCMC convergence diagnostics: effective sample size, Gelman-Rubin R-hat,
//! Geweke z-scores, autocorrelation, integrated autocorrelation time, and
//! thinning recommendations.
//!
//! All diagnostics operate on post-burn-in chains. Functions taking a single
//! `&[f64]` treat the slice as one scalar chain; functions taking `&[Vec<f64>]`
//! treat it as a matrix of draws (one row per iteration) and return one value
//! per parameter dimension.

/// Sample autocorrelation of a chain at a given lag.
///
/// # Panics
/// Panics if `samples` has fewer than 2 elements.
#[must_use]
pub fn autocorrelation(samples: &[f64], lag: usize) -> f64 {
    assert!(samples.len() >= 2, "autocorrelation needs at least 2 samples");
    let n = samples.len();
    if lag >= n {
        return 0.0;
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var: f64 = samples.iter().map(|&x| (x - mean) * (x - mean)).sum();
    if var == 0.0 {
        return 0.0;
    }
    let cov: f64 = samples[..n - lag]
        .iter()
        .zip(&samples[lag..])
        .map(|(&x, &y)| (x - mean) * (y - mean))
        .sum();
    cov / var
}

/// Autocorrelation function for lags `0..=max_lag`.
#[must_use]
pub fn autocorrelation_function(samples: &[f64], max_lag: usize) -> Vec<f64> {
    let max_lag = max_lag.min(samples.len().saturating_sub(2));
    (0..=max_lag).map(|lag| autocorrelation(samples, lag)).collect()
}

/// Integrated autocorrelation time: `1 + 2 * sum_{k>=1} rho_k`, using Geyer's
/// initial positive sequence estimator (truncates before the second negative
/// correlation) so the estimate is always positive.
#[must_use]
pub fn integrated_autocorrelation_time(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 4 {
        return 1.0;
    }
    let max_lag = (n - 1).min(1000);
    let mut sum = 0.0;
    let mut negatives_seen = 0;
    for lag in 1..=max_lag {
        let rho = autocorrelation(samples, lag);
        if rho < 0.0 {
            negatives_seen += 1;
            if negatives_seen >= 2 {
                break;
            }
        } else {
            negatives_seen = 0;
        }
        sum += rho;
    }
    (1.0 + 2.0 * sum.max(0.0)).max(1.0)
}

/// Effective sample size: `n / IACT`. Values near `n` indicate an
/// approximately independent chain; values much smaller than `n` indicate
/// strong autocorrelation (thinning may help).
#[must_use]
pub fn effective_sample_size(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 4 {
        return n as f64;
    }
    n as f64 / integrated_autocorrelation_time(samples)
}

/// Per-dimension effective sample size for a matrix of draws.
#[must_use]
pub fn effective_sample_size_multi(samples: &[Vec<f64>]) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let dim = samples[0].len();
    (0..dim)
        .map(|d| {
            let col: Vec<f64> = samples.iter().map(|s| s[d]).collect();
            effective_sample_size(&col)
        })
        .collect()
}

/// Gelman-Rubin potential scale reduction factor `R-hat` for `m` chains of
/// `n` draws each (per parameter dimension). Each chain is a matrix of draws,
/// one row per iteration. `R-hat <= 1.1` is the usual convergence heuristic;
/// `1.0` means the between- and within-chain variances agree.
///
/// # Panics
/// Panics if `chains` is empty, chains have unequal lengths, any chain has
/// fewer than 2 draws, or rows have inconsistent dimension.
#[must_use]
pub fn gelman_rubin(chains: &[Vec<Vec<f64>>]) -> Vec<f64> {
    assert!(!chains.is_empty(), "gelman_rubin needs at least one chain");
    let m = chains.len() as f64;
    let n = chains[0].len();
    assert!(n >= 2, "gelman_rubin needs at least 2 draws per chain");
    assert!(
        chains.iter().all(|c| c.len() == n),
        "all chains must have the same number of draws"
    );
    let dim = chains[0][0].len();
    assert!(
        chains.iter().all(|c| c.iter().all(|row| row.len() == dim)),
        "all draws must have the same dimension"
    );

    let chain_means: Vec<Vec<f64>> = chains
        .iter()
        .map(|chain| {
            (0..dim)
                .map(|d| chain.iter().map(|s| s[d]).sum::<f64>() / n as f64)
                .collect()
        })
        .collect();

    let overall_mean: Vec<f64> = (0..dim)
        .map(|d| chain_means.iter().map(|cm| cm[d]).sum::<f64>() / m)
        .collect();

    let mut rhat = vec![0.0; dim];
    for d in 0..dim {
        let mut b = 0.0;
        let mut w = 0.0;
        for (j, cm) in chain_means.iter().enumerate() {
            b += (cm[d] - overall_mean[d]).powi(2);
            let within: f64 = chains[j]
                .iter()
                .map(|s| (s[d] - cm[d]).powi(2))
                .sum::<f64>()
                / (n as f64 - 1.0);
            w += within;
        }
        b *= n as f64 / (m - 1.0);
        w /= m;
        if w == 0.0 {
            rhat[d] = f64::NAN;
            continue;
        }
        let var_hat = (n as f64 - 1.0) / n as f64 * w + b / n as f64;
        rhat[d] = (var_hat / w).sqrt();
    }
    rhat
}

/// Geweke z-score comparing the first `fraction_a` of the chain against the
/// last `fraction_b`. The variances are estimated with batch means, so the
/// statistic is roughly standard-normal under convergence; `|z| > 2` suggests
/// the chain has not yet converged or the windows are too small.
///
/// # Panics
/// Panics if `fraction_a + fraction_b > 1` or the windows contain fewer than
/// 2 draws each.
#[must_use]
pub fn geweke(samples: &[f64], fraction_a: f64, fraction_b: f64) -> f64 {
    assert!(
        fraction_a + fraction_b <= 1.0,
        "geweke windows must not overlap"
    );
    let n = samples.len();
    let na = ((fraction_a * n as f64) as usize).max(2);
    let nb = ((fraction_b * n as f64) as usize).max(2);
    assert!(na + nb <= n, "geweke windows exceed chain length");

    let a = &samples[..na];
    let b = &samples[n - nb..];

    let mean_a = a.iter().sum::<f64>() / na as f64;
    let mean_b = b.iter().sum::<f64>() / nb as f64;

    // Batch-means variance estimator (spectral density at zero, approximately).
    let var_a = batch_variance(a, mean_a);
    let var_b = batch_variance(b, mean_b);

    let denom = (var_a / na as f64 + var_b / nb as f64).sqrt();
    if denom == 0.0 {
        return f64::NAN;
    }
    (mean_a - mean_b) / denom
}

/// Batch-means variance estimator: split the sample into ~16 batches, compute
/// the variance of batch means, and rescale by batch size.
fn batch_variance(samples: &[f64], overall_mean: f64) -> f64 {
    let n = samples.len();
    const BATCHES: usize = 16;
    if n < 2 * BATCHES {
        return samples
            .iter()
            .map(|&x| (x - overall_mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
    }
    let batch_size = n / BATCHES;
    let batch_means: Vec<f64> = (0..BATCHES)
        .map(|b| {
            samples[b * batch_size..(b + 1) * batch_size].iter().sum::<f64>()
                / batch_size as f64
        })
        .collect();
    let mean_of_means = batch_means.iter().sum::<f64>() / BATCHES as f64;
    let var_of_means = batch_means
        .iter()
        .map(|&x| (x - mean_of_means).powi(2))
        .sum::<f64>()
        / (BATCHES - 1) as f64;
    var_of_means * batch_size as f64
}

/// Suggested thinning interval: the smallest lag at which the autocorrelation
/// drops below `threshold` (default 0.1), or `None` if it never does within
/// the observed lags.
#[must_use]
pub fn thinning_interval(samples: &[f64], threshold: f64) -> Option<usize> {
    let n = samples.len();
    let max_lag = (n - 1).min(1000);
    for lag in 1..=max_lag {
        if autocorrelation(samples, lag).abs() < threshold {
            return Some(lag);
        }
    }
    None
}

/// Descriptive summary of one scalar chain.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainSummary {
    /// Number of draws.
    pub n: usize,
    /// Sample mean.
    pub mean: f64,
    /// Sample standard deviation.
    pub std_dev: f64,
    /// Effective sample size.
    pub ess: f64,
    /// Integrated autocorrelation time.
    pub iact: f64,
    /// Autocorrelation at lag 1.
    pub rho1: f64,
}

/// Summary statistics for a scalar chain.
#[must_use]
pub fn summarize(samples: &[f64]) -> ChainSummary {
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        samples
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        0.0
    };
    ChainSummary {
        n,
        mean,
        std_dev: var.sqrt(),
        ess: effective_sample_size(samples),
        iact: integrated_autocorrelation_time(samples),
        rho1: if n >= 2 { autocorrelation(samples, 1) } else { 0.0 },
    }
}

/// Hailperin-consistency helper for CI text in tests.
#[doc(hidden)]
pub fn _assert_ci95(samples: &[f64], true_mean: f64) {
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    assert!(
        (mean - true_mean).abs() < 4.0 * se,
        "95% CI should contain the true mean: mean {mean}, true {true_mean}, se {se}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn independent_normal_chain(n: usize, mu: f64) -> Vec<f64> {
        let mut rng = Rng::new(7);
        let dist = crate::distributions::Normal { mu, sigma: 1.0 };
        (0..n).map(|_| dist.sample(&mut rng)).collect()
    }

    #[test]
    fn autocorrelation_is_one_at_lag_zero() {
        let x = independent_normal_chain(1000, 0.0);
        assert!((autocorrelation(&x, 0) - 1.0).abs() < 1e-12);
        assert!(autocorrelation(&x, 1).abs() < 0.1);
    }

    #[test]
    fn autocorrelation_detects_deterministic_trend() {
        // A strictly increasing ramp is strongly positively autocorrelated
        // (the common-mean estimator gives rho_1 ~ 0.97, not exactly 1).
        let x: Vec<f64> = (0..100).map(f64::from).collect();
        let r1 = autocorrelation(&x, 1);
        assert!(r1 > 0.9 && r1 < 1.0, "rho1 {r1} should be near 1");
        assert!(autocorrelation(&x, 5) < r1, "autocorrelation should decay");
    }

    #[test]
    fn ess_is_near_n_for_independent_chain() {
        let x = independent_normal_chain(2000, 0.0);
        let ess = effective_sample_size(&x);
        assert!(ess > 1000.0, "ESS {ess} too low for an iid chain");
        assert!(ess <= 2000.0 + 1e-9);
    }

    #[test]
    fn ess_is_low_for_strongly_autocorrelated_chain() {
        let mut rng = Rng::new(3);
        // AR(1) with phi = 0.95: heavily autocorrelated.
        let mut x = Vec::with_capacity(2000);
        let mut v = 0.0;
        for _ in 0..2000 {
            v = 0.95 * v + crate::distributions::Normal { mu: 0.0, sigma: 0.3 }.sample(&mut rng);
            x.push(v);
        }
        let ess = effective_sample_size(&x);
        assert!(ess < 400.0, "ESS {ess} should be small for AR(1) phi=0.95");
    }

    #[test]
    fn gelman_rubin_is_minimal_for_identical_chains() {
        // Identical chains give B = 0, so the corrected R-hat is
        // sqrt((n-1)/n) < 1 rather than exactly 1.
        let chain: Vec<Vec<f64>> = independent_normal_chain(500, 0.0)
            .into_iter()
            .map(|x| vec![x])
            .collect();
        let chains = vec![chain.clone(), chain.clone(), chain.clone()];
        let rhat = gelman_rubin(&chains);
        assert!(
            (rhat[0] - (499.0_f64 / 500.0).sqrt()).abs() < 1e-9,
            "rhat {}",
            rhat[0]
        );
    }

    #[test]
    fn gelman_rubin_accepts_converged_chains() {
        let chains: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|_| {
                independent_normal_chain(1000, 0.0)
                    .into_iter()
                    .map(|x| vec![x])
                    .collect()
            })
            .collect();
        let rhat = gelman_rubin(&chains);
        assert!(rhat[0] < 1.1, "R-hat {} should be < 1.1", rhat[0]);
    }

    #[test]
    fn gelman_rubin_detects_divergent_chains() {
        let chains: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|i| {
                independent_normal_chain(1000, f64::from(i) * 2.0)
                    .into_iter()
                    .map(|x| vec![x])
                    .collect()
            })
            .collect();
        let rhat = gelman_rubin(&chains);
        assert!(rhat[0] > 1.5, "R-hat {} should be large", rhat[0]);
    }

    #[test]
    fn geweke_accepts_stationary_chain() {
        let x = independent_normal_chain(4000, 0.0);
        let z = geweke(&x, 0.1, 0.5);
        assert!(z.is_finite());
        assert!(z.abs() < 3.0, "Geweke z {z} too extreme for stationary chain");
    }

    #[test]
    fn geweke_rejects_drift() {
        let x: Vec<f64> = (0..4000).map(|i| i as f64 * 0.01).collect();
        let z = geweke(&x, 0.1, 0.5);
        assert!(z.abs() > 3.0, "Geweke z {z} should detect drift");
    }

    #[test]
    fn thinning_recommendation_found_for_ar1() {
        let mut rng = Rng::new(5);
        let mut x = Vec::with_capacity(2000);
        let mut v = 0.0;
        for _ in 0..2000 {
            v = 0.9 * v + crate::distributions::Normal { mu: 0.0, sigma: 0.5 }.sample(&mut rng);
            x.push(v);
        }
        let interval = thinning_interval(&x, 0.1);
        assert!(interval.is_some());
        assert!(interval.unwrap() >= 2);
    }

    #[test]
    fn summary_reports_sensible_values() {
        let x = independent_normal_chain(1000, 3.0);
        let s = summarize(&x);
        assert_eq!(s.n, 1000);
        assert!((s.mean - 3.0).abs() < 0.2);
        assert!(s.ess > 0.0 && s.ess <= 1000.0);
        assert!(s.iact >= 1.0);
        assert!(s.rho1.abs() < 0.2);
    }

    #[test]
    fn multi_dimension_ess_and_rhat() {
        let mut rng = Rng::new(11);
        let dist = crate::distributions::Normal { mu: 0.0, sigma: 1.0 };
        let chains: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|_| (0..600).map(|_| vec![dist.sample(&mut rng), dist.sample(&mut rng)]).collect())
            .collect();
        let rhat = gelman_rubin(&chains);
        assert_eq!(rhat.len(), 2);
        assert!(rhat.iter().all(|&r| r < 1.1));
        let ess = effective_sample_size_multi(&chains[0]);
        assert_eq!(ess.len(), 2);
        assert!(ess[0] > 300.0);
    }
}
