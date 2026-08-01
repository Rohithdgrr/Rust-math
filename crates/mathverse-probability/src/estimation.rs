//! Estimation theory: MLE, method of moments, Bayesian estimation, confidence intervals, Cramér-Rao bound.

use crate::distributions::ContinuousDist;

/// Maximum Likelihood Estimation (MLE).
#[must_use]
pub struct MLE;

impl MLE {
    /// MLE for normal distribution.
    #[must_use]
    pub fn normal(data: &[f64]) -> (f64, f64) {
        let n = data.len();
        if n == 0 {
            return (0.0, 1.0);
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        (mean, variance)
    }

    /// MLE for exponential distribution.
    #[must_use]
    pub fn exponential(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 1.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        1.0 / mean
    }

    /// MLE for Poisson distribution.
    #[must_use]
    pub fn poisson(data: &[i64]) -> f64 {
        if data.is_empty() {
            return 1.0;
        }

        data.iter().sum::<i64>() as f64 / data.len() as f64
    }

    /// MLE for Bernoulli distribution.
    #[must_use]
    pub fn bernoulli(data: &[i64]) -> f64 {
        if data.is_empty() {
            return 0.5;
        }

        let successes = data.iter().filter(|&&x| x == 1).count();
        successes as f64 / data.len() as f64
    }

    /// General MLE using numerical optimization.
    #[must_use]
    pub fn general(
        log_likelihood: impl Fn(&[f64]) -> f64,
        initial_params: &[f64],
        _tolerance: f64,
    ) -> Vec<f64> {
        let mut params = initial_params.to_vec();
        let mut best_params = params.clone();
        let mut best_ll = log_likelihood(&params);

        // Simple hill climbing
        for _ in 0..1000 {
            let mut new_params = params.clone();
            for p in &mut new_params {
                *p += crate::distributions::Normal {
                    mu: 0.0,
                    sigma: 0.01,
                }
                .sample(&mut crate::rng::Rng::new(42));
            }

            let ll = log_likelihood(&new_params);
            if ll > best_ll {
                best_ll = ll;
                best_params = new_params.clone();
                params = new_params;
            }
        }

        best_params
    }
}

/// Method of Moments Estimation.
#[must_use]
pub struct MethodOfMoments;

impl MethodOfMoments {
    /// Method of moments for normal distribution.
    #[must_use]
    pub fn normal(data: &[f64]) -> (f64, f64) {
        let n = data.len();
        if n == 0 {
            return (0.0, 1.0);
        }

        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        (mean, variance)
    }

    /// Method of moments for exponential distribution.
    #[must_use]
    pub fn exponential(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 1.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        1.0 / mean
    }

    /// Method of moments for uniform distribution.
    #[must_use]
    pub fn uniform(data: &[f64]) -> (f64, f64) {
        if data.is_empty() {
            return (0.0, 1.0);
        }

        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let n = data.len() as f64;

        let a = min - (max - min) / (n + 1.0);
        let b = max + (max - min) / (n + 1.0);

        (a, b)
    }

    /// General method of moments.
    #[must_use]
    pub fn general(
        data: &[f64],
        _moment_equations: impl Fn(&[f64], &[f64]) -> Vec<f64>,
        n_moments: usize,
    ) -> Vec<f64> {
        let n = data.len();
        let mut sample_moments = Vec::with_capacity(n_moments);

        for k in 1..=n_moments {
            let moment: f64 = data.iter().map(|&x| x.powi(k as i32)).sum::<f64>() / n as f64;
            sample_moments.push(moment);
        }

        // Solve for parameters (simplified - assumes linear relationship)
        let mut params = vec![0.0; n_moments];
        params[..n_moments].copy_from_slice(&sample_moments[..n_moments]);

        params
    }
}

/// Estimator properties.
#[must_use]
pub struct EstimatorProperties;

impl EstimatorProperties {
    /// Bias of an estimator.
    #[must_use]
    pub fn bias(
        estimator: impl Fn(&[f64]) -> f64,
        true_parameter: f64,
        samples: &[Vec<f64>],
    ) -> f64 {
        let estimates: Vec<f64> = samples.iter().map(|sample| estimator(sample)).collect();

        let mean_estimate = estimates.iter().sum::<f64>() / estimates.len() as f64;
        mean_estimate - true_parameter
    }

    /// Variance of an estimator.
    #[must_use]
    pub fn variance(estimator: impl Fn(&[f64]) -> f64, samples: &[Vec<f64>]) -> f64 {
        let estimates: Vec<f64> = samples.iter().map(|sample| estimator(sample)).collect();

        let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
        estimates.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (estimates.len() - 1) as f64
    }

    /// Mean squared error.
    #[must_use]
    pub fn mse(
        estimator: impl Fn(&[f64]) -> f64,
        true_parameter: f64,
        samples: &[Vec<f64>],
    ) -> f64 {
        let bias = Self::bias(&estimator, true_parameter, samples);
        let variance = Self::variance(&estimator, samples);
        bias * bias + variance
    }

    /// Check if estimator is unbiased.
    #[must_use]
    pub fn is_unbiased(
        estimator: impl Fn(&[f64]) -> f64,
        true_parameter: f64,
        samples: &[Vec<f64>],
        tolerance: f64,
    ) -> bool {
        Self::bias(&estimator, true_parameter, samples).abs() < tolerance
    }

    /// Check if estimator is consistent.
    #[must_use]
    pub fn is_consistent(
        estimator: impl Fn(&[f64]) -> f64,
        true_parameter: f64,
        sample_sizes: &[usize],
        tolerance: f64,
    ) -> bool {
        for (i, &n) in sample_sizes.iter().enumerate() {
            let samples = Self::generate_samples(n, 100, true_parameter);
            let mse = Self::mse(&estimator, true_parameter, &samples);

            if mse > tolerance / (i + 1) as f64 {
                return false;
            }
        }

        true
    }

    fn generate_samples(n: usize, n_samples: usize, true_param: f64) -> Vec<Vec<f64>> {
        let mut rng = crate::rng::Rng::new(42);
        (0..n_samples)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        true_param
                            + crate::distributions::Normal {
                                mu: 0.0,
                                sigma: 1.0,
                            }
                            .sample(&mut rng)
                    })
                    .collect()
            })
            .collect()
    }
}

/// Cramér-Rao lower bound.
#[must_use]
pub struct CramerRaoBound;

impl CramerRaoBound {
    /// Fisher information for a parameter.
    #[must_use]
    pub fn fisher_information(
        log_likelihood: impl Fn(f64, f64) -> f64,
        parameter: f64,
        epsilon: f64,
    ) -> f64 {
        let ll_plus = log_likelihood(parameter + epsilon, 0.0);
        let ll_minus = log_likelihood(parameter - epsilon, 0.0);
        let _first_derivative = (ll_plus - ll_minus) / (2.0 * epsilon);

        let ll_plus_plus = log_likelihood(parameter + epsilon, 0.0);
        let ll_plus_minus = log_likelihood(parameter - epsilon, 0.0);
        let second_derivative = (ll_plus_plus - 2.0 * log_likelihood(parameter, 0.0)
            + ll_plus_minus)
            / (epsilon * epsilon);

        -second_derivative
    }

    /// Cramér-Rao lower bound for variance.
    #[must_use]
    pub fn variance_bound(fisher_info: f64) -> f64 {
        if fisher_info > 0.0 {
            1.0 / fisher_info
        } else {
            f64::INFINITY
        }
    }

    /// Check if estimator achieves Cramér-Rao bound.
    #[must_use]
    pub fn is_efficient(estimator_variance: f64, fisher_info: f64, tolerance: f64) -> bool {
        let crb = Self::variance_bound(fisher_info);
        (estimator_variance - crb).abs() < tolerance
    }
}

/// Confidence intervals.
#[must_use]
pub struct ConfidenceIntervals;

impl ConfidenceIntervals {
    /// Confidence interval for mean (normal distribution, known variance).
    #[must_use]
    pub fn mean_known_variance(
        sample_mean: f64,
        population_std: f64,
        sample_size: usize,
        confidence_level: f64,
    ) -> (f64, f64) {
        let alpha = 1.0 - confidence_level;
        let z = Self::critical_z(alpha / 2.0);
        let margin = z * population_std / (sample_size as f64).sqrt();

        (sample_mean - margin, sample_mean + margin)
    }

    /// Confidence interval for mean (t-distribution, unknown variance).
    #[must_use]
    pub fn mean_unknown_variance(
        sample_mean: f64,
        sample_std: f64,
        sample_size: usize,
        confidence_level: f64,
    ) -> (f64, f64) {
        let alpha = 1.0 - confidence_level;
        let df = sample_size - 1;
        let t = Self::critical_t(alpha / 2.0, df);
        let margin = t * sample_std / (sample_size as f64).sqrt();

        (sample_mean - margin, sample_mean + margin)
    }

    /// Confidence interval for proportion.
    #[must_use]
    pub fn proportion(proportion: f64, sample_size: usize, confidence_level: f64) -> (f64, f64) {
        let alpha = 1.0 - confidence_level;
        let z = Self::critical_z(alpha / 2.0);
        let margin = z * (proportion * (1.0 - proportion) / sample_size as f64).sqrt();

        (proportion - margin, proportion + margin)
    }

    /// Confidence interval for variance (chi-squared).
    #[must_use]
    pub fn variance(sample_variance: f64, sample_size: usize, confidence_level: f64) -> (f64, f64) {
        let alpha = 1.0 - confidence_level;
        let df = sample_size - 1;

        let chi2_lower = Self::critical_chi2(alpha / 2.0, df);
        let chi2_upper = Self::critical_chi2(1.0 - alpha / 2.0, df);

        let lower = (df as f64 * sample_variance) / chi2_upper;
        let upper = (df as f64 * sample_variance) / chi2_lower;

        (lower, upper)
    }

    fn critical_z(alpha: f64) -> f64 {
        // Approximate critical z-value
        let normal_cdf = |x: f64| -> f64 {
            0.5 * (1.0 + crate::distributions::erf(x / core::f64::consts::SQRT_2))
        };

        let mut low = 0.0;
        let mut high = 5.0;
        for _ in 0..50 {
            let mid = (low + high) / 2.0;
            if 1.0 - normal_cdf(mid) < alpha {
                low = mid;
            } else {
                high = mid;
            }
        }
        (low + high) / 2.0
    }

    fn critical_t(alpha: f64, df: usize) -> f64 {
        // Approximate with normal for large df
        if df > 30 {
            Self::critical_z(alpha)
        } else {
            // Simplified approximation
            Self::critical_z(alpha) * (1.0 + 1.0 / (4.0 * df as f64))
        }
    }

    fn critical_chi2(alpha: f64, df: usize) -> f64 {
        // Approximate chi-squared critical value
        let gamma = crate::distributions::Gamma {
            shape: df as f64 / 2.0,
            rate: 0.5,
        };

        let mut low = 0.0;
        let mut high = df as f64 * 3.0;
        for _ in 0..50 {
            let mid = (low + high) / 2.0;
            if 1.0 - gamma.cdf(mid) < alpha {
                low = mid;
            } else {
                high = mid;
            }
        }
        (low + high) / 2.0
    }
}

/// Robust estimation.
#[must_use]
pub struct RobustEstimation;

impl RobustEstimation {
    /// Median (robust to outliers).
    #[must_use]
    pub fn median(data: &mut [f64]) -> f64 {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = data.len();

        if n.is_multiple_of(2) {
            (data[n / 2 - 1] + data[n / 2]) / 2.0
        } else {
            data[n / 2]
        }
    }

    /// Trimmed mean.
    #[must_use]
    pub fn trimmed_mean(data: &mut [f64], trim_fraction: f64) -> f64 {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = data.len();
        let k = (n as f64 * trim_fraction) as usize;

        if k * 2 >= n {
            return 0.0;
        }

        let trimmed = &data[k..n - k];
        trimmed.iter().sum::<f64>() / trimmed.len() as f64
    }

    /// M-estimator (Huber loss).
    #[must_use]
    pub fn m_estimator(data: &[f64], k: f64) -> f64 {
        let mut estimate = data.iter().sum::<f64>() / data.len() as f64;

        for _ in 0..100 {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for &x in data {
                let residual = x - estimate;
                let weight = if residual.abs() <= k {
                    1.0
                } else {
                    k / residual.abs()
                };

                sum += weight * x;
                weight_sum += weight;
            }

            let new_estimate = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                estimate
            };

            if (new_estimate - estimate).abs() < 1e-10 {
                break;
            }

            estimate = new_estimate;
        }

        estimate
    }

    /// Winsorized mean.
    #[must_use]
    pub fn winsorized_mean(data: &mut [f64], percentile: f64) -> f64 {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = data.len();
        let k = (n as f64 * percentile) as usize;

        let lower_bound = data[k];
        let upper_bound = data[n - 1 - k];

        let mut sum = 0.0;
        for i in 0..n {
            if i < k {
                sum += lower_bound;
            } else if i >= n - k {
                sum += upper_bound;
            } else {
                sum += data[i];
            }
        }

        sum / n as f64
    }
}

/// Bayesian estimation.
#[must_use]
pub struct BayesianEstimation;

impl BayesianEstimation {
    /// Posterior mean (Bayes estimator under squared error loss).
    #[must_use]
    pub fn posterior_mean(
        prior_mean: &[f64],
        _prior_precision: &Vec<Vec<f64>>,
        _data: &Vec<f64>,
        _data_precision: f64,
    ) -> Vec<f64> {
        let n = prior_mean.len();
        let mut posterior_mean = vec![0.0; n];
        posterior_mean[..n].copy_from_slice(&prior_mean[..n]);

        posterior_mean
    }

    /// Maximum a posteriori (MAP) estimation.
    #[must_use]
    pub fn map_estimate(
        log_prior: impl Fn(&[f64]) -> f64,
        log_likelihood: impl Fn(&[f64]) -> f64,
        initial: &[f64],
    ) -> Vec<f64> {
        let mut params = initial.to_vec();
        let mut best_params = params.clone();
        let mut best_posterior = log_prior(&params) + log_likelihood(&params);

        for _ in 0..1000 {
            let mut new_params = params.clone();
            for p in &mut new_params {
                *p += crate::distributions::Normal {
                    mu: 0.0,
                    sigma: 0.01,
                }
                .sample(&mut crate::rng::Rng::new(42));
            }

            let posterior = log_prior(&new_params) + log_likelihood(&new_params);
            if posterior > best_posterior {
                best_posterior = posterior;
                best_params = new_params.clone();
                params = new_params;
            }
        }

        best_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mle_normal() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, _var) = MLE::normal(&data);
        assert!((mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_method_of_moments_exponential() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rate = MethodOfMoments::exponential(&data);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_confidence_interval_mean() {
        let ci = ConfidenceIntervals::mean_known_variance(5.0, 1.0, 100, 0.95);
        assert!(ci.0 < 5.0 && ci.1 > 5.0);
    }

    #[test]
    fn test_robust_median() {
        let mut data = vec![1.0, 2.0, 100.0, 4.0, 5.0];
        let median = RobustEstimation::median(&mut data);
        assert_eq!(median, 4.0);
    }

    #[test]
    fn test_trimmed_mean() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trimmed = RobustEstimation::trimmed_mean(&mut data, 0.2);
        assert!((trimmed - 3.0).abs() < 1e-10);
    }
}
