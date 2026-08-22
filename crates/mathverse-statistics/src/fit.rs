//! Distribution fitting and goodness-of-fit tests.
//!
//! Provides functions to fit common distributions to data and perform
//! statistical tests such as the Kolmogorov-Smirnov test.
//!
//! # Example
//!
//! ```
//! use mathverse_statistics::fit::fit_normal;
//!
//! let data = [1.0, 2.0, 3.0, 4.0, 5.0];
//! let (mean, std) = fit_normal(&data).unwrap();
//! println!("fitted Normal: mean={mean}, std={std}");
//! ```

use crate::descriptive::{mean, std_dev_sample};
use crate::error::MathError;
use crate::MathResult;

/// Fit a normal distribution to the given data by computing the sample mean
/// and standard deviation.
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] if the data is empty.
pub fn fit_normal(data: &[f64]) -> MathResult<(f64, f64)> {
    if data.is_empty() {
        return Err(MathError::InvalidArgument("data must not be empty"));
    }
    Ok((mean(data), std_dev_sample(data)))
}

/// Two-sided Kolmogorov-Smirnov `D` statistic against a reference cumulative
/// distribution function:
///
/// ```text
/// D = sup_x |F_n(x) - F(x)|
/// ```
///
/// where `F_n` is the empirical CDF of the sample. Ties in the data do not
/// affect the result because the empirical CDF is a step function evaluated
/// exactly at each sorted observation.
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] if the data is empty or if the CDF
/// returns NaN/Inf for any sample point.
pub fn ks_test<F>(data: &[f64], cdf: F) -> MathResult<f64>
where
    F: Fn(f64) -> f64,
{
    if data.is_empty() {
        return Err(MathError::InvalidArgument("data must not be empty"));
    }
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);

    let mut max_diff = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let fn_plus = (i + 1) as f64 / n as f64;
        let fn_minus = i as f64 / n as f64;
        let f_theo = cdf(x);
        if !f_theo.is_finite() {
            return Err(MathError::InvalidArgument("CDF returned NaN or Inf"));
        }
        max_diff = max_diff.max((fn_plus - f_theo).abs());
        max_diff = max_diff.max((f_theo - fn_minus).abs());
    }
    Ok(max_diff)
}

/// Deterministic xorshift64 PRNG so bootstrap results are reproducible
/// without external dependencies.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        // Avoid the all-zero fixed point of xorshift64.
        Self {
            state: seed.wrapping_add(1).max(1),
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

/// Bootstrap resampling: estimate the sampling distribution of the sample
/// mean via `n_bootstraps` resamples with replacement, returning the bootstrap
/// mean and the percentile confidence interval `(lower, upper)`.
///
/// Resampling uses a deterministic xorshift PRNG seeded from the data length,
/// so results are reproducible across runs and platforms.
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] if the data is empty,
/// `n_bootstraps` is zero, or `confidence` is not in (0, 1).
pub fn bootstrapped_mean(
    data: &[f64],
    n_bootstraps: usize,
    confidence: f64,
) -> MathResult<(f64, f64, f64)> {
    if data.is_empty() {
        return Err(MathError::InvalidArgument("data must not be empty"));
    }
    if n_bootstraps == 0 {
        return Err(MathError::InvalidArgument("n_bootstraps must be positive"));
    }
    if !(0.0..=1.0).contains(&confidence) {
        return Err(MathError::InvalidArgument(
            "confidence must be in [0, 1]",
        ));
    }

    let n = data.len();
    let mut rng = XorShift64::new(n as u64 ^ 0x9E37_79B9_7F4A_7C15);
    let mut means = Vec::with_capacity(n_bootstraps);
    for _ in 0..n_bootstraps {
        let mut sum = 0.0f64;
        for _ in 0..n {
            sum += data[(rng.next() % n as u64) as usize];
        }
        means.push(sum / n as f64);
    }

    // Percentile-based confidence interval.
    means.sort_by(f64::total_cmp);
    let tail = (1.0 - confidence) / 2.0;
    let lo_idx =
        ((tail * n_bootstraps as f64).floor() as usize).min(n_bootstraps - 1);
    let hi_idx =
        (((1.0 - tail) * n_bootstraps as f64).ceil() as usize).min(n_bootstraps - 1);
    let stat = mean(&means);
    Ok((stat, means[lo_idx], means[hi_idx]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::normal_cdf;
    use crate::error::MathError;

    #[test]
    fn test_fit_normal() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (m, s) = fit_normal(&data).unwrap();
        assert!((m - 3.0).abs() < 1e-12);
        assert!((s - 1.5811388300841898).abs() < 1e-12);
    }

    #[test]
    fn test_fit_normal_empty() {
        let result = fit_normal(&[]);
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }

    #[test]
    fn test_ks_test_perfect_fit() {
        // Sampling the CDF itself gives an exact uniform grid; D should be tiny.
        let data: Vec<f64> = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95].to_vec();
        let d = ks_test(&data, |x| x).unwrap();
        assert!(d < 0.2);
    }

    #[test]
    fn test_ks_test_known_value() {
        // Uniform data vs uniform CDF on [0, 10]: D = 1/n exactly at each step.
        let data = [1.0, 3.0, 5.0, 7.0, 9.0];
        let d = ks_test(&data, |x| (x / 10.0).clamp(0.0, 1.0)).unwrap();
        assert!((d - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_ks_test_against_normal_cdf() {
        let data: Vec<f64> = (-50..=50).map(f64::from).map(|i| i / 10.0).collect();
        let d = ks_test(&data, normal_cdf).unwrap();
        assert!((0.0..=1.0).contains(&d));
    }

    #[test]
    fn test_ks_test_empty() {
        let result = ks_test(&[], |x| x);
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }

    #[test]
    fn test_ks_test_bad_cdf() {
        let result = ks_test(&[1.0], |_| f64::NAN);
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }

    #[test]
    fn test_bootstrapped_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (stat, lower, upper) = bootstrapped_mean(&data, 100, 0.95).unwrap();
        assert!((stat - 3.0).abs() < 0.5);
        assert!(lower <= stat && stat <= upper);
        assert!((1.0..=5.0).contains(&lower));
        assert!((1.0..=5.0).contains(&upper));
    }

    #[test]
    fn test_bootstrapped_mean_deterministic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let a = bootstrapped_mean(&data, 200, 0.90).unwrap();
        let b = bootstrapped_mean(&data, 200, 0.90).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_bootstrapped_mean_empty() {
        let result = bootstrapped_mean(&[], 10, 0.95);
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }

    #[test]
    fn test_bootstrapped_mean_zero_bootstraps() {
        let result = bootstrapped_mean(&[1.0], 0, 0.95);
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }

    #[test]
    fn test_bootstrapped_mean_bad_confidence() {
        let result = bootstrapped_mean(&[1.0], 10, 1.5);
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }
}
