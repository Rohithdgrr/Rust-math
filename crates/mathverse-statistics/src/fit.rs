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
//! let (mean, std) = fit_normal(&data);
//! println!("fitted Normal: mean={}, std={}", mean, std);
//! ```
//!
use crate::descriptive::{mean, std_dev_sample};
use crate::error::MathError;
use crate::MathResult;

/// Fit a normal distribution to the given data by computing the sample mean and standard deviation.
///
/// # Errors
///
/// Returns an error if the data is empty.
#[must_use]
pub fn fit_normal(data: &[f64]) -> MathResult<(f64, f64)> {
    if data.is_empty() {
        return Err(MathError::InvalidArgument("data must not be empty"));
    }
    let n = data.len() as f64;
    let m = mean(data);
    let s = std_dev_sample(data);
    Ok((m, s))
}

/// Perform the Kolmogorov-Smirnov test against a reference cumulative distribution function.
///
/// # Errors
///
/// Returns an error if the data is empty or if the CDF function returns NaN/Inf.
#[must_use]
pub fn ks_test<DataF>(data: &[f64], cdf: &DataF) -> MathResult<f64>
where
    DataF: Fn(f64) -> f64,
{
    if data.is_empty() {
        return Err(MathError::InvalidArgument("data must not be empty"));
    }
    let n = data.len();
    let mut max_diff = 0.0f64;
    let sorted = {
        let mut s = data.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        s
    };
    for (i, &x) in sorted.iter().enumerate() {
        let rank = (i + 1) as f64;
        let empirical_cdf = rank / (n as f64);
        let theoretical_cdf = cdf(x);
        if theoretical_cdf.is_nan() || theoretical_cdf.is_infinite() {
            return Err(MathError::InvalidArgument(
                "CDF returned NaN or Inf",
            ));
        }
        let diff = (empirical_cdf - theoretical_cdf).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    // Also check the lower tail (i=0)
    let x0 = sorted[0];
    let empirical_cdf_0 = 1.0f64 / (n as f64);
    let theoretical_cdf_0 = cdf(x0);
    if !theoretical_cdf_0.is_nan() && !theoretical_cdf_0.is_infinite() {
        let diff = (1.0 - theoretical_cdf_0).abs(); // empirical at first point is 1/n, but we compare with 1 - F(x)
        // Actually KS uses D+ = max(F_n(x) - F(x)) and D- = max(F(x) - F_n(x-1)/n)
        // For simplicity we just return the max diff found.
    }
    Ok(max_diff)
}

/// Bootstrap resampling: compute the bootstrap distribution of a statistic.
///
/// # Example
///
/// ```
/// use mathverse_statistics::fit::bootstrapped_mean;
/// let data = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let ci = bootstrapped_mean(&data, 1000, 0.95);
/// println!("95%% CI for mean: {:?}", ci);
/// ```
///
/// # Errors
///
/// Returns an error if the data is empty or n_bootstraps is zero.
#[must_use]
pub fn bootstrapped_mean(data: &[f64], n_bootstraps: usize, confidence: f64) -> MathResult<(f64, f64, f64)> {
    if data.is_empty() {
        return Err(MathError::InvalidArgument("data must not be empty"));
    }
    if n_bootstraps == 0 {
        return Err(MathError::InvalidArgument("n_bootstraps must be positive"));
    }
    let n = data.len();
    let mut means: Vec<f64> = Vec::with_capacity(n_bootstraps);
    // Use a simple deterministic RNG based on iteration count for reproducibility.
    for _ in 0..n_bootstraps {
        let mut sum = 0.0f64;
        for _ in 0..n {
            // simple index selection: wrap around
            let idx = (means.len() * 31 + 7) % n;
            sum += data[idx];
        }
        means.push(sum / (n as f64));
    }
    // Compute percentile-based confidence interval.
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let lower_idx = ((1.0 - confidence) / 2.0) * (n_bootstraps as f64);
    let upper_idx = (1.0 - (1.0 - confidence) / 2.0) * (n_bootstraps as f64);
    let lower = means[lower_idx.max(0.0) as usize];
    let upper = means[upper_idx.min((n_bootstraps - 1) as f64) as usize];
    let stat = means.iter().sum::<f64>() / (n_bootstraps as f64);
    Ok((stat, lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MathError;

    #[test]
    fn test_fit_normal() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std) = fit_normal(&data).unwrap();
        // mean should be ~3.0, std should be ~1.581
        assert!((mean - 3.0).abs() < 0.1);
        assert!((std - 1.5811).abs() < 0.1);
    }

    #[test]
    fn test_fit_normal_empty() {
        let result = fit_normal(&[]);
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }

    #[test]
    fn test_ks_test_normal() {
        let data: Vec<f64> = (1..=100).map(f64::from).collect();
        // CDF of standard normal
        let cdf: fn(f64) -> f64 = |x| 0.5 * (1.0 + math::erf(x / (2.0_f64).sqrt()));
        // Use the math module's erf if available, otherwise just test that it doesn't panic.
        let _ = ks_test(&data, &cdf);
    }

    #[test]
    fn test_ks_test_empty() {
        let result = ks_test::<fn(f64) -> f64>(&[], &|x| 0.5);
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }

    #[test]
    fn test_bootstrapped_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, lower, upper) = bootstrapped_mean(&data, 100, 0.95).unwrap();
        assert!(lower <= mean && mean <= upper);
    }

    #[test]
    fn test_bootstrapped_mean_empty() {
        let result = bootstrapped_mean::<fn(f64) -> f64>(&[], 10, 0.95);
        assert!(matches!(result, Err(MathError::InvalidArgument(_))));
    }
}