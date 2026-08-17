//! Online (streaming) statistics: one-pass Welford mean/variance/skewness/
//! kurtosis, Chan et al. parallel merge, and slice-based covariance helpers.
//!
//! All accumulators are `no_std`-compatible and numerically stable: the
//! variance estimate never relies on `sum(x^2) - n·mean^2`, so large offsets
//! do not catastrophically cancel.

/// Streaming accumulator for the first four central moments (Welford's
/// algorithm). Feed samples with [`StreamingStats::update`] and combine
/// partial results with [`StreamingStats::merge`] (Chan et al.).
#[must_use]
#[derive(Clone, Debug, Default)]
pub struct StreamingStats {
    count: u64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    min: f64,
    max: f64,
}

impl StreamingStats {
    /// Create an empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of observations seen so far.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Running arithmetic mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Smallest observation seen so far (or `+inf` if empty).
    #[must_use]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Largest observation seen so far (or `-inf` if empty).
    #[must_use]
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Add one observation, updating all four moments in O(1).
    pub fn update(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        let n1 = self.count;
        self.count += 1;
        let n = self.count;
        if n == 1 {
            self.mean = x;
            self.m2 = 0.0;
            self.m3 = 0.0;
            self.m4 = 0.0;
            self.min = x;
            self.max = x;
            return;
        }
        let delta = x - self.mean;
        let delta_n = delta / n as f64;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1 as f64;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n as f64 * n as f64 - 3.0 * n as f64 + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n as f64 - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    /// Sample variance (Bessel-corrected). `NaN` with fewer than 2 samples.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        self.m2 / (self.count as f64 - 1.0)
    }

    /// Population variance. `NaN` when empty.
    #[must_use]
    pub fn population_variance(&self) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        self.m2 / self.count as f64
    }

    /// Sample standard deviation.
    #[must_use]
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sample skewness (Fisher-Pearson `g1`). `NaN` with fewer than 3 samples
    /// or zero variance.
    #[must_use]
    pub fn skewness(&self) -> f64 {
        if self.count < 3 || self.m2 == 0.0 {
            return f64::NAN;
        }
        let n = self.count as f64;
        (n * n - n).sqrt() / (n - 2.0) * self.m3 / self.m2.powf(1.5)
    }

    /// Sample excess kurtosis. `NaN` with fewer than 4 samples or zero
    /// variance. Zero indicates a Gaussian-shaped distribution.
    #[must_use]
    pub fn kurtosis(&self) -> f64 {
        if self.count < 4 || self.m2 == 0.0 {
            return f64::NAN;
        }
        let n = self.count as f64;
        let m2sq = self.m2 * self.m2;
        (n - 1.0) / ((n - 2.0) * (n - 3.0))
            * ((n + 1.0) * self.m4 / m2sq - 3.0 * (n - 1.0))
            + 3.0
    }

    /// Combine another accumulator into `self` (Chan et al. parallel merge).
    /// The two streams may have been updated independently; the result is
    /// identical (up to floating point) to updating `self` with all
    /// observations in order.
    pub fn merge(&mut self, other: &StreamingStats) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let n_a = self.count as f64;
        let n_b = other.count as f64;
        let n = n_a + n_b;
        let delta = other.mean - self.mean;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;

        self.mean = (n_a * self.mean + n_b * other.mean) / n;
        self.m4 = self.m4
            + other.m4
            + delta4 * n_a * n_b * (n_a * n_a - n_a * n_b + n_b * n_b) / (n * n * n)
            + 6.0 * delta2 * (n_a * n_a * other.m2 + n_b * n_b * self.m2) / (n * n)
            + 4.0 * delta * (n_a * other.m3 - n_b * self.m3) / n;
        self.m3 = self.m3
            + other.m3
            + delta3 * n_a * n_b * (n_a - n_b) / (n * n)
            + 3.0 * delta * (n_a * other.m2 - n_b * self.m2) / n;
        self.m2 = self.m2 + other.m2 + delta2 * n_a * n_b / n;
        self.count += other.count;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }

    /// Convenience: feed every element of `samples`.
    pub fn extend(&mut self, samples: &[f64]) {
        for &x in samples {
            self.update(x);
        }
    }

    /// Statistics of `samples` in one pass.
    #[must_use]
    pub fn from_slice(samples: &[f64]) -> Self {
        let mut s = Self::new();
        s.extend(samples);
        s
    }
}

/// Arithmetic mean of a slice. `NaN` if empty.
#[must_use]
pub fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Unbiased sample variance. `NaN` with fewer than 2 elements.
#[must_use]
pub fn variance(samples: &[f64]) -> f64 {
    StreamingStats::from_slice(samples).variance()
}

/// Population variance. `NaN` if empty.
#[must_use]
pub fn population_variance(samples: &[f64]) -> f64 {
    StreamingStats::from_slice(samples).population_variance()
}

/// Sample standard deviation.
#[must_use]
pub fn stddev(samples: &[f64]) -> f64 {
    StreamingStats::from_slice(samples).stddev()
}

/// Sample covariance of two paired slices.
///
/// # Panics
/// Panics if the slices differ in length or are empty.
#[must_use]
pub fn covariance(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "covariance slices must have equal length");
    assert!(!x.is_empty(), "covariance needs at least one sample");
    let n = x.len() as f64;
    let x_mean = mean(x);
    let y_mean = mean(y);
    x.iter()
        .zip(y)
        .map(|(&a, &b)| (a - x_mean) * (b - y_mean))
        .sum::<f64>()
        / (n - 1.0)
}

/// Pearson correlation coefficient in `[-1, 1]`. `NaN` if either variance is
/// zero.
#[must_use]
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let sx = stddev(x);
    let sy = stddev(y);
    if sx == 0.0 || sy == 0.0 {
        return f64::NAN;
    }
    covariance(x, y) / (sx * sy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn welford_matches_naive() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = StreamingStats::from_slice(&data);
        assert!((s.mean() - 5.0).abs() < 1e-12);
        assert!((s.variance() - 4.571_428_571_428_571).abs() < 1e-12);
        assert!((s.stddev() - 2.138_089_935_299_395).abs() < 1e-9);
        assert_eq!(s.min(), 2.0);
        assert_eq!(s.max(), 9.0);
        assert_eq!(s.count(), 8);
    }

    #[test]
    fn skewness_and_kurtosis_of_normal_are_zero() {
        let mut rng = Rng::new(7);
        let n = crate::distributions::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        let mut s = StreamingStats::new();
        for _ in 0..50_000 {
            s.update(n.sample(&mut rng));
        }
        assert!(s.skewness().abs() < 0.05, "skew {}", s.skewness());
        assert!(s.kurtosis().abs() < 0.15, "kurt {}", s.kurtosis());
        assert!((s.mean()).abs() < 0.02);
        assert!((s.stddev() - 1.0).abs() < 0.02);
    }

    #[test]
    fn skewness_sign_detects_asymmetry() {
        let mut s = StreamingStats::new();
        for x in 0..10_000 {
            s.update(x as f64);
        }
        assert!(s.skewness() > 0.0);
        let mut s2 = StreamingStats::new();
        for x in 0..10_000 {
            s2.update((9_999 - x) as f64);
        }
        assert!(s2.skewness() < 0.0);
    }

    #[test]
    fn merge_matches_single_stream() {
        let mut rng = Rng::new(42);
        let normal = crate::distributions::Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        let mut whole = StreamingStats::new();
        let mut left = StreamingStats::new();
        let mut right = StreamingStats::new();
        for i in 0..10_000 {
            let x = normal.sample(&mut rng);
            whole.update(x);
            if i < 4_000 {
                left.update(x);
            } else {
                right.update(x);
            }
        }
        left.merge(&right);
        assert_eq!(left.count(), whole.count());
        assert!((left.mean() - whole.mean()).abs() < 1e-12);
        assert!((left.variance() - whole.variance()).abs() < 1e-12);
        assert!((left.skewness() - whole.skewness()).abs() < 1e-9);
        assert!((left.kurtosis() - whole.kurtosis()).abs() < 1e-9);
        assert_eq!(left.min(), whole.min());
        assert_eq!(left.max(), whole.max());
    }

    #[test]
    fn merging_into_empty_or_from_empty_is_identity() {
        let a = StreamingStats::from_slice(&[1.0, 2.0, 3.0]);
        let mut empty = StreamingStats::new();
        empty.merge(&a);
        assert!((empty.mean() - 2.0).abs() < 1e-12);
        let mut a2 = a.clone();
        let b = StreamingStats::new();
        a2.merge(&b);
        assert!((a2.mean() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn correlation_is_one_for_linear_data() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((correlation(&x, &y) - 1.0).abs() < 1e-12);
        let yneg = [-2.0, -4.0, -6.0, -8.0, -10.0];
        assert!((correlation(&x, &yneg) + 1.0).abs() < 1e-12);
        let flat = [3.0, 3.0, 3.0, 3.0, 3.0];
        assert!(correlation(&x, &flat).is_nan());
    }

    #[test]
    fn handles_non_finite_inputs() {
        let mut s = StreamingStats::new();
        s.update(1.0);
        s.update(f64::NAN);
        s.update(f64::INFINITY);
        s.update(3.0);
        assert_eq!(s.count(), 2);
        assert_eq!(s.min(), 1.0);
        assert_eq!(s.max(), 3.0);
    }
}
