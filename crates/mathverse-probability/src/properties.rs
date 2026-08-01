//! Distribution properties: moments, quantiles, skewness, kurtosis, MGF, characteristic functions.

use crate::distributions::Distribution;

/// Extended distribution properties beyond basic moments.
pub trait DistributionProperties: Distribution {
    /// Skewness (third standardized moment).
    fn skewness(&self) -> f64;

    /// Excess kurtosis (fourth standardized moment minus 3).
    fn kurtosis(&self) -> f64;

    /// nth raw moment E[X^n].
    fn moment(&self, n: u32) -> f64;

    /// nth central moment E[(X - μ)^n].
    fn central_moment(&self, n: u32) -> f64;

    /// Quantile function (inverse CDF) for probability p.
    fn quantile(&self, p: f64) -> f64;

    /// Median (50th percentile).
    fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    /// Mode (most likely value).
    fn mode(&self) -> f64;

    /// Moment generating function M(t) = E[e^(tX)].
    fn mgf(&self, t: f64) -> f64;

    /// Characteristic function φ(t) = E[e^(itX)].
    fn characteristic_function(&self, t: f64) -> f64;
}

/// Helper for computing moments from samples.
pub struct SampleMoments {
    pub count: usize,
    pub mean: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl SampleMoments {
    /// Compute sample moments from data.
    pub fn from_samples(data: &[f64]) -> Self {
        let n = data.len();
        if n == 0 {
            return SampleMoments {
                count: 0,
                mean: f64::NAN,
                variance: f64::NAN,
                skewness: f64::NAN,
                kurtosis: f64::NAN,
            };
        }

        let mean = data.iter().sum::<f64>() / n as f64;

        let variance = if n > 1 {
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };

        let (skewness, kurtosis) = if n > 3 && variance > 0.0 {
            let m3 = data.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n as f64;
            let m4 = data.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n as f64;
            let std = variance.sqrt();
            let skew = m3 / (std.powi(3));
            let kurt = m4 / (std.powi(4)) - 3.0;
            (skew, kurt)
        } else {
            (0.0, 0.0)
        };

        SampleMoments {
            count: n,
            mean,
            variance,
            skewness,
            kurtosis,
        }
    }

    /// Compute percentile from sorted samples.
    #[must_use]
    pub fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return f64::NAN;
        }
        if p <= 0.0 {
            return sorted_data[0];
        }
        if p >= 1.0 {
            return sorted_data[sorted_data.len() - 1];
        }

        let n = sorted_data.len();
        let index = (p * (n - 1) as f64) as usize;
        let fraction = p * (n - 1) as f64 - index as f64;

        if index + 1 < n {
            sorted_data[index] * (1.0 - fraction) + sorted_data[index + 1] * fraction
        } else {
            sorted_data[index]
        }
    }

    /// Compute interquartile range (IQR).
    #[must_use]
    pub fn iqr(sorted_data: &[f64]) -> f64 {
        let q75 = Self::percentile(sorted_data, 0.75);
        let q25 = Self::percentile(sorted_data, 0.25);
        q75 - q25
    }
}

/// Order statistics for samples.
pub struct OrderStatistics;

impl OrderStatistics {
    /// Compute k-th order statistic (k-th smallest value, 1-indexed).
    pub fn kth_smallest(data: &mut [f64], k: usize) -> Option<f64> {
        if k == 0 || k > data.len() {
            return None;
        }
        Self::quickselect(data, k - 1, 0, data.len());
        Some(data[k - 1])
    }

    fn quickselect(data: &mut [f64], k: usize, left: usize, right: usize) {
        if left >= right {
            return;
        }

        let pivot_index = Self::partition(data, left, right);

        if k == pivot_index {
        } else if k < pivot_index {
            Self::quickselect(data, k, left, pivot_index);
        } else {
            Self::quickselect(data, k, pivot_index + 1, right);
        }
    }

    fn partition(data: &mut [f64], left: usize, right: usize) -> usize {
        let pivot = data[right - 1];
        let mut i = left;
        for j in left..right - 1 {
            if data[j] < pivot {
                data.swap(i, j);
                i += 1;
            }
        }
        data.swap(i, right - 1);
        i
    }

    /// Compute minimum and maximum.
    #[must_use]
    pub fn range(data: &[f64]) -> (f64, f64) {
        if data.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min, max)
    }
}

/// Percentile calculations using various methods.
pub enum PercentileMethod {
    Nearest,
    Linear,
    Midpoint,
    Lower,
    Higher,
}

impl PercentileMethod {
    pub fn compute(&self, sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return f64::NAN;
        }
        if p <= 0.0 {
            return sorted_data[0];
        }
        if p >= 1.0 {
            return sorted_data[sorted_data.len() - 1];
        }

        let n = sorted_data.len();
        match self {
            PercentileMethod::Nearest => {
                let index = (p * n as f64).round() as usize;
                sorted_data[index.min(n - 1)]
            }
            PercentileMethod::Linear => {
                let index = p * (n - 1) as f64;
                let lower = index.floor() as usize;
                let upper = (index.ceil() as usize).min(n - 1);
                let fraction = index - lower as f64;
                if lower == upper {
                    sorted_data[lower]
                } else {
                    sorted_data[lower] * (1.0 - fraction) + sorted_data[upper] * fraction
                }
            }
            PercentileMethod::Midpoint => {
                let index = p * (n - 1) as f64;
                let lower = index.floor() as usize;
                let upper = (index.ceil() as usize).min(n - 1);
                if lower == upper {
                    sorted_data[lower]
                } else {
                    (sorted_data[lower] + sorted_data[upper]) / 2.0
                }
            }
            PercentileMethod::Lower => {
                let index = (p * n as f64).floor() as usize;
                sorted_data[index.min(n - 1)]
            }
            PercentileMethod::Higher => {
                let index = (p * n as f64).ceil() as usize;
                sorted_data[index.min(n - 1)]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_moments() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let moments = SampleMoments::from_samples(&data);
        assert!((moments.mean - 3.0).abs() < 1e-10);
        assert!((moments.variance - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = SampleMoments::percentile(&data, 0.5);
        assert!((median - 5.5).abs() < 1e-10);

        let q25 = SampleMoments::percentile(&data, 0.25);
        assert!((q25 - 3.25).abs() < 1e-10);
    }

    #[test]
    fn test_order_statistics() {
        let mut data = vec![5.0, 3.0, 1.0, 4.0, 2.0];
        let min = OrderStatistics::kth_smallest(&mut data, 1).unwrap();
        assert!((min - 1.0).abs() < 1e-10);

        let max = OrderStatistics::kth_smallest(&mut data, 5).unwrap();
        assert!((max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_range() {
        let data = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        let (min, max) = OrderStatistics::range(&data);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_iqr() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let iqr = SampleMoments::iqr(&data);
        assert!((iqr - 4.5).abs() < 1e-10);
    }
}
