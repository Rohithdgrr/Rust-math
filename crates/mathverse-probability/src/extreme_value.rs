//! Extreme value theory: Gumbel, Fréchet, Weibull distributions, return periods, peak over threshold.

use crate::F64Ext;

/// Gumbel distribution (Type I extreme value distribution).
#[must_use]
pub struct GumbelEVD {
    pub location: f64,
    pub scale: f64,
}

impl GumbelEVD {
    pub fn new(location: f64, scale: f64) -> Result<Self, String> {
        if scale <= 0.0 {
            return Err("Scale must be positive".to_string());
        }

        Ok(GumbelEVD { location, scale })
    }

    /// PDF of Gumbel distribution.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        (-z - (-z).exp()).exp() / self.scale
    }

    /// CDF of Gumbel distribution.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.location) / self.scale;
        (-(-z).exp()).exp()
    }

    /// Quantile function (inverse CDF).
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }
        self.location - self.scale * (-(-p.ln()).ln())
    }

    /// Mean of Gumbel distribution.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.location + self.scale * 0.577_215_664_901_532_9_f64
    }

    /// Variance of Gumbel distribution.
    #[must_use]
    pub fn variance(&self) -> f64 {
        (core::f64::consts::PI * core::f64::consts::PI / 6.0) * self.scale * self.scale
    }
}

/// Fréchet distribution (Type II extreme value distribution).
#[must_use]
pub struct FrechetEVD {
    pub location: f64,
    pub scale: f64,
    pub shape: f64,
}

impl FrechetEVD {
    pub fn new(location: f64, scale: f64, shape: f64) -> Result<Self, String> {
        if scale <= 0.0 || shape <= 0.0 {
            return Err("Scale and shape must be positive".to_string());
        }

        Ok(FrechetEVD {
            location,
            scale,
            shape,
        })
    }

    /// PDF of Fréchet distribution.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= self.location {
            return 0.0;
        }

        let z = (x - self.location) / self.scale;
        let alpha = self.shape;

        (alpha / self.scale) * z.powf(-alpha - 1.0) * (-z.powf(-alpha)).exp()
    }

    /// CDF of Fréchet distribution.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= self.location {
            return 0.0;
        }

        let z = (x - self.location) / self.scale;
        (-z.powf(-self.shape)).exp()
    }

    /// Quantile function.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }
        self.location + self.scale * (-p.ln()).powf(-1.0 / self.shape)
    }

    /// Mean (exists only for shape > 1).
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.shape > 1.0 {
            let gamma = (1.0 - 1.0 / self.shape).gamma();
            self.location + self.scale * gamma
        } else {
            f64::INFINITY
        }
    }

    /// Variance (exists only for shape > 2).
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.shape > 2.0 {
            let gamma1 = (1.0 - 1.0 / self.shape).gamma();
            let gamma2 = (1.0 - 2.0 / self.shape).gamma();
            self.scale * self.scale * (gamma2 - gamma1 * gamma1)
        } else {
            f64::INFINITY
        }
    }
}

/// Weibull distribution (Type III extreme value distribution).
#[must_use]
pub struct WeibullEVD {
    pub location: f64,
    pub scale: f64,
    pub shape: f64,
}

impl WeibullEVD {
    pub fn new(location: f64, scale: f64, shape: f64) -> Result<Self, String> {
        if scale <= 0.0 || shape <= 0.0 {
            return Err("Scale and shape must be positive".to_string());
        }

        Ok(WeibullEVD {
            location,
            scale,
            shape,
        })
    }

    /// PDF of Weibull EVD.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        if x >= self.location {
            return 0.0;
        }

        let z = (self.location - x) / self.scale;
        let alpha = self.shape;

        (alpha / self.scale) * z.powf(alpha - 1.0) * (-z.powf(alpha)).exp()
    }

    /// CDF of Weibull EVD.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        if x >= self.location {
            return 1.0;
        }

        let z = (self.location - x) / self.scale;
        (-(-z.powf(self.shape)).exp()).exp()
    }

    /// Quantile function.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }
        self.location - self.scale * (-(1.0 - p).ln()).powf(1.0 / self.shape)
    }

    /// Mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        let gamma = (1.0 + 1.0 / self.shape).gamma();
        self.location - self.scale * gamma
    }

    /// Variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        let g1 = (1.0 + 1.0 / self.shape).gamma();
        let g2 = (1.0 + 2.0 / self.shape).gamma();
        self.scale * self.scale * (g2 - g1 * g1)
    }
}

/// Generalized Extreme Value (GEV) distribution.
#[must_use]
pub struct GEVDistribution {
    pub location: f64,
    pub scale: f64,
    pub shape: f64, // ξ: shape parameter
}

impl GEVDistribution {
    pub fn new(location: f64, scale: f64, shape: f64) -> Result<Self, String> {
        if scale <= 0.0 {
            return Err("Scale must be positive".to_string());
        }

        Ok(GEVDistribution {
            location,
            scale,
            shape,
        })
    }

    /// PDF of GEV distribution.
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        let xi = self.shape;
        let mu = self.location;
        let sigma = self.scale;

        let z = if xi != 0.0 {
            1.0 + xi * (x - mu) / sigma
        } else {
            (x - mu) / sigma
        };

        if z <= 0.0 {
            return 0.0;
        }

        let t = if xi != 0.0 {
            z.powf(-1.0 / xi)
        } else {
            (-z).exp()
        };

        let exponent = if xi != 0.0 {
            -(1.0 + xi) * z.ln() - t
        } else {
            -z - t
        };

        exponent.exp() / sigma
    }

    /// CDF of GEV distribution.
    #[must_use]
    pub fn cdf(&self, x: f64) -> f64 {
        let xi = self.shape;
        let mu = self.location;
        let sigma = self.scale;

        let z = if xi != 0.0 {
            1.0 + xi * (x - mu) / sigma
        } else {
            (x - mu) / sigma
        };

        if z <= 0.0 {
            return if xi > 0.0 { 0.0 } else { 1.0 };
        }

        if xi != 0.0 {
            (-z.powf(-1.0 / xi)).exp()
        } else {
            (-z).exp()
        }
    }

    /// Quantile function.
    #[must_use]
    pub fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 || p >= 1.0 {
            return f64::NAN;
        }

        let xi = self.shape;
        let mu = self.location;
        let sigma = self.scale;

        let y = -(-p.ln()).ln();

        if xi != 0.0 {
            mu + sigma * (y.powf(-xi) - 1.0) / xi
        } else {
            mu - sigma * y
        }
    }

    /// Mean (exists for shape < 1).
    #[must_use]
    pub fn mean(&self) -> f64 {
        let xi = self.shape;
        let mu = self.location;
        let sigma = self.scale;

        if xi < 1.0 && xi != 0.0 {
            let gamma = (1.0 - xi).gamma();
            mu + sigma * (gamma - 1.0) / xi
        } else if xi == 0.0 {
            mu + sigma * 0.577_215_664_901_532_9_f64
        } else {
            f64::INFINITY
        }
    }

    /// Variance (exists for shape < 0.5).
    #[must_use]
    pub fn variance(&self) -> f64 {
        let xi = self.shape;
        let sigma = self.scale;

        if xi < 0.5 && xi != 0.0 {
            let gamma1 = (1.0 - 2.0 * xi).gamma();
            let gamma2 = (1.0 - xi).gamma();
            sigma * sigma * (gamma1 - gamma2 * gamma2) / (xi * xi)
        } else if xi == 0.0 {
            (core::f64::consts::PI * core::f64::consts::PI / 6.0) * sigma * sigma
        } else {
            f64::INFINITY
        }
    }
}

/// Return period analysis.
#[must_use]
pub struct ReturnPeriod;

impl ReturnPeriod {
    /// Return period for a given value.
    #[must_use]
    pub fn from_value(annual_exceedance_probability: f64) -> f64 {
        if annual_exceedance_probability <= 0.0 {
            return f64::INFINITY;
        }
        1.0 / annual_exceedance_probability
    }

    /// Annual exceedance probability for a given return period.
    #[must_use]
    pub fn from_period(return_period: f64) -> f64 {
        if return_period <= 0.0 {
            return 1.0;
        }
        1.0 / return_period
    }

    /// Return level for a given return period using GEV.
    #[must_use]
    pub fn return_level(gev: &GEVDistribution, return_period: f64) -> f64 {
        let p = 1.0 - 1.0 / return_period;
        gev.quantile(p)
    }

    /// Return period for a given value using GEV.
    #[must_use]
    pub fn return_period_for_value(gev: &GEVDistribution, value: f64) -> f64 {
        let p = gev.cdf(value);
        if p >= 1.0 {
            return f64::INFINITY;
        }
        1.0 / (1.0 - p)
    }
}

/// Peak Over Threshold (POT) method.
#[must_use]
pub struct PeakOverThreshold {
    pub threshold: f64,
    pub data: Vec<f64>,
}

impl PeakOverThreshold {
    #[must_use]
    pub fn new(threshold: f64, data: Vec<f64>) -> Self {
        let excesses: Vec<f64> = data
            .into_iter()
            .filter(|&x| x > threshold)
            .map(|x| x - threshold)
            .collect();

        PeakOverThreshold {
            threshold,
            data: excesses,
        }
    }

    /// Fit Generalized Pareto Distribution (GPD) to excesses.
    pub fn fit_gpd(&self) -> Result<(f64, f64), String> {
        if self.data.is_empty() {
            return Err("No data above threshold".to_string());
        }

        // Simplified method of moments estimation
        let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
        let variance = self.data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (self.data.len() - 1) as f64;

        let shape =
            -0.5 * (1.0 - mean * mean / variance).signum() * (1.0 - mean * mean / variance).sqrt();

        let scale = mean * (1.0 - 2.0 * shape) / (1.0 - shape);

        Ok((scale, shape))
    }

    /// Number of exceedances per year.
    #[must_use]
    pub fn exceedance_rate(&self, total_years: f64) -> f64 {
        self.data.len() as f64 / total_years
    }

    /// Return level using POT method.
    #[must_use]
    pub fn return_level(
        &self,
        scale: f64,
        shape: f64,
        exceedance_rate: f64,
        return_period: f64,
    ) -> f64 {
        let n = exceedance_rate * return_period;
        let p = 1.0 - 1.0 / n;

        let quantile = if shape != 0.0 {
            scale * (p.powf(-shape) - 1.0) / shape
        } else {
            -scale * p.ln()
        };

        self.threshold + quantile
    }
}

/// Block maxima method.
#[must_use]
pub struct BlockMaxima {
    pub block_size: usize,
    pub data: Vec<f64>,
}

impl BlockMaxima {
    #[must_use]
    pub fn new(block_size: usize, data: Vec<f64>) -> Self {
        BlockMaxima { block_size, data }
    }

    /// Compute block maxima.
    #[must_use]
    pub fn compute(&self) -> Vec<f64> {
        let n_blocks = self.data.len().div_ceil(self.block_size);
        let mut maxima = Vec::with_capacity(n_blocks);

        for i in 0..n_blocks {
            let start = i * self.block_size;
            let end = (start + self.block_size).min(self.data.len());

            if start < self.data.len() {
                let block_max = self.data[start..end]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                maxima.push(block_max);
            }
        }

        maxima
    }

    /// Fit GEV to block maxima using method of moments (simplified).
    pub fn fit_gev(&self) -> Result<(f64, f64, f64), String> {
        let maxima = self.compute();
        if maxima.len() < 3 {
            return Err("Insufficient data".to_string());
        }

        let mean = maxima.iter().sum::<f64>() / maxima.len() as f64;
        let variance =
            maxima.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (maxima.len() - 1) as f64;

        let skewness = maxima.iter().map(|&x| (x - mean).powi(3)).sum::<f64>()
            / maxima.len() as f64
            / variance.powf(1.5);

        // Simplified parameter estimation
        let shape = -0.1 * skewness.signum(); // Very rough approximation
        let scale = variance.sqrt();
        let location = mean - scale * 0.577_215_664_901_532_9_f64;

        Ok((location, scale, shape))
    }
}

/// Extreme value indices.
#[must_use]
pub struct ExtremeValueIndices {
    pub data: Vec<f64>,
}

impl ExtremeValueIndices {
    #[must_use]
    pub fn new(data: Vec<f64>) -> Self {
        ExtremeValueIndices { data }
    }

    /// r-largest order statistics.
    #[must_use]
    pub fn r_largest(&self, r: usize) -> Vec<Vec<f64>> {
        let n = self.data.len();
        let block_size = n / r;
        let mut result = Vec::new();

        for i in 0..r {
            let start = i * block_size;
            let end = if i < r - 1 { (i + 1) * block_size } else { n };

            let mut block = self.data[start..end].to_vec();
            block.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let r_largest: Vec<f64> = block.into_iter().rev().take(r).collect();
            result.push(r_largest);
        }

        result
    }

    /// Declustering of exceedances.
    #[must_use]
    pub fn decluster(&self, threshold: f64, min_gap: usize) -> Vec<f64> {
        let mut clusters = Vec::new();
        let mut current_cluster = Vec::new();
        let mut last_exceedance_idx = None;

        for (i, &x) in self.data.iter().enumerate() {
            if x > threshold {
                if let Some(last_idx) = last_exceedance_idx {
                    if i - last_idx >= min_gap && !current_cluster.is_empty() {
                        clusters.push(
                            current_cluster
                                .iter()
                                .cloned()
                                .fold(f64::NEG_INFINITY, f64::max),
                        );
                        current_cluster.clear();
                    }
                }
                current_cluster.push(x);
                last_exceedance_idx = Some(i);
            }
        }

        if !current_cluster.is_empty() {
            clusters.push(
                current_cluster
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max),
            );
        }

        clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gumbel_evd() {
        let gev = GumbelEVD::new(0.0, 1.0).unwrap();
        let mean = gev.mean();
        assert!((mean - 0.5772156649015329).abs() < 1e-10);

        let cdf = gev.cdf(0.0);
        assert!((cdf - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_frechet_evd() {
        let gev = FrechetEVD::new(0.0, 1.0, 2.0).unwrap();
        let mean = gev.mean();
        assert!(mean.is_finite());
    }

    #[test]
    fn test_weibull_evd() {
        let gev = WeibullEVD::new(1.0, 1.0, 2.0).unwrap();
        let mean = gev.mean();
        assert!(mean < 1.0);
    }

    #[test]
    fn test_gev_distribution() {
        let gev = GEVDistribution::new(0.0, 1.0, 0.0).unwrap();
        let mean = gev.mean();
        assert!((mean - 0.5772156649015329).abs() < 1e-10);
    }

    #[test]
    fn test_return_period() {
        let rp = ReturnPeriod::from_value(0.1);
        assert!((rp - 10.0).abs() < 1e-10);

        let aep = ReturnPeriod::from_period(10.0);
        assert!((aep - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_pot() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let pot = PeakOverThreshold::new(5.0, data);
        assert_eq!(pot.data.len(), 5);
    }

    #[test]
    fn test_block_maxima() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let bm = BlockMaxima::new(10, data);
        let maxima = bm.compute();
        assert_eq!(maxima.len(), 10);
    }
}
