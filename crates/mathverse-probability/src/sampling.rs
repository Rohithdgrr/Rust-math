//! Sampling methods: rejection sampling, importance sampling, stratified sampling, bootstrap, jackknife.

use crate::rng::Rng;

/// Rejection sampling.
pub struct RejectionSampling;

impl RejectionSampling {
    /// Sample from target distribution using rejection sampling.
    pub fn sample<F1, F2>(
        target_pdf: F1,
        proposal_pdf: F2,
        proposal_sample: impl Fn(&mut Rng) -> f64,
        m: f64,
        n_samples: usize,
        rng: &mut Rng,
    ) -> Vec<f64>
    where
        F1: Fn(f64) -> f64,
        F2: Fn(f64) -> f64,
    {
        let mut samples = Vec::new();
        
        while samples.len() < n_samples {
            let x = proposal_sample(rng);
            let u = rng.uniform();
            
            if u < target_pdf(x) / (m * proposal_pdf(x)) {
                samples.push(x);
            }
        }
        
        samples
    }

    /// Adaptive rejection sampling (simplified).
    pub fn adaptive_sample(
        target_log_pdf: impl Fn(f64) -> f64,
        initial_bounds: (f64, f64),
        n_samples: usize,
        rng: &mut Rng,
    ) -> Vec<f64> {
        let mut samples = Vec::new();
        let (a, b) = initial_bounds;
        
        // Simplified: use uniform proposal within bounds
        for _ in 0..n_samples {
            let x = a + (b - a) * rng.uniform();
            let log_target = target_log_pdf(x);
            let log_proposal = ((b - a).ln()).neg();
            
            let u = rng.uniform();
            if u.ln() < log_target - log_proposal {
                samples.push(x);
            }
        }
        
        samples
    }
}

/// Importance sampling.
pub struct ImportanceSampling;

impl ImportanceSampling {
    /// Estimate expectation using importance sampling.
    pub fn estimate<F>(
        target_log_pdf: impl Fn(f64) -> f64,
        proposal_log_pdf: impl Fn(f64) -> f64,
        proposal_sample: impl Fn(&mut Rng) -> f64,
        integrand: F,
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64)
    where
        F: Fn(f64) -> f64,
    {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..n_samples {
            let x = proposal_sample(rng);
            let log_weight = target_log_pdf(x) - proposal_log_pdf(x);
            let weight = log_weight.exp();
            let value = integrand(x) * weight;
            
            sum += value;
            sum_sq += value * value;
        }
        
        let mean = sum / n_samples as f64;
        let variance = (sum_sq / n_samples as f64 - mean * mean) / n_samples as f64;
        
        (mean, variance.sqrt())
    }

    /// Self-normalized importance sampling.
    pub fn self_normalized<F>(
        target_log_pdf: impl Fn(f64) -> f64,
        proposal_log_pdf: impl Fn(f64) -> f64,
        proposal_sample: impl Fn(&mut Rng) -> f64,
        integrand: F,
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64)
    where
        F: Fn(f64) -> f64,
    {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut weighted_sum_sq = 0.0;
        
        for _ in 0..n_samples {
            let x = proposal_sample(rng);
            let log_weight = target_log_pdf(x) - proposal_log_pdf(x);
            let weight = log_weight.exp();
            
            weighted_sum += integrand(x) * weight;
            weight_sum += weight;
            weighted_sum_sq += (integrand(x) * weight).powi(2);
        }
        
        let mean = weighted_sum / weight_sum;
        let variance = (weighted_sum_sq / weight_sum - mean * mean) / n_samples as f64;
        
        (mean, variance.sqrt())
    }
}

/// Stratified sampling.
pub struct StratifiedSampling;

impl StratifiedSampling {
    /// Stratified sampling in 1D.
    pub fn sample<F>(
        integrand: F,
        a: f64,
        b: f64,
        n_strata: usize,
        samples_per_stratum: usize,
        rng: &mut Rng,
    ) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let stratum_width = (b - a) / n_strata as f64;
        let mut sum = 0.0;
        
        for i in 0..n_strata {
            let stratum_start = a + i as f64 * stratum_width;
            let mut stratum_sum = 0.0;
            
            for _ in 0..samples_per_stratum {
                let x = stratum_start + stratum_width * rng.uniform();
                stratum_sum += integrand(x);
            }
            
            sum += stratum_sum / samples_per_stratum as f64;
        }
        
        sum * stratum_width
    }

    /// Latin hypercube sampling.
    pub fn latin_hypercube(
        dim: usize,
        n_samples: usize,
        rng: &mut Rng,
    ) -> Vec<Vec<f64>> {
        let mut samples = vec![vec![0.0; dim]; n_samples];
        
        for d in 0..dim {
            let mut perm: Vec<usize> = (0..n_samples).collect();
            // Fisher-Yates shuffle
            for i in (1..n_samples).rev() {
                let j = rng.below(i as u64 + 1) as usize;
                perm.swap(i, j);
            }
            
            for i in 0..n_samples {
                samples[i][d] = (perm[i] as f64 + rng.uniform()) / n_samples as f64;
            }
        }
        
        samples
    }
}

/// Resampling methods.
pub struct Resampling;

impl Resampling {
    /// Bootstrap resampling.
    pub fn bootstrap(data: &[f64], n_bootstrap: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        let n = data.len();
        let mut bootstrap_samples = Vec::new();
        
        for _ in 0..n_bootstrap {
            let mut sample = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = rng.below(n as u64) as usize;
                sample.push(data[idx]);
            }
            bootstrap_samples.push(sample);
        }
        
        bootstrap_samples
    }

    /// Bootstrap confidence interval.
    pub fn bootstrap_ci(
        data: &[f64],
        statistic: impl Fn(&[f64]) -> f64,
        alpha: f64,
        n_bootstrap: usize,
        rng: &mut Rng,
    ) -> (f64, f64) {
        let bootstrap_stats: Vec<f64> = Self::bootstrap(data, n_bootstrap, rng)
            .iter()
            .map(|sample| statistic(sample))
            .collect();
        
        let mut sorted = bootstrap_stats.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
        
        (sorted[lower_idx], sorted[upper_idx.min(n_bootstrap - 1)])
    }

    /// Jackknife resampling.
    pub fn jackknife(data: &[f64], statistic: impl Fn(&[f64]) -> f64) -> Vec<f64> {
        let n = data.len();
        let mut jackknife_stats = Vec::new();
        
        for i in 0..n {
            let sample: Vec<f64> = data.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &x)| x)
                .collect();
            jackknife_stats.push(statistic(&sample));
        }
        
        jackknife_stats
    }

    /// Jackknife estimate of variance.
    pub fn jackknife_variance(data: &[f64], statistic: impl Fn(&[f64]) -> f64) -> f64 {
        let jackknife_stats = Self::jackknife(data, statistic);
        let n = data.len() as f64;
        let mean: f64 = jackknife_stats.iter().sum::<f64>() / jackknife_stats.len() as f64;
        
        let variance: f64 = jackknife_stats.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() * (n - 1.0) / n;
        
        variance
    }
}

/// Variance reduction techniques.
pub struct VarianceReduction;

impl VarianceReduction {
    /// Antithetic variates.
    pub fn antithetic_variates<F>(
        integrand: F,
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64)
    where
        F: Fn(f64) -> f64,
    {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..n_samples {
            let u = rng.uniform();
            let x1 = integrand(u);
            let x2 = integrand(1.0 - u);
            let avg = (x1 + x2) / 2.0;
            
            sum += avg;
            sum_sq += avg * avg;
        }
        
        let mean = sum / n_samples as f64;
        let variance = (sum_sq / n_samples as f64 - mean * mean) / n_samples as f64;
        
        (mean, variance.sqrt())
    }

    /// Control variates.
    pub fn control_variates<F, G>(
        integrand: F,
        control: G,
        control_mean: f64,
        n_samples: usize,
        rng: &mut Rng,
    ) -> (f64, f64)
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
    {
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_y_sq = 0.0;
        
        for _ in 0..n_samples {
            let u = rng.uniform();
            let x = integrand(u);
            let y = control(u);
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_y_sq += y * y;
        }
        
        let mean_x = sum_x / n_samples as f64;
        let mean_y = sum_y / n_samples as f64;
        
        // Compute optimal coefficient c
        let cov_xy = (sum_xy / n_samples as f64 - mean_x * mean_y);
        let var_y = (sum_y_sq / n_samples as f64 - mean_y * mean_y);
        let c = if var_y > 0.0 { cov_xy / var_y } else { 0.0 };
        
        // Controlled estimate
        let controlled_mean = mean_x - c * (mean_y - control_mean);
        
        // Estimate variance (simplified)
        let variance = var_y * (1.0 - c * c) / n_samples as f64;
        
        (controlled_mean, variance.sqrt())
    }
}

/// Sequential Monte Carlo (Particle Filter).
pub struct ParticleFilter<T> {
    pub particles: Vec<T>,
    pub weights: Vec<f64>,
}

impl<T: Clone> ParticleFilter<T> {
    pub fn new(initial_particles: Vec<T>) -> Self {
        let n = initial_particles.len();
        ParticleFilter {
            particles: initial_particles,
            weights: vec![1.0 / n as f64; n],
        }
    }

    /// Resample particles based on weights.
    pub fn resample(&mut self, rng: &mut Rng) {
        let n = self.particles.len();
        let mut new_particles = Vec::with_capacity(n);
        let mut cumulative = Vec::with_capacity(n);
        
        let mut acc = 0.0;
        for &w in &self.weights {
            acc += w;
            cumulative.push(acc);
        }
        
        for _ in 0..n {
            let u = rng.uniform();
            let idx = cumulative.binary_search_by(|&x| x.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|i| i);
            new_particles.push(self.particles[idx.min(n - 1)].clone());
        }
        
        self.particles = new_particles;
        self.weights = vec![1.0 / n as f64; n];
    }

    /// Update particle weights.
    pub fn update_weights(&mut self, new_weights: Vec<f64>) {
        let n = self.weights.len();
        for i in 0..n {
            self.weights[i] *= new_weights[i];
        }
        
        // Normalize weights
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    /// Effective sample size.
    pub fn ess(&self) -> f64 {
        let sum_sq: f64 = self.weights.iter().map(|&w| w * w).sum();
        if sum_sq > 0.0 {
            1.0 / sum_sq
        } else {
            0.0
        }
    }
}

/// Quasi-Monte Carlo sequences.
pub struct QuasiMonteCarlo;

impl QuasiMonteCarlo {
    /// Sobol sequence (simplified 1D).
    pub fn sobol_1d(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i + 1) as f64 / n as f64).collect()
    }

    /// Halton sequence.
    pub fn halton(dim: usize, n: usize) -> Vec<Vec<f64>> {
        let mut samples = Vec::new();
        let mut bases = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
        
        for i in 0..n {
            let mut point = Vec::new();
            for d in 0..dim {
                let base = bases[d % bases.len()];
                point.push(Self::halton_number(i, base));
            }
            samples.push(point);
        }
        
        samples
    }

    fn halton_number(index: usize, base: usize) -> f64 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f64;
        let mut i = index;
        
        while i > 0 {
            result += f * (i % base) as f64;
            i /= base;
            f /= base as f64;
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratified_sampling() {
        let mut rng = Rng::new(42);
        let result = StratifiedSampling::sample(
            |x| x * x,
            0.0,
            1.0,
            10,
            100,
            &mut rng,
        );
        assert!((result - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_bootstrap() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng = Rng::new(42);
        let bootstrap_samples = Resampling::bootstrap(&data, 100, &mut rng);
        assert_eq!(bootstrap_samples.len(), 100);
        assert_eq!(bootstrap_samples[0].len(), 5);
    }

    #[test]
    fn test_latin_hypercube() {
        let mut rng = Rng::new(42);
        let samples = StratifiedSampling::latin_hypercube(3, 100, &mut rng);
        assert_eq!(samples.len(), 100);
        assert_eq!(samples[0].len(), 3);
    }

    #[test]
    fn test_antithetic_variates() {
        let mut rng = Rng::new(42);
        let (mean, _err) = VarianceReduction::antithetic_variates(
            |x| x,
            1000,
            &mut rng,
        );
        assert!((mean - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_particle_filter() {
        let particles = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut pf = ParticleFilter::new(particles);
        let mut rng = Rng::new(42);
        
        pf.update_weights(vec![0.1, 0.2, 0.3, 0.2, 0.2]);
        pf.resample(&mut rng);
        
        assert_eq!(pf.particles.len(), 5);
    }
}
