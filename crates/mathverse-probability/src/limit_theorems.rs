//! Limit theorems: Law of Large Numbers, Central Limit Theorem, convergence types, large deviations.

use crate::rng::Rng;

/// Law of Large Numbers (Weak and Strong).
pub struct LawOfLargeNumbers;

impl LawOfLargeNumbers {
    /// Weak Law of Large Numbers: sample mean converges to expected value.
    pub fn weak_law_check(
        samples: &[f64],
        expected_mean: f64,
        tolerance: f64,
    ) -> bool {
        let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        (sample_mean - expected_mean).abs() < tolerance
    }

    /// Strong Law of Large Numbers: almost sure convergence check.
    pub fn strong_law_check(
        sample_means: &[f64],
        expected_mean: f64,
        tolerance: f64,
    ) -> bool {
        // Check if sample means converge to expected value
        for &mean in sample_means {
            if (mean - expected_mean).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Simulate convergence of sample means.
    pub fn simulate_convergence<F>(
        sampler: F,
        expected_mean: f64,
        max_n: usize,
        rng: &mut Rng,
    ) -> Vec<f64>
    where
        F: Fn(&mut Rng) -> f64,
    {
        let mut sample_means = Vec::new();
        let mut sum = 0.0;
        
        for n in 1..=max_n {
            sum += sampler(rng);
            sample_means.push(sum / n as f64);
        }
        
        sample_means
    }
}

/// Central Limit Theorem.
pub struct CentralLimitTheorem;

impl CentralLimitTheorem {
    /// Check if sample distribution approximates normal.
    pub fn normality_test(samples: &[f64]) -> (f64, f64) {
        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = variance.sqrt();
        
        // Standardize samples
        let standardized: Vec<f64> = samples.iter()
            .map(|&x| (x - mean) / std)
            .collect();
        
        // Compute skewness and kurtosis
        let skewness = standardized.iter().map(|&x| x.powi(3)).sum::<f64>() / n as f64;
        let kurtosis = standardized.iter().map(|&x| x.powi(4)).sum::<f64>() / n as f64 - 3.0;
        
        (skewness, kurtosis)
    }

    /// Simulate CLT convergence.
    pub fn simulate_clt<F>(
        sampler: F,
        sample_size: usize,
        n_samples: usize,
        rng: &mut Rng,
    ) -> Vec<f64>
    where
        F: Fn(&mut Rng) -> f64,
    {
        let mut sample_means = Vec::new();
        
        for _ in 0..n_samples {
            let mut sum = 0.0;
            for _ in 0..sample_size {
                sum += sampler(rng);
            }
            sample_means.push(sum / sample_size as f64);
        }
        
        sample_means
    }

    /// Standardized sample means (should be approximately N(0,1)).
    pub fn standardized_means(
        sample_means: &[f64],
        population_mean: f64,
        population_std: f64,
        sample_size: usize,
    ) -> Vec<f64> {
        sample_means.iter()
            .map(|&x| (x - population_mean) / (population_std / (sample_size as f64).sqrt()))
            .collect()
    }
}

/// Convergence types.
pub enum ConvergenceType {
    InProbability,
    AlmostSure,
    InDistribution,
    InMean,
}

impl ConvergenceType {
    /// Check convergence in probability.
    pub fn in_probability(
        sequence: &[f64],
        limit: f64,
        epsilon: f64,
        tolerance: f64,
    ) -> bool {
        let n = sequence.len();
        let count = sequence.iter()
            .filter(|&&x| (x - limit).abs() > epsilon)
            .count();
        
        (count as f64 / n as f64) < tolerance
    }

    /// Check almost sure convergence.
    pub fn almost_sure(
        sequence: &[f64],
        limit: f64,
        epsilon: f64,
    ) -> bool {
        // Almost sure convergence: P(lim X_n = X) = 1
        // Check if sequence eventually stays within epsilon
        let mut consecutive_within = 0;
        let threshold = sequence.len() / 10;
        
        for &x in sequence {
            if (x - limit).abs() < epsilon {
                consecutive_within += 1;
            } else {
                consecutive_within = 0;
            }
            
            if consecutive_within > threshold {
                return true;
            }
        }
        
        false
    }

    /// Check convergence in distribution (via CDF comparison).
    pub fn in_distribution(
        sequence: &[f64],
        target_cdf: impl Fn(f64) -> f64,
        tolerance: f64,
    ) -> bool {
        let mut sorted = sequence.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted.len();
        for i in 0..n {
            let empirical_cdf = (i + 1) as f64 / n as f64;
            let target = target_cdf(sorted[i]);
            
            if (empirical_cdf - target).abs() > tolerance {
                return false;
            }
        }
        
        true
    }

    /// Check convergence in mean (L1 convergence).
    pub fn in_mean(
        sequence: &[f64],
        limit: f64,
        tolerance: f64,
    ) -> bool {
        let mean_abs_diff: f64 = sequence.iter()
            .map(|&x| (x - limit).abs())
            .sum::<f64>() / sequence.len() as f64;
        
        mean_abs_diff < tolerance
    }
}

/// Large deviation theory.
pub struct LargeDeviations;

impl LargeDeviations {
    /// Cramér's theorem approximation (simplified).
    pub fn cramers_bound(
        sample_mean: f64,
        true_mean: f64,
        rate_function: impl Fn(f64) -> f64,
        n: usize,
    ) -> f64 {
        let x = sample_mean;
        let mu = true_mean;
        let rate = rate_function(x);
        
        // P(|X_n - mu| >= |x - mu|) ≈ exp(-n * rate)
        (-n as f64 * rate).exp()
    }

    /// Chernoff bound.
    pub fn chernoff_bound(
        sample_mean: f64,
        true_mean: f64,
        moment_generating: impl Fn(f64) -> f64,
        n: usize,
    ) -> f64 {
        let t = if sample_mean > true_mean { 0.1 } else { -0.1 };
        let mgf_t = moment_generating(t);
        let bound = (mgf_t * (-t * true_mean).exp()).powf(n as f64);
        
        bound
    }

    /// Rate function for Bernoulli distribution.
    pub fn bernoulli_rate(p: f64, x: f64) -> f64 {
        if x < 0.0 || x > 1.0 {
            return f64::INFINITY;
        }
        
        let kl = if p > 0.0 && x > 0.0 {
            x * (x / p).ln()
        } else {
            0.0
        } + if p < 1.0 && x < 1.0 {
            (1.0 - x) * ((1.0 - x) / (1.0 - p)).ln()
        } else {
            0.0
        };
        
        kl
    }

    /// Rate function for Normal distribution.
    pub fn normal_rate(mu: f64, sigma: f64, x: f64) -> f64 {
        0.5 * ((x - mu) / sigma).powi(2)
    }
}

/// Berry-Esseen theorem (rate of convergence in CLT).
pub struct BerryEsseen;

impl BerryEsseen {
    /// Berry-Esseen bound.
    pub fn bound(
        third_absolute_moment: f64,
        variance: f64,
        n: usize,
    ) -> f64 {
        let c = 0.4748; // Optimal constant
        let sigma = variance.sqrt();
        
        c * third_absolute_moment / (sigma.powi(3) * (n as f64).sqrt())
    }

    /// Compute third absolute central moment.
    pub fn third_absolute_moment(samples: &[f64]) -> f64 {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        samples.iter()
            .map(|&x| (x - mean).abs().powi(3))
            .sum::<f64>() / samples.len() as f64
    }

    /// Check if CLT approximation is within Berry-Esseen bound.
    pub fn check_approximation(
        samples: &[f64],
        target_cdf: impl Fn(f64) -> f64,
    ) -> bool {
        let third_moment = Self::third_absolute_moment(samples);
        let variance = samples.iter().map(|&x| x * x).sum::<f64>() / samples.len() as f64;
        let n = samples.len();
        
        let bound = Self::bound(third_moment, variance, n);
        
        // Check maximum difference between empirical and target CDF
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut max_diff = 0.0;
        for &x in &sorted {
            let empirical = sorted.iter().filter(|&&y| y <= x).count() as f64 / n as f64;
            let target = target_cdf(x);
            max_diff = max_diff.max((empirical - target).abs());
        }
        
        max_diff < bound
    }
}

/// Renewal theory.
pub struct RenewalTheory;

impl RenewalTheory {
    /// Elementary renewal theorem.
    pub fn elementary_renewal_rate(
        interarrival_times: &[f64],
        time_horizon: f64,
    ) -> f64 {
        let mean_interarrival = interarrival_times.iter().sum::<f64>() / interarrival_times.len() as f64;
        
        if mean_interarrival > 0.0 {
            time_horizon / mean_interarrival
        } else {
            0.0
        }
    }

    /// Renewal function (expected number of renewals by time t).
    pub fn renewal_function(
        interarrival_times: &[f64],
        max_time: f64,
        dt: f64,
    ) -> Vec<f64> {
        let mut renewal_counts = Vec::new();
        let mut current_time = 0.0;
        let mut count = 0;
        let mut idx = 0;
        
        while current_time <= max_time {
            renewal_counts.push(count);
            current_time += dt;
            
            while idx < interarrival_times.len() && current_time >= interarrival_times[idx] {
                count += 1;
                idx += 1;
            }
        }
        
        renewal_counts
    }

    /// Key renewal theorem (simplified).
    pub fn key_renewal_theorem(
        renewal_function: &[f64],
        mean_interarrival: f64,
        dt: f64,
    ) -> f64 {
        if renewal_function.is_empty() || mean_interarrival == 0.0 {
            return 0.0;
        }
        
        // Long-run average renewal rate
        let final_count = renewal_function.last().unwrap();
        let total_time = renewal_function.len() as f64 * dt;
        
        final_count / total_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weak_law() {
        let samples: Vec<f64> = (0..1000).map(|_| 0.5).collect();
        assert!(LawOfLargeNumbers::weak_law_check(&samples, 0.5, 0.01));
    }

    #[test]
    fn test_clt_normality() {
        let mut rng = Rng::new(42);
        let samples = CentralLimitTheorem::simulate_clt(
            |r| r.uniform(),
            30,
            1000,
            &mut rng,
        );
        
        let (skewness, kurtosis) = CentralLimitTheorem::normality_test(&samples);
        // For normal distribution, skewness ≈ 0, kurtosis ≈ 0
        assert!(skewness.abs() < 0.5);
        assert!(kurtosis.abs() < 1.0);
    }

    #[test]
    fn test_convergence_in_probability() {
        let sequence: Vec<f64> = (0..1000).map(|i| 0.5 + 1.0 / (i + 1) as f64).collect();
        assert!(ConvergenceType::in_probability(&sequence, 0.5, 0.1, 0.1));
    }

    #[test]
    fn test_bernoulli_rate() {
        let rate = LargeDeviations::bernoulli_rate(0.5, 0.6);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_berry_esseen() {
        let samples: Vec<f64> = (0..100).map(|_| 0.5).collect();
        let third_moment = BerryEsseen::third_absolute_moment(&samples);
        assert_eq!(third_moment, 0.0);
    }

    #[test]
    fn test_renewal_theory() {
        let interarrival = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let rate = RenewalTheory::elementary_renewal_rate(&interarrival, 10.0);
        assert!((rate - 10.0).abs() < 0.1);
    }
}
