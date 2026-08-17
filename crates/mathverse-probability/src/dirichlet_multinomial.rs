//! Categorical, multinomial, Dirichlet and Dirichlet-multinomial
//! distributions: exact pmf/log-pmf and dependency-free sampling.

use crate::distributions::Gamma;
use crate::rng::Rng;
use crate::special::ln_gamma;

/// Discrete distribution over `k` categories with probabilities `probs`.
#[must_use]
#[derive(Clone, Debug)]
pub struct Categorical {
    pub probs: Vec<f64>,
}

impl Categorical {
    /// Validate that `probs` are non-negative, finite and sum to 1.
    pub fn new(probs: Vec<f64>) -> Result<Self, String> {
        let sum: f64 = probs.iter().sum();
        if probs.is_empty()
            || !sum.is_finite()
            || probs.iter().any(|&p| !p.is_finite() || p < 0.0)
            || (sum - 1.0).abs() > 1e-9
        {
            return Err("categorical probabilities must be finite, non-negative and sum to 1".into());
        }
        Ok(Self { probs })
    }

    /// Draw a category index in `0..k` (binary search on the CDF).
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.uniform();
        let mut acc = 0.0;
        for (i, &p) in self.probs.iter().enumerate() {
            acc += p;
            if u <= acc {
                return i;
            }
        }
        self.probs.len() - 1
    }

    /// Probability of category `k`.
    #[must_use]
    pub fn pmf(&self, k: usize) -> f64 {
        self.probs.get(k).copied().unwrap_or(0.0)
    }

    /// Log probability of category `k`.
    #[must_use]
    pub fn log_pmf(&self, k: usize) -> f64 {
        self.probs.get(k).map(|&p| p.ln()).unwrap_or(f64::NEG_INFINITY)
    }

    /// The probability vector itself.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        self.probs.clone()
    }
}

/// Multinomial distribution: `n` independent trials over categories with
/// probabilities `probs`.
#[must_use]
#[derive(Clone, Debug)]
pub struct Multinomial {
    pub n: u64,
    pub probs: Vec<f64>,
}

impl Multinomial {
    /// Validate `n > 0` and `probs` as in [`Categorical::new`].
    pub fn new(n: u64, probs: Vec<f64>) -> Result<Self, String> {
        if n == 0 {
            return Err("multinomial trials n must be positive".into());
        }
        let _ = Categorical::new(probs.clone())?;
        Ok(Self { n, probs })
    }

    /// Draw a count vector summing to `n` by tallying `n` categorical draws.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> Vec<u64> {
        let cat = Categorical {
            probs: self.probs.clone(),
        };
        let mut counts = vec![0u64; self.probs.len()];
        for _ in 0..self.n {
            counts[cat.sample(rng)] += 1;
        }
        counts
    }

    /// Exact log probability mass of `counts`:
    /// `ln(n!) - Σ ln(k_i!) + Σ k_i·ln(p_i)`.
    ///
    /// # Panics
    /// Panics if `counts.len() != probs.len()` or `counts` does not sum to `n`.
    #[must_use]
    pub fn log_pmf(&self, counts: &[u64]) -> f64 {
        assert_eq!(
            counts.len(),
            self.probs.len(),
            "multinomial counts and probs must have equal length"
        );
        let total: u64 = counts.iter().sum();
        assert_eq!(
            total, self.n,
            "multinomial counts must sum to the number of trials"
        );
        let mut log_p = ln_gamma(total as f64 + 1.0);
        for (&k, &p) in counts.iter().zip(&self.probs) {
            log_p -= ln_gamma(k as f64 + 1.0);
            if k > 0 {
                log_p += k as f64 * p.ln();
            }
        }
        log_p
    }

    /// Exact probability mass of `counts`.
    #[must_use]
    pub fn pmf(&self, counts: &[u64]) -> f64 {
        self.log_pmf(counts).exp()
    }

    /// Mean count vector `n·p`.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        self.probs.iter().map(|&p| p * self.n as f64).collect()
    }

    /// `Cov(counts_i, counts_j) = n·p_i·(δ_ij - p_j)`.
    #[must_use]
    pub fn covariance(&self, i: usize, j: usize) -> f64 {
        if i >= self.probs.len() || j >= self.probs.len() {
            return f64::NAN;
        }
        let pi = self.probs[i];
        let pj = self.probs[j];
        let delta = if i == j { 1.0 } else { 0.0 };
        self.n as f64 * pi * (delta - pj)
    }
}

/// Dirichlet distribution over the probability simplex with concentration
/// parameters `alphas`.
#[must_use]
#[derive(Clone, Debug)]
pub struct Dirichlet {
    pub alphas: Vec<f64>,
}

impl Dirichlet {
    /// Validate that all `alphas` are positive.
    pub fn new(alphas: Vec<f64>) -> Result<Self, String> {
        if alphas.is_empty()
            || alphas.iter().any(|&a| !a.is_finite() || a <= 0.0)
        {
            return Err("dirichlet alphas must be finite and positive".into());
        }
        Ok(Self { alphas })
    }

    /// Draw from the simplex: normalized independent gamma variates.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let mut draws = Vec::with_capacity(self.alphas.len());
        let mut sum = 0.0;
        for &a in &self.alphas {
            let g = Gamma {
                shape: a,
                rate: 1.0,
            }
            .sample(rng);
            draws.push(g);
            sum += g;
        }
        for x in &mut draws {
            *x /= sum;
        }
        draws
    }

    /// Log density at `x` (must lie on the simplex).
    #[must_use]
    pub fn log_pdf(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.alphas.len(), "dirichlet x has wrong dimension");
        let alpha_sum: f64 = self.alphas.iter().sum();
        let mut log_d = ln_gamma(alpha_sum);
        for (&a, &xi) in self.alphas.iter().zip(x) {
            if xi <= 0.0 {
                return f64::NEG_INFINITY;
            }
            log_d -= ln_gamma(a);
            log_d += (a - 1.0) * xi.ln();
        }
        log_d
    }

    /// Density at `x`.
    #[must_use]
    pub fn pdf(&self, x: &[f64]) -> f64 {
        self.log_pdf(x).exp()
    }

    /// Mean vector `α / Σα`.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        let sum: f64 = self.alphas.iter().sum();
        self.alphas.iter().map(|&a| a / sum).collect()
    }

    /// Marginal variance of component `i`: `α_i(Σα - α_i)/(Σα²(Σα+1))`.
    #[must_use]
    pub fn variance(&self, i: usize) -> f64 {
        let a = self.alphas.get(i).copied().unwrap_or(f64::NAN);
        let sum: f64 = self.alphas.iter().sum();
        a * (sum - a) / (sum * sum * (sum + 1.0))
    }
}

/// Dirichlet-multinomial (compound) distribution: a multinomial whose
/// category probabilities are themselves drawn from a Dirichlet.
#[must_use]
#[derive(Clone, Debug)]
pub struct DirichletMultinomial {
    pub n: u64,
    pub alphas: Vec<f64>,
}

impl DirichletMultinomial {
    /// Validate `n > 0` and positive `alphas`.
    pub fn new(n: u64, alphas: Vec<f64>) -> Result<Self, String> {
        if n == 0 {
            return Err("dirichlet-multinomial trials n must be positive".into());
        }
        let _ = Dirichlet::new(alphas.clone())?;
        Ok(Self { n, alphas })
    }

    /// Draw a count vector: sample `π ~ Dirichlet(α)`, then `counts ~
    /// Multinomial(n, π)`.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> Vec<u64> {
        let dir = Dirichlet {
            alphas: self.alphas.clone(),
        };
        let pi = dir.sample(rng);
        let multi = Multinomial {
            n: self.n,
            probs: pi,
        };
        multi.sample(rng)
    }

    /// Exact log probability mass:
    /// `ln(n!) - Σ ln(k_i!) + ln Γ(Σα) - Σ ln Γ(α_i) + Σ ln Γ(α_i + k_i) - ln Γ(Σα + n)`.
    ///
    /// # Panics
    /// Panics if `counts.len() != alphas.len()` or `counts` does not sum to `n`.
    #[must_use]
    pub fn log_pmf(&self, counts: &[u64]) -> f64 {
        assert_eq!(
            counts.len(),
            self.alphas.len(),
            "dirichlet-multinomial counts and alphas must have equal length"
        );
        let total: u64 = counts.iter().sum();
        assert_eq!(
            total, self.n,
            "dirichlet-multinomial counts must sum to the number of trials"
        );
        let alpha_sum: f64 = self.alphas.iter().sum();
        let mut log_p = ln_gamma(self.n as f64 + 1.0) + ln_gamma(alpha_sum);
        for (&k, &a) in counts.iter().zip(&self.alphas) {
            log_p -= ln_gamma(k as f64 + 1.0) + ln_gamma(a);
            log_p += ln_gamma(a + k as f64);
        }
        log_p - ln_gamma(alpha_sum + self.n as f64)
    }

    /// Exact probability mass of `counts`.
    #[must_use]
    pub fn pmf(&self, counts: &[u64]) -> f64 {
        self.log_pmf(counts).exp()
    }

    /// Mean count vector `n·α / Σα`.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        let dir_mean = Dirichlet {
            alphas: self.alphas.clone(),
        }
        .mean();
        dir_mean.iter().map(|&m| m * self.n as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorical_sums_and_samples() {
        let cat = Categorical::new(vec![0.5, 0.3, 0.2]).unwrap();
        assert!(Categorical::new(vec![0.5, 0.4]).is_err());
        assert!((cat.pmf(0) - 0.5).abs() < 1e-12);
        assert_eq!(cat.pmf(5), 0.0);
        let mut rng = Rng::new(11);
        let mut counts = [0usize; 3];
        for _ in 0..100_000 {
            counts[cat.sample(&mut rng)] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            let f = c as f64 / 100_000.0;
            assert!((f - cat.probs[i]).abs() < 0.01, "cat {i}: {f}");
        }
    }

    #[test]
    fn multinomial_pmf_matches_hand_calc() {
        let m = Multinomial::new(10, vec![0.5, 0.3, 0.2]).unwrap();
        let p = m.pmf(&[5, 3, 2]);
        let expected = 2520.0 * 0.5f64.powi(5) * 0.3f64.powi(3) * 0.2f64.powi(2);
        assert!((p - expected).abs() < 1e-12, "p {p} expected {expected}");
        assert!((m.log_pmf(&[5, 3, 2]) - expected.ln()).abs() < 1e-12);
        assert!((m.covariance(0, 0) - 10.0 * 0.5 * 0.5).abs() < 1e-12);
        assert!((m.covariance(0, 1) + 10.0 * 0.5 * 0.3).abs() < 1e-12);
    }

    #[test]
    fn multinomial_samples_sum_to_n() {
        let m = Multinomial::new(50, vec![0.1, 0.2, 0.7]).unwrap();
        let mut rng = Rng::new(3);
        for _ in 0..100 {
            let counts = m.sample(&mut rng);
            assert_eq!(counts.iter().sum::<u64>(), 50);
        }
    }

    #[test]
    fn dirichlet_uniform_has_constant_pdf() {
        let d = Dirichlet::new(vec![1.0, 1.0, 1.0]).unwrap();
        assert!((d.pdf(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]) - 2.0).abs() < 1e-9);
        assert!((d.pdf(&[0.1, 0.2, 0.7]) - 2.0).abs() < 1e-9);
        assert!(d.pdf(&[0.0, 0.5, 0.5]) == 0.0);
    }

    #[test]
    fn dirichlet_samples_lie_on_simplex() {
        let d = Dirichlet::new(vec![2.0, 3.0, 5.0]).unwrap();
        let mut rng = Rng::new(9);
        for _ in 0..100 {
            let x = d.sample(&mut rng);
            let sum: f64 = x.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9);
            assert!(x.iter().all(|&v| v > 0.0 && v < 1.0));
        }
        let mean = d.mean();
        let total: f64 = mean.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn dirichlet_multinomial_uniform_prior_is_uniform_over_compositions() {
        let dm = DirichletMultinomial::new(10, vec![1.0, 1.0, 1.0]).unwrap();
        let expected = 1.0 / 66.0; // C(12, 2) compositions
        assert!((dm.pmf(&[5, 3, 2]) - expected).abs() < 1e-12);
        assert!((dm.pmf(&[10, 0, 0]) - expected).abs() < 1e-12);
        assert!((dm.pmf(&[0, 7, 3]) - expected).abs() < 1e-12);
        let mean = dm.mean();
        assert!((mean[0] - 10.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn dirichlet_multinomial_samples_sum_to_n() {
        let dm = DirichletMultinomial::new(20, vec![1.0, 2.0, 3.0]).unwrap();
        let mut rng = Rng::new(5);
        for _ in 0..50 {
            let counts = dm.sample(&mut rng);
            assert_eq!(counts.iter().sum::<u64>(), 20);
        }
    }
}
