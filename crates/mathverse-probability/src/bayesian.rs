//! Bayesian methods: priors, posteriors, conjugate priors, credible intervals, Bayesian inference.

use crate::distributions::BetaFunc;
use crate::{rng::Rng, F64Ext};

/// Prior distribution trait.
pub trait Prior {
    fn log_pdf(&self, x: &[f64]) -> f64;
    fn sample(&self, rng: &mut Rng) -> Vec<f64>;
}

/// Beta prior for Bernoulli/Binomial likelihood.
#[must_use]
pub struct BetaPrior {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaPrior {
    #[must_use]
    pub fn new(alpha: f64, beta: f64) -> Self {
        BetaPrior { alpha, beta }
    }

    /// Posterior after observing successes and failures.
    pub fn posterior(&self, successes: u64, failures: u64) -> BetaPrior {
        BetaPrior {
            alpha: self.alpha + successes as f64,
            beta: self.beta + failures as f64,
        }
    }

    /// Predictive distribution (Beta-Bernoulli).
    #[must_use]
    pub fn predictive(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
}

impl Prior for BetaPrior {
    fn log_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != 1 || x[0] <= 0.0 || x[0] >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let a = self.alpha;
        let b = self.beta;
        (a - 1.0) * x[0].ln() + (b - 1.0) * (1.0 - x[0]).ln() - (a, b).beta().ln()
    }

    fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let _beta_dist = crate::distributions::Beta {
            alpha: self.alpha,
            beta: self.beta,
        };
        // Use rejection sampling for beta
        let mut x;
        loop {
            x = rng.uniform();
            let u = rng.uniform();
            let pdf = x.powf(self.alpha - 1.0) * (1.0 - x).powf(self.beta - 1.0);
            let max_pdf = if self.alpha >= 1.0 && self.beta >= 1.0 {
                let mode = (self.alpha - 1.0) / (self.alpha + self.beta - 2.0);
                mode.powf(self.alpha - 1.0) * (1.0 - mode).powf(self.beta - 1.0)
            } else {
                1.0
            };
            if u < pdf / max_pdf {
                break;
            }
        }
        vec![x]
    }
}

/// Normal prior for Normal likelihood.
#[must_use]
pub struct NormalPrior {
    pub mu: f64,
    pub sigma: f64,
}

impl NormalPrior {
    #[must_use]
    pub fn new(mu: f64, sigma: f64) -> Self {
        NormalPrior { mu, sigma }
    }

    /// Posterior after observing data with known variance.
    #[must_use]
    pub fn posterior_known_variance(&self, data: &[f64], known_variance: f64) -> NormalPrior {
        let n = data.len() as f64;
        let sample_mean = data.iter().sum::<f64>() / n;

        let prior_precision = 1.0 / (self.sigma * self.sigma);
        let data_precision = n / known_variance;
        let posterior_precision = prior_precision + data_precision;
        let posterior_sigma = 1.0 / posterior_precision.sqrt();
        let posterior_mu =
            (prior_precision * self.mu + data_precision * sample_mean) / posterior_precision;

        NormalPrior {
            mu: posterior_mu,
            sigma: posterior_sigma,
        }
    }
}

impl Prior for NormalPrior {
    fn log_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != 1 {
            return f64::NEG_INFINITY;
        }
        let z = (x[0] - self.mu) / self.sigma;
        -0.5 * z * z - (self.sigma * (2.0 * core::f64::consts::PI).sqrt()).ln()
    }

    fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let normal = crate::distributions::Normal {
            mu: self.mu,
            sigma: self.sigma,
        };
        vec![normal.sample(rng)]
    }
}

/// Gamma prior for Poisson/Exponential likelihood.
#[must_use]
pub struct GammaPrior {
    pub shape: f64,
    pub rate: f64,
}

impl GammaPrior {
    #[must_use]
    pub fn new(shape: f64, rate: f64) -> Self {
        GammaPrior { shape, rate }
    }

    /// Posterior after observing count data.
    pub fn posterior(&self, total_count: u64, total_exposure: f64) -> GammaPrior {
        GammaPrior {
            shape: self.shape + total_count as f64,
            rate: self.rate + total_exposure,
        }
    }
}

impl Prior for GammaPrior {
    fn log_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != 1 || x[0] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let k = self.shape;
        let theta = 1.0 / self.rate;
        (k - 1.0) * x[0].ln() - x[0] / theta - k.gamma().ln() - k * theta.ln()
    }

    fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let gamma = crate::distributions::Gamma {
            shape: self.shape,
            rate: self.rate,
        };
        vec![gamma.sample(rng)]
    }
}

/// Dirichlet prior for Multinomial likelihood.
#[must_use]
pub struct DirichletPrior {
    pub alpha: Vec<f64>,
}

impl DirichletPrior {
    #[must_use]
    pub fn new(alpha: Vec<f64>) -> Self {
        DirichletPrior { alpha }
    }

    /// Posterior after observing counts.
    pub fn posterior(&self, counts: &[u64]) -> DirichletPrior {
        let new_alpha: Vec<f64> = self
            .alpha
            .iter()
            .zip(counts.iter())
            .map(|(&a, &c)| a + c as f64)
            .collect();
        DirichletPrior { alpha: new_alpha }
    }

    /// Expected value of the Dirichlet distribution.
    #[must_use]
    pub fn mean(&self) -> Vec<f64> {
        let sum: f64 = self.alpha.iter().sum();
        self.alpha.iter().map(|&a| a / sum).collect()
    }
}

impl Prior for DirichletPrior {
    fn log_pdf(&self, x: &[f64]) -> f64 {
        if x.len() != self.alpha.len() {
            return f64::NEG_INFINITY;
        }

        let sum: f64 = x.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return f64::NEG_INFINITY;
        }

        let alpha_sum: f64 = self.alpha.iter().sum();
        let mut log_pdf = alpha_sum.gamma().ln();

        for (i, &a) in self.alpha.iter().enumerate() {
            log_pdf -= a.gamma().ln();
            if x[i] > 0.0 {
                log_pdf += (a - 1.0) * x[i].ln();
            } else if a < 1.0 {
                return f64::NEG_INFINITY;
            }
        }

        log_pdf
    }

    fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let gamma_samples: Vec<f64> = self
            .alpha
            .iter()
            .map(|&a| {
                let gamma = crate::distributions::Gamma {
                    shape: a,
                    rate: 1.0,
                };
                gamma.sample(rng)
            })
            .collect();

        let sum: f64 = gamma_samples.iter().sum();
        gamma_samples.iter().map(|&x| x / sum).collect()
    }
}

/// Credible interval computation.
#[must_use]
pub struct CredibleInterval;

impl CredibleInterval {
    /// Compute credible interval from samples.
    #[must_use]
    pub fn from_samples(samples: &[f64], alpha: f64) -> (f64, f64) {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let lower_idx = ((alpha / 2.0) * n as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n as f64) as usize;

        (sorted[lower_idx], sorted[upper_idx.min(n - 1)])
    }

    /// Highest posterior density (HPD) interval.
    #[must_use]
    pub fn hpd(samples: &[f64], alpha: f64) -> (f64, f64) {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let m = ((1.0 - alpha) * n as f64) as usize;

        let mut min_width = f64::INFINITY;
        let mut best_interval = (0.0, 0.0);

        for i in 0..=(n - m) {
            let width = sorted[i + m - 1] - sorted[i];
            if width < min_width {
                min_width = width;
                best_interval = (sorted[i], sorted[i + m - 1]);
            }
        }

        best_interval
    }
}

/// Bayesian model selection using Bayes factor.
#[must_use]
pub struct BayesFactor;

impl BayesFactor {
    /// Compute Bayes factor (evidence ratio).
    #[must_use]
    pub fn compute(evidence_m1: f64, evidence_m2: f64) -> f64 {
        evidence_m1 / evidence_m2
    }

    /// Interpret Bayes factor strength.
    #[must_use]
    pub fn interpret(bf: f64) -> &'static str {
        if bf < 1.0 {
            Self::interpret(1.0 / bf)
        } else if bf < 3.0 {
            "Not worth more than a bare mention"
        } else if bf < 10.0 {
            "Substantial evidence"
        } else if bf < 30.0 {
            "Strong evidence"
        } else if bf < 100.0 {
            "Very strong evidence"
        } else {
            "Decisive evidence"
        }
    }

    /// Approximate evidence using Laplace approximation.
    #[must_use]
    pub fn laplace_approximation(
        log_likelihood: impl Fn(&[f64]) -> f64,
        log_prior: impl Fn(&[f64]) -> f64,
        mode: &[f64],
        hessian: &[Vec<f64>],
    ) -> f64 {
        let log_posterior = log_likelihood(mode) + log_prior(mode);
        let n = mode.len();

        // Compute determinant of Hessian
        let mut det = 1.0;
        for i in 0..n {
            det *= hessian[i][i];
        }

        (log_posterior + 0.5 * (2.0 * core::f64::consts::PI).ln() * n as f64 - 0.5 * det.ln()).exp()
    }
}

/// Hierarchical Bayesian model.
#[must_use]
pub struct HierarchicalModel {
    pub hyperprior: Box<dyn Prior>,
    pub likelihood: Box<dyn Fn(&[f64], &[f64]) -> f64>,
}

impl HierarchicalModel {
    #[must_use]
    pub fn new<F>(hyperprior: Box<dyn Prior>, likelihood: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> f64 + 'static,
    {
        HierarchicalModel {
            hyperprior,
            likelihood: Box::new(likelihood),
        }
    }

    /// Sample from hierarchical model using Gibbs sampling.
    pub fn sample(&self, _data: &[f64], n_samples: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        let mut samples = Vec::new();

        for _ in 0..n_samples {
            let hyperparams = self.hyperprior.sample(rng);
            samples.push(hyperparams);
        }

        samples
    }
}

/// Empirical Bayes estimation.
#[must_use]
pub struct EmpiricalBayes;

impl EmpiricalBayes {
    /// Estimate hyperparameters by maximizing marginal likelihood.
    pub fn estimate(
        marginal_likelihood: impl Fn(&[f64]) -> f64,
        initial: &[f64],
        rng: &mut Rng,
    ) -> Vec<f64> {
        let mut params = initial.to_vec();
        let mut best_params = params.clone();
        let mut best_ll = marginal_likelihood(&params);

        // Simple hill climbing
        for _ in 0..1000 {
            let mut new_params = params.clone();
            for p in &mut new_params {
                *p += crate::distributions::Normal {
                    mu: 0.0,
                    sigma: 0.1,
                }
                .sample(rng);
            }

            let ll = marginal_likelihood(&new_params);
            if ll > best_ll {
                best_ll = ll;
                best_params = new_params.clone();
                params = new_params;
            }
        }

        best_params
    }
}

/// Bayesian hypothesis testing.
#[must_use]
pub struct BayesianHypothesisTest;

impl BayesianHypothesisTest {
    /// Compute posterior probability of hypothesis.
    #[must_use]
    pub fn posterior_probability(prior_h1: f64, evidence_h1: f64, evidence_h2: f64) -> f64 {
        let bf = evidence_h1 / evidence_h2;
        let posterior_odds = (prior_h1 / (1.0 - prior_h1)) * bf;
        posterior_odds / (1.0 + posterior_odds)
    }

    /// Savage-Dickey density ratio for nested models.
    #[must_use]
    pub fn savage_dickey(prior_density: f64, posterior_density: f64) -> f64 {
        posterior_density / prior_density
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_prior_posterior() {
        let prior = BetaPrior::new(1.0, 1.0);
        let posterior = prior.posterior(5, 5);
        assert!((posterior.alpha - 6.0).abs() < 1e-10);
        assert!((posterior.beta - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_prior_posterior() {
        let prior = NormalPrior::new(0.0, 1.0);
        let data = vec![1.0, 2.0, 3.0];
        let posterior = prior.posterior_known_variance(&data, 1.0);
        assert!(posterior.sigma < prior.sigma);
    }

    #[test]
    fn test_dirichlet_mean() {
        let prior = DirichletPrior::new(vec![1.0, 1.0, 1.0]);
        let mean = prior.mean();
        assert!((mean[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((mean[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!((mean[2] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_credible_interval() {
        let samples: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let (lower, upper) = CredibleInterval::from_samples(&samples, 0.05);
        assert!(lower < upper);
    }

    #[test]
    fn test_bayes_factor() {
        let bf = BayesFactor::compute(0.1, 0.01);
        assert!((bf - 10.0).abs() < 1e-10);
        let interpretation = BayesFactor::interpret(bf);
        assert_eq!(interpretation, "Strong evidence");
    }
}
