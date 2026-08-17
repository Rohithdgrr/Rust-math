//! Variational inference: a mean-field Gaussian variational family, Monte
//! Carlo ELBO estimation, and black-box variational inference (BBVI) with
//! reparameterization gradients.

use crate::distributions::Normal;
use crate::rng::Rng;

/// Independent-Gaussian variational family over `dim` parameters, stored as
/// `(mean, log_sigma)` so the scale is always positive.
#[must_use]
#[derive(Clone, Debug)]
pub struct MeanFieldGaussian {
    /// Component means.
    pub mu: Vec<f64>,
    /// Component log standard deviations.
    pub log_sigma: Vec<f64>,
}

impl MeanFieldGaussian {
    /// Standard normal init for all components.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            mu: vec![0.0; dim],
            log_sigma: vec![0.0; dim],
        }
    }

    /// Mean vector.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mu
    }

    /// Standard deviation vector.
    #[must_use]
    pub fn stddev(&self) -> Vec<f64> {
        self.log_sigma.iter().map(|&l| l.exp()).collect()
    }

    /// Differential entropy `0.5·Σ ln(2πe·σᵢ²)`.
    #[must_use]
    pub fn entropy(&self) -> f64 {
        let dim = self.mu.len() as f64;
        0.5 * dim * (2.0 * core::f64::consts::PI * core::f64::consts::E).ln()
            + self.log_sigma.iter().sum::<f64>()
    }

    /// Log density at `z`.
    #[must_use]
    pub fn log_prob(&self, z: &[f64]) -> f64 {
        let mut lp = 0.0;
        for (i, &zi) in z.iter().enumerate() {
            let sigma = self.log_sigma[i].exp();
            lp += -(0.5 * (2.0 * core::f64::consts::PI).ln() + self.log_sigma[i])
                - 0.5 * ((zi - self.mu[i]) / sigma).powi(2);
        }
        lp
    }

    /// Reparameterized sample `z = μ + σ·ε`, `ε ~ N(0, I)`.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let normal = Normal {
            mu: 0.0,
            sigma: 1.0,
        };
        self.mu
            .iter()
            .zip(&self.log_sigma)
            .map(|(&m, &ls)| m + ls.exp() * normal.sample(rng))
            .collect()
    }
}

/// Monte Carlo estimate of the ELBO:
/// `L(q) = (1/S)Σ_s log p(z_s) + H(q)` with `z_s ~ q`.
#[must_use]
pub fn elbo_estimate(
    log_joint: &dyn Fn(&[f64]) -> f64,
    q: &MeanFieldGaussian,
    n_samples: usize,
    rng: &mut Rng,
) -> f64 {
    let mut sum = 0.0;
    for _ in 0..n_samples {
        sum += log_joint(&q.sample(rng));
    }
    sum / n_samples as f64 + q.entropy()
}

/// Central-difference gradient of `f` at `x`.
fn numeric_grad(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    let mut grad = vec![0.0; x.len()];
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[i] += h;
        xm[i] -= h;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * h);
    }
    grad
}

/// Result of a BBVI run.
#[must_use]
#[derive(Clone, Debug)]
pub struct BBVIResult {
    /// Fitted variational distribution.
    pub q: MeanFieldGaussian,
    /// ELBO estimate after each iteration.
    pub elbo_history: Vec<f64>,
}

/// Black-box variational inference: maximize the ELBO of `log_joint` over
/// the mean-field Gaussian family using reparameterization gradients and
/// SGD with a decaying step size.
#[must_use]
pub fn bbvi(
    log_joint: &dyn Fn(&[f64]) -> f64,
    dim: usize,
    n_iterations: usize,
    n_samples: usize,
    learning_rate: f64,
    rng: &mut Rng,
) -> BBVIResult {
    let mut q = MeanFieldGaussian::new(dim);
    let mut elbo_history = Vec::with_capacity(n_iterations);
    let normal = Normal {
        mu: 0.0,
        sigma: 1.0,
    };
    for t in 0..n_iterations {
        // Stochastic reparameterization gradient.
        let mut grad_mu = vec![0.0; dim];
        let mut grad_ls = vec![0.0; dim];
        for _ in 0..n_samples {
            let eps: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
            let z: Vec<f64> = q
                .mu
                .iter()
                .zip(&q.log_sigma)
                .zip(&eps)
                .map(|((&m, &ls), &e)| m + ls.exp() * e)
                .collect();
            let dz = numeric_grad(log_joint, &z);
            for i in 0..dim {
                grad_mu[i] += dz[i];
                grad_ls[i] += dz[i] * q.log_sigma[i].exp() * eps[i];
            }
        }
        let step = learning_rate / (1.0 + t as f64).sqrt();
        for i in 0..dim {
            q.mu[i] += step * grad_mu[i] / n_samples as f64;
            q.log_sigma[i] += step * (grad_ls[i] / n_samples as f64 + 1.0); // +∇H = +1
        }
        // Safety clamp to avoid numerical blowup.
        for ls in &mut q.log_sigma {
            *ls = ls.clamp(-6.0, 3.0);
        }
        if t % 10 == 0 || t == n_iterations - 1 {
            elbo_history.push(elbo_estimate(log_joint, &q, 100, rng));
        }
    }
    BBVIResult { q, elbo_history }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbvi_recovers_gaussian_posterior() {
        // Posterior of θ ~ N(0,1) after 20 obs with mean 1.5, σ_obs = 1:
        // θ ~ N(30/21, 1/21).
        let n_obs = 20.0;
        let obs_mean = 1.5;
        let log_joint = |z: &[f64]| {
            let theta = z[0];
            -0.5 * theta * theta
                - n_obs * 0.5 * (2.0 * core::f64::consts::PI).ln()
                - n_obs / 2.0 * (theta - obs_mean).powi(2)
        };
        let mut rng = Rng::new(3);
        let result = bbvi(&log_joint, 1, 3000, 20, 0.05, &mut rng);
        let mu_true = 30.0 / 21.0;
        let sigma_true = (1.0_f64 / 21.0).sqrt();
        let q = &result.q;
        assert!(
            (q.mu[0] - mu_true).abs() < 0.15,
            "mu {} vs {}",
            q.mu[0],
            mu_true
        );
        assert!(
            (q.log_sigma[0].exp() - sigma_true).abs() < 0.05,
            "sigma {} vs {}",
            q.log_sigma[0].exp(),
            sigma_true
        );
        // ELBO improves over the run.
        assert!(result.elbo_history.len() >= 2);
        assert!(*result.elbo_history.last().unwrap() > result.elbo_history[0]);
    }

    #[test]
    fn bbvi_two_dimensional_diagonal() {
        // Independent targets: z1 ~ N(0, 1), z2 ~ N(2, 4).
        let log_joint = |z: &[f64]| {
            -0.5 * z[0] * z[0] - 0.5 * ((z[1] - 2.0) / 2.0).powi(2)
        };
        let mut rng = Rng::new(5);
        let result = bbvi(&log_joint, 2, 10_000, 40, 0.1, &mut rng);
        assert!((result.q.mu[0]).abs() < 0.15, "mu0 {}", result.q.mu[0]);
        assert!((result.q.mu[1] - 2.0).abs() < 0.15, "mu1 {}", result.q.mu[1]);
        assert!((result.q.log_sigma[0].exp() - 1.0).abs() < 0.15, "s0");
        assert!((result.q.log_sigma[1].exp() - 2.0).abs() < 0.3, "s1");
    }

    #[test]
    fn elbo_is_finite_and_sensible() {
        let log_joint = |z: &[f64]| -0.5 * z[0] * z[0] - 0.5 * z[1] * z[1];
        let q = MeanFieldGaussian::new(2);
        let mut rng = Rng::new(7);
        let elbo = elbo_estimate(&log_joint, &q, 5000, &mut rng);
        // The target closure omits the -ln(2π) normalization, so
        // E[log p] = -0.5·E[z0²] - 0.5·E[z1²] = -1 and H = ln(2πe):
        // ELBO = ln(2πe) - 1.
        let expected = (2.0 * core::f64::consts::PI * core::f64::consts::E).ln() - 1.0;
        assert!(
            (elbo - expected).abs() < 0.05,
            "elbo {} vs {}",
            elbo,
            expected
        );
        assert!((q.entropy() - (2.0 * core::f64::consts::PI * core::f64::consts::E).ln()).abs() < 1e-9);
    }

    #[test]
    fn mean_field_sample_and_log_prob_consistent() {
        let q = MeanFieldGaussian {
            mu: vec![1.0, -2.0],
            log_sigma: vec![0.0, 0.5],
        };
        let mut rng = Rng::new(9);
        let z = q.sample(&mut rng);
        assert_eq!(z.len(), 2);
        // Monte Carlo mean/sd of samples tracks the parameters.
        let n = 200_000;
        let mut s1 = crate::online_stats::StreamingStats::new();
        let mut s2 = crate::online_stats::StreamingStats::new();
        for _ in 0..n {
            let z = q.sample(&mut rng);
            s1.update(z[0]);
            s2.update(z[1]);
        }
        assert!((s1.mean() - 1.0).abs() < 0.02);
        assert!((s2.mean() + 2.0).abs() < 0.02);
        assert!((s1.stddev() - 1.0).abs() < 0.02);
        assert!((s2.stddev() - 0.5f64.exp()).abs() < 0.02);
    }
}
