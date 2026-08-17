//! Expectation-maximization: a generic driver plus a full-covariance
//! Gaussian mixture model (GMM) with responsibilities, scoring and sampling.

use crate::distributions::Normal;
use crate::kalman::{cholesky, mat_mul, mat_t};
use crate::rng::Rng;

/// A parametric latent-variable model that EM can fit.
pub trait EMModel {
    /// Unnormalized log-posterior weight of each component for one
    /// observation given the current parameters (i.e. `ln w_k + ln p_k`).
    fn log_responsibilities(&self, params: &[f64], observation: &[f64]) -> Vec<f64>;

    /// Maximization step: new parameters (same length as `params`) given
    /// normalized responsibilities `r[i][k]`.
    fn m_step(
        &self,
        params: &[f64],
        observations: &[Vec<f64>],
        responsibilities: &[Vec<f64>],
    ) -> Vec<f64>;
}

/// Result of a generic EM run.
#[must_use]
#[derive(Clone, Debug)]
pub struct EMResult {
    /// Parameters at convergence.
    pub params: Vec<f64>,
    /// Observed-data log-likelihood per iteration (non-decreasing).
    pub log_likelihoods: Vec<f64>,
    /// Number of iterations executed.
    pub iterations: usize,
}

/// Run expectation-maximization. `params` is the initial parameter vector.
///
/// # Errors
/// Returns an error if `observations` is empty.
pub fn em<M: EMModel>(
    model: &M,
    mut params: Vec<f64>,
    observations: &[Vec<f64>],
    max_iterations: usize,
    tolerance: f64,
) -> Result<EMResult, String> {
    if observations.is_empty() {
        return Err("em needs at least one observation".into());
    }
    if params.is_empty() {
        return Err("em needs at least one parameter".into());
    }
    let n = observations.len();
    let mut responsibilities = vec![vec![0.0; 0]; n];
    let mut log_likelihoods = Vec::with_capacity(max_iterations);
    let mut prev_ll = f64::NEG_INFINITY;
    let mut iterations = 0;
    for _ in 0..max_iterations {
        iterations += 1;
        // E-step.
        let mut ll = 0.0;
        for (i, obs) in observations.iter().enumerate() {
            let log_rho = model.log_responsibilities(&params, obs);
            let max_r = log_rho
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = log_rho.iter().map(|&lr| (lr - max_r).exp()).sum();
            let log_z = max_r + sum_exp.ln();
            ll += log_z;
            responsibilities[i] = log_rho.iter().map(|&lr| (lr - log_z).exp()).collect();
        }
        log_likelihoods.push(ll);
        if ll.is_finite() && (ll - prev_ll).abs() <= tolerance * (1.0 + ll.abs()) {
            break;
        }
        prev_ll = ll;
        // M-step.
        params = model.m_step(&params, observations, &responsibilities);
    }
    Ok(EMResult {
        params,
        log_likelihoods,
        iterations,
    })
}

/// Shallow copy of the lower-triangular part that EM uses to represent a
/// covariance. A covariance `Σ` is stored as its Cholesky factor `L`
/// (`Σ = L·Lᵀ`), diagonal entries in log space so the matrix is always
/// positive definite.
struct CovChol {
    log_diag: Vec<f64>,
    lower: Vec<Vec<f64>>,
}

impl CovChol {
    fn new(d: usize) -> Self {
        Self {
            log_diag: vec![0.0; d],
            lower: vec![vec![0.0; d]; d],
        }
    }

    fn from_matrix(m: &[Vec<f64>]) -> Result<Self, String> {
        let d = m.len();
        let l = cholesky(m)?;
        let mut out = Self::new(d);
        for i in 0..d {
            out.log_diag[i] = l[i][i].ln();
            for j in 0..i {
                out.lower[i][j] = l[i][j];
            }
        }
        Ok(out)
    }

    /// `-0.5·(d·ln 2π + ln det Σ + (x-μ)ᵀΣ⁻¹(x-μ))`.
    fn log_pdf(&self, mu: &[f64], x: &[f64]) -> f64 {
        let d = mu.len();
        let mut log_det = 0.0;
        for i in 0..d {
            log_det += 2.0 * self.log_diag[i];
        }
        let b: Vec<f64> = x.iter().zip(mu).map(|(&xi, &mi)| xi - mi).collect();
        // Solve L·y = b.
        let mut y = vec![0.0; d];
        for i in 0..d {
            let mut s = b[i];
            for j in 0..i {
                s -= self.lower[i][j] * y[j];
            }
            y[i] = s / self.log_diag[i].exp();
        }
        let quad: f64 = y.iter().map(|v| v * v).sum();
        -0.5 * (d as f64 * (2.0 * core::f64::consts::PI).ln() + log_det + quad)
    }
}

/// Gaussian mixture model with per-component full covariances, fitted by EM.
#[must_use]
#[derive(Clone, Debug)]
pub struct GaussianMixture {
    /// Mixture weights, summing to 1.
    pub weights: Vec<f64>,
    /// Component means `[k][feature]`.
    pub means: Vec<Vec<f64>>,
    /// Component covariances `[k][i][j]`.
    pub covariances: Vec<Vec<Vec<f64>>>,
}

impl GaussianMixture {
    /// Fit `n_components` Gaussians to `data` (rows are samples) by EM.
    /// Returns the per-iteration log-likelihood history.
    ///
    /// Initialization is deterministic: the first `n_components` rows seed
    /// the means, the data's per-feature variance seeds the covariances.
    ///
    /// # Errors
    /// Returns an error if `data` is empty, `n_components` is zero or larger
    /// than the data, or the fit diverges.
    pub fn fit(
        &mut self,
        data: &[Vec<f64>],
        n_components: usize,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<Vec<f64>, String> {
        if data.is_empty() {
            return Err("gmm fit needs data".into());
        }
        let n = data.len();
        let d = data[0].len();
        if n_components == 0 || n_components > n {
            return Err("gmm needs 1 <= n_components <= n_samples".into());
        }
        if data.iter().any(|row| row.len() != d) {
            return Err("gmm data rows have unequal length".into());
        }

        // Deterministic init.
        let weights = vec![1.0 / n_components as f64; n_components];
        let means: Vec<Vec<f64>> = data[..n_components].to_vec();
        let mut covs = Vec::with_capacity(n_components);
        let mut feat_var = vec![0.0; d];
        let mut feat_mean = vec![0.0; d];
        for row in data {
            for (j, &x) in row.iter().enumerate() {
                feat_mean[j] += x;
            }
        }
        for m in &mut feat_mean {
            *m /= n as f64;
        }
        for row in data {
            for (j, &x) in row.iter().enumerate() {
                feat_var[j] += (x - feat_mean[j]).powi(2);
            }
        }
        for v in &mut feat_var {
            *v /= n as f64;
            *v = v.max(1e-6);
        }
        for _ in 0..n_components {
            let mut c = vec![vec![0.0; d]; d];
            for j in 0..d {
                c[j][j] = feat_var[j];
            }
            covs.push(c);
        }

        let model = GMMModel {
            n_components,
            n_features: d,
        };
        let mut params = GMMModel::flatten(&weights, &means, &covs);
        let result = em(&model, params.clone(), data, max_iterations, tolerance)?;
        params = result.params;

        let mut ll_hist = result.log_likelihoods;
        let flat = GMMModel::unflatten(&params, n_components, d);
        if ll_hist.last().map_or(false, |&ll| ll.is_nan() || ll.is_infinite()) {
            return Err("gmm fit diverged".into());
        }
        self.weights = flat.0.clone();
        self.means = flat.1.clone();
        self.covariances = flat.2.clone();
        // Recompute the exact log-likelihood at the final parameters.
        let ll = self.log_score_multi(data);
        ll_hist.push(ll);
        Ok(ll_hist)
    }

    /// Posterior component membership probabilities for `x`.
    #[must_use]
    pub fn responsibilities(&self, x: &[f64]) -> Vec<f64> {
        let k = self.weights.len();
        let mut log_rho = Vec::with_capacity(k);
        for j in 0..k {
            let mut s = self.weights[j].ln();
            match CovChol::from_matrix(&self.covariances[j]) {
                Ok(c) => s += c.log_pdf(&self.means[j], x),
                Err(_) => s = f64::NEG_INFINITY,
            }
            log_rho.push(s);
        }
        let max_r = log_rho.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = log_rho.iter().map(|&lr| (lr - max_r).exp()).sum();
        log_rho
            .iter()
            .map(|&lr| (lr - max_r).exp() / sum_exp)
            .collect()
    }

    /// Mixture density at `x`.
    #[must_use]
    pub fn score(&self, x: &[f64]) -> f64 {
        let mut s = 0.0;
        for j in 0..self.weights.len() {
            if let Ok(c) = CovChol::from_matrix(&self.covariances[j]) {
                s += self.weights[j] * c.log_pdf(&self.means[j], x).exp();
            }
        }
        s
    }

    /// Log mixture density at `x`.
    #[must_use]
    pub fn log_score(&self, x: &[f64]) -> f64 {
        self.score(x).ln()
    }

    /// Average log density over `data`.
    #[must_use]
    pub fn log_score_multi(&self, data: &[Vec<f64>]) -> f64 {
        data.iter().map(|row| self.log_score(row)).sum::<f64>() / data.len() as f64
    }

    /// Number of free parameters (for information criteria).
    #[must_use]
    pub fn n_parameters(&self) -> usize {
        let d = self.means[0].len();
        (self.weights.len() - 1) + self.weights.len() * (d + d * (d + 1) / 2)
    }

    /// Bayesian information criterion on `data` (lower is better).
    #[must_use]
    pub fn bic(&self, data: &[Vec<f64>]) -> f64 {
        -2.0 * self.log_score_multi(data) * data.len() as f64
            + self.n_parameters() as f64 * (data.len() as f64).ln()
    }

    /// Akaike information criterion on `data` (lower is better).
    #[must_use]
    pub fn aic(&self, data: &[Vec<f64>]) -> f64 {
        -2.0 * self.log_score_multi(data) * data.len() as f64 + 2.0 * self.n_parameters() as f64
    }

    /// Draw a sample: pick a component by weight, then `μ + L·z`.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let mut u = rng.uniform();
        let mut k = 0;
        for (i, &w) in self.weights.iter().enumerate() {
            if u < w {
                k = i;
                break;
            }
            u -= w;
        }
        let d = self.means[0].len();
        if let Ok(l) = cholesky(&self.covariances[k]) {
            let mut x = self.means[k].clone();
            for i in 0..d {
                let z = Normal { mu: 0.0, sigma: 1.0 }.sample(rng);
                for j in 0..d {
                    x[j] += l[j][i] * z;
                }
            }
            x
        } else {
            self.means[k].clone()
        }
    }
}

/// Internal flat-parameter model used by the EM driver for the GMM.
struct GMMModel {
    n_components: usize,
    n_features: usize,
}

impl GMMModel {
    /// Layout: `[ln a_0..ln a_k] [μ_0..μ_{k-1}] [L_0..L_{k-1}]` where weights
    /// are `softmax(a)` and each `L` stores `log_diag` then the strict lower
    /// triangle.
    fn flatten(weights: &[f64], means: &[Vec<f64>], covs: &[Vec<Vec<f64>>]) -> Vec<f64> {
        let k = weights.len();
        let d = means[0].len();
        let mut params = Vec::with_capacity(k * (1 + d + d * (d + 1) / 2));
        for &w in weights {
            params.push(w.ln());
        }
        for mean in means {
            params.extend_from_slice(mean);
        }
        for cov in covs {
            let l = cholesky(cov).unwrap_or_else(|_| {
                let mut eye = vec![vec![0.0; d]; d];
                for i in 0..d {
                    eye[i][i] = 1.0;
                }
                eye
            });
            for i in 0..d {
                params.push(l[i][i].ln());
                for j in 0..i {
                    params.push(l[i][j]);
                }
            }
        }
        params
    }

    fn unflatten(params: &[f64], k: usize, d: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let mut idx = 0;
        let mut a = vec![0.0; k];
        for ai in &mut a {
            *ai = params[idx];
            idx += 1;
        }
        let max_a = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = a.iter().map(|&v| (v - max_a).exp()).sum();
        let weights: Vec<f64> = a.iter().map(|&v| (v - max_a).exp() / sum_exp).collect();
        let mut means = vec![vec![0.0; d]; k];
        for mean in &mut means {
            mean.copy_from_slice(&params[idx..idx + d]);
            idx += d;
        }
        let mut covs = vec![vec![vec![0.0; d]; d]; k];
        for c in &mut covs {
            for i in 0..d {
                c[i][i] = params[idx].exp();
                idx += 1;
                for j in 0..i {
                    c[i][j] = params[idx];
                    idx += 1;
                }
            }
            // Rebuild full L^T to make it symmetric: L·Lᵀ with the stored
            // strict lower part.
            let mut l = vec![vec![0.0; d]; d];
            for i in 0..d {
                l[i][i] = c[i][i].sqrt();
                for j in 0..i {
                    l[i][j] = c[i][j];
                }
            }
            *c = mat_mul(&l, &mat_t(&l));
        }
        (weights, means, covs)
    }
}

impl EMModel for GMMModel {
    fn log_responsibilities(&self, params: &[f64], observation: &[f64]) -> Vec<f64> {
        let k = self.n_components;
        let d = self.n_features;
        let (weights, means, covs) = Self::unflatten(params, k, d);
        let mut out = Vec::with_capacity(k);
        for j in 0..k {
            match CovChol::from_matrix(&covs[j]) {
                Ok(c) => out.push(weights[j].ln() + c.log_pdf(&means[j], observation)),
                Err(_) => out.push(f64::NEG_INFINITY),
            }
        }
        out
    }

    fn m_step(
        &self,
        _params: &[f64],
        observations: &[Vec<f64>],
        responsibilities: &[Vec<f64>],
    ) -> Vec<f64> {
        let k = self.n_components;
        let d = self.n_features;
        let n = observations.len();
        let mut r_sum = vec![0.0; k];
        for r in responsibilities {
            for (j, &rj) in r.iter().enumerate() {
                r_sum[j] += rj;
            }
        }
        let mut weights = vec![0.0; k];
        for (j, &s) in r_sum.iter().enumerate() {
            weights[j] = (s / n as f64).max(1e-300);
        }
        let mut means = vec![vec![0.0; d]; k];
        for (i, obs) in observations.iter().enumerate() {
            for j in 0..k {
                for (t, &x) in obs.iter().enumerate() {
                    means[j][t] += responsibilities[i][j] * x;
                }
            }
        }
        for j in 0..k {
            if r_sum[j] > 0.0 {
                for m in &mut means[j] {
                    *m /= r_sum[j];
                }
            }
        }
        let mut covs = vec![vec![vec![0.0; d]; d]; k];
        for (i, obs) in observations.iter().enumerate() {
            for j in 0..k {
                if r_sum[j] <= 0.0 {
                    continue;
                }
                let diff: Vec<f64> = obs.iter().zip(&means[j]).map(|(&x, &m)| x - m).collect();
                let od = outer(diff);
                for a in 0..d {
                    for b in 0..d {
                        covs[j][a][b] += responsibilities[i][j] * od[a][b];
                    }
                }
            }
        }
        for j in 0..k {
            if r_sum[j] > 0.0 {
                for a in 0..d {
                    for b in 0..d {
                        covs[j][a][b] /= r_sum[j];
                    }
                }
                // Keep the covariance positive definite: jitter the diagonal
                // only when the Cholesky factor would fail (preserves exact
                // EM monotonicity in the regular case).
                if cholesky(&covs[j]).is_err() {
                    let jitter = 1e-9 * (1.0 + covs[j][0][0].abs());
                    for a in 0..d {
                        covs[j][a][a] += jitter;
                    }
                }
            }
        }
        Self::flatten(&weights, &means, &covs)
    }
}

fn outer(x: Vec<f64>) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = x[i] * x[j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gmm_recovers_two_components_1d() {
        let mut rng = Rng::new(7);
        let n = 1000;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let mu = if rng.uniform() < 0.5 { -3.0 } else { 3.0 };
            data.push(vec![
                mu + Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng),
            ]);
        }
        let mut gmm = GaussianMixture {
            weights: Vec::new(),
            means: Vec::new(),
            covariances: Vec::new(),
        };
        let ll_hist = gmm.fit(&data, 2, 300, 1e-8).unwrap();
        assert!(ll_hist.len() > 1, "no EM iterations recorded");

        let mut means = gmm.means.iter().map(|m| m[0]).collect::<Vec<_>>();
        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((means[0] + 3.0).abs() < 0.25, "mean0 {}", means[0]);
        assert!((means[1] - 3.0).abs() < 0.25, "mean1 {}", means[1]);
        let mut weights = gmm.weights.clone();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((weights[0] - 0.5).abs() < 0.05, "w {}", weights[0]);
        assert!((gmm.covariances[0][0][0]).abs() - 1.0 < 0.2);

        // Responsibilities normalize and score is a density.
        let r = gmm.responsibilities(&data[0]);
        assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let score = gmm.score(&data[0]);
        assert!(score > 0.0 && score < 1.0);

        // Sampling respects the mixture.
        let mut rng2 = Rng::new(11);
        let sample = gmm.sample(&mut rng2);
        assert_eq!(sample.len(), 1);
    }

    #[test]
    fn gmm_2d_recovers_well_separated_components() {
        let mut rng = Rng::new(13);
        let n = 800;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let center = if rng.uniform() < 0.5 {
                (0.0, 0.0)
            } else {
                (4.0, 4.0)
            };
            data.push(vec![
                center.0 + Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng),
                center.1 + Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng),
            ]);
        }
        let mut gmm = GaussianMixture {
            weights: Vec::new(),
            means: Vec::new(),
            covariances: Vec::new(),
        };
        gmm.fit(&data, 2, 300, 1e-8).unwrap();
        let mut ms = gmm.means.clone();
        ms.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        assert!((ms[0][0]).abs() < 0.3, "m0 {}", ms[0][0]);
        assert!((ms[1][0] - 4.0).abs() < 0.3, "m1 {}", ms[1][0]);
        assert!((ms[0][1]).abs() < 0.3 && (ms[1][1] - 4.0).abs() < 0.3);
    }

    #[test]
    fn gmm_log_likelihood_is_monotone() {
        let mut rng = Rng::new(17);
        let mut data = Vec::new();
        for _ in 0..300 {
            data.push(vec![
                Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng),
                Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng),
            ]);
        }
        let mut gmm = GaussianMixture {
            weights: Vec::new(),
            means: Vec::new(),
            covariances: Vec::new(),
        };
        let ll_hist = gmm.fit(&data, 3, 100, 0.0).unwrap();
        for w in ll_hist.windows(2).take(50) {
            assert!(
                w[1] >= w[0] - 1e-9,
                "log-likelihood decreased: {} -> {}",
                w[0],
                w[1]
            );
        }
        // BIC/AIC finite.
        assert!(gmm.bic(&data).is_finite());
        assert!(gmm.aic(&data).is_finite());
    }

    /// Bernoulli mixture used to test the generic EM driver.
    /// Params: `[a_0, a_1, logit(p_0), logit(p_1)]`, weights `softmax(a)`.
    struct BernoulliMixture {
        k: usize,
    }

    impl BernoulliMixture {
        fn weight(j: usize, a: &[f64]) -> f64 {
            let max_a = a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = a.iter().map(|&v| (v - max_a).exp()).sum();
            (a[j] - max_a).exp() / sum
        }
    }

    impl EMModel for BernoulliMixture {
        fn log_responsibilities(&self, params: &[f64], observation: &[f64]) -> Vec<f64> {
            let x = observation[0];
            (0..self.k)
                .map(|j| {
                    let w = Self::weight(j, &params[..self.k]);
                    let logit = params[self.k + j];
                    let p = logit.exp() / (1.0 + logit.exp()).max(1e-300);
                    w.ln() + x * p.ln() + (1.0 - x) * (1.0 - p).ln()
                })
                .collect()
        }

        fn m_step(
            &self,
            _params: &[f64],
            observations: &[Vec<f64>],
            responsibilities: &[Vec<f64>],
        ) -> Vec<f64> {
            let n = observations.len();
            let mut r_sum = vec![0.0; self.k];
            for r in responsibilities {
                for (j, &rj) in r.iter().enumerate() {
                    r_sum[j] += rj;
                }
            }
            let mut params = Vec::with_capacity(2 * self.k);
            for &s in &r_sum {
                let w = (s / n as f64).clamp(1e-9, 1.0 - 1e-9);
                params.push((w / (1.0 - w)).ln());
            }
            for j in 0..self.k {
                let mut num = 0.0;
                for (i, obs) in observations.iter().enumerate() {
                    num += responsibilities[i][j] * obs[0];
                }
                if r_sum[j] > 0.0 {
                    let p = (num / r_sum[j]).clamp(1e-9, 1.0 - 1e-9);
                    params.push((p / (1.0 - p)).ln());
                } else {
                    params.push(0.0);
                }
            }
            params
        }
    }

    #[test]
    fn generic_em_fits_bernoulli_mixture() {
        let mut rng = Rng::new(19);
        let mut data = Vec::new();
        for _ in 0..600 {
            let p = if rng.uniform() < 0.5 { 0.1 } else { 0.8 };
            let x = if rng.uniform() < p { 1.0 } else { 0.0 };
            data.push(vec![x]);
        }
        let model = BernoulliMixture { k: 2 };
        let result = em(&model, vec![0.0, 0.0, 0.0, 0.0], &data, 200, 1e-9).unwrap();
        assert!(result.iterations > 1);
        // p values recovered near 0.1 and 0.8 (order may swap).
        let mut ps: Vec<f64> = result
            .params
            .iter()
            .map(|&logit| logit.exp() / (1.0 + logit.exp()))
            .collect();
        ps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((ps[0] - 0.1).abs() < 0.05, "p0 {}", ps[0]);
        assert!((ps[1] - 0.8).abs() < 0.05, "p1 {}", ps[1]);
        // Monotone log-likelihood.
        assert!(
            result
                .log_likelihoods
                .windows(2)
                .all(|w| w[1] >= w[0] - 1e-9)
        );
    }

    #[test]
    fn gmm_errors_on_bad_input() {
        let mut gmm = GaussianMixture {
            weights: Vec::new(),
            means: Vec::new(),
            covariances: Vec::new(),
        };
        assert!(gmm.fit(&[], 2, 100, 1e-8).is_err());
        assert!(gmm.fit(&[vec![1.0], vec![2.0]], 3, 100, 1e-8).is_err());
        assert!(gmm.fit(&[vec![1.0, 2.0], vec![3.0]], 1, 100, 1e-8).is_err());
    }
}