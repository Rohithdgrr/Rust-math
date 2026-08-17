//! Kalman filtering and smoothing: linear Kalman filter, extended Kalman
//! filter (EKF), unscented Kalman filter (UKF) and the Rauch-Tung-Striebel
//! (RTS) smoother.
//!
//! Matrices are `Vec<Vec<f64>>` in row-major order; all linear algebra is
//! dependency-free Gaussian elimination.

pub(crate) fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (m, k, n) = (a.len(), b.len(), b[0].len());
    let mut out = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..k {
                s += a[i][t] * b[t][j];
            }
            out[i][j] = s;
        }
    }
    out
}

pub(crate) fn mat_t(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (m, n) = (a.len(), a[0].len());
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

pub(crate) fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(row_a, row_b)| row_a.iter().zip(row_b).map(|(&x, &y)| x + y).collect())
        .collect()
}

pub(crate) fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(row_a, row_b)| row_a.iter().zip(row_b).map(|(&x, &y)| x - y).collect())
        .collect()
}

/// Matrix-vector product `A·x`.
fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(&a, &b)| a * b).sum())
        .collect()
}

/// Outer product `x·xᵀ`.
pub(crate) fn outer(x: &[f64]) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = x[i] * x[j];
        }
    }
    out
}

pub(crate) fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

pub(crate) fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(&x, &y)| x - y).collect()
}

/// Matrix inverse by Gauss-Jordan elimination.
///
/// # Errors
/// Returns an error if the matrix is singular.
pub(crate) fn mat_inv(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .cloned()
                .chain((0..n).map(|j| if i == j { 1.0 } else { 0.0 }))
                .collect()
        })
        .collect();
    for col in 0..n {
        let mut pivot = col;
        for r in col + 1..n {
            if aug[r][col].abs() > aug[pivot][col].abs() {
                pivot = r;
            }
        }
        if aug[pivot][col].abs() < 1e-12 {
            return Err("matrix is singular".into());
        }
        aug.swap(col, pivot);
        let div = aug[col][col];
        for c in 0..2 * n {
            aug[col][c] /= div;
        }
        for r in 0..n {
            if r != col {
                let factor = aug[r][col];
                for c in 0..2 * n {
                    aug[r][c] -= factor * aug[col][c];
                }
            }
        }
    }
    Ok(aug
        .iter()
        .map(|row| row[n..].to_vec())
        .collect())
}

/// Cholesky decomposition `A = L·Lᵀ` (lower triangular).
///
/// # Errors
/// Returns an error if `A` is not positive definite.
pub(crate) fn cholesky(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 1e-12 {
                    return Err("matrix is not positive definite".into());
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Ok(l)
}

/// Linear Kalman filter over a state vector `x` with covariance `P`.
#[must_use]
#[derive(Clone, Debug)]
pub struct KalmanFilter {
    /// State estimate.
    pub x: Vec<f64>,
    /// State covariance.
    pub p: Vec<Vec<f64>>,
}

impl KalmanFilter {
    /// Create a filter with initial state `x0` and covariance `P0`.
    pub fn new(x0: Vec<f64>, p0: Vec<Vec<f64>>) -> Result<Self, String> {
        let n = x0.len();
        if p0.len() != n || p0.iter().any(|row| row.len() != n) {
            return Err("kalman initial covariance has wrong dimension".into());
        }
        Ok(Self { x: x0, p: p0 })
    }

    /// Prediction step: `x = F·x`, `P = F·P·Fᵀ + Q`.
    pub fn predict(&mut self, f: &[Vec<f64>], q: &[Vec<f64>]) {
        self.x = mat_vec_mul(f, &self.x);
        let fpf = mat_mul(&mat_mul(f, &self.p), &mat_t(f));
        self.p = mat_add(&fpf, q);
    }

    /// Update step with measurement `z`, model `z = H·x + v`, `v ~ N(0, R)`.
    /// Returns the innovation `z - H·x`.
    ///
    /// # Errors
    /// Returns an error if `S = H·P·Hᵀ + R` is singular.
    pub fn update(
        &mut self,
        z: &[f64],
        h: &[Vec<f64>],
        r: &[Vec<f64>],
    ) -> Result<Vec<f64>, String> {
        let hx = mat_vec_mul(h, &self.x);
        let innovation = vec_sub(z, &hx);
        let s = mat_add(&mat_mul(&mat_mul(h, &self.p), &mat_t(h)), r);
        let s_inv = mat_inv(&s)?;
        let k = mat_mul(&mat_mul(&self.p, &mat_t(h)), &s_inv);
        self.x = vec_add(&self.x, &mat_vec_mul(&k, &innovation));
        let kh = mat_mul(&k, h);
        // Joseph form for numerical stability: P = (I - KH)P(I - KH)ᵀ + KRKᵀ.
        let n = self.x.len();
        let idn: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let ikh = mat_sub(&idn, &kh);
        let term1 = mat_mul(&mat_mul(&ikh, &self.p), &mat_t(&ikh));
        let term2 = mat_mul(&mat_mul(&k, r), &mat_t(&k));
        self.p = mat_add(&term1, &term2);
        Ok(innovation)
    }

    /// Convenience: `predict` then `update`.
    pub fn step(
        &mut self,
        z: &[f64],
        f: &[Vec<f64>],
        h: &[Vec<f64>],
        q: &[Vec<f64>],
        r: &[Vec<f64>],
    ) -> Result<Vec<f64>, String> {
        self.predict(f, q);
        self.update(z, h, r)
    }
}

/// Extended Kalman filter for `x_{t+1} = f(x_t) + w`,
/// `z_t = h(x_t) + v`, with analytic Jacobians.
#[must_use]
#[derive(Clone, Debug)]
pub struct ExtendedKalmanFilter {
    /// State estimate.
    pub x: Vec<f64>,
    /// State covariance.
    pub p: Vec<Vec<f64>>,
}

impl ExtendedKalmanFilter {
    /// Create a filter with initial state `x0` and covariance `P0`.
    pub fn new(x0: Vec<f64>, p0: Vec<Vec<f64>>) -> Result<Self, String> {
        KalmanFilter::new(x0, p0).map(|kf| Self { x: kf.x, p: kf.p })
    }

    /// Prediction step: `x = f(x)`, `P = F·P·Fᵀ + Q` with `F` the Jacobian
    /// of `f` evaluated at the current state.
    pub fn predict(
        &mut self,
        f: impl Fn(&[f64]) -> Vec<f64>,
        f_jac: impl Fn(&[f64]) -> Vec<Vec<f64>>,
        q: &[Vec<f64>],
    ) {
        self.x = f(&self.x);
        let fj = f_jac(&self.x);
        let fpf = mat_mul(&mat_mul(&fj, &self.p), &mat_t(&fj));
        self.p = mat_add(&fpf, q);
    }

    /// Update step with `z = h(x) + v`. Returns the innovation.
    ///
    /// # Errors
    /// Returns an error if `S = H·P·Hᵀ + R` is singular.
    pub fn update(
        &mut self,
        z: &[f64],
        h: impl Fn(&[f64]) -> Vec<f64>,
        h_jac: impl Fn(&[f64]) -> Vec<Vec<f64>>,
        r: &[Vec<f64>],
    ) -> Result<Vec<f64>, String> {
        let hx = h(&self.x);
        let innovation = vec_sub(z, &hx);
        let hj = h_jac(&self.x);
        let s = mat_add(&mat_mul(&mat_mul(&hj, &self.p), &mat_t(&hj)), r);
        let s_inv = mat_inv(&s)?;
        let k = mat_mul(&mat_mul(&self.p, &mat_t(&hj)), &s_inv);
        self.x = vec_add(&self.x, &mat_vec_mul(&k, &innovation));
        let kh = mat_mul(&k, &hj);
        let n = self.x.len();
        let idn: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let ikh = mat_sub(&idn, &kh);
        let term1 = mat_mul(&mat_mul(&ikh, &self.p), &mat_t(&ikh));
        let term2 = mat_mul(&mat_mul(&k, r), &mat_t(&k));
        self.p = mat_add(&term1, &term2);
        Ok(innovation)
    }
}

/// Unscented Kalman filter: no Jacobians required, state propagation through
/// sigma points.
#[must_use]
#[derive(Clone, Debug)]
pub struct UnscentedKalmanFilter {
    /// State estimate.
    pub x: Vec<f64>,
    /// State covariance.
    pub p: Vec<Vec<f64>>,
    /// Sigma-point spread (default 1e-3).
    pub alpha: f64,
    /// Prior kurtosis of `x` (Gaussian: 2).
    pub beta: f64,
    /// Secondary scaling parameter.
    pub kappa: f64,
}

impl UnscentedKalmanFilter {
    /// Create a filter with initial state `x0` and covariance `P0`.
    pub fn new(x0: Vec<f64>, p0: Vec<Vec<f64>>) -> Result<Self, String> {
        let _ = KalmanFilter::new(x0.clone(), p0.clone())?;
        Ok(Self {
            x: x0,
            p: p0,
            alpha: 1e-3,
            beta: 2.0,
            kappa: 0.0,
        })
    }

    /// Generate sigma points and their weights.
    fn sigma_points(&self) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let n = self.x.len();
        let lambda = self.alpha * self.alpha * (n as f64 + self.kappa) - n as f64;
        let scale = (n as f64 + lambda).max(0.0);
        let l = cholesky(&self.p).unwrap_or_else(|_| {
            // Fallback: tiny jitter on the diagonal.
            let mut jitter = self.p.clone();
            for (i, row) in jitter.iter_mut().enumerate() {
                row[i] += 1e-9;
            }
            cholesky(&jitter).unwrap_or_else(|_| {
                let mut d = vec![vec![0.0; n]; n];
                for i in 0..n {
                    d[i][i] = 1e-6;
                }
                cholesky(&d).unwrap()
            })
        });
        let mut points = vec![self.x.clone()];
        for i in 0..n {
            let plus: Vec<f64> = (0..n).map(|k| self.x[k] + scale.sqrt() * l[k][i]).collect();
            let minus: Vec<f64> = (0..n).map(|k| self.x[k] - scale.sqrt() * l[k][i]).collect();
            points.push(plus);
            points.push(minus);
        }
        let wm0 = lambda / (n as f64 + lambda);
        let wc0 = wm0 + (1.0 - self.alpha * self.alpha + self.beta);
        let wi = 1.0 / (2.0 * (n as f64 + lambda));
        let mut wm = vec![wm0];
        let mut wc = vec![wc0];
        for _ in 1..points.len() {
            wm.push(wi);
            wc.push(wi);
        }
        (points, wm, wc)
    }

    /// Prediction step: propagate sigma points through `f`, recompute mean
    /// and covariance, add process noise `Q`.
    pub fn predict(&mut self, f: impl Fn(&[f64]) -> Vec<f64>, q: &[Vec<f64>]) {
        let (points, wm, wc) = self.sigma_points();
        let n = self.x.len();
        let ys: Vec<Vec<f64>> = points.iter().map(|p| f(p)).collect();
        let mut mean = vec![0.0; n];
        for (y, &w) in ys.iter().zip(&wm) {
            for (k, &yk) in y.iter().enumerate() {
                mean[k] += w * yk;
            }
        }
        let mut cov = q.to_vec();
        for (y, &w) in ys.iter().zip(&wc) {
            let d: Vec<f64> = y.iter().zip(&mean).map(|(&a, &b)| a - b).collect();
            let od = outer(&d);
            for i in 0..n {
                for j in 0..n {
                    cov[i][j] += w * od[i][j];
                }
            }
        }
        self.x = mean;
        self.p = cov;
    }

    /// Update step with `z = h(x) + v`. Returns the innovation.
    ///
    /// # Errors
    /// Returns an error if the innovation covariance is singular.
    pub fn update(
        &mut self,
        z: &[f64],
        h: impl Fn(&[f64]) -> Vec<f64>,
        r: &[Vec<f64>],
    ) -> Result<Vec<f64>, String> {
        let (points, wm, wc) = self.sigma_points();
        let n = self.x.len();
        let m = z.len();
        let zs: Vec<Vec<f64>> = points.iter().map(|p| h(p)).collect();
        let mut z_mean = vec![0.0; m];
        for (y, &w) in zs.iter().zip(&wm) {
            for (k, &yk) in y.iter().enumerate() {
                z_mean[k] += w * yk;
            }
        }
        let mut pzz = r.to_vec();
        let mut pxz = vec![vec![0.0; m]; n];
        for (i, (y, &w)) in zs.iter().zip(&wc).enumerate() {
            let dz: Vec<f64> = y.iter().zip(&z_mean).map(|(&a, &b)| a - b).collect();
            let odz = outer(&dz);
            for a in 0..m {
                for b in 0..m {
                    pzz[a][b] += w * odz[a][b];
                }
            }
            let dx: Vec<f64> = points[i].iter().zip(&self.x).map(|(&a, &b)| a - b).collect();
            for a in 0..n {
                for b in 0..m {
                    pxz[a][b] += w * dx[a] * dz[b];
                }
            }
        }
        let pzz_inv = mat_inv(&pzz)?;
        let k = mat_mul(&pxz, &pzz_inv);
        let innovation = vec_sub(z, &z_mean);
        self.x = vec_add(&self.x, &mat_vec_mul(&k, &innovation));
        let kp = mat_mul(&k, &mat_t(&pxz));
        self.p = mat_sub(&self.p, &kp);
        Ok(innovation)
    }
}

/// Rauch-Tung-Striebel smoothing over a complete forward pass. Takes the
/// filtered states/covariances, one-step predictions and the (constant)
/// transition matrix `F`; returns `(smoothed states, smoothed covariances)`.
///
/// # Errors
/// Returns an error if the predicted covariances are singular or the input
/// shapes are inconsistent.
pub fn rts_smoother(
    x_filt: &[Vec<f64>],
    p_filt: &[Vec<Vec<f64>>],
    x_pred: &[Vec<f64>],
    p_pred: &[Vec<Vec<f64>>],
    f: &[Vec<f64>],
) -> Result<(Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>), String> {
    let n = x_filt.len();
    if n == 0 || x_pred.len() != n || p_filt.len() != n || p_pred.len() != n {
        return Err("rts_smoother input shapes do not match".into());
    }
    let mut xs = x_filt.to_vec();
    let mut ps = p_filt.to_vec();
    for t in (0..n - 1).rev() {
        let f_t = mat_t(f);
        let fp = mat_mul(&mat_mul(&p_filt[t], &f_t), &mat_inv(&p_pred[t + 1])?);
        let xs_t1 = vec_sub(&xs[t + 1], &x_pred[t + 1]);
        xs[t] = vec_add(&x_filt[t], &mat_vec_mul(&fp, &xs_t1));
        let inner = mat_sub(&ps[t + 1], &p_pred[t + 1]);
        ps[t] = mat_add(
            &p_filt[t],
            &mat_mul(&mat_mul(&fp, &inner), &mat_t(&fp)),
        );
    }
    Ok((xs, ps))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::Normal;
    use crate::rng::Rng;

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
            / (a.len() as f64).max(1.0)
    }

    #[test]
    fn kalman_random_walk_tracks() {
        let mut rng = Rng::new(1);
        let q: f64 = 0.01;
        let r: f64 = 1.0;
        let mut true_x = 0.0;
        let mut zs = Vec::new();
        let mut truth = Vec::new();
        for _ in 0..200 {
            true_x += q.sqrt() * Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng);
            zs.push(true_x + r.sqrt() * Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng));
            truth.push(true_x);
        }
        let mut kf = KalmanFilter::new(vec![0.0], vec![vec![100.0]]).unwrap();
        let f = vec![vec![1.0]];
        let h = vec![vec![1.0]];
        let qq = vec![vec![q]];
        let rr = vec![vec![r]];
        let mut estimates = Vec::new();
        for z in &zs {
            kf.step(&[*z], &f, &h, &qq, &rr).unwrap();
            estimates.push(kf.x[0]);
        }
        let err = rmse(&estimates, &truth);
        assert!(err < 0.35, "rmse {err}");
        assert!(kf.p[0][0] < 0.5, "steady-state variance {}", kf.p[0][0]);
    }

    #[test]
    fn kalman_constant_velocity_recovers_speed() {
        let mut rng = Rng::new(2);
        let dt = 0.1;
        let v_true = 3.0;
        let mut x = 0.0;
        let mut truth = Vec::new();
        let mut zs = Vec::new();
        for _ in 0..150 {
            x += v_true * dt;
            truth.push(x);
            zs.push(x + 0.5 * Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng));
        }
        let f = vec![vec![1.0, dt], vec![0.0, 1.0]];
        let h = vec![vec![1.0, 0.0]];
        let q = vec![vec![0.001, 0.0], vec![0.0, 0.001]];
        let r = vec![vec![0.25]];
        let mut kf = KalmanFilter::new(vec![0.0, 0.0], vec![vec![10.0, 0.0], vec![0.0, 10.0]]).unwrap();
        for z in &zs {
            kf.step(&[*z], &f, &h, &q, &r).unwrap();
        }
        assert!(
            (kf.x[1] - v_true).abs() < 0.3,
            "velocity {}",
            kf.x[1]
        );
        assert!((kf.x[0] - truth[149]).abs() < 1.0);
    }

    #[test]
    fn ekf_and_ukf_converge_on_nonlinear_model() {
        // x_{t+1} = 0.5x + 0.1x²/(1+x²) + w, z = x³ + v.
        let mut rng = Rng::new(3);
        let f = |x: &[f64]| vec![0.5 * x[0] + 0.1 * x[0] * x[0] / (1.0 + x[0] * x[0])];
        let f_jac = |x: &[f64]| {
            let x = x[0];
            vec![vec![0.5 + 0.2 * x / (1.0 + x * x).powi(2)]]
        };
        let h = |x: &[f64]| vec![x[0].powi(3)];
        let h_jac = |x: &[f64]| vec![vec![3.0 * x[0] * x[0]]];
        let q = vec![vec![0.01]];
        let r = vec![vec![0.25]];

        let mut true_x: f64 = 1.0;
        let mut zs = Vec::new();
        let mut truth = Vec::new();
        for _ in 0..120 {
            true_x = 0.5 * true_x + 0.1 * true_x * true_x / (1.0 + true_x * true_x);
            truth.push(true_x);
            zs.push(true_x.powi(3) + 0.5 * Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng));
        }

        let mut ekf = ExtendedKalmanFilter::new(vec![0.0], vec![vec![5.0]]).unwrap();
        for z in &zs {
            ekf.predict(&f, &f_jac, &q);
            ekf.update(&[*z], &h, &h_jac, &r).unwrap();
        }
        assert!(
            (ekf.x[0] - truth[119]).abs() < 0.25,
            "ekf {}",
            ekf.x[0]
        );

        let mut ukf = UnscentedKalmanFilter::new(vec![0.0], vec![vec![5.0]]).unwrap();
        for z in &zs {
            ukf.predict(&f, &q);
            ukf.update(&[*z], &h, &r).unwrap();
        }
        assert!(
            (ukf.x[0] - truth[119]).abs() < 0.25,
            "ukf {}",
            ukf.x[0]
        );
    }

    #[test]
    fn rts_smoother_beats_filter() {
        let mut rng = Rng::new(4);
        let q: f64 = 0.05;
        let r: f64 = 1.0;
        let mut true_x = 0.0;
        let mut zs = Vec::new();
        let mut truth = Vec::new();
        for _ in 0..150 {
            true_x += q.sqrt() * Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng);
            zs.push(true_x + r.sqrt() * Normal { mu: 0.0, sigma: 1.0 }.sample(&mut rng));
            truth.push(true_x);
        }
        let f = vec![vec![1.0]];
        let h = vec![vec![1.0]];
        let qq = vec![vec![q]];
        let rr = vec![vec![r]];
        let mut kf = KalmanFilter::new(vec![0.0], vec![vec![50.0]]).unwrap();
        let mut x_filt = Vec::new();
        let mut p_filt = Vec::new();
        let mut x_pred = Vec::new();
        let mut p_pred = Vec::new();
        for z in &zs {
            kf.predict(&f, &qq);
            x_pred.push(kf.x.clone());
            p_pred.push(kf.p.clone());
            kf.update(&[*z], &h, &rr).unwrap();
            x_filt.push(kf.x.clone());
            p_filt.push(kf.p.clone());
        }
        let (xs, _ps) = rts_smoother(&x_filt, &p_filt, &x_pred, &p_pred, &f).unwrap();
        let filt_err = rmse(&x_filt.iter().map(|v| v[0]).collect::<Vec<_>>(), &truth);
        let sm_err = rmse(&xs.iter().map(|v| v[0]).collect::<Vec<_>>(), &truth);
        assert!(sm_err <= filt_err + 1e-9, "smooth {sm_err} vs filter {filt_err}");
        // Smoother error should be meaningfully smaller.
        assert!(sm_err < filt_err * 0.8 + 1e-9, "smooth {sm_err} vs filter {filt_err}");
    }
}
