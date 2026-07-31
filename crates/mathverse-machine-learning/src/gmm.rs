//! Gaussian Mixture Model via Expectation-Maximization.

use mathverse_core::error::MathResult;

/// GMM result.
#[derive(Debug, Clone)]
pub struct GmmResult {
    pub weights: Vec<f64>,
    pub means: Vec<Vec<f64>>,
    pub covariances: Vec<Vec<Vec<f64>>>,
    pub responsibilities: Vec<Vec<f64>>,
    pub log_likelihood: f64,
    pub n_iters: usize,
}

/// Fit a Gaussian Mixture Model via EM.
pub fn fit_gmm(x: &[Vec<f64>], k: usize, max_iters: usize, tol: f64) -> MathResult<GmmResult> {
    let n = x.len();
    let p = x[0].len();
    // Initialize: k-means then use those
    let mut weights = vec![1.0 / k as f64; k];
    let mut means = Vec::new();
    let mut covariances = Vec::new();
    // Use first k points as initial means (simple)
    for i in 0..k {
        means.push(x[i * n / k].clone());
        covariances.push(vec![vec![1.0; p]; p]);
    }
    let mut responsibilities = vec![vec![0.0; k]; n];
    let mut prev_ll = f64::NEG_INFINITY;

    for iter in 0..max_iters {
        // E-step
        for i in 0..n {
            let mut max_log = f64::NEG_INFINITY;
            let mut log_probs = Vec::with_capacity(k);
            for c in 0..k {
                let lp = weights[c].ln() + log_gaussian(&x[i], &means[c], &covariances[c]);
                log_probs.push(lp);
                if lp > max_log { max_log = lp; }
            }
            let sum_exp: f64 = log_probs.iter().map(|lp| (lp - max_log).exp()).sum();
            for c in 0..k {
                responsibilities[i][c] = (log_probs[c] - max_log).exp() / sum_exp;
            }
        }
        // M-step
        for c in 0..k {
            let nk: f64 = (0..n).map(|i| responsibilities[i][c]).sum();
            if nk < 1e-10 { continue; }
            weights[c] = nk / n as f64;
            for j in 0..p {
                means[c][j] = (0..n).map(|i| responsibilities[i][c] * x[i][j]).sum::<f64>() / nk;
            }
            for j1 in 0..p {
                for j2 in 0..p {
                    covariances[c][j1][j2] = (0..n).map(|i| {
                        responsibilities[i][c] * (x[i][j1] - means[c][j1]) * (x[i][j2] - means[c][j2])
                    }).sum::<f64>() / nk;
                    // Regularize diagonal
                    if j1 == j2 { covariances[c][j1][j2] += 1e-6; }
                }
            }
        }
        // Log-likelihood
        let ll: f64 = (0..n).map(|i| {
            let probs: Vec<f64> = (0..k).map(|c| {
                weights[c] * gaussian(&x[i], &means[c], &covariances[c])
            }).collect();
            probs.iter().sum::<f64>().ln()
        }).sum();
        if (ll - prev_ll).abs() < tol {
            return Ok(GmmResult { weights, means, covariances, responsibilities, log_likelihood: ll, n_iters: iter + 1 });
        }
        prev_ll = ll;
    }
    let ll: f64 = (0..n).map(|i| {
        let sum: f64 = (0..k).map(|c| {
            weights[c] * gaussian(&x[i], &means[c], &covariances[c])
        }).sum();
        sum.ln()
    }).sum();
    Ok(GmmResult { weights, means, covariances, responsibilities, log_likelihood: ll, n_iters: max_iters })
}

/// Predict most likely component for each sample.
pub fn predict(r: &GmmResult, x: &[Vec<f64>]) -> Vec<usize> {
    x.iter().map(|xi| {
        let probs: Vec<f64> = (0..r.weights.len()).map(|c| {
            r.weights[c] * gaussian(xi, &r.means[c], &r.covariances[c])
        }).collect();
        probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }).collect()
}

fn gaussian(x: &[f64], mean: &[f64], cov: &[Vec<f64>]) -> f64 {
    log_gaussian(x, mean, cov).exp()
}

fn log_gaussian(x: &[f64], mean: &[f64], cov: &[Vec<f64>]) -> f64 {
    let p = x.len();
    let det = determinant_2d(cov).max(1e-20);
    let mut quad = 0.0;
    for i in 0..p {
        for j in 0..p {
            quad += (x[i] - mean[i]) * cov_inv_ij(cov, i, j, det) * (x[j] - mean[j]);
        }
    }
    -0.5 * (p as f64 * (2.0 * std::f64::consts::PI).ln() + det.ln() + quad)
}

fn determinant_2d(m: &[Vec<f64>]) -> f64 {
    let p = m.len();
    if p == 1 { return m[0][0]; }
    if p == 2 { return m[0][0] * m[1][1] - m[0][1] * m[1][0]; }
    // LU decomposition for general case
    let mut det = 1.0;
    let mut lu = m.to_vec();
    for i in 0..p {
        let mut max_val = lu[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..p {
            if lu[k][i].abs() > max_val { max_val = lu[k][i].abs(); max_row = k; }
        }
        if max_row != i { lu.swap(i, max_row); det = -det; }
        if lu[i][i].abs() < 1e-15 { return 0.0; }
        det *= lu[i][i];
        for k in (i + 1)..p {
            let factor = lu[k][i] / lu[i][i];
            for j in (i + 1)..p { lu[k][j] -= factor * lu[i][j]; }
        }
    }
    det
}

fn cov_inv_ij(cov: &[Vec<f64>], i: usize, j: usize, det: f64) -> f64 {
    let p = cov.len();
    if p == 1 { return 1.0 / cov[0][0]; }
    if p == 2 {
        return if i == j { cov[1 - i][1 - j] / det } else { -cov[i][j] / det };
    }
    // General: cofactor / det
    let cofactor = cofactor(cov, i, j);
    cofactor / det
}

fn cofactor(m: &[Vec<f64>], row: usize, col: usize) -> f64 {
    let p = m.len();
    let mut sub = Vec::new();
    for i in 0..p {
        if i == row { continue; }
        for j in 0..p {
            if j == col { continue; }
            sub.push(m[i][j]);
        }
    }
    let sign = if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
    let n = p - 1;
    if n == 1 { return sign * sub[0]; }
    if n == 2 { return sign * (sub[0] * sub[3] - sub[1] * sub[2]); }
    sign * det_small(&sub, n)
}

fn det_small(m: &[f64], n: usize) -> f64 {
    if n == 1 { return m[0]; }
    if n == 2 { return m[0] * m[3] - m[1] * m[2]; }
    let mut det = 0.0;
    for j in 0..n {
        let mut sub = Vec::new();
        for i in 1..n {
            for k in 0..n {
                if k == j { continue; }
                sub.push(m[i * n + k]);
            }
        }
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        det += sign * m[j] * det_small(&sub, n - 1);
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters() {
        let mut x = Vec::new();
        for _ in 0..20 { x.push(vec![randn(), 0.0]); }
        for _ in 0..20 { x.push(vec![randn() + 10.0, 0.0]); }
        let r = fit_gmm(&x, 2, 50, 1e-6).unwrap();
        let labels = predict(&r, &x);
        assert_eq!(labels[0], labels[5]);
        assert_ne!(labels[0], labels[25]);
    }

    fn randn() -> f64 {
        use std::cell::Cell;
        thread_local! { static S: Cell<u64> = Cell::new(0x1234_5678); }
        S.with(|s| {
            let mut x = s.get();
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            s.set(x);
            let u1 = (x as f64) / (u64::MAX as f64).max(1e-30);
            let u2 = ((x >> 32) as f64) / (u64::MAX as f64).max(1e-30);
            (-2.0 * u1.max(1e-30).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
    }
}
