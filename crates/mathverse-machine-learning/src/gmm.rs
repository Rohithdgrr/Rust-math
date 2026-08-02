//! Gaussian Mixture Model via Expectation-Maximization.

use mathverse_core::error::MathResult;

/// GMM result.
#[derive(Debug, Clone)]
pub struct GmmResult {
    /// Mixing weight for each component.
    pub weights: Vec<f64>,
    /// Mean vector for each component.
    pub means: Vec<Vec<f64>>,
    /// Covariance matrix for each component.
    pub covariances: Vec<Vec<Vec<f64>>>,
    /// Per-sample responsibility of each component.
    pub responsibilities: Vec<Vec<f64>>,
    /// Final log-likelihood of the data under the fitted model.
    pub log_likelihood: f64,
    /// Number of EM iterations executed.
    pub n_iters: usize,
}

/// Fit a Gaussian Mixture Model via EM.
#[must_use]
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
                if lp > max_log {
                    max_log = lp;
                }
            }
            let sum_exp: f64 = log_probs.iter().map(|lp| (lp - max_log).exp()).sum();
            for c in 0..k {
                responsibilities[i][c] = (log_probs[c] - max_log).exp() / sum_exp;
            }
        }
        // M-step
        for c in 0..k {
            let nk: f64 = (0..n).map(|i| responsibilities[i][c]).sum();
            if nk < 1e-10 {
                continue;
            }
            weights[c] = nk / n as f64;
            for j in 0..p {
                means[c][j] = (0..n)
                    .map(|i| responsibilities[i][c] * x[i][j])
                    .sum::<f64>()
                    / nk;
            }
            for j1 in 0..p {
                for j2 in 0..p {
                    covariances[c][j1][j2] = (0..n)
                        .map(|i| {
                            responsibilities[i][c]
                                * (x[i][j1] - means[c][j1])
                                * (x[i][j2] - means[c][j2])
                        })
                        .sum::<f64>()
                        / nk;
                    // Regularize diagonal
                    if j1 == j2 {
                        covariances[c][j1][j2] += 1e-6;
                    }
                }
            }
        }
        // Log-likelihood
        let ll: f64 = (0..n)
            .map(|i| {
                let probs: Vec<f64> = (0..k)
                    .map(|c| weights[c] * gaussian(&x[i], &means[c], &covariances[c]))
                    .collect();
                probs.iter().sum::<f64>().ln()
            })
            .sum();
        if (ll - prev_ll).abs() < tol {
            return Ok(GmmResult {
                weights,
                means,
                covariances,
                responsibilities,
                log_likelihood: ll,
                n_iters: iter + 1,
            });
        }
        prev_ll = ll;
    }
    let ll: f64 = (0..n)
        .map(|i| {
            let sum: f64 = (0..k)
                .map(|c| weights[c] * gaussian(&x[i], &means[c], &covariances[c]))
                .sum();
            sum.ln()
        })
        .sum();
    Ok(GmmResult {
        weights,
        means,
        covariances,
        responsibilities,
        log_likelihood: ll,
        n_iters: max_iters,
    })
}

/// Predict most likely component for each sample.
#[must_use]
pub fn predict(r: &GmmResult, x: &[Vec<f64>]) -> Vec<usize> {
    x.iter()
        .map(|xi| {
            let probs: Vec<f64> = (0..r.weights.len())
                .map(|c| r.weights[c] * gaussian(xi, &r.means[c], &r.covariances[c]))
                .collect();
            probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

fn gaussian(x: &[f64], mean: &[f64], cov: &[Vec<f64>]) -> f64 {
    log_gaussian(x, mean, cov).exp()
}

fn log_gaussian(x: &[f64], mean: &[f64], cov: &[Vec<f64>]) -> f64 {
    let p = x.len();
    
    // Use Cholesky decomposition for numerical stability
    if let Ok((l, log_det)) = cholesky_with_logdet(cov) {
        // Solve L * y = (x - mean) using forward substitution
        let diff: Vec<f64> = x.iter().zip(mean.iter()).map(|(xi, mi)| xi - mi).collect();
        let y = forward_substitution(&l, &diff);
        
        // Compute quadratic form: y^T * y
        let quad: f64 = y.iter().map(|yi| yi * yi).sum();
        
        -0.5 * (p as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + quad)
    } else {
        // Fallback to slower but more robust method if Cholesky fails
        let det = determinant_lu(cov).max(1e-20);
        let inv = matrix_inverse(cov, det);
        let mut quad = 0.0;
        for i in 0..p {
            for j in 0..p {
                quad += (x[i] - mean[i]) * inv[i][j] * (x[j] - mean[j]);
            }
        }
        -0.5 * (p as f64 * (2.0 * std::f64::consts::PI).ln() + det.ln() + quad)
    }
}

/// Cholesky decomposition with log determinant for SPD matrices.
/// Returns (L, log(det)) where A = L * L^T and det(A) = (prod(diag(L)))^2
fn cholesky_with_logdet(a: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, f64), ()> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            
            if j == i {
                // Diagonal elements
                for k in 0..j {
                    sum += l[j][k] * l[j][k];
                }
                let diag = a[j][j] - sum;
                if diag <= 0.0 {
                    return Err(()); // Not positive definite
                }
                l[j][j] = diag.sqrt();
            } else {
                // Non-diagonal elements
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    
    // Log determinant = 2 * sum(log(diag(L)))
    let log_det: f64 = (0..n).map(|i| l[i][i].ln()).sum::<f64>() * 2.0;
    
    Ok((l, log_det))
}

/// Forward substitution: solve L * y = b for y
fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i][i];
    }
    
    y
}

/// LU decomposition for determinant (fallback)
fn determinant_lu(m: &[Vec<f64>]) -> f64 {
    let p = m.len();
    if p == 1 {
        return m[0][0];
    }
    if p == 2 {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }
    
    let mut det = 1.0;
    let mut lu = m.to_vec();
    
    for i in 0..p {
        // Partial pivoting
        let mut max_val = lu[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..p {
            if lu[k][i].abs() > max_val {
                max_val = lu[k][i].abs();
                max_row = k;
            }
        }
        if max_row != i {
            lu.swap(i, max_row);
            det = -det;
        }
        
        if lu[i][i].abs() < 1e-15 {
            return 0.0;
        }
        
        det *= lu[i][i];
        
        for k in (i + 1)..p {
            let factor = lu[k][i] / lu[i][i];
            for j in (i + 1)..p {
                lu[k][j] -= factor * lu[i][j];
            }
        }
    }
    
    det
}

/// Matrix inverse using Gaussian elimination (fallback)
fn matrix_inverse(m: &[Vec<f64>], det: f64) -> Vec<Vec<f64>> {
    let n = m.len();
    if n == 1 {
        return vec![vec![1.0 / m[0][0]]];
    }
    if n == 2 {
        return vec![
            vec![m[1][1] / det, -m[0][1] / det],
            vec![-m[1][0] / det, m[0][0] / det],
        ];
    }
    
    // Gaussian elimination for larger matrices
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = m[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    
    // Forward elimination
    for i in 0..n {
        // Pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);
        
        let pivot = aug[i][i];
        if pivot.abs() < 1e-15 {
            continue; // Singular matrix
        }
        
        // Normalize row
        for j in 0..2 * n {
            aug[i][j] /= pivot;
        }
        
        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..2 * n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }
    
    // Extract inverse
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters() {
        let mut x = Vec::new();
        for _ in 0..20 {
            x.push(vec![randn(), 0.0]);
        }
        for _ in 0..20 {
            x.push(vec![randn() + 10.0, 0.0]);
        }
        let r = fit_gmm(&x, 2, 50, 1e-6).unwrap();
        let labels = predict(&r, &x);
        assert_eq!(labels[0], labels[5]);
        assert_ne!(labels[0], labels[25]);
    }

    #[test]
    fn single_component() {
        let x: Vec<Vec<f64>> = (0..50).map(|_| vec![0.1, 0.2]).collect();
        let r = fit_gmm(&x, 1, 20, 1e-6).unwrap();
        assert_eq!(r.weights.len(), 1);
    }

    #[test]
    fn test_cholesky_stability() {
        // Test that Cholesky works for well-conditioned matrices
        // Fixed data: cluster A around (0,0), cluster B around (5,5)
        let mut x = Vec::new();
        for i in 0..20 {
            x.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
        }
        for i in 0..20 {
            x.push(vec![5.0 + i as f64 * 0.1, 5.0 + i as f64 * 0.1]);
        }
        
        let r = fit_gmm(&x, 2, 50, 1e-6).unwrap();
        assert_eq!(r.weights.len(), 2);
        // All points in the same cluster should share a label
        let labels = predict(&r, &x);
        assert_eq!(labels[0], labels[10]); // both in cluster A
        assert_eq!(labels[20], labels[30]); // both in cluster B
    }

    fn randn() -> f64 {
        use std::cell::Cell;
        thread_local! { static S: Cell<u64> = Cell::new(0x1234_5678); }
        S.with(|s| {
            let mut x = s.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.set(x);
            let u1 = (x as f64) / (u64::MAX as f64).max(1e-30);
            let u2 = ((x >> 32) as f64) / (u64::MAX as f64).max(1e-30);
            (-2.0 * u1.max(1e-30).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
    }
}
