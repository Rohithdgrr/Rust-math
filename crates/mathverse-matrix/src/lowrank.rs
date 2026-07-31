//! Low-rank approximation using truncated SVD and other methods.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Low-rank approximation result.
#[derive(Debug, Clone)]
pub struct LowRankApproximation {
    pub u: Matrix,      // Left singular vectors (truncated)
    pub s: Vec<f64>,    // Singular values (truncated)
    pub vt: Matrix,     // Right singular vectors (truncated)
    pub rank: usize,    // Approximation rank
    pub error: f64,     // Approximation error
}

/// Low-rank approximation methods.
pub struct LowRankApprox;

impl LowRankApprox {
    /// Truncated SVD approximation: A ≈ U_k Σ_k V_k^T.
    pub fn truncated_svd(m: &Matrix, k: usize) -> MathResult<LowRankApproximation> {
        let svd = m.svd()?;
        let k = k.min(svd.s.len());
        
        // Truncate to rank k
        let mut u_trunc = Matrix::zeros(m.rows, k);
        let mut s_trunc = Vec::with_capacity(k);
        let mut vt_trunc = Matrix::zeros(k, m.cols);
        
        for j in 0..k {
            s_trunc.push(svd.s[j]);
            for i in 0..m.rows {
                u_trunc.set(i, j, svd.u.get(i, j));
            }
            for i in 0..m.cols {
                vt_trunc.set(j, i, svd.vt.get(j, i));
            }
        }
        
        // Compute approximation error
        let approx = u_trunc.mul(&Matrix::diagonal(&s_trunc))?.mul(&vt_trunc)?;
        let error = m.sub(&approx)?;
        let error_norm = crate::norms::MatrixNorms::frobenius(&error);
        
        Ok(LowRankApproximation {
            u: u_trunc,
            s: s_trunc,
            vt: vt_trunc,
            rank: k,
            error: error_norm,
        })
    }

    /// Low-rank approximation based on energy threshold.
    pub fn energy_based(m: &Matrix, energy_threshold: f64) -> MathResult<LowRankApproximation> {
        let svd = m.svd()?;
        let total_energy: f64 = svd.s.iter().map(|&s| s * s).sum();
        
        let mut cumulative = 0.0;
        let mut k = 0;
        
        for &s in &svd.s {
            cumulative += s * s;
            k += 1;
            if cumulative / total_energy >= energy_threshold {
                break;
            }
        }
        
        Self::truncated_svd(m, k)
    }

    /// Low-rank approximation based on tolerance.
    pub fn tolerance_based(m: &Matrix, tolerance: f64) -> MathResult<LowRankApproximation> {
        let svd = m.svd()?;
        let sigma_max = svd.s[0];
        
        let k = svd.s.iter()
            .take_while(|&&s| s > tolerance * sigma_max)
            .count();
        
        Self::truncated_svd(m, k.max(1))
    }

    /// Randomized SVD for large matrices.
    pub fn randomized_svd(m: &Matrix, k: usize, oversampling: usize) -> MathResult<LowRankApproximation> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        let target_rank = k + oversampling;
        
        // Generate random projection matrix
        let mut rng = crate::rng::Rng::new(42);
        let omega = Matrix::zeros(m_cols, target_rank);
        for i in 0..m_cols {
            for j in 0..target_rank {
                omega.set(i, j, rng.uniform() * 2.0 - 1.0);
            }
        }
        
        // Form Y = A * Omega
        let y = m.mul(&omega)?;
        
        // QR decomposition of Y
        let qr_y = y.qr()?;
        let q = qr_y.q;
        
        // Form B = Q^T * A
        let qt = q.transpose();
        let b = qt.mul(m)?;
        
        // SVD of B (smaller matrix)
        let svd_b = b.svd()?;
        
        // Truncate to rank k
        let k_actual = k.min(svd_b.s.len());
        
        // Transform back: U = Q * U_B
        let mut u_trunc = Matrix::zeros(m_rows, k_actual);
        for j in 0..k_actual {
            for i in 0..m_rows {
                let mut sum = 0.0;
                for l in 0..q.cols {
                    sum += q.get(i, l) * svd_b.u.get(l, j);
                }
                u_trunc.set(i, j, sum);
            }
        }
        
        let s_trunc = svd_b.s[..k_actual].to_vec();
        let vt_trunc = Matrix::zeros(k_actual, m_cols);
        for i in 0..k_actual {
            for j in 0..m_cols {
                vt_trunc.set(i, j, svd_b.vt.get(i, j));
            }
        }
        
        // Compute error
        let approx = u_trunc.mul(&Matrix::diagonal(&s_trunc))?.mul(&vt_trunc)?;
        let error = m.sub(&approx)?;
        let error_norm = crate::norms::MatrixNorms::frobenius(&error);
        
        Ok(LowRankApproximation {
            u: u_trunc,
            s: s_trunc,
            vt: vt_trunc,
            rank: k_actual,
            error: error_norm,
        })
    }

    /// Reconstruct matrix from low-rank approximation.
    pub fn reconstruct(approx: &LowRankApproximation) -> MathResult<Matrix> {
        approx.u.mul(&Matrix::diagonal(&approx.s))?.mul(&approx.vt)
    }

    /// Compute relative error of approximation.
    pub fn relative_error(m: &Matrix, approx: &LowRankApproximation) -> f64 {
        let original_norm = crate::norms::MatrixNorms::frobenius(m);
        if original_norm > 0.0 {
            approx.error / original_norm
        } else {
            approx.error
        }
    }
}

/// Rank selection strategies.
pub struct RankSelection;

impl RankSelection {
    /// Optimal rank based on singular value gap.
    pub fn optimal_gap(singular_values: &[f64]) -> usize {
        if singular_values.len() < 2 {
            return singular_values.len();
        }
        
        let mut max_gap = 0.0;
        let mut optimal_rank = singular_values.len();
        
        for i in 0..(singular_values.len() - 1) {
            let gap = (singular_values[i] - singular_values[i + 1]).abs();
            if gap > max_gap {
                max_gap = gap;
                optimal_rank = i + 1;
            }
        }
        
        optimal_rank
    }

    /// Rank based on explained variance ratio.
    pub fn explained_variance(singular_values: &[f64], threshold: f64) -> usize {
        let total: f64 = singular_values.iter().map(|&s| s * s).sum();
        let mut cumulative = 0.0;
        
        for (i, &s) in singular_values.iter().enumerate() {
            cumulative += s * s;
            if cumulative / total >= threshold {
                return i + 1;
            }
        }
        
        singular_values.len()
    }

    /// Rank using Gavish-Donoho method.
    pub fn gavish_donoho(singular_values: &[f64], matrix_size: usize) -> usize {
        let n = matrix_size as f64;
        let beta = singular_values.len() as f64 / n;
        let tau = 0.56 * beta.powf(3.0) - 0.95 * beta.powf(1.5) + 1.43 * beta.powf(0.5);
        let threshold = tau * (n as f64).sqrt();
        
        singular_values.iter()
            .take_while(|&&s| s > threshold)
            .count()
            .max(1)
    }

    /// Cross-validation for rank selection.
    pub fn cross_validation(
        m: &Matrix,
        max_rank: usize,
        folds: usize,
    ) -> MathResult<usize> {
        let mut best_rank = 1;
        let mut best_error = f64::INFINITY;
        
        for rank in 1..=max_rank.min(m.rows.min(m.cols)) {
            let approx = LowRankApprox::truncated_svd(m, rank)?;
            let error = LowRankApprox::relative_error(m, &approx);
            
            if error < best_error {
                best_error = error;
                best_rank = rank;
            }
        }
        
        Ok(best_rank)
    }
}

/// Matrix completion (missing data imputation).
pub struct MatrixCompletion;

impl MatrixCompletion {
    /// Soft-impute algorithm for matrix completion.
    pub fn soft_impute(
        observed: &Matrix,
        mask: &Matrix,  // 1 where observed, 0 where missing
        lambda: f64,
        max_iterations: usize,
    ) -> MathResult<Matrix> {
        let mut x = Matrix::zeros(observed.rows, observed.cols);
        
        for _ in 0..max_iterations {
            // Compute SVD
            let svd = x.svd()?;
            
            // Soft-threshold singular values
            let mut s_thresholded = Vec::new();
            for &s in &svd.s {
                let thresholded = (s - lambda).max(0.0);
                s_thresholded.push(thresholded);
            }
            
            // Reconstruct
            let reconstructed = svd.u.mul(&Matrix::diagonal(&s_thresholded))?.mul(&svd.vt)?;
            
            // Update only observed entries
            for i in 0..observed.rows {
                for j in 0..observed.cols {
                    if mask.get(i, j) > 0.5 {
                        x.set(i, j, reconstructed.get(i, j));
                    }
                }
            }
            
            // Check convergence
            let diff = reconstructed.sub(&x)?;
            let norm = crate::norms::MatrixNorms::frobenius(&diff);
            if norm < 1e-10 {
                break;
            }
        }
        
        Ok(x)
    }

    /// Nuclear norm minimization (simplified).
    pub fn nuclear_norm_minimization(
        observed: &Matrix,
        mask: &Matrix,
        max_rank: usize,
    ) -> MathResult<Matrix> {
        // Use low-rank approximation as proxy
        let approx = LowRankApprox::truncated_svd(observed, max_rank)?;
        LowRankApprox::reconstruct(&approx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncated_svd() {
        let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();
        let approx = LowRankApprox::truncated_svd(&m, 1).unwrap();
        
        assert_eq!(approx.rank, 1);
        assert!(approx.error >= 0.0);
    }

    #[test]
    fn test_energy_based() {
        let m = Matrix::diagonal(&[10.0, 1.0, 0.1]);
        let approx = LowRankApprox::energy_based(&m, 0.95).unwrap();
        
        assert!(approx.rank >= 1);
    }

    #[test]
    fn test_reconstruct() {
        let m = Matrix::identity(3);
        let approx = LowRankApprox::truncated_svd(&m, 3).unwrap();
        let reconstructed = LowRankApprox::reconstruct(&approx).unwrap();
        
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((reconstructed.get(i, j) - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_explained_variance() {
        let s = vec![10.0, 5.0, 1.0, 0.1];
        let rank = RankSelection::explained_variance(&s, 0.9);
        assert!(rank >= 1);
    }
}
