//! Matrix rank: computation, estimation, and rank-deficient matrices.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Matrix rank computation.
pub struct MatrixRank;

impl MatrixRank {
    /// Compute rank using SVD (numerical rank).
    pub fn compute(m: &Matrix, tolerance: f64) -> MathResult<usize> {
        let svd = m.svd()?;
        let sigma_max = svd.s[0];
        
        let rank = svd.s.iter()
            .filter(|&&s| s > tolerance * sigma_max)
            .count();
        
        Ok(rank)
    }

    /// Compute rank using QR decomposition with column pivoting.
    pub fn qr_rank(m: &Matrix, tolerance: f64) -> MathResult<usize> {
        let qr = m.qr()?;
        let n = qr.r.cols.min(qr.r.rows);
        
        let rank = (0..n)
            .filter(|&i| qr.r.get(i, i).abs() > tolerance)
            .count();
        
        Ok(rank)
    }

    /// Compute exact rank for integer matrices (Gaussian elimination).
    pub fn exact(m: &Matrix) -> MathResult<usize> {
        let mut a = m.clone();
        let (rows, cols) = (a.rows, a.cols);
        let mut rank = 0;
        
        for col in 0..cols {
            // Find pivot
            let mut pivot_row = rank;
            while pivot_row < rows && a.get(pivot_row, col).abs() < 1e-10 {
                pivot_row += 1;
            }
            
            if pivot_row == rows {
                continue;
            }
            
            // Swap rows
            if pivot_row != rank {
                for j in col..cols {
                    let temp = a.get(rank, j);
                    a.set(rank, j, a.get(pivot_row, j));
                    a.set(pivot_row, j, temp);
                }
            }
            
            // Eliminate below
            let pivot = a.get(rank, col);
            for i in (rank + 1)..rows {
                let factor = a.get(i, col) / pivot;
                for j in col..cols {
                    a.set(i, j, a.get(i, j) - factor * a.get(rank, j));
                }
            }
            
            rank += 1;
            if rank == rows {
                break;
            }
        }
        
        Ok(rank)
    }

    /// Estimate rank without full SVD (randomized algorithm).
    pub fn estimate(m: &Matrix, oversampling: usize) -> MathResult<usize> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        let k = m_rows.min(m_cols) + oversampling;
        
        // Generate random projection matrix
        let mut rng = crate::rng::Rng::new(42);
        let mut omega = Matrix::zeros(m_cols, k);
        for i in 0..m_cols {
            for j in 0..k {
                omega.set(i, j, rng.uniform() * 2.0 - 1.0);
            }
        }
        
        // Form Y = A * Omega
        let y = m.mul(&omega)?;
        
        // QR decomposition of Y
        let qr_y = y.qr()?;
        
        // Form B = Q^T * A
        let qt = qr_y.q.transpose();
        let b = qt.mul(m)?;
        
        // SVD of B (smaller matrix)
        let svd_b = b.svd()?;
        
        let sigma_max = svd_b.s[0];
        let tolerance = 1e-10;
        let rank = svd_b.s.iter()
            .filter(|&&s| s > tolerance * sigma_max)
            .count();
        
        Ok(rank.min(m_rows.min(m_cols)))
    }

    /// Numerical rank with adaptive tolerance.
    pub fn adaptive(m: &Matrix) -> MathResult<usize> {
        let svd = m.svd()?;
        let sigma_max = svd.s[0];
        
        // Adaptive tolerance based on machine epsilon and matrix size
        let n = m.rows.max(m.cols) as f64;
        let tolerance = n * sigma_max * f64::EPSILON;
        
        let rank = svd.s.iter()
            .filter(|&&s| s > tolerance)
            .count();
        
        Ok(rank)
    }

    /// Check if matrix is full rank.
    pub fn is_full_rank(m: &Matrix, tolerance: f64) -> MathResult<bool> {
        let rank = Self::compute(m, tolerance)?;
        let min_dim = m.rows.min(m.cols);
        Ok(rank == min_dim)
    }

    /// Rank deficiency: min(m,n) - rank(A).
    pub fn deficiency(m: &Matrix, tolerance: f64) -> MathResult<usize> {
        let rank = Self::compute(m, tolerance)?;
        let min_dim = m.rows.min(m.cols);
        Ok(min_dim - rank)
    }

    /// Effective rank based on singular value energy.
    pub fn energy_based(m: &Matrix, energy_threshold: f64) -> MathResult<usize> {
        let svd = m.svd()?;
        let total_energy: f64 = svd.s.iter().map(|&s| s * s).sum();
        
        let mut cumulative = 0.0;
        let mut rank = 0;
        
        for &s in &svd.s {
            cumulative += s * s;
            rank += 1;
            if cumulative / total_energy >= energy_threshold {
                break;
            }
        }
        
        Ok(rank)
    }
}

/// Rank properties and analysis.
pub struct RankProperties;

impl RankProperties {
    /// Check rank inequality: rank(AB) ≤ min(rank(A), rank(B)).
    pub fn rank_inequality(
        a: &Matrix,
        b: &Matrix,
        tolerance: f64,
    ) -> MathResult<bool> {
        let rank_a = MatrixRank::compute(a, tolerance)?;
        let rank_b = MatrixRank::compute(b, tolerance)?;
        
        let prod = a.mul(b)?;
        let rank_ab = MatrixRank::compute(&prod, tolerance)?;
        
        Ok(rank_ab <= rank_a.min(rank_b))
    }

    /// Check Sylvester's rank inequality: rank(A) + rank(B) - n ≤ rank(AB).
    pub fn sylvester_inequality(
        a: &Matrix,
        b: &Matrix,
        tolerance: f64,
    ) -> MathResult<bool> {
        if a.cols != b.rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let rank_a = MatrixRank::compute(a, tolerance)?;
        let rank_b = MatrixRank::compute(b, tolerance)?;
        let n = a.cols;
        
        let prod = a.mul(b)?;
        let rank_ab = MatrixRank::compute(&prod, tolerance)?;
        
        Ok(rank_a + rank_b - n <= rank_ab)
    }

    /// Rank of sum: rank(A + B) ≤ rank(A) + rank(B).
    pub fn sum_inequality(
        a: &Matrix,
        b: &Matrix,
        tolerance: f64,
    ) -> MathResult<bool> {
        let rank_a = MatrixRank::compute(a, tolerance)?;
        let rank_b = MatrixRank::compute(b, tolerance)?;
        
        let sum = a.add(b)?;
        let rank_sum = MatrixRank::compute(&sum, tolerance)?;
        
        Ok(rank_sum <= rank_a + rank_b)
    }

    /// Rank of transpose: rank(A) = rank(A^T).
    pub fn transpose_equality(m: &Matrix, tolerance: f64) -> MathResult<bool> {
        let rank = MatrixRank::compute(m, tolerance)?;
        let rank_t = MatrixRank::compute(&m.transpose(), tolerance)?;
        Ok(rank == rank_t)
    }

    /// Nullity (dimension of null space): nullity(A) = n - rank(A).
    pub fn nullity(m: &Matrix, tolerance: f64) -> MathResult<usize> {
        let rank = MatrixRank::compute(m, tolerance)?;
        Ok(m.cols - rank)
    }
}

/// Rank-revealing decompositions.
pub struct RankRevealing;

impl RankRevealing {
    /// QR with column pivoting for rank determination.
    pub fn qr_pivoting(m: &Matrix) -> MathResult<(Matrix, Vec<usize>)> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        let mut a = m.clone();
        let mut pivots: Vec<usize> = (0..m_cols).collect();
        let mut rank = 0;
        
        for k in 0..m_cols.min(m_rows) {
            // Find column with maximum norm
            let mut max_col = k;
            let mut max_norm = 0.0;
            
            for j in k..m_cols {
                let norm: f64 = (0..m_rows)
                    .map(|i| a.get(i, j).abs())
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt();
                if norm > max_norm {
                    max_norm = norm;
                    max_col = j;
                }
            }
            
            if max_norm < 1e-10 {
                break;
            }
            
            // Swap columns
            if max_col != k {
                pivots.swap(k, max_col);
                for i in 0..m_rows {
                    let temp = a.get(i, k);
                    a.set(i, k, a.get(i, max_col));
                    a.set(i, max_col, temp);
                }
            }
            
            // Householder reflection
            let mut x: Vec<f64> = (k..m_rows).map(|i| a.get(i, k)).collect();
            let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            
            if norm_x < 1e-10 {
                break;
            }
            
            let alpha = if x[0] >= 0.0 { -norm_x } else { norm_x };
            x[0] -= alpha;
            let vn = x.iter().map(|w| w * w).sum::<f64>().sqrt();
            
            if vn > 1e-10 {
                for w in &mut x {
                    *w /= vn;
                }
                
                for j in k..m_cols {
                    let dot: f64 = x.iter()
                        .enumerate()
                        .map(|(o, &vv)| vv * a.get(k + o, j))
                        .sum();
                    for (o, &vv) in x.iter().enumerate() {
                        a.set(k + o, j, a.get(k + o, j) - 2.0 * vv * dot);
                    }
                }
            }
            
            rank += 1;
        }
        
        Ok((a, pivots))
    }

    /// SVD-based rank revealing decomposition.
    pub fn svd_rank_reveal(m: &Matrix, tolerance: f64) -> MathResult<(Matrix, Matrix, Vec<f64>)> {
        let svd = m.svd()?;
        let sigma_max = svd.s[0];
        
        // Truncate small singular values
        let truncated_s: Vec<f64> = svd.s.iter()
            .map(|&s| if s > tolerance * sigma_max { s } else { 0.0 })
            .collect();
        
        let rank = truncated_s.iter().filter(|&&s| s > 0.0).count();
        
        // Truncate U and V
        let mut u_trunc = Matrix::zeros(m.rows, rank);
        let mut v_trunc = Matrix::zeros(rank, m.cols);
        
        for j in 0..rank {
            for i in 0..m.rows {
                u_trunc.set(i, j, svd.u.get(i, j));
            }
            for i in 0..m.cols {
                v_trunc.set(j, i, svd.vt.get(j, i));
            }
        }
        
        Ok((u_trunc, v_trunc, truncated_s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_rank() {
        let m = Matrix::identity(3);
        let rank = MatrixRank::compute(&m, 1e-10).unwrap();
        assert_eq!(rank, 3);
    }

    #[test]
    fn test_rank_deficient() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
        let rank = MatrixRank::compute(&m, 1e-10).unwrap();
        assert_eq!(rank, 1);
    }

    #[test]
    fn test_nullity() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
        let nullity = RankProperties::nullity(&m, 1e-10).unwrap();
        assert_eq!(nullity, 1);
    }

    #[test]
    fn test_transpose_equality() {
        let m = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();
        assert!(RankProperties::transpose_equality(&m, 1e-10).unwrap());
    }

    #[test]
    fn test_energy_based_rank() {
        let m = Matrix::diagonal(&[10.0, 1.0, 0.1]);
        let rank = MatrixRank::energy_based(&m, 0.99).unwrap();
        assert!(rank >= 1);
    }
}
