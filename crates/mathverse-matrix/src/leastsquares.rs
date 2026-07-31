//! Least squares solvers: QR-based, normal equations, SVD-based.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Least squares result.
#[derive(Debug, Clone)]
pub struct LeastSquaresResult {
    pub solution: mathverse_vector::Vector,
    pub residuals: mathverse_vector::Vector,
    pub residual_norm: f64,
    pub rank: usize,
}

/// Least squares solvers.
pub struct LeastSquares;

impl LeastSquares {
    /// Solve least squares using QR decomposition: min ||Ax - b||.
    pub fn qr_solve(m: &Matrix, b: &mathverse_vector::Vector) -> MathResult<LeastSquaresResult> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        
        if b.len() != m_rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let qr = m.qr()?;
        let q = qr.q;
        let r = qr.r;
        
        // Compute Q^T b
        let qt_b = q.transpose().mul_vec(b)?;
        
        // Solve R x = Q^T b (only first m_cols rows)
        let mut x = vec![0.0; m_cols];
        for i in (0..m_cols).rev() {
            let mut sum = qt_b.get(i);
            for j in (i + 1)..m_cols {
                sum -= r.get(i, j) * x[j];
            }
            
            if r.get(i, i).abs() < 1e-15 {
                return Err(MathError::InvalidArgument("singular matrix in QR least squares"));
            }
            
            x[i] = sum / r.get(i, i);
        }
        
        // Compute residuals
        let ax = m.mul_vec(&mathverse_vector::Vector::new(x.clone()))?;
        let residuals = b.sub(&ax)?;
        let residual_norm = crate::norms::MatrixNorms::frobenius(&residuals.to_matrix());
        
        let rank = crate::rank::MatrixRank::compute(m, 1e-10)?;
        
        Ok(LeastSquaresResult {
            solution: mathverse_vector::Vector::new(x),
            residuals,
            residual_norm,
            rank,
        })
    }

    /// Solve using normal equations: A^T A x = A^T b.
    pub fn normal_equations(m: &Matrix, b: &mathverse_vector::Vector) -> MathResult<LeastSquaresResult> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        
        if b.len() != m_rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let ata = m.transpose().mul(m)?;
        let atb = m.transpose().mul_vec(b)?;
        
        let x = ata.solve(&atb)?;
        
        // Compute residuals
        let ax = m.mul_vec(&x)?;
        let residuals = b.sub(&ax)?;
        let residual_norm = crate::norms::MatrixNorms::frobenius(&residuals.to_matrix());
        
        let rank = crate::rank::MatrixRank::compute(m, 1e-10)?;
        
        Ok(LeastSquaresResult {
            solution: x,
            residuals,
            residual_norm,
            rank,
        })
    }

    /// Solve using SVD (handles rank-deficient matrices).
    pub fn svd_solve(m: &Matrix, b: &mathverse_vector::Vector, tolerance: f64) -> MathResult<LeastSquaresResult> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        
        if b.len() != m_rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let svd = m.svd()?;
        let sigma_max = svd.s[0];
        
        // Compute pseudoinverse of singular values
        let mut sigma_plus = vec![0.0; svd.s.len()];
        let mut rank = 0;
        for (i, &s) in svd.s.iter().enumerate() {
            if s > tolerance * sigma_max {
                sigma_plus[i] = 1.0 / s;
                rank += 1;
            }
        }
        
        // x = V Σ^+ U^T b
        let sigma_plus_mat = Matrix::diagonal(&sigma_plus);
        let vt_sigma = sigma_plus_mat.mul(&svd.vt)?;
        let ut_b = svd.u.transpose().mul_vec(b)?;
        let x = vt_sigma.mul_vec(&ut_b)?;
        
        // Compute residuals
        let ax = m.mul_vec(&x)?;
        let residuals = b.sub(&ax)?;
        let residual_norm = crate::norms::MatrixNorms::frobenius(&residuals.to_matrix());
        
        Ok(LeastSquaresResult {
            solution: x,
            residuals,
            residual_norm,
            rank,
        })
    }

    /// Weighted least squares: min ||W^(1/2)(Ax - b)||.
    pub fn weighted(
        m: &Matrix,
        b: &mathverse_vector::Vector,
        weights: &[f64],
    ) -> MathResult<LeastSquaresResult> {
        if weights.len() != m.rows {
            return Err(MathError::DimensionMismatch);
        }
        
        // Apply weights
        let mut w_sqrt_m = m.clone();
        let mut w_sqrt_b = b.clone();
        
        for i in 0..m.rows {
            let w = weights[i].sqrt();
            for j in 0..m.cols {
                w_sqrt_m.set(i, j, w * m.get(i, j));
            }
            w_sqrt_b.set(i, w * b.get(i));
        }
        
        Self::qr_solve(&w_sqrt_m, &w_sqrt_b)
    }

    /// Constrained least squares: min ||Ax - b|| subject to Cx = d.
    pub fn constrained(
        m: &Matrix,
        b: &mathverse_vector::Vector,
        c: &Matrix,
        d: &mathverse_vector::Vector,
    ) -> MathResult<LeastSquaresResult> {
        // Use Lagrange multipliers
        let n = m.cols;
        let p = c.rows;
        
        // Form augmented system: [A^T A  C^T] [x] = [A^T b]
        //                   [C      0 ] [λ]   [d     ]
        
        let ata = m.transpose().mul(m)?;
        let atb = m.transpose().mul_vec(b)?;
        let ct = c.transpose();
        
        let mut augmented = Matrix::zeros(n + p, n + p);
        let mut augmented_b = vec![0.0; n + p];
        
        for i in 0..n {
            for j in 0..n {
                augmented.set(i, j, ata.get(i, j));
            }
            augmented_b[i] = atb.get(i);
        }
        
        for i in 0..p {
            for j in 0..n {
                augmented.set(n + i, j, c.get(i, j));
                augmented.set(j, n + i, ct.get(j, i));
            }
            augmented_b[n + i] = d.get(i);
        }
        
        let solution_vec = augmented.solve(&mathverse_vector::Vector::new(augmented_b))?;
        
        // Extract x (first n elements)
        let x = mathverse_vector::Vector::new(solution_vec.data[..n].to_vec());
        
        // Compute residuals
        let ax = m.mul_vec(&x)?;
        let residuals = b.sub(&ax)?;
        let residual_norm = crate::norms::MatrixNorms::frobenius(&residuals.to_matrix());
        
        let rank = crate::rank::MatrixRank::compute(m, 1e-10)?;
        
        Ok(LeastSquaresResult {
            solution: x,
            residuals,
            residual_norm,
            rank,
        })
    }

    /// Total least squares: minimize ||[A b] * [x; -1]||.
    pub fn total_least_squares(m: &Matrix, b: &mathverse_vector::Vector) -> MathResult<LeastSquaresResult> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        
        // Form augmented matrix [A | b]
        let mut augmented = Matrix::zeros(m_rows, m_cols + 1);
        for i in 0..m_rows {
            for j in 0..m_cols {
                augmented.set(i, j, m.get(i, j));
            }
            augmented.set(i, m_cols, b.get(i));
        }
        
        // SVD of augmented matrix
        let svd = augmented.svd()?;
        
        // Solution is last column of V (corresponding to smallest singular value)
        let v_last = svd.vt.row(m_cols);
        
        // Normalize: x = -v[0:n] / v[n]
        let v_n = v_last[m_cols];
        if v_n.abs() < 1e-15 {
            return Err(MathError::InvalidArgument("TLS solution is degenerate"));
        }
        
        let x: Vec<f64> = v_last[..m_cols].iter().map(|&v| -v / v_n).collect();
        
        // Compute residuals
        let ax = m.mul_vec(&mathverse_vector::Vector::new(x.clone()))?;
        let residuals = b.sub(&ax)?;
        let residual_norm = crate::norms::MatrixNorms::frobenius(&residuals.to_matrix());
        
        let rank = crate::rank::MatrixRank::compute(m, 1e-10)?;
        
        Ok(LeastSquaresResult {
            solution: mathverse_vector::Vector::new(x),
            residuals,
            residual_norm,
            rank,
        })
    }

    /// Non-negative least squares: min ||Ax - b|| subject to x >= 0.
    pub fn non_negative(
        m: &Matrix,
        b: &mathverse_vector::Vector,
        max_iterations: usize,
    ) -> MathResult<LeastSquaresResult> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        
        // Active set method (simplified)
        let mut x = vec![0.0; m_cols];
        let mut active_set = vec![false; m_cols];
        
        for _ in 0..max_iterations {
            // Compute gradient: A^T (Ax - b)
            let ax = m.mul_vec(&mathverse_vector::Vector::new(x.clone()))?;
            let residual = ax.sub(b)?;
            let gradient = m.transpose().mul_vec(&residual)?;
            
            // Find most negative gradient among inactive variables
            let mut max_grad = 0.0;
            let mut max_idx = None;
            
            for (i, &active) in active_set.iter().enumerate() {
                if !active && gradient.get(i) < max_grad {
                    max_grad = gradient.get(i);
                    max_idx = Some(i);
                }
            }
            
            if let Some(idx) = max_idx {
                if max_grad >= -1e-10 {
                    break; // Optimal
                }
                active_set[idx] = true;
            } else {
                break;
            }
            
            // Solve reduced problem
            let active_indices: Vec<usize> = active_set.iter()
                .enumerate()
                .filter(|(_, &active)| active)
                .map(|(i, _)| i)
                .collect();
            
            if active_indices.is_empty() {
                break;
            }
            
            let mut a_reduced = Matrix::zeros(m_rows, active_indices.len());
            for (j, &idx) in active_indices.iter().enumerate() {
                for i in 0..m_rows {
                    a_reduced.set(i, j, m.get(i, idx));
                }
            }
            
            let x_reduced = Self::qr_solve(&a_reduced, b)?.solution;
            
            // Update solution
            for (j, &idx) in active_indices.iter().enumerate() {
                x[idx] = x_reduced.get(j).max(0.0);
                if x[idx] < 1e-15 {
                    active_set[idx] = false;
                }
            }
        }
        
        // Compute residuals
        let ax = m.mul_vec(&mathverse_vector::Vector::new(x.clone()))?;
        let residuals = b.sub(&ax)?;
        let residual_norm = crate::norms::MatrixNorms::frobenius(&residuals.to_matrix());
        
        let rank = crate::rank::MatrixRank::compute(m, 1e-10)?;
        
        Ok(LeastSquaresResult {
            solution: mathverse_vector::Vector::new(x),
            residuals,
            residual_norm,
            rank,
        })
    }
}

/// Least squares utilities.
pub struct LeastSquaresUtils;

impl LeastSquaresUtils {
    /// Compute R-squared: 1 - ||r||^2 / ||b - mean(b)||^2.
    pub fn r_squared(result: &LeastSquaresResult, b: &mathverse_vector::Vector) -> f64 {
        let residual_ss = result.residual_norm * result.residual_norm;
        
        let mean_b = b.data.iter().sum::<f64>() / b.len() as f64;
        let total_ss: f64 = b.data.iter().map(|&x| (x - mean_b).powi(2)).sum();
        
        if total_ss > 0.0 {
            1.0 - residual_ss / total_ss
        } else {
            0.0
        }
    }

    /// Compute condition number of least squares problem.
    pub fn condition_number(m: &Matrix) -> MathResult<f64> {
        crate::condition::ConditionNumber::least_squares(m)
    }

    /// Leverage values (diagonal of hat matrix).
    pub fn leverage_values(m: &Matrix) -> MathResult<Vec<f64>> {
        let qr = m.qr()?;
        let q = qr.q;
        
        let mut leverage = Vec::new();
        for i in 0..q.rows {
            let mut sum = 0.0;
            for j in 0..q.cols {
                sum += q.get(i, j) * q.get(i, j);
            }
            leverage.push(sum);
        }
        
        Ok(leverage)
    }

    /// Cook's distance for influence analysis.
    pub fn cooks_distance(
        result: &LeastSquaresResult,
        m: &Matrix,
        leverage: &[f64],
    ) -> MathResult<Vec<f64>> {
        let n = m.rows;
        let p = m.cols;
        let mse = result.residual_norm * result.residual_norm / (n - p) as f64;
        
        let mut cooks = Vec::new();
        for i in 0..n {
            let residual = result.residuals.get(i);
            let leverage_i = leverage[i];
            
            if leverage_i >= 1.0 {
                cooks.push(f64::INFINITY);
            } else {
                let cook = residual * residual / (mse * p as f64) * leverage_i / (1.0 - leverage_i).powi(2);
                cooks.push(cook);
            }
        }
        
        Ok(cooks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_least_squares() {
        let m = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0]]).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0, 2.0]);
        
        let result = LeastSquares::qr_solve(&m, &b).unwrap();
        
        assert!(result.solution.len() == 2);
        assert!(result.residual_norm >= 0.0);
    }

    #[test]
    fn test_normal_equations() {
        let m = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 2.0]]).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0]);
        
        let result = LeastSquares::normal_equations(&m, &b).unwrap();
        
        assert!(result.solution.len() == 2);
    }

    #[test]
    fn test_svd_least_squares() {
        let m = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0]]).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0, 2.0]);
        
        let result = LeastSquares::svd_solve(&m, &b, 1e-10).unwrap();
        
        assert!(result.solution.len() == 2);
    }

    #[test]
    fn test_weighted_least_squares() {
        let m = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 2.0]]).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0]);
        let weights = vec![1.0, 2.0];
        
        let result = LeastSquares::weighted(&m, &b, &weights).unwrap();
        
        assert!(result.solution.len() == 2);
    }
}
