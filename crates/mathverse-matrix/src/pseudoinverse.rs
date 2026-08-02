//! Moore-Penrose pseudoinverse: computation and applications.

use crate::Matrix;
use mathverse_core::error::MathResult;

/// Moore-Penrose pseudoinverse.
pub struct Pseudoinverse;

impl Pseudoinverse {
    /// Compute pseudoinverse using SVD: A⁺ = V Σ⁺ Uᵀ.
    pub fn compute(m: &Matrix, tolerance: f64) -> MathResult<Matrix> {
        let svd = m.svd()?;
        let sigma_max = svd.s[0];
        
        // Compute Σ⁺ (reciprocal of non-zero singular values)
        let mut sigma_plus = vec![0.0; svd.s.len()];
        for (i, &s) in svd.s.iter().enumerate() {
            if s > tolerance * sigma_max {
                sigma_plus[i] = 1.0 / s;
            }
        }
        
        // A⁺ = V Σ⁺ Uᵀ
        let sigma_plus_mat = Matrix::diagonal(&sigma_plus);
        let v_sigma = svd.vt.transpose().mul(&sigma_plus_mat)?;
        let pinv = v_sigma.mul(&svd.u.transpose())?;
        
        Ok(pinv)
    }

    /// Compute pseudoinverse for tall matrices (m > n) using normal equations.
    pub fn tall(m: &Matrix, _tolerance: f64) -> MathResult<Matrix> {
        let ata = m.transpose().mul(m)?;
        let inv_ata = ata.inverse()?;
        let pinv = inv_ata.mul(&m.transpose())?;
        Ok(pinv)
    }

    /// Compute pseudoinverse for wide matrices (m < n) using normal equations.
    pub fn wide(m: &Matrix, _tolerance: f64) -> MathResult<Matrix> {
        let aat = m.mul(&m.transpose())?;
        let inv_aat = aat.inverse()?;
        let pinv = m.transpose().mul(&inv_aat)?;
        Ok(pinv)
    }

    /// Least squares solution using pseudoinverse: x = A⁺ b.
    pub fn least_squares(m: &Matrix, b: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        let pinv = Self::compute(m, 1e-10)?;
        pinv.mul_vec(b)
    }

    /// Minimum norm solution for underdetermined system.
    pub fn minimum_norm(m: &Matrix, b: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        let pinv = Self::compute(m, 1e-10)?;
        pinv.mul_vec(b)
    }

    /// Check if pseudoinverse satisfies Moore-Penrose conditions.
    pub fn verify_conditions(
        m: &Matrix,
        pinv: &Matrix,
        tolerance: f64,
    ) -> MathResult<bool> {
        // Condition 1: A A⁺ A = A
        let a_pinv_a = m.mul(pinv)?.mul(m)?;
        let cond1 = Self::matrices_equal(m, &a_pinv_a, tolerance);
        
        // Condition 2: A⁺ A A⁺ = A⁺
        let pinv_a_pinv = pinv.mul(m)?.mul(pinv)?;
        let cond2 = Self::matrices_equal(pinv, &pinv_a_pinv, tolerance);
        
        // Condition 3: (A A⁺)ᵀ = A A⁺
        let a_pinv = m.mul(pinv)?;
        let a_pinv_t = a_pinv.transpose();
        let cond3 = Self::matrices_equal(&a_pinv, &a_pinv_t, tolerance);
        
        // Condition 4: (A⁺ A)ᵀ = A⁺ A
        let pinv_a = pinv.mul(m)?;
        let pinv_a_t = pinv_a.transpose();
        let cond4 = Self::matrices_equal(&pinv_a, &pinv_a_t, tolerance);
        
        Ok(cond1 && cond2 && cond3 && cond4)
    }

    fn matrices_equal(a: &Matrix, b: &Matrix, tolerance: f64) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        
        for i in 0..a.rows {
            for j in 0..a.cols {
                if (a.get(i, j) - b.get(i, j)).abs() > tolerance {
                    return false;
                }
            }
        }
        
        true
    }

    /// Iterative refinement of pseudoinverse.
    pub fn iterative_refinement(
        m: &Matrix,
        tolerance: f64,
        max_iterations: usize,
    ) -> MathResult<Matrix> {
        let mut pinv = Self::compute(m, tolerance)?;
        
        for _ in 0..max_iterations {
            let residual = m.mul(&pinv)?;
            let identity = Matrix::identity(residual.rows.min(residual.cols));
            let error = residual.sub(&identity)?;
            
            if crate::norms::MatrixNorms::frobenius(&error) < tolerance {
                break;
            }
            
            let correction = Self::compute(&error, tolerance)?;
            pinv = pinv.sub(&correction)?;
        }
        
        Ok(pinv)
    }

    /// Block pseudoinverse for large matrices.
    pub fn block(m: &Matrix, block_size: usize) -> MathResult<Matrix> {
        let (rows, cols) = (m.rows, m.cols);
        let mut pinv = Matrix::zeros(cols, rows);
        
        for bi in (0..rows).step_by(block_size) {
            for bj in (0..cols).step_by(block_size) {
                let i_end = (bi + block_size).min(rows);
                let j_end = (bj + block_size).min(cols);
                
                // Extract block
                let mut block = Matrix::zeros(i_end - bi, j_end - bj);
                for i in bi..i_end {
                    for j in bj..j_end {
                        block.set(i - bi, j - bj, m.get(i, j));
                    }
                }
                
                let block_pinv = Self::compute(&block, 1e-10)?;
                
                for i in bj..j_end {
                    for j in bi..i_end {
                        pinv.set(i, j, block_pinv.get(i - bj, j - bi));
                    }
                }
            }
        }
        
        Ok(pinv)
    }
}

/// Pseudoinverse applications.
pub struct PseudoinverseApplications;

impl PseudoinverseApplications {
    /// Solve linear system Ax = b (least squares for overdetermined).
    pub fn solve(
        m: &Matrix,
        b: &mathverse_vector::Vector,
    ) -> MathResult<mathverse_vector::Vector> {
        Pseudoinverse::least_squares(m, b)
    }

    /// Compute projection onto column space of A: P = A A⁺.
    pub fn column_projection(m: &Matrix) -> MathResult<Matrix> {
        let pinv = Pseudoinverse::compute(m, 1e-10)?;
        m.mul(&pinv)
    }

    /// Compute projection onto row space of A: P = A⁺ A.
    pub fn row_projection(m: &Matrix) -> MathResult<Matrix> {
        let pinv = Pseudoinverse::compute(m, 1e-10)?;
        pinv.mul(m)
    }

    /// Compute projection onto null space of A: I - A⁺ A.
    pub fn null_projection(m: &Matrix) -> MathResult<Matrix> {
        let pinv = Pseudoinverse::compute(m, 1e-10)?;
        let row_proj = pinv.mul(m)?;
        let identity = Matrix::identity(row_proj.rows);
        identity.sub(&row_proj)
    }

    /// Compute projection onto left null space of A: I - A A⁺.
    pub fn left_null_projection(m: &Matrix) -> MathResult<Matrix> {
        let pinv = Pseudoinverse::compute(m, 1e-10)?;
        let col_proj = m.mul(&pinv)?;
        let identity = Matrix::identity(col_proj.rows);
        identity.sub(&col_proj)
    }

    /// Minimum norm least squares solution.
    pub fn minimum_norm_least_squares(
        m: &Matrix,
        b: &mathverse_vector::Vector,
    ) -> MathResult<mathverse_vector::Vector> {
        Pseudoinverse::least_squares(m, b)
    }

    /// Damped least squares (Tikhonov regularization): x = (AᵀA + λI)⁻¹ Aᵀ b.
    pub fn tikhonov(
        m: &Matrix,
        b: &mathverse_vector::Vector,
        lambda: f64,
    ) -> MathResult<mathverse_vector::Vector> {
        let ata = m.transpose().mul(m)?;
        let n = ata.rows;
        let mut regularized = ata.clone();
        
        for i in 0..n {
            regularized.set(i, i, regularized.get(i, i) + lambda);
        }
        
        let inv = regularized.inverse()?;
        let atb = m.transpose().mul_vec(b)?;
        inv.mul_vec(&atb)
    }

    /// Truncated SVD pseudoinverse (rank-k approximation).
    pub fn truncated_svd(m: &Matrix, k: usize) -> MathResult<Matrix> {
        let svd = m.svd()?;
        let k = k.min(svd.s.len());
        
        // Truncate singular values
        let mut s_trunc = vec![0.0; k];
        for i in 0..k {
            s_trunc[i] = svd.s[i];
        }
        
        // Truncate U and V
        let mut u_trunc = Matrix::zeros(m.rows, k);
        let mut v_trunc = Matrix::zeros(k, m.cols);
        
        for j in 0..k {
            for i in 0..m.rows {
                u_trunc.set(i, j, svd.u.get(i, j));
            }
            for i in 0..m.cols {
                v_trunc.set(j, i, svd.vt.get(j, i));
            }
        }
        
        // Compute pseudoinverse of truncated SVD
        let mut s_plus = vec![0.0; k];
        for i in 0..k {
            if s_trunc[i] > 1e-10 {
                s_plus[i] = 1.0 / s_trunc[i];
            }
        }
        
        let s_plus_mat = Matrix::diagonal(&s_plus);
        let vt_s = s_plus_mat.mul(&v_trunc)?;
        u_trunc.transpose().mul(&vt_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pseudoinverse_identity() {
        let m = Matrix::identity(3);
        let pinv = Pseudoinverse::compute(&m, 1e-10).unwrap();
        assert!(Pseudoinverse::verify_conditions(&m, &pinv, 1e-10).unwrap());
    }

    #[test]
    fn test_pseudoinverse_full_rank() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let pinv = Pseudoinverse::compute(&m, 1e-10).unwrap();
        let should_be_identity = m.mul(&pinv).unwrap();
        let identity = Matrix::identity(2);
        
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((should_be_identity.get(i, j) - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_least_squares() {
        let m = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0]]).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0, 2.0]);
        let x = PseudoinverseApplications::solve(&m, &b).unwrap();
        assert!(x.data.len() == 2);
    }

    #[test]
    fn test_column_projection() {
        let m = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]).unwrap();
        let proj = PseudoinverseApplications::column_projection(&m).unwrap();
        assert!(Pseudoinverse::matrices_equal(&proj, &m, 1e-10));
    }

    #[test]
    fn test_tikhonov() {
        let m = Matrix::from_rows(&[&[1.0, 1.0], &[1.0, 2.0]]).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0]);
        let x = PseudoinverseApplications::tikhonov(&m, &b, 0.1).unwrap();
        assert!(x.data.len() == 2);
    }
}
