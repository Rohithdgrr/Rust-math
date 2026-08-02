//! Polar decomposition: A = UP where U is orthogonal and P is positive semi-definite.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Polar decomposition result.
#[derive(Debug, Clone)]
pub struct PolarDecomposition {
    pub u: Matrix,  // Orthogonal/unitary matrix
    pub p: Matrix,  // Positive semi-definite matrix
}

/// Polar decomposition implementation.
pub struct PolarDecompositionImpl;

impl PolarDecompositionImpl {
    /// Compute polar decomposition using SVD: A = U Σ V^T = (U V^T)(V Σ V^T).
    pub fn compute(m: &Matrix) -> MathResult<PolarDecomposition> {
        let svd = m.svd()?;
        
        // U = U V^T
        let u = svd.u.mul(&svd.vt)?;
        
        // P = V Σ V^T
        let sigma = Matrix::diagonal(&svd.s);
        let v_sigma = svd.vt.transpose().mul(&sigma)?;
        let p = v_sigma.mul(&svd.vt)?;
        
        Ok(PolarDecomposition { u, p })
    }

    /// Compute polar decomposition using Newton iteration (for square matrices).
    pub fn newton(m: &Matrix, max_iterations: usize, tolerance: f64) -> MathResult<PolarDecomposition> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let _n = m.rows;
        let mut x = m.clone();
        let mut x_inv = m.inverse()?;
        
        for _ in 0..max_iterations {
            let x_new = x.add(&x_inv.transpose())?.scale(0.5);
            
            let diff = x_new.sub(&x)?;
            let norm = crate::norms::MatrixNorms::frobenius(&diff);
            
            x = x_new;
            x_inv = x.inverse()?;
            
            if norm < tolerance {
                break;
            }
        }
        
        // U = X (orthogonal)
        let u = x.clone();
        
        // P = U^T A
        let p = u.transpose().mul(m)?;
        
        Ok(PolarDecomposition { u, p })
    }

    /// Compute polar decomposition using SVD with sign correction.
    pub fn compute_with_sign(m: &Matrix) -> MathResult<PolarDecomposition> {
        let svd = m.svd()?;
        
        // Compute sign of determinant
        let det_u = svd.u.det()?;
        let det_vt = svd.vt.det()?;
        let sign = (det_u * det_vt).signum();
        
        // Adjust U to have det = 1 if needed
        let mut u = svd.u.mul(&svd.vt)?;
        if sign < 0.0 {
            // Flip sign of last column
            for i in 0..u.rows {
                u.set(i, u.cols - 1, -u.get(i, u.cols - 1));
            }
        }
        
        // P = V Σ V^T
        let sigma = Matrix::diagonal(&svd.s);
        let v_sigma = svd.vt.transpose().mul(&sigma)?;
        let p = v_sigma.mul(&svd.vt)?;
        
        Ok(PolarDecomposition { u, p })
    }

    /// Verify polar decomposition: A = UP, U^T U = I, P = P^T ≥ 0.
    pub fn verify(
        m: &Matrix,
        polar: &PolarDecomposition,
        tolerance: f64,
    ) -> MathResult<bool> {
        // Check A = UP
        let up = polar.u.mul(&polar.p)?;
        let diff = m.sub(&up)?;
        let norm_diff = crate::norms::MatrixNorms::frobenius(&diff);
        
        // Check U^T U = I
        let ut_u = polar.u.transpose().mul(&polar.u)?;
        let identity = Matrix::identity(polar.u.rows);
        let diff_orth = ut_u.sub(&identity)?;
        let norm_orth = crate::norms::MatrixNorms::frobenius(&diff_orth);
        
        // Check P = P^T
        let p_t = polar.p.transpose();
        let diff_sym = polar.p.sub(&p_t)?;
        let norm_sym = crate::norms::MatrixNorms::frobenius(&diff_sym);
        
        // Check P ≥ 0 (positive semi-definite)
        let is_psd = crate::positivedefinite::PositiveDefinite::is_positive_semi_definite(&polar.p, tolerance);
        
        Ok(norm_diff < tolerance && norm_orth < tolerance && norm_sym < tolerance && is_psd)
    }
}

/// Polar decomposition applications.
pub struct PolarApplications;

impl PolarApplications {
    /// Matrix sign function using polar decomposition.
    pub fn matrix_sign(m: &Matrix) -> MathResult<Matrix> {
        let polar = PolarDecompositionImpl::compute(m)?;
        Ok(polar.u)
    }

    /// Orthogonal Procrustes problem: find orthogonal Q minimizing ||AQ - B||_F.
    pub fn orthogonal_procrustes(a: &Matrix, b: &Matrix) -> MathResult<Matrix> {
        let (m, n) = (a.rows, a.cols);
        if b.rows != m || b.cols != n {
            return Err(MathError::DimensionMismatch);
        }
        
        // Compute M = A^T B
        let m_mat = a.transpose().mul(b)?;
        
        // Polar decomposition of M
        let polar = PolarDecompositionImpl::compute(&m_mat)?;
        
        // Q = U from polar decomposition
        Ok(polar.u)
    }

    /// Nearest orthogonal matrix (in Frobenius norm).
    pub fn nearest_orthogonal(m: &Matrix) -> MathResult<Matrix> {
        PolarDecompositionImpl::compute(m).map(|p| p.u)
    }

    /// Nearest positive semi-definite matrix.
    pub fn nearest_psd(m: &Matrix) -> MathResult<Matrix> {
        PolarDecompositionImpl::compute(m).map(|p| p.p)
    }

    /// Symmetric polar decomposition (for symmetric matrices).
    pub fn symmetric(m: &Matrix) -> MathResult<PolarDecomposition> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let (vals, vecs) = m.eigen_symmetric()?;
        let n = m.rows;
        
        // U = I (for symmetric matrices, U is sign of eigenvalues)
        let mut u = Matrix::zeros(n, n);
        let mut p = Matrix::zeros(n, n);
        
        for i in 0..n {
            let sign = if vals[i] >= 0.0 { 1.0 } else { -1.0 };
            u.set(i, i, sign);
            p.set(i, i, vals[i].abs());
        }
        
        // Transform back
        let u_transformed = vecs.mul(&u)?.mul(&vecs.transpose())?;
        let p_transformed = vecs.mul(&p)?.mul(&vecs.transpose())?;
        
        Ok(PolarDecomposition {
            u: u_transformed,
            p: p_transformed,
        })
    }

    /// Compute matrix absolute value: |A| = (A^T A)^(1/2) = P from polar decomposition.
    pub fn matrix_absolute(m: &Matrix) -> MathResult<Matrix> {
        PolarDecompositionImpl::compute(m).map(|p| p.p)
    }

    /// Compute matrix sign using iterative method.
    pub fn matrix_sign_iterative(m: &Matrix, max_iterations: usize, tolerance: f64) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut x = m.clone();
        let mut x_inv = m.inverse()?;
        
        for _ in 0..max_iterations {
            let x_new = x.add(&x_inv.transpose())?.scale(0.5);
            
            let diff = x_new.sub(&x)?;
            let norm = crate::norms::MatrixNorms::frobenius(&diff);
            
            x = x_new;
            x_inv = x.inverse()?;
            
            if norm < tolerance {
                break;
            }
        }
        
        Ok(x)
    }
}

/// Polar decomposition for rectangular matrices.
pub struct RectangularPolar;

impl RectangularPolar {
    /// Polar decomposition for tall matrices (m > n): A = U [P; 0].
    pub fn tall(m: &Matrix) -> MathResult<PolarDecomposition> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        
        let svd = m.svd()?;
        
        // U = U V^T (m x m)
        let u_full = svd.u.mul(&svd.vt)?;
        
        // P = V Σ V^T (n x n)
        let sigma = Matrix::diagonal(&svd.s);
        let v_sigma = svd.vt.transpose().mul(&sigma)?;
        let p = v_sigma.mul(&svd.vt)?;
        
        // Pad P to match dimensions
        let mut p_padded = Matrix::zeros(m_rows, m_cols);
        for i in 0..m_cols {
            for j in 0..m_cols {
                p_padded.set(i, j, p.get(i, j));
            }
        }
        
        Ok(PolarDecomposition {
            u: u_full,
            p: p_padded,
        })
    }

    /// Polar decomposition for wide matrices (m < n): A = [U 0] P.
    pub fn wide(m: &Matrix) -> MathResult<PolarDecomposition> {
        let (m_rows, m_cols) = (m.rows, m.cols);
        
        let svd = m.svd()?;
        
        // U = U V^T (m x m)
        let u = svd.u.mul(&svd.vt)?;
        
        // P = V Σ V^T (n x n)
        let sigma = Matrix::diagonal(&svd.s);
        let v_sigma = svd.vt.transpose().mul(&sigma)?;
        let p = v_sigma.mul(&svd.vt)?;
        
        // Pad U to match dimensions
        let mut u_padded = Matrix::zeros(m_rows, m_cols);
        for i in 0..m_rows {
            for j in 0..m_rows {
                u_padded.set(i, j, u.get(i, j));
            }
        }
        
        Ok(PolarDecomposition {
            u: u_padded,
            p,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polar_decomposition() {
        let m = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let polar = PolarDecompositionImpl::compute(&m).unwrap();
        
        // Verify A = UP
        let up = polar.u.mul(&polar.p).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((up.get(i, j) - m.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_polar_verify() {
        let m = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let polar = PolarDecompositionImpl::compute(&m).unwrap();
        
        assert!(PolarDecompositionImpl::verify(&m, &polar, 1e-10).unwrap());
    }

    #[test]
    fn test_matrix_sign() {
        let m = Matrix::identity(2);
        let sign = PolarApplications::matrix_sign(&m).unwrap();
        
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((sign.get(i, j) - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_nearest_orthogonal() {
        let m = Matrix::from_rows(&[&[1.0, 0.1], &[0.1, 1.0]]).unwrap();
        let nearest = PolarApplications::nearest_orthogonal(&m).unwrap();
        
        // Check orthogonality
        let nt_n = nearest.transpose().mul(&nearest).unwrap();
        let _identity = Matrix::identity(2);
        
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((nt_n.get(i, j) - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_absolute() {
        let m = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, -3.0]]).unwrap();
        let abs_m = PolarApplications::matrix_absolute(&m).unwrap();
        
        // Should be positive
        assert!(abs_m.get(0, 0) > 0.0);
        assert!(abs_m.get(1, 1) > 0.0);
    }
}
