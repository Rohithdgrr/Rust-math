//! General eigenvalue decomposition for non-symmetric matrices.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// General eigenvalue decomposition result.
#[derive(Debug, Clone)]
pub struct GeneralEigenDecomposition {
    pub eigenvalues: Vec<f64>,  // May include complex pairs
    pub eigenvectors: Matrix,   // Right eigenvectors as columns
    pub left_eigenvectors: Option<Matrix>,  // Left eigenvectors
}

/// General eigenvalue decomposition using QR algorithm.
pub struct GeneralEigen;

impl GeneralEigen {
    /// Compute eigenvalues and eigenvectors for general matrix using QR iteration.
    pub fn compute(m: &Matrix) -> MathResult<GeneralEigenDecomposition> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        
        // First reduce to Hessenberg form
        let mut h = Self::hessenberg(m)?;
        let mut q = Matrix::identity(n);
        
        // QR iteration with shifts
        for _ in 0..100 {
            // Wilkinson shift
            let submatrix = Self::extract_submatrix(&h, n - 2, n - 2, 2, 2);
            let trace = submatrix.get(0, 0) + submatrix.get(1, 1);
            let det = submatrix.get(0, 0) * submatrix.get(1, 1) - submatrix.get(0, 1) * submatrix.get(1, 0);
            let discriminant = trace * trace - 4.0 * det;
            let shift = if discriminant >= 0.0 {
                (trace + discriminant.sqrt()) / 2.0
            } else {
                trace / 2.0
            };
            
            let shifted = h.sub(&Matrix::identity(n).scale(shift))?;
            let qr_result = shifted.qr()?;
            h = qr_result.r.mul(&qr_result.q)?.add(&Matrix::identity(n).scale(shift))?;
            q = q.mul(&qr_result.q)?;
            
            // Check for convergence
            if Self::is_upper_triangular(&h, 1e-10) {
                break;
            }
        }
        
        // Extract eigenvalues from diagonal
        let eigenvalues: Vec<f64> = (0..n).map(|i| h.get(i, i)).collect();
        
        // Compute eigenvectors
        let eigenvectors = Self::compute_eigenvectors(m, &eigenvalues)?;
        
        Ok(GeneralEigenDecomposition {
            eigenvalues,
            eigenvectors,
            left_eigenvectors: None,
        })
    }

    /// Reduce to upper Hessenberg form.
    fn hessenberg(m: &Matrix) -> MathResult<Matrix> {
        let n = m.rows;
        let mut h = m.clone();
        
        for k in 0..(n - 2) {
            let mut x: Vec<f64> = (k + 1..n).map(|i| h.get(i, k)).collect();
            let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            
            if norm_x < 1e-15 {
                continue;
            }
            
            let alpha = if x[0] >= 0.0 { -norm_x } else { norm_x };
            x[0] -= alpha;
            let vn = x.iter().map(|w| w * w).sum::<f64>().sqrt();
            
            if vn > 1e-15 {
                for w in &mut x {
                    *w /= vn;
                }
                
                for j in k..n {
                    let dot: f64 = x.iter()
                        .enumerate()
                        .map(|(o, &vv)| vv * h.get(k + 1 + o, j))
                        .sum();
                    for (o, &vv) in x.iter().enumerate() {
                        h.set(k + 1 + o, j, h.get(k + 1 + o, j) - 2.0 * vv * dot);
                    }
                }
                
                for i in 0..n {
                    let dot: f64 = x.iter()
                        .enumerate()
                        .map(|(o, &vv)| h.get(i, k + 1 + o) * vv)
                        .sum();
                    for (o, &vv) in x.iter().enumerate() {
                        h.set(i, k + 1 + o, h.get(i, k + 1 + o) - 2.0 * dot * vv);
                    }
                }
            }
        }
        
        Ok(h)
    }

    fn extract_submatrix(m: &Matrix, row: usize, col: usize, rows: usize, cols: usize) -> Matrix {
        let mut sub = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                sub.set(i, j, m.get(row + i, col + j));
            }
        }
        sub
    }

    fn is_upper_triangular(m: &Matrix, tolerance: f64) -> bool {
        for i in 1..m.rows {
            for j in 0..i.min(m.cols) {
                if m.get(i, j).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Compute eigenvectors using inverse iteration.
    fn compute_eigenvectors(m: &Matrix, eigenvalues: &[f64]) -> MathResult<Matrix> {
        let n = m.rows;
        let mut eigenvectors = Matrix::zeros(n, n);
        
        for (j, &lambda) in eigenvalues.iter().enumerate() {
            // Solve (A - lambda*I) * v = 0 using inverse iteration
            let shifted = m.sub(&Matrix::identity(n).scale(lambda + 1e-12))?;
            
            // Start with random vector
            let mut v = vec![1.0; n];
            
            for _ in 0..10 {
                // Solve (A - lambda*I) * v_new = v
                let v_vec = mathverse_vector::Vector::new(v.clone());
                let v_new = shifted.solve(&v_vec)?;
                v = v_new.data;
                
                // Normalize
                let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    v = v.iter().map(|x| x / norm).collect();
                }
            }
            
            for i in 0..n {
                eigenvectors.set(i, j, v[i]);
            }
        }
        
        Ok(eigenvectors)
    }

    /// Compute left eigenvectors (w^T A = lambda w^T).
    pub fn left_eigenvectors(m: &Matrix) -> MathResult<GeneralEigenDecomposition> {
        let decomp = Self::compute(m)?;
        let left = Self::compute_eigenvectors(&m.transpose(), &decomp.eigenvalues)?;
        
        Ok(GeneralEigenDecomposition {
            eigenvalues: decomp.eigenvalues,
            eigenvectors: decomp.eigenvectors,
            left_eigenvectors: Some(left.transpose()),
        })
    }

    /// Check if matrix is diagonalizable.
    pub fn is_diagonalizable(m: &Matrix, tolerance: f64) -> MathResult<bool> {
        let decomp = Self::compute(m)?;
        let n = m.rows;
        
        // Check if eigenvectors are linearly independent
        let rank = crate::rank::MatrixRank::compute(&decomp.eigenvectors, tolerance)?;
        Ok(rank == n)
    }

    /// Jordan canonical form (simplified - returns eigenvalue decomposition if diagonalizable).
    pub fn jordan_form(m: &Matrix) -> MathResult<(Matrix, Matrix)> {
        let decomp = Self::compute(m)?;
        
        if Self::is_diagonalizable(m, 1e-10)? {
            let jordan = Matrix::diagonal(&decomp.eigenvalues);
            Ok((jordan, decomp.eigenvectors))
        } else {
            // For non-diagonalizable matrices, return Schur form
            let schur = crate::schur::SchurDecompositionImpl::compute(m)?;
            Ok((schur.t, schur.q))
        }
    }
}

/// Eigenvalue sensitivity analysis.
pub struct EigenvalueSensitivity;

impl EigenvalueSensitivity {
    /// Condition number of eigenvalue: κ = ||y|| * ||x|| / |y^T x|.
    pub fn eigenvalue_condition(
        m: &Matrix,
        eigenvalue: f64,
        eigenvector: &[f64],
        left_eigenvector: &[f64],
    ) -> f64 {
        let x_norm: f64 = eigenvector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let y_norm: f64 = left_eigenvector.iter().map(|y| y * y).sum::<f64>().sqrt();
        
        let dot: f64 = eigenvector.iter()
            .zip(left_eigenvector.iter())
            .map(|(x, y)| x * y)
            .sum();
        
        if dot.abs() > 1e-15 {
            x_norm * y_norm / dot.abs()
        } else {
            f64::INFINITY
        }
    }

    /// Pseudospectrum: set of eigenvalues of A + E for ||E|| < epsilon.
    pub fn pseudospectrum_radius(m: &Matrix, epsilon: f64) -> f64 {
        // Simplified: return epsilon as radius
        epsilon
    }

    /// Spectral abscissa (maximum real part of eigenvalues).
    pub fn spectral_abscissa(m: &Matrix) -> MathResult<f64> {
        let decomp = GeneralEigen::compute(m)?;
        Ok(decomp.eigenvalues.iter()
            .map(|&lambda| lambda)
            .fold(f64::NEG_INFINITY, f64::max))
    }

    /// Spectral radius (maximum absolute value of eigenvalues).
    pub fn spectral_radius(m: &Matrix) -> MathResult<f64> {
        let decomp = GeneralEigen::compute(m)?;
        Ok(decomp.eigenvalues.iter()
            .map(|&lambda| lambda.abs())
            .fold(f64::NEG_INFINITY, f64::max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_general_eigen_symmetric() {
        let m = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 2.0]]).unwrap();
        let decomp = GeneralEigen::compute(&m).unwrap();
        
        assert!(decomp.eigenvalues.len() == 2);
        // Should be approximately 3 and 1
        assert!((decomp.eigenvalues[0] - 3.0).abs() < 0.1 || (decomp.eigenvalues[1] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_spectral_radius() {
        let m = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 2.0]]).unwrap();
        let radius = EigenvalueSensitivity::spectral_radius(&m).unwrap();
        assert!((radius - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_spectral_abscissa() {
        let m = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 1.0]]).unwrap();
        let abscissa = EigenvalueSensitivity::spectral_abscissa(&m).unwrap();
        assert!((abscissa - 2.0).abs() < 0.1);
    }
}
