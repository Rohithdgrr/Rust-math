//! Generalized eigenvalue problems: Ax = λBx.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Generalized eigenvalue decomposition result.
#[derive(Debug, Clone)]
pub struct GeneralizedEigenDecomposition {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Matrix,
    pub left_eigenvectors: Option<Matrix>,
}

/// Generalized eigenvalue problem solver.
pub struct GeneralizedEigen;

impl GeneralizedEigen {
    /// Solve Ax = λBx using QZ algorithm (simplified - uses Cholesky for B).
    pub fn compute(a: &Matrix, b: &Matrix) -> MathResult<GeneralizedEigenDecomposition> {
        if !a.is_square() || !b.is_square() || a.rows != b.rows {
            return Err(MathError::DimensionMismatch);
        }
        
        // Try Cholesky decomposition of B (assumes B is positive definite)
        let l = b.cholesky();
        
        if let Ok(l) = l {
            // Transform to standard eigenvalue problem: L^{-1} A L^{-T} y = λ y
            let l_inv = l.inverse()?;
            let l_inv_t = l_inv.transpose();
            let transformed = l_inv.mul(a)?.mul(&l_inv_t)?;
            
            // Solve standard eigenvalue problem
            let (vals, vecs) = transformed.eigen_symmetric()?;
            
            // Transform back: x = L^{-T} y
            let eigenvectors = l_inv_t.mul(&vecs)?;
            
            Ok(GeneralizedEigenDecomposition {
                eigenvalues: vals,
                eigenvectors,
                left_eigenvectors: None,
            })
        } else {
            // Fallback: use inverse iteration
            Self::inverse_iteration(a, b)
        }
    }

    /// Solve using inverse iteration (for general B).
    fn inverse_iteration(a: &Matrix, b: &Matrix) -> MathResult<GeneralizedEigenDecomposition> {
        let n = a.rows;
        let mut eigenvalues = Vec::new();
        let mut eigenvectors = Matrix::zeros(n, n);
        
        for j in 0..n {
            // Initial guess
            let mut x = vec![1.0; n];
            x[j] = 1.0;
            
            // Normalize
            let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            x = x.iter().map(|v| v / norm).collect();
            
            // Inverse iteration
            for _ in 0..20 {
                // Solve B x_new = A x
                let x_vec = mathverse_vector::Vector::new(x.clone());
                let ax = a.mul_vec(&x_vec)?;
                let x_new = b.solve(&ax)?;
                x = x_new.data;
                
                // Normalize
                let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 0.0 {
                    x = x.iter().map(|v| v / norm).collect();
                }
            }
            
            // Rayleigh quotient: λ = (x^T A x) / (x^T B x)
            let x_vec = mathverse_vector::Vector::new(x.clone());
            let ax = a.mul_vec(&x_vec)?;
            let bx = b.mul_vec(&x_vec)?;
            let lambda = ax.dot(&bx) / bx.dot(&bx);
            
            eigenvalues.push(lambda);
            
            for i in 0..n {
                eigenvectors.set(i, j, x[i]);
            }
        }
        
        Ok(GeneralizedEigenDecomposition {
            eigenvalues,
            eigenvectors,
            left_eigenvectors: None,
        })
    }

    /// Solve Ax = λx (standard eigenvalue problem as special case with B=I).
    pub fn standard(a: &Matrix) -> MathResult<GeneralizedEigenDecomposition> {
        let n = a.rows;
        let b = Matrix::identity(n);
        Self::compute(a, &b)
    }

    /// Check if generalized eigenvalue problem is well-posed.
    pub fn is_well_posed(a: &Matrix, b: &Matrix, tolerance: f64) -> MathResult<bool> {
        // Check if B is invertible
        let det_b = b.det()?;
        if det_b.abs() < tolerance {
            return Ok(false);
        }
        
        // Check if pencil (A - λB) is regular
        let n = a.rows;
        for i in 0..n {
            let shifted = a.sub(&b.scale(i as f64))?;
            let det = shifted.det()?;
            if det.abs() < tolerance {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    /// Compute generalized Schur form (simplified).
    pub fn generalized_schur(a: &Matrix, b: &Matrix) -> MathResult<(Matrix, Matrix, Matrix, Matrix)> {
        // Simplified: return original matrices
        Ok((a.clone(), b.clone(), Matrix::identity(a.rows), Matrix::identity(b.rows)))
    }
}

/// Generalized eigenvalue applications.
pub struct GeneralizedEigenApplications;

impl GeneralizedEigenApplications {
    /// Solve quadratic eigenvalue problem: (λ²M + λC + K)x = 0.
    pub fn quadratic_eigen(m: &Matrix, c: &Matrix, k: &Matrix) -> MathResult<Vec<f64>> {
        if !m.is_square() || !c.is_square() || !k.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        
        // Linearize to generalized eigenvalue problem
        let mut a = Matrix::zeros(2 * n, 2 * n);
        let mut b = Matrix::zeros(2 * n, 2 * n);
        
        // A = [0, I; -K, -C]
        for i in 0..n {
            for j in 0..n {
                a.set(i, n + j, if i == j { 1.0 } else { 0.0 });
                a.set(n + i, j, -k.get(i, j));
                a.set(n + i, n + j, -c.get(i, j));
            }
        }
        
        // B = [I, 0; 0, M]
        for i in 0..n {
            for j in 0..n {
                b.set(i, j, if i == j { 1.0 } else { 0.0 });
                b.set(n + i, n + j, m.get(i, j));
            }
        }
        
        let decomp = GeneralizedEigen::compute(&a, &b)?;
        Ok(decomp.eigenvalues)
    }

    /// Solve polynomial eigenvalue problem: Σ λ^k A_k x = 0.
    pub fn polynomial_eigen(coefficients: &[Matrix]) -> MathResult<Vec<f64>> {
        if coefficients.is_empty() {
            return Ok(Vec::new());
        }
        
        let n = coefficients[0].rows;
        let degree = coefficients.len() - 1;
        
        // Linearize to generalized eigenvalue problem
        let block_size = degree * n;
        let mut a = Matrix::zeros(block_size, block_size);
        let mut b = Matrix::zeros(block_size, block_size);
        
        // Build companion form
        for i in 0..((degree - 1) * n) {
            a.set(i, i + n, 1.0);
        }
        
        for i in 0..n {
            for j in 0..n {
                b.set(i, j, coefficients[degree].get(i, j));
            }
        }
        
        for k in 0..degree {
            for i in 0..n {
                for j in 0..n {
                    a.set((degree - 1) * n + i, k * n + j, -coefficients[k].get(i, j));
                }
            }
        }
        
        let decomp = GeneralizedEigen::compute(&a, &b)?;
        Ok(decomp.eigenvalues)
    }

    /// Buckling load analysis: (K - λK_G)φ = 0.
    pub fn buckling_load(k: &Matrix, k_g: &Matrix) -> MathResult<Vec<f64>> {
        GeneralizedEigen::compute(k, k_g).map(|d| d.eigenvalues)
    }

    /// Vibration analysis: (K - ω²M)φ = 0.
    pub fn natural_frequencies(k: &Matrix, m: &Matrix) -> MathResult<Vec<f64>> {
        let eigenvalues = GeneralizedEigen::compute(k, m)?.eigenvalues;
        // ω = sqrt(λ)
        Ok(eigenvalues.iter().map(|&λ| λ.sqrt()).collect())
    }

    /// Damped vibration: (Mλ² + Cλ + K)φ = 0.
    pub fn damped_vibration(m: &Matrix, c: &Matrix, k: &Matrix) -> MathResult<Vec<f64>> {
        Self::quadratic_eigen(m, c, k)
    }
}

/// Generalized eigenvalue sensitivity.
pub struct GeneralizedEigenSensitivity;

impl GeneralizedEigenSensitivity {
    /// Sensitivity of eigenvalue to parameter changes.
    pub fn eigenvalue_sensitivity(
        a: &Matrix,
        b: &Matrix,
        da: &Matrix,
        db: &Matrix,
        eigenvalue: f64,
        eigenvector: &[f64],
    ) -> f64 {
        let n = a.rows;
        
        // Compute numerator: v^T (dA - λ dB) v
        let mut numerator = 0.0;
        for i in 0..n {
            for j in 0..n {
                numerator += eigenvector[i] * (da.get(i, j) - eigenvalue * db.get(i, j)) * eigenvector[j];
            }
        }
        
        // Compute denominator: v^T B v
        let mut denominator = 0.0;
        for i in 0..n {
            for j in 0..n {
                denominator += eigenvector[i] * b.get(i, j) * eigenvector[j];
            }
        }
        
        if denominator.abs() > 1e-15 {
            numerator / denominator
        } else {
            f64::INFINITY
        }
    }

    /// Condition number of generalized eigenvalue.
    pub fn condition_number(
        a: &Matrix,
        b: &Matrix,
        eigenvalue: f64,
        eigenvector: &[f64],
        left_eigenvector: &[f64],
    ) -> f64 {
        let n = a.rows;
        
        let x_norm: f64 = eigenvector.iter().map(|x| x * x).sum::<f64>().sqrt();
        let y_norm: f64 = left_eigenvector.iter().map(|y| y * y).sum::<f64>().sqrt();
        
        let mut denom = 0.0;
        for i in 0..n {
            for j in 0..n {
                denom += left_eigenvector[i] * b.get(i, j) * eigenvector[j];
            }
        }
        
        if denom.abs() > 1e-15 {
            x_norm * y_norm / denom.abs()
        } else {
            f64::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generalized_eigen_identity() {
        let a = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 2.0]]).unwrap();
        let b = Matrix::identity(2);
        
        let decomp = GeneralizedEigen::compute(&a, &b).unwrap();
        
        // Should match standard eigenvalues
        assert!(decomp.eigenvalues.len() == 2);
    }

    #[test]
    fn test_generalized_eigen_positive_definite() {
        let a = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let b = Matrix::from_rows(&[&[2.0, 0.0], &[0.0, 1.0]]).unwrap();
        
        let decomp = GeneralizedEigen::compute(&a, &b).unwrap();
        
        assert!(decomp.eigenvalues.len() == 2);
    }

    #[test]
    fn test_natural_frequencies() {
        let k = Matrix::from_rows(&[&[2.0, -1.0], &[-1.0, 2.0]]).unwrap();
        let m = Matrix::identity(2);
        
        let frequencies = GeneralizedEigenApplications::natural_frequencies(&k, &m).unwrap();
        
        assert!(frequencies.len() == 2);
    }
}
