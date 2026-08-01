//! Schur decomposition: A = Q T Q^T where T is quasi-triangular.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Schur decomposition result.
#[derive(Debug, Clone)]
pub struct SchurDecomposition {
    pub q: Matrix,  // Orthogonal matrix
    pub t: Matrix,  // Quasi-triangular (upper triangular with 2x2 blocks for complex eigenvalues)
}

/// Schur decomposition for general matrices.
pub struct SchurDecompositionImpl;

impl SchurDecompositionImpl {
    /// Compute Schur decomposition using QR iteration.
    pub fn compute(m: &Matrix) -> MathResult<SchurDecomposition> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        let mut h = Self::hessenberg(m)?;
        let mut q = Matrix::identity(n);
        
        // QR iteration
        for _ in 0..100 {
            let qr_result = h.qr()?;
            h = qr_result.r.mul(&qr_result.q)?;
            q = q.mul(&qr_result.q)?;
            
            // Check for convergence
            if Self::is_upper_triangular(&h, 1e-10) {
                break;
            }
        }
        
        Ok(SchurDecomposition { q, t: h })
    }

    /// Reduce matrix to upper Hessenberg form.
    fn hessenberg(m: &Matrix) -> MathResult<Matrix> {
        let n = m.rows;
        let mut h = m.clone();
        
        for k in 0..(n - 2) {
            // Householder reflection
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
                
                // Apply Householder to rows
                for j in k..n {
                    let dot: f64 = x.iter()
                        .enumerate()
                        .map(|(o, &vv)| vv * h.get(k + 1 + o, j))
                        .sum();
                    for (o, &vv) in x.iter().enumerate() {
                        h.set(k + 1 + o, j, h.get(k + 1 + o, j) - 2.0 * vv * dot);
                    }
                }
                
                // Apply Householder to columns
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

    /// Check if matrix is upper triangular (within tolerance).
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

    /// Real Schur form for matrices with complex eigenvalues.
    pub fn real_schur(m: &Matrix) -> MathResult<SchurDecomposition> {
        Self::compute(m)
    }

    /// Complex Schur form (simplified - returns real for now).
    pub fn complex_schur(m: &Matrix) -> MathResult<SchurDecomposition> {
        Self::compute(m)
    }

    /// Extract eigenvalues from Schur form.
    pub fn eigenvalues(schur: &SchurDecomposition) -> Vec<f64> {
        let n = schur.t.rows;
        let mut eigenvalues = Vec::new();
        
        let mut i = 0;
        while i < n {
            if i + 1 < n && schur.t.get(i + 1, i).abs() > 1e-10 {
                // 2x2 block - complex conjugate pair
                let a = schur.t.get(i, i);
                let b = schur.t.get(i, i + 1);
                let c = schur.t.get(i + 1, i);
                let d = schur.t.get(i + 1, i + 1);
                
                // Eigenvalues of 2x2 block
                let trace = a + d;
                let det = a * d - b * c;
                let discriminant = trace * trace - 4.0 * det;
                
                if discriminant >= 0.0 {
                    eigenvalues.push((trace + discriminant.sqrt()) / 2.0);
                    eigenvalues.push((trace - discriminant.sqrt()) / 2.0);
                } else {
                    // Complex eigenvalues - store real parts
                    eigenvalues.push(trace / 2.0);
                    eigenvalues.push(trace / 2.0);
                }
                i += 2;
            } else {
                eigenvalues.push(schur.t.get(i, i));
                i += 1;
            }
        }
        
        eigenvalues
    }
}

/// Schur form applications.
pub struct SchurApplications;

impl SchurApplications {
    /// Compute matrix exponential using Schur decomposition.
    pub fn exp(m: &Matrix) -> MathResult<Matrix> {
        let schur = SchurDecompositionImpl::compute(m)?;
        let exp_t = Self::exp_triangular(&schur.t)?;
        schur.q.mul(&exp_t)?.mul(&schur.q.transpose())
    }

    /// Exponential of triangular matrix.
    fn exp_triangular(t: &Matrix) -> MathResult<Matrix> {
        let n = t.rows;
        let mut exp_t = Matrix::zeros(n, n);
        
        // Diagonal elements
        for i in 0..n {
            exp_t.set(i, i, t.get(i, i).exp());
        }
        
        // Upper triangular part
        for j in 1..n {
            for i in (0..j).rev() {
                let mut sum = 0.0;
                for k in (i + 1)..=j {
                    sum += t.get(i, k) * exp_t.get(k, j);
                }
                let a_ii = t.get(i, i);
                let a_jj = t.get(i, i);
                if (a_ii - a_jj).abs() > 1e-15 {
                    exp_t.set(i, j, (exp_t.get(i, i) - exp_t.get(j, j)) / (a_ii - a_jj) + sum / (a_ii - a_jj));
                } else {
                    exp_t.set(i, j, exp_t.get(i, i) + sum);
                }
            }
        }
        
        Ok(exp_t)
    }

    /// Compute matrix function using Schur decomposition.
    pub fn matrix_function(m: &Matrix, f: impl Fn(f64) -> f64) -> MathResult<Matrix> {
        let schur = SchurDecompositionImpl::compute(m)?;
        let f_t = Self::apply_function_triangular(&schur.t, &f)?;
        schur.q.mul(&f_t)?.mul(&schur.q.transpose())
    }

    /// Apply function to triangular matrix.
    fn apply_function_triangular(t: &Matrix, f: &impl Fn(f64) -> f64) -> MathResult<Matrix> {
        let n = t.rows;
        let mut f_t = Matrix::zeros(n, n);
        
        // Diagonal elements
        for i in 0..n {
            f_t.set(i, i, f(t.get(i, i)));
        }
        
        // Upper triangular part (Parlett's recurrence)
        for j in 1..n {
            for i in (0..j).rev() {
                let mut sum = 0.0;
                for k in (i + 1)..=j {
                    sum += t.get(i, k) * f_t.get(k, j);
                }
                let a_ii = t.get(i, i);
                let a_jj = t.get(j, j);
                if (a_ii - a_jj).abs() > 1e-15 {
                    f_t.set(i, j, (f(a_ii) - f(a_jj)) / (a_ii - a_jj) + sum / (a_ii - a_jj));
                } else {
                    // Use derivative for repeated eigenvalues
                    let h = 1e-8;
                    let df = (f(a_ii + h) - f(a_ii)) / h;
                    f_t.set(i, j, df + sum);
                }
            }
        }
        
        Ok(f_t)
    }

    /// Solve Sylvester equation using Schur form.
    pub fn solve_sylvester(a: &Matrix, b: &Matrix, c: &Matrix) -> MathResult<Matrix> {
        let schur_a = SchurDecompositionImpl::compute(a)?;
        let schur_b = SchurDecompositionImpl::compute(&b.transpose())?;
        
        let qta_c = schur_a.q.transpose().mul(c)?;
        let qta_c_qb = qta_c.mul(&schur_b.q)?;
        
        let y = Self::solve_sylvester_triangular(&schur_a.t, &schur_b.t, &qta_c_qb)?;
        
        schur_a.q.mul(&y)?.mul(&schur_b.q.transpose())
    }

    /// Solve Sylvester equation with triangular matrices.
    fn solve_sylvester_triangular(a: &Matrix, b: &Matrix, c: &Matrix) -> MathResult<Matrix> {
        let m = a.rows;
        let n = b.cols;
        let mut x = Matrix::zeros(m, n);
        
        for j in 0..n {
            for i in (0..m).rev() {
                let mut sum = c.get(i, j);
                
                for k in (i + 1)..m {
                    sum -= a.get(i, k) * x.get(k, j);
                }
                
                for k in (j + 1)..n {
                    sum -= x.get(i, k) * b.get(k, j);
                }
                
                let denom = a.get(i, i) + b.get(j, j);
                if denom.abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("singular Sylvester equation"));
                }
                
                x.set(i, j, sum / denom);
            }
        }
        
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schur_decomposition() {
        let m = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let schur = SchurDecompositionImpl::compute(&m).unwrap();
        
        // Verify A = Q T Q^T
        let reconstructed = schur.q.mul(&schur.t)?.mul(&schur.q.transpose())?;
        
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed.get(i, j) - m.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_schur_eigenvalues() {
        let m = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 2.0]]).unwrap();
        let schur = SchurDecompositionImpl::compute(&m).unwrap();
        let eigenvalues = SchurDecompositionImpl::eigenvalues(&schur);
        
        // Should be 3 and 1
        assert!(eigenvalues.len() == 2);
        assert!((eigenvalues[0] - 3.0).abs() < 1e-10 || (eigenvalues[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_schur_exp() {
        let m = Matrix::identity(2);
        let exp_m = SchurApplications::exp(&m).unwrap();
        
        let e = core::f64::consts::E;
        assert!((exp_m.get(0, 0) - e).abs() < 1e-10);
        assert!((exp_m.get(1, 1) - e).abs() < 1e-10);
    }
}
