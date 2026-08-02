//! LDL decomposition for symmetric indefinite matrices.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// LDL decomposition result: A = L D L^T where L is unit lower triangular and D is diagonal.
#[derive(Debug, Clone)]
pub struct LdlDecomposition {
    pub l: Matrix,      // Unit lower triangular
    pub d: Matrix,      // Diagonal matrix
}

/// LDL decomposition for symmetric matrices.
pub struct LdlDecompositionImpl;

impl LdlDecompositionImpl {
    /// Compute LDL decomposition: A = L D L^T.
    pub fn compute(m: &Matrix) -> MathResult<LdlDecomposition> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        if !m.is_symmetric(1e-10) {
            return Err(MathError::InvalidArgument("matrix must be symmetric"));
        }
        
        let n = m.rows;
        let mut l = Matrix::zeros(n, n);
        let mut d = Matrix::zeros(n, n);
        
        for i in 0..n {
            l.set(i, i, 1.0);
            
            for j in 0..=i {
                let mut sum = m.get(i, j);
                
                for k in 0..j {
                    sum -= l.get(i, k) * d.get(k, k) * l.get(j, k);
                }
                
                if i == j {
                    d.set(i, i, sum);
                    if d.get(i, i).abs() < 1e-15 {
                        return Err(MathError::InvalidArgument("zero pivot in LDL decomposition"));
                    }
                } else {
                    l.set(i, j, sum / d.get(j, j));
                }
            }
        }
        
        Ok(LdlDecomposition { l, d })
    }

    /// LDL decomposition with pivoting for stability.
    pub fn compute_pivoted(m: &Matrix) -> MathResult<(LdlDecomposition, Vec<usize>)> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        let mut a = m.clone();
        let mut pivots: Vec<usize> = (0..n).collect();
        let mut l = Matrix::zeros(n, n);
        let mut d = Matrix::zeros(n, n);
        
        for k in 0..n {
            // Find pivot
            let mut max_val = a.get(k, k).abs();
            let mut max_idx = k;
            
            for i in (k + 1)..n {
                let val = a.get(i, i).abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            if max_val < 1e-15 {
                return Err(MathError::InvalidArgument("singular matrix in LDL"));
            }
            
            // Swap rows and columns
            if max_idx != k {
                pivots.swap(k, max_idx);
                for j in 0..n {
                    let temp = a.get(k, j);
                    a.set(k, j, a.get(max_idx, j));
                    a.set(max_idx, j, temp);
                }
                for j in 0..n {
                    let temp = a.get(j, k);
                    a.set(j, k, a.get(j, max_idx));
                    a.set(j, max_idx, temp);
                }
            }
            
            d.set(k, k, a.get(k, k));
            l.set(k, k, 1.0);
            
            for i in (k + 1)..n {
                l.set(i, k, a.get(i, k) / d.get(k, k));
            }
            
            for j in (k + 1)..n {
                for i in j..n {
                    a.set(i, j, a.get(i, j) - l.get(i, k) * d.get(k, k) * l.get(j, k));
                }
            }
        }
        
        Ok((LdlDecomposition { l, d }, pivots))
    }

    /// Solve Ax = b using LDL decomposition.
    pub fn solve(ldl: &LdlDecomposition, b: &mathverse_vector::Vector) -> MathResult<mathverse_vector::Vector> {
        let n = ldl.l.rows;
        if b.len() != n {
            return Err(MathError::DimensionMismatch);
        }
        
        // Solve L y = b (forward substitution)
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = b.get(i);
            for j in 0..i {
                sum -= ldl.l.get(i, j) * y[j];
            }
            y[i] = sum;
        }
        
        // Solve D z = y
        let mut z = vec![0.0; n];
        for i in 0..n {
            let d_val = ldl.d.get(i, i);
            if d_val.abs() < 1e-15 {
                return Err(MathError::InvalidArgument("zero diagonal in D"));
            }
            z[i] = y[i] / d_val;
        }
        
        // Solve L^T x = z (backward substitution)
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = z[i];
            for j in (i + 1)..n {
                sum -= ldl.l.get(j, i) * x[j];
            }
            x[i] = sum;
        }
        
        Ok(mathverse_vector::Vector::new(x))
    }

    /// Compute determinant from LDL: det(A) = det(D) = Π d_i.
    pub fn determinant(ldl: &LdlDecomposition) -> f64 {
        let mut det = 1.0;
        for i in 0..ldl.d.rows {
            det *= ldl.d.get(i, i);
        }
        det
    }

    /// Compute inverse using LDL decomposition.
    pub fn inverse(ldl: &LdlDecomposition) -> MathResult<Matrix> {
        let n = ldl.l.rows;
        let mut inv = Matrix::zeros(n, n);
        
        for k in 0..n {
            let mut e = vec![0.0; n];
            e[k] = 1.0;
            let col = Self::solve(ldl, &mathverse_vector::Vector::new(e))?;
            
            for i in 0..n {
                inv.set(i, k, col.get(i));
            }
        }
        
        Ok(inv)
    }
}

/// Bunch-Kaufman decomposition for symmetric indefinite matrices.
pub struct BunchKaufman;

impl BunchKaufman {
    /// Bunch-Kaufman decomposition: P A P^T = L T L^T where T is block diagonal.
    pub fn compute(m: &Matrix) -> MathResult<(Matrix, Matrix, Vec<usize>)> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        if !m.is_symmetric(1e-10) {
            return Err(MathError::InvalidArgument("matrix must be symmetric"));
        }
        
        let n = m.rows;
        let mut a = m.clone();
        let mut l = Matrix::identity(n);
        let mut pivots: Vec<usize> = (0..n).collect();
        
        let mut k = 0;
        while k < n - 1 {
            let lambda = a.get(k, k).abs();
            let sigma = a.get(k + 1, k + 1).abs();
            let gamma = a.get(k, k + 1).abs();
            
            if lambda >= sigma {
                if lambda >= gamma {
                    // 1x1 pivot
                    let pivot = a.get(k, k);
                    if pivot.abs() < 1e-15 {
                        return Err(MathError::InvalidArgument("zero pivot"));
                    }
                    
                    for i in (k + 1)..n {
                        let factor = a.get(i, k) / pivot;
                        l.set(i, k, factor);
                        
                        for j in k..n {
                            a.set(i, j, a.get(i, j) - factor * a.get(k, j));
                        }
                    }
                    k += 1;
                } else {
                    // 2x2 pivot
                    Self::apply_2x2_pivot(&mut a, &mut l, k, &mut pivots);
                    k += 2;
                }
            } else {
                if sigma >= gamma {
                    // 1x1 pivot (swap)
                    pivots.swap(k, k + 1);
                    Self::swap_rows_cols(&mut a, k, k + 1);
                    Self::swap_rows(&mut l, k, k + 1);
                    
                    let pivot = a.get(k, k);
                    if pivot.abs() < 1e-15 {
                        return Err(MathError::InvalidArgument("zero pivot"));
                    }
                    
                    for i in (k + 1)..n {
                        let factor = a.get(i, k) / pivot;
                        l.set(i, k, factor);
                        
                        for j in k..n {
                            a.set(i, j, a.get(i, j) - factor * a.get(k, j));
                        }
                    }
                    k += 1;
                } else {
                    // 2x2 pivot
                    Self::apply_2x2_pivot(&mut a, &mut l, k, &mut pivots);
                    k += 2;
                }
            }
        }
        
        Ok((l, a, pivots))
    }
    
    fn apply_2x2_pivot(a: &mut Matrix, l: &mut Matrix, k: usize, _pivots: &mut Vec<usize>) {
        let n = a.rows;
        let d11 = a.get(k, k);
        let d22 = a.get(k + 1, k + 1);
        let d12 = a.get(k, k + 1);
        
        let det = d11 * d22 - d12 * d12;
        if det.abs() < 1e-15 {
            return;
        }
        
        for i in (k + 2)..n {
            let alpha = a.get(i, k);
            let beta = a.get(i, k + 1);
            
            l.set(i, k, (d22 * alpha - d12 * beta) / det);
            l.set(i, k + 1, (d11 * beta - d12 * alpha) / det);
            
            for j in (k + 2)..n {
                a.set(i, j, a.get(i, j) - l.get(i, k) * a.get(k, j) - l.get(i, k + 1) * a.get(k + 1, j));
            }
        }
    }
    
    fn swap_rows_cols(m: &mut Matrix, i: usize, j: usize) {
        let n = m.rows;
        for k in 0..n {
            let temp = m.get(i, k);
            m.set(i, k, m.get(j, k));
            m.set(j, k, temp);
        }
        for k in 0..n {
            let temp = m.get(k, i);
            m.set(k, i, m.get(k, j));
            m.set(k, j, temp);
        }
    }
    
    fn swap_rows(m: &mut Matrix, i: usize, j: usize) {
        let n = m.cols;
        for k in 0..n {
            let temp = m.get(i, k);
            m.set(i, k, m.get(j, k));
            m.set(j, k, temp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ldl_decomposition() {
        let m = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let ldl = LdlDecompositionImpl::compute(&m).unwrap();
        
        // Verify A = L D L^T
        let ldl_t = ldl.l.transpose();
        let ld = ldl.l.mul(&ldl.d).unwrap();
        let reconstructed = ld.mul(&ldl_t).unwrap();
        
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed.get(i, j) - m.get(i, j)).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_ldl_solve() {
        let m = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let ldl = LdlDecompositionImpl::compute(&m).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0]);
        
        let x = LdlDecompositionImpl::solve(&ldl, &b).unwrap();
        let back = m.mul_vec(&x).unwrap();
        
        assert!((back.get(0) - 1.0).abs() < 1e-10);
        assert!((back.get(1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ldl_determinant() {
        let m = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let ldl = LdlDecompositionImpl::compute(&m).unwrap();
        let det_ldl = LdlDecompositionImpl::determinant(&ldl);
        let det = m.det().unwrap();
        
        assert!((det_ldl - det).abs() < 1e-10);
    }
}
