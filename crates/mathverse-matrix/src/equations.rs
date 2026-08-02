//! Matrix equation solvers: Sylvester, Lyapunov, Stein, and related equations.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Sylvester equation: AX + XB = C.
pub struct SylvesterEquation;

impl SylvesterEquation {
    /// Solve Sylvester equation using Bartels-Stewart algorithm.
    pub fn solve(a: &Matrix, b: &Matrix, c: &Matrix) -> MathResult<Matrix> {
        if !a.is_square() || !b.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        if c.rows != a.rows || c.cols != b.cols {
            return Err(MathError::DimensionMismatch);
        }
        
        // Reduce A and B to Schur form
        let schur_a = crate::schur::SchurDecompositionImpl::compute(a)?;
        let schur_b = crate::schur::SchurDecompositionImpl::compute(b)?;
        
        // Transform C: C' = Q_a^T C Q_b
        let qta_c = schur_a.q.transpose().mul(c)?;
        let c_prime = qta_c.mul(&schur_b.q)?;
        
        // Solve triangular Sylvester equation
        let y = Self::solve_triangular(&schur_a.t, &schur_b.t, &c_prime)?;
        
        // Transform back: X = Q_a Y Q_b^T
        schur_a.q.mul(&y)?.mul(&schur_b.q.transpose())
    }

    /// Solve Sylvester equation with triangular matrices.
    fn solve_triangular(a: &Matrix, b: &Matrix, c: &Matrix) -> MathResult<Matrix> {
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

    /// Solve using Kronecker product formulation: vec(X) = (I ⊗ A + B^T ⊗ I)^{-1} vec(C).
    pub fn solve_kronecker(a: &Matrix, b: &Matrix, c: &Matrix) -> MathResult<Matrix> {
        let m = a.rows;
        let n = b.cols;
        
        // Build Kronecker sum matrix
        let i_m = Matrix::identity(m);
        let i_n = Matrix::identity(n);

        let kron_a = crate::kronecker::KroneckerProduct::compute(&i_n, a);
        let kron_bt = crate::kronecker::KroneckerProduct::compute(&b.transpose(), &i_m);
        
        let kron_sum = kron_a.add(&kron_bt)?;
        
        // Vectorize C
        let mut c_vec = vec![0.0; m * n];
        for j in 0..n {
            for i in 0..m {
                c_vec[j * m + i] = c.get(i, j);
            }
        }
        
        // Solve linear system
        let c_vec_mat = mathverse_vector::Vector::new(c_vec);
        let x_vec = kron_sum.solve(&c_vec_mat)?;
        
        // Reshape back to matrix
        let mut x = Matrix::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                x.set(i, j, x_vec.get(j * m + i));
            }
        }
        
        Ok(x)
    }

    /// Check if Sylvester equation has unique solution.
    pub fn has_unique_solution(a: &Matrix, b: &Matrix) -> bool {
        let (vals_a, _) = match a.eigen_symmetric() {
            Ok(result) => result,
            Err(_) => return false,
        };
        
        let (vals_b, _) = match b.eigen_symmetric() {
            Ok(result) => result,
            Err(_) => return false,
        };
        
        // Check if λ_i + μ_j ≠ 0 for all i, j
        for &lambda in &vals_a {
            for &mu in &vals_b {
                if (lambda + mu).abs() < 1e-10 {
                    return false;
                }
            }
        }
        
        true
    }
}

/// Lyapunov equation: AX + XA^T = Q.
pub struct LyapunovEquation;

impl LyapunovEquation {
    /// Solve Lyapunov equation using Bartels-Stewart algorithm.
    pub fn solve(a: &Matrix, q: &Matrix) -> MathResult<Matrix> {
        if !a.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        if q.rows != a.rows || q.cols != a.rows {
            return Err(MathError::DimensionMismatch);
        }
        
        // Reduce A to Schur form
        let schur_a = crate::schur::SchurDecompositionImpl::compute(a)?;
        
        // Transform Q: Q' = Q_a^T Q Q_a
        let qta_q = schur_a.q.transpose().mul(q)?;
        let q_prime = qta_q.mul(&schur_a.q)?;
        
        // Solve triangular Lyapunov equation
        let y = Self::solve_triangular(&schur_a.t, &q_prime)?;
        
        // Transform back: X = Q_a Y Q_a^T
        schur_a.q.mul(&y)?.mul(&schur_a.q.transpose())
    }

    /// Solve Lyapunov equation with triangular matrix.
    fn solve_triangular(a: &Matrix, q: &Matrix) -> MathResult<Matrix> {
        let n = a.rows;
        let mut x = Matrix::zeros(n, n);
        
        for j in (0..n).rev() {
            for i in (j..n).rev() {
                let mut sum = q.get(i, j);
                
                for k in (i + 1)..n {
                    sum -= a.get(i, k) * x.get(k, j);
                }
                
                for k in (j + 1)..n {
                    sum -= x.get(i, k) * a.get(j, k);
                }
                
                let denom = a.get(i, i) + a.get(j, j);
                if denom.abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("singular Lyapunov equation"));
                }
                
                x.set(i, j, sum / denom);
                x.set(j, i, x.get(i, j)); // Symmetric
            }
        }
        
        Ok(x)
    }

    /// Solve using Kronecker product: vec(X) = (I ⊗ A + A ⊗ I)^{-1} vec(Q).
    pub fn solve_kronecker(a: &Matrix, q: &Matrix) -> MathResult<Matrix> {
        let n = a.rows;
        
        let i_n = Matrix::identity(n);
        let kron_a = crate::kronecker::KroneckerProduct::compute(&i_n, a);
        let kron_a_t = crate::kronecker::KroneckerProduct::compute(&a, &i_n);
        
        let kron_sum = kron_a.add(&kron_a_t)?;
        
        // Vectorize Q
        let mut q_vec = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                q_vec[j * n + i] = q.get(i, j);
            }
        }
        
        let q_vec_mat = mathverse_vector::Vector::new(q_vec);
        let x_vec = kron_sum.solve(&q_vec_mat)?;
        
        // Reshape
        let mut x = Matrix::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                x.set(i, j, x_vec.get(j * n + i));
            }
        }
        
        Ok(x)
    }

    /// Check if Lyapunov equation has unique solution (A stable).
    pub fn has_unique_solution(a: &Matrix) -> bool {
        let (vals, _) = match a.eigen_symmetric() {
            Ok(result) => result,
            Err(_) => return false,
        };
        
        // Check if all eigenvalues have negative real parts
        vals.iter().all(|&v| v < 0.0)
    }

    /// Solve discrete Lyapunov equation: AXA^T - X + Q = 0.
    pub fn solve_discrete(a: &Matrix, q: &Matrix) -> MathResult<Matrix> {
        let _n = a.rows;
        
        // Use iterative method
        let mut x = q.clone();
        
        for _ in 0..100 {
            let axa = a.mul(&x)?.mul(&a.transpose())?;
            let x_new = axa.add(&q)?;
            
            let diff = x_new.sub(&x)?;
            let norm = crate::norms::MatrixNorms::frobenius(&diff);
            
            x = x_new;
            
            if norm < 1e-10 {
                break;
            }
        }
        
        Ok(x)
    }
}

/// Stein equation: X - AXB = C.
pub struct SteinEquation;

impl SteinEquation {
    /// Solve Stein equation using Kronecker product.
    pub fn solve(a: &Matrix, b: &Matrix, c: &Matrix) -> MathResult<Matrix> {
        let m = a.rows;
        let n = b.cols;
        
        let i_m = Matrix::identity(m);
        let i_n = Matrix::identity(n);

        let kron_a = crate::kronecker::KroneckerProduct::compute(&b.transpose(), a);
        let kron_i = crate::kronecker::KroneckerProduct::compute(&i_n, &i_m);
        
        let kron_diff = kron_i.sub(&kron_a)?;
        
        // Vectorize C
        let mut c_vec = vec![0.0; m * n];
        for j in 0..n {
            for i in 0..m {
                c_vec[j * m + i] = c.get(i, j);
            }
        }
        
        let c_vec_mat = mathverse_vector::Vector::new(c_vec);
        let x_vec = kron_diff.solve(&c_vec_mat)?;
        
        // Reshape
        let mut x = Matrix::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                x.set(i, j, x_vec.get(j * m + i));
            }
        }
        
        Ok(x)
    }

    /// Check if Stein equation has unique solution.
    pub fn has_unique_solution(a: &Matrix, b: &Matrix) -> bool {
        let (vals_a, _) = match a.eigen_symmetric() {
            Ok(result) => result,
            Err(_) => return false,
        };
        
        let (vals_b, _) = match b.eigen_symmetric() {
            Ok(result) => result,
            Err(_) => return false,
        };
        
        // Check if λ_i * μ_j ≠ 1 for all i, j
        for &lambda in &vals_a {
            for &mu in &vals_b {
                if (lambda * mu - 1.0).abs() < 1e-10 {
                    return false;
                }
            }
        }
        
        true
    }
}

/// General Sylvester equation: Σ A_k X B_k = C.
pub struct GeneralSylvester;

impl GeneralSylvester {
    /// Solve using iterative method (simplified).
    pub fn solve(
        a_list: &[Matrix],
        b_list: &[Matrix],
        c: &Matrix,
    ) -> MathResult<Matrix> {
        if a_list.len() != b_list.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = c.rows;
        let mut x = Matrix::zeros(n, n);
        
        // Fixed-point iteration
        for _ in 0..100 {
            let mut residual = c.clone();
            
            for (a, b) in a_list.iter().zip(b_list.iter()) {
                let axb = a.mul(&x)?.mul(b)?;
                residual = residual.sub(&axb)?;
            }
            
            let norm = crate::norms::MatrixNorms::frobenius(&residual);
            if norm < 1e-10 {
                break;
            }
            
            // Simple update (not optimal)
            x = x.add(&residual)?;
        }
        
        Ok(x)
    }
}

/// Riccati equation: A^T X + X A - X B R^{-1} B^T X + Q = 0.
pub struct RiccatiEquation;

impl RiccatiEquation {
    /// Solve continuous-time algebraic Riccati equation using Kleinman algorithm.
    pub fn solve_continuous(
        a: &Matrix,
        b: &Matrix,
        q: &Matrix,
        r: &Matrix,
        max_iterations: usize,
    ) -> MathResult<Matrix> {
        if !a.is_square() || !q.is_square() || a.rows != q.rows {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = a.rows;
        let r_inv = r.inverse()?;
        
        // Start with X = 0
        let mut x = Matrix::zeros(n, n);
        
        for _ in 0..max_iterations {
            // Compute A_cl = A - B R^{-1} B^T X
            let br_inv = b.mul(&r_inv)?;
            let br_inv_bt = br_inv.mul(&b.transpose())?;
            let br_inv_bt_x = br_inv_bt.mul(&x)?;
            let a_cl = a.sub(&br_inv_bt_x)?;
            
            // Solve Lyapunov: A_cl^T X + X A_cl = -Q - X B R^{-1} B^T X
            let rhs = q.add(&x.mul(&br_inv_bt_x)?)?;
            let rhs = rhs.scale(-1.0);
            
            let x_new = LyapunovEquation::solve(&a_cl.transpose(), &rhs)?;
            
            // Check convergence
            let diff = x_new.sub(&x)?;
            let norm = crate::norms::MatrixNorms::frobenius(&diff);
            
            x = x_new;
            
            if norm < 1e-10 {
                break;
            }
        }
        
        Ok(x)
    }

    /// Solve discrete-time algebraic Riccati equation.
    pub fn solve_discrete(
        a: &Matrix,
        b: &Matrix,
        q: &Matrix,
        r: &Matrix,
        max_iterations: usize,
    ) -> MathResult<Matrix> {
        let n = a.rows;
        
        let mut x = Matrix::zeros(n, n);
        
        for _ in 0..max_iterations {
            // X_new = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q
            let ax = a.transpose().mul(&x)?;
            let axa = ax.mul(a)?;
            
            let bx = b.transpose().mul(&x)?;
            let bxb = bx.mul(b)?;
            let r_plus = r.add(&bxb)?;
            let r_plus_inv = r_plus.inverse()?;
            
            // A^T X B (R + B^T X B)^{-1} B^T X A
            let axb = ax.mul(b)?;
            let quadratic = axb.mul(&r_plus_inv)?.mul(&bx.mul(a)?)?;
            
            let x_new = axa.sub(&quadratic)?.add(q)?;
            
            let diff = x_new.sub(&x)?;
            let norm = crate::norms::MatrixNorms::frobenius(&diff);
            
            x = x_new;
            
            if norm < 1e-10 {
                break;
            }
        }
        
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sylvester_equation() {
        let a = Matrix::from_rows(&[&[1.0, 0.0], &[0.0, 2.0]]).unwrap();
        let b = Matrix::from_rows(&[&[3.0, 0.0], &[0.0, 4.0]]).unwrap();
        let c = Matrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]).unwrap();
        
        let x = SylvesterEquation::solve(&a, &b, &c).unwrap();
        
        // Verify AX + XB = C
        let ax = a.mul(&x).unwrap();
        let xb = x.mul(&b).unwrap();
        let ax_plus_xb = ax.add(&xb).unwrap();
        
        for i in 0..2 {
            for j in 0..2 {
                assert!((ax_plus_xb.get(i, j) - c.get(i, j)).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_lyapunov_equation() {
        let a = Matrix::from_rows(&[&[-1.0, 0.0], &[0.0, -2.0]]).unwrap();
        let q = Matrix::identity(2);
        
        let x = LyapunovEquation::solve(&a, &q).unwrap();
        
        // Verify AX + XA^T = Q
        let ax = a.mul(&x).unwrap();
        let xat = x.mul(&a.transpose()).unwrap();
        let ax_plus_xat = ax.add(&xat).unwrap();
        
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((ax_plus_xat.get(i, j) - want).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_stein_equation() {
        let a = Matrix::from_rows(&[&[0.5, 0.0], &[0.0, 0.3]]).unwrap();
        let b = Matrix::identity(2);
        let c = Matrix::identity(2);
        
        let x = SteinEquation::solve(&a, &b, &c).unwrap();
        
        // Verify X - AXB = C
        let axb = a.mul(&x).unwrap().mul(&b).unwrap();
        let x_minus_axb = x.sub(&axb).unwrap();
        
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((x_minus_axb.get(i, j) - want).abs() < 1e-8);
            }
        }
    }
}
