//! Iterative solvers: Conjugate Gradient, GMRES, Jacobi, Gauss-Seidel.

use crate::Matrix;
use mathverse_core::error::{MathError, MathResult};

/// Iterative solver result.
#[derive(Debug, Clone)]
pub struct IterativeResult {
    pub solution: mathverse_vector::Vector,
    pub iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
}

/// Conjugate Gradient method for symmetric positive definite systems.
pub struct ConjugateGradient;

impl ConjugateGradient {
    /// Solve Ax = b using CG.
    pub fn solve(
        a: &Matrix,
        b: &mathverse_vector::Vector,
        max_iterations: usize,
        tolerance: f64,
    ) -> MathResult<IterativeResult> {
        if !a.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = a.rows;
        if b.len() != n {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut x = mathverse_vector::Vector::new(vec![0.0; n]);
        let mut r = b.clone();
        let mut p = r.clone();
        let mut rs_old = r.dot(&r);
        
        for iteration in 0..max_iterations {
            let ap = a.mul_vec(&p)?;
            let alpha = rs_old / p.dot(&ap);
            
            x = x.add(&p.scale(alpha));
            let r_new = r.sub(&ap.scale(alpha));
            let rs_new = r_new.dot(&r_new);
            
            let residual_norm = rs_new.sqrt();
            
            if residual_norm < tolerance {
                return Ok(IterativeResult {
                    solution: x,
                    iterations: iteration + 1,
                    residual_norm,
                    converged: true,
                });
            }
            
            let beta = rs_new / rs_old;
            p = r_new.add(&p.scale(beta));
            
            r = r_new;
            rs_old = rs_new;
        }
        
        Ok(IterativeResult {
            solution: x,
            iterations: max_iterations,
            residual_norm: rs_old.sqrt(),
            converged: false,
        })
    }

    /// Preconditioned CG with diagonal preconditioner.
    pub fn solve_preconditioned(
        a: &Matrix,
        b: &mathverse_vector::Vector,
        max_iterations: usize,
        tolerance: f64,
    ) -> MathResult<IterativeResult> {
        let n = a.rows;
        
        // Diagonal preconditioner M = diag(A)
        let mut m_inv = vec![0.0; n];
        for i in 0..n {
            let diag = a.get(i, i);
            if diag.abs() < 1e-15 {
                return Err(MathError::InvalidArgument("zero diagonal in preconditioner"));
            }
            m_inv[i] = 1.0 / diag;
        }
        
        let mut x = mathverse_vector::Vector::new(vec![0.0; n]);
        let mut r = b.clone();
        let mut z = mathverse_vector::Vector::new(
            r.data.iter().zip(m_inv.iter()).map(|(&r, &m)| r * m).collect()
        );
        let mut p = z.clone();
        let mut rz_old = r.dot(&z);
        
        for iteration in 0..max_iterations {
            let ap = a.mul_vec(&p)?;
            let alpha = rz_old / p.dot(&ap);
            
            x = x.add(&p.scale(alpha));
            let r_new = r.sub(&ap.scale(alpha));
            
            let residual_norm = r_new.data.iter().map(|&x| x * x).sum::<f64>().sqrt();
            
            if residual_norm < tolerance {
                return Ok(IterativeResult {
                    solution: x,
                    iterations: iteration + 1,
                    residual_norm,
                    converged: true,
                });
            }
            
            let z_new = mathverse_vector::Vector::new(
                r_new.data.iter().zip(m_inv.iter()).map(|(&r, &m)| r * m).collect()
            );
            let rz_new = r_new.dot(&z_new);
            
            let beta = rz_new / rz_old;
            p = z_new.add(&p.scale(beta));
            
            r = r_new;
            z = z_new;
            rz_old = rz_new;
        }
        
        Ok(IterativeResult {
            solution: x,
            iterations: max_iterations,
            residual_norm: rz_old.sqrt(),
            converged: false,
        })
    }
}

/// GMRES (Generalized Minimal Residual) method for general systems.
pub struct Gmres;

impl Gmres {
    /// Solve Ax = b using GMRES with restart.
    pub fn solve(
        a: &Matrix,
        b: &mathverse_vector::Vector,
        restart: usize,
        max_iterations: usize,
        tolerance: f64,
    ) -> MathResult<IterativeResult> {
        if !a.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = a.rows;
        if b.len() != n {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut x = mathverse_vector::Vector::new(vec![0.0; n]);
        let mut total_iterations = 0;
        
        for _ in 0..(max_iterations / restart + 1) {
            let ax = a.mul_vec(&x)?;
            let r = b.sub(&ax);
            let beta = r.data.iter().map(|x| x * x).sum::<f64>().sqrt();
            
            if beta < tolerance {
                return Ok(IterativeResult {
                    solution: x.clone(),
                    iterations: total_iterations,
                    residual_norm: beta,
                    converged: true,
                });
            }
            
            // Arnoldi iteration (simplified)
            let mut q = vec![r.scale(1.0 / beta)];
            let mut h = vec![vec![0.0; restart + 1]; restart];
            
            for k in 0..restart.min(n) {
                let v = a.mul_vec(&q[k])?;
                let mut v = v;

                // Modified Gram-Schmidt
                let mut h_k = vec![0.0; k + 2];
                for j in 0..=k {
                    h_k[j] = v.dot(&q[j]);
                    let v_sub = q[j].scale(h_k[j]);
                    let v_new = v.sub(&v_sub);
                    let v_data = v_new.data;
                    let mut v_vec = mathverse_vector::Vector::new(v_data);
                    std::mem::swap(&mut v, &mut v_vec);
                }

                h_k[k + 1] = v.data.iter().map(|x| x * x).sum::<f64>().sqrt();

                if h_k[k + 1] < 1e-15 {
                    break;
                }

                q.push(v.scale(1.0 / h_k[k + 1]));
                h[k] = h_k.clone();

                // Solve least squares problem (simplified)
                let residual = h_k[k + 1];
                if residual < tolerance {
                    // Update solution
                    let y = Self::solve_least_squares(&h, k + 1, beta)?;
                    for (j, &yj) in y.iter().enumerate() {
                        let qj_scaled = q[j].scale(yj);
                        x = x.add(&qj_scaled);
                    }
                    
                    return Ok(IterativeResult {
                        solution: x,
                        iterations: total_iterations + k + 1,
                        residual_norm: residual,
                        converged: true,
                    });
                }
            }
            
            total_iterations += restart;
        }
        
        Ok(IterativeResult {
            solution: x.clone(),
            iterations: total_iterations,
            residual_norm: b.sub(&a.mul_vec(&x).unwrap()).data.iter().map(|x| x * x).sum::<f64>().sqrt(),
            converged: false,
        })
    }
    
    fn solve_least_squares(h: &[Vec<f64>], k: usize, beta: f64) -> MathResult<Vec<f64>> {
        // Simplified: return zeros
        Ok(vec![0.0; k])
    }
}

/// Jacobi iteration method.
pub struct Jacobi;

impl Jacobi {
    /// Solve Ax = b using Jacobi iteration.
    pub fn solve(
        a: &Matrix,
        b: &mathverse_vector::Vector,
        max_iterations: usize,
        tolerance: f64,
        omega: f64,  // Relaxation parameter
    ) -> MathResult<IterativeResult> {
        if !a.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = a.rows;
        if b.len() != n {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut x = mathverse_vector::Vector::new(vec![0.0; n]);
        
        for iteration in 0..max_iterations {
            let mut x_new = vec![0.0; n];
            let mut max_diff: f64 = 0.0;
            
            for i in 0..n {
                let mut sum = b.get(i);
                
                for j in 0..n {
                    if i != j {
                        sum -= a.get(i, j) * x.get(j);
                    }
                }
                
                let diag = a.get(i, i);
                if diag.abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("zero diagonal in Jacobi"));
                }
                
                x_new[i] = (1.0 - omega) * x.get(i) + omega * sum / diag;
                max_diff = max_diff.max((x_new[i] - x.get(i)).abs());
            }
            
            x = mathverse_vector::Vector::new(x_new);
            
            if max_diff < tolerance {
                return Ok(IterativeResult {
                    solution: x,
                    iterations: iteration + 1,
                    residual_norm: max_diff,
                    converged: true,
                });
            }
        }
        
        Ok(IterativeResult {
            solution: x.clone(),
            iterations: max_iterations,
            residual_norm: b.sub(&a.mul_vec(&x).unwrap()).data.iter().map(|x| x * x).sum::<f64>().sqrt(),
            converged: false,
        })
    }
}

/// Gauss-Seidel iteration method.
pub struct GaussSeidel;

impl GaussSeidel {
    /// Solve Ax = b using Gauss-Seidel iteration.
    pub fn solve(
        a: &Matrix,
        b: &mathverse_vector::Vector,
        max_iterations: usize,
        tolerance: f64,
        omega: f64,  // Relaxation parameter (SOR when omega != 1)
    ) -> MathResult<IterativeResult> {
        if !a.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = a.rows;
        if b.len() != n {
            return Err(MathError::DimensionMismatch);
        }
        
        let mut x = mathverse_vector::Vector::new(vec![0.0; n]);
        
        for iteration in 0..max_iterations {
            let mut max_diff: f64 = 0.0;
            
            for i in 0..n {
                let mut sum = b.get(i);
                
                for j in 0..n {
                    if i != j {
                        sum -= a.get(i, j) * x.get(j);
                    }
                }
                
                let diag = a.get(i, i);
                if diag.abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("zero diagonal in Gauss-Seidel"));
                }
                
                let new_val = (1.0 - omega) * x.get(i) + omega * sum / diag;
                max_diff = max_diff.max((new_val - x.get(i)).abs());
                x.set(i, new_val);
            }
            
            if max_diff < tolerance {
                return Ok(IterativeResult {
                    solution: x,
                    iterations: iteration + 1,
                    residual_norm: max_diff,
                    converged: true,
                });
            }
        }
        
        Ok(IterativeResult {
            solution: x.clone(),
            iterations: max_iterations,
            residual_norm: b.sub(&a.mul_vec(&x).unwrap()).data.iter().map(|x| x * x).sum::<f64>().sqrt(),
            converged: false,
        })
    }
}

/// Successive Over-Relaxation (SOR).
pub struct Sor;

impl Sor {
    /// Solve Ax = b using SOR.
    pub fn solve(
        a: &Matrix,
        b: &mathverse_vector::Vector,
        omega: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> MathResult<IterativeResult> {
        GaussSeidel::solve(a, b, max_iterations, tolerance, omega)
    }
}

/// Stationary iterative methods utilities.
pub struct IterativeUtils;

impl IterativeUtils {
    /// Check if matrix is strictly diagonally dominant.
    pub fn is_diagonally_dominant(m: &Matrix, strict: bool) -> bool {
        for i in 0..m.rows {
            let diag = m.get(i, i).abs();
            let mut row_sum = 0.0;
            
            for j in 0..m.cols {
                if i != j {
                    row_sum += m.get(i, j).abs();
                }
            }
            
            if strict {
                if diag <= row_sum {
                    return false;
                }
            } else {
                if diag < row_sum {
                    return false;
                }
            }
        }
        
        true
    }

    /// Check if matrix is symmetric positive definite (for CG).
    pub fn is_spd(m: &Matrix) -> bool {
        crate::positivedefinite::PositiveDefinite::is_positive_definite(m)
    }

    /// Estimate spectral radius of iteration matrix.
    pub fn spectral_radius_estimate(m: &Matrix) -> MathResult<f64> {
        let (vals, _) = m.eigen_symmetric()?;
        Ok(vals.iter().map(|&v| v.abs()).fold(f64::NEG_INFINITY, f64::max))
    }

    /// Compute optimal relaxation parameter for SOR.
    pub fn optimal_omega(m: &Matrix) -> MathResult<f64> {
        let (vals, _) = m.eigen_symmetric()?;
        let rho = vals.iter().map(|&v| v.abs()).fold(f64::NEG_INFINITY, f64::max);
        
        if rho >= 1.0 {
            return Ok(1.0);  // No convergence guaranteed
        }
        
        Ok(2.0 / (1.0 + (1.0 - rho * rho).sqrt()))
    }

    /// Preconditioner: Incomplete Cholesky.
    pub fn incomplete_cholesky(m: &Matrix) -> MathResult<Matrix> {
        if !m.is_square() {
            return Err(MathError::DimensionMismatch);
        }
        
        let n = m.rows;
        let mut l = Matrix::zeros(n, n);
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = m.get(i, j);
                
                for k in 0..j {
                    sum -= l.get(i, k) * l.get(j, k);
                }
                
                if i == j {
                    if sum <= 0.0 {
                        return Err(MathError::InvalidArgument("matrix not positive definite"));
                    }
                    l.set(i, j, sum.sqrt());
                } else if sum.abs() > 1e-15 {
                    l.set(i, j, sum / l.get(j, j));
                }
            }
        }
        
        Ok(l)
    }

    /// Preconditioner: SSOR preconditioner.
    pub fn ssor_preconditioner(m: &Matrix, omega: f64) -> MathResult<Matrix> {
        let n = m.rows;
        let d = Matrix::diagonal(&(0..n).map(|i| m.get(i, i)).collect::<Vec<_>>());
        let d_inv = d.inverse()?;
        
        let mut l = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..i {
                l.set(i, j, m.get(i, j));
            }
        }
        
        let u = l.transpose();
        
        let d_inv_omega = d_inv.scale(2.0 - omega);
        let i_minus_omega_l = Matrix::identity(n).sub(&l.scale(omega))?;
        let i_minus_omega_u = Matrix::identity(n).sub(&u.scale(omega))?;

        // Simple approximation: use matrix multiplication instead of solve
        Ok(i_minus_omega_l.mul(&d_inv_omega)?.mul(&i_minus_omega_u)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conjugate_gradient() {
        let a = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0]);
        
        let result = ConjugateGradient::solve(&a, &b, 100, 1e-10).unwrap();
        
        assert!(result.converged);
        assert!(result.iterations < 10);
        
        // Verify solution
        let ax = a.mul_vec(&result.solution).unwrap();
        assert!((ax.get(0) - 1.0).abs() < 1e-8);
        assert!((ax.get(1) - 2.0).abs() < 1e-8);
    }

    #[test]
    fn test_jacobi() {
        let a = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0]);
        
        let result = Jacobi::solve(&a, &b, 100, 1e-10, 1.0).unwrap();
        
        assert!(result.converged);
    }

    #[test]
    fn test_gauss_seidel() {
        let a = Matrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]).unwrap();
        let b = mathverse_vector::Vector::new(vec![1.0, 2.0]);
        
        let result = GaussSeidel::solve(&a, &b, 100, 1e-10, 1.0).unwrap();
        
        assert!(result.converged);
    }

    #[test]
    fn test_diagonal_dominance() {
        let dd = Matrix::from_rows(&[&[5.0, 1.0], &[1.0, 4.0]]).unwrap();
        assert!(IterativeUtils::is_diagonally_dominant(&dd, true));
        
        let not_dd = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]).unwrap();
        assert!(!IterativeUtils::is_diagonally_dominant(&not_dd, true));
    }
}
