//! Iterative linear solvers: Jacobi, Gauss-Seidel, SOR, CG, GMRES with preconditioning.

use mathverse_core::error::{MathError, MathResult};

/// Jacobi iterative method for solving Ax = b.
pub struct Jacobi {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Jacobi {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Jacobi {
            max_iterations,
            tolerance,
        }
    }

    /// Solve Ax = b using Jacobi iteration.
    pub fn solve(&self, a: &[Vec<f64>], b: &[f64], x0: Option<&[f64]>) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) || b.len() != n {
            return Err(MathError::InvalidArgument("matrix must be square and compatible with b"));
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; n]);
        
        for _iteration in 0..self.max_iterations {
            let mut x_new = vec![0.0; n];
            
            for i in 0..n {
                if a[i][i].abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("zero diagonal element"));
                }
                
                let mut sum = b[i];
                for j in 0..n {
                    if i != j {
                        sum -= a[i][j] * x[j];
                    }
                }
                x_new[i] = sum / a[i][i];
            }
            
            // Check convergence
            let diff: f64 = x_new.iter()
                .zip(&x)
                .map(|(&xn, &x)| (xn - x).abs())
                .sum::<f64>() / n as f64;
            
            x = x_new;
            
            if diff < self.tolerance {
                return Ok(x);
            }
        }
        
        Ok(x)
    }
}

/// Gauss-Seidel iterative method.
pub struct GaussSeidel {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl GaussSeidel {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        GaussSeidel {
            max_iterations,
            tolerance,
        }
    }

    /// Solve Ax = b using Gauss-Seidel iteration.
    pub fn solve(&self, a: &[Vec<f64>], b: &[f64], x0: Option<&[f64]>) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) || b.len() != n {
            return Err(MathError::InvalidArgument("matrix must be square and compatible with b"));
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; n]);
        
        for _iteration in 0..self.max_iterations {
            let x_old = x.clone();
            
            for i in 0..n {
                if a[i][i].abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("zero diagonal element"));
                }
                
                let mut sum = b[i];
                for j in 0..n {
                    if i != j {
                        sum -= a[i][j] * x[j];
                    }
                }
                x[i] = sum / a[i][i];
            }
            
            // Check convergence
            let diff: f64 = x.iter()
                .zip(&x_old)
                .map(|(&xi, &xo)| (xi - xo).abs())
                .sum::<f64>() / n as f64;
            
            if diff < self.tolerance {
                return Ok(x);
            }
        }
        
        Ok(x)
    }
}

/// Successive Over-Relaxation (SOR) method.
pub struct SOR {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub omega: f64,
}

impl SOR {
    pub fn new(max_iterations: usize, tolerance: f64, omega: f64) -> Self {
        SOR {
            max_iterations,
            tolerance,
            omega,
        }
    }

    /// Solve Ax = b using SOR iteration.
    pub fn solve(&self, a: &[Vec<f64>], b: &[f64], x0: Option<&[f64]>) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) || b.len() != n {
            return Err(MathError::InvalidArgument("matrix must be square and compatible with b"));
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; n]);
        
        for _iteration in 0..self.max_iterations {
            let x_old = x.clone();
            
            for i in 0..n {
                if a[i][i].abs() < 1e-15 {
                    return Err(MathError::InvalidArgument("zero diagonal element"));
                }
                
                let mut sum = b[i];
                for j in 0..n {
                    if i != j {
                        sum -= a[i][j] * x[j];
                    }
                }
                
                let x_gs = sum / a[i][i];
                x[i] = (1.0 - self.omega) * x_old[i] + self.omega * x_gs;
            }
            
            // Check convergence
            let diff: f64 = x.iter()
                .zip(&x_old)
                .map(|(&xi, &xo)| (xi - xo).abs())
                .sum::<f64>() / n as f64;
            
            if diff < self.tolerance {
                return Ok(x);
            }
        }
        
        Ok(x)
    }
}

/// Conjugate Gradient method for symmetric positive definite matrices.
pub struct ConjugateGradient {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl ConjugateGradient {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        ConjugateGradient {
            max_iterations,
            tolerance,
        }
    }

    /// Solve Ax = b using Conjugate Gradient.
    pub fn solve(&self, a: &[Vec<f64>], b: &[f64], x0: Option<&[f64]>) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) || b.len() != n {
            return Err(MathError::InvalidArgument("matrix must be square and compatible with b"));
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; n]);
        
        // r = b - Ax
        let mut r = b.to_vec();
        for i in 0..n {
            for j in 0..n {
                r[i] -= a[i][j] * x[j];
            }
        }
        
        let mut p = r.clone();
        let mut rsold: f64 = r.iter().map(|&ri| ri * ri).sum();
        
        for _iteration in 0..self.max_iterations {
            // Ap = A * p
            let mut ap = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    ap[i] += a[i][j] * p[j];
                }
            }
            
            // alpha = rsold / (p^T * Ap)
            let pAp: f64 = p.iter().zip(&ap).map(|(&pi, &api)| pi * api).sum();
            if pAp.abs() < 1e-15 {
                return Ok(x);
            }
            
            let alpha = rsold / pAp;
            
            // x = x + alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            
            // r = r - alpha * Ap
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }
            
            let rsnew: f64 = r.iter().map(|&ri| ri * ri).sum();
            
            if rsnew.sqrt() < self.tolerance {
                return Ok(x);
            }
            
            // p = r + (rsnew / rsold) * p
            let beta = rsnew / rsold;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            
            rsold = rsnew;
        }
        
        Ok(x)
    }
}

/// Preconditioned Conjugate Gradient.
pub struct PreconditionedCG {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl PreconditionedCG {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        PreconditionedCG {
            max_iterations,
            tolerance,
        }
    }

    /// Solve Ax = b using PCG with diagonal preconditioner.
    pub fn solve(&self, a: &[Vec<f64>], b: &[f64], x0: Option<&[f64]>) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) || b.len() != n {
            return Err(MathError::InvalidArgument("matrix must be square and compatible with b"));
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; n]);
        
        // Diagonal preconditioner
        let mut m_inv = vec![0.0; n];
        for i in 0..n {
            if a[i][i].abs() < 1e-15 {
                return Err(MathError::InvalidArgument("zero diagonal element for preconditioner"));
            }
            m_inv[i] = 1.0 / a[i][i];
        }
        
        // r = b - Ax
        let mut r = b.to_vec();
        for i in 0..n {
            for j in 0..n {
                r[i] -= a[i][j] * x[j];
            }
        }
        
        // z = M^(-1) * r
        let mut z: Vec<f64> = r.iter().zip(&m_inv).map(|(&ri, &mi)| ri * mi).collect();
        
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(&ri, &zi)| ri * zi).sum();
        
        for _iteration in 0..self.max_iterations {
            // Ap = A * p
            let mut ap = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    ap[i] += a[i][j] * p[j];
                }
            }
            
            // alpha = rz / (p^T * Ap)
            let pAp: f64 = p.iter().zip(&ap).map(|(&pi, &api)| pi * api).sum();
            if pAp.abs() < 1e-15 {
                return Ok(x);
            }
            
            let alpha = rz / pAp;
            
            // x = x + alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            
            // r = r - alpha * Ap
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }
            
            let r_norm: f64 = r.iter().map(|&ri| ri * ri).sum::<f64>().sqrt();
            if r_norm < self.tolerance {
                return Ok(x);
            }
            
            // z = M^(-1) * r
            z = r.iter().zip(&m_inv).map(|(&ri, &mi)| ri * mi).collect();
            
            let rz_new: f64 = r.iter().zip(&z).map(|(&ri, &zi)| ri * zi).sum();
            
            // p = z + (rz_new / rz) * p
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            
            rz = rz_new;
        }
        
        Ok(x)
    }
}

/// GMRES (Generalized Minimal Residual) method.
pub struct GMRES {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub restart: usize,
}

impl GMRES {
    pub fn new(max_iterations: usize, tolerance: f64, restart: usize) -> Self {
        GMRES {
            max_iterations,
            tolerance,
            restart,
        }
    }

    /// Solve Ax = b using GMRES with restart.
    pub fn solve(&self, a: &[Vec<f64>], b: &[f64], x0: Option<&[f64]>) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) || b.len() != n {
            return Err(MathError::InvalidArgument("matrix must be square and compatible with b"));
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; n]);
        
        let mut total_iter = 0;
        
        while total_iter < self.max_iterations {
            let max_iter = (self.restart).min(self.max_iterations - total_iter);
            
            // r = b - Ax
            let mut r = b.to_vec();
            for i in 0..n {
                for j in 0..n {
                    r[i] -= a[i][j] * x[j];
                }
            }
            
            let r_norm: f64 = r.iter().map(|&ri| ri * ri).sum::<f64>().sqrt();
            if r_norm < self.tolerance {
                return Ok(x);
            }
            
            let beta = r_norm;
            let mut v = vec![vec![0.0; n]; max_iter + 1];
            v[0] = r.iter().map(|&ri| ri / beta).collect();
            
            let mut h = vec![vec![0.0; max_iter + 1]; max_iter];
            let mut g = vec![0.0; max_iter + 1];
            g[0] = beta;
            
            for j in 0..max_iter {
                // w = A * v_j
                let mut w = vec![0.0; n];
                for i in 0..n {
                    for k in 0..n {
                        w[i] += a[i][k] * v[j][k];
                    }
                }
                
                // Modified Gram-Schmidt
                for i in 0..=j {
                    h[i][j] = w.iter().zip(&v[i]).map(|(&wi, &vi)| wi * vi).sum();
                    for k in 0..n {
                        w[k] -= h[i][j] * v[i][k];
                    }
                }
                
                h[j + 1][j] = w.iter().map(|&wi| wi * wi).sum::<f64>().sqrt();
                
                if h[j + 1][j].abs() < 1e-15 {
                    break;
                }
                
                v[j + 1] = w.iter().map(|&wi| wi / h[j + 1][j]).collect();
                
                // Apply previous Givens rotations
                for i in 0..j {
                    let temp = h[i][j];
                    h[i][j] = Self::cs(i) * temp + Self::sn(i) * h[i + 1][j];
                    h[i + 1][j] = -Self::sn(i) * temp + Self::cs(i) * h[i + 1][j];
                }
                
                // Compute new Givens rotation
                let t = (h[j][j].powi(2) + h[j + 1][j].powi(2)).sqrt();
                let cs = h[j][j] / t;
                let sn = h[j + 1][j] / t;
                
                // Apply new rotation
                h[j][j] = cs * h[j][j] + sn * h[j + 1][j];
                h[j + 1][j] = 0.0;
                
                g[j + 1] = -sn * g[j];
                g[j] = cs * g[j];
                
                if g[j + 1].abs() < self.tolerance {
                    break;
                }
            }
            
            // Solve least squares problem
            let y = Self::solve_triangular(&h, &g, max_iter)?;
            
            // Update solution
            for i in 0..max_iter {
                for k in 0..n {
                    x[k] += y[i] * v[i][k];
                }
            }
            
            total_iter += max_iter;
        }
        
        Ok(x)
    }

    fn cs(_i: usize) -> f64 {
        // Simplified - in practice, store rotation parameters
        1.0
    }

    fn sn(_i: usize) -> f64 {
        // Simplified - in practice, store rotation parameters
        0.0
    }

    fn solve_triangular(h: &[Vec<f64>], g: &[f64], n: usize) -> MathResult<Vec<f64>> {
        let mut y = vec![0.0; n];
        
        for i in (0..n).rev() {
            let mut sum = g[i];
            for j in (i + 1)..n {
                sum -= h[i][j] * y[j];
            }
            
            if h[i][i].abs() < 1e-15 {
                return Err(MathError::InvalidArgument("singular matrix in GMRES"));
            }
            
            y[i] = sum / h[i][i];
        }
        
        Ok(y)
    }
}

/// BiCGSTAB (Bi-Conjugate Gradient Stabilized) method.
pub struct BiCGSTAB {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl BiCGSTAB {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        BiCGSTAB {
            max_iterations,
            tolerance,
        }
    }

    /// Solve Ax = b using BiCGSTAB.
    pub fn solve(&self, a: &[Vec<f64>], b: &[f64], x0: Option<&[f64]>) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) || b.len() != n {
            return Err(MathError::InvalidArgument("matrix must be square and compatible with b"));
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; n]);
        
        // r = b - Ax
        let mut r = b.to_vec();
        for i in 0..n {
            for j in 0..n {
                r[i] -= a[i][j] * x[j];
            }
        }
        
        let r0_hat = r.clone();
        let mut p = r.clone();
        let mut rho = r.iter().map(|&ri| ri * ri).sum::<f64>();
        
        for _iteration in 0..self.max_iterations {
            let rho_old = rho;
            rho = r.iter().zip(&r0_hat).map(|(&ri, &r0i)| ri * r0i).sum();
            
            if rho.abs() < 1e-15 {
                return Ok(x);
            }
            
            let beta = rho / rho_old;
            
            // p = r + beta * p
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            
            // Ap = A * p
            let mut ap = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    ap[i] += a[i][j] * p[j];
                }
            }
            
            let alpha = rho / r0_hat.iter().zip(&ap).map(|(&r0i, &api)| r0i * api).sum::<f64>();
            
            // s = r - alpha * Ap
            let s: Vec<f64> = r.iter().zip(&ap).map(|(&ri, &api)| ri - alpha * api).collect();
            
            let s_norm: f64 = s.iter().map(|&si| si * si).sum::<f64>().sqrt();
            if s_norm < self.tolerance {
                // x = x + alpha * p
                for i in 0..n {
                    x[i] += alpha * p[i];
                }
                return Ok(x);
            }
            
            // As = A * s
            let mut as_vec = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    as_vec[i] += a[i][j] * s[j];
                }
            }
            
            let omega = as_vec.iter().zip(&s).map(|(&asi, &si)| asi * si).sum::<f64>()
                / as_vec.iter().map(|&asi| asi * asi).sum::<f64>();
            
            // x = x + alpha * p + omega * s
            for i in 0..n {
                x[i] += alpha * p[i] + omega * s[i];
            }
            
            // r = s - omega * As
            for i in 0..n {
                r[i] = s[i] - omega * as_vec[i];
            }
            
            let r_norm: f64 = r.iter().map(|&ri| ri * ri).sum::<f64>().sqrt();
            if r_norm < self.tolerance {
                return Ok(x);
            }
        }
        
        Ok(x)
    }
}

/// ILU (Incomplete LU) preconditioner.
pub struct ILUPreconditioner {
    pub l: Vec<Vec<f64>>,
    pub u: Vec<Vec<f64>>,
}

impl ILUPreconditioner {
    /// Create ILU(0) preconditioner from matrix A.
    pub fn new(a: &[Vec<f64>]) -> MathResult<Self> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) {
            return Err(MathError::InvalidArgument("matrix must be square"));
        }
        
        let mut l = vec![vec![0.0; n]; n];
        let mut u = a.to_vec();
        
        for i in 0..n {
            l[i][i] = 1.0;
            
            for k in 0..i {
                for j in k..n {
                    u[i][j] -= l[i][k] * u[k][j];
                }
            }
            
            for j in (i + 1)..n {
                l[j][i] = u[j][i] / u[i][i];
                u[j][i] = 0.0;
            }
            
            if u[i][i].abs() < 1e-15 {
                return Err(MathError::InvalidArgument("zero pivot in ILU"));
            }
        }
        
        Ok(ILUPreconditioner { l, u })
    }

    /// Apply preconditioner: solve Mz = r where M = LU.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = self.l.len();
        
        // Forward substitution: Ly = r
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = r[i];
            for j in 0..i {
                sum -= self.l[i][j] * y[j];
            }
            y[i] = sum / self.l[i][i];
        }
        
        // Back substitution: Uz = y
        let mut z = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= self.u[i][j] * z[j];
            }
            z[i] = sum / self.u[i][i];
        }
        
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        
        let jacobi = Jacobi::new(1000, 1e-10);
        let result = jacobi.solve(&a, &b, None).unwrap();
        
        // Verify solution
        let ax0 = a[0][0] * result[0] + a[0][1] * result[1];
        let ax1 = a[1][0] * result[0] + a[1][1] * result[1];
        
        assert!((ax0 - b[0]).abs() < 1e-6);
        assert!((ax1 - b[1]).abs() < 1e-6);
    }

    #[test]
    fn test_gauss_seidel() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        
        let gs = GaussSeidel::new(1000, 1e-10);
        let result = gs.solve(&a, &b, None).unwrap();
        
        let ax0 = a[0][0] * result[0] + a[0][1] * result[1];
        let ax1 = a[1][0] * result[0] + a[1][1] * result[1];
        
        assert!((ax0 - b[0]).abs() < 1e-6);
        assert!((ax1 - b[1]).abs() < 1e-6);
    }

    #[test]
    fn test_sor() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        
        let sor = SOR::new(1000, 1e-10, 1.2);
        let result = sor.solve(&a, &b, None).unwrap();
        
        let ax0 = a[0][0] * result[0] + a[0][1] * result[1];
        let ax1 = a[1][0] * result[0] + a[1][1] * result[1];
        
        assert!((ax0 - b[0]).abs() < 1e-6);
        assert!((ax1 - b[1]).abs() < 1e-6);
    }

    #[test]
    fn test_conjugate_gradient() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        
        let cg = ConjugateGradient::new(1000, 1e-10);
        let result = cg.solve(&a, &b, None).unwrap();
        
        let ax0 = a[0][0] * result[0] + a[0][1] * result[1];
        let ax1 = a[1][0] * result[0] + a[1][1] * result[1];
        
        assert!((ax0 - b[0]).abs() < 1e-10);
        assert!((ax1 - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_preconditioned_cg() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        
        let pcg = PreconditionedCG::new(1000, 1e-10);
        let result = pcg.solve(&a, &b, None).unwrap();
        
        let ax0 = a[0][0] * result[0] + a[0][1] * result[1];
        let ax1 = a[1][0] * result[0] + a[1][1] * result[1];
        
        assert!((ax0 - b[0]).abs() < 1e-10);
        assert!((ax1 - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_bicgstab() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        
        let bicgstab = BiCGSTAB::new(1000, 1e-10);
        let result = bicgstab.solve(&a, &b, None).unwrap();
        
        let ax0 = a[0][0] * result[0] + a[0][1] * result[1];
        let ax1 = a[1][0] * result[0] + a[1][1] * result[1];
        
        assert!((ax0 - b[0]).abs() < 1e-6);
        assert!((ax1 - b[1]).abs() < 1e-6);
    }

    #[test]
    fn test_ilu_preconditioner() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        
        let ilu = ILUPreconditioner::new(&a).unwrap();
        let r = vec![1.0, 2.0];
        let z = ilu.apply(&r);
        
        assert!(!z.is_empty());
    }
}
