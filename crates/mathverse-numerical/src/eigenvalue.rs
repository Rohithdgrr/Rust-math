//! Eigenvalue solvers: power method, inverse iteration, QR algorithm, Lanczos.

use mathverse_core::error::{MathError, MathResult};

/// Power method for finding dominant eigenvalue.
pub struct PowerMethod {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl PowerMethod {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        PowerMethod {
            max_iterations,
            tolerance,
        }
    }

    /// Find dominant eigenvalue and eigenvector.
    pub fn compute(&self, a: &[Vec<f64>], x0: Option<&[f64]>) -> MathResult<(f64, Vec<f64>)> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) {
            return Err(MathError::InvalidArgument("matrix must be square"));
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| {
            (0..n).map(|_| rand::random::<f64>()).collect()
        });
        
        // Normalize initial vector
        let norm: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        x = x.iter().map(|&xi| xi / norm).collect();
        
        let mut lambda = 0.0;
        
        for _iteration in 0..self.max_iterations {
            // y = Ax
            let mut y = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    y[i] += a[i][j] * x[j];
                }
            }
            
            // Rayleigh quotient
            let numerator: f64 = x.iter().zip(&y).map(|(&xi, &yi)| xi * yi).sum();
            let lambda_new = numerator;
            
            // Check convergence
            if (lambda_new - lambda).abs() < self.tolerance {
                lambda = lambda_new;
                break;
            }
            
            lambda = lambda_new;
            
            // Normalize y
            let norm: f64 = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return Err(MathError::NotConverged("zero norm in power method"));
            }
            x = y.iter().map(|&yi| yi / norm).collect();
        }
        
        Ok((lambda, x))
    }
}

/// Inverse power method for finding eigenvalue closest to shift.
pub struct InversePowerMethod {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub shift: f64,
}

impl InversePowerMethod {
    pub fn new(max_iterations: usize, tolerance: f64, shift: f64) -> Self {
        InversePowerMethod {
            max_iterations,
            tolerance,
            shift,
        }
    }

    /// Find eigenvalue closest to shift.
    pub fn compute(&self, a: &[Vec<f64>], x0: Option<&[f64]>) -> MathResult<(f64, Vec<f64>)> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) {
            return Err(MathError::InvalidArgument("matrix must be square"));
        }
        
        // Form A - shift*I
        let mut a_shift = a.to_vec();
        for i in 0..n {
            a_shift[i][i] -= self.shift;
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| {
            (0..n).map(|_| rand::random::<f64>()).collect()
        });
        
        let norm: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        x = x.iter().map(|&xi| xi / norm).collect();
        
        let mut lambda = 0.0;
        
        for _iteration in 0..self.max_iterations {
            // Solve (A - shift*I) * y = x using Gaussian elimination
            let y = Self::solve_linear(&a_shift, &x)?;
            
            // Rayleigh quotient for inverse
            let numerator: f64 = x.iter().zip(&y).map(|(&xi, &yi)| xi * yi).sum();
            let lambda_new = 1.0 / numerator + self.shift;
            
            if (lambda_new - lambda).abs() < self.tolerance {
                lambda = lambda_new;
                break;
            }
            
            lambda = lambda_new;
            
            let norm: f64 = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return Err(MathError::NotConverged("zero norm in inverse power method"));
            }
            x = y.iter().map(|&yi| yi / norm).collect();
        }
        
        Ok((lambda, x))
    }

    fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> MathResult<Vec<f64>> {
        let n = a.len();
        let mut a = a.to_vec();
        let mut b = b.to_vec();
        
        // Forward elimination
        for i in 0..n {
            let mut pivot = i;
            for j in (i + 1)..n {
                if a[j][i].abs() > a[pivot][i].abs() {
                    pivot = j;
                }
            }
            
            if a[pivot][i].abs() < 1e-15 {
                return Err(MathError::InvalidArgument("singular matrix"));
            }
            
            a.swap(i, pivot);
            b.swap(i, pivot);
            
            for j in (i + 1)..n {
                let factor = a[j][i] / a[i][i];
                for k in i..n {
                    a[j][k] -= factor * a[i][k];
                }
                b[j] -= factor * b[i];
            }
        }
        
        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= a[i][j] * x[j];
            }
            x[i] = sum / a[i][i];
        }
        
        Ok(x)
    }
}

/// Rayleigh quotient iteration (accelerated inverse iteration).
pub struct RayleighQuotientIteration {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl RayleighQuotientIteration {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        RayleighQuotientIteration {
            max_iterations,
            tolerance,
        }
    }

    /// Find eigenvalue using Rayleigh quotient iteration.
    pub fn compute(&self, a: &[Vec<f64>], x0: Option<&[f64]>) -> MathResult<(f64, Vec<f64>)> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) {
            return Err(MathError::InvalidArgument("matrix must be square"));
        }
        
        let mut x = x0.map(|v| v.to_vec()).unwrap_or_else(|| {
            (0..n).map(|_| rand::random::<f64>()).collect()
        });
        
        let norm: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        x = x.iter().map(|&xi| xi / norm).collect();
        
        let mut lambda = 0.0;
        
        for _iteration in 0..self.max_iterations {
            // Rayleigh quotient
            let mut ax = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    ax[i] += a[i][j] * x[j];
                }
            }
            
            let numerator: f64 = x.iter().zip(&ax).map(|(&xi, &axi)| xi * axi).sum();
            let lambda_new = numerator;
            
            if (lambda_new - lambda).abs() < self.tolerance {
                lambda = lambda_new;
                break;
            }
            
            lambda = lambda_new;
            
            // Form A - lambda*I
            let mut a_shift = a.to_vec();
            for i in 0..n {
                a_shift[i][i] -= lambda;
            }
            
            // Solve (A - lambda*I) * y = x. A singular shifted matrix means
            // lambda has converged to an eigenvalue, so treat that as success.
            let y = match InversePowerMethod::solve_linear(&a_shift, &x) {
                Ok(y) => y,
                Err(_) => break,
            };
            
            let norm: f64 = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return Err(MathError::NotConverged("zero norm in RQI"));
            }
            x = y.iter().map(|&yi| yi / norm).collect();
        }
        
        Ok((lambda, x))
    }
}

/// QR algorithm for computing all eigenvalues.
pub struct QRAlgorithm {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl QRAlgorithm {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        QRAlgorithm {
            max_iterations,
            tolerance,
        }
    }

    /// Compute all eigenvalues using QR algorithm.
    pub fn compute(&self, a: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) {
            return Err(MathError::InvalidArgument("matrix must be square"));
        }
        
        let mut a = a.to_vec();
        
        for _iteration in 0..self.max_iterations {
            // Check for convergence (upper triangular)
            if Self::is_upper_triangular(&a, self.tolerance) {
                let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
                return Ok(eigenvalues);
            }
            
            // QR decomposition using Gram-Schmidt
            let (q, r) = Self::qr_decomposition(&a)?;
            
            // A = R * Q
            let mut a_new = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        a_new[i][j] += r[i][k] * q[k][j];
                    }
                }
            }
            
            a = a_new;
        }
        
        // Return diagonal as eigenvalues
        let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        Ok(eigenvalues)
    }

    fn qr_decomposition(a: &[Vec<f64>]) -> MathResult<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let n = a.len();
        let mut q = vec![vec![0.0; n]; n];
        let mut r = vec![vec![0.0; n]; n];
        
        for j in 0..n {
            // Copy column j of A
            let mut col = vec![0.0; n];
            for i in 0..n {
                col[i] = a[i][j];
            }
            
            // Subtract projections onto previous Q columns
            for k in 0..j {
                let r_kj: f64 = (0..n).map(|i| q[i][k] * col[i]).sum();
                r[k][j] = r_kj;
                
                for i in 0..n {
                    col[i] -= r_kj * q[i][k];
                }
            }
            
            // Normalize
            let norm: f64 = col.iter().map(|&c| c * c).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return Err(MathError::InvalidArgument("zero norm in QR decomposition"));
            }
            
            r[j][j] = norm;
            
            for i in 0..n {
                q[i][j] = col[i] / norm;
            }
        }
        
        Ok((q, r))
    }

    fn is_upper_triangular(a: &[Vec<f64>], tolerance: f64) -> bool {
        for i in 1..a.len() {
            for j in 0..i {
                if a[i][j].abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

/// Lanczos algorithm for symmetric matrices.
pub struct Lanczos {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Lanczos {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Lanczos {
            max_iterations,
            tolerance,
        }
    }

    /// Compute eigenvalues using Lanczos algorithm.
    pub fn compute(&self, a: &[Vec<f64>], num_eigenvalues: usize) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) {
            return Err(MathError::InvalidArgument("matrix must be square"));
        }
        
        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                if (a[i][j] - a[j][i]).abs() > 1e-10 {
                    return Err(MathError::InvalidArgument("matrix must be symmetric"));
                }
            }
        }
        
        let m = num_eigenvalues.min(n);
        
        // Lanczos iteration
        let mut q = vec![vec![0.0; n]; m + 1];
        let mut alpha = vec![0.0; m];
        let mut beta = vec![0.0; m];
        
        // Initial vector
        let norm: f64 = (0..n).map(|_| rand::random::<f64>().powi(2)).sum::<f64>().sqrt();
        for i in 0..n {
            q[0][i] = rand::random::<f64>() / norm;
        }
        
        let mut r = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                r[i] += a[i][j] * q[0][j];
            }
        }
        
        alpha[0] = (0..n).map(|i| q[0][i] * r[i]).sum();
        
        for i in 0..n {
            r[i] -= alpha[0] * q[0][i];
        }
        
        beta[0] = (0..n).map(|i| r[i] * r[i]).sum::<f64>().sqrt();
        
        for j in 1..m {
            if beta[j - 1].abs() < 1e-15 {
                break;
            }
            
            for i in 0..n {
                q[j][i] = r[i] / beta[j - 1];
            }
            
            // r = A * q_j
            for i in 0..n {
                r[i] = 0.0;
                for k in 0..n {
                    r[i] += a[i][k] * q[j][k];
                }
            }
            
            alpha[j] = (0..n).map(|i| q[j][i] * r[i]).sum();
            
            for i in 0..n {
                r[i] -= alpha[j] * q[j][i] + beta[j - 1] * q[j - 1][i];
            }
            
            beta[j] = (0..n).map(|i| r[i] * r[i]).sum::<f64>().sqrt();
        }
        
        // Solve tridiagonal eigenvalue problem (simplified)
        Self::tridiagonal_eigenvalues(&alpha, &beta)
    }

    fn tridiagonal_eigenvalues(alpha: &[f64], beta: &[f64]) -> MathResult<Vec<f64>> {
        let n = alpha.len();
        if n == 0 {
            return Ok(vec![]);
        }
        
        // Use QR algorithm on tridiagonal matrix
        let mut t = vec![vec![0.0; n]; n];
        for i in 0..n {
            t[i][i] = alpha[i];
            if i > 0 {
                t[i][i - 1] = beta[i - 1];
                t[i - 1][i] = beta[i - 1];
            }
        }
        
        let qr = QRAlgorithm::new(1000, 1e-10);
        qr.compute(&t)
    }
}

/// Subspace iteration for computing multiple eigenvalues.
pub struct SubspaceIteration {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub subspace_size: usize,
}

impl SubspaceIteration {
    pub fn new(max_iterations: usize, tolerance: f64, subspace_size: usize) -> Self {
        SubspaceIteration {
            max_iterations,
            tolerance,
            subspace_size,
        }
    }

    /// Compute multiple eigenvalues using subspace iteration.
    pub fn compute(&self, a: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) {
            return Err(MathError::InvalidArgument("matrix must be square"));
        }
        
        let m = self.subspace_size.min(n);
        
        // Initialize random subspace
        let mut q = vec![vec![0.0; n]; m];
        for i in 0..m {
            let norm: f64 = (0..n).map(|_| rand::random::<f64>().powi(2)).sum::<f64>().sqrt();
            for j in 0..n {
                q[i][j] = rand::random::<f64>() / norm;
            }
        }
        
        for _iteration in 0..self.max_iterations {
            // Z = A * Q
            let mut z = vec![vec![0.0; n]; m];
            for i in 0..m {
                for j in 0..n {
                    for k in 0..n {
                        z[i][j] += a[j][k] * q[i][k];
                    }
                }
            }
            
            // QR decomposition of Z
            let (q_new, _) = QRAlgorithm::qr_decomposition(&Self::transpose(&z))?;
            
            // Check convergence
            let diff: f64 = q.iter()
                .zip(&q_new)
                .map(|(qi, qni)| {
                    qi.iter().zip(qni).map(|(&qij, &qnij)| (qij - qnij).powi(2)).sum::<f64>()
                })
                .sum::<f64>()
                .sqrt();
            
            q = Self::transpose(&q_new);
            
            if diff < self.tolerance {
                break;
            }
        }
        
        // Project A onto subspace: A_proj = Q^T * A * Q
        let mut a_proj = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                for k in 0..n {
                    for l in 0..n {
                        a_proj[i][j] += q[k][i] * a[k][l] * q[l][j];
                    }
                }
            }
        }
        
        // Compute eigenvalues of projected matrix
        let qr = QRAlgorithm::new(1000, 1e-10);
        qr.compute(&a_proj)
    }

    fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let rows = m.len();
        let cols = m[0].len();
        let mut result = vec![vec![0.0; rows]; cols];
        
        for i in 0..rows {
            for j in 0..cols {
                result[j][i] = m[i][j];
            }
        }
        
        result
    }
}

/// Jacobi method for symmetric eigenvalue problems.
pub struct JacobiEigenvalue {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl JacobiEigenvalue {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        JacobiEigenvalue {
            max_iterations,
            tolerance,
        }
    }

    /// Compute eigenvalues using Jacobi method.
    pub fn compute(&self, a: &[Vec<f64>]) -> MathResult<Vec<f64>> {
        let n = a.len();
        if a.iter().any(|row| row.len() != n) {
            return Err(MathError::InvalidArgument("matrix must be square"));
        }
        
        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                if (a[i][j] - a[j][i]).abs() > 1e-10 {
                    return Err(MathError::InvalidArgument("matrix must be symmetric"));
                }
            }
        }
        
        let mut a = a.to_vec();
        
        for _iteration in 0..self.max_iterations {
            // Find largest off-diagonal element
            let mut max_val = 0.0;
            let (mut p, mut q) = (0, 0);
            
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            
            if max_val < self.tolerance {
                break;
            }
            
            // Compute rotation angle
            let theta = if a[p][p] == a[q][q] {
                core::f64::consts::PI / 4.0
            } else {
                0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
            };
            
            let c = theta.cos();
            let s = theta.sin();
            
            // Apply rotation
            for i in 0..n {
                if i != p && i != q {
                    let a_ip = a[i][p];
                    let a_iq = a[i][q];
                    a[i][p] = c * a_ip - s * a_iq;
                    a[p][i] = a[i][p];
                    a[i][q] = s * a_ip + c * a_iq;
                    a[q][i] = a[i][q];
                }
            }
            
            let a_pp = a[p][p];
            let a_qq = a[q][q];
            let a_pq = a[p][q];
            
            a[p][p] = c * c * a_pp + s * s * a_qq - 2.0 * s * c * a_pq;
            a[q][q] = s * s * a_pp + c * c * a_qq + 2.0 * s * c * a_pq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
        }
        
        let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        Ok(eigenvalues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_method() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        
        let pm = PowerMethod::new(1000, 1e-10);
        let (lambda, _) = pm.compute(&a, None).unwrap();
        
        // Dominant eigenvalue should be 3
        assert!((lambda - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_inverse_power_method() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        
        let ipm = InversePowerMethod::new(1000, 1e-10, 1.5);
        let (lambda, _) = ipm.compute(&a, None).unwrap();
        
        // Should find eigenvalue closest to 1.5 (which is 1)
        assert!((lambda - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_rayleigh_quotient_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        
        let rqi = RayleighQuotientIteration::new(100, 1e-10);
        let (lambda, _) = rqi.compute(&a, None).unwrap();
        
        // Should converge to an eigenvalue
        assert!(lambda > 0.0);
    }

    #[test]
    fn test_qr_algorithm() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        
        let qr = QRAlgorithm::new(1000, 1e-10);
        let mut eigenvalues = qr.compute(&a).unwrap();
        
        // Eigenvalues should be 3 and 1
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigenvalues[0] - 1.0).abs() < 1e-6);
        assert!((eigenvalues[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_lanczos() {
        let a = vec![vec![2.0, 1.0, 0.0], vec![1.0, 2.0, 1.0], vec![0.0, 1.0, 2.0]];
        
        let lanczos = Lanczos::new(100, 1e-10);
        let eigenvalues = lanczos.compute(&a, 2).unwrap();
        
        assert_eq!(eigenvalues.len(), 2);
    }

    #[test]
    fn test_jacobi_eigenvalue() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        
        let jacobi = JacobiEigenvalue::new(1000, 1e-10);
        let mut eigenvalues = jacobi.compute(&a).unwrap();
        
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigenvalues[0] - 1.0).abs() < 1e-6);
        assert!((eigenvalues[1] - 3.0).abs() < 1e-6);
    }
}
