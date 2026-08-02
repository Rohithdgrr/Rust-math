pub fn lu_decompose(a: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<usize>)> {
    let n = a.len();
    if n == 0 || a.iter().any(|r| r.len() != n) { return None; }
    
    let mut l = vec![vec![0.0; n]; n];
    let mut u = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    
    for i in 0..n {
        // Partial pivoting: find the row with maximum absolute value in column i
        let mut max_row = i;
        let mut max_val = u[i][i].abs();
        for k in (i+1)..n {
            if u[k][i].abs() > max_val {
                max_val = u[k][i].abs();
                max_row = k;
            }
        }
        
        // Swap rows if necessary
        if max_row != i {
            u.swap(i, max_row);
            perm.swap(i, max_row);
            // Swap the already-computed part of L (only columns 0..i-1 are valid)
            for k in 0..i {
                let temp = l[i][k];
                l[i][k] = l[max_row][k];
                l[max_row][k] = temp;
            }
        }
        
        // Check for singularity
        if u[i][i].abs() < 1e-15 { return None; }
        
        // Compute U's row i and L's column i
        for k in i..n {
            let mut sum = 0.0;
            for j in 0..i { sum += l[i][j] * u[j][k]; }
            u[i][k] = u[i][k] - sum;
        }
        
        l[i][i] = 1.0;
        for k in i+1..n {
            let mut sum = 0.0;
            for j in 0..i { sum += l[k][j] * u[j][i]; }
            l[k][i] = (u[k][i] - sum) / u[i][i];
        }
    }
    
    Some((l, u, perm))
}

pub fn qr_decompose(a: &[Vec<f64>]) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let (m, n) = (a.len(), a[0].len());
    let mut q = vec![vec![0.0; n]; m]; // Q is m×n with orthonormal columns
    let mut r = vec![vec![0.0; n]; n]; // R is n×n upper triangular
    
    for j in 0..n {
        // Start with the j-th column of A
        let mut v: Vec<f64> = (0..m).map(|i| a[i][j]).collect();
        
        // Modified Gram-Schmidt: orthogonalize against each previous q vector
        for i in 0..j {
            let dot: f64 = (0..m).map(|k| q[k][i] * v[k]).sum();
            r[i][j] = dot;
            // Subtract projection immediately (this is the key difference from classical GS)
            for k in 0..m { 
                v[k] -= dot * q[k][i]; 
            }
        }
        
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 { return None; }
        r[j][j] = norm;
        
        for k in 0..m { 
            q[k][j] = v[k] / norm; 
        }
    }
    
    Some((q, r))
}

pub fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..i { sum += l[i][k] * l[i][k]; }
        let diag = a[i][i] - sum;
        if diag <= 1e-15 { return None; }
        l[i][i] = diag.sqrt();
        for j in i+1..n {
            let mut sum = 0.0;
            for k in 0..i { sum += l[j][k] * l[i][k]; }
            l[j][i] = (a[j][i] - sum) / l[i][i];
        }
    }
    Some(l)
}

pub fn solve_lu(l: &[Vec<f64>], u: &[Vec<f64>], perm: &[usize], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    
    // Apply permutation to b
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[perm[i]];
    }
    
    // Forward substitution (solve Ly = Pb)
    let mut y = vec![0.0; n];
    for i in 0..n { 
        y[i] = pb[i] - (0..i).map(|j| l[i][j] * y[j]).sum::<f64>(); 
    }
    
    // Backward substitution (solve Ux = y)
    let mut x = vec![0.0; n];
    for i in (0..n).rev() { 
        x[i] = (y[i] - (i+1..n).map(|j| u[i][j] * x[j]).sum::<f64>()) / u[i][i]; 
    }
    x
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

pub fn eigenvalue_2x2(a: [[f64; 2]; 2]) -> (Complex, Complex) {
    let trace = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = trace * trace - 4.0 * det;
    
    if disc >= 0.0 {
        // Real eigenvalues
        let sqrt_disc = disc.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        (Complex { re: lambda1, im: 0.0 }, Complex { re: lambda2, im: 0.0 })
    } else {
        // Complex eigenvalues
        let sqrt_disc_abs = (-disc).sqrt();
        let real_part = trace / 2.0;
        let imag_part = sqrt_disc_abs / 2.0;
        (
            Complex { re: real_part, im: imag_part },
            Complex { re: real_part, im: -imag_part }
        )
    }
}

pub fn power_iteration(a: &[Vec<f64>], max_iter: usize, tol: f64) -> Option<(Vec<f64>, f64)> {
    let n = a.len();
    if n == 0 { return None; }
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda = 0.0;
    for _ in 0..max_iter {
        let mut w = vec![0.0; n];
        for i in 0..n { for j in 0..n { w[i] += a[i][j] * v[j]; } }
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 { break; }
        for i in 0..n { v[i] = w[i] / norm; }
        let mut new_lambda = 0.0;
        for i in 0..n { new_lambda += v[i] * (0..n).map(|j| a[i][j] * v[j]).sum::<f64>(); }
        if (new_lambda - lambda).abs() < tol { return Some((v, new_lambda)); }
        lambda = new_lambda;
    }
    Some((v, lambda))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_test() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let (l, u, perm) = lu_decompose(&a).unwrap();
        let x = solve_lu(&l, &u, &perm, &[5.0, 7.0]);
        assert!((x[0] - 1.6).abs() < 1e-10);
    }

    #[test]
    fn lu_pivoting_test() {
        // Test matrix that requires pivoting: [[0,1],[1,0]]
        let a = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let (l, u, perm) = lu_decompose(&a).unwrap();
        let x = solve_lu(&l, &u, &perm, &[1.0, 0.0]);
        // Solution should be [0, 1]
        assert!((x[0] - 0.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn lu_ill_conditioned() {
        // Test with a matrix that has small but non-zero pivots
        let a = vec![vec![1e-10, 1.0], vec![1.0, 1.0]];
        let (l, u, perm) = lu_decompose(&a).unwrap();
        let x = solve_lu(&l, &u, &perm, &[1.0, 2.0]);
        // Solution should exist and be reasonable
        assert!(x[0].is_finite());
        assert!(x[1].is_finite());
    }

    #[test]
    fn qr_test() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (q, _r) = qr_decompose(&a).unwrap();
        assert!((q[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn qr_orthogonality_test() {
        // Test that Q^T Q is close to identity
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let (q, _) = qr_decompose(&a).unwrap();
        
        let (m, n) = (q.len(), q[0].len());
        // Compute Q^T Q
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..m).map(|k| q[k][i] * q[k][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-10, "Q^T Q[{},{}] = {}, expected {}", i, j, dot, expected);
            }
        }
    }

    #[test]
    fn qr_ill_conditioned() {
        // Test QR with a nearly rank-deficient matrix
        let a = vec![vec![1.0, 2.0], vec![1.0, 2.0 + 1e-8], vec![1.0, 2.0 + 2e-8]];
        let (q, _r) = qr_decompose(&a).unwrap();
        
        // Check that Q is still orthogonal
        let (m, n) = (q.len(), q[0].len());
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..m).map(|k| q[k][i] * q[k][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-6, "Q^T Q[{},{}] = {}, expected {}", i, j, dot, expected);
            }
        }
    }

    #[test]
    fn cholesky_test() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky(&a).unwrap();
        assert!((l[0][0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn cholesky_spd() {
        // Test with a symmetric positive definite matrix
        let a = vec![vec![4.0, 12.0, -16.0], vec![12.0, 37.0, -43.0], vec![-16.0, -43.0, 98.0]];
        let l = cholesky(&a).unwrap();
        
        // Verify L * L^T = A
        let n = a.len();
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..=j.min(i) {
                    sum += l[i][k] * l[j][k];
                }
                assert!((sum - a[i][j]).abs() < 1e-8, "LL^T[{},{}] = {}, expected {}", i, j, sum, a[i][j]);
            }
        }
    }

    #[test]
    fn eigen_2x2() {
        let (lambda1, lambda2) = eigenvalue_2x2([[1.0, 0.0], [0.0, 2.0]]);
        assert!((lambda1.re - 2.0).abs() < 1e-10 || (lambda1.re - 1.0).abs() < 1e-10);
        assert!((lambda2.re - 2.0).abs() < 1e-10 || (lambda2.re - 1.0).abs() < 1e-10);
        assert_eq!(lambda1.im, 0.0);
        assert_eq!(lambda2.im, 0.0);
    }

    #[test]
    fn eigen_2x2_complex() {
        // Test rotation matrix [[0, -1], [1, 0]] which has eigenvalues ±i
        let (lambda1, lambda2) = eigenvalue_2x2([[0.0, -1.0], [1.0, 0.0]]);
        assert!((lambda1.re - 0.0).abs() < 1e-10);
        assert!((lambda1.im - 1.0).abs() < 1e-10 || (lambda1.im - (-1.0)).abs() < 1e-10);
        assert!((lambda2.re - 0.0).abs() < 1e-10);
        assert!((lambda2.im - 1.0).abs() < 1e-10 || (lambda2.im - (-1.0)).abs() < 1e-10);
        // They should be complex conjugates
        assert!((lambda1.im + lambda2.im).abs() < 1e-10);
    }
}
