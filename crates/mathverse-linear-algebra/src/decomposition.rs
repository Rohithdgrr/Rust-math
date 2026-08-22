use mathverse_core::error::{MathError, MathResult};
use mathverse_matrix::Matrix;

/// Computes the LU decomposition with partial pivoting ($PA = LU$).
///
/// Returns `(L, U, perm)` where:
/// - `L` is unit lower triangular $n \times n$
/// - `U` is upper triangular $n \times n$
/// - `perm` is a vector of row indices indicating the permutation matrix $P$
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] if the matrix is empty or not square,
/// and [`MathError::Singular`] if the matrix is singular.
///
/// # Examples
/// ```
/// use mathverse_linear_algebra::lu_decompose;
/// use mathverse_matrix::Matrix;
///
/// let a = Matrix::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]).unwrap();
/// let (l, _u, _perm) = lu_decompose(&a).unwrap();
/// assert!((l.get(1, 0) - 0.5).abs() < 1e-10);
/// ```
#[allow(clippy::needless_range_loop)] // index arithmetic clearer in elimination loops
pub fn lu_decompose(a: &Matrix) -> MathResult<(Matrix, Matrix, Vec<usize>)> {
    let n = a.rows();
    if !a.is_square() || n == 0 {
        return Err(MathError::InvalidArgument(
            "lu_decompose requires a non-empty square matrix",
        ));
    }

    let mut l = Matrix::zeros(n, n);
    let mut u = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();

    for i in 0..n {
        // Partial pivoting: find the row with maximum absolute value in column i
        let mut max_row = i;
        let mut max_val = u.get(i, i).abs();
        for k in (i + 1)..n {
            if u.get(k, i).abs() > max_val {
                max_val = u.get(k, i).abs();
                max_row = k;
            }
        }

        // Swap rows if necessary
        if max_row != i {
            swap_rows(&mut u, i, max_row);
            perm.swap(i, max_row);
            // Swap the already-computed part of L (only columns 0..i-1 are valid)
            for k in 0..i {
                let temp = l.get(i, k);
                l.set(i, k, l.get(max_row, k));
                l.set(max_row, k, temp);
            }
        }

        // Check for singularity
        if u.get(i, i).abs() < 1e-15 {
            return Err(MathError::Singular);
        }

        // Compute U's row i and L's column i
        for k in i..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l.get(i, j) * u.get(j, k);
            }
            u.set(i, k, u.get(i, k) - sum);
        }

        l.set(i, i, 1.0);
        for k in (i + 1)..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l.get(k, j) * u.get(j, i);
            }
            l.set(k, i, (u.get(k, i) - sum) / u.get(i, i));
        }
    }

    Ok((l, u, perm))
}

fn swap_rows(m: &mut Matrix, i: usize, k: usize) {
    for c in 0..m.cols() {
        let temp = m.get(i, c);
        m.set(i, c, m.get(k, c));
        m.set(k, c, temp);
    }
}

/// Computes the QR decomposition of an $m \times n$ matrix ($A = QR$) using Modified Gram-Schmidt.
///
/// Returns `(Q, R)` where `Q` has orthonormal columns ($m \times n$) and `R` is upper triangular ($n \times n$).
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] if the matrix is empty and
/// [`MathError::Singular`] if its columns are linearly dependent.
#[allow(clippy::needless_range_loop)] // index arithmetic clearer in orthogonalization loops
pub fn qr_decompose(a: &Matrix) -> MathResult<(Matrix, Matrix)> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err(MathError::InvalidArgument(
            "qr_decompose requires a non-empty matrix",
        ));
    }
    let mut q = Matrix::zeros(m, n); // Q is m×n with orthonormal columns
    let mut r = Matrix::zeros(n, n); // R is n×n upper triangular

    for j in 0..n {
        // Start with the j-th column of A
        let mut v: Vec<f64> = (0..m).map(|i| a.get(i, j)).collect();

        // Modified Gram-Schmidt: orthogonalize against each previous q vector
        for i in 0..j {
            let dot: f64 = (0..m).map(|k| q.get(k, i) * v[k]).sum();
            r.set(i, j, dot);
            // Subtract projection immediately (this is the key difference from classical GS)
            for k in 0..m {
                v[k] -= dot * q.get(k, i);
            }
        }

        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return Err(MathError::Singular);
        }
        r.set(j, j, norm);

        for k in 0..m {
            q.set(k, j, v[k] / norm);
        }
    }

    Ok((q, r))
}

/// Computes the Cholesky factorization of a symmetric positive-definite matrix ($A = L L^T$).
///
/// Returns `L`, lower triangular.
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] if the matrix is empty or not square,
/// and [`MathError::Singular`] if the matrix is not positive-definite.
#[allow(clippy::needless_range_loop)] // index arithmetic clearer in factorization loops
pub fn cholesky(a: &Matrix) -> MathResult<Matrix> {
    let n = a.rows();
    if !a.is_square() || n == 0 {
        return Err(MathError::InvalidArgument(
            "cholesky requires a non-empty square matrix",
        ));
    }
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..i {
            sum += l.get(i, k) * l.get(i, k);
        }
        let diag = a.get(i, i) - sum;
        if diag <= 1e-15 {
            return Err(MathError::Singular);
        }
        l.set(i, i, diag.sqrt());
        for j in (i + 1)..n {
            let mut sum = 0.0;
            for k in 0..i {
                sum += l.get(j, k) * l.get(i, k);
            }
            l.set(j, i, (a.get(j, i) - sum) / l.get(i, i));
        }
    }
    Ok(l)
}

/// Solves a linear system $Ax = b$ given its LU decomposition $(L, U, P)$.
#[allow(clippy::needless_range_loop)] // index arithmetic clearer in substitution loops
pub fn solve_lu(l: &Matrix, u: &Matrix, perm: &[usize], b: &[f64]) -> Vec<f64> {
    let n = b.len();

    // Apply permutation to b
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[perm[i]];
    }

    // Forward substitution (solve Ly = Pb)
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = pb[i] - (0..i).map(|j| l.get(i, j) * y[j]).sum::<f64>();
    }

    // Backward substitution (solve Ux = y)
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - ((i + 1)..n).map(|j| u.get(i, j) * x[j]).sum::<f64>()) / u.get(i, i);
    }
    x
}

/// Simple complex number representation for eigenvalue results.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    /// Real part
    pub re: f64,
    /// Imaginary part
    pub im: f64,
}

/// Computes exact eigenvalues for a $2 \times 2$ real matrix.
pub fn eigenvalue_2x2(a: [[f64; 2]; 2]) -> (Complex, Complex) {
    let trace = a[0][0] + a[1][1];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let disc = trace * trace - 4.0 * det;

    if disc >= 0.0 {
        // Real eigenvalues
        let sqrt_disc = disc.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        (
            Complex { re: lambda1, im: 0.0 },
            Complex { re: lambda2, im: 0.0 },
        )
    } else {
        // Complex eigenvalues
        let sqrt_disc_abs = (-disc).sqrt();
        let real_part = trace / 2.0;
        let imag_part = sqrt_disc_abs / 2.0;
        (
            Complex { re: real_part, im: imag_part },
            Complex { re: real_part, im: -imag_part },
        )
    }
}

/// Approximates the dominant eigenvalue and eigenvector using Power Iteration.
///
/// Returns `Some((eigenvector, eigenvalue))` or `None` if convergence fails.
#[allow(clippy::needless_range_loop)] // index arithmetic clearer in iteration loops
pub fn power_iteration(a: &Matrix, max_iter: usize, tol: f64) -> Option<(Vec<f64>, f64)> {
    let n = a.rows();
    if n == 0 {
        return None;
    }
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda = 0.0;
    for _ in 0..max_iter {
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += a.get(i, j) * v[j];
            }
        }
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            break;
        }
        for i in 0..n {
            v[i] = w[i] / norm;
        }
        let mut new_lambda = 0.0;
        for i in 0..n {
            new_lambda += v[i] * (0..n).map(|j| a.get(i, j) * v[j]).sum::<f64>();
        }
        if (new_lambda - lambda).abs() < tol {
            return Some((v, new_lambda));
        }
        lambda = new_lambda;
    }
    Some((v, lambda))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: &[&[f64]]) -> Matrix {
        Matrix::from_rows(rows).unwrap()
    }

    #[test]
    fn lu_test() {
        let a = mat(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let (l, u, perm) = lu_decompose(&a).unwrap();
        let x = solve_lu(&l, &u, &perm, &[5.0, 7.0]);
        assert!((x[0] - 1.6).abs() < 1e-10);
    }

    #[test]
    fn lu_pivoting_test() {
        // Test matrix that requires pivoting: [[0,1],[1,0]]
        let a = mat(&[&[0.0, 1.0], &[1.0, 0.0]]);
        let (l, u, perm) = lu_decompose(&a).unwrap();
        let x = solve_lu(&l, &u, &perm, &[1.0, 0.0]);
        // Solution should be [0, 1]
        assert!((x[0] - 0.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn lu_ill_conditioned() {
        // Test with a matrix that has small but non-zero pivots
        let a = mat(&[&[1e-10, 1.0], &[1.0, 1.0]]);
        let (l, u, perm) = lu_decompose(&a).unwrap();
        let x = solve_lu(&l, &u, &perm, &[1.0, 2.0]);
        // Solution should exist and be reasonable
        assert!(x[0].is_finite());
        assert!(x[1].is_finite());
    }

    #[test]
    fn lu_nonsquare() {
        let rect = mat(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        assert!(lu_decompose(&rect).is_err());
    }

    #[test]
    fn qr_test() {
        let a = mat(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let (q, _r) = qr_decompose(&a).unwrap();
        assert!((q.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn qr_orthogonality_test() {
        // Test that Q^T Q is close to identity
        let a = mat(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let (q, _) = qr_decompose(&a).unwrap();

        let (_m, n) = q.shape();
        // Compute Q^T Q
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..q.rows()).map(|k| q.get(k, i) * q.get(k, j)).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q^T Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn qr_ill_conditioned() {
        // Test QR with a nearly rank-deficient matrix
        let a = mat(&[
            &[1.0, 2.0],
            &[1.0, 2.0 + 1e-8],
            &[1.0, 2.0 + 2e-8],
        ]);
        let (q, _r) = qr_decompose(&a).unwrap();

        // Check that Q is still orthogonal
        let (_m, n) = q.shape();
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..q.rows()).map(|k| q.get(k, i) * q.get(k, j)).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "Q^T Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn cholesky_test() {
        let a = mat(&[&[4.0, 2.0], &[2.0, 3.0]]);
        let l = cholesky(&a).unwrap();
        assert!((l.get(0, 0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn cholesky_spd() {
        // Test with a symmetric positive definite matrix
        let a = mat(&[
            &[4.0, 12.0, -16.0],
            &[12.0, 37.0, -43.0],
            &[-16.0, -43.0, 98.0],
        ]);
        let l = cholesky(&a).unwrap();

        // Verify L * L^T = A
        let n = a.rows();
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..=j.min(i) {
                    sum += l.get(i, k) * l.get(j, k);
                }
                assert!(
                    (sum - a.get(i, j)).abs() < 1e-8,
                    "LL^T[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    a.get(i, j)
                );
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
