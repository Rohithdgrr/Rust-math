//! Linear system solvers: LU-based, QR-based, Gaussian, least-squares.

use mathverse_matrix::Matrix;

/// Solves $Ax = b$ using pre-computed LU decomposition and permutation vector.
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

/// Solves $Ax = b$ using pre-computed QR decomposition ($Q^T b = R x$).
pub fn solve_qr(q: &Matrix, r: &Matrix, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut qt_b = vec![0.0; n];
    for i in 0..n {
        qt_b[i] = (0..n).map(|j| q.get(j, i) * b[j]).sum();
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (qt_b[i] - ((i + 1)..n).map(|j| r.get(i, j) * x[j]).sum::<f64>()) / r.get(i, i);
    }
    x
}

/// Solves a 2x2 linear system $Ax = b$ analytically using Cramer's rule.
pub fn solve_2x2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-15 {
        return None;
    }
    Some([
        (b[0] * a[1][1] - b[1] * a[0][1]) / det,
        (a[0][0] * b[1] - a[1][0] * b[0]) / det,
    ])
}

/// Solves a 3x3 linear system $Ax = b$ analytically using Cramer's rule.
#[allow(clippy::similar_names)] // matrix entries a[i][j] are inherently similar
pub fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-15 {
        return None;
    }
    Some([
        (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
            + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
            / det,
        (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2]) - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
            / det,
        (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1]) - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
            + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
            / det,
    ])
}

/// Solves $Ax = b$ using Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)] // index arithmetic clearer in elimination loops
pub fn solve_gauss(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.rows();
    if !a.is_square() || n == 0 || b.len() != n {
        return None;
    }
    // Build the augmented matrix [a | b]
    let mut aug = Matrix::zeros(n, n + 1);
    for i in 0..n {
        for c in 0..n {
            aug.set(i, c, a.get(i, c));
        }
        aug.set(i, n, b[i]);
    }
    for col in 0..n {
        let mut max_row = col;
        for r in col + 1..n {
            if aug.get(r, col).abs() > aug.get(max_row, col).abs() {
                max_row = r;
            }
        }
        swap_rows(&mut aug, col, max_row);
        if aug.get(col, col).abs() < 1e-15 {
            return None;
        }
        for r in col + 1..n {
            let f = aug.get(r, col) / aug.get(col, col);
            for c in col..=n {
                aug.set(r, c, aug.get(r, c) - f * aug.get(col, c));
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (aug.get(i, n) - ((i + 1)..n).map(|j| aug.get(i, j) * x[j]).sum::<f64>())
            / aug.get(i, i);
    }
    Some(x)
}

fn swap_rows(m: &mut Matrix, i: usize, k: usize) {
    for c in 0..m.cols() {
        let temp = m.get(i, c);
        m.set(i, c, m.get(k, c));
        m.set(k, c, temp);
    }
}

/// Solves an overdetermined linear system $Ax \approx b$ in the least-squares sense using QR decomposition.
pub fn ls_solve(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    // Use QR decomposition for better numerical stability
    // Solve Ax = b via QR: Q^T b = R x, then back substitution
    let (q, r) = crate::decomposition::qr_decompose(a).ok()?;

    // Compute Q^T b
    let (m, n) = a.shape();
    let mut qtb = vec![0.0; n];
    for i in 0..n {
        qtb[i] = (0..m).map(|j| q.get(j, i) * b[j]).sum();
    }

    // Back substitution on R (upper triangular)
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if r.get(i, i).abs() < 1e-15 {
            return None; // Rank deficient
        }
        x[i] = (qtb[i] - ((i + 1)..n).map(|j| r.get(i, j) * x[j]).sum::<f64>()) / r.get(i, i);
    }

    Some(x)
}

/// Computes the residual norm $\|b - Ax\|_2$.
pub fn residual_norm(a: &Matrix, b: &[f64], x: &[f64]) -> f64 {
    let m = a.rows();
    let mut sum = 0.0;
    for i in 0..m {
        let r = b[i] - (0..x.len()).map(|j| a.get(i, j) * x[j]).sum::<f64>();
        sum += r * r;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(rows: &[&[f64]]) -> Matrix {
        Matrix::from_rows(rows).unwrap()
    }

    #[test]
    fn lu_solve() {
        let a = mat(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let (l, u, perm) = crate::decomposition::lu_decompose(&a).unwrap();
        let x = solve_lu(&l, &u, &perm, &[5.0, 7.0]);
        assert!((x[0] - 1.6).abs() < 1e-10);
    }

    #[test]
    fn gauss_test() {
        let a = mat(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let x = solve_gauss(&a, &[5.0, 7.0]).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
    }

    #[test]
    fn least_squares() {
        let a = mat(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
        let b = vec![1.0, 2.0, 3.5];
        let x = ls_solve(&a, &b).unwrap();
        // The least squares solution should be computed via QR
        // Verify it minimizes the residual
        let residual = residual_norm(&a, &b, &x);
        assert!(residual < 0.5);
    }

    #[test]
    fn least_squares_overdetermined() {
        // Test with a classic overdetermined system
        let a = mat(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0], &[1.0, 4.0]]);
        let b = vec![6.0, 5.0, 7.0, 10.0];
        let x = ls_solve(&a, &b).unwrap();

        // Verify the solution minimizes the residual
        let residual = residual_norm(&a, &b, &x);
        // The solution should be reasonable
        assert!(residual < 5.0);
        // Solution should be finite
        assert!(x[0].is_finite());
        assert!(x[1].is_finite());
    }

    #[test]
    fn solve_2x2_exact() {
        let a = [[2.0, 1.0], [1.0, 3.0]];
        let b = [5.0, 7.0];
        let x = solve_2x2(a, b).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn solve_3x3_exact() {
        let a = [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let b = [5.0, 7.0, 3.0];
        let x = solve_3x3(a, b).unwrap();
        // Verify solution works
        let r0 = a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2];
        let r1 = a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2];
        let r2 = a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2];
        assert!((r0 - b[0]).abs() < 1e-10);
        assert!((r1 - b[1]).abs() < 1e-10);
        assert!((r2 - b[2]).abs() < 1e-10);
    }
}
