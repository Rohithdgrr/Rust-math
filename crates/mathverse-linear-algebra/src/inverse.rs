//! Matrix inverse via Gauss-Jordan elimination.
//!
//! Computes the inverse of a square matrix by augmenting it with the
//! identity matrix and applying row operations.
//!
//! Returns `None` if the matrix is singular or non-square.

use crate::solve::solve_gauss;

/// Compute the inverse of a square matrix `a`.
///
/// # Examples
///
/// ```rust
/// use mathverse_linear_algebra::inverse::matrix_inverse;
///
/// let a = vec![
///     vec![4.0, 7.0],
///     vec![2.0, 6.0],
/// ];
/// let inv = matrix_inverse(&a).unwrap();
/// // a * inv should be approximately the identity
/// assert!((inv[0][0] - 0.6).abs() < 1e-10);
/// ```
pub fn matrix_inverse(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 || a.iter().any(|r| r.len() != n) {
        return None;
    }
    // Augment [a | I]
    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = a[i].clone();
        row.extend(std::iter::repeat(0.0).take(n));
        row[n + i] = 1.0;
        aug.push(row);
    }
    // Solve each column of the identity
    for col in 0..n {
        let b: Vec<f64> = (0..n).map(|r| if r == col { 1.0 } else { 0.0 }).collect();
        let x = solve_gauss(&aug, &b)?;
        for r in 0..n {
            aug[r][n + col] = x[r];
        }
    }
    // Extract the inverse from the augmented matrix
    Some(aug.iter().map(|r| r[n..].to_vec()).collect())
}