//! Matrix inverse via Gauss-Jordan elimination.
//!
//! Computes the inverse of a square matrix by augmenting it with the
//! identity matrix and applying row operations.

use mathverse_core::error::{MathError, MathResult};
use mathverse_matrix::Matrix;

use crate::solve::solve_gauss;

/// Compute the inverse of a square matrix `a`.
///
/// # Errors
///
/// Returns [`MathError::InvalidArgument`] for empty matrices,
/// [`MathError::DimensionMismatch`] for non-square matrices and
/// [`MathError::Singular`] for singular matrices.
///
/// # Examples
///
/// ```rust
/// use mathverse_linear_algebra::inverse::matrix_inverse;
/// use mathverse_matrix::Matrix;
///
/// let a = Matrix::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]).unwrap();
/// let inv = matrix_inverse(&a).unwrap();
/// // a * inv should be approximately the identity
/// assert!((inv.get(0, 0) - 0.6).abs() < 1e-10);
/// ```
#[allow(clippy::needless_range_loop)] // index arithmetic clearer in augmentation loops
pub fn matrix_inverse(a: &Matrix) -> MathResult<Matrix> {
    let n = a.rows();
    if n == 0 {
        return Err(MathError::InvalidArgument(
            "matrix_inverse requires a non-empty matrix",
        ));
    }
    if !a.is_square() {
        return Err(MathError::DimensionMismatch);
    }
    // Solve A x = e_col for each column of the identity
    let mut inv = Matrix::zeros(n, n);
    for col in 0..n {
        let b: Vec<f64> = (0..n).map(|r| if r == col { 1.0 } else { 0.0 }).collect();
        let x = solve_gauss(a, &b).ok_or(MathError::Singular)?;
        for r in 0..n {
            inv.set(r, col, x[r]);
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_round_trip() {
        let a = Matrix::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]).unwrap();
        let inv = matrix_inverse(&a).unwrap();
        assert!((inv.get(0, 0) - 0.6).abs() < 1e-10);
        let prod = a.mul(&inv).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((prod.get(i, j) - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn inverse_singular_and_nonsquare() {
        let sing = Matrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]).unwrap();
        assert_eq!(matrix_inverse(&sing), Err(MathError::Singular));
        let rect = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();
        assert_eq!(matrix_inverse(&rect), Err(MathError::DimensionMismatch));
    }
}
