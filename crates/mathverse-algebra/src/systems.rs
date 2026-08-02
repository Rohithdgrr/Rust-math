//! # Linear Systems
//!
//! Solve 2×2 and 3×3 linear systems `Ax = b` via Cramer's rule.
//!
//! All operations return [`Result<T, AlgebraError>`](crate::Result).
//!
//! ## Examples
//!
//! ```rust
//! use mathverse_algebra::systems::solve_2x2;
//!
//! // x + y = 4, x - y = 2 → x = 3, y = 1
//! let (a, b) = ([1.0, 1.0, 1.0, -1.0], [4.0, 2.0]);
//! let x = solve_2x2(&a, &b).unwrap();
//! assert_eq!(x, [3.0, 1.0]);
//! ```

use crate::determinant::{det_2x2, det_3x3};
use crate::{AlgebraError, TOL};

/// Solve a 2×2 system via Cramer's rule.
///
/// Matrix `a` is row-major `[a, b, c, d]` representing `ax + by = e`, `cx + dy = f`.
/// Vector `b` is `[e, f]`.
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] if `det(a) ≈ 0`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::systems::solve_2x2;
///
/// // x + y = 4, x - y = 2
/// let a = [1.0, 1.0, 1.0, -1.0];
/// let b = [4.0, 2.0];
/// let x = solve_2x2(&a, &b).unwrap();
/// assert_eq!(x, [3.0, 1.0]);
/// ```
pub fn solve_2x2(a: &[f64; 4], b: &[f64; 2]) -> Result<[f64; 2], AlgebraError> {
    let d = det_2x2(a);
    if d.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    Ok([
        (b[0] * a[3] - b[1] * a[1]) / d,
        (b[1] * a[0] - b[0] * a[2]) / d,
    ])
}

/// Solve a 3×3 system via Cramer's rule.
///
/// Matrix `a` is row-major (9 elements). Vector `b` is 3 elements.
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] if `det(a) ≈ 0`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::systems::solve_3x3;
///
/// // Identity
/// let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// let b = [4.0, 5.0, 6.0];
/// let x = solve_3x3(&a, &b).unwrap();
/// assert_eq!(x, [4.0, 5.0, 6.0]);
/// ```
pub fn solve_3x3(a: &[f64; 9], b: &[f64; 3]) -> Result<[f64; 3], AlgebraError> {
    let d = det_3x3(a);
    if d.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    let dx = det_3x3(&[b[0], a[1], a[2], b[1], a[4], a[5], b[2], a[7], a[8]]);
    let dy = det_3x3(&[a[0], b[0], a[2], a[3], b[1], a[5], a[6], b[2], a[8]]);
    let dz = det_3x3(&[a[0], a[1], b[0], a[3], a[4], b[1], a[6], a[7], b[2]]);
    Ok([dx / d, dy / d, dz / d])
}

/// Solve a 2×2 system from a `&[f64]` slice (convenience wrapper).
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] or [`AlgebraError::DimensionMismatch`].
pub fn solve_system_2x2(a: &[f64], b: &[f64]) -> Result<[f64; 2], AlgebraError> {
    if a.len() != 4 || b.len() != 2 {
        return Err(AlgebraError::DimensionMismatch {
            expected: 6,
            actual: a.len() + b.len(),
        });
    }
    Ok([
        (b[0] * a[3] - b[1] * a[1]) / det_2x2(a),
        (b[1] * a[0] - b[0] * a[2]) / det_2x2(a),
    ])
}

/// Solve a 3×3 system from `&[f64]` slices (convenience wrapper).
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] or [`AlgebraError::DimensionMismatch`].
pub fn solve_system_3x3(a: &[f64], b: &[f64]) -> Result<[f64; 3], AlgebraError> {
    if a.len() != 9 || b.len() != 3 {
        return Err(AlgebraError::DimensionMismatch {
            expected: 12,
            actual: a.len() + b.len(),
        });
    }
    let d = det_3x3(a);
    if d.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    let dx = det_3x3(&[b[0], a[1], a[2], b[1], a[4], a[5], b[2], a[7], a[8]]);
    let dy = det_3x3(&[a[0], b[0], a[2], a[3], b[1], a[5], a[6], b[2], a[8]]);
    let dz = det_3x3(&[a[0], a[1], b[0], a[3], a[4], b[1], a[6], a[7], b[2]]);
    Ok([dx / d, dy / d, dz / d])
}

/// Compute the residual `‖Ax - b‖∞` for a 2×2 system.
#[must_use]
pub fn residual_2x2(a: &[f64; 4], x: &[f64; 2], b: &[f64; 2]) -> f64 {
    let r0 = (a[0] * x[0] + a[1] * x[1] - b[0]).abs();
    let r1 = (a[2] * x[0] + a[3] * x[1] - b[1]).abs();
    r0.max(r1)
}

/// Compute the residual `‖Ax - b‖∞` for a 3×3 system.
#[must_use]
pub fn residual_3x3(a: &[f64; 9], x: &[f64; 3], b: &[f64; 3]) -> f64 {
    let r0 = (a[0] * x[0] + a[1] * x[1] + a[2] * x[2] - b[0]).abs();
    let r1 = (a[3] * x[0] + a[4] * x[1] + a[5] * x[2] - b[1]).abs();
    let r2 = (a[6] * x[0] + a[7] * x[1] + a[8] * x[2] - b[2]).abs();
    r0.max(r1).max(r2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2x2() {
        let a = [2.0, 1.0, 5.0, 3.0];
        let b = [4.0, 7.0];
        let x = solve_2x2(&a, &b).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-9);
        assert!((x[1] - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn test_3x3() {
        let a = [2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0];
        let b = [4.0, 8.0, 10.0];
        let x = solve_3x3(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 1.0).abs() < 1e-9);
        assert!((x[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_singular() {
        let a = [1.0, 2.0, 2.0, 4.0];
        let b = [3.0, 6.0];
        assert_eq!(solve_2x2(&a, &b), Err(AlgebraError::Singular));
    }

    #[test]
    fn test_identity() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [3.0, 4.0];
        let x = solve_2x2(&a, &b).unwrap();
        assert_eq!(x, [3.0, 4.0]);
    }

    #[test]
    fn residual_test() {
        let a = [2.0, 1.0, 5.0, 3.0];
        let x = [5.0, -2.0];
        let b = [4.0, 7.0];
        let r = residual_2x2(&a, &x, &b);
        assert!(r < 1e-9);
    }
}
