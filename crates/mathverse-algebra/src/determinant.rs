//! # Determinant, Inverse, Cramer's Rule, Rank, and Trace
//!
//! Determinant, inverse, and Cramer's rule for 2×2 and 3×3 matrices.
//! Also includes rank and trace helpers.
//!
//! Matrices are represented as `&[f64]` slices in **row-major** order:
//! `[a, b, c, d]` → `[[a, b], [c, d]]`.
//!
//! ## Examples
//!
//! ```rust
//! use mathverse_algebra::determinant::{det_2x2, inverse_2x2, cramer_rule_2x2};
//!
//! let m = [1.0, 2.0, 3.0, 4.0];
//! assert_eq!(det_2x2(&m), -2.0);
//!
//! let inv = inverse_2x2(&m).unwrap();
//! assert!((inv[0] - (-2.0)).abs() < 1e-9);
//!
//! // [1x + 2y = 5, 3x + 4y = 6]
//! let x = cramer_rule_2x2(&m, &[5.0, 6.0]).unwrap();
//! assert!((x[0] - (-4.0)).abs() < 1e-9);
//! assert!((x[1] - 4.5).abs() < 1e-9);
//! ```

use crate::{AlgebraError, TOL};

/// 2×2 determinant `ad - bc`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::det_2x2;
///
/// assert_eq!(det_2x2(&[1.0, 2.0, 3.0, 4.0]), -2.0);
/// ```
#[inline]
#[must_use]
pub fn det_2x2(m: &[f64]) -> f64 {
    m[0] * m[3] - m[1] * m[2]
}

/// 3×3 determinant by cofactor expansion along the first row.
///
/// `m` is 9 elements: `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::det_3x3;
///
/// // Identity matrix
/// let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// assert_eq!(det_3x3(&m), 1.0);
/// ```
#[must_use]
pub fn det_3x3(m: &[f64]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// 2×2 inverse via adjugate, returning [`Err(AlgebraError::Singular)`] if `det ≈ 0`.
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] if the matrix is singular.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::inverse_2x2;
///
/// let inv = inverse_2x2(&[1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert!((inv[0] - (-2.0)).abs() < 1e-9);
/// ```
pub fn inverse_2x2(m: &[f64]) -> Result<[f64; 4], AlgebraError> {
    let d = det_2x2(m);
    if d.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    Ok([m[3] / d, -m[1] / d, -m[2] / d, m[0] / d])
}

/// 3×3 inverse using the classical adjoint method.
///
/// Returns [`Err(AlgebraError::Singular)`] if `det ≈ 0`.
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] if the matrix is singular.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::inverse_3x3;
///
/// let m = [2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0];
/// let inv = inverse_3x3(&m).unwrap();
/// assert!((inv[0] - 0.5).abs() < 1e-9);
/// ```
pub fn inverse_3x3(m: &[f64]) -> Result<[f64; 9], AlgebraError> {
    let d = det_3x3(m);
    if d.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    let inv = [
        m[4] * m[8] - m[5] * m[7],
        m[2] * m[7] - m[1] * m[8],
        m[1] * m[5] - m[2] * m[4],
        m[5] * m[6] - m[3] * m[8],
        m[0] * m[8] - m[2] * m[6],
        m[2] * m[3] - m[0] * m[5],
        m[3] * m[7] - m[4] * m[6],
        m[1] * m[6] - m[0] * m[7],
        m[0] * m[4] - m[1] * m[3],
    ];
    Ok(inv.map(|x| x / d))
}

/// 2×2 Cramer's rule: solve `[m] x = b`.
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] if `det(m) ≈ 0`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::cramer_rule_2x2;
///
/// // x + 2y = 5, 3x + 4y = 6
/// let m = [1.0, 2.0, 3.0, 4.0];
/// let b = [5.0, 6.0];
/// let x = cramer_rule_2x2(&m, &b).unwrap();
/// assert!((x[0] - (-4.0)).abs() < 1e-9);
/// assert!((x[1] - 4.5).abs() < 1e-9);
/// ```
pub fn cramer_rule_2x2(m: &[f64], b: &[f64]) -> Result<[f64; 2], AlgebraError> {
    let d = det_2x2(m);
    if d.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    Ok([
        (b[0] * m[3] - b[1] * m[1]) / d,
        (b[1] * m[0] - b[0] * m[2]) / d,
    ])
}

/// 3×3 Cramer's rule: solve `[m] x = b`.
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] if `det(m) ≈ 0`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::cramer_rule_3x3;
///
/// // Identity matrix
/// let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// let b = [1.0, 2.0, 3.0];
/// let x = cramer_rule_3x3(&m, &b).unwrap();
/// assert_eq!(x, [1.0, 2.0, 3.0]);
/// ```
pub fn cramer_rule_3x3(m: &[f64], b: &[f64]) -> Result<[f64; 3], AlgebraError> {
    let d = det_3x3(m);
    if d.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    let dx = det_3x3(&[b[0], m[1], m[2], b[1], m[4], m[5], b[2], m[7], m[8]]);
    let dy = det_3x3(&[m[0], b[0], m[2], m[3], b[1], m[5], m[6], b[2], m[8]]);
    let dz = det_3x3(&[m[0], m[1], b[0], m[3], m[4], b[1], m[6], m[7], b[2]]);
    Ok([dx / d, dy / d, dz / d])
}

/// Trace of a 2×2 matrix: `m00 + m11`.
#[inline]
#[must_use]
pub fn trace_2x2(m: &[f64]) -> f64 {
    m[0] + m[3]
}

/// Trace of a 3×3 matrix: `m00 + m11 + m22`.
#[inline]
#[must_use]
pub fn trace_3x3(m: &[f64]) -> f64 {
    m[0] + m[4] + m[8]
}

/// Rank of a 2×2 matrix (0, 1, or 2).
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::rank_2x2;
///
/// assert_eq!(rank_2x2(&[1.0, 0.0, 0.0, 1.0]), 2);
/// assert_eq!(rank_2x2(&[1.0, 2.0, 2.0, 4.0]), 1);
/// assert_eq!(rank_2x2(&[0.0, 0.0, 0.0, 0.0]), 0);
/// ```
#[must_use]
pub fn rank_2x2(m: &[f64]) -> u8 {
    if det_2x2(m).abs() > TOL {
        return 2;
    }
    if m.iter().any(|&x| x.abs() > TOL) {
        return 1;
    }
    0
}

/// Rank of a 3×3 matrix (0, 1, 2, or 3).
///
/// Checks `det_3x3` first, then all 2×2 submatrices, then nonzero entries.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::rank_3x3;
///
/// assert_eq!(rank_3x3(&[1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]), 3);
/// assert_eq!(rank_3x3(&[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), 0);
/// ```
#[must_use]
pub fn rank_3x3(m: &[f64]) -> u8 {
    if det_3x3(m).abs() > TOL {
        return 3;
    }
    let submatrices = [
        [m[0], m[1], m[3], m[4]],
        [m[0], m[2], m[3], m[5]],
        [m[0], m[2], m[6], m[8]],
        [m[1], m[2], m[4], m[5]],
        [m[1], m[2], m[7], m[8]],
        [m[4], m[5], m[7], m[8]],
        [m[3], m[4], m[6], m[7]],
        [m[3], m[5], m[6], m[8]],
    ];
    for s in &submatrices {
        if det_2x2(s).abs() > TOL {
            return 2;
        }
    }
    if m.iter().any(|&x| x.abs() > TOL) {
        return 1;
    }
    0
}

/// Solve a 2×2 system `[m] x = b` using Cramer's rule.
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] if the system is singular.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::solve_matrix_2x2;
///
/// let m = [2.0, 1.0, 5.0, 3.0];
/// let b = [4.0, 7.0];
/// let x = solve_matrix_2x2(&m, &b).unwrap();
/// assert!((x[0] - 5.0).abs() < 1e-9);
/// assert!((x[1] - (-2.0)).abs() < 1e-9);
/// ```
pub fn solve_matrix_2x2(m: &[f64], b: &[f64]) -> Result<[f64; 2], AlgebraError> {
    cramer_rule_2x2(m, b)
}

/// Solve a 3×3 system `[m] x = b` using Cramer's rule.
///
/// # Errors
///
/// Returns [`AlgebraError::Singular`] if the system is singular.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::determinant::solve_matrix_3x3;
///
/// let m = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
/// let b = [6.0, 8.0, 9.0];
/// let x = solve_matrix_3x3(&m, &b).unwrap();
/// assert!((x[0] - 6.0).abs() < 1e-9);
/// assert!((x[1] - 4.0).abs() < 1e-9);
/// assert!((x[2] - 3.0).abs() < 1e-9);
/// ```
pub fn solve_matrix_3x3(m: &[f64], b: &[f64]) -> Result<[f64; 3], AlgebraError> {
    cramer_rule_3x3(m, b)
}

/// Singular: matrix determinant is effectively zero.
pub fn is_singular_2x2(m: &[f64]) -> bool {
    det_2x2(m).abs() < TOL
}

/// Singular: matrix determinant is effectively zero.
pub fn is_singular_3x3(m: &[f64]) -> bool {
    det_3x3(m).abs() < TOL
}

/// Compute the cofactor matrix of a 3×3 matrix.
#[must_use]
pub fn cofactor_matrix_3x3(m: &[f64]) -> [f64; 9] {
    [
        m[4] * m[8] - m[5] * m[7],
        -(m[3] * m[8] - m[5] * m[6]),
        m[3] * m[7] - m[4] * m[6],
        -(m[1] * m[8] - m[2] * m[7]),
        m[0] * m[8] - m[2] * m[6],
        -(m[0] * m[7] - m[1] * m[6]),
        m[1] * m[5] - m[2] * m[4],
        -(m[0] * m[5] - m[2] * m[3]),
        m[0] * m[4] - m[1] * m[3],
    ]
}

/// Compute the adjugate (classical adjoint) of a 3×3 matrix.
#[must_use]
pub fn adjugate_3x3(m: &[f64]) -> [f64; 9] {
    let c = cofactor_matrix_3x3(m);
    // transpose
    [
        c[0], c[3], c[6],
        c[1], c[4], c[7],
        c[2], c[5], c[8],
    ]
}

/// Multiply two 3×3 matrices.
#[must_use]
pub fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut r = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    r
}

/// Multiply a 3×3 matrix by a 3-vector.
#[must_use]
pub fn mat3_vec(m: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// Multiply two 2×2 matrices.
#[must_use]
pub fn mat2_mul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

/// Multiply a 2×2 matrix by a 2-vector.
#[must_use]
pub fn mat2_vec(m: &[f64; 4], v: &[f64; 2]) -> [f64; 2] {
    [m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1]]
}

/// 2×2 identity matrix.
#[must_use]
pub fn identity_2x2() -> [f64; 4] {
    [1.0, 0.0, 0.0, 1.0]
}

/// 3×3 identity matrix.
#[must_use]
pub fn identity_3x3() -> [f64; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

/// Characteristic polynomial coefficients of a 2×2 matrix.
///
/// Returns `[1, -(trace), (determinant)]` — the monic polynomial in
/// **lowest-degree-first** order.
#[must_use]
pub fn characteristic_poly_2x2(m: &[f64]) -> [f64; 3] {
    [1.0, -trace_2x2(m), det_2x2(m)]
}

/// Characteristic polynomial coefficients of a 3×3 matrix.
///
/// Returns `[1, -(trace), (sum of principal 2×2 minors), -(determinant)]`
/// in lowest-degree-first order.
#[must_use]
pub fn characteristic_poly_3x3(m: &[f64]) -> [f64; 4] {
    let t = trace_3x3(m);
    let m11 = m[4] * m[8] - m[5] * m[7];
    let m22 = m[0] * m[8] - m[2] * m[6];
    let m33 = m[0] * m[4] - m[1] * m[3];
    [1.0, -t, m11 + m22 + m33, -det_3x3(m)]
}

/// Eigenvalues of a 2×2 matrix.
///
/// Returns the real roots of the characteristic polynomial.
#[must_use]
pub fn eigenvalues_2x2(m: &[f64]) -> Vec<f64> {
    let [c0, c1, c2] = characteristic_poly_2x2(m);
    crate::roots::solve_quadratic(c2, c1, c0)
}

/// Eigenvalues of a 3×3 matrix.
///
/// Returns the real roots of the characteristic polynomial.
#[must_use]
pub fn eigenvalues_3x3(m: &[f64]) -> Vec<f64> {
    let [c0, c1, c2, c3] = characteristic_poly_3x3(m);
    crate::roots::solve_cubic(c3, c2, c1, c0)
}

/// Matrix `A - λI` for 2×2.
#[must_use]
pub fn matrix_minus_lambda_2x2(m: &[f64], lambda: f64) -> [f64; 4] {
    [m[0] - lambda, m[1], m[2], m[3] - lambda]
}

/// Matrix `A - λI` for 3×3.
#[must_use]
pub fn matrix_minus_lambda_3x3(m: &[f64], lambda: f64) -> [f64; 9] {
    [
        m[0] - lambda, m[1], m[2],
        m[3], m[4] - lambda, m[5],
        m[6], m[7], m[8] - lambda,
    ]
}

// Backward-compatible aliases

/// Deprecated: use [`det_2x2`] instead.
#[deprecated(since = "0.1.1", note = "renamed to det_2x2")]
#[must_use]
pub fn det2(m: &[f64]) -> f64 {
    det_2x2(m)
}

/// Deprecated: use [`det_3x3`] instead.
#[deprecated(since = "0.1.1", note = "renamed to det_3x3")]
#[must_use]
pub fn det3(m: &[f64]) -> f64 {
    det_3x3(m)
}

/// Deprecated: use [`inverse_2x2`] instead.
#[deprecated(since = "0.1.1", note = "renamed to inverse_2x2")]
pub fn inverse2(m: &[f64]) -> Result<[f64; 4], AlgebraError> {
    inverse_2x2(m)
}

/// Deprecated: use [`inverse_3x3`] instead.
#[deprecated(since = "0.1.1", note = "renamed to inverse_3x3")]
pub fn inverse3(m: &[f64]) -> Result<[f64; 9], AlgebraError> {
    inverse_3x3(m)
}

/// Deprecated: use [`cramer_rule_2x2`] instead.
#[deprecated(since = "0.1.1", note = "renamed to cramer_rule_2x2")]
pub fn cramers_rule_2x2(m: &[f64], b: &[f64]) -> Result<[f64; 2], AlgebraError> {
    cramer_rule_2x2(m, b)
}

/// Deprecated: use [`cramer_rule_3x3`] instead.
#[deprecated(since = "0.1.1", note = "renamed to cramer_rule_3x3")]
pub fn cramers_rule_3x3(m: &[f64], b: &[f64]) -> Result<[f64; 3], AlgebraError> {
    cramer_rule_3x3(m, b)
}

/// Deprecated: use [`solve_matrix_2x2`] instead.
#[deprecated(since = "0.1.1", note = "renamed to solve_matrix_2x2")]
pub fn solve2x2(m: &[f64], b: &[f64]) -> Result<[f64; 2], AlgebraError> {
    solve_matrix_2x2(m, b)
}

/// Deprecated: use [`solve_matrix_3x3`] instead.
#[deprecated(since = "0.1.1", note = "renamed to solve_matrix_3x3")]
pub fn solve3x3(m: &[f64], b: &[f64]) -> Result<[f64; 3], AlgebraError> {
    solve_matrix_3x3(m, b)
}

/// Deprecated: use [`is_singular_2x2`] instead.
#[deprecated(since = "0.1.1", note = "renamed to is_singular_2x2")]
#[must_use]
pub fn is_singular2(m: &[f64]) -> bool {
    is_singular_2x2(m)
}

/// Deprecated: use [`is_singular_3x3`] instead.
#[deprecated(since = "0.1.1", note = "renamed to is_singular_3x3")]
#[must_use]
pub fn is_singular3(m: &[f64]) -> bool {
    is_singular_3x3(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det2_test() {
        assert_eq!(det_2x2(&[1.0, 2.0, 3.0, 4.0]), -2.0);
    }

    #[test]
    fn det3_test() {
        let m = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(det_3x3(&m), 0.0);
    }

    #[test]
    fn inverse_test() {
        let inv = inverse_2x2(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((inv[0] - (-2.0)).abs() < 1e-9);
        assert!((inv[1] - 1.0).abs() < 1e-9);
        assert!((inv[2] - 1.5).abs() < 1e-9);
        assert!((inv[3] - (-0.5)).abs() < 1e-9);
    }

    #[test]
    fn singular_matrix() {
        let m = [1.0, 2.0, 2.0, 4.0];
        assert!(inverse_2x2(&m).is_err());
    }

    #[test]
    fn cramer_test() {
        let m = [2.0, 1.0, 5.0, 3.0];
        let b = [4.0, 7.0];
        let x = cramer_rule_2x2(&m, &b).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-9);
        assert!((x[1] - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn trace_test() {
        assert_eq!(trace_2x2(&[1.0, 2.0, 3.0, 4.0]), 5.0);
        assert_eq!(trace_3x3(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]), 15.0);
    }

    #[test]
    fn rank_test() {
        assert_eq!(rank_2x2(&[1.0, 0.0, 0.0, 1.0]), 2);
        assert_eq!(rank_2x2(&[1.0, 2.0, 2.0, 4.0]), 1);
        assert_eq!(rank_2x2(&[0.0, 0.0, 0.0, 0.0]), 0);
    }

    #[test]
    fn eigenvalues_test() {
        let m = [2.0, 1.0, 1.0, 2.0];
        let mut eigs = eigenvalues_2x2(&m);
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(eigs, vec![1.0, 3.0]);
    }

    #[test]
    fn identity_test() {
        let i = identity_2x2();
        let m = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mat2_mul(&i, &m), m);
    }
}
