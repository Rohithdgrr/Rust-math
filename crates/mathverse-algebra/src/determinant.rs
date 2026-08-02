//! 2x2 and 3x3 determinants, matrix inverses, and Cramer's rule.

use crate::{AlgebraError, Result, TOL};

/// Determinant of a 2x2 matrix `[[a, b], [c, d]]`: `ad - bc`.
///
/// ```
/// # use mathverse_algebra::determinant::det_2x2;
/// assert_eq!(det_2x2(1.0, 2.0, 3.0, 4.0), -2.0);
/// ```
#[inline]
#[must_use]
pub fn det_2x2(a: f64, b: f64, c: f64, d: f64) -> f64 {
    a * d - b * c
}

/// Determinant of a 3x3 matrix (row-major):
/// `a(ei - fh) - b(di - fg) + c(dh - eg)`.
///
/// ```
/// # use mathverse_algebra::determinant::det_3x3;
/// assert_eq!(det_3x3(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0), 1.0);
/// ```
#[inline]
#[must_use]
pub fn det_3x3(
    a: f64, b: f64, c: f64,
    d: f64, e: f64, f: f64,
    g: f64, h: f64, i: f64,
) -> f64 {
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Inverse of a 2x2 matrix. Returns `None` if singular.
///
/// ```
/// # use mathverse_algebra::determinant::inverse_2x2;
/// let inv = inverse_2x2(4.0, 7.0, 2.0, 6.0).unwrap();
/// assert!((inv[0][0] - 0.6).abs() < 1e-12);
/// ```
#[must_use]
pub fn inverse_2x2(a: f64, b: f64, c: f64, d: f64) -> Result<[[f64; 2]; 2]> {
    let det = det_2x2(a, b, c, d);
    if det.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    let inv_det = 1.0 / det;
    Ok([[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]])
}

/// Inverse of a 3x3 matrix (row-major). Returns `None` if singular.
#[must_use]
pub fn inverse_3x3(
    a: f64, b: f64, c: f64,
    d: f64, e: f64, f: f64,
    g: f64, h: f64, i: f64,
) -> Result<[[f64; 3]; 3]> {
    let det = det_3x3(a, b, c, d, e, f, g, h, i);
    if det.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    let inv_det = 1.0 / det;
    Ok([
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ])
}

/// Cramer's rule for a 2x2 system:
/// `a*x + b*y = e`, `c*x + d*y = f`.
///
/// Returns `Some((x, y))` or `None` if the system is singular.
///
/// ```
/// # use mathverse_algebra::determinant::cramer_rule_2x2;
/// let sol = cramer_rule_2x2(1.0, 1.0, 0.0, 1.0, 3.0, 5.0).unwrap();
/// assert!((sol.0 - (-2.0)).abs() < 1e-12);
/// assert!((sol.1 - 5.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn cramer_rule_2x2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Result<(f64, f64)> {
    let det = det_2x2(a, b, c, d);
    if det.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    let x = det_2x2(e, b, f, d) / det;
    let y = det_2x2(a, e, c, f) / det;
    Ok((x, y))
}

/// Cramer's rule for a 3x3 system.
///
/// `a*x + b*y + c*z = j`, `d*x + e*y + f*z = k`, `g*x + h*y + i*z = l`.
///
/// Returns `Some((x, y, z))` or `None` if singular.
#[must_use]
pub fn cramer_rule_3x3(
    a: f64, b: f64, c: f64,
    d: f64, e: f64, f: f64,
    g: f64, h: f64, i: f64,
    j: f64, k: f64, l: f64,
) -> Result<(f64, f64, f64)> {
    let det = det_3x3(a, b, c, d, e, f, g, h, i);
    if det.abs() < TOL {
        return Err(AlgebraError::Singular);
    }
    let x = det_3x3(j, b, c, k, e, f, l, h, i) / det;
    let y = det_3x3(a, j, c, d, k, f, g, l, i) / det;
    let z = det_3x3(a, b, j, d, e, k, g, h, l) / det;
    Ok((x, y, z))
}

/// Trace of a 2x2 matrix: `a + d`.
#[inline]
#[must_use]
pub fn trace_2x2(a: f64, _b: f64, _c: f64, d: f64) -> f64 {
    a + d
}

/// Trace of a 3x3 matrix: `a + e + i`.
#[inline]
#[must_use]
pub fn trace_3x3(a: f64, _b: f64, _c: f64, _d: f64, e: f64, _f: f64, _g: f64, _h: f64, i: f64) -> f64 {
    a + e + i
}

/// Rank of a 2x2 matrix (0, 1, or 2).
#[must_use]
pub fn rank_2x2(a: f64, b: f64, c: f64, d: f64) -> u8 {
    if det_2x2(a, b, c, d).abs() > TOL {
        2
    } else if a.abs() > TOL || b.abs() > TOL || c.abs() > TOL || d.abs() > TOL {
        1
    } else {
        0
    }
}

/// Rank of a 3x3 matrix (0, 1, 2, or 3).
#[must_use]
pub fn rank_3x3(
    a: f64, b: f64, c: f64,
    d: f64, e: f64, f: f64,
    g: f64, h: f64, i: f64,
) -> u8 {
    if det_3x3(a, b, c, d, e, f, g, h, i).abs() > TOL {
        return 3;
    }
    // Check all 2x2 minors
    let minors = [
        det_2x2(a, b, d, e),
        det_2x2(a, c, d, f),
        det_2x2(b, c, e, f),
        det_2x2(a, b, g, h),
        det_2x2(a, c, g, i),
        det_2x2(b, c, h, i),
        det_2x2(d, e, g, h),
        det_2x2(d, f, g, i),
        det_2x2(e, f, h, i),
    ];
    if minors.iter().any(|m| m.abs() > TOL) {
        2
    } else if [a, b, c, d, e, f, g, h, i].iter().any(|x| x.abs() > TOL) {
        1
    } else {
        0
    }
}

// Legacy aliases for backward compatibility
#[deprecated(note = "use det_2x2")]
pub use det_2x2 as det2;
#[deprecated(note = "use det_3x3")]
pub use det_3x3 as det3;
#[deprecated(note = "use inverse_2x2")]
pub use inverse_2x2 as inverse2;
#[deprecated(note = "use inverse_3x3")]
pub use inverse_3x3 as inverse3;

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn det_2x2_test() {
        assert_eq!(det_2x2(1.0, 2.0, 3.0, 4.0), -2.0);
    }

    #[test]
    fn det_3x3_test() {
        assert_eq!(det_3x3(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0), 1.0);
    }

    #[test]
    fn inv_2x2_test() {
        let inv = inverse_2x2(4.0, 7.0, 2.0, 6.0).unwrap();
        assert!(approx(inv[0][0], 0.6));
        assert!(approx(inv[0][1], -0.7));
        assert!(approx(inv[1][0], -0.2));
        assert!(approx(inv[1][1], 0.4));
    }

    #[test]
    fn inv_2x2_singular() {
        assert_eq!(inverse_2x2(1.0, 2.0, 2.0, 4.0), Err(AlgebraError::Singular));
    }

    #[test]
    fn inv_3x3_test() {
        let inv = inverse_3x3(2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0).unwrap();
        assert!(approx(inv[0][0], 0.5));
        assert!(approx(inv[1][1], 0.25));
        assert!(approx(inv[2][2], 0.125));
    }

    #[test]
    fn cramer2() {
        let sol = cramer_rule_2x2(1.0, 1.0, 0.0, 1.0, 3.0, 5.0).unwrap();
        assert!(approx(sol.0, -2.0));
        assert!(approx(sol.1, 5.0));
    }

    #[test]
    fn cramer3() {
        let sol = cramer_rule_3x3(
            1.0, 1.0, 1.0,
            2.0, 3.0, 1.0,
            1.0, 2.0, 3.0,
            6.0, 11.0, 14.0,
        ).unwrap();
        assert!(approx(sol.0, 1.0));
        assert!(approx(sol.1, 2.0));
        assert!(approx(sol.2, 3.0));
    }

    #[test]
    fn cramer_singular() {
        assert_eq!(cramer_rule_2x2(1.0, 2.0, 2.0, 4.0, 5.0, 10.0), Err(AlgebraError::Singular));
    }

    #[test]
    fn trace_test() {
        assert_eq!(trace_2x2(1.0, 2.0, 3.0, 4.0), 5.0);
        assert_eq!(trace_3x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), 15.0);
    }

    #[test]
    fn rank_test() {
        assert_eq!(rank_2x2(1.0, 0.0, 0.0, 1.0), 2);
        assert_eq!(rank_2x2(1.0, 2.0, 2.0, 4.0), 1);
        assert_eq!(rank_2x2(0.0, 0.0, 0.0, 0.0), 0);
        assert_eq!(rank_3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), 3);
        assert_eq!(rank_3x3(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0), 1);
    }
}
