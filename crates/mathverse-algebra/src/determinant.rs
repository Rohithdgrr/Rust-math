//! 2×2 and 3×3 determinants, matrix inverses, and Cramer's rule.
//!
//! These are the closed-form linear-algebra primitives that belong alongside
//! polynomial algebra (e.g. resultants, system solving).

const TOL: f64 = 1e-12;

/// Determinant of a 2×2 matrix `[[a, b], [c, d]]`: `ad − bc`.
///
/// ```
/// # use mathverse_algebra::determinant::det2;
/// assert_eq!(det2(1.0, 2.0, 3.0, 4.0), -2.0);
/// ```
pub fn det2(a: f64, b: f64, c: f64, d: f64) -> f64 {
    a * d - b * c
}

/// Determinant of a 3×3 matrix (row-major):
/// `a(ei − fh) − b(di − fg) + c(dh − eg)`.
///
/// ```
/// # use mathverse_algebra::determinant::det3;
/// assert_eq!(det3(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0), -24.0);
/// ```
pub fn det3(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64, i: f64) -> f64 {
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Inverse of a 2×2 matrix. Returns `None` if singular.
///
/// ```
/// # use mathverse_algebra::determinant::inverse2;
/// let inv = inverse2(4.0, 7.0, 2.0, 6.0).unwrap();
/// assert!((inv[0][0] - 0.6).abs() < 1e-12);
/// ```
pub fn inverse2(a: f64, b: f64, c: f64, d: f64) -> Option<[[f64; 2]; 2]> {
    let det = det2(a, b, c, d);
    if det.abs() < TOL {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]])
}

/// Inverse of a 3×3 matrix (row-major). Returns `None` if singular.
pub fn inverse3(
    a: f64, b: f64, c: f64,
    d: f64, e: f64, f: f64,
    g: f64, h: f64, i: f64,
) -> Option<[[f64; 3]; 3]> {
    let det = det3(a, b, c, d, e, f, g, h, i);
    if det.abs() < TOL {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ])
}

/// Cramer's rule for a 2×2 system:
/// `a·x + b·y = e`, `c·x + d·y = f`.
///
/// Returns `Some((x, y))` or `None` if the system is singular.
///
/// ```
/// # use mathverse_algebra::determinant::cramers_rule_2x2;
/// let sol = cramers_rule_2x2(1.0, 1.0, 0.0, 1.0, 3.0, 5.0).unwrap();
/// assert!((sol.0 - (-2.0)).abs() < 1e-12);
/// assert!((sol.1 - 5.0).abs() < 1e-12);
/// ```
pub fn cramer_rule_2x2(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Option<(f64, f64)> {
    let det = det2(a, b, c, d);
    if det.abs() < TOL {
        return None;
    }
    let x = det2(e, b, f, d) / det;
    let y = det2(a, e, c, f) / det;
    Some((x, y))
}

/// Cramer's rule for a 3×3 system.
///
/// `a·x + b·y + c·z = j`, `d·x + e·y + f·z = k`, `g·x + h·y + i·z = l`.
///
/// Returns `Some((x, y, z))` or `None` if singular.
pub fn cramer_rule_3x3(
    a: f64, b: f64, c: f64,
    d: f64, e: f64, f: f64,
    g: f64, h: f64, i: f64,
    j: f64, k: f64, l: f64,
) -> Option<(f64, f64, f64)> {
    let det = det3(a, b, c, d, e, f, g, h, i);
    if det.abs() < TOL {
        return None;
    }
    let x = det3(j, b, c, k, e, f, l, h, i) / det;
    let y = det3(a, j, c, d, k, f, g, l, i) / det;
    let z = det3(a, b, j, d, e, k, g, h, l) / det;
    Some((x, y, z))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn det2x2() {
        assert_eq!(det2(1.0, 2.0, 3.0, 4.0), -2.0);
    }

    #[test]
    fn det3x3() {
        assert_eq!(det3(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0), -24.0);
    }

    #[test]
    fn inv2() {
        let inv = inverse2(4.0, 7.0, 2.0, 6.0).unwrap();
        assert!(approx(inv[0][0], 0.6));
        assert!(approx(inv[0][1], -0.7));
        assert!(approx(inv[1][0], -0.2));
        assert!(approx(inv[1][1], 0.4));
    }

    #[test]
    fn inv2_singular() {
        assert!(inverse2(1.0, 2.0, 2.0, 4.0).is_none());
    }

    #[test]
    fn inv3() {
        let inv = inverse3(2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0).unwrap();
        assert!(approx(inv[0][0], 0.5));
        assert!(approx(inv[1][1], 0.25));
        assert!(approx(inv[2][2], 0.125));
    }

    #[test]
    fn cramer2() {
        // x + y = 3, y = 5 → x = -2, y = 5
        let sol = cramer_rule_2x2(1.0, 1.0, 0.0, 1.0, 3.0, 5.0).unwrap();
        assert!(approx(sol.0, -2.0));
        assert!(approx(sol.1, 5.0));
    }

    #[test]
    fn cramer3() {
        // x + y + z = 6, 2x + 3y + z = 11, x + 2y + 3z = 13 → x=1, y=2, z=3
        let sol = cramer_rule_3x3(
            1.0, 1.0, 1.0,
            2.0, 3.0, 1.0,
            1.0, 2.0, 3.0,
            6.0, 11.0, 13.0,
        ).unwrap();
        assert!(approx(sol.0, 1.0));
        assert!(approx(sol.1, 2.0));
        assert!(approx(sol.2, 3.0));
    }

    #[test]
    fn cramer_singular() {
        assert!(cramer_rule_2x2(1.0, 2.0, 2.0, 4.0, 5.0, 10.0).is_none());
    }
}