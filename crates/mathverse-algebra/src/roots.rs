//! # Roots
//!
//! Polynomial equation solvers and discriminant/Vieta helpers.
//!
//! All solvers return **real** roots only. Polynomials are passed as
//! coefficient slices **lowest-degree first** (matching [`Polynomial`](crate::Polynomial)).
//!
//! ## Supported Degrees
//!
//! | Degree | Solver | Method |
//! |---|---|---|
//! | 1 | [`solve_linear`] | Direct division |
//! | 2 | [`solve_quadratic`] | Quadratic formula |
//! | 3 | [`solve_cubic`] | Cardano's method (trigonometric for 3 real roots) |
//! | 4 | [`solve_quartic`] | Ferrari's method |
//! | >4 | `solve` returns `[]` | — |
//!
//! ## Examples
//!
//! ```rust
//! use mathverse_algebra::roots::{solve_quadratic, solve_cubic};
//!
//! // x^2 - 5x + 6 = 0 → x = 2, 3
//! let r = solve_quadratic(1.0, -5.0, 6.0);
//! assert_eq!(r, vec![2.0, 3.0]);
//!
//! // x^3 - 6x^2 + 11x - 6 = 0 → x = 1, 2, 3
//! let r = solve_cubic(1.0, -6.0, 11.0, -6.0);
//! assert_eq!(r, vec![1.0, 2.0, 3.0]);
//! ```

use crate::polynomial::Polynomial;
use crate::TOL;

/// Dispatch solver based on polynomial degree.
///
/// Returns the real roots of `coeffs` (lowest-degree first). Degree 0/1
/// return `[]` / one root; degrees 2–4 use the closed-form solvers;
/// degree > 4 returns `[]`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::roots::solve;
///
/// let r = solve(&[6.0, -5.0, 1.0]); // x^2 - 5x + 6
/// assert_eq!(r, vec![2.0, 3.0]);
/// ```
#[must_use]
pub fn solve(coeffs: &[f64]) -> Vec<f64> {
    let p = Polynomial::from_coeffs(coeffs);
    solve_polynomial(&p)
}

/// Dispatch solver for a [`Polynomial`].
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::{Polynomial, roots::solve_polynomial};
///
/// let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]);
/// let r = solve_polynomial(&p);
/// assert_eq!(r, vec![2.0, 3.0]);
/// ```
#[must_use]
pub fn solve_polynomial(p: &Polynomial) -> Vec<f64> {
    let c = p.coeffs();
    let n = c.len().saturating_sub(1);
    match n {
        0 => vec![],
        1 => solve_linear(c[1], c[0])
            .into_iter()
            .collect(),
        2 => solve_quadratic(c[2], c[1], c[0]),
        3 => solve_cubic(c[3], c[2], c[1], c[0]),
        4 => solve_quartic(c[4], c[3], c[2], c[1], c[0]),
        _ => vec![],
    }
}

/// Solve `ax + b = 0`. Returns `Some(root)` when `a != 0`, else `None`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::roots::solve_linear;
///
/// assert_eq!(solve_linear(2.0, -6.0), Some(3.0));
/// assert_eq!(solve_linear(0.0, 1.0), None);
/// ```
#[inline]
#[must_use]
pub fn solve_linear(a: f64, b: f64) -> Option<f64> {
    if a.abs() < TOL {
        None
    } else {
        Some(-b / a)
    }
}

/// Discriminant of `ax^2 + bx + c`: `b^2 - 4ac`.
///
/// Positive → two real roots. Zero → one repeated root. Negative → no real roots.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::roots::quadratic_discriminant;
///
/// assert_eq!(quadratic_discriminant(1.0, -5.0, 6.0), 1.0);  // two roots
/// assert_eq!(quadratic_discriminant(1.0, -4.0, 4.0), 0.0);  // one root
/// assert_eq!(quadratic_discriminant(1.0, 0.0, 1.0), -4.0);  // no real roots
/// ```
#[inline]
#[must_use]
pub fn quadratic_discriminant(a: f64, b: f64, c: f64) -> f64 {
    b * b - 4.0 * a * c
}

/// Solve `ax^2 + bx + c = 0`, returning real roots (0, 1, or 2 values).
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::roots::solve_quadratic;
///
/// let r = solve_quadratic(1.0, -5.0, 6.0); // (x-2)(x-3)
/// assert_eq!(r, vec![2.0, 3.0]);
/// ```
#[must_use]
pub fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() < TOL {
        return solve_linear(b, c).into_iter().collect();
    }
    let disc = quadratic_discriminant(a, b, c);
    if disc < -TOL {
        return vec![];
    }
    if disc.abs() < TOL {
        return vec![-b / (2.0 * a)];
    }
    let sq = disc.sqrt();
    let mut roots = vec![(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)];
    roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
    roots
}

/// Discriminant of the cubic `ax^3 + bx^2 + cx + d`.
///
/// Positive → one real root. Zero → repeated root. Negative → three real roots.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::roots::cubic_discriminant;
///
/// // (x-1)(x-2)(x-3): three distinct real roots → disc > 0
/// assert!(cubic_discriminant(1.0, -6.0, 11.0, -6.0) > 0.0);
/// ```
#[inline]
#[must_use]
pub fn cubic_discriminant(a: f64, b: f64, c: f64, d: f64) -> f64 {
    18.0 * a * b * c * d
        - 4.0 * b * b * b * d
        + b * b * c * c
        - 4.0 * a * c * c * c
        - 27.0 * a * a * d * d
}

/// Solve `ax^3 + bx^2 + cx + d = 0` using Cardano's method.
///
/// Returns real roots (1 or 3). The *casus irreducibilis* (three real roots,
/// discriminant < 0) is handled via the trigonometric form.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::roots::solve_cubic;
///
/// // x^3 - 8 = 0 → x = 2
/// let r = solve_cubic(1.0, 0.0, 0.0, -8.0);
/// assert_eq!(r, vec![2.0]);
///
/// // (x-1)(x-2)(x-3) → x = 1, 2, 3
/// let r = solve_cubic(1.0, -6.0, 11.0, -6.0);
/// assert_eq!(r, vec![1.0, 2.0, 3.0]);
/// ```
#[must_use]
pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    if a.abs() < TOL {
        return solve_quadratic(b, c, d);
    }
    let bn = b / a;
    let cn = c / a;
    let dn = d / a;
    let p = cn - bn * bn / 3.0;
    let q = (2.0 * bn * bn * bn - 9.0 * bn * cn + 27.0 * dn) / 27.0;
    let disc = (q * q / 4.0) + (p * p * p / 27.0);
    let shift = -bn / 3.0;

    if disc > TOL {
        let u = (-q / 2.0 + disc.sqrt()).cbrt();
        let v = (-q / 2.0 - disc.sqrt()).cbrt();
        vec![shift + u + v]
    } else if disc.abs() < TOL {
        if q.abs() < TOL {
            vec![shift]
        } else {
            let u = -q.cbrt() / 2.0;
            let v = if p.abs() < TOL { 0.0 } else { 3.0 * q / p };
            vec![shift + u - v / 3.0, shift + 2.0 * u]
        }
    } else {
        let r = (-p * p * p / 27.0).sqrt();
        let phi = (-q / (2.0 * r)).clamp(-1.0, 1.0).acos();
        let m = 2.0 * r.cbrt();
        let t1 = m * (phi / 3.0).cos();
        let t2 = m * ((phi + 2.0 * std::f64::consts::PI) / 3.0).cos();
        let t3 = m * ((phi + 4.0 * std::f64::consts::PI) / 3.0).cos();
        let mut roots = vec![shift + t1, shift + t2, shift + t3];
        roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
        roots
    }
}

/// Discriminant of the quartic `ax^4 + bx^3 + cx^2 + dx + e`.
#[inline]
#[must_use]
pub fn quartic_discriminant(a: f64, b: f64, c: f64, d: f64, e: f64) -> f64 {
    256.0 * a * a * a * e * e * e
        - 192.0 * a * a * b * d * e * e
        - 128.0 * a * a * c * c * e * e
        + 144.0 * a * a * c * d * d * e
        - 27.0 * a * a * d * d * d * d
        + 144.0 * a * b * b * c * e * e
        - 6.0 * a * b * b * d * d * e
        - 80.0 * a * b * c * c * d * e
        + 18.0 * a * b * c * d * d * d
        + 16.0 * a * c * c * c * c * e
        - 4.0 * a * c * c * c * d * d
        - 27.0 * b * b * b * b * e * e
        + 18.0 * b * b * b * c * d * e
        - 4.0 * b * b * b * d * d * d
        - 4.0 * b * b * c * c * c * e
        + b * b * c * c * d * d
}

/// Solve `ax^4 + bx^3 + cx^2 + dx + e = 0` via Ferrari's method.
///
/// Returns real roots (0, 2, or 4).
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::roots::solve_quartic;
///
/// // (x-1)(x-2)(x-3)(x-4)
/// let r = solve_quartic(1.0, -10.0, 35.0, -50.0, 24.0);
/// assert_eq!(r, vec![1.0, 2.0, 3.0, 4.0]);
/// ```
#[must_use]
pub fn solve_quartic(a: f64, b: f64, c: f64, d: f64, e: f64) -> Vec<f64> {
    if a.abs() < TOL {
        return solve_cubic(b, c, d, e);
    }
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let e = e / a;

    let p = c - 3.0 * b * b / 8.0;
    let q = d - b * c / 2.0 + b * b * b / 8.0;
    let r = e - b * d / 4.0 + b * b * c / 16.0 - 3.0 * b * b * b * b / 256.0;
    let shift = -b / 4.0;

    let rc = solve_cubic(1.0, 2.0 * p, p * p - 4.0 * r, -q * q);

    let mut roots = Vec::new();

    for &u in &rc {
        if u < -TOL {
            continue;
        }
        let u = u.max(0.0);
        if u < TOL {
            if q.abs() < TOL {
                let zs = solve_quadratic(1.0, p, r);
                for &z in &zs {
                    if z >= -TOL {
                        let y = z.max(0.0).sqrt();
                        roots.push(y);
                        if y > TOL {
                            roots.push(-y);
                        }
                    }
                }
            }
            break;
        }

        let a_coeff = u.sqrt();
        let b_coeff = (p + u - q / a_coeff) / 2.0;
        let c_coeff = (p + u + q / a_coeff) / 2.0;

        roots.extend(solve_quadratic(1.0, a_coeff, b_coeff));
        roots.extend(solve_quadratic(1.0, -a_coeff, c_coeff));
        break;
    }

    let mut real_roots: Vec<f64> = roots.iter().map(|&y| shift + y).collect();
    real_roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
    real_roots.dedup_by(|x, y| (*x - *y).abs() < TOL);
    real_roots
}

// ---------------------------------------------------------------------------
// Vieta's formulas
// ---------------------------------------------------------------------------

/// Sum of roots of `ax^2 + bx + c`: `-b/a`.
#[inline]
#[must_use]
pub fn vieta_quadratic_sum(a: f64, b: f64) -> f64 {
    -b / a
}

/// Product of roots of `ax^2 + bx + c`: `c/a`.
#[inline]
#[must_use]
pub fn vieta_quadratic_product(a: f64, c: f64) -> f64 {
    c / a
}

/// Sum of roots of `ax^3 + bx^2 + cx + d`: `-b/a`.
#[inline]
#[must_use]
pub fn vieta_cubic_sum(a: f64, b: f64) -> f64 {
    -b / a
}

/// Sum of pairwise products of roots of `ax^3 + bx^2 + cx + d`: `c/a`.
#[inline]
#[must_use]
pub fn vieta_cubic_pairwise(a: f64, c: f64) -> f64 {
    c / a
}

/// Product of roots of `ax^3 + bx^2 + cx + d`: `-d/a`.
#[inline]
#[must_use]
pub fn vieta_cubic_product(a: f64, d: f64) -> f64 {
    -d / a
}

/// Sum of roots of degree-`n` polynomial `a_n x^n + ... + a_0`: `-a_{n-1}/a_n`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::roots::vieta_sum;
///
/// // x^2 - 5x + 6: roots 2, 3 → sum = 5
/// assert_eq!(vieta_sum(&[6.0, -5.0, 1.0]), 5.0);
/// ```
#[must_use]
pub fn vieta_sum(coeffs: &[f64]) -> f64 {
    let n = coeffs.len().saturating_sub(1);
    if n == 0 {
        return 0.0;
    }
    -coeffs[n - 1] / coeffs[n]
}

/// Product of roots of degree-`n` polynomial: `(-1)^n * a_0/a_n`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::roots::vieta_product;
///
/// // x^2 - 5x + 6: roots 2, 3 → product = 6
/// assert_eq!(vieta_product(&[6.0, -5.0, 1.0]), 6.0);
/// ```
#[must_use]
pub fn vieta_product(coeffs: &[f64]) -> f64 {
    let n = coeffs.len().saturating_sub(1);
    if n == 0 {
        return 0.0;
    }
    let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
    sign * coeffs[0] / coeffs[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn linear() {
        assert_eq!(solve_linear(2.0, -6.0), Some(3.0));
        assert_eq!(solve_linear(0.0, 1.0), None);
    }

    #[test]
    fn quadratic_two_roots() {
        let r = solve_quadratic(1.0, -5.0, 6.0);
        assert_eq!(r.len(), 2);
        assert!(approx(r[0], 2.0) && approx(r[1], 3.0));
    }

    #[test]
    fn quadratic_one_root() {
        let r = solve_quadratic(1.0, -4.0, 4.0);
        assert_eq!(r.len(), 1);
        assert!(approx(r[0], 2.0));
    }

    #[test]
    fn quadratic_no_real() {
        let r = solve_quadratic(1.0, 0.0, 1.0);
        assert!(r.is_empty());
    }

    #[test]
    fn cubic_one_root() {
        let r = solve_cubic(1.0, 0.0, 0.0, -8.0);
        assert_eq!(r.len(), 1);
        assert!(approx(r[0], 2.0));
    }

    #[test]
    fn cubic_three_roots() {
        let r = solve_cubic(1.0, -6.0, 11.0, -6.0);
        assert_eq!(r.len(), 3);
        assert!(approx(r[0], 1.0));
        assert!(approx(r[1], 2.0));
        assert!(approx(r[2], 3.0));
    }

    #[test]
    fn quartic_four_roots() {
        let r = solve_quartic(1.0, -10.0, 35.0, -50.0, 24.0);
        assert_eq!(r.len(), 4);
        for (i, &v) in r.iter().enumerate() {
            assert!(approx(v, (i + 1) as f64));
        }
    }

    #[test]
    fn quartic_biquadratic() {
        let r = solve_quartic(1.0, 0.0, -5.0, 0.0, 4.0);
        assert_eq!(r.len(), 4);
        assert!(approx(r[0], -2.0));
        assert!(approx(r[1], -1.0));
        assert!(approx(r[2], 1.0));
        assert!(approx(r[3], 2.0));
    }

    #[test]
    fn discriminants() {
        assert!(approx(quadratic_discriminant(1.0, 0.0, 1.0), -4.0));
        assert!(approx(cubic_discriminant(1.0, -6.0, 11.0, -6.0), 4.0));
    }

    #[test]
    fn vieta_quadratic() {
        assert!(approx(vieta_quadratic_sum(1.0, -5.0), 5.0));
        assert!(approx(vieta_quadratic_product(1.0, 6.0), 6.0));
    }

    #[test]
    fn vieta_cubic() {
        assert!(approx(vieta_cubic_sum(1.0, -6.0), 6.0));
        assert!(approx(vieta_cubic_pairwise(1.0, 11.0), 11.0));
        assert!(approx(vieta_cubic_product(1.0, -6.0), 6.0));
    }

    #[test]
    fn vieta_general() {
        let coeffs = &[-6.0, 11.0, -6.0, 1.0];
        assert!(approx(vieta_sum(coeffs), 6.0));
        assert!(approx(vieta_product(coeffs), 6.0));
    }

    #[test]
    fn solve_dispatch() {
        let r = solve(&[6.0, -5.0, 1.0]);
        assert_eq!(r.len(), 2);
        assert!(approx(r[0], 2.0));
        assert!(approx(r[1], 3.0));
    }
}
