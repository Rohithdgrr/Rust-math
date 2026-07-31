//! Polynomial equation solvers and discriminant/Vieta helpers.
//!
//! All solvers return **real** roots only. Polynomials are passed as
//! coefficient slices **lowest-degree first** (matching [`Polynomial`](crate::Polynomial)).

use crate::polynomial::Polynomial;

/// Tolerance for treating a coefficient as zero.
const TOL: f64 = 1e-12;

/// Dispatch solver based on polynomial degree.
///
/// Returns the real roots of `coeffs` (lowest-degree first). Degree 0/1
/// return `[]` / one root; degrees 2–4 use the closed-form solvers;
/// degree > 4 returns `[]`.
pub fn solve(coeffs: &[f64]) -> Vec<f64> {
    let p = Polynomial::from_coeffs(coeffs);
    solve_polynomial(&p)
}

/// Dispatch solver for a [`Polynomial`].
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

/// Solve `ax + b = 0`. Returns `Some(root)` when `a ≠ 0`, else `None`.
pub fn solve_linear(a: f64, b: f64) -> Option<f64> {
    if a.abs() < TOL {
        None
    } else {
        Some(-b / a)
    }
}

/// Discriminant of `ax² + bx + c`: `b² − 4ac`.
pub fn quadratic_discriminant(a: f64, b: f64, c: f64) -> f64 {
    b * b - 4.0 * a * c
}

/// Solve `ax² + bx + c = 0`, returning real roots (0, 1, or 2 values).
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

/// Discriminant of the cubic `ax³ + bx² + cx + d`.
pub fn cubic_discriminant(a: f64, b: f64, c: f64, d: f64) -> f64 {
    18.0 * a * b * c * d
        - 4.0 * b * b * b * d
        + b * b * c * c
        - 4.0 * a * c * c * c
        - 27.0 * a * a * d * d
}

/// Solve `ax³ + bx² + cx + d = 0` using Cardano's method.
///
/// Returns real roots (1 or 3). The *casus irreducibilis* (three real roots,
/// discriminant < 0) is handled via the trigonometric form.
pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    if a.abs() < TOL {
        return solve_quadratic(b, c, d);
    }
    let bn = b / a;
    let cn = c / a;
    let dn = d / a;
    // Depressed cubic: t³ + pt + q = 0  where x = t - b/(3a)
    let p = cn - bn * bn / 3.0;
    let q = (2.0 * bn * bn * bn - 9.0 * bn * cn + 27.0 * dn) / 27.0;
    let disc = (q * q / 4.0) + (p * p * p / 27.0);

    let shift = -bn / 3.0;

    if disc > TOL {
        // One real root.
        let u = (-q / 2.0 + disc.sqrt()).cbrt();
        let v = (-q / 2.0 - disc.sqrt()).cbrt();
        vec![shift + u + v]
    } else if disc.abs() < TOL {
        // Multiple root.
        if q.abs() < TOL {
            // Triple root.
            vec![shift]
        } else {
            let u = -q.cbrt() / 2.0;
            let v = if p.abs() < TOL { 0.0 } else { 3.0 * q / p };
            vec![shift + u - v / 3.0, shift + 2.0 * u]
        }
    } else {
        // Three real roots — trigonometric form.
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

/// Discriminant of the quartic `ax⁴ + bx³ + cx² + dx + e`.
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

/// Solve `ax⁴ + bx³ + cx² + dx + e = 0` via Ferrari's method.
///
/// Returns real roots (0, 2, or 4).
pub fn solve_quartic(a: f64, b: f64, c: f64, d: f64, e: f64) -> Vec<f64> {
    if a.abs() < TOL {
        return solve_cubic(b, c, d, e);
    }
    // Normalize: x⁴ + Bx³ + Cx² + Dx + E = 0
    let b = b / a;
    let c = c / a;
    let d = d / a;
    let e = e / a;

    // Depressed quartic: y⁴ + py² + qy + r = 0,  x = y - B/4
    let p = c - 3.0 * b * b / 8.0;
    let q = d - b * c / 2.0 + b * b * b / 8.0;
    let r = e - b * d / 4.0 + b * b * c / 16.0 - 3.0 * b * b * b * b / 256.0;
    let shift = -b / 4.0;

    // Resolvent cubic: z³ - (p²/8) z - r = 0  ... using the standard form
    // m³ - (p/2)m² - r m + (rp/2 - q²/8) = 0 is more stable; use Ferrari's.
    // We solve the resolvent cubic for m:
    // m³ - (p/2) m² - r m + (rp/2 - q²/8) = 0
    let rc = solve_cubic(1.0, -p / 2.0, -r, r * p / 2.0 - q * q / 8.0);
    // Pick the largest real root (most stable for Ferrari).
    let m = rc.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut roots = Vec::new();

    // Factor the quartic into two quadratics.
    let disc1 = m * m - 4.0 * r;
    if disc1 >= -TOL {
        let sq = if disc1 < 0.0 { 0.0 } else { disc1.sqrt() };
        let a1 = m;
        let b1 = q / 2.0;
        // (y² + a1)(y² + b1 y + ...) — use the standard Ferrari split.
        let q2 = q / 2.0;
        let sqrt_m = if m < 0.0 && m.abs() < TOL { 0.0 } else { m };
        // Quadratic 1: y² + (q/(2√m)) y + (m/2 - p/2 - q/(2√m))  -- only valid m>0
        if m.abs() > TOL {
            let coeff = q / (2.0 * m);
            let const1 = m / 2.0 - p / 2.0 - coeff;
            let c1 = m / 2.0 - p / 2.0 + coeff;
            let r1 = solve_quadratic(1.0, coeff, const1);
            let r2 = solve_quadratic(1.0, -coeff, c1);
            roots.extend(r1);
            roots.extend(r2);
        } else {
            // m ≈ 0: depressed quartic y⁴ + py² + qy + r = 0 with resolvent m=0
            // Fall back to solving the biquadratic-like form.
            let r1 = solve_quadratic(1.0, q.sqrt(), p);
            let r2 = solve_quadratic(1.0, -q.sqrt(), p);
            roots.extend(r1);
            roots.extend(r2);
        }
    }

    // Apply the shift and deduplicate.
    let mut real_roots: Vec<f64> = roots.iter().map(|&y| shift + y).collect();
    real_roots.sort_by(|x, y| x.partial_cmp(y).unwrap());
    real_roots.dedup_by(|x, y| (*x - *y).abs() < TOL);
    real_roots
}

// ---------------------------------------------------------------------------
// Vieta's formulas
// ---------------------------------------------------------------------------

/// Sum of roots of `ax² + bx + c`: `−b/a`.
pub fn vieta_quadratic_sum(a: f64, b: f64) -> f64 {
    -b / a
}

/// Product of roots of `ax² + bx + c`: `c/a`.
pub fn vieta_quadratic_product(a: f64, c: f64) -> f64 {
    c / a
}

/// Sum of roots of `ax³ + bx² + cx + d`: `−b/a`.
pub fn vieta_cubic_sum(a: f64, b: f64) -> f64 {
    -b / a
}

/// Sum of pairwise products of roots of `ax³ + bx² + cx + d`: `c/a`.
pub fn vieta_cubic_pairwise(a: f64, c: f64) -> f64 {
    c / a
}

/// Product of roots of `ax³ + bx² + cx + d`: `−d/a`.
pub fn vieta_cubic_product(a: f64, d: f64) -> f64 {
    -d / a
}

/// Sum of roots of degree-`n` polynomial `aₙxⁿ + … + a₀`: `−aₙ₋₁/aₙ`.
pub fn vieta_sum(coeffs: &[f64]) -> f64 {
    let n = coeffs.len().saturating_sub(1);
    if n == 0 {
        return 0.0;
    }
    -coeffs[n - 1] / coeffs[n]
}

/// Product of roots of degree-`n` polynomial: `(−1)ⁿ · a₀/aₙ`.
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
        let r = solve_quadratic(1.0, -5.0, 6.0); // (x-2)(x-3)
        assert_eq!(r.len(), 2);
        assert!(approx(r[0], 2.0) && approx(r[1], 3.0));
    }

    #[test]
    fn quadratic_one_root() {
        let r = solve_quadratic(1.0, -4.0, 4.0); // (x-2)²
        assert_eq!(r.len(), 1);
        assert!(approx(r[0], 2.0));
    }

    #[test]
    fn quadratic_no_real() {
        let r = solve_quadratic(1.0, 0.0, 1.0); // x² + 1
        assert!(r.is_empty());
    }

    #[test]
    fn cubic_one_root() {
        let r = solve_cubic(1.0, 0.0, 0.0, -8.0); // x³ - 8
        assert_eq!(r.len(), 1);
        assert!(approx(r[0], 2.0));
    }

    #[test]
    fn cubic_three_roots() {
        let r = solve_cubic(1.0, -6.0, 11.0, -6.0); // (x-1)(x-2)(x-3)
        assert_eq!(r.len(), 3);
        assert!(approx(r[0], 1.0));
        assert!(approx(r[1], 2.0));
        assert!(approx(r[2], 3.0));
    }

    #[test]
    fn quartic_four_roots() {
        let r = solve_quartic(1.0, -10.0, 35.0, -50.0, 24.0); // (x-1)(x-2)(x-3)(x-4)
        assert_eq!(r.len(), 4);
        for (i, &v) in r.iter().enumerate() {
            assert!(approx(v, (i + 1) as f64));
        }
    }

    #[test]
    fn quartic_biquadratic() {
        let r = solve_quartic(1.0, 0.0, -5.0, 0.0, 4.0); // x⁴ - 5x² + 4 = (x²-1)(x²-4)
        assert_eq!(r.len(), 4);
        assert!(approx(r[0], -2.0));
        assert!(approx(r[1], -1.0));
        assert!(approx(r[2], 1.0));
        assert!(approx(r[3], 2.0));
    }

    #[test]
    fn discriminants() {
        assert!(approx(quadratic_discriminant(1.0, 0.0, 1.0), -4.0));
        assert!(approx(cubic_discriminant(1.0, -6.0, 11.0, -6.0), 4.0)); // 3 distinct real roots
    }

    #[test]
    fn vieta_quadratic() {
        // x² - 5x + 6 = 0, roots 2 and 3
        assert!(approx(vieta_quadratic_sum(1.0, -5.0), 5.0));
        assert!(approx(vieta_quadratic_product(1.0, 6.0), 6.0));
    }

    #[test]
    fn vieta_cubic() {
        // x³ - 6x² + 11x - 6, roots 1,2,3
        assert!(approx(vieta_cubic_sum(1.0, -6.0), 6.0));
        assert!(approx(vieta_cubic_pairwise(1.0, 11.0), 11.0));
        assert!(approx(vieta_cubic_product(1.0, -6.0), 6.0));
    }

    #[test]
    fn vieta_general() {
        // x³ - 6x² + 11x - 6
        let coeffs = [-6.0, 11.0, -6.0, 1.0];
        assert!(approx(vieta_sum(&coeffs), 6.0));
        assert!(approx(vieta_product(&coeffs), 6.0));
    }

    #[test]
    fn solve_dispatch() {
        let r = solve(&[6.0, -5.0, 1.0]); // x² - 5x + 6
        assert_eq!(r.len(), 2);
        assert!(approx(r[0], 2.0));
        assert!(approx(r[1], 3.0));
    }
}
