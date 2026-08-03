//! # Interpolation
//!
//! Lagrange and Newton polynomial interpolation from data points.
//!
//! ## Examples
//!
//! ```rust
//! use mathverse_algebra::interpolate::{lagrange, newton};
//!
//! // Points (0,1), (1,3), (2,7) → p(x) = x^2 + x + 1
//! let xs = [0.0, 1.0, 2.0];
/// let ys = [1.0, 3.0, 7.0];
/// assert!((lagrange(&xs, &ys)(2.0) - 7.0).abs() < 1e-10);
/// assert!((newton(&xs, &ys)(2.0) - 7.0).abs() < 1e-10);
/// ```

use crate::polynomial::Polynomial;

/// Lagrange interpolation.
///
/// Returns a [`Polynomial`] that passes through every `(xs[i], ys[i])`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::interpolate::lagrange;
///
/// // (0,1), (1,2), (2,5)
/// let xs = [0.0, 1.0, 2.0];
/// let ys = [1.0, 2.0, 5.0];
/// let p = lagrange(&xs, &ys);
/// assert!((p.eval(0.0) - 1.0).abs() < 1e-10);
/// assert!((p.eval(1.0) - 2.0).abs() < 1e-10);
/// assert!((p.eval(2.0) - 5.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn lagrange(xs: &[f64], ys: &[f64]) -> Polynomial {
    let n = xs.len();
    let mut result = Polynomial::constant(0.0);
    for i in 0..n {
        let mut basis = Polynomial::constant(1.0);
        for j in 0..n {
            if i == j {
                continue;
            }
            let basis_factor = Polynomial::from_coeffs(&[-xs[j], 1.0]);
            basis = basis * basis_factor * Polynomial::constant(1.0 / (xs[i] - xs[j]));
        }
        result = result + basis * Polynomial::constant(ys[i]);
    }
    result
}

/// Divided differences table.
///
/// Returns a vector of divided differences `dd[k]` (the first row of each column),
/// which are the Newton coefficients.
#[must_use]
pub fn divided_differences(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut dd: Vec<Vec<f64>> = (0..n).map(|i| vec![ys[i]]).collect();
    for j in 1..n {
        for i in 0..n - j {
            let diff = dd[i + 1][j - 1] - dd[i][j - 1];
            let denom = xs[i + j] - xs[i];
            dd[i].push(if denom.abs() < crate::TOL { 0.0 } else { diff / denom });
        }
    }
    dd.remove(0)
}

/// Newton interpolation using divided differences.
///
/// Returns a [`Polynomial`] that passes through every `(xs[i], ys[i])`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::interpolate::newton;
///
/// let xs = [0.0, 1.0, 2.0];
/// let ys = [1.0, 2.0, 5.0];
/// let p = newton(&xs, &ys);
/// assert!((p.eval(0.0) - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn newton(xs: &[f64], ys: &[f64]) -> Polynomial {
    let dd = divided_differences(xs, ys);
    let n = xs.len();
    let mut result = Polynomial::constant(dd[0]);
    let mut basis = Polynomial::constant(1.0);
    for i in 1..n {
        basis = basis * Polynomial::from_coeffs(&[-xs[i - 1], 1.0]);
        result = result + basis.clone() * Polynomial::constant(dd[i]);
    }
    result
}

/// Evaluate a Newton-form polynomial at a single point without constructing the full polynomial.
///
/// Uses the nested multiplication form directly from the divided-difference table.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::interpolate::evaluate_newton;
///
/// let xs = [0.0, 1.0, 2.0];
/// let ys = [1.0, 2.0, 5.0];
/// assert!((evaluate_newton(&xs, &ys, 1.5) - 3.25).abs() < 1e-10);
/// ```
#[must_use]
pub fn evaluate_newton(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let dd = divided_differences(xs, ys);
    let n = xs.len();
    let mut result = dd[n - 1];
    for i in (0..n - 1).rev() {
        result = dd[i] + (x - xs[i]) * result;
    }
    result
}

/// Vandermonde matrix (n×n) for interpolation nodes `xs`.
///
/// Row `i`, column `j` is `xs[i]^j`.
#[must_use]
pub fn vandermonde(xs: &[f64]) -> Vec<Vec<f64>> {
    let n = xs.len();
    (0..n)
        .map(|i| (0..n).map(|j| xs[i].powi(j as i32)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrange_test() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 3.0, 7.0];
        let p = lagrange(&xs, &ys);
        // Interpolant is x^2 + x + 1 → p(0.5) = 1.75
        let err = (p.eval(0.5) - 1.75).abs();
        assert!(err < 1e-10);
        for i in 0..3 {
            assert!((p.eval(xs[i]) - ys[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn newton_test() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 3.0, 7.0];
        let p = newton(&xs, &ys);
        for i in 0..3 {
            assert!((p.eval(xs[i]) - ys[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn evaluate_newton_test() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 3.0, 7.0];
        let val = evaluate_newton(&xs, &ys, 0.5);
        let p = newton(&xs, &ys);
        assert!((val - p.eval(0.5)).abs() < 1e-12);
    }

    #[test]
    fn vandermonde_test() {
        let xs = [0.0, 1.0, 2.0];
        let v = vandermonde(&xs);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(v[1], vec![1.0, 1.0, 1.0]);
        assert_eq!(v[2], vec![1.0, 2.0, 4.0]);
    }
}
