//! Nonlinear equation solving: root finding and systems of equations.
//!
//! ## Migration Notice (v0.2.0)
//! The single-variable root-finding functions in this module now delegate to
//! [`mathverse_numerical`] for the canonical implementations. The API remains
//! compatible but returns `Option` instead of `Result` for convenience.
//!
//! For advanced methods (Brent, Muller, Halley, etc.) and full `Result` error handling,
//! use [`mathverse_numerical::root`] or [`mathverse_numerical`] directly.

use mathverse_numerical as num;

/// Newton's method with analytically provided derivative.
///
/// **Note:** This is a convenience wrapper that returns `Option<f64>`.
/// For full error handling with `Result`, use [`mathverse_numerical::newton_raphson`].
///
/// # Examples
/// ```
/// use mathverse_equations::nonlinear::newton;
/// let root = newton(|x| x*x - 2.0, |x| 2.0*x, 1.0, 1e-10, 100).unwrap();
/// assert!((root - 1.41421).abs() < 1e-4);
/// ```
pub fn newton(f: impl Fn(f64) -> f64, df: impl Fn(f64) -> f64, x0: f64, tol: f64, max_iter: usize) -> Option<f64> {
    num::newton_raphson(&f, &df, x0, tol, max_iter).ok()
}

/// Secant method (derivative-free Newton-like).
///
/// **Note:** This is a convenience wrapper that returns `Option<f64>`.
/// For full error handling with `Result`, use [`mathverse_numerical::root::secant`].
///
/// # Examples
/// ```
/// use mathverse_equations::nonlinear::secant;
/// let root = secant(|x| x*x - 2.0, 1.0, 2.0, 1e-10, 100).unwrap();
/// assert!((root - 1.41421).abs() < 1e-4);
/// ```
pub fn secant(f: impl Fn(f64) -> f64, x0: f64, x1: f64, tol: f64, max_iter: usize) -> Option<f64> {
    num::root::secant(&f, x0, x1, tol, max_iter).ok()
}

/// Bisection method on a bracketing interval.
///
/// **Note:** This is a convenience wrapper that returns `Option<f64>`.
/// For full error handling with `Result`, use [`mathverse_numerical::bisection`].
///
/// # Examples
/// ```
/// use mathverse_equations::nonlinear::bisection;
/// let root = bisection(|x| x*x - 2.0, 0.0, 2.0, 1e-10).unwrap();
/// assert!((root - 1.41421).abs() < 1e-4);
/// ```
pub fn bisection(f: impl Fn(f64) -> f64, a0: f64, b0: f64, tol: f64) -> Option<f64> {
    num::bisection(&f, a0, b0, tol).ok()
}

/// Newton's method for a system of nonlinear equations.
pub fn newton_system(f: &[impl Fn(&[f64]) -> f64], j: &impl Fn(&[f64]) -> Vec<Vec<f64>>, x0: &[f64], tol: f64, max_iter: usize) -> Option<Vec<f64>> {
    let n = x0.len();
    let mut x = x0.to_vec();
    for _ in 0..max_iter {
        let fx: Vec<f64> = f.iter().map(|fi| fi(&x)).collect();
        if fx.iter().map(|v| v * v).sum::<f64>().sqrt() < tol { return Some(x); }
        let jac = j(&x);
        let dx = super::matrix_eq::solve_gauss(&jac, &fx)?;
        for i in 0..n { x[i] -= dx[i]; }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_sqrt() {
        let x = newton(|x| x*x - 2.0, |x| 2.0*x, 1.0, 1e-15, 100).unwrap();
        assert!((x - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn secant_method() {
        let x = secant(|x| x*x - 2.0, 1.0, 2.0, 1e-15, 100).unwrap();
        assert!((x - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn bisect() {
        let x = bisection(|x| x*x - 2.0, 0.0, 2.0, 1e-12).unwrap();
        assert!((x - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn newton_system_test() {
        let f: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
            Box::new(|x| x[0]*x[0] + x[1]*x[1] - 1.0),
            Box::new(|x| x[0] - x[1]),
        ];
        let j = |x: &[f64]| vec![
            vec![2.0*x[0], 2.0*x[1]],
            vec![1.0, -1.0],
        ];
        let result = newton_system(&f, &j, &[0.5, 0.5], 1e-12, 100).unwrap();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((result[0] - expected).abs() < 1e-10);
        assert!((result[1] - expected).abs() < 1e-10);
    }
}
