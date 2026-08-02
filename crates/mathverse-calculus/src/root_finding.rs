//! Root finding algorithms for solving f(x) = 0.
//!
//! For the full suite of root-finding methods (Brent, Muller, Householder, etc.),
//! see [`mathverse_numerical::root`]. The methods here are tightly integrated
//! with the calculus crate's automatic differentiation for `newton_raphson_auto`.

use mathverse_core::error::{MathError, MathResult};
use crate::derivative::derivative;

/// Bisection method for finding roots of f(x) = 0.
///
/// Requires f(a) and f(b) to have opposite signs.
/// Returns error if the interval does not bracket a root.
///
/// ```
/// use mathverse_calculus::root_finding::bisection;
/// // Find root of x² - 4 = 0 on [1, 3]
/// let root = bisection(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
/// assert!((root - 2.0).abs() < 1e-8);
/// ```
pub fn bisection(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> MathResult<f64> {
    let fa = f(a);
    let fb = f(b);
    if fa * fb > 0.0 {
        return Err(MathError::InvalidArgument("interval does not bracket a root"));
    }
    let mut a = a;
    let mut b = b;
    let mut fa = fa;
    let mut _fb = fb;
    for _ in 0..max_iter {
        let c = (a + b) / 2.0;
        let fc = f(c);
        
        if (b - a).abs() < tol || fc.abs() < tol {
            return Ok(c);
        }
        
        if fa * fc < 0.0 {
            b = c;
            _fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    
    Err(MathError::NotConverged("bisection max iterations exceeded"))
}

/// Newton-Raphson method for finding roots of f(x) = 0.
///
/// Requires the derivative of f. Converges quadratically near simple roots.
/// Returns error if derivative is zero or max iterations exceeded.
///
/// ```
/// use mathverse_calculus::root_finding::newton_raphson;
/// // Find root of x² - 4 = 0 starting from 3
/// let f = |x| x * x - 4.0;
/// let df = |x| 2.0 * x;
/// let root = newton_raphson(&f, &df, 3.0, 1e-10, 100).unwrap();
/// assert!((root - 2.0).abs() < 1e-8);
/// ```
pub fn newton_raphson(
    f: &dyn Fn(f64) -> f64,
    df: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> MathResult<f64> {
    let mut x = x0;
    
    for _ in 0..max_iter {
        let fx = f(x);
        let dfx = df(x);
        
        if dfx.abs() < 1e-15 {
            return Err(MathError::InvalidArgument("derivative is zero"));
        }
        
        let x_new = x - fx / dfx;
        
        if (x_new - x).abs() < tol || fx.abs() < tol {
            return Ok(x_new);
        }
        
        x = x_new;
    }
    
    Err(MathError::NotConverged("newton max iterations exceeded"))
}

/// Secant method for finding roots of f(x) = 0.
///
/// Does not require derivative; approximates it using finite differences.
/// Converges superlinearly (order ~1.618).
///
/// ```
/// use mathverse_calculus::root_finding::secant;
/// // Find root of x² - 4 = 0 starting from 1 and 3
/// let root = secant(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
/// assert!((root - 2.0).abs() < 1e-8);
/// ```
pub fn secant(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    x1: f64,
    tol: f64,
    max_iter: usize,
) -> MathResult<f64> {
    let mut x_prev = x0;
    let mut x_curr = x1;
    let mut f_prev = f(x_prev);
    let mut f_curr = f(x_curr);
    
    for _ in 0..max_iter {
        let denominator = f_curr - f_prev;
        
        if denominator.abs() < 1e-15 {
            return Err(MathError::InvalidArgument("denominator is zero"));
        }
        
        let x_new = x_curr - f_curr * (x_curr - x_prev) / denominator;
        
        if (x_new - x_curr).abs() < tol || f_curr.abs() < tol {
            return Ok(x_new);
        }
        
        x_prev = x_curr;
        f_prev = f_curr;
        x_curr = x_new;
        f_curr = f(x_curr);
    }
    
    Err(MathError::NotConverged("secant max iterations exceeded"))
}

/// False position (regula falsi) method for finding roots.
///
/// Combines bisection with secant method. Always maintains bracketing.
///
/// ```
/// use mathverse_calculus::root_finding::false_position;
/// // Find root of x² - 4 = 0 on [1, 3]
/// let root = false_position(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
/// assert!((root - 2.0).abs() < 1e-8);
/// ```
pub fn false_position(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> MathResult<f64> {
    let fa = f(a);
    let fb = f(b);
    if fa * fb > 0.0 {
        return Err(MathError::InvalidArgument("interval does not bracket a root"));
    }
    let mut a = a;
    let mut b = b;
    let mut fa = fa;
    let mut fb = fb;
    for _ in 0..max_iter {
        let c = (a * fb - b * fa) / (fb - fa);
        let fc = f(c);
        
        if (b - a).abs() < tol || fc.abs() < tol {
            return Ok(c);
        }
        
        if fa * fc < 0.0 {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    
    Err(MathError::NotConverged("false position max iterations exceeded"))
}

/// Newton-Raphson method with numerical derivative.
///
/// Automatically computes derivative using central differences.
///
/// ```
/// use mathverse_calculus::root_finding::newton_raphson_auto;
/// // Find root of x² - 4 = 0 starting from 3
/// let root = newton_raphson_auto(&|x| x * x - 4.0, 3.0, 1e-10, 100).unwrap();
/// assert!((root - 2.0).abs() < 1e-8);
/// ```
pub fn newton_raphson_auto(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> MathResult<f64> {
    let mut x = x0;
    
    for _ in 0..max_iter {
        let fx = f(x);
        let dfx = derivative(f, x);
        
        if dfx.abs() < 1e-15 {
            return Err(MathError::InvalidArgument("derivative is zero"));
        }
        
        let x_new = x - fx / dfx;
        
        if (x_new - x).abs() < tol || fx.abs() < tol {
            return Ok(x_new);
        }
        
        x = x_new;
    }
    
    Err(MathError::NotConverged("newton max iterations exceeded"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisection_test() {
        let root = bisection(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
        
        // Test with negative root
        let root_neg = bisection(&|x| x * x - 4.0, -3.0, -1.0, 1e-10, 100).unwrap();
        assert!((root_neg + 2.0).abs() < 1e-8);
    }

    #[test]
    fn newton_raphson_test() {
        let f = |x| x * x - 4.0;
        let df = |x| 2.0 * x;
        let root = newton_raphson(&f, &df, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
        
        // Test with cubic
        let f2 = |x| x * x * x - 8.0;
        let df2 = |x| 3.0 * x * x;
        let root2 = newton_raphson(&f2, &df2, 2.5, 1e-10, 100).unwrap();
        assert!((root2 - 2.0).abs() < 1e-8);
    }

    #[test]
    fn secant_test() {
        let root = secant(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn false_position_test() {
        let root = false_position(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn newton_raphson_auto_test() {
        let root = newton_raphson_auto(&|x| x * x - 4.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
    }

    #[test]
    fn error_cases() {
        // Interval doesn't bracket root
        assert!(bisection(&|x| x * x + 1.0, 0.0, 1.0, 1e-10, 100).is_err());
        assert!(false_position(&|x| x * x + 1.0, 0.0, 1.0, 1e-10, 100).is_err());
        
        // Zero derivative
        let f = |x| 1.0;
        let df = |x| 0.0;
        assert!(newton_raphson(&f, &df, 1.0, 1e-10, 100).is_err());
    }

    #[test]
    fn max_iter_returns_err() {
        // f(x)=x³-2, root at 1.2599...; bracket [1,2]. Midpoint 1.5, not a root.
        let f = |x| x * x * x - 2.0;
        assert!(bisection(&f, 1.0, 2.0, 1e-15, 2).is_err());
        assert!(secant(&f, 1.0, 2.0, 1e-15, 2).is_err());
        assert!(false_position(&f, 1.0, 2.0, 1e-15, 2).is_err());
        assert!(newton_raphson(&f, &|x| 3.0 * x * x, 1.5, 1e-15, 2).is_err());
        assert!(newton_raphson_auto(&f, 1.5, 1e-15, 2).is_err());
    }
}
