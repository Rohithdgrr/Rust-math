//! Root finding algorithms for solving f(x) = 0.
//!
//! This module re-exports the comprehensive root-finding suite from [`mathverse_numerical`]
//! and adds calculus-specific conveniences like automatic differentiation via
//! [`newton_raphson_auto`].
//!
//! For advanced methods (Brent, Muller, Halley, etc.), use [`mathverse_numerical::root`] directly.

use mathverse_core::error::{MathError, MathResult};
use crate::derivative::derivative;

// Re-export core root-finding methods from mathverse-numerical
pub use mathverse_numerical::{bisection, newton_raphson};

// Re-export advanced methods from mathverse-numerical::root
pub use mathverse_numerical::root::{
    secant, false_position, brent, muller, illinois, steffensen, halley, householder, fixed_point
};


/// Newton-Raphson method with automatic numerical derivative.
///
/// Uses central differences to compute the derivative automatically.
/// This is a convenience wrapper unique to the calculus module.
///
/// For better performance when the derivative is known analytically,
/// use [`newton_raphson`] directly with the analytical derivative.
///
/// # Examples
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

/// Find critical points of a function (where f'(x) = 0).
///
/// This is a convenience function that uses numerical differentiation
/// to find where the derivative is zero.
///
/// # Examples
/// ```
/// use mathverse_calculus::root_finding::find_critical_point;
/// 
/// let f = |x: f64| x * x * x - 3.0 * x;  // Has critical points at x = ±1
/// let critical = find_critical_point(&f, 0.5, 1e-10, 100).unwrap();
/// assert!((critical - 1.0).abs() < 1e-8);
/// ```
pub fn find_critical_point(
    f: &dyn Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> MathResult<f64> {
    // Find root of f'(x) using numerical derivatives for both f' and f''
    let df = |x: f64| derivative(f, x);
    let ddf = |x: f64| {
        let h = (1e-8_f64 * (1.0 + x.abs())).sqrt();
        (derivative(f, x + h) - derivative(f, x - h)) / (2.0 * h)
    };
    
    // Use Newton-Raphson on the derivative
    let mut x = x0;
    for _ in 0..max_iter {
        let dfx = df(x);
        let ddfx = ddf(x);
        
        if ddfx.abs() < 1e-15 {
            return Err(MathError::InvalidArgument("second derivative is zero"));
        }
        
        let x_new = x - dfx / ddfx;
        
        if (x_new - x).abs() < tol || dfx.abs() < tol {
            return Ok(x_new);
        }
        
        x = x_new;
    }
    
    Err(MathError::NotConverged("critical point search exceeded max iterations"))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_raphson_auto_test() {
        let root = newton_raphson_auto(&|x| x * x - 4.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
        
        // Test with cubic
        let root2 = newton_raphson_auto(&|x| x * x * x - 8.0, 3.0, 1e-10, 100).unwrap();
        assert!((root2 - 2.0).abs() < 1e-8);
    }

    #[test]
    fn find_critical_point_test() {
        // f(x) = x³ - 3x has critical points at ±1
        let f = |x: f64| x * x * x - 3.0 * x;
        
        // Find positive critical point
        let cp = find_critical_point(&f, 0.5, 1e-10, 100).unwrap();
        assert!((cp - 1.0).abs() < 1e-8);
        
        // Find negative critical point
        let cp2 = find_critical_point(&f, -0.5, 1e-10, 100).unwrap();
        assert!((cp2 + 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_re_exported_methods() {
        // Test that re-exported methods work correctly
        let root = bisection(&|x| x * x - 4.0, 1.0, 3.0, 1e-10).unwrap();
        assert!((root - 2.0).abs() < 1e-8);
        
        let root2 = newton_raphson(&|x| x * x - 4.0, &|x| 2.0 * x, 3.0, 1e-10, 100).unwrap();
        assert!((root2 - 2.0).abs() < 1e-8);
        
        let root3 = secant(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
        assert!((root3 - 2.0).abs() < 1e-8);
    }
}
