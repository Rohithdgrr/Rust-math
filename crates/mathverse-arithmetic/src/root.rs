//! Advanced root operations: nth roots, inverse roots, root properties.

use mathverse_core::error::{MathError, MathResult};

/// Advanced root operations.
pub struct Root;

impl Root {
    /// Nth root: x^(1/n).
    pub fn nth_root(x: f64, n: u32) -> MathResult<f64> {
        if n == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        if x >= 0.0 {
            Ok(x.powf(1.0 / n as f64))
        } else {
            // Handle negative bases
            if n % 2 == 0 {
                return Err(MathError::InvalidArgument("even root of negative number"));
            }
            Ok(-((-x).powf(1.0 / n as f64)))
        }
    }

    /// Square root: x^(1/2).
    pub fn square_root(x: f64) -> MathResult<f64> {
        if x < 0.0 {
            return Err(MathError::InvalidArgument("square root of negative number"));
        }
        Ok(x.sqrt())
    }

    /// Cube root: x^(1/3).
    pub fn cube_root(x: f64) -> f64 {
        if x >= 0.0 {
            x.powf(1.0 / 3.0)
        } else {
            -((-x).powf(1.0 / 3.0))
        }
    }

    /// Fourth root: x^(1/4).
    pub fn fourth_root(x: f64) -> MathResult<f64> {
        Self::nth_root(x, 4)
    }

    /// Inverse square root: 1/sqrt(x).
    pub fn inverse_square_root(x: f64) -> MathResult<f64> {
        if x <= 0.0 {
            return Err(MathError::InvalidArgument("inverse square root of non-positive number"));
        }
        Ok(1.0 / x.sqrt())
    }

    /// Inverse nth root: 1/x^(1/n).
    pub fn inverse_nth_root(x: f64, n: u32) -> MathResult<f64> {
        let root = Self::nth_root(x, n)?;
        if root == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        Ok(1.0 / root)
    }

    /// Root of unity: e^(2πi/n) - returns real part for n=1,2,4.
    pub fn root_of_unity(n: u32) -> f64 {
        match n {
            1 => 1.0,
            2 => -1.0,
            4 => 0.0,
            _ => (2.0 * core::f64::consts::PI / n as f64).cos(),
        }
    }

    /// Principal root using Newton's method.
    pub fn principal_root(x: f64, n: u32, tolerance: f64) -> MathResult<f64> {
        if n == 0 {
            return Err(MathError::DivisionByZero);
        }
        
        if x == 0.0 {
            return Ok(0.0);
        }
        
        let mut guess = x / n as f64;
        
        for _ in 0..100 {
            let guess_pow_n_minus_1 = guess.powf((n - 1) as f64);
            let new_guess = ((n as f64 - 1.0) * guess + x / guess_pow_n_minus_1) / n as f64;
            
            if (new_guess - guess).abs() < tolerance {
                return Ok(new_guess);
            }
            
            guess = new_guess;
        }
        
        Ok(guess)
    }

    /// Nested radical: sqrt(a + sqrt(b + sqrt(c + ...))).
    pub fn nested_radical(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut result = values[values.len() - 1];
        
        for &value in values.iter().rev().skip(1) {
            result = (value + result).sqrt();
        }
        
        result
    }

    /// Continued radical: sqrt(a + sqrt(a + sqrt(a + ...))).
    pub fn continued_radical(a: f64, iterations: u32) -> f64 {
        let mut result = a.sqrt();
        
        for _ in 1..iterations {
            result = (a + result).sqrt();
        }
        
        result
    }

    /// Golden ratio: (1 + sqrt(5)) / 2.
    pub fn golden_ratio() -> f64 {
        (1.0 + 5.0_f64.sqrt()) / 2.0
    }

    /// Silver ratio: 1 + sqrt(2).
    pub fn silver_ratio() -> f64 {
        1.0 + 2.0_f64.sqrt()
    }

    /// Bronze ratio: 3 + sqrt(13) / 2.
    pub fn bronze_ratio() -> f64 {
        (3.0 + 13.0_f64.sqrt()) / 2.0
    }
}

/// Root properties and identities.
pub struct RootProperties;

impl RootProperties {
    /// Check if root is rational (for perfect powers).
    pub fn is_rational_root(x: f64, n: u32) -> bool {
        if x < 0.0 && n % 2 == 0 {
            return false;
        }
        
        let abs_x = x.abs();
        let root = abs_x.powf(1.0 / n as f64);
        
        // Check if root is close to an integer
        let rounded = root.round();
        (root - rounded).abs() < 1e-10
    }

    /// Simplify radical: sqrt(a*b) = sqrt(a)*sqrt(b) if a,b are perfect squares.
    pub fn simplify_radical(x: f64) -> (f64, f64) {
        // Return (coefficient, remaining radicand)
        let mut coefficient = 1.0;
        let mut radicand = x;
        
        for i in 2..=100 {
            while ((radicand / (i as f64).powi(2)).round() - radicand / (i as f64).powi(2)).abs() < 1e-10 {
                radicand /= (i as f64).powi(2);
                coefficient *= i as f64;
            }
        }
        
        (coefficient, radicand)
    }

    /// Rationalize denominator: 1/sqrt(a) = sqrt(a)/a.
    pub fn rationalize_denominator(a: f64) -> MathResult<(f64, f64)> {
        if a <= 0.0 {
            return Err(MathError::InvalidArgument("a must be positive"));
        }
        
        let sqrt_a = a.sqrt();
        Ok((sqrt_a, a))
    }

    /// Conjugate radical: a + sqrt(b) and a - sqrt(b).
    pub fn conjugate(a: f64, b: f64) -> (f64, f64) {
        let sqrt_b = if b >= 0.0 { b.sqrt() } else { f64::NAN };
        ((a + sqrt_b), (a - sqrt_b))
    }

    /// Multiply conjugates: (a + sqrt(b))(a - sqrt(b)) = a^2 - b.
    pub fn multiply_conjugates(a: f64, b: f64) -> f64 {
        a * a - b
    }

    /// Root mean square: sqrt((x1^2 + x2^2 + ... + xn^2) / n).
    pub fn root_mean_square(values: &[f64]) -> MathResult<f64> {
        if values.is_empty() {
            return Err(MathError::InvalidArgument("empty slice"));
        }
        
        let sum_of_squares: f64 = values.iter().map(|&x| x * x).sum();
        (sum_of_squares / values.len() as f64).sqrt()
    }

    /// Geometric mean: (x1 * x2 * ... * xn)^(1/n).
    pub fn geometric_mean(values: &[f64]) -> MathResult<f64> {
        if values.is_empty() {
            return Err(MathError::InvalidArgument("empty slice"));
        }
        
        let product: f64 = values.iter().product();
        if product < 0.0 {
            return Err(MathError::InvalidArgument("geometric mean undefined for negative product"));
        }
        
        Ok(product.powf(1.0 / values.len() as f64))
    }

    /// Harmonic mean: n / (1/x1 + 1/x2 + ... + 1/xn).
    pub fn harmonic_mean(values: &[f64]) -> MathResult<f64> {
        if values.is_empty() {
            return Err(MathError::InvalidArgument("empty slice"));
        }
        
        let sum_reciprocals: f64 = values.iter()
            .map(|&x| {
                if x == 0.0 {
                    return f64::INFINITY;
                }
                1.0 / x
            })
            .sum();
        
        if sum_reciprocals == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        Ok(values.len() as f64 / sum_reciprocals)
    }
}

/// Root solving methods.
pub struct RootSolving;

impl RootSolving {
    /// Bisection method for finding roots of f(x) = 0.
    pub fn bisection(
        f: impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
        max_iterations: u32,
    ) -> MathResult<f64> {
        let fa = f(a);
        let fb = f(b);
        
        if fa * fb > 0.0 {
            return Err(MathError::InvalidArgument("function must have opposite signs at endpoints"));
        }
        
        let mut a = a;
        let mut b = b;
        
        for _ in 0..max_iterations {
            let c = (a + b) / 2.0;
            let fc = f(c);
            
            if (b - a).abs() < tolerance || fc.abs() < tolerance {
                return Ok(c);
            }
            
            if fa * fc < 0.0 {
                b = c;
            } else {
                a = c;
            }
        }
        
        Ok((a + b) / 2.0)
    }

    /// Newton-Raphson method for finding roots.
    pub fn newton_raphson(
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
        x0: f64,
        tolerance: f64,
        max_iterations: u32,
    ) -> MathResult<f64> {
        let mut x = x0;
        
        for _ in 0..max_iterations {
            let fx = f(x);
            let dfx = df(x);
            
            if dfx.abs() < 1e-15 {
                return Err(MathError::InvalidArgument("derivative too small"));
            }
            
            let x_new = x - fx / dfx;
            
            if (x_new - x).abs() < tolerance {
                return Ok(x_new);
            }
            
            x = x_new;
        }
        
        Ok(x)
    }

    /// Secant method (derivative-free Newton).
    pub fn secant(
        f: impl Fn(f64) -> f64,
        x0: f64,
        x1: f64,
        tolerance: f64,
        max_iterations: u32,
    ) -> MathResult<f64> {
        let mut x_prev = x0;
        let mut x_curr = x1;
        
        for _ in 0..max_iterations {
            let f_prev = f(x_prev);
            let f_curr = f(x_curr);
            
            if (f_curr - f_prev).abs() < 1e-15 {
                return Err(MathError::InvalidArgument("function values too close"));
            }
            
            let x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev);
            
            if (x_new - x_curr).abs() < tolerance {
                return Ok(x_new);
            }
            
            x_prev = x_curr;
            x_curr = x_new;
        }
        
        Ok(x_curr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nth_root() {
        assert!((Root::nth_root(8.0, 3).unwrap() - 2.0).abs() < 1e-10);
        assert!((Root::nth_root(-8.0, 3).unwrap() + 2.0).abs() < 1e-10);
        assert!((Root::nth_root(16.0, 4).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_square_root() {
        assert!((Root::inverse_square_root(4.0).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_principal_root() {
        assert!((Root::principal_root(27.0, 3, 1e-10).unwrap() - 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_nested_radical() {
        let result = Root::nested_radical(&[2.0, 2.0, 2.0]);
        assert!(result > 1.5);
    }

    #[test]
    fn test_golden_ratio() {
        let phi = Root::golden_ratio();
        assert!((phi - 1.6180339887).abs() < 1e-9);
    }

    #[test]
    fn test_rms() {
        let values = vec![1.0, 2.0, 3.0];
        let rms = RootProperties::root_mean_square(&values).unwrap();
        assert!((rms - 2.160246899).abs() < 1e-8);
    }

    #[test]
    fn test_geometric_mean() {
        let values = vec![1.0, 2.0, 4.0];
        let gm = RootProperties::geometric_mean(&values).unwrap();
        assert!((gm - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_mean() {
        let values = vec![1.0, 2.0, 4.0];
        let hm = RootProperties::harmonic_mean(&values).unwrap();
        assert!((hm - 1.714285714).abs() < 1e-8);
    }

    #[test]
    fn test_bisection() {
        let f = |x: f64| x * x - 4.0;
        let root = RootSolving::bisection(f, 0.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_newton_raphson() {
        let f = |x: f64| x * x - 4.0;
        let df = |x: f64| 2.0 * x;
        let root = RootSolving::newton_raphson(f, df, 1.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-10);
    }
}
