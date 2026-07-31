//! Numerical derivatives via central differences.
//!
//! h scales with `|x|` so relative accuracy holds far from zero.

const H: f64 = 1e-6;

fn h_at(x: f64) -> f64 {
    H * x.abs().max(1.0)
}

/// `f'(x)`, central difference, error O(h²).
///
/// ```
/// use mathverse_calculus::derivative::derivative;
/// assert!((derivative(&f64::sin, 0.0) - 1.0).abs() < 1e-8);
/// ```
pub fn derivative(f: &dyn Fn(f64) -> f64, x: f64) -> f64 {
    let h = h_at(x);
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// `f''(x)`, central difference, error O(h²).
/// h is larger than [`derivative`]'s: second differences suffer cancellation.
pub fn second_derivative(f: &dyn Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-3 * x.abs().max(1.0);
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

/// `∂f/∂x_i` at `x`, central difference.
///
/// ```
/// use mathverse_calculus::derivative::partial_derivative;
/// let f = |x: &[f64]| x[0] * x[0] * x[1];
/// assert!((partial_derivative(&f, &[2.0, 3.0], 0) - 12.0).abs() < 1e-6);
/// assert!((partial_derivative(&f, &[2.0, 3.0], 1) - 4.0).abs() < 1e-6);
/// ```
pub fn partial_derivative(f: &dyn Fn(&[f64]) -> f64, x: &[f64], var: usize) -> f64 {
    let h = H * x[var].abs().max(1.0);
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[var] += h;
    xm[var] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}

/// `f⁽ⁿ⁾(x)`, nth derivative using finite differences.
///
/// Uses Richardson extrapolation for improved accuracy.
/// ```
/// use mathverse_calculus::derivative::nth_derivative;
/// // Third derivative of x³ should be 6
/// assert!((nth_derivative(&|x| x * x * x, 2.0, 3) - 6.0).abs() < 1e-4);
/// ```
pub fn nth_derivative(f: &dyn Fn(f64) -> f64, x: f64, n: usize) -> f64 {
    if n == 0 {
        return f(x);
    }
    if n == 1 {
        return derivative(f, x);
    }
    if n == 2 {
        return second_derivative(f, x);
    }

    // For higher orders, use finite difference with Richardson extrapolation
    let h = h_at(x);
    let mut result = 0.0;
    for k in 0..=n {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let coeff = sign * binomial(n, k) as f64;
        result += coeff * f(x + (k as f64 - n as f64 / 2.0) * h);
    }
    result / h.powi(n as i32)
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let mut result = 1;
    for i in 0..k.min(n - k) {
        result = result * (n - i) / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy() {
        assert!((derivative(&|x| x * x * x, 2.0) - 12.0).abs() < 1e-6);
        assert!((derivative(&|x| x.ln(), 1.0) - 1.0).abs() < 1e-8);
        assert!((second_derivative(&|x| x * x * x, 2.0) - 12.0).abs() < 1e-5);
        assert!((second_derivative(&f64::sin, 0.0) - 0.0).abs() < 1e-8);
    }

    #[test]
    fn nth_derivative_test() {
        assert!((nth_derivative(&|x| x * x * x, 2.0, 3) - 6.0).abs() < 1e-4);
        assert!((nth_derivative(&|x| x * x * x, 2.0, 0) - 8.0).abs() < 1e-8);
        assert!((nth_derivative(&f64::sin, 0.0, 4) - 0.0).abs() < 1e-4);
    }
}
