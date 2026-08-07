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
    let mut scratch = x.to_vec();
    scratch[var] = x[var] + h;
    let fp = f(&scratch);
    scratch[var] = x[var] - h;
    let fm = f(&scratch);
    (fp - fm) / (2.0 * h)
}

/// `f⁽ⁿ⁾(x)`, nth derivative via a central finite-difference stencil.
///
/// Returns `(value, estimated_error)` using Richardson extrapolation.
/// The error estimate is the difference between the h and h/2 stencils,
/// scaled by `1/(2ⁿ - 1)`.
///
/// ```
/// use mathverse_calculus::derivative::nth_derivative;
/// // Third derivative of x³ should be 6
/// let (v, e) = nth_derivative(&|x| x * x * x, 2.0, 3);
/// assert!((v - 6.0).abs() < e * 10.0);
/// ```
pub fn nth_derivative(f: &dyn Fn(f64) -> f64, x: f64, n: usize) -> (f64, f64) {
    if n == 0 {
        return (f(x), 0.0);
    }
    if n == 1 {
        let h = h_at(x);
        let val = (f(x + h) - f(x - h)) / (2.0 * h);
        let val2h = (f(x + 2.0 * h) - f(x - 2.0 * h)) / (4.0 * h);
        let err = (val - val2h).abs() / 3.0;
        return (val, err);
    }
    if n == 2 {
        let h = 1e-3 * x.abs().max(1.0);
        let val = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
        let val2h = (f(x + 2.0 * h) - 2.0 * f(x) + f(x - 2.0 * h)) / (4.0 * h * h);
        let err = (val - val2h).abs() / 3.0;
        return (val, err);
    }

    // Optimal step for central nth-order finite difference: h ~ ε^{1/(n+2)} * scale.
    let scale = x.abs().max(1.0);
    let h = f64::EPSILON.powf(1.0 / (n as f64 + 2.0)) * scale;

    let compute = |h: f64| -> f64 {
        let mut result = 0.0;
        for k in 0..=n {
            let sign = if (n - k).is_multiple_of(2) { 1.0 } else { -1.0 };
            let coeff = sign * binomial(n, k) as f64;
            result += coeff * f(x + (k as f64 - n as f64 / 2.0) * h);
        }
        result / h.powi(n as i32)
    };

    let val = compute(h);
    let val2 = compute(h / 2.0);
    let err = (val - val2).abs() / (2.0_f64.powi(n as i32) - 1.0);
    (val, err)
}

/// Discrete gradient: `np.gradient` equivalent.
/// Second-order accurate central differences on interior,
/// first-order forward/backward on boundaries.
///
    /// ```
    /// use mathverse_calculus::derivative::discrete_gradient;
    /// let y = vec![1.0, 4.0, 9.0, 16.0]; // x² at x=1,2,3,4
    /// let g = discrete_gradient(&y, 1.0);
    /// assert!((g[0] - 3.0).abs() < 1e-10); // forward diff: (4-1)/1 = 3
    /// assert!((g[1] - 4.0).abs() < 1e-10); // central diff: (9-1)/2 = 4
    /// assert!((g[3] - 7.0).abs() < 1e-10); // backward diff: (16-9)/1 = 7
    /// ```
pub fn discrete_gradient(y: &[f64], dx: f64) -> Vec<f64> {
    let n = y.len();
    let mut grad = vec![0.0; n];
    if n <= 1 {
        return grad;
    }
    // Boundaries: first-order
    grad[0] = (y[1] - y[0]) / dx;
    grad[n - 1] = (y[n - 1] - y[n - 2]) / dx;
    // Interior: central difference
    for i in 1..n - 1 {
        grad[i] = (y[i + 1] - y[i - 1]) / (2.0 * dx);
    }
    grad
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
        let (v0, _) = nth_derivative(&|x| x * x * x, 2.0, 0);
        assert!((v0 - 8.0).abs() < 1e-8);

        let (v1, _) = nth_derivative(&f64::sin, 0.0, 1);
        assert!((v1 - 1.0).abs() < 1e-8);

        let (v3, e3) = nth_derivative(&|x| x.powi(5), 1.0, 3);
        assert!((v3 - 60.0).abs() < e3 * 10.0 + 1e-6, "n=3 got {v3} ± {e3}");

        let (v5, e5) = nth_derivative(&|x| x.powi(5), 1.0, 5);
        assert!((v5 - 120.0).abs() < e5 * 10.0 + 0.01, "n=5 got {v5} ± {e5}");

        // 4th derivative of sin at 0 should be 0
        let (v4, e4) = nth_derivative(&f64::sin, 0.0, 4);
        assert!(v4.abs() < e4 * 10.0 + 1e-4, "n=4 got {v4} ± {e4}");
    }

    #[test]
    fn discrete_gradient_test() {
        // x² at x=1,2,3,4 → gradient should be 2x
        let y = vec![1.0, 4.0, 9.0, 16.0];
        let g = discrete_gradient(&y, 1.0);
        assert!((g[0] - 3.0).abs() < 1e-10); // forward diff: (4-1)/1 = 3
        assert!((g[1] - 4.0).abs() < 1e-10); // central diff: (9-1)/2 = 4
        assert!((g[2] - 6.0).abs() < 1e-10); // central diff: (16-4)/2 = 6
        assert!((g[3] - 7.0).abs() < 1e-10); // backward diff: (16-9)/1 = 7

        // Constant function → zero gradient
        let c = vec![5.0; 4];
        let gc = discrete_gradient(&c, 1.0);
        assert!(gc.iter().all(|&v| v.abs() < 1e-10));

        // Empty and single-element
        assert!(discrete_gradient(&[], 1.0).is_empty());
        assert_eq!(discrete_gradient(&[1.0], 1.0), vec![0.0]);
    }
}
