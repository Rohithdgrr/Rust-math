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

/// `f⁽ⁿ⁾(x)`, nth derivative via a central finite-difference stencil.
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

    // Optimal step for an nth-order finite difference: h ~ ε^{1/(n+1)} * scale.
    // Uses machine epsilon (2.22e-16), not the 1e-6 used by h_at().
    // h_at() gives ε^{1/2} which is optimal for n=1 but too small for n>2,
    // amplifying roundoff catastrophically (e.g. n=3: roundoff/h³ ≈ 200).
    let scale = x.abs().max(1.0);
    let h = f64::EPSILON.powf(1.0 / (n as f64 + 1.0)) * scale;
    let mut result = 0.0;
    for k in 0..=n {
        // (-1)^(n-k): sign flips with the stencil's distance from the center,
        // not with k alone. (-1)^k here flips the result for every odd n.
        let sign = if (n - k) % 2 == 0 { 1.0 } else { -1.0 };
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
        // Odd orders used to come back negated by the stencil sign bug.
        let v3 = nth_derivative(&|x| x.powi(5), 1.0, 3);
        assert!((v3 - 60.0).abs() < 1e-2, "n=3 got {v3}");
        let v5 = nth_derivative(&|x| x.powi(5), 1.0, 5);
        assert!((v5 - 120.0).abs() < 10.0, "n=5 got {v5}");
        assert!((nth_derivative(&f64::sin, 0.0, 1) - 1.0).abs() < 1e-8);
    }
}
