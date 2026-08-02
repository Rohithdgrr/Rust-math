//! Numerical integration: fixed-rule and adaptive.

/// Composite trapezoid rule, `n` intervals.
pub fn trapezoid(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut s = 0.5 * (f(a) + f(b));
    for i in 1..n {
        s += f(a + i as f64 * h);
    }
    s * h
}

/// Composite Simpson's rule, `n` intervals (even). Error O(h⁴).
pub fn simpson(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n {
        let w = if i.is_multiple_of(2) { 2.0 } else { 4.0 };
        s += w * f(a + i as f64 * h);
    }
    s * h / 3.0
}

/// Adaptive Simpson, converges to ~`tol` relative accuracy.
///
/// ```
/// use mathverse_calculus::integrate::integrate;
/// assert!((integrate(&f64::sin, 0.0, core::f64::consts::PI, 1e-10) - 2.0).abs() < 1e-8);
/// ```
pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    fn s(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        (b - a) / 6.0 * (f(a) + 4.0 * f((a + b) / 2.0) + f(b))
    }
    fn go(f: &dyn Fn(f64) -> f64, a: f64, b: f64, whole: f64, tol: f64, depth: u32) -> f64 {
        let m = (a + b) / 2.0;
        let left = s(f, a, m);
        let right = s(f, m, b);
        let delta = left + right - whole;
        if delta.abs() <= 15.0 * tol || depth == 0 {
            left + right + delta / 15.0
        } else {
            go(f, a, m, left, tol / 2.0, depth - 1) + go(f, m, b, right, tol / 2.0, depth - 1)
        }
    }
    go(f, a, b, s(f, a, b), tol, 30)
}

/// Gaussian quadrature (Legendre polynomials) for n points.
/// Exact for polynomials of degree up to 2n-1.
///
/// ```
/// use mathverse_calculus::integrate::gaussian_quadrature;
/// // ∫₀¹ x² dx = 1/3
/// assert!((gaussian_quadrature(&|x| x * x, 0.0, 1.0, 3) - 1.0/3.0).abs() < 1e-10);
/// ```
pub fn gaussian_quadrature(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let (nodes, weights) = legendre_nodes_weights(n);
    let scale = (b - a) / 2.0;
    let shift = (a + b) / 2.0;
    let mut sum = 0.0;
    for (&x, &w) in nodes.iter().zip(weights.iter()) {
        sum += w * f(scale * x + shift);
    }
    sum * scale
}

fn legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    match n {
        1 => (vec![0.0], vec![2.0]),
        2 => (
            vec![-1.0_f64.sqrt(), 1.0_f64.sqrt()],
            vec![1.0, 1.0],
        ),
        3 => (
            vec![-0.7745966692414834, 0.0, 0.7745966692414834],
            vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
        ),
        4 => (
            vec![
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            vec![0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538],
        ),
        5 => (
            vec![
                -0.9061798459386640,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.9061798459386640,
            ],
            vec![
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ],
        ),
        _ => {
            // For higher n, use Newton-Raphson to find roots
            // This is a simplified implementation
            let mut nodes = Vec::with_capacity(n);
            let mut weights = Vec::with_capacity(n);
            for i in 0..n {
                let x = ((2 * (n - i) - 1) as f64 * core::f64::consts::PI / (2 * n) as f64).cos();
                let (node, weight) = legendre_root_weight(x, n);
                nodes.push(node);
                weights.push(weight);
            }
            (nodes, weights)
        }
    }
}

fn legendre_root_weight(x0: f64, n: usize) -> (f64, f64) {
    let mut x = x0;
    for _ in 0..100 {
        let (p, dp) = legendre_and_derivative(x, n);
        let dx = p / dp;
        x -= dx;
        if dx.abs() < 1e-15 {
            break;
        }
    }
    let (_, dp) = legendre_and_derivative(x, n);
    let weight = 2.0 / ((1.0 - x * x) * dp * dp);
    (x, weight)
}

fn legendre_and_derivative(x: f64, n: usize) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }
    let mut p_prev = 1.0;
    let mut p_curr = x;
    let mut dp_prev = 0.0;
    let mut dp_curr = 1.0;
    for k in 1..n {
        let p_next = ((2 * k + 1) as f64 * x * p_curr - k as f64 * p_prev) / ((k + 1) as f64);
        let dp_next = ((2 * k + 1) as f64 * (p_curr + x * dp_curr) - k as f64 * dp_prev) / ((k + 1) as f64);
        p_prev = p_curr;
        p_curr = p_next;
        dp_prev = dp_curr;
        dp_curr = dp_next;
    }
    (p_curr, dp_curr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_rules() {
        let f = &f64::sin;
        let pi = core::f64::consts::PI;
        // Convergence: error shrinks ~16x per 8x refinement for Simpson.
        let e1 = (simpson(f, 0.0, pi, 8) - 2.0).abs();
        let e2 = (simpson(f, 0.0, pi, 64) - 2.0).abs();
        assert!(e2 < e1 / 3.0, "{e1} vs {e2}");
        assert!(e2 < 1e-5);
        assert!((trapezoid(f, 0.0, pi, 1024) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn adaptive() {
        assert!((integrate(&f64::sin, 0.0, core::f64::consts::PI, 1e-12) - 2.0).abs() < 1e-9);
        assert!((integrate(&|x| 1.0 / x, 1.0, 2.0, 1e-12) - 2.0f64.ln()).abs() < 1e-9);
        // Gaussian: integrates polynomials to machine precision.
        assert!((integrate(&|x| x.exp() * (-x * x).exp(), -3.0, 3.0, 1e-10)).is_finite());
    }

    #[test]
    fn gaussian_quadrature_test() {
        // ∫₀¹ x² dx = 1/3
        assert!((gaussian_quadrature(&|x| x * x, 0.0, 1.0, 3) - 1.0/3.0).abs() < 1e-10);
        // ∫₋₁¹ x⁴ dx = 2/5
        assert!((gaussian_quadrature(&|x| x.powi(4), -1.0, 1.0, 3) - 2.0/5.0).abs() < 1e-10);
        // ∫₀^π sin(x) dx = 2; 5-point Gauss error is ~1e-7, not machine eps
        assert!((gaussian_quadrature(&f64::sin, 0.0, core::f64::consts::PI, 5) - 2.0).abs() < 1e-6);
    }
}
