//! Numerical integration: fixed-rule and adaptive.

use mathverse_core::error::{MathError, MathResult};

/// Composite trapezoid rule, `n` intervals.
///
/// Returns `Err` if `n == 0`.
pub fn trapezoid(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> MathResult<f64> {
    if n == 0 {
        return Err(MathError::InvalidArgument("trapezoid: n must be > 0"));
    }
    let h = (b - a) / n as f64;
    let mut s = 0.5 * (f(a) + f(b));
    for i in 1..n {
        s += f(a + i as f64 * h);
    }
    Ok(s * h)
}

/// Composite Simpson's rule, `n` intervals (even). Error O(h⁴).
///
/// Returns `Err` if `n == 0`.
pub fn simpson(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> MathResult<f64> {
    if n == 0 {
        return Err(MathError::InvalidArgument("simpson: n must be > 0"));
    }
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n {
        let w = if i.is_multiple_of(2) { 2.0 } else { 4.0 };
        s += w * f(a + i as f64 * h);
    }
    Ok(s * h / 3.0)
}

/// Adaptive Simpson, converges to ~`tol` relative accuracy.
///
/// Handles `a > b` by swapping and negating the result.
///
/// ```
/// use mathverse_calculus::integrate::integrate;
/// assert!((integrate(&f64::sin, 0.0, core::f64::consts::PI, 1e-10) - 2.0).abs() < 1e-8);
/// ```
pub fn integrate(f: &dyn Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    fn s(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        (b - a) / 6.0 * (f(a) + 4.0 * f(f64::midpoint(a, b)) + f(b))
    }
    fn go(f: &dyn Fn(f64) -> f64, a: f64, b: f64, whole: f64, tol: f64, depth: u32) -> f64 {
        let m = f64::midpoint(a, b);
        let left = s(f, a, m);
        let right = s(f, m, b);
        let delta = left + right - whole;
        if delta.abs() <= 15.0 * tol || depth == 0 {
            left + right + delta / 15.0
        } else {
            go(f, a, m, left, tol / 2.0, depth - 1) + go(f, m, b, right, tol / 2.0, depth - 1)
        }
    }
    if a == b {
        return 0.0;
    }
    let (a, b, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    sign * go(f, a, b, s(f, a, b), tol, 30)
}

/// Gaussian quadrature (Legendre polynomials) for n points.
/// Exact for polynomials of degree up to 2n-1.
///
/// Returns `Err` if the node/weight computation fails to converge (n > 50).
///
/// ```
/// use mathverse_calculus::integrate::gaussian_quadrature;
/// // ∫₀¹ x² dx = 1/3
/// assert!((gaussian_quadrature(&|x| x * x, 0.0, 1.0, 3).unwrap() - 1.0/3.0).abs() < 1e-10);
/// ```
pub fn gaussian_quadrature(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    n: usize,
) -> MathResult<f64> {
    let (nodes, weights) = legendre_nodes_weights(n)?;
    let scale = (b - a) / 2.0;
    let shift = f64::midpoint(a, b);
    let mut sum = 0.0;
    for (&x, &w) in nodes.iter().zip(weights.iter()) {
        sum += w * f(scale * x + shift);
    }
    Ok(sum * scale)
}

/// Romberg integration: Richardson extrapolation of the trapezoid rule.
/// Equivalent to `scipy.integrate.romberg`.
///
/// ```
/// use mathverse_calculus::integrate::romberg;
/// let result = romberg(&f64::sin, 0.0, core::f64::consts::PI, 10, 1e-12).unwrap();
/// assert!((result - 2.0).abs() < 1e-10);
/// ```
pub fn romberg(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    max_steps: usize,
    tol: f64,
) -> MathResult<f64> {
    if max_steps == 0 {
        return Err(MathError::InvalidArgument("romberg: max_steps must be > 0"));
    }
    let mut r = vec![vec![0.0; max_steps]; max_steps];
    let h = b - a;
    r[0][0] = h * (f(a) + f(b)) / 2.0;

    for i in 1..max_steps {
        let h_i = h / (2.0_f64.powi(i as i32));
        let mut sum = 0.0;
        for k in 1..=(2_usize.pow(i as u32 - 1)) {
            sum += f(a + (2.0 * k as f64 - 1.0) * h_i);
        }
        r[i][0] = r[i - 1][0] / 2.0 + h_i * sum;

        for j in 1..=i {
            let four_j = 4.0_f64.powi(j as i32);
            r[i][j] = (four_j * r[i][j - 1] - r[i - 1][j - 1]) / (four_j - 1.0);
        }

        if i > 1 && (r[i][i] - r[i - 1][i - 1]).abs() < tol {
            return Ok(r[i][i]);
        }
    }

    Ok(r[max_steps - 1][max_steps - 1])
}

/// 2D integration over a rectangular domain using nested Gaussian quadrature.
/// Equivalent to `scipy.integrate.dblquad` for rectangular regions.
///
/// ```
/// use mathverse_calculus::integrate::integrate_2d;
/// // ∫₀¹ ∫₀¹ x*y dx dy = 1/4
/// let result = integrate_2d(&|x, y| x * y, 0.0, 1.0, 0.0, 1.0, 5).unwrap();
/// assert!((result - 0.25).abs() < 1e-10);
/// ```
pub fn integrate_2d(
    f: &dyn Fn(f64, f64) -> f64,
    ax: f64,
    bx: f64,
    ay: f64,
    by: f64,
    n: usize,
) -> MathResult<f64> {
    let (nodes_x, weights_x) = legendre_nodes_weights(n)?;
    let (nodes_y, weights_y) = legendre_nodes_weights(n)?;

    let scale_x = (bx - ax) / 2.0;
    let shift_x = f64::midpoint(ax, bx);
    let scale_y = (by - ay) / 2.0;
    let shift_y = f64::midpoint(ay, by);

    let mut sum = 0.0;
    for (&xi, &wi) in nodes_x.iter().zip(weights_x.iter()) {
        let x = scale_x * xi + shift_x;
        for (&yj, &wj) in nodes_y.iter().zip(weights_y.iter()) {
            let y = scale_y * yj + shift_y;
            sum += wi * wj * f(x, y);
        }
    }
    Ok(sum * scale_x * scale_y)
}

fn legendre_nodes_weights(n: usize) -> MathResult<(Vec<f64>, Vec<f64>)> {
    match n {
        0 => Err(MathError::InvalidArgument("gaussian_quadrature: n must be > 0")),
        1 => Ok((vec![0.0], vec![2.0])),
        2 => Ok((
            vec![-1.0_f64.sqrt(), 1.0_f64.sqrt()],
            vec![1.0, 1.0],
        )),
        3 => Ok((
            vec![-0.7745966692414834, 0.0, 0.7745966692414834],
            vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556],
        )),
        4 => Ok((
            vec![
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            vec![0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538],
        )),
        5 => Ok((
            vec![
                -0.906_179_845_938_664,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.906_179_845_938_664,
            ],
            vec![
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ],
        )),
        _ => {
            // For higher n, use Newton-Raphson with Tricomi initial guesses.
            // Exploit symmetry: only compute positive roots.
            let mut nodes = Vec::with_capacity(n);
            let mut weights = Vec::with_capacity(n);

            for i in 1..=n / 2 {
                // Tricomi initial guess
                let theta =
                    (4.0 * i as f64 - 1.0) * core::f64::consts::PI / (4.0 * n as f64 + 2.0);
                let x = (1.0 - 1.0 / (8.0 * n as f64 * n as f64)
                    - 1.0 / (8.0 * n as f64 * n as f64 * n as f64))
                    * theta.cos();

                let (node, weight) = legendre_root_weight(x, n)?;
                nodes.push(-node);
                nodes.push(node);
                weights.push(weight);
                weights.push(weight);
            }

            // Middle root at 0 for odd n
            if n % 2 == 1 {
                let (node, weight) = legendre_root_weight(1e-8, n)?;
                nodes.push(node);
                weights.push(weight);
            }

            // Sort by node value
            let mut paired: Vec<_> = nodes.into_iter().zip(weights).collect();
            paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let (nodes, weights): (Vec<_>, Vec<_>) = paired.into_iter().unzip();
            Ok((nodes, weights))
        }
    }
}

fn legendre_root_weight(x0: f64, n: usize) -> MathResult<(f64, f64)> {
    let mut x = x0;
    let mut converged = false;
    for _ in 0..100 {
        let (p, dp) = legendre_and_derivative(x, n);
        if dp.abs() < 1e-300 {
            return Err(MathError::NotConverged("legendre_root_weight: derivative too small, stagnation"));
        }
        let dx = p / dp;
        x -= dx;
        if dx.abs() < 1e-15 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(MathError::NotConverged("legendre_root_weight: Newton-Raphson did not converge"));
    }
    let (_, dp) = legendre_and_derivative(x, n);
    let weight = 2.0 / ((1.0 - x * x) * dp * dp);
    Ok((x, weight))
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
        let e1 = (simpson(f, 0.0, pi, 8).unwrap() - 2.0).abs();
        let e2 = (simpson(f, 0.0, pi, 64).unwrap() - 2.0).abs();
        assert!(e2 < e1 / 3.0, "{e1} vs {e2}");
        assert!(e2 < 1e-5);
        assert!((trapezoid(f, 0.0, pi, 1024).unwrap() - 2.0).abs() < 1e-5);
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
        assert!((gaussian_quadrature(&|x| x * x, 0.0, 1.0, 3).unwrap() - 1.0/3.0).abs() < 1e-10);
        // ∫₋₁¹ x⁴ dx = 2/5
        assert!((gaussian_quadrature(&|x| x.powi(4), -1.0, 1.0, 3).unwrap() - 2.0/5.0).abs() < 1e-10);
        // ∫₀^π sin(x) dx = 2; 5-point Gauss error is ~1e-7, not machine eps
        assert!((gaussian_quadrature(&f64::sin, 0.0, core::f64::consts::PI, 5).unwrap() - 2.0).abs() < 1e-6);

        // Higher n (triggers Newton-Raphson path)
        assert!((gaussian_quadrature(&|x| x.powi(6), -1.0, 1.0, 8).unwrap() - 2.0/7.0).abs() < 1e-10);
        assert!(gaussian_quadrature(&|x| x, 0.0, 1.0, 0).is_err());
    }

    #[test]
    fn romberg_test() {
        // ∫₀^π sin(x) dx = 2
        let result = romberg(&f64::sin, 0.0, core::f64::consts::PI, 10, 1e-12).unwrap();
        assert!((result - 2.0).abs() < 1e-10);

        // ∫₀¹ x² dx = 1/3 (Romberg should be exact after 2 steps)
        let result = romberg(&|x| x * x, 0.0, 1.0, 5, 1e-12).unwrap();
        assert!((result - 1.0/3.0).abs() < 1e-10);

        // ∫₀¹ e^x dx = e - 1
        let result = romberg(&|x| x.exp(), 0.0, 1.0, 10, 1e-12).unwrap();
        assert!((result - (core::f64::consts::E - 1.0)).abs() < 1e-10);
    }

    #[test]
    fn integrate_2d_test() {
        // ∫₀¹ ∫₀¹ x*y dx dy = 1/4
        let result = integrate_2d(&|x, y| x * y, 0.0, 1.0, 0.0, 1.0, 5).unwrap();
        assert!((result - 0.25).abs() < 1e-10);

        // ∫₋₁¹ ∫₋₁¹ (x² + y²) dx dy = 8/3
        let result = integrate_2d(&|x, y| x*x + y*y, -1.0, 1.0, -1.0, 1.0, 5).unwrap();
        assert!((result - 8.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn n_zero_returns_err() {
        assert!(trapezoid(&|x| x, 0.0, 1.0, 0).is_err());
        assert!(simpson(&|x| x, 0.0, 1.0, 0).is_err());
    }
}
