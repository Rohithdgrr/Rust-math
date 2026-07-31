//! Numerical helpers: integration, differentiation, and numerical methods.

use mathverse_core::error::{MathError, MathResult};

/// Numerical integration methods.
pub struct Integration;

impl Integration {
    /// Trapezoidal rule: ∫[a,b] f(x) dx ≈ (b-a)/2 * (f(a) + f(b)).
    pub fn trapezoidal(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        
        for i in 1..n {
            let x = a + i as f64 * h;
            sum += f(x);
        }
        
        sum * h
    }

    /// Simpson's rule: ∫[a,b] f(x) dx ≈ (b-a)/6 * (f(a) + 4f((a+b)/2) + f(b)).
    pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        if n % 2 != 0 {
            return Self::trapezoidal(f, a, b, n);
        }
        
        let h = (b - a) / n as f64;
        let mut sum = f(a) + f(b);
        
        for i in 1..n {
            let x = a + i as f64 * h;
            let weight = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * f(x);
        }
        
        sum * h / 3.0
    }

    /// Simpson's 3/8 rule: ∫[a,b] f(x) dx ≈ (b-a)/8 * (f(a) + 3f(a+h) + 3f(a+2h) + f(b)).
    pub fn simpson_3_8(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        if n % 3 != 0 {
            return Self::simpson(f, a, b, n);
        }
        
        let h = (b - a) / n as f64;
        let mut sum = f(a) + f(b);
        
        for i in 1..n {
            let x = a + i as f64 * h;
            let weight = if i % 3 == 0 { 2.0 } else { 3.0 };
            sum += weight * f(x);
        }
        
        sum * 3.0 * h / 8.0
    }

    /// Midpoint rule: ∫[a,b] f(x) dx ≈ (b-a) * f((a+b)/2).
    pub fn midpoint(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.0;
        
        for i in 0..n {
            let x = a + (i as f64 + 0.5) * h;
            sum += f(x);
        }
        
        sum * h
    }

    /// Gaussian quadrature (Legendre polynomials).
    pub fn gaussian_legendre(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let (nodes, weights) = Self::legendre_nodes_weights(n);
        
        let mut sum = 0.0;
        for (&x, &w) in nodes.iter().zip(weights.iter()) {
            let transformed = 0.5 * (b - a) * x + 0.5 * (a + b);
            sum += w * f(transformed);
        }
        
        0.5 * (b - a) * sum
    }

    /// Legendre polynomial nodes and weights.
    fn legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
        match n {
            1 => (vec![0.0], vec![2.0]),
            2 => (vec![-0.577350269, 0.577350269], vec![1.0, 1.0]),
            3 => (vec![-0.774596669, 0.0, 0.774596669], vec![0.555555556, 0.888888889, 0.555555556]),
            4 => (vec![-0.861136312, -0.339981044, 0.339981044, 0.861136312],
                vec![0.347854845, 0.652145155, 0.652145155, 0.347854845]),
            5 => (vec![-0.906179846, -0.538469310, 0.0, 0.538469310, 0.906179846],
                vec![0.236926885, 0.478628670, 0.568888889, 0.478628670, 0.236926885]),
            _ => {
                // Fallback to trapezoidal for higher n
                let nodes: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * i as f64 / (n - 1) as f64).collect();
                let weights = vec![2.0 / n as f64; n];
                (nodes, weights)
            }
        }
    }

    /// Romberg integration (Richardson extrapolation).
    pub fn romberg(f: impl Fn(f64) -> f64, a: f64, b: f64, max_levels: usize) -> f64 {
        let mut r = vec![vec![0.0; max_levels]; max_levels];
        
        // First column: trapezoidal rule with increasing subdivisions
        for k in 0..max_levels {
            let n = 2_usize.pow(k as u32);
            r[k][0] = Self::trapezoidal(&f, a, b, n);
        }
        
        // Richardson extrapolation
        for j in 1..max_levels {
            for k in j..max_levels {
                let factor = 4.0_f64.powi(j as i32);
                r[k][j] = (factor * r[k][j - 1] - r[k - 1][j - 1]) / (factor - 1.0);
            }
        }
        
        r[max_levels - 1][max_levels - 1]
    }

    /// Adaptive Simpson's rule.
    pub adaptive_simpson(
        f: impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
        max_depth: usize,
    ) -> f64 {
        Self::adaptive_simpson_recursive(&f, a, b, tolerance, max_depth, f(a), f(b), f((a + b) / 2.0))
    }

    fn adaptive_simpson_recursive(
        f: &impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
        depth: usize,
        fa: f64,
        fb: f64,
        fm: f64,
    ) -> f64 {
        let h = b - a;
        let m = (a + b) / 2.0;
        
        let whole = h / 6.0 * (fa + 4.0 * fm + fb);
        
        let fl = f((a + m) / 2.0);
        let fr = f((m + b) / 2.0);
        
        let left = h / 12.0 * (fa + 4.0 * fl + fm);
        let right = h / 12.0 * (fm + 4.0 * fr + fb);
        
        if depth == 0 || (left + right - whole).abs() < 15.0 * tolerance {
            left + right + (left + right - whole) / 15.0
        } else {
            Self::adaptive_simpson_recursive(f, a, m, tolerance / 2.0, depth - 1, fa, fm, fl)
                + Self::adaptive_simpson_recursive(f, m, b, tolerance / 2.0, depth - 1, fm, fb, fr)
        }
    }

    /// Monte Carlo integration.
    pub fn monte_carlo(f: impl Fn(f64) -> f64, a: f64, b: f64, samples: usize) -> (f64, f64) {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for _ in 0..samples {
            let x = a + (b - a) * rand::random::<f64>();
            let y = f(x);
            sum += y;
            sum_sq += y * y;
        }
        
        let mean = sum / samples as f64;
        let variance = (sum_sq / samples as f64 - mean * mean).max(0.0);
        let error = (variance / samples as f64).sqrt() * (b - a);
        
        (mean * (b - a), error)
    }

    /// Improper integral from a to infinity.
    pub fn improper_infinite(f: impl Fn(f64) -> f64, a: f64, tolerance: f64) -> MathResult<f64> {
        let mut total = 0.0;
        let mut b = a + 1.0;
        let mut contribution;
        
        loop {
            contribution = Self::simpson(&f, a, b, 100);
            total += contribution;
            
            if contribution.abs() < tolerance {
                return Ok(total);
            }
            
            a = b;
            b *= 2.0;
            
            if b > 1e10 {
                return Err(MathError::InvalidArgument("integral does not converge"));
            }
        }
    }
}

/// Numerical differentiation methods.
pub struct Differentiation;

impl Differentiation {
    /// Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h.
    pub fn forward_difference(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x)) / h
    }

    /// Backward difference: f'(x) ≈ (f(x) - f(x-h)) / h.
    pub fn backward_difference(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x) - f(x - h)) / h
    }

    /// Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h).
    pub fn central_difference(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    /// Second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h².
    pub fn second_derivative(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
    }

    /// Third derivative: f'''(x) ≈ (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (2h³).
    pub fn third_derivative(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + 2.0 * h) - 2.0 * f(x + h) + 2.0 * f(x - h) - f(x - 2.0 * h)) / (2.0 * h * h * h)
    }

    /// Fourth derivative: f''''(x) ≈ (f(x+2h) - 4f(x+h) + 6f(x) - 4f(x-h) + f(x-2h)) / h⁴.
    pub fn fourth_derivative(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + 2.0 * h) - 4.0 * f(x + h) + 6.0 * f(x) - 4.0 * f(x - h) + f(x - 2.0 * h)) / (h * h * h * h)
    }

    /// Gradient of multivariate function.
    pub fn gradient(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
        let mut grad = Vec::with_capacity(x.len());
        
        for i in 0..x.len() {
            let mut x_plus = x.to_vec();
            x_plus[i] += h;
            let f_plus = f(&x_plus);
            
            let mut x_minus = x.to_vec();
            x_minus[i] -= h;
            let f_minus = f(&x_minus);
            
            grad.push((f_plus - f_minus) / (2.0 * h));
        }
        
        grad
    }

    /// Hessian matrix of multivariate function.
    pub fn hessian(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut hess = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..=i {
                let mut x_pp = x.to_vec();
                x_pp[i] += h;
                x_pp[j] += h;
                let f_pp = f(&x_pp);
                
                let mut x_pm = x.to_vec();
                x_pp[i] += h;
                x_pm[j] -= h;
                let f_pm = f(&x_pm);
                
                let mut x_mp = x.to_vec();
                x_mp[i] -= h;
                x_mp[j] += h;
                let f_mp = f(&x_mp);
                
                let mut x_mm = x.to_vec();
                x_mm[i] -= h;
                x_mm[j] -= h;
                let f_mm = f(&x_mm);
                
                let h_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
                hess[i][j] = h_ij;
                hess[j][i] = h_ij;
            }
        }
        
        hess
    }

    /// Jacobian matrix of vector function.
    pub fn jacobian(f: impl Fn(&[f64]) -> Vec<f64>, x: &[f64], h: f64) -> Vec<Vec<f64>> {
        let m = f(x).len();
        let n = x.len();
        let mut jac = vec![vec![0.0; n]; m];
        
        for j in 0..n {
            let mut x_plus = x.to_vec();
            x_plus[j] += h;
            let f_plus = f(&x_plus);
            
            let mut x_minus = x.to_vec();
            x_minus[j] -= h;
            let f_minus = f(&x_minus);
            
            for i in 0..m {
                jac[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * h);
            }
        }
        
        jac
    }

    /// Partial derivative.
    pub fn partial_derivative(f: impl Fn(&[f64]) -> f64, x: &[f64], var: usize, h: f64) -> f64 {
        let mut x_plus = x.to_vec();
        x_plus[var] += h;
        let f_plus = f(&x_plus);
        
        let mut x_minus = x.to_vec();
        x_minus[var] -= h;
        let f_minus = f(&x_minus);
        
        (f_plus - f_minus) / (2.0 * h)
    }

    /// Directional derivative.
    pub fn directional_derivative(f: impl Fn(&[f64]) -> f64, x: &[f64], direction: &[f64], h: f64) -> f64 {
        let n = x.len();
        let mut x_plus = Vec::with_capacity(n);
        
        for i in 0..n {
            x_plus.push(x[i] + h * direction[i]);
        }
        
        let f_plus = f(&x_plus);
        let f_x = f(x);
        
        (f_plus - f_x) / h
    }
}

/// Root finding methods.
pub struct RootFinding;

impl RootFinding {
    /// Bisection method for f(x) = 0.
    pub fn bisection(
        f: impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
        max_iterations: usize,
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

    /// Newton-Raphson method.
    pub fn newton_raphson(
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
        x0: f64,
        tolerance: f64,
        max_iterations: usize,
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

    /// Secant method (derivative-free).
    pub fn secant(
        f: impl Fn(f64) -> f64,
        x0: f64,
        x1: f64,
        tolerance: f64,
        max_iterations: usize,
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

    /// Fixed point iteration: x = g(x).
    pub fn fixed_point(
        g: impl Fn(f64) -> f64,
        x0: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> MathResult<f64> {
        let mut x = x0;
        
        for _ in 0..max_iterations {
            let x_new = g(x);
            
            if (x_new - x).abs() < tolerance {
                return Ok(x_new);
            }
            
            x = x_new;
        }
        
        Ok(x)
    }
}

/// Optimization methods.
pub struct Optimization;

impl Optimization {
    /// Golden section search for minimum of f(x) on [a, b].
    pub fn golden_section_min(
        f: impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        tolerance: f64,
    ) -> f64 {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let resphi = 2.0 - phi;
        
        let mut a = a;
        let mut b = b;
        
        let mut c = b - resphi * (b - a);
        let mut d = a + resphi * (b - a);
        
        while (b - a).abs() > tolerance {
            if f(c) < f(d) {
                b = d;
                d = c;
                c = b - resphi * (b - a);
            } else {
                a = c;
                c = d;
                d = a + resphi * (b - a);
            }
        }
        
        (a + b) / 2.0
    }

    /// Gradient descent for multivariate function.
    pub fn gradient_descent(
        f: impl Fn(&[f64]) -> f64,
        grad: impl Fn(&[f64]) -> Vec<f64>,
        x0: &[f64],
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> (Vec<f64>, usize, f64) {
        let mut x = x0.to_vec();
        
        for iteration in 0..max_iterations {
            let gradient = grad(&x);
            let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < tolerance {
                return (x, iteration, grad_norm);
            }
            
            for i in 0..x.len() {
                x[i] -= learning_rate * gradient[i];
            }
        }
        
        let gradient = grad(&x);
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        
        (x, max_iterations, grad_norm)
    }

    /// Line search (backtracking).
    pub fn line_search(
        f: impl Fn(&[f64]) -> f64,
        x: &[f64],
        direction: &[f64],
        initial_step: f64,
        alpha: f64,
        beta: f64,
    ) -> f64 {
        let mut step = initial_step;
        
        loop {
            let x_new: Vec<f64> = x.iter().zip(direction.iter())
                .map(|(&xi, &di)| xi + step * di)
                .collect();
            
            let f_new = f(&x_new);
            let f_x = f(x);
            
            let grad_f_x = Differentiation::gradient(&f, x, 1e-6);
            let directional_derivative: f64 = grad_f_x.iter().zip(direction.iter())
                .map(|(&g, &d)| g * d)
                .sum();
            
            if f_new <= f_x + alpha * step * directional_derivative {
                return step;
            }
            
            step *= beta;
            
            if step < 1e-15 {
                return step;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trapezoidal() {
        let result = Integration::trapezoidal(|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_simpson() {
        let result = Integration::simpson(|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_legendre() {
        let result = Integration::gaussian_legendre(|x| x * x, -1.0, 1.0, 3);
        assert!((result - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_central_difference() {
        let result = Differentiation::central_difference(|x| x * x, 2.0, 1e-6);
        assert!((result - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_second_derivative() {
        let result = Differentiation::second_derivative(|x| x * x, 2.0, 1e-6);
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient() {
        let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
        let grad = Differentiation::gradient(&f, &[1.0, 2.0], 1e-6);
        
        assert!((grad[0] - 2.0).abs() < 1e-5);
        assert!((grad[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_bisection() {
        let root = RootFinding::bisection(|x| x * x - 4.0, 0.0, 3.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_newton_raphson() {
        let root = RootFinding::newton_raphson(|x| x * x - 4.0, |x| 2.0 * x, 1.0, 1e-10, 100).unwrap();
        assert!((root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_golden_section() {
        let min = Optimization::golden_section_min(|x| (x - 2.0).powi(2), 0.0, 4.0, 1e-10);
        assert!((min - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_descent() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)];
        
        let (x, iterations, _) = Optimization::gradient_descent(&f, &grad, &[0.0, 0.0], 0.1, 100, 1e-10);
        
        assert!(iterations < 100);
        assert!((x[0] - 1.0).abs() < 0.1);
        assert!((x[1] - 2.0).abs() < 0.1);
    }
}
