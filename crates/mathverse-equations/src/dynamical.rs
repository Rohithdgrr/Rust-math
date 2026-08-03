//! Discrete dynamical systems: fixed-point iteration, cobweb, Lyapunov exponents, period detection.

/// Fixed-point iteration: converges `x_{n+1} = g(x_n)`.
pub fn fixed_point(g: impl Fn(f64) -> f64, x0: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut x = x0;
    for _ in 0..max_iter {
        let x_new = g(x);
        if (x_new - x).abs() < tol { return Some(x_new); }
        x = x_new;
    }
    Some(x)
}

pub fn iterate_map(f: impl Fn(f64) -> f64, x0: f64, n: usize) -> Vec<f64> {
    let mut result = vec![x0];
    let mut x = x0;
    for _ in 0..n {
        x = f(x);
        result.push(x);
    }
    result
}

pub fn cobweb(g: impl Fn(f64) -> f64, x0: f64, n: usize) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    let mut x = x0;
    for _ in 0..n {
        let y = g(x);
        result.push((x, y));
        result.push((y, y));
        x = y;
    }
    result
}

pub fn lyapunov_exponent(f: impl Fn(f64) -> f64, df: impl Fn(f64) -> f64, x0: f64, n: usize) -> f64 {
    let mut x = x0;
    let mut sum = 0.0;
    for _ in 0..n {
        let d = df(x).abs();
        if d < 1e-30 { return f64::NEG_INFINITY; }
        sum += d.ln();
        x = f(x);
    }
    sum / n as f64
}

pub fn iterate_to_fixed_point(g: impl Fn(f64) -> f64, x0: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut x = x0;
    for _ in 0..max_iter {
        let x_new = g(x);
        if (x_new - x).abs() < tol { return Some(x_new); }
        x = x_new;
    }
    None
}

pub fn orbit(g: impl Fn(f64) -> f64, x0: f64, n: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(n + 1);
    let mut x = x0;
    result.push(x);
    for _ in 0..n {
        x = g(x);
        result.push(x);
    }
    result
}

pub fn period(g: impl Fn(f64) -> f64, x0: f64, tol: f64, max_period: usize) -> Option<usize> {
    let mut x = x0;
    for _ in 0..100 { x = g(x); } // transient
    let y = x;
    for p in 1..=max_period {
        x = g(x);
        if (x - y).abs() < tol { return Some(p); }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_pt() {
        let x = fixed_point(|x| (x + 2.0 / x) / 2.0, 1.0, 1e-15, 100).unwrap();
        assert!((x - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn lyapunov() {
        // r=4 logistic map is chaotic, Lyapunov exponent = ln(2) ≈ 0.693
        let l = lyapunov_exponent(|x| 4.0 * x * (1.0 - x), |x| 4.0 - 8.0 * x, 0.3, 10000);
        assert!(l > 0.5);
        assert!((l - 2.0_f64.ln()).abs() < 0.1);
    }

    #[test]
    fn lyapunov_stable() {
        // r=2 logistic map converges to x=0.5, negative Lyapunov exponent
        let l = lyapunov_exponent(|x| 2.0 * x * (1.0 - x), |x| 2.0 - 4.0 * x, 0.3, 1000);
        assert!(l < 0.0);
    }

    #[test]
    fn period_test() {
        // r=3.2 logistic map has a stable period-2 orbit
        let p = period(|x| 3.2 * x * (1.0 - x), 0.3, 1e-10, 10);
        assert_eq!(p, Some(2));
    }
}
