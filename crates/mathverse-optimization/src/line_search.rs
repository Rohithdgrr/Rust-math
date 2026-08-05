//! Line search methods: backtracking, Wolfe conditions, Armijo, golden section, Fibonacci.

/// Backtracking line search satisfying the Armijo sufficient decrease condition.
pub fn backtracking(f: &dyn Fn(&[f64]) -> f64, grad: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], direction: &[f64], alpha0: f64, rho: f64, c: f64) -> f64 {
    let fx = f(x);
    let g: f64 = x.iter().zip(grad(x)).zip(direction).map(|((_xi, gi), di)| gi * di).sum();
    let mut alpha = alpha0;
    loop {
        let mut next = x.to_vec();
        for i in 0..x.len() { next[i] += alpha * direction[i]; }
        if f(&next) <= fx + c * alpha * g { break; }
        alpha *= rho;
        if alpha < 1e-16 { break; }
    }
    alpha
}

pub fn wolfe_line_search(f: &dyn Fn(&[f64]) -> f64, grad: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], direction: &[f64], alpha0: f64, c1: f64, c2: f64) -> f64 {
    let fx = f(x);
    let gx: f64 = grad(x).iter().zip(direction).map(|(g, d)| g * d).sum();
    let (mut a_lo, mut a_hi) = (0.0, alpha0);
    let mut alpha = alpha0;
    for _ in 0..50 {
        let mut next = x.to_vec();
        for i in 0..x.len() { next[i] += alpha * direction[i]; }
        let f_alpha = f(&next);
        let g_alpha: f64 = grad(&next).iter().zip(direction).map(|(g, d)| g * d).sum();
        if f_alpha > fx + c1 * alpha * gx { a_hi = alpha; }
        else if g_alpha < c2 * gx { a_lo = alpha; }
        else { return alpha; }
        alpha = (a_lo + a_hi) / 2.0;
    }
    alpha
}

pub fn armijo(f: &dyn Fn(&[f64]) -> f64, grad: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], direction: &[f64], alpha0: f64, c: f64, beta: f64) -> f64 {
    let fx = f(x);
    let gx: f64 = grad(x).iter().zip(direction).map(|(g, d)| g * d).sum();
    let mut alpha = alpha0;
    loop {
        let mut next = x.to_vec();
        for i in 0..x.len() { next[i] += alpha * direction[i]; }
        if f(&next) <= fx + c * alpha * gx { break; }
        alpha *= beta;
    }
    alpha
}

pub fn golden_section_search(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64) -> f64 {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let (mut l, mut r) = (a, b);
    while (r - l).abs() > tol {
        let x1 = r - (r - l) / phi;
        let x2 = l + (r - l) / phi;
        if f(x1) < f(x2) { r = x2; } else { l = x1; }
    }
    (l + r) / 2.0
}

pub fn fibonacci_search(_f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let (mut l, mut r) = (a, b);
    let (mut x1, mut x2) = (0.0, 0.0);
    for i in (1..n).rev() {
        if x1 < x2 { r = x2; x2 = x1; x1 = l + (r - l) * (i - 1) as f64 / (i + 1) as f64; }
        else { l = x1; x1 = x2; x2 = r - (r - l) * (i - 1) as f64 / (i + 1) as f64; }
    }
    (l + r) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backtracking_test() {
        let alpha = backtracking(&|x: &[f64]| x[0]*x[0], &|x: &[f64]| vec![2.0*x[0]], &[2.0], &[-1.0], 1.0, 0.5, 0.001);
        assert!(alpha > 0.0 && alpha <= 1.0);
    }

    #[test]
    fn golden_test() {
        let x = golden_section_search(|x| (x-3.0).powi(2), 0.0, 6.0, 1e-8);
        assert!((x - 3.0).abs() < 1e-6);
    }
}
