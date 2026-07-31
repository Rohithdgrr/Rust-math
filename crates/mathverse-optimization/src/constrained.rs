pub fn lagrangian(f: &dyn Fn(&[f64]) -> f64, g: &[dyn Fn(&[f64]) -> f64], x0: &[f64], lr: f64, tol: f64, max_iters: usize) -> Vec<f64> {
    let n = x0.len();
    let m = g.len();
    let mut x = x0.to_vec();
    let mut lambda = vec![0.0; m];
    for _ in 0..max_iters {
        let gx: Vec<f64> = g.iter().map(|gi| gi(&x)).collect();
        let mut grad_x = vec![0.0; n];
        for i in 0..n {
            let dx = 1e-6;
            let mut f_plus = f(&x);
            let mut f_minus = f(&x);
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += dx;
            x_minus[i] -= dx;
            f_plus = f(&x_plus);
            f_minus = f(&x_minus);
            grad_x[i] = (f_plus - f_minus) / (2.0 * dx);
            for j in 0..m {
                let mut g_plus = x.clone();
                let mut g_minus = x.clone();
                g_plus[i] += dx;
                g_minus[i] -= dx;
                grad_x[i] -= lambda[j] * (g[j](&g_plus) - g[j](&g_minus)) / (2.0 * dx);
            }
        }
        let mut next = x.clone();
        for i in 0..n { next[i] -= lr * grad_x[i]; }
        for j in 0..m { lambda[j] += lr * g[j](&next); }
        if next.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < tol { return next; }
        x = next;
    }
    x
}

pub fn penalty_method(f: &dyn Fn(&[f64]) -> f64, g: &[dyn Fn(&[f64]) -> f64], x0: &[f64], mu: f64, tol: f64, max_outer: usize, max_inner: usize) -> Vec<f64> {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut mu = mu;
    for _ in 0..max_outer {
        let penalized = |x: &[f64]| -> f64 {
            let mut val = f(x);
            for gi in g { let v = gi(x); if v > 0.0 { val += mu * v * v; } }
            val
        };
        x = crate::gradient::gradient_descent(&|x: &[f64]| {
            let mut grad = vec![0.0; n];
            let dx = 1e-6;
            for i in 0..n {
                let mut xp = x.to_vec();
                let mut xm = x.to_vec();
                xp[i] += dx;
                xm[i] -= dx;
                grad[i] = (penalized(&xp) - penalized(&xm)) / (2.0 * dx);
            }
            grad
        }, &x, 0.01, 1e-10, max_inner);
        mu *= 10.0;
        if g.iter().all(|gi| gi(&x).abs() < tol) { break; }
    }
    x
}

pub fn augmented_lagrangian(f: &dyn Fn(&[f64]) -> f64, g: &[dyn Fn(&[f64]) -> f64], x0: &[f64], mu: f64, tol: f64, max_outer: usize) -> Vec<f64> {
    let n = x0.len();
    let m = g.len();
    let mut x = x0.to_vec();
    let mut lambda = vec![0.0; m];
    let mut mu = mu;
    for _ in 0..max_outer {
        let al = |x: &[f64]| -> f64 {
            let mut val = f(x);
            for j in 0..m {
                let gj = g[j](x);
                val += lambda[j] * gj + mu / 2.0 * gj * gj;
            }
            val
        };
        x = crate::gradient::gradient_descent(&|x: &[f64]| {
            let mut grad = vec![0.0; n];
            let dx = 1e-6;
            for i in 0..n {
                let mut xp = x.to_vec();
                let mut xm = x.to_vec();
                xp[i] += dx;
                xm[i] -= dx;
                grad[i] = (al(&xp) - al(&xm)) / (2.0 * dx);
            }
            grad
        }, &x, 0.01, 1e-10, 1000);
        for j in 0..m { lambda[j] += mu * g[j](&x); }
        mu *= 2.0;
        if g.iter().all(|gi| gi(&x).abs() < tol) { break; }
    }
    x
}

pub fn project_gradient(x: &[f64], grad: &[f64], bounds: &[(f64, f64)], lr: f64) -> Vec<f64> {
    x.iter().zip(grad).zip(bounds).map(|((xi, gi), &(lo, hi))| {
        let next = xi - lr * gi;
        next.clamp(lo, hi)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrangian_test() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let g: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![Box::new(|x| x[0] + x[1] - 1.0)];
        let x = lagrangian(&f, &g, &[0.5, 0.5], 0.01, 1e-8, 10000);
        assert!((x[0] - 0.5).abs() < 0.1);
    }
}
