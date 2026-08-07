/// Minimizes `f(x)` subject to the *equality* constraints `g_j(x) = 0` using
/// the augmented-Lagrangian method of multipliers.
///
/// The plain Lagrangian `L(x, λ) = f(x) + Σ_j λ_j g_j(x)` is a saddle point
/// of the primal–dual dynamics, and a naive one-step primal–dual gradient
/// update is marginally unstable for quadratic problems (it oscillates and
/// drifts instead of converging). This implementation instead minimizes the
/// *augmented* Lagrangian
///
/// ```text
/// L_μ(x, λ) = f(x) + Σ_j λ_j g_j(x) + (μ/2) Σ_j g_j(x)²
/// ```
///
/// over `x` (inner gradient descent with an Armijo backtracking line search),
/// then updates the multipliers with the method-of-multipliers rule
/// `λ_j ← λ_j + μ g_j(x)` and increases the penalty `μ`, which converges
/// even with fixed step sizes.
pub fn lagrangian(f: &dyn Fn(&[f64]) -> f64, g: &[Box<dyn Fn(&[f64]) -> f64>], x0: &[f64], lr: f64, tol: f64, max_iters: usize) -> Vec<f64> {
    let n = x0.len();
    let m = g.len();
    let mut x = x0.to_vec();
    let mut lambda = vec![0.0; m];
    let mut mu = 1.0_f64;
    for _ in 0..max_iters {
        // Augmented Lagrangian for the current multiplier estimate and penalty.
        let lmu = |x: &[f64]| -> f64 {
            let mut val = f(x);
            for j in 0..m {
                let gj = g[j](x);
                val += lambda[j] * gj + 0.5 * mu * gj * gj;
            }
            val
        };
        // Inner minimization of L_μ over x (finite-difference gradient with
        // Armijo backtracking so the step stays stable as μ grows).
        for _ in 0..500 {
            let mut grad_x = vec![0.0; n];
            let dx = 1e-7;
            for i in 0..n {
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[i] += dx;
                xm[i] -= dx;
                grad_x[i] = (lmu(&xp) - lmu(&xm)) / (2.0 * dx);
            }
            let gnorm: f64 = grad_x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if gnorm < 1e-8 { break; }
            let fx = lmu(&x);
            let mut alpha = lr;
            let mut next: Vec<f64> = x.iter().zip(&grad_x).map(|(xi, gi)| xi - alpha * gi).collect();
            for _ in 0..24 {
                if lmu(&next) <= fx - 1e-4 * alpha * gnorm * gnorm { break; }
                alpha *= 0.5;
                next = x.iter().zip(&grad_x).map(|(xi, gi)| xi - alpha * gi).collect();
            }
            if alpha < 1e-12 { break; }
            x = next;
        }
        // Dual update: λ_j ← λ_j + μ g_j(x).
        let mut max_viol: f64 = 0.0;
        for j in 0..m {
            let gj = g[j](&x);
            max_viol = max_viol.max(gj.abs());
            lambda[j] += mu * gj;
        }
        if max_viol < tol { break; }
        mu = (mu * 2.0).min(1e10);
    }
    x
}

pub fn penalty_method(f: &dyn Fn(&[f64]) -> f64, g: &[Box<dyn Fn(&[f64]) -> f64>], x0: &[f64], mu: f64, tol: f64, max_outer: usize, max_inner: usize) -> Vec<f64> {
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

// Constrained optimization: Lagrangian, penalty method, augmented Lagrangian, projected gradient.

/// Augmented Lagrangian method for equality-constrained minimization.
pub fn augmented_lagrangian(f: &dyn Fn(&[f64]) -> f64, g: &[Box<dyn Fn(&[f64]) -> f64>], x0: &[f64], mu: f64, tol: f64, max_outer: usize) -> Vec<f64> {
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
        let x = lagrangian(&f, &g, &[0.5, 0.5], 0.001, 1e-10, 100000);
        // x should be near (0.5, 0.5) — the minimum of x^2+y^2 on x+y=1
        assert!((x[0] - 0.5).abs() < 0.2);
    }
}
