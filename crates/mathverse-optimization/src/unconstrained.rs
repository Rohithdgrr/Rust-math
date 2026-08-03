pub fn newton_min(f: &dyn Fn(&[f64]) -> f64, grad: &dyn Fn(&[f64]) -> Vec<f64>, hessian: &dyn Fn(&[f64]) -> Vec<Vec<f64>>, x0: &[f64], tol: f64, max_iters: usize) -> Vec<f64> {
    let mut x = x0.to_vec();
    for _ in 0..max_iters {
        let g = grad(&x);
        let h = hessian(&x);
        let n = x.len();
        let mut det = 1.0;
        let mut aug: Vec<Vec<f64>> = (0..n).map(|i| {
            let mut row = h[i].clone();
            row.push(-g[i]);
            row
        }).collect();
        for col in 0..n {
            let mut max_row = col;
            for r in col+1..n { if aug[r][col].abs() > aug[max_row][col].abs() { max_row = r; } }
            aug.swap(col, max_row);
            if aug[col][col].abs() < 1e-15 { return x; }
            for r in col+1..n {
                let f = aug[r][col] / aug[col][col];
                for c in col..=n { aug[r][c] -= f * aug[col][c]; }
            }
        }
        let mut dx = vec![0.0; n];
        for i in (0..n).rev() { dx[i] = (aug[i][n] - (i+1..n).map(|j| aug[i][j]*dx[j]).sum::<f64>()) / aug[i][i]; }
        let mut next = x.clone();
        for i in 0..n { next[i] += dx[i]; }
        if next.iter().zip(&x).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < tol { return next; }
        x = next;
    }
    x
}

// Unconstrained optimization: Newton's method, BFGS, conjugate gradient.

/// BFGS quasi-Newton optimizer.
pub fn bfgs(grad: &dyn Fn(&[f64]) -> Vec<f64>, x0: &[f64], tol: f64, max_iters: usize) -> Vec<f64> {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut h = vec![vec![0.0; n]; n];
    for i in 0..n { h[i][i] = 1.0; }
    let mut g = grad(&x);
    for _ in 0..max_iters {
        let mut p = vec![0.0; n];
        for i in 0..n { p[i] = -(0..n).map(|j| h[i][j] * g[j]).sum::<f64>(); }
        let mut next = x.clone();
        for i in 0..n { next[i] += p[i]; }
        let g_new = grad(&next);
        let s: Vec<f64> = next.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        if sy.abs() > 1e-30 {
            let rho = 1.0 / sy;
            let mut new_h = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    new_h[i][j] = h[i][j] + rho * (s[i] * s[j] + y[i] * y[j])
                        - rho * rho * y[i] * y[j] * (0..n).map(|k| s[k] * y[k]).sum::<f64>();
                }
            }
            h = new_h;
        }
        if g_new.iter().map(|v| v * v).sum::<f64>().sqrt() < tol { return next; }
        x = next;
        g = g_new;
    }
    x
}

pub fn conjugate_gradient(a: &[Vec<f64>], b: &[f64], x0: &[f64], tol: f64, max_iters: usize) -> Vec<f64> {
    let n = b.len();
    let mut x = x0.to_vec();
    let mut r: Vec<f64> = b.iter().enumerate().map(|(i, bi)| bi - (0..n).map(|j| a[i][j] * x[j]).sum::<f64>()).collect();
    let mut p = r.clone();
    for _ in 0..max_iters {
        let ap: Vec<f64> = (0..n).map(|i| (0..n).map(|j| a[i][j] * p[j]).sum()).collect();
        let r_dot_r: f64 = r.iter().map(|v| v * v).sum();
        let p_ap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        if p_ap.abs() < 1e-30 { break; }
        let alpha = r_dot_r / p_ap;
        let mut next = x.clone();
        for i in 0..n { next[i] += alpha * p[i]; }
        let mut r_new = r.clone();
        for i in 0..n { r_new[i] -= alpha * ap[i]; }
        let r_new_dot: f64 = r_new.iter().map(|v| v * v).sum();
        if r_new_dot.sqrt() < tol { return next; }
        let beta = r_new_dot / r_dot_r;
        let mut new_p = r_new.clone();
        for i in 0..n { new_p[i] += beta * p[i]; }
        x = next;
        r = r_new;
        p = new_p;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfgs_test() {
        let x = bfgs(&|x: &[f64]| x.iter().map(|v| 2.0 * v).collect(), &[10.0, -10.0], 1e-6, 10000);
        assert!(x.iter().all(|v| v.abs() < 1e-4));
    }
}
