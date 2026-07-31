//! Linear regression, Ridge, and Lasso.

use mathverse_core::error::{MathError, MathResult};

/// Linear regression result.
#[derive(Debug, Clone)]
pub struct LinearResult {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub r_squared: f64,
    pub residuals: Vec<f64>,
}

/// Ordinary least squares linear regression.
/// `x`: [n_samples × n_features], `y`: [n_samples].
pub fn fit(x: &[Vec<f64>], y: &[f64]) -> MathResult<LinearResult> {
    let n = y.len();
    let p = x[0].len();
    // Build X^T X and X^T y (with intercept column)
    let mut xtx = vec![vec![0.0; p + 1]; p + 1];
    let mut xty = vec![0.0; p + 1];
    for i in 0..n {
        xtx[0][0] += 1.0;
        for j in 0..p {
            xtx[0][j + 1] += x[i][j];
            xtx[j + 1][0] += x[i][j];
            xty[j + 1] += x[i][j] * y[i];
            for k in 0..p {
                xtx[j + 1][k + 1] += x[i][j] * x[i][k];
            }
        }
        xty[0] += y[i];
    }
    let beta = solve_symmetric(&xtx, &xty)?;
    let intercept = beta[0];
    let coefficients = beta[1..].to_vec();
    let mut residuals = Vec::with_capacity(n);
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        let pred = intercept + x[i].iter().zip(&coefficients).map(|(xi, ci)| xi * ci).sum::<f64>();
        let r = y[i] - pred;
        residuals.push(r);
        ss_res += r * r;
        ss_tot += (y[i] - y_mean).powi(2);
    }
    let r_squared = if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot };
    Ok(LinearResult { coefficients, intercept, r_squared, residuals })
}

/// Predict using fitted coefficients.
pub fn predict(x: &[Vec<f64>], coefficients: &[f64], intercept: f64) -> Vec<f64> {
    x.iter().map(|row| intercept + row.iter().zip(coefficients).map(|(xi, ci)| xi * ci).sum::<f64>()).collect()
}

/// Ridge regression (L2 penalty).
pub fn fit_ridge(x: &[Vec<f64>], y: &[f64], alpha: f64) -> MathResult<LinearResult> {
    let n = y.len();
    let p = x[0].len();
    let mut xtx = vec![vec![0.0; p + 1]; p + 1];
    let mut xty = vec![0.0; p + 1];
    for i in 0..n {
        xtx[0][0] += 1.0;
        for j in 0..p {
            xtx[0][j + 1] += x[i][j];
            xtx[j + 1][0] += x[i][j];
            xty[j + 1] += x[i][j] * y[i];
            for k in 0..p {
                xtx[j + 1][k + 1] += x[i][j] * x[i][k];
            }
        }
        xty[0] += y[i];
    }
    // Add L2 penalty (skip intercept)
    for j in 1..=p {
        xtx[j][j] += alpha;
    }
    let beta = solve_symmetric(&xtx, &xty)?;
    let intercept = beta[0];
    let coefficients = beta[1..].to_vec();
    let mut residuals = Vec::with_capacity(n);
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        let pred = intercept + x[i].iter().zip(&coefficients).map(|(xi, ci)| xi * ci).sum::<f64>();
        let r = y[i] - pred;
        residuals.push(r);
        ss_res += r * r;
        ss_tot += (y[i] - y_mean).powi(2);
    }
    let r_squared = if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot };
    Ok(LinearResult { coefficients, intercept, r_squared, residuals })
}

/// Lasso regression (L1 penalty) via coordinate descent.
pub fn fit_lasso(x: &[Vec<f64>], y: &[f64], alpha: f64, max_iters: usize, tol: f64) -> MathResult<LinearResult> {
    let n = y.len();
    let p = x[0].len();
    let mut coef = vec![0.0; p];
    let mut intercept = y.iter().sum::<f64>() / n as f64;
    // Precompute column means and residual
    let mut residual: Vec<f64> = y.iter().enumerate().map(|(i, &yi)| {
        yi - intercept - x[i].iter().zip(&coef).map(|(xi, ci)| xi * ci).sum::<f64>()
    }).collect();
    for _iter in 0..max_iters {
        let mut max_change = 0.0;
        for j in 0..p {
            let xj: Vec<f64> = x.iter().map(|row| row[j]).collect();
            let rho: f64 = xj.iter().zip(&residual).map(|(xj, r)| xj * r).sum();
            let z: f64 = xj.iter().map(|xj| xj * xj).sum();
            let old = coef[j];
            coef[j] = soft_threshold(rho, alpha) / z.max(1e-10);
            let change = (coef[j] - old).abs();
            if change > max_change { max_change = change; }
            // Update residual
            for i in 0..n {
                residual[i] -= x[i][j] * (coef[j] - old);
            }
        }
        // Update intercept
        let mean_res: f64 = residual.iter().sum::<f64>() / n as f64;
        intercept += mean_res;
        for r in residual.iter_mut() { *r -= mean_res; }
        if max_change < tol { break; }
    }
    let mut residuals = Vec::with_capacity(n);
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    for i in 0..n {
        let pred = intercept + x[i].iter().zip(&coef).map(|(xi, ci)| xi * ci).sum::<f64>();
        let r = y[i] - pred;
        residuals.push(r);
        ss_res += r * r;
        ss_tot += (y[i] - y_mean).powi(2);
    }
    let r_squared = if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot };
    Ok(LinearResult { coefficients: coef, intercept, r_squared, residuals })
}

fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda { x - lambda } else if x < -lambda { x + lambda } else { 0.0 }
}

fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> MathResult<Vec<f64>> {
    let n = b.len();
    let a = a.to_vec();
    let b = b.to_vec();
    // Cholesky decomposition
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j { sum += l[i][k] * l[j][k]; }
            let diag = a[i][i] - sum;
            if diag <= 0.0 { return Err(MathError::Singular); }
            l[i][j] = if i == j { diag.sqrt() } else { (a[i][j] - sum) / l[j][j] };
        }
    }
    // Forward solve Ly = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let sum: f64 = (0..i).map(|k| l[i][k] * y[k]).sum();
        y[i] = (b[i] - sum) / l[i][i];
    }
    // Back solve L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|k| l[k][i] * x[k]).sum();
        x[i] = (y[i] - sum) / l[i][i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-6;

    #[test]
    fn fit_perfect_linear() {
        let x: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..5).map(|i| 2.0 * i as f64 + 1.0).collect();
        let r = fit(&x, &y).unwrap();
        assert!((r.coefficients[0] - 2.0).abs() < E);
        assert!((r.intercept - 1.0).abs() < E);
        assert!((r.r_squared - 1.0).abs() < E);
    }

    #[test]
    fn fit_2d() {
        let x = vec![vec![1.0, 2.0], vec![2.0, 1.0], vec![3.0, 4.0], vec![4.0, 3.0]];
        let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 3.0 * r[1]).collect();
        let r = fit(&x, &y).unwrap();
        assert!((r.coefficients[0] - 2.0).abs() < E);
        assert!((r.coefficients[1] - 3.0).abs() < E);
        assert!((r.intercept).abs() < E);
    }

    #[test]
    fn ridge_test() {
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..10).map(|i| 2.0 * i as f64 + 1.0).collect();
        let r = fit_ridge(&x, &y, 0.1).unwrap();
        // Should be close to OLS
        assert!((r.coefficients[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn predict_test() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let coef = vec![2.0];
        let intercept = 1.0;
        let preds = predict(&x, &coef, intercept);
        assert_eq!(preds, vec![3.0, 5.0, 7.0]);
    }
}
