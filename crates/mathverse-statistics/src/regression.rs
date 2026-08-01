//! Regression: polynomial, multiple (OLS), logistic, weighted least squares.

/// Polynomial regression: fit y = Σ βᵢxⁱ (degree `d`).
/// Returns coefficients `[β₀, β₁, ..., β_d]` (lowest degree first).
pub fn polynomial_regression(xs: &[f64], ys: &[f64], degree: usize) -> Vec<f64> {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len();
    let d = degree + 1;
    // Build normal equations: (X^T X) β = X^T y
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];
    for i in 0..n {
        let xi = xs[i];
        let yi = ys[i];
        for j in 0..d {
            xty[j] += xi.powi(j as i32) * yi;
            for (k, xtx_jk) in xtx[j].iter_mut().enumerate() {
                *xtx_jk += xi.powi((j + k) as i32);
            }
        }
    }
    gaussian_elimination(&mut xtx, &mut xty)
}

/// Multiple linear regression (OLS): y = Xβ + ε.
/// `xs` is `n × p` (row-major), `ys` is length `n`.
/// Returns coefficients `[β₀, β₁, ..., β_p]`.
pub fn multiple_regression(xs: &[&[f64]], ys: &[f64]) -> Vec<f64> {
    let n = ys.len();
    let p = xs[0].len();
    // Add intercept column (all 1s)
    let d = p + 1;
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];
    for i in 0..n {
        for j in 0..d {
            let xj = if j == 0 { 1.0 } else { xs[i][j - 1] };
            xty[j] += xj * ys[i];
            for k in 0..d {
                let xk = if k == 0 { 1.0 } else { xs[i][k - 1] };
                xtx[j][k] += xj * xk;
            }
        }
    }
    gaussian_elimination(&mut xtx, &mut xty)
}

/// Weighted least squares: y = Xβ + ε, Var(εᵢ) = σ²/wᵢ.
/// `xs` is `n × p` (row-major), `ys` length `n`, `weights` length `n`.
pub fn weighted_least_squares(xs: &[&[f64]], ys: &[f64], weights: &[f64]) -> Vec<f64> {
    let n = ys.len();
    let p = xs[0].len();
    let d = p + 1;
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];
    for i in 0..n {
        let w = weights[i];
        for j in 0..d {
            let xj = if j == 0 { 1.0 } else { xs[i][j - 1] };
            xty[j] += w * xj * ys[i];
            for k in 0..d {
                let xk = if k == 0 { 1.0 } else { xs[i][k - 1] };
                xtx[j][k] += w * xj * xk;
            }
        }
    }
    gaussian_elimination(&mut xtx, &mut xty)
}

/// Logistic regression (binary): P(y=1) = sigmoid(Xβ).
/// Returns coefficients via iteratively reweighted least squares (IRLS).
pub fn logistic_regression(xs: &[&[f64]], ys: &[f64], max_iter: usize, lr: f64) -> Vec<f64> {
    let n = ys.len();
    let p = xs[0].len();
    let d = p + 1;
    let mut beta = vec![0.0; d];

    for _ in 0..max_iter {
        // Compute predictions
        let mut eta = vec![0.0; n];
        for i in 0..n {
            eta[i] = beta[0];
            for j in 0..p {
                eta[i] += beta[j + 1] * xs[i][j];
            }
        }
        let mu: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();

        // Working response and weights
        let mut z = vec![0.0; n];
        let mut w = vec![0.0; n];
        for i in 0..n {
            w[i] = mu[i] * (1.0 - mu[i]);
            if w[i] < 1e-15 { w[i] = 1e-15; }
            z[i] = eta[i] + (ys[i] - mu[i]) / w[i];
        }

        // WLS step
        let mut xtx = vec![vec![0.0; d]; d];
        let mut xty = vec![0.0; d];
        for i in 0..n {
            for j in 0..d {
                let xj = if j == 0 { 1.0 } else { xs[i][j - 1] };
                xty[j] += w[i] * xj * z[i];
                for k in 0..d {
                    let xk = if k == 0 { 1.0 } else { xs[i][k - 1] };
                    xtx[j][k] += w[i] * xj * xk;
                }
            }
        }
        let delta = gaussian_elimination(&mut xtx, &mut xty);
        for j in 0..d {
            beta[j] += lr * delta[j];
        }
    }
    beta
}

/// Predicted probability from logistic regression coefficients.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// R² for polynomial/multiple regression.
pub fn r_squared(xs: &[&[f64]], ys: &[f64], coeffs: &[f64]) -> f64 {
    let n = ys.len();
    let my = ys.iter().sum::<f64>() / n as f64;
    let mut sst = 0.0;
    let mut ssr = 0.0;
    for i in 0..n {
        let pred = predict(xs[i], coeffs);
        sst += (ys[i] - my).powi(2);
        ssr += (ys[i] - pred).powi(2);
    }
    if sst == 0.0 { return 1.0; }
    1.0 - ssr / sst
}

/// Predict y from a single row and coefficients.
pub fn predict(row: &[f64], coeffs: &[f64]) -> f64 {
    let mut y = coeffs[0];
    for j in 0..row.len() {
        y += coeffs[j + 1] * row[j];
    }
    y
}

/// Predict y from polynomial coefficients: y = Σ βᵢxⁱ.
pub fn predict_poly(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().enumerate().map(|(i, &c)| c * x.powi(i as i32)).sum()
}

/// Residuals from regression.
pub fn residuals(xs: &[&[f64]], ys: &[f64], coeffs: &[f64]) -> Vec<f64> {
    ys.iter().enumerate().map(|(i, &y)| y - predict(xs[i], coeffs)).collect()
}

/// Mean squared error.
pub fn mse(ys: &[f64], predicted: &[f64]) -> f64 {
    ys.iter().zip(predicted).map(|(y, p)| (y - p).powi(2)).sum::<f64>() / ys.len() as f64
}

/// Root mean squared error.
pub fn rmse(ys: &[f64], predicted: &[f64]) -> f64 {
    mse(ys, predicted).sqrt()
}

/// Mean absolute error.
pub fn mae(ys: &[f64], predicted: &[f64]) -> f64 {
    ys.iter().zip(predicted).map(|(y, p)| (y - p).abs()).sum::<f64>() / ys.len() as f64
}

// ---------------------------------------------------------------------------
// Gaussian elimination with partial pivoting
// ---------------------------------------------------------------------------

fn gaussian_elimination(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        // Pivot
        let max_row = (i..n).max_by(|&r, &s| a[r][i].abs().partial_cmp(&a[s][i].abs()).unwrap()).unwrap();
        a.swap(i, max_row);
        b.swap(i, max_row);
        let pivot = a[i][i];
        if pivot.abs() < 1e-30 {
            continue;
        }
        // Eliminate below
        for j in (i + 1)..n {
            let factor = a[j][i] / pivot;
            let (a_top, a_bot) = a.split_at_mut(j);
            for (k, a_jk) in a_bot[0].iter_mut().enumerate().skip(i) {
                *a_jk -= factor * a_top[i][k];
            }
            b[j] -= factor * b[i];
        }
    }
    // Back-substitute
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_degree1() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [3.0, 5.0, 7.0, 9.0];
        let c = polynomial_regression(&xs, &ys, 1);
        assert!((c[1] - 2.0).abs() < 1e-10); // slope
        assert!((c[0] - 1.0).abs() < 1e-10); // intercept
    }

    #[test]
    fn multiple_regression_test() {
        let xs: Vec<&[f64]> = vec![&[1.0, 2.0], &[2.0, 1.0], &[3.0, 4.0]];
        let ys = [7.0, 8.0, 17.0];
        // y = 3*x1 + 2*x2 + 0
        let c = multiple_regression(&xs, &ys);
        assert!((c[1] - 3.0).abs() < 1e-8);
        assert!((c[2] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn logistic_test() {
        let xs: Vec<&[f64]> = vec![&[1.0], &[2.0], &[3.0], &[4.0], &[5.0]];
        let ys = [0.0, 0.0, 1.0, 1.0, 1.0];
        let c = logistic_regression(&xs, &ys, 200, 0.1);
        assert!(c[1] > 0.0); // positive coefficient
    }

    #[test]
    fn r_squared_test() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [3.0, 5.0, 7.0, 9.0];
        let c = polynomial_regression(&xs, &ys, 1);
        let xs_refs: Vec<&[f64]> = xs.iter().map(|x| std::slice::from_ref(x)).collect();
        let r2 = r_squared(&xs_refs, &ys, &c);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn predict_poly_test() {
        let c = [1.0, 2.0, 3.0]; // 1 + 2x + 3x²
        assert!((predict_poly(0.0, &c) - 1.0).abs() < 1e-12);
        assert!((predict_poly(1.0, &c) - 6.0).abs() < 1e-12);
        assert!((predict_poly(2.0, &c) - 17.0).abs() < 1e-12);
    }
}
