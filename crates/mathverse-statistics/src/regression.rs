//! Regression: polynomial, multiple (OLS), logistic, weighted least squares.

use mathverse_core::error::{MathError, MathResult};

/// Polynomial regression: fit y = Σ βᵢxⁱ (degree `d`).
/// Returns coefficients `[β₀, β₁, ..., β_d]` (lowest degree first).
///
/// # Errors
///
/// Returns [`MathError::DimensionMismatch`] if `xs` and `ys` differ in length.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::polynomial_regression;
///
/// let xs = [1.0, 2.0, 3.0, 4.0];
/// let ys = [3.0, 5.0, 7.0, 9.0];
/// let c = polynomial_regression(&xs, &ys, 1).unwrap();
/// assert!((c[1] - 2.0).abs() < 1e-10); // slope
/// ```
pub fn polynomial_regression(xs: &[f64], ys: &[f64], degree: usize) -> MathResult<Vec<f64>> {
    if xs.len() != ys.len() {
        return Err(MathError::DimensionMismatch);
    }
    if xs.is_empty() {
        return Err(MathError::InvalidArgument("need at least one point"));
    }
    let n = xs.len();
    let d = degree + 1;
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];
    for i in 0..n {
        let xi = xs[i];
        let yi = ys[i];
        for j in 0..d {
            xty[j] = xi.powi(j as i32).mul_add(yi, xty[j]);
            for (k, xtx_jk) in xtx[j].iter_mut().enumerate() {
                *xtx_jk += xi.powi((j + k) as i32);
            }
        }
    }
    Ok(gaussian_elimination(&mut xtx, &mut xty))
}

/// Multiple linear regression (OLS): y = Xβ + ε.
/// `xs` is `n × p` (row-major), `ys` is length `n`.
/// Returns coefficients `[β₀, β₁, ..., β_p]`.
///
/// # Errors
///
/// Returns [`MathError::DimensionMismatch`] if `xs` and `ys` differ in length,
/// or [`MathError::InvalidArgument`] if inputs are empty.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::multiple_regression;
///
/// let xs: Vec<&[f64]> = vec![&[1.0, 2.0], &[2.0, 1.0], &[3.0, 4.0]];
/// let ys = [7.0, 8.0, 17.0];
/// let c = multiple_regression(&xs, &ys).unwrap();
/// assert!((c[1] - 3.0).abs() < 1e-8);
/// ```
pub fn multiple_regression(xs: &[&[f64]], ys: &[f64]) -> MathResult<Vec<f64>> {
    if xs.is_empty() || ys.is_empty() {
        return Err(MathError::InvalidArgument("need at least one data point"));
    }
    if xs.len() != ys.len() {
        return Err(MathError::DimensionMismatch);
    }
    let n = ys.len();
    let p = xs[0].len();
    let d = p + 1;
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];
    for i in 0..n {
        for j in 0..d {
            let xj = if j == 0 { 1.0 } else { xs[i][j - 1] };
            xty[j] = xj.mul_add(ys[i], xty[j]);
            for k in 0..d {
                let xk = if k == 0 { 1.0 } else { xs[i][k - 1] };
                xtx[j][k] = xj.mul_add(xk, xtx[j][k]);
            }
        }
    }
    Ok(gaussian_elimination(&mut xtx, &mut xty))
}

/// Weighted least squares: y = Xβ + ε, Var(εᵢ) = σ²/wᵢ.
/// `xs` is `n × p` (row-major), `ys` length `n`, `weights` length `n`.
/// Returns coefficients `[β₀, β₁, ..., β_p]`.
///
/// # Errors
///
/// Returns [`MathError::DimensionMismatch`] if input lengths differ,
/// or [`MathError::InvalidArgument`] if inputs are empty.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::weighted_least_squares;
///
/// let xs: Vec<&[f64]> = vec![&[1.0], &[2.0], &[3.0]];
/// let ys = [2.0, 4.0, 6.0];
/// let w = [1.0, 1.0, 1.0];
/// let c = weighted_least_squares(&xs, &ys, &w).unwrap();
/// assert!((c[1] - 2.0).abs() < 1e-8);
/// ```
pub fn weighted_least_squares(xs: &[&[f64]], ys: &[f64], weights: &[f64]) -> MathResult<Vec<f64>> {
    if xs.is_empty() || ys.is_empty() {
        return Err(MathError::InvalidArgument("need at least one data point"));
    }
    if xs.len() != ys.len() || ys.len() != weights.len() {
        return Err(MathError::DimensionMismatch);
    }
    let n = ys.len();
    let p = xs[0].len();
    let d = p + 1;
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];
    for i in 0..n {
        let w = weights[i];
        for j in 0..d {
            let xj = if j == 0 { 1.0 } else { xs[i][j - 1] };
            xty[j] = w.mul_add(xj.mul_add(ys[i], 0.0), xty[j]);
            for k in 0..d {
                let xk = if k == 0 { 1.0 } else { xs[i][k - 1] };
                xtx[j][k] = w.mul_add(xj * xk, xtx[j][k]);
            }
        }
    }
    Ok(gaussian_elimination(&mut xtx, &mut xty))
}

/// Logistic regression (binary): P(y=1) = sigmoid(Xβ).
/// Returns coefficients via iteratively reweighted least squares (IRLS).
///
/// # Errors
///
/// Returns [`MathError::DimensionMismatch`] if `xs` and `ys` differ in length,
/// or [`MathError::InvalidArgument`] if inputs are empty.
///
/// # Examples
///
/// ```
/// use mathverse_statistics::logistic_regression;
///
/// let xs: Vec<&[f64]> = vec![&[1.0], &[2.0], &[3.0], &[4.0], &[5.0]];
/// let ys = [0.0, 0.0, 1.0, 1.0, 1.0];
/// let c = logistic_regression(&xs, &ys, 200, 0.1).unwrap();
/// assert!(c[1] > 0.0); // positive coefficient
/// ```
pub fn logistic_regression(
    xs: &[&[f64]],
    ys: &[f64],
    max_iter: usize,
    lr: f64,
) -> MathResult<Vec<f64>> {
    if xs.is_empty() || ys.is_empty() {
        return Err(MathError::InvalidArgument("need at least one data point"));
    }
    if xs.len() != ys.len() {
        return Err(MathError::DimensionMismatch);
    }
    let n = ys.len();
    let p = xs[0].len();
    let d = p + 1;
    let mut beta = vec![0.0; d];
    let mut xtx = vec![vec![0.0; d]; d];
    let mut xty = vec![0.0; d];

    for _ in 0..max_iter {
        let mut eta = vec![0.0; n];
        for i in 0..n {
            eta[i] = beta[0];
            for j in 0..p {
                eta[i] += beta[j + 1] * xs[i][j];
            }
        }
        let mu: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();

        let mut z = vec![0.0; n];
        let mut w = vec![0.0; n];
        for i in 0..n {
            w[i] = mu[i].mul_add(1.0 - mu[i], 1e-15).max(1e-15);
            z[i] = eta[i] + (ys[i] - mu[i]) / w[i];
        }

        xtx.iter_mut().for_each(|row| row.fill(0.0));
        xty.fill(0.0);
        for i in 0..n {
            for j in 0..d {
                let xj = if j == 0 { 1.0 } else { xs[i][j - 1] };
                xty[j] = w[i].mul_add(xj * z[i], xty[j]);
                for k in 0..d {
                    let xk = if k == 0 { 1.0 } else { xs[i][k - 1] };
                    xtx[j][k] = w[i].mul_add(xj * xk, xtx[j][k]);
                }
            }
        }
        let delta = gaussian_elimination(&mut xtx, &mut xty);
        for j in 0..d {
            beta[j] += lr * delta[j];
        }
    }
    Ok(beta)
}

/// Logistic function: `1 / (1 + e^{-x})`.
#[must_use]
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// R² for polynomial/multiple regression.
#[must_use]
#[inline]
pub fn r_squared(xs: &[&[f64]], ys: &[f64], coeffs: &[f64]) -> f64 {
    let n = ys.len();
    if n == 0 {
        return 0.0;
    }
    let my = ys.iter().sum::<f64>() / n as f64;
    let mut sst = 0.0;
    let mut ssr = 0.0;
    for i in 0..n {
        let pred = predict(xs[i], coeffs);
        sst += (ys[i] - my).powi(2);
        ssr += (ys[i] - pred).powi(2);
    }
    if sst == 0.0 {
        return 1.0;
    }
    1.0 - ssr / sst
}

/// Predict y from a single row and coefficients.
#[must_use]
#[inline]
pub fn predict(row: &[f64], coeffs: &[f64]) -> f64 {
    let mut y = coeffs[0];
    for j in 0..row.len() {
        y += coeffs[j + 1] * row[j];
    }
    y
}

/// Predict y from polynomial coefficients: y = Σ βᵢxⁱ.
#[must_use]
#[inline]
pub fn predict_poly(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().enumerate().map(|(i, &c)| c * x.powi(i as i32)).sum()
}

/// Residuals from regression.
#[must_use]
pub fn residuals(xs: &[&[f64]], ys: &[f64], coeffs: &[f64]) -> Vec<f64> {
    ys.iter()
        .zip(xs)
        .map(|(&y, row)| y - predict(row, coeffs))
        .collect()
}

/// Mean squared error.
#[must_use]
#[inline]
pub fn mse(ys: &[f64], predicted: &[f64]) -> f64 {
    let n = ys.len().max(1);
    ys.iter()
        .zip(predicted)
        .map(|(y, p)| (y - p).powi(2))
        .sum::<f64>()
        / n as f64
}

/// Root mean squared error.
#[must_use]
#[inline]
pub fn rmse(ys: &[f64], predicted: &[f64]) -> f64 {
    mse(ys, predicted).sqrt()
}

/// Mean absolute error.
#[must_use]
#[inline]
pub fn mae(ys: &[f64], predicted: &[f64]) -> f64 {
    let n = ys.len().max(1);
    ys.iter()
        .zip(predicted)
        .map(|(y, p)| (y - p).abs())
        .sum::<f64>()
        / n as f64
}

// ---------------------------------------------------------------------------
// Gaussian elimination with partial pivoting
// ---------------------------------------------------------------------------

fn gaussian_elimination(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        let max_row = (i..n)
            .max_by(|&r, &s| a[r][i].abs().partial_cmp(&a[s][i].abs()).unwrap())
            .unwrap();
        a.swap(i, max_row);
        b.swap(i, max_row);
        let pivot = a[i][i];
        if pivot.abs() < 1e-30 {
            continue;
        }
        for j in (i + 1)..n {
            let factor = a[j][i] / pivot;
            let (a_top, a_bot) = a.split_at_mut(j);
            for (k, a_jk) in a_bot[0].iter_mut().enumerate().skip(i) {
                *a_jk -= factor * a_top[i][k];
            }
            b[j] -= factor * b[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - sum) / a[i][i];
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
        let c = polynomial_regression(&xs, &ys, 1).unwrap();
        assert!((c[1] - 2.0).abs() < 1e-10);
        assert!((c[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn polynomial_dim_mismatch() {
        assert_eq!(
            polynomial_regression(&[1.0, 2.0], &[3.0], 1),
            Err(MathError::DimensionMismatch)
        );
    }

    #[test]
    fn multiple_regression_test() {
        let xs: Vec<&[f64]> = vec![&[1.0, 2.0], &[2.0, 1.0], &[3.0, 4.0]];
        let ys = [7.0, 8.0, 17.0];
        let c = multiple_regression(&xs, &ys).unwrap();
        assert!((c[1] - 3.0).abs() < 1e-8);
        assert!((c[2] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn logistic_test() {
        let xs: Vec<&[f64]> = vec![&[1.0], &[2.0], &[3.0], &[4.0], &[5.0]];
        let ys = [0.0, 0.0, 1.0, 1.0, 1.0];
        let c = logistic_regression(&xs, &ys, 200, 0.1).unwrap();
        assert!(c[1] > 0.0);
    }

    #[test]
    fn r_squared_test() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [3.0, 5.0, 7.0, 9.0];
        let c = polynomial_regression(&xs, &ys, 1).unwrap();
        let xs_refs: Vec<&[f64]> = xs.iter().map(|x| core::slice::from_ref(x)).collect();
        let r2 = r_squared(&xs_refs, &ys, &c);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn predict_poly_test() {
        let c = [1.0, 2.0, 3.0];
        assert!((predict_poly(0.0, &c) - 1.0).abs() < 1e-12);
        assert!((predict_poly(1.0, &c) - 6.0).abs() < 1e-12);
        assert!((predict_poly(2.0, &c) - 17.0).abs() < 1e-12);
    }

    #[test]
    fn error_metrics_test() {
        let ys = [1.0, 2.0, 3.0];
        let pred = [1.1, 2.1, 2.9];
        assert!(mse(&ys, &pred) > 0.0);
        assert!(rmse(&ys, &pred) > 0.0);
        assert!(mae(&ys, &pred) > 0.0);
    }
}
