//! Complex-valued regression solvers: least-squares, ridge, and LASSO
//! for fitting complex-valued models to complex-valued data.
//!
//! These solvers are essential for complex-valued signal processing,
//! direction-of-arrival estimation, channel estimation in communications,
//! and MRI reconstruction.
//!
//! # Module overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`least_squares`] | Ordinary least-squares via normal equations |
//! | [`ridge`] | Tikhonov-regularized least-squares |
//! | [`lasso_coordinate`] | Coordinate-descent LASSO for complex signals |
//! | [`predict`] | Evaluate linear model: y = X·w + b |

use crate::matrix::ComplexMatrix;
use crate::Complex;
use mathverse_core::error::{MathError, MathResult};

/// Solve the ordinary least-squares problem: `min ||X·w − y||²`.
///
/// Uses the normal equations: `w = (Xᴴ·X)⁻¹·Xᴴ·y`.
///
/// # Arguments
/// * `x` — design matrix of shape `(n, p)`, `n ≥ p`
/// * `y` — response vector of length `n`
///
/// # Returns
/// Weight vector `w` of length `p`.
///
/// # Errors
/// [`MathError::DimensionMismatch`] if dimensions are incompatible.
/// [`MathError::Singular`] if `Xᴴ·X` is singular.
pub fn least_squares(x: &ComplexMatrix, y: &[Complex]) -> MathResult<Vec<Complex>> {
    let (n, p) = (x.rows, x.cols);
    if y.len() != n {
        return Err(MathError::DimensionMismatch);
    }
    if n < p {
        return Err(MathError::InvalidArgument(
            "Underdetermined system: n < p",
        ));
    }

    // Xᴴ (conjugate transpose)
    let xh = x.hermitian();

    // Xᴴ·X
    let xhx = xh.mul(x)?;

    // Xᴴ·y: multiply xh (p×n) by y column vector (n×1)
    let y_col = ComplexMatrix::from_data(y.to_vec(), n, 1);
    let xhy_mat = xh.mul(&y_col)?;
    let xhy: Vec<Complex> = (0..p).map(|i| xhy_mat.get(i, 0)).collect();

    // Solve via LU decomposition
    let w = xhx.solve(&xhy).ok_or(MathError::Singular)?;
    Ok(w)
}

/// Solve Tikhonov-regularized least-squares (ridge regression):
/// `min ||X·w − y||² + λ·||w||²`.
///
/// Equivalent to solving: `(Xᴴ·X + λ·I)·w = Xᴴ·y`.
///
/// # Arguments
/// * `x` — design matrix of shape `(n, p)`
/// * `y` — response vector of length `n`
/// * `lambda` — regularization strength (must be ≥ 0)
///
/// # Returns
/// Regularized weight vector `w` of length `p`.
///
/// # Errors
/// [`MathError::DimensionMismatch`] if dimensions are incompatible.
/// [`MathError::Singular`] if the system is singular.
pub fn ridge(
    x: &ComplexMatrix,
    y: &[Complex],
    lambda: f64,
) -> MathResult<Vec<Complex>> {
    let (n, p) = (x.rows, x.cols);
    if y.len() != n {
        return Err(MathError::DimensionMismatch);
    }
    if lambda < 0.0 {
        return Err(MathError::InvalidArgument(
            "Regularization lambda must be ≥ 0",
        ));
    }

    let xh = x.hermitian();
    let mut xhx = xh.mul(x)?;

    // Add λ·I to diagonal
    for i in 0..p {
        xhx[(i, i)] = xhx[(i, i)] + Complex::real(lambda);
    }

    let y_col = ComplexMatrix::from_data(y.to_vec(), n, 1);
    let xhy_mat = xh.mul(&y_col)?;
    let xhy: Vec<Complex> = (0..p).map(|i| xhy_mat.get(i, 0)).collect();

    let w = xhx.solve(&xhy).ok_or(MathError::Singular)?;
    Ok(w)
}

/// Coordinate-descent LASSO for complex-valued signals.
///
/// Minimizes: `min (1/2n)·||X·w − y||² + λ·||w||₁`
/// where `||w||₁ = Σ|Re(w_j)| + |Im(w_j)|`.
///
/// Performs `max_iter` soft-thresholding iterations.
///
/// # Arguments
/// * `x` — design matrix of shape `(n, p)`
/// * `y` — response vector of length `n`
/// * `lambda` — L1 regularization strength
/// * `max_iter` — maximum number of coordinate descent iterations
/// * `tol` — convergence tolerance
///
/// # Returns
/// Sparse weight vector `w` of length `p`.
pub fn lasso_coordinate(
    x: &ComplexMatrix,
    y: &[Complex],
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> MathResult<Vec<Complex>> {
    let (n, p) = (x.rows, x.cols);
    if y.len() != n {
        return Err(MathError::DimensionMismatch);
    }

    let mut w = vec![Complex::zero(); p];
    let n_f = n as f64;

    // Precompute column norms (squared)
    let col_norms: Vec<f64> = (0..p)
        .map(|j| {
            let mut norm_sq = 0.0;
            for i in 0..n {
                let v = x[(i, j)];
                norm_sq += v.norm_sq();
            }
            norm_sq / n_f
        })
        .collect();

    for _ in 0..max_iter {
        let mut max_change = 0.0;

        for j in 0..p {
            // Compute residual: r = y - X·w
            let mut residual: Vec<Complex> = y.to_vec();
            for i in 0..n {
                for k in 0..p {
                    residual[i] = residual[i] - x[(i, k)] * w[k];
                }
                residual[i] = residual[i] + x[(i, j)] * w[j];
            }

            // Compute X_jᴴ·r / n
            let mut grad = Complex::zero();
            for i in 0..n {
                grad = grad + x[(i, j)].conjugate() * residual[i];
            }
            grad = grad / Complex::real(n_f);

            // Soft-thresholding update
            let old = w[j];
            let norm_j = col_norms[j];
            if norm_j > 1e-15 {
                let scale = 1.0 / norm_j;
                let z = grad * Complex::real(scale);
                // Soft-threshold: sign(z) * max(|z| - λ/norm_j, 0)
                let abs_z = z.norm();
                let threshold = lambda / (norm_j * n_f);
                if abs_z > threshold {
                    w[j] = z * Complex::real((abs_z - threshold) / abs_z);
                } else {
                    w[j] = Complex::zero();
                }
            }

            let change = (w[j] - old).norm();
            if change > max_change {
                max_change = change;
            }
        }

        if max_change < tol {
            break;
        }
    }

    Ok(w)
}

/// Evaluate a linear model: `y_pred = X·w`.
///
/// # Arguments
/// * `x` — feature matrix of shape `(n, p)`
/// * `w` — weight vector of length `p`
///
/// # Returns
/// Prediction vector of length `n`.
///
/// # Errors
/// [`MathError::DimensionMismatch`] if `w.len() != x.cols`.
pub fn predict(x: &ComplexMatrix, w: &[Complex]) -> MathResult<Vec<Complex>> {
    if w.len() != x.cols {
        return Err(MathError::DimensionMismatch);
    }
    let w_col = ComplexMatrix::from_data(w.to_vec(), w.len(), 1);
    let result = x.mul(&w_col)?;
    Ok((0..result.rows).map(|i| result.get(i, 0)).collect())
}

/// Compute the mean squared error between predicted and actual values.
///
/// # Panics
/// If `predicted.len() != actual.len()`.
pub fn mse(predicted: &[Complex], actual: &[Complex]) -> f64 {
    assert_eq!(predicted.len(), actual.len());
    let n = predicted.len() as f64;
    let total: f64 = predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (*p - *a).norm_sq())
        .sum();
    total / n
}

/// Compute R² (coefficient of determination) for complex-valued predictions.
///
/// `R² = 1 − SS_res / SS_tot` where `SS_res = Σ|y_i − ŷ_i|²` and
/// `SS_tot = Σ|y_i − ȳ|²`.
///
/// # Panics
/// If `predicted.len() != actual.len()`.
pub fn r_squared(predicted: &[Complex], actual: &[Complex]) -> f64 {
    assert_eq!(predicted.len(), actual.len());
    let mean: Complex = actual
        .iter()
        .copied()
        .fold(Complex::zero(), |a, b| a + b)
        / Complex::real(actual.len() as f64);

    let ss_res: f64 = predicted
        .iter()
        .zip(actual.iter())
        .map(|(p, a)| (*p - *a).norm_sq())
        .sum();

    let ss_tot: f64 = actual
        .iter()
        .map(|a| (*a - mean).norm_sq())
        .sum();

    if ss_tot < 1e-15 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_matrix(rows: usize, cols: usize, data: &[f64]) -> ComplexMatrix {
        let cdata: Vec<Complex> = data.iter().map(|&v| Complex::real(v)).collect();
        ComplexMatrix::from_data(cdata, rows, cols)
    }

    #[test]
    fn least_squares_perfect_fit() {
        // y = 2x + 1 (real-valued, should find w = [1, 2])
        let x = make_matrix(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let y: Vec<Complex> = vec![
            Complex::real(1.0),
            Complex::real(3.0),
            Complex::real(5.0),
            Complex::real(7.0),
        ];
        let w = least_squares(&x, &y).unwrap();
        assert!((w[0].re - 1.0).abs() < 1e-8);
        assert!((w[1].re - 2.0).abs() < 1e-8);
    }

    #[test]
    fn least_squares_complex() {
        // Fit y = (1+i)·x to data
        let x = make_matrix(3, 1, &[1.0, 2.0, 3.0]);
        let coeff = Complex::new(1.0, 1.0);
        let y: Vec<Complex> = (1..=3).map(|i| coeff * Complex::real(f64::from(i))).collect();
        let w = least_squares(&x, &y).unwrap();
        assert!((w[0] - coeff).norm() < 1e-8);
    }

    #[test]
    fn ridge_reduces_coefficients() {
        let x = make_matrix(10, 2, &[
            1.0, 0.5, 1.0, 1.0, 1.0, 1.5, 1.0, 2.0, 1.0, 2.5,
            1.0, 3.0, 1.0, 3.5, 1.0, 4.0, 1.0, 4.5, 1.0, 5.0,
        ]);
        let y: Vec<Complex> = (0..10)
            .map(|i| Complex::real(f64::from(i) * 2.0 + 1.0))
            .collect();

        let w_ols = least_squares(&x, &y).unwrap();
        let w_ridge = ridge(&x, &y, 1.0).unwrap();

        // Ridge coefficients should be smaller in magnitude
        let norm_ols: f64 = w_ols.iter().map(super::super::Complex::norm).sum();
        let norm_ridge: f64 = w_ridge.iter().map(super::super::Complex::norm).sum();
        assert!(norm_ridge < norm_ols + 1e-10);
    }

    #[test]
    fn lasso_sparsity() {
        // Only first feature is relevant
        let x = make_matrix(20, 3, &{
            let mut d = Vec::new();
            for i in 0..20 {
                d.push(1.0);
                d.push(f64::from(i));
                d.push((f64::from(i) * 0.1).sin());
            }
            d
        });
        let y: Vec<Complex> = (0..20).map(|i| Complex::real(f64::from(i) + 1.0)).collect();

        let w = lasso_coordinate(&x, &y, 0.5, 100, 1e-8).unwrap();

        // With strong regularization, some weights should be near zero
        let non_zero = w.iter().filter(|w| w.norm() > 1e-6).count();
        assert!(non_zero < 3, "Expected sparsity, got {non_zero} non-zero weights");
    }

    #[test]
    fn predict_basic() {
        let x = make_matrix(3, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let w = vec![Complex::real(1.0), Complex::real(2.0)];
        let y = predict(&x, &w).unwrap();
        assert!((y[0].re - 3.0).abs() < EPS);
        assert!((y[1].re - 5.0).abs() < EPS);
        assert!((y[2].re - 7.0).abs() < EPS);
    }

    const EPS: f64 = 1e-10;

    #[test]
    fn mse_zero_for_perfect() {
        let pred = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let actual = pred.clone();
        assert!(mse(&pred, &actual) < EPS);
    }

    #[test]
    fn r_squared_perfect_fit() {
        let actual = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 1.0)];
        let predicted = actual.clone();
        let r2 = r_squared(&predicted, &actual);
        assert!((r2 - 1.0).abs() < EPS);
    }

    #[test]
    fn dimension_mismatch() {
        let x = make_matrix(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = vec![Complex::real(1.0); 2]; // Wrong length
        assert!(least_squares(&x, &y).is_err());
    }
}
