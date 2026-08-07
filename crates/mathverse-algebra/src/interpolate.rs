//! # Interpolation
//!
//! Lagrange and Newton polynomial interpolation from data points.
//!
//! ## Examples
//!
//! ```rust
//! use mathverse_algebra::interpolate::{lagrange, newton};
//!
//! // Points (0,1), (1,3), (2,7) → p(x) = x^2 + x + 1
//! let xs = [0.0, 1.0, 2.0];
/// let ys = [1.0, 3.0, 7.0];
/// assert!((lagrange(&xs, &ys)(2.0) - 7.0).abs() < 1e-10);
/// assert!((newton(&xs, &ys)(2.0) - 7.0).abs() < 1e-10);
/// ```

use crate::polynomial::Polynomial;

/// Lagrange interpolation.
///
/// Returns a [`Polynomial`] that passes through every `(xs[i], ys[i])`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::interpolate::lagrange;
///
/// // (0,1), (1,2), (2,5)
/// let xs = [0.0, 1.0, 2.0];
/// let ys = [1.0, 2.0, 5.0];
/// let p = lagrange(&xs, &ys);
/// assert!((p.eval(0.0) - 1.0).abs() < 1e-10);
/// assert!((p.eval(1.0) - 2.0).abs() < 1e-10);
/// assert!((p.eval(2.0) - 5.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn lagrange(xs: &[f64], ys: &[f64]) -> Polynomial {
    let n = xs.len();
    let mut result = Polynomial::constant(0.0);
    for i in 0..n {
        let mut basis = Polynomial::constant(1.0);
        for j in 0..n {
            if i == j {
                continue;
            }
            let basis_factor = Polynomial::from_coeffs(&[-xs[j], 1.0]);
            basis = basis * basis_factor * Polynomial::constant(1.0 / (xs[i] - xs[j]));
        }
        result = result + basis * Polynomial::constant(ys[i]);
    }
    result
}

/// Divided differences table.
///
/// Returns a vector of divided differences `dd[k]` (the first row of each column),
/// which are the Newton coefficients.
#[must_use]
pub fn divided_differences(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut dd: Vec<Vec<f64>> = (0..n).map(|i| vec![ys[i]]).collect();
    for j in 1..n {
        for i in 0..n - j {
            let diff = dd[i + 1][j - 1] - dd[i][j - 1];
            let denom = xs[i + j] - xs[i];
            dd[i].push(if denom.abs() < crate::TOL { 0.0 } else { diff / denom });
        }
    }
    dd.remove(0)
}

/// Newton interpolation using divided differences.
///
/// Returns a [`Polynomial`] that passes through every `(xs[i], ys[i])`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::interpolate::newton;
///
/// let xs = [0.0, 1.0, 2.0];
/// let ys = [1.0, 2.0, 5.0];
/// let p = newton(&xs, &ys);
/// assert!((p.eval(0.0) - 1.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn newton(xs: &[f64], ys: &[f64]) -> Polynomial {
    let dd = divided_differences(xs, ys);
    let n = xs.len();
    let mut result = Polynomial::constant(dd[0]);
    let mut basis = Polynomial::constant(1.0);
    for i in 1..n {
        basis = basis * Polynomial::from_coeffs(&[-xs[i - 1], 1.0]);
        result = result + basis.clone() * Polynomial::constant(dd[i]);
    }
    result
}

/// Evaluate a Newton-form polynomial at a single point without constructing the full polynomial.
///
/// Uses the nested multiplication form directly from the divided-difference table.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::interpolate::evaluate_newton;
///
/// let xs = [0.0, 1.0, 2.0];
/// let ys = [1.0, 2.0, 5.0];
/// assert!((evaluate_newton(&xs, &ys, 1.5) - 3.25).abs() < 1e-10);
/// ```
#[must_use]
pub fn evaluate_newton(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let dd = divided_differences(xs, ys);
    let n = xs.len();
    let mut result = dd[n - 1];
    for i in (0..n - 1).rev() {
        result = dd[i] + (x - xs[i]) * result;
    }
    result
}

/// Vandermonde matrix (n×n) for interpolation nodes `xs`.
///
/// Row `i`, column `j` is `xs[i]^j`.
#[must_use]
pub fn vandermonde(xs: &[f64]) -> Vec<Vec<f64>> {
    let n = xs.len();
    (0..n)
        .map(|i| (0..n).map(|j| xs[i].powi(j as i32)).collect())
        .collect()
}

/// Least-squares polynomial fit (NumPy `polyfit` equivalent).
///
/// Fits a degree-`degree` polynomial `p(x) = c0 + c1 x + ... + cd x^d` that
/// minimizes `sum_i (p(xs[i]) - ys[i])²`. The returned [`Polynomial`]
/// coefficients are ordered **lowest-degree first**, matching
/// [`Polynomial::from_coeffs`].
///
/// The normal equations `(VᵀV)c = Vᵀy` are solved by Gaussian elimination
/// with partial pivoting.
///
/// # Numerical notes
///
/// Solving the normal equations squares the condition number of the
/// Vandermonde matrix, which loses precision for high degrees or wide `x`
/// ranges. The fit is reliable for low degrees (typically ≤ 8) and
/// well-conditioned node sets; for high-degree fits, scale/shift `x` to
/// `[-1, 1]` first or prefer a QR/SVD-based solver.
///
/// # Errors
///
/// - `MathError::DimensionMismatch` if `xs` and `ys` differ in length.
/// - `MathError::InvalidArgument` if there are fewer points than
///   `degree + 1` coefficients, or if the system is singular (e.g. duplicate
///   `x` values with an underdetermined fit).
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::interpolate::polyfit;
///
/// // Exact fit through three points of y = x² + x + 1
/// let xs = [0.0, 1.0, 2.0];
/// let ys = [1.0, 3.0, 7.0];
/// let p = polyfit(&xs, &ys, 2).unwrap();
/// assert!((p.eval(0.0) - 1.0).abs() < 1e-9);
/// assert!((p.eval(1.0) - 3.0).abs() < 1e-9);
/// assert!((p.eval(2.0) - 7.0).abs() < 1e-9);
/// ```
pub fn polyfit(
    xs: &[f64],
    ys: &[f64],
    degree: usize,
) -> mathverse_core::error::MathResult<Polynomial> {
    use mathverse_core::error::MathError;

    if xs.len() != ys.len() {
        return Err(MathError::DimensionMismatch);
    }
    let n = xs.len();
    let m = degree + 1;
    if n < m {
        return Err(MathError::InvalidArgument(
            "polyfit: need at least degree+1 data points",
        ));
    }

    // Build the normal equations A = VᵀV, b = Vᵀy (columns = powers of x).
    let mut a = vec![vec![0.0; m]; m];
    let mut b = vec![0.0; m];
    for (&x, &y) in xs.iter().zip(ys) {
        let mut powers = vec![1.0; m];
        for j in 1..m {
            powers[j] = powers[j - 1] * x;
        }
        for j in 0..m {
            b[j] += powers[j] * y;
            for k in 0..m {
                a[j][k] += powers[j] * powers[k];
            }
        }
    }

    let coeffs = solve_linear_system(&a, &b)?;
    Ok(Polynomial::from_coeffs(&coeffs))
}

/// Solve `A x = b` by Gaussian elimination with partial pivoting.
///
/// Returns `MathError::InvalidArgument` when the matrix is singular.
fn solve_linear_system(
    a: &[Vec<f64>],
    b: &[f64],
) -> mathverse_core::error::MathResult<Vec<f64>> {
    use mathverse_core::error::MathError;

    let n = a.len();
    debug_assert_eq!(n, b.len());
    // Augmented matrix.
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivot: largest |entry| in this column at or below the diagonal.
        let mut pivot = col;
        for r in col + 1..n {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        if m[pivot][col].abs() < crate::TOL {
            return Err(MathError::InvalidArgument(
                "polyfit: singular system (e.g. duplicate x values)",
            ));
        }
        m.swap(col, pivot);

        let pivot_val = m[col][col];
        for r in col + 1..n {
            let factor = m[r][col] / pivot_val;
            if factor == 0.0 {
                continue;
            }
            for c in col..=n {
                m[r][c] -= factor * m[col][c];
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut acc = m[r][n];
        for c in r + 1..n {
            acc -= m[r][c] * x[c];
        }
        x[r] = acc / m[r][r];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrange_test() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 3.0, 7.0];
        let p = lagrange(&xs, &ys);
        // Interpolant is x^2 + x + 1 → p(0.5) = 1.75
        let err = (p.eval(0.5) - 1.75).abs();
        assert!(err < 1e-10);
        for i in 0..3 {
            assert!((p.eval(xs[i]) - ys[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn newton_test() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 3.0, 7.0];
        let p = newton(&xs, &ys);
        for i in 0..3 {
            assert!((p.eval(xs[i]) - ys[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn evaluate_newton_test() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 3.0, 7.0];
        let val = evaluate_newton(&xs, &ys, 0.5);
        let p = newton(&xs, &ys);
        assert!((val - p.eval(0.5)).abs() < 1e-12);
    }

    #[test]
    fn vandermonde_test() {
        let xs = [0.0, 1.0, 2.0];
        let v = vandermonde(&xs);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(v[1], vec![1.0, 1.0, 1.0]);
        assert_eq!(v[2], vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn polyfit_exact() {
        // y = x² + x + 1 through 3 points → exact quadratic.
        let xs = [0.0, 1.0, 2.0];
        let ys = [1.0, 3.0, 7.0];
        let p = polyfit(&xs, &ys, 2).unwrap();
        for (i, (&x, &y)) in xs.iter().zip(&ys).enumerate() {
            assert!((p.eval(x) - y).abs() < 1e-9, "point {i}");
        }
    }

    #[test]
    fn polyfit_least_squares() {
        // Noisy line y = 2x + 1; a degree-1 fit recovers slope/intercept.
        let xs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0 + (x * 0.37).sin() * 1e-6).collect();
        let p = polyfit(&xs, &ys, 1).unwrap();
        assert!((p.coeffs()[0] - 1.0).abs() < 1e-6, "intercept {}", p.coeffs()[0]);
        assert!((p.coeffs()[1] - 2.0).abs() < 1e-6, "slope {}", p.coeffs()[1]);
    }

    #[test]
    fn polyfit_errors() {
        assert!(polyfit(&[1.0, 2.0], &[3.0], 1).is_err()); // length mismatch
        assert!(polyfit(&[1.0], &[2.0], 2).is_err()); // too few points
        // Duplicate x with underdetermined system → singular.
        assert!(polyfit(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0], 2).is_err());
    }

    #[test]
    fn polyfit_constant() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [5.0, 5.0, 5.0, 5.0];
        let p = polyfit(&xs, &ys, 0).unwrap();
        assert!((p.eval(0.0) - 5.0).abs() < 1e-9);
    }
}
