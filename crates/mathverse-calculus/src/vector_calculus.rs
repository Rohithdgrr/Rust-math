//! Vector calculus: gradient, divergence, curl, Laplacian (numerical).
//!
//! Scalar fields: `&dyn Fn(&[f64]) -> f64`. Vector fields: `&dyn Fn(&[f64]) -> Vec<f64>`.

use crate::derivative::{partial_derivative, second_derivative};
use mathverse_core::error::{MathError, MathResult};

/// `∇f`: partial derivatives in every coordinate.
///
/// ```
/// use mathverse_calculus::vector_calculus::gradient;
/// let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
/// let g = gradient(&f, &[1.0, 2.0]);
/// assert!((g[0] - 2.0).abs() < 1e-6 && (g[1] - 4.0).abs() < 1e-6);
/// ```
pub fn gradient(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    (0..x.len()).map(|i| partial_derivative(f, x, i)).collect()
}

/// `∇·F`: sum of `∂F_i/∂x_i`.
///
/// Returns [`MathError::DimensionMismatch`] if `f` returns a different number
/// of components than `x` has coordinates.
pub fn divergence(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> MathResult<f64> {
    let fx = f(x);
    if fx.len() != x.len() {
        return Err(MathError::DimensionMismatch);
    }
    Ok((0..x.len())
        .map(|i| partial_derivative(&|p: &[f64]| f(p)[i], x, i))
        .sum())
}

/// `∇×F` in 3D; [`MathError::DimensionMismatch`] if `x` isn't length 3.
///
/// ```
/// use mathverse_calculus::vector_calculus::curl;
/// let f = |x: &[f64]| vec![x[1], x[2], x[0]];
/// let c = curl(&f, &[1.0, 2.0, 3.0]).unwrap();
/// assert!(c.iter().all(|&v| (v + 1.0).abs() < 1e-6));
/// ```
pub fn curl(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> MathResult<Vec<f64>> {
    if x.len() != 3 {
        return Err(MathError::DimensionMismatch);
    }
    if f(x).len() < 3 {
        return Err(MathError::DimensionMismatch);
    }
    let fz = |p: &[f64]| f(p)[2];
    let fy = |p: &[f64]| f(p)[1];
    let fx = |p: &[f64]| f(p)[0];
    Ok(vec![
        partial_derivative(&fz, x, 1) - partial_derivative(&fy, x, 2),
        partial_derivative(&fx, x, 2) - partial_derivative(&fz, x, 0),
        partial_derivative(&fy, x, 0) - partial_derivative(&fx, x, 1),
    ])
}

/// `∇²f`: sum of second partials.
pub fn laplacian(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> f64 {
    // Use global scale for consistent step across all dimensions
    let scale = x.iter().map(|&xi| xi.abs()).fold(0.0, f64::max).max(1.0);
    let mut scratch = x.to_vec();
    let h = 1e-3 * scale;
    (0..x.len())
        .map(|i| {
            let val = {
                let mut fi = |t: f64| {
                    scratch[i] = t;
                    f(&scratch)
                };
                (fi(x[i] + h) - 2.0 * fi(x[i]) + fi(x[i] - h)) / (h * h)
            };
            scratch[i] = x[i]; // restore
            val
        })
        .sum()
}

/// Jacobian matrix `J_ij = ∂F_i/∂x_j` at point `x`.
///
/// Returns a flattened vector where J[i][j] is stored at index i*m + j.
///
/// ```
/// use mathverse_calculus::vector_calculus::jacobian;
/// let f = |x: &[f64]| vec![x[0] * x[1], x[0] + x[1]];
/// let j = jacobian(&f, &[2.0, 3.0]);
/// // J = [[3, 2], [1, 1]]
/// assert!((j[0] - 3.0).abs() < 1e-6); // ∂F₀/∂x₀
/// assert!((j[1] - 2.0).abs() < 1e-6); // ∂F₀/∂x₁
/// assert!((j[2] - 1.0).abs() < 1e-6); // ∂F₁/∂x₀
/// assert!((j[3] - 1.0).abs() < 1e-6); // ∂F₁/∂x₁
/// ```
pub fn jacobian(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> Vec<f64> {
    let m = f(x).len();
    let n = x.len();
    let mut j = vec![0.0; m * n];
    for i in 0..m {
        for j_idx in 0..n {
            j[i * n + j_idx] = partial_derivative(&|p: &[f64]| f(p)[i], x, j_idx);
        }
    }
    j
}

/// Hessian matrix `H_ij = ∂²f/∂x_i∂x_j` at point `x`.
///
/// Returns a flattened vector where H[i][j] is stored at index i*n + j.
/// The Hessian is symmetric for smooth functions.
///
/// ```
/// use mathverse_calculus::vector_calculus::hessian;
/// let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
/// let h = hessian(&f, &[1.0, 2.0]);
/// // H = [[2, 0], [0, 2]]
/// assert!((h[0] - 2.0).abs() < 1e-4); // ∂²f/∂x₀²
/// assert!((h[1] - 0.0).abs() < 1e-4); // ∂²f/∂x₀∂x₁
/// assert!((h[2] - 0.0).abs() < 1e-4); // ∂²f/∂x₁∂x₀
/// assert!((h[3] - 2.0).abs() < 1e-4); // ∂²f/∂x₁²
/// ```
pub fn hessian(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut h = vec![0.0; n * n];
    // Global scale for consistent step across all dimensions
    let scale = x.iter().map(|&xi| xi.abs()).fold(0.0, f64::max).max(1.0);
    let step = f64::EPSILON.powf(0.25) * scale; // ε^(1/4) optimal for mixed partials

    for i in 0..n {
        // Diagonal: use second_derivative directly
        h[i * n + i] = second_derivative(
            &|t| {
                let mut p = x.to_vec();
                p[i] = t;
                f(&p)
            },
            x[i],
        );
    }

    // Off-diagonal: 4-point stencil with theoretically optimal step
    for i in 0..n {
        for j in (i + 1)..n {
            let mut p1 = x.to_vec();
            let mut p2 = x.to_vec();
            let mut p3 = x.to_vec();
            let mut p4 = x.to_vec();
            p1[i] += step;
            p1[j] += step;
            p2[i] += step;
            p2[j] -= step;
            p3[i] -= step;
            p3[j] += step;
            p4[i] -= step;
            p4[j] -= step;
            let mixed = (f(&p1) - f(&p2) - f(&p3) + f(&p4)) / (4.0 * step * step);
            h[i * n + j] = mixed;
            h[j * n + i] = mixed; // exact symmetry
        }
    }
    h
}

/// Directional derivative `∇f·v` at point `x` in direction `v`.
///
/// The direction vector `v` is automatically normalized.
/// Returns [`MathError::DimensionMismatch`] if `v.len() != x.len()`.
///
/// ```
/// use mathverse_calculus::vector_calculus::directional_derivative;
/// let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
/// let v = vec![1.0, 0.0]; // x-direction
/// assert!((directional_derivative(&f, &[1.0, 2.0], &v).unwrap() - 2.0).abs() < 1e-6);
/// ```
pub fn directional_derivative(f: &dyn Fn(&[f64]) -> f64, x: &[f64], v: &[f64]) -> MathResult<f64> {
    if v.len() != x.len() {
        return Err(MathError::DimensionMismatch);
    }
    let grad = gradient(f, x);
    let norm: f64 = v.iter().map(|&vi| vi * vi).sum::<f64>().sqrt();
    if norm == 0.0 {
        return Ok(0.0);
    }
    Ok(grad.iter().zip(v.iter()).map(|(&gi, &vi)| gi * vi / norm).sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_and_laplacian() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
        let g = gradient(&f, &[1.0, 2.0, 3.0]);
        assert!((g[0] - 2.0).abs() < 1e-6);
        assert!((g[1] - 4.0).abs() < 1e-6);
        assert!((g[2] - 6.0).abs() < 1e-6);
        assert!((laplacian(&f, &[1.0, 2.0, 3.0]) - 6.0).abs() < 1e-4);
    }

    #[test]
    fn divergence_and_curl() {
        let f = |x: &[f64]| x.to_vec();
        assert!((divergence(&f, &[1.0, 2.0, 3.0]).unwrap() - 3.0).abs() < 1e-6);
        let rot = |x: &[f64]| vec![x[1], x[2], x[0]];
        let c = curl(&rot, &[1.0, 2.0, 3.0]).unwrap();
        assert!(c.iter().all(|&v| (v + 1.0).abs() < 1e-6));
        // irrotational: gradient field has zero curl
        let g = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1], 2.0 * x[2]];
        assert!(curl(&g, &[1.0, 2.0, 3.0]).unwrap().iter().all(|&v| v.abs() < 1e-6));
        assert!(curl(&rot, &[1.0, 2.0]).is_err());
        // field returning fewer components than x has coordinates
        assert!(divergence(&|x: &[f64]| vec![x[0]], &[1.0, 2.0]).is_err());
        assert!(curl(&|x: &[f64]| vec![x[0]], &[1.0, 2.0, 3.0]).is_err());
        // field returning MORE components than x has coordinates (strict check)
        assert!(divergence(&|x: &[f64]| vec![x[0], 0.0, 0.0], &[1.0]).is_err());
    }

    #[test]
    fn jacobian_test() {
        let f = |x: &[f64]| vec![x[0] * x[1], x[0] + x[1]];
        let j = jacobian(&f, &[2.0, 3.0]);
        assert!((j[0] - 3.0).abs() < 1e-6);
        assert!((j[1] - 2.0).abs() < 1e-6);
        assert!((j[2] - 1.0).abs() < 1e-6);
        assert!((j[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hessian_test() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let h = hessian(&f, &[1.0, 2.0]);
        assert!((h[0] - 2.0).abs() < 1e-4);
        assert!((h[1] - 0.0).abs() < 1e-4);
        assert!((h[2] - 0.0).abs() < 1e-4);
        assert!((h[3] - 2.0).abs() < 1e-4);
        // mixed partials of x*y: symmetric matrix
        let f2 = |x: &[f64]| x[0] * x[1];
        let h2 = hessian(&f2, &[3.0, 4.0]);
        assert!((h2[1] - 1.0).abs() < 1e-4 && (h2[2] - 1.0).abs() < 1e-4);
        assert!((h2[1] - h2[2]).abs() < 1e-12);
    }

    #[test]
    fn directional_derivative_test() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let v = vec![1.0, 0.0];
        assert!((directional_derivative(&f, &[1.0, 2.0], &v).unwrap() - 2.0).abs() < 1e-6);
        let v2 = vec![0.0, 1.0];
        assert!((directional_derivative(&f, &[1.0, 2.0], &v2).unwrap() - 4.0).abs() < 1e-6);
        // Zero direction → 0
        assert!((directional_derivative(&f, &[1.0, 2.0], &[0.0, 0.0]).unwrap()).abs() < 1e-10);
        // Dimension mismatch
        assert!(directional_derivative(&f, &[1.0, 2.0], &[1.0]).is_err());
    }
}
