//! Complex-valued activation functions and their derivatives for
//! complex-valued neural networks.
//!
//! These activations operate element-wise on [`Complex`] values and are
//! designed for use in architectures like complex-valued CNNs, RNNs, and
//! transformers used in speech enhancement, direction-of-arrival estimation,
//! and MRI reconstruction.
//!
//! # Module overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`complex_relu`] | Complex `ReLU`: max(0, z) applied to real and imaginary parts |
//! | [`complex_sigmoid`] | 1 / (1 + e^(-z)) with full complex arithmetic |
//! | [`complex_tanh`] | tanh(z) with complex identity |
//! | [`complex_softmax`] | Column-wise softmax for complex vectors |
//! | [`mod_relu`] | Phase-preserving activation: `ReLU`(|z|) · e^(i·arg(z)) |
//! | [`complex_leaky_relu`] | Leaky `ReLU` applied independently to re and im |
//! | [`complex_elu`] | ELU with complex-aware negative region |
//! | [`complex_gelu`] | GELU approximation via erf |
//!
//! Each activation has a corresponding `_grad` function that computes
//! the derivative `d/dz` (where it exists), suitable for backpropagation
//! through complex layers.

use crate::Complex;

/// Complex `ReLU`: apply real-ReLU independently to real and imaginary parts.
///
/// `f(z) = max(0, Re(z)) + i · max(0, Im(z))`
pub fn complex_relu(z: Complex) -> Complex {
    Complex::new(z.re.max(0.0), z.im.max(0.0))
}

/// Derivative of [`complex_relu`]: 1 where active, 0 where saturated.
pub fn complex_relu_grad(z: Complex) -> Complex {
    Complex::new(
        if z.re > 0.0 { 1.0 } else { 0.0 },
        if z.im > 0.0 { 1.0 } else { 0.0 },
    )
}

/// Complex sigmoid: `σ(z) = 1 / (1 + e^(-z))` with full complex arithmetic.
///
/// Uses the numerically stable form: for Re(z) ≥ 0, compute directly;
/// for Re(z) < 0, use `e^z / (1 + e^z)` to avoid overflow.
pub fn complex_sigmoid(z: Complex) -> Complex {
    if z.re >= 0.0 {
        let ez = (-z).exp();
        Complex::one() / (Complex::one() + ez)
    } else {
        let ez = z.exp();
        ez / (Complex::one() + ez)
    }
}

/// Derivative of [`complex_sigmoid`]: `σ(z) · (1 − σ(z))`.
pub fn complex_sigmoid_grad(z: Complex) -> Complex {
    let s = complex_sigmoid(z);
    s * (Complex::one() - s)
}

/// Complex hyperbolic tangent: `tanh(z)` using the full complex formula.
///
/// This is mathematically identical to `Complex::tanh()` but provided
/// as an activation for API consistency.
pub fn complex_tanh(z: Complex) -> Complex {
    z.tanh()
}

/// Derivative of [`complex_tanh`]: `1 − tanh²(z)`.
pub fn complex_tanh_grad(z: Complex) -> Complex {
    let t = z.tanh();
    Complex::one() - t * t
}

/// Column-wise complex softmax over a slice of complex values.
///
/// `softmax(z_k) = e^(z_k) / Σ_j e^(z_j)` computed with max-subtraction
/// for numerical stability.
pub fn complex_softmax(values: &[Complex]) -> Vec<Complex> {
    if values.is_empty() {
        return Vec::new();
    }
    let max_val = values
        .iter()
        .map(|v| v.re)
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<Complex> = values
        .iter()
        .map(|v| (*v - Complex::real(max_val)).exp())
        .collect();
    let sum: Complex = exps.iter().copied().fold(Complex::zero(), |a, b| a + b);
    exps.into_iter().map(|e| e / sum).collect()
}

/// Modulus `ReLU` (modReLU): phase-preserving activation.
///
/// `f(z) = ReLU(|z|) · e^(i·arg(z))`
///
/// Zeroes out small-magnitude complex numbers while preserving the phase
/// of larger ones. Used in complex-valued RNNs (e.g. Deep Complex Networks).
pub fn mod_relu(z: Complex) -> Complex {
    let r = z.norm();
    if r > 0.0 {
        z * (Complex::real(r.max(0.0)) / Complex::real(r))
    } else {
        Complex::zero()
    }
}

/// Derivative of [`mod_relu`]: `e^(i·arg(z))` where |z| > 0, else 0.
pub fn mod_relu_grad(z: Complex) -> Complex {
    let r = z.norm();
    if r > 0.0 {
        z / Complex::real(r)
    } else {
        Complex::zero()
    }
}

/// Complex leaky `ReLU`: apply leaky-ReLU independently to re and im.
///
/// `f(z) = max(0, Re(z)) + α·min(0, Re(z)) + i·[max(0, Im(z)) + α·min(0, Im(z))]`
pub fn complex_leaky_relu(z: Complex, alpha: f64) -> Complex {
    Complex::new(
        if z.re >= 0.0 { z.re } else { alpha * z.re },
        if z.im >= 0.0 { z.im } else { alpha * z.im },
    )
}

/// Derivative of [`complex_leaky_relu`].
pub fn complex_leaky_relu_grad(z: Complex, alpha: f64) -> Complex {
    Complex::new(
        if z.re >= 0.0 { 1.0 } else { alpha },
        if z.im >= 0.0 { 1.0 } else { alpha },
    )
}

/// Complex ELU: Exponential Linear Unit applied to re and im.
///
/// `f(z) = z` if z ≥ 0, else `α · (e^z − 1)` (applied independently per component).
pub fn complex_elu(z: Complex, alpha: f64) -> Complex {
    Complex::new(
        if z.re >= 0.0 {
            z.re
        } else {
            alpha * (z.re.exp() - 1.0)
        },
        if z.im >= 0.0 {
            z.im
        } else {
            alpha * (z.im.exp() - 1.0)
        },
    )
}

/// Derivative of [`complex_elu`].
pub fn complex_elu_grad(z: Complex, alpha: f64) -> Complex {
    Complex::new(
        if z.re >= 0.0 {
            1.0
        } else {
            alpha * z.re.exp()
        },
        if z.im >= 0.0 {
            1.0
        } else {
            alpha * z.im.exp()
        },
    )
}

/// Complex GELU (Gaussian Error Linear Unit) approximation.
///
/// Uses the tanh approximation: `GELU(z) ≈ 0.5·z·(1 + tanh(√(2/π)·(z + 0.044715·z³)))`.
/// Applied independently to real and imaginary parts.
pub fn complex_gelu(z: Complex) -> Complex {
    let c = (2.0 / core::f64::consts::PI).sqrt();
    let inner = Complex::real(c) * (z + Complex::real(0.044715) * z * z * z);
    Complex::real(0.5) * z * (Complex::one() + inner.tanh())
}

/// Derivative of [`complex_gelu`] (numerical approximation).
pub fn complex_gelu_grad(z: Complex) -> Complex {
    let h = 1e-7;
    (complex_gelu(z + Complex::real(h)) - complex_gelu(z - Complex::real(h)))
        / Complex::real(2.0 * h)
}

/// Apply an activation function to every element of a matrix.
pub fn apply_activation(
    m: &crate::ComplexMatrix,
    f: &dyn Fn(Complex) -> Complex,
) -> crate::ComplexMatrix {
    let mut result = m.clone();
    for val in &mut result.data {
        *val = f(*val);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn relu_basic() {
        assert_eq!(complex_relu(Complex::new(3.0, -2.0)), Complex::new(3.0, 0.0));
        assert_eq!(complex_relu(Complex::new(-1.0, 4.0)), Complex::new(0.0, 4.0));
        assert_eq!(complex_relu(Complex::zero()), Complex::zero());
    }

    #[test]
    fn relu_grad_values() {
        let g = complex_relu_grad(Complex::new(2.0, -3.0));
        assert_eq!(g, Complex::new(1.0, 0.0));
        let g2 = complex_relu_grad(Complex::new(-1.0, 5.0));
        assert_eq!(g2, Complex::new(0.0, 1.0));
    }

    #[test]
    fn sigmoid_at_zero() {
        let s = complex_sigmoid(Complex::zero());
        assert!((s.re - 0.5).abs() < EPS);
        assert!(s.im.abs() < EPS);
    }

    #[test]
    fn sigmoid_large_positive() {
        let s = complex_sigmoid(Complex::real(20.0));
        assert!((s.re - 1.0).abs() < 1e-8);
    }

    #[test]
    fn sigmoid_large_negative() {
        let s = complex_sigmoid(Complex::real(-20.0));
        assert!(s.re.abs() < 1e-8);
    }

    #[test]
    fn sigmoid_grad_equals_s_times_one_minus_s() {
        let z = Complex::new(1.5, -0.7);
        let g = complex_sigmoid_grad(z);
        let s = complex_sigmoid(z);
        let expected = s * (Complex::one() - s);
        assert!((g - expected).norm() < 1e-12);
    }

    #[test]
    fn tanh_at_zero() {
        let t = complex_tanh(Complex::zero());
        assert!(t.norm() < EPS);
    }

    #[test]
    fn tanh_imaginary_unit() {
        // tanh(i) = i·tan(1)
        let t = complex_tanh(Complex::i());
        assert!((t.re).abs() < EPS);
        assert!((t.im - 1.0_f64.tan()).abs() < 1e-10);
    }

    #[test]
    fn tanh_grad_identity() {
        let z = Complex::new(0.5, 1.2);
        let g = complex_tanh_grad(z);
        let t = z.tanh();
        let expected = Complex::one() - t * t;
        assert!((g - expected).norm() < 1e-12);
    }

    #[test]
    fn softmax_sums_to_one() {
        let vals = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 1.0), Complex::new(0.5, -0.5)];
        let s = complex_softmax(&vals);
        let total: Complex = s.iter().copied().fold(Complex::zero(), |a, b| a + b);
        assert!((total - Complex::one()).norm() < 1e-12);
    }

    #[test]
    fn softmax_equal_inputs() {
        let vals = vec![Complex::real(1.0); 4];
        let s = complex_softmax(&vals);
        for v in &s {
            assert!((v.re - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn mod_relu_zeroes_small() {
        // mod_relu preserves magnitude for nonzero inputs
        let z = Complex::new(0.01, 0.02);
        let r = mod_relu(z);
        assert!((r.norm() - z.norm()).abs() < 1e-12);
        // Zero input stays zero
        let r0 = mod_relu(Complex::zero());
        assert!(r0.norm() < 1e-15);
    }

    #[test]
    fn mod_relu_preserves_phase() {
        let z = Complex::new(3.0, 4.0);
        let r = mod_relu(z);
        let phase_diff = (r.arg() - z.arg()).abs();
        assert!(phase_diff < 1e-12 || (phase_diff - core::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn leaky_relu_positive_part() {
        let z = Complex::new(5.0, 3.0);
        let r = complex_leaky_relu(z, 0.01);
        assert!((r.re - 5.0).abs() < EPS);
        assert!((r.im - 3.0).abs() < EPS);
    }

    #[test]
    fn leaky_relu_negative_part() {
        let z = Complex::new(-5.0, -3.0);
        let r = complex_leaky_relu(z, 0.1);
        assert!((r.re - (-0.5)).abs() < EPS);
        assert!((r.im - (-0.3)).abs() < EPS);
    }

    #[test]
    fn elu_positive_is_identity() {
        let z = Complex::new(2.0, 3.0);
        let e = complex_elu(z, 1.0);
        assert!((e - z).norm() < EPS);
    }

    #[test]
    fn elu_negative_has_exponential() {
        let z = Complex::real(-2.0);
        let e = complex_elu(z, 1.0);
        let expected = (-2.0_f64).exp() - 1.0;
        assert!((e.re - expected).abs() < 1e-10);
    }

    #[test]
    fn gelu_near_zero() {
        // GELU(0) = 0
        let g = complex_gelu(Complex::zero());
        assert!(g.norm() < 1e-12);
    }

    #[test]
    fn gelu_large_positive() {
        // GELU(x) ≈ x for large positive x
        let g = complex_gelu(Complex::real(10.0));
        assert!((g.re - 10.0).abs() < 0.01);
    }
}
