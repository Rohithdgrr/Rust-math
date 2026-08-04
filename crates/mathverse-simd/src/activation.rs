//! SIMD-friendly activation function kernels.

/// Sigmoid: `out[i] = 1 / (1 + exp(-a[i]))`.
///
/// Uses the numerically stable formulation to avoid overflow.
#[inline]
pub fn sigmoid(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = if v >= 0.0 {
            1.0 / (1.0 + (-v).exp())
        } else {
            let ev = v.exp();
            ev / (1.0 + ev)
        };
    }
}

/// ReLU: `out[i] = max(0, a[i])`.
#[inline]
pub fn relu(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.max(0.0);
    }
}

/// Leaky ReLU: `out[i] = a[i] > 0 ? a[i] : alpha * a[i]`.
#[inline]
pub fn leaky_relu(a: &[f64], alpha: f64, out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = if *v > 0.0 { *v } else { alpha * v };
    }
}

/// GELU approximation: `out[i] = 0.5 * a[i] * (1 + tanh(sqrt(2/pi) * (a[i] + 0.044715 * a[i]^3)))`.
#[inline]
pub fn gelu(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
    const COEFF: f64 = 0.044_715;
    for (o, &v) in out.iter_mut().zip(a) {
        let inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
        *o = 0.5 * v * (1.0 + inner.tanh());
    }
}

/// SiLU / Swish: `out[i] = a[i] * sigmoid(a[i])`.
#[inline]
pub fn silu(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        let s = if *v >= 0.0 {
            1.0 / (1.0 + (-v).exp())
        } else {
            let ev = v.exp();
            ev / (1.0 + ev)
        };
        *o = v * s;
    }
}

/// Softmax: `out[i] = exp(a[i]) / sum(exp(a[j]))`.
///
/// Uses the max-subtraction trick for numerical stability.
#[inline]
pub fn softmax(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    if a.is_empty() {
        return;
    }

    let max_val = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for (o, &v) in out.iter_mut().zip(a) {
        *o = (v - max_val).exp();
        sum += *o;
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
}

/// ELU: `out[i] = a[i] > 0 ? a[i] : alpha * (exp(a[i]) - 1)`.
#[inline]
pub fn elu(a: &[f64], alpha: f64, out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = if *v > 0.0 {
            *v
        } else {
            alpha * (v.exp() - 1.0)
        };
    }
}

/// Tanh activation: `out[i] = tanh(a[i])`.
#[inline]
pub fn tanh_act(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.tanh();
    }
}

/// Mish: `out[i] = a[i] * tanh(ln(1 + exp(a[i])))`.
#[inline]
pub fn mish(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        let sp = (1.0 + v.exp()).ln();
        *o = v * sp.tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let a = [0.0, 100.0, -100.0];
        let mut out = [0.0; 3];
        sigmoid(&a, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-15);
        assert!((out[1] - 1.0).abs() < 1e-10);
        assert!((out[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu() {
        let a = [-1.0, 0.0, 1.0, 2.0];
        let mut out = [0.0; 4];
        relu(&a, &mut out);
        assert_eq!(out, [0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_softmax() {
        let a = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        softmax(&a, &mut out);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // Softmax preserves order
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
    }

    #[test]
    fn test_gelu() {
        let a = [0.0];
        let mut out = [0.0; 1];
        gelu(&a, &mut out);
        assert!((out[0]).abs() < 1e-15);
    }

    #[test]
    fn test_silu() {
        let a = [0.0];
        let mut out = [0.0; 1];
        silu(&a, &mut out);
        assert!((out[0]).abs() < 1e-15);
    }
}
