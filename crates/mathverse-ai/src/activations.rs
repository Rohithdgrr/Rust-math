//! Activation functions: element-wise and softmax variants with derivatives.

use crate::tensor::Tensor;
use mathverse_core::error::MathResult;

/// ReLU: max(0, x)
pub fn relu(t: &Tensor) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| x.max(0.0)).collect() }
}

/// Derivative of ReLU (1 where x > 0, else 0).
pub fn relu_grad(t: &Tensor) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect() }
}

/// LeakyReLU: x if x > 0, else slope * x.
pub fn leaky_relu(t: &Tensor, slope: f64) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| if x > 0.0 { x } else { slope * x }).collect() }
}

/// Derivative of LeakyReLU (1 where x > 0, else `slope`).
pub fn leaky_relu_grad(t: &Tensor, slope: f64) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| if x > 0.0 { 1.0 } else { slope }).collect() }
}

/// PReLU: x if x > 0, else alpha * x (per-channel alpha).
pub fn prelu(t: &Tensor, alpha: &Tensor) -> MathResult<Tensor> {
    let target = crate::tensor::broadcast_shapes(&t.shape, &alpha.shape)?;
    let a = t.broadcast_to(&target)?;
    let b = alpha.broadcast_to(&target)?;
    let data: Vec<f64> = a.data.iter().zip(&b.data)
        .map(|(&x, &a)| if x > 0.0 { x } else { a * x })
        .collect();
    Ok(Tensor { shape: target, data })
}

/// ELU: x if x > 0, else alpha * (exp(x) - 1).
pub fn elu(t: &Tensor, alpha: f64) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter()
        .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
        .collect() }
}

/// SELU: lambda * (x if x > 0 else alpha * (exp(x) - 1)).
/// lambda ≈ 1.0507, alpha ≈ 1.6733.
pub fn selu(t: &Tensor) -> Tensor {
    let lambda = 1.0507009873554805;
    let alpha = 1.6732632423543772;
    Tensor { shape: t.shape.clone(), data: t.data.iter()
        .map(|&x| if x > 0.0 { lambda * x } else { lambda * alpha * (x.exp() - 1.0) })
        .collect() }
}

/// GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³))).
pub fn gelu(t: &Tensor) -> Tensor {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| {
        let inner = c * (x + 0.044715 * x.powi(3));
        0.5 * x * (1.0 + inner.tanh())
    }).collect() }
}

/// Derivative of GELU (tanh approximation).
pub fn gelu_grad(t: &Tensor) -> Tensor {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| {
        let inner = c * (x + 0.044715 * x.powi(3));
        let tanh_val = inner.tanh();
        let sech2 = 1.0 - tanh_val * tanh_val;
        let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
        0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner
    }).collect() }
}

/// Sigmoid: 1 / (1 + exp(-x)), numerically stable.
pub fn sigmoid(t: &Tensor) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| {
        if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) }
        else { let e = x.exp(); e / (1.0 + e) }
    }).collect() }
}

/// Derivative of [`sigmoid`].
pub fn sigmoid_grad(t: &Tensor) -> Tensor {
    let s = sigmoid(t);
    Tensor { shape: t.shape.clone(), data: s.data.iter().map(|&v| v * (1.0 - v)).collect() }
}

/// Hyperbolic tangent.
pub fn tanh(t: &Tensor) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|x| x.tanh()).collect() }
}

/// Derivative of [`tanh`].
pub fn tanh_grad(t: &Tensor) -> Tensor {
    let h = tanh(t);
    Tensor { shape: h.shape.clone(), data: h.data.iter().map(|&v| 1.0 - v * v).collect() }
}

/// Swish / SiLU: x * sigmoid(x).
pub fn swish(t: &Tensor) -> Tensor {
    let s = sigmoid(t);
    Tensor { shape: t.shape.clone(), data: t.data.iter().zip(&s.data).map(|(&x, &s)| x * s).collect() }
}

/// Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
pub fn mish(t: &Tensor) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| {
        let sp = (1.0 + x.exp()).ln();
        x * sp.tanh()
    }).collect() }
}

/// Softplus: ln(1 + exp(x)), numerically stable.
pub fn softplus(t: &Tensor) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| {
        if x > 20.0 { x } else if x < -20.0 { 0.0 } else { (1.0 + x.exp()).ln() }
    }).collect() }
}

/// Softsign: x / (1 + |x|).
pub fn softsign(t: &Tensor) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| x / (1.0 + x.abs())).collect() }
}

/// Hard sigmoid: clamp(x/6 + 0.5, 0, 1).
pub fn hard_sigmoid(t: &Tensor) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|&x| (x / 6.0 + 0.5).clamp(0.0, 1.0)).collect() }
}

/// Hard tanh: clamp(x, -1, 1).
pub fn hard_tanh(t: &Tensor) -> Tensor {
    Tensor { shape: t.shape.clone(), data: t.data.iter().map(|x| x.clamp(-1.0, 1.0)).collect() }
}

/// Identity (pass-through).
pub fn identity(t: &Tensor) -> Tensor { t.clone() }

/// Softmax along axis with numerical stability (max-subtracted).
pub fn softmax(t: &Tensor, axis: usize) -> MathResult<Tensor> {
    if axis >= t.shape.len() {
        return Err(mathverse_core::error::MathError::InvalidArgument("axis out of range"));
    }
    let axis_size = t.shape[axis];
    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = t.shape[axis + 1..].iter().product();
    let mut out = vec![0.0; t.numel()];

    for i in 0..outer {
        for j in 0..inner {
            // Find max for numerical stability
            let mut max_val = f64::NEG_INFINITY;
            for k in 0..axis_size {
                let idx = i * axis_size * inner + k * inner + j;
                if t.data[idx] > max_val { max_val = t.data[idx]; }
            }
            // Compute exp and sum
            let mut sum = 0.0;
            for k in 0..axis_size {
                let idx = i * axis_size * inner + k * inner + j;
                let e = (t.data[idx] - max_val).exp();
                out[idx] = e;
                sum += e;
            }
            // Normalize
            for k in 0..axis_size {
                let idx = i * axis_size * inner + k * inner + j;
                out[idx] /= sum;
            }
        }
    }
    Ok(Tensor { shape: t.shape.clone(), data: out })
}

/// Log-softmax: computed directly as x - max - log(sum(exp(x - max))) for numerical stability.
pub fn log_softmax(t: &Tensor, axis: usize) -> MathResult<Tensor> {
    if axis >= t.shape.len() {
        return Err(mathverse_core::error::MathError::InvalidArgument("axis out of range"));
    }
    let axis_size = t.shape[axis];
    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = t.shape[axis + 1..].iter().product();
    let mut out = vec![0.0; t.numel()];

    for i in 0..outer {
        for j in 0..inner {
            let mut max_val = f64::NEG_INFINITY;
            for k in 0..axis_size {
                let idx = i * axis_size * inner + k * inner + j;
                if t.data[idx] > max_val { max_val = t.data[idx]; }
            }
            let mut sum_exp = 0.0;
            for k in 0..axis_size {
                let idx = i * axis_size * inner + k * inner + j;
                sum_exp += (t.data[idx] - max_val).exp();
            }
            let log_sum = sum_exp.ln();
            for k in 0..axis_size {
                let idx = i * axis_size * inner + k * inner + j;
                out[idx] = t.data[idx] - max_val - log_sum;
            }
        }
    }
    Ok(Tensor { shape: t.shape.clone(), data: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-6;

    #[test]
    fn relu_test() {
        let t = Tensor::new(&[5], &[-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let r = relu(&t);
        assert_eq!(r.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
        let g = relu_grad(&t);
        assert_eq!(g.data, vec![0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn sigmoid_test() {
        let t = Tensor::new(&[3], &[-1.0, 0.0, 1.0]).unwrap();
        let s = sigmoid(&t);
        assert!((s.data[1] - 0.5).abs() < E);
        assert!((s.data[0] + s.data[2] - 1.0).abs() < E);
    }

    #[test]
    fn softmax_test() {
        let t = Tensor::new(&[2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
        let s = softmax(&t, 1).unwrap();
        // Each row sums to 1
        let row0: f64 = (0..3).map(|j| s.data[j]).sum();
        assert!((row0 - 1.0).abs() < E);
        // Largest element has highest probability
        assert!(s.data[2] > s.data[0]);
    }

    #[test]
    fn softmax_stability() {
        let t = Tensor::new(&[3], &[1000.0, 1001.0, 1002.0]).unwrap();
        let s = softmax(&t, 0).unwrap();
        let sum: f64 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < E);
        assert!(s.data[2] > s.data[0]);
    }

    #[test]
    fn gelu_test() {
        let t = Tensor::new(&[1], &[0.0]).unwrap();
        let g = gelu(&t);
        assert!(g.data[0].abs() < E); // gelu(0) = 0
    }

    #[test]
    fn tanh_test() {
        let t = Tensor::new(&[1], &[0.0]).unwrap();
        let h = tanh(&t);
        assert!(h.data[0].abs() < E);
    }

    #[test]
    fn softplus_test() {
        let t = Tensor::new(&[1], &[0.0]).unwrap();
        let s = softplus(&t);
        assert!((s.data[0] - 2.0_f64.ln()).abs() < E); // ln(1 + e^0) = ln(2)
    }

    #[test]
    fn swish_test() {
        let t = Tensor::new(&[1], &[0.0]).unwrap();
        let s = swish(&t);
        assert!(s.data[0].abs() < E); // swish(0) = 0
    }

    #[test]
    fn mish_test() {
        let t = Tensor::new(&[1], &[0.0]).unwrap();
        let m = mish(&t);
        assert!(m.data[0].abs() < E); // mish(0) = 0
    }

    #[test]
    fn log_softmax_test() {
        let t = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
        let ls = log_softmax(&t, 0).unwrap();
        let sum: f64 = ls.data.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < E);
    }
}
