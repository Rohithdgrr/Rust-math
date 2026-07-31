//! Neural network layers: Linear, Conv1d, Conv2d, LayerNorm, BatchNorm, Dropout.

use crate::tensor::Tensor;
use mathverse_core::error::MathResult;

/// Linear (fully connected) layer: y = x @ W^T + b.
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let std = (2.0 / in_features as f64).sqrt();
        let weight = Tensor::randn(&[out_features, in_features]).mul_scalar(std);
        let bias = Tensor::zeros(&[out_features]);
        Self { weight, bias, in_features, out_features }
    }

    /// Xavier/Glorot initialization.
    pub fn xavier(in_features: usize, out_features: usize) -> Self {
        let std = (2.0 / (in_features + out_features) as f64).sqrt();
        let weight = Tensor::randn(&[out_features, in_features]).mul_scalar(std);
        let bias = Tensor::zeros(&[out_features]);
        Self { weight, bias, in_features, out_features }
    }

    /// Forward pass: [*, in_features] -> [*, out_features] (flattens leading dims).
    pub fn forward(&self, x: &Tensor) -> MathResult<Tensor> {
        let orig_shape = x.shape.clone();
        let batch_dims: usize = orig_shape[..orig_shape.len().saturating_sub(1)].iter().product();
        let flat = x.reshape(&[batch_dims, self.in_features])?;
        let wt = self.weight.transpose()?;
        let out = flat.matmul(&wt)?;
        let mut out_shape = orig_shape;
        let last = out_shape.len() - 1;
        out_shape[last] = self.out_features;
        let bias_2d = self.bias.unsqueeze(0).broadcast_to(&[batch_dims, self.out_features])?;
        out.add(&bias_2d)?.reshape(&out_shape)
    }

    /// Number of parameters.
    pub fn num_params(&self) -> usize { self.in_features * self.out_features + self.out_features }
}

/// Layer normalization with learnable gamma (scale) and beta (shift).
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub eps: f64,
    pub normalized_shape: usize,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize, eps: f64) -> Self {
        Self {
            gamma: Tensor::ones(&[normalized_shape]),
            beta: Tensor::zeros(&[normalized_shape]),
            eps,
            normalized_shape,
        }
    }

    /// Forward: normalize over last dimension, then affine.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let nd = x.shape.len();
        let d = self.normalized_shape;
        let outer: usize = x.shape[..nd - 1].iter().product();
        let mut out = vec![0.0; x.numel()];
        for i in 0..outer {
            let start = i * d;
            let slice = &x.data[start..start + d];
            let mu: f64 = slice.iter().sum::<f64>() / d as f64;
            let var: f64 = slice.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + self.eps).sqrt();
            for j in 0..d {
                let norm = (x.data[start + j] - mu) * inv;
                out[start + j] = self.gamma.data[j] * norm + self.beta.data[j];
            }
        }
        Tensor { shape: x.shape.clone(), data: out }
    }
}

/// Batch normalization with learnable gamma, beta, running mean/var.
pub struct BatchNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub eps: f64,
    pub momentum: f64,
    pub num_features: usize,
}

impl BatchNorm {
    pub fn new(num_features: usize, eps: f64, momentum: f64) -> Self {
        Self {
            gamma: Tensor::ones(&[num_features]),
            beta: Tensor::zeros(&[num_features]),
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            eps,
            momentum,
            num_features,
        }
    }

    /// Forward pass over first dimension.
    pub fn forward(&self, x: &Tensor, training: bool) -> MathResult<Tensor> {
        if x.shape.len() < 2 { return Err(mathverse_core::error::MathError::InvalidArgument("BatchNorm needs >= 2D input")); }
        let batch = x.shape[0];
        let feature_size: usize = x.shape[1..].iter().product();
        let per_sample = x.numel() / batch;
        let mut out = vec![0.0; x.numel()];
        for f in 0..feature_size {
            let (mu, var) = if training {
                let mut sum = 0.0;
                let mut sum2 = 0.0;
                for b in 0..batch {
                    let v = x.data[b * per_sample + f];
                    sum += v;
                    sum2 += v * v;
                }
                let mu = sum / batch as f64;
                let var = sum2 / batch as f64 - mu * mu;
                (mu, var)
            } else {
                (self.running_mean.data[f], self.running_var.data[f])
            };
            let inv = 1.0 / (var + self.eps).sqrt();
            for b in 0..batch {
                let idx = b * per_sample + f;
                let norm = (x.data[idx] - mu) * inv;
                out[idx] = self.gamma.data[f] * norm + self.beta.data[f];
            }
        }
        Ok(Tensor { shape: x.shape.clone(), data: out })
    }
}

/// Dropout layer (inverted dropout).
pub struct Dropout {
    pub p: f64,
}

impl Dropout {
    pub fn new(p: f64) -> Self { Self { p } }

    /// Forward: randomly zero out elements with probability p, scale by 1/(1-p).
    pub fn forward(&self, x: &Tensor, training: bool) -> Tensor {
        if !training || self.p == 0.0 { return x.clone(); }
        let scale = 1.0 / (1.0 - self.p);
        use std::cell::Cell;
        thread_local! { static S: Cell<u64> = Cell::new(0x1234_5678); }
        let data: Vec<f64> = x.data.iter().map(|&v| {
            S.with(|s| {
                let mut x = s.get();
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                s.set(x);
                let u = (x as f64) / (u64::MAX as f64);
                if u < self.p { 0.0 } else { v * scale }
            })
        }).collect();
        Tensor { shape: x.shape.clone(), data }
    }
}

/// 1D convolution: x [batch, in_channels, length], w [out_channels, in_channels, kernel_size].
pub fn conv1d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> MathResult<Tensor> {
    let (batch, in_ch, len) = (x.shape[0], x.shape[1], x.shape[2]);
    let (out_ch, _, kernel_size) = (weight.shape[0], weight.shape[1], weight.shape[2]);
    let out_len = (len + 2 * padding - kernel_size) / stride + 1;
    let mut out_data = vec![0.0; batch * out_ch * out_len];
    for b in 0..batch {
        for oc in 0..out_ch {
            for o in 0..out_len {
                let mut val = 0.0;
                for ic in 0..in_ch {
                    for k in 0..kernel_size {
                        let inp = o * stride + k;
                        if inp >= padding && inp - padding < len {
                            let iv = x.data[b * in_ch * len + ic * len + (inp - padding)];
                            let wv = weight.data[oc * in_ch * kernel_size + ic * kernel_size + k];
                            val += iv * wv;
                        }
                    }
                }
                if let Some(bias) = bias { val += bias.data[oc]; }
                out_data[b * out_ch * out_len + oc * out_len + o] = val;
            }
        }
    }
    Ok(Tensor { shape: vec![batch, out_ch, out_len], data: out_data })
}

/// 2D convolution: x [batch, in_ch, h, w], w [out_ch, in_ch, kh, kw].
pub fn conv2d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: usize, padding: usize) -> MathResult<Tensor> {
    let (batch, in_ch, ih, iw) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let (out_ch, _, kh, kw) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
    let oh = (ih + 2 * padding - kh) / stride + 1;
    let ow = (iw + 2 * padding - kw) / stride + 1;
    let mut out_data = vec![0.0; batch * out_ch * oh * ow];
    for b in 0..batch {
        for oc in 0..out_ch {
            for i in 0..oh {
                for j in 0..ow {
                    let mut val = 0.0;
                    for ic in 0..in_ch {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let inp_i = i * stride + ki;
                                let inp_j = j * stride + kj;
                                if inp_i >= padding && inp_i - padding < ih && inp_j >= padding && inp_j - padding < iw {
                                    let iv = x.data[b * in_ch * ih * iw + ic * ih * iw + (inp_i - padding) * iw + (inp_j - padding)];
                                    let wv = weight.data[oc * in_ch * kh * kw + ic * kh * kw + ki * kw + kj];
                                    val += iv * wv;
                                }
                            }
                        }
                    }
                    if let Some(bias) = bias { val += bias.data[oc]; }
                    out_data[b * out_ch * oh * ow + oc * oh * ow + i * ow + j] = val;
                }
            }
        }
    }
    Ok(Tensor { shape: vec![batch, out_ch, oh, ow], data: out_data })
}

/// Max pool 1D.
pub fn max_pool1d(x: &Tensor, kernel_size: usize, stride: usize) -> MathResult<Tensor> {
    let (batch, ch, len) = (x.shape[0], x.shape[1], x.shape[2]);
    let out_len = (len - kernel_size) / stride + 1;
    let mut out_data = vec![0.0; batch * ch * out_len];
    for b in 0..batch {
        for c in 0..ch {
            for o in 0..out_len {
                let start = o * stride;
                let mut max_val = f64::NEG_INFINITY;
                for k in 0..kernel_size {
                    let v = x.data[b * ch * len + c * len + start + k];
                    if v > max_val { max_val = v; }
                }
                out_data[b * ch * out_len + c * out_len + o] = max_val;
            }
        }
    }
    Ok(Tensor { shape: vec![batch, ch, out_len], data: out_data })
}

/// Average pool 2D.
pub fn avg_pool2d(x: &Tensor, kernel_size: usize) -> MathResult<Tensor> {
    let (batch, ch, ih, iw) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let oh = ih / kernel_size;
    let ow = iw / kernel_size;
    let mut out_data = vec![0.0; batch * ch * oh * ow];
    let area = (kernel_size * kernel_size) as f64;
    for b in 0..batch {
        for c in 0..ch {
            for i in 0..oh {
                for j in 0..ow {
                    let mut sum = 0.0;
                    for ki in 0..kernel_size {
                        for kj in 0..kernel_size {
                            sum += x.data[b * ch * ih * iw + c * ih * iw + (i * kernel_size + ki) * iw + (j * kernel_size + kj)];
                        }
                    }
                    out_data[b * ch * oh * ow + c * oh * ow + i * ow + j] = sum / area;
                }
            }
        }
    }
    Ok(Tensor { shape: vec![batch, ch, oh, ow], data: out_data })
}

/// Global average pool: reduces spatial dims to 1x1.
pub fn global_avg_pool(x: &Tensor) -> MathResult<Tensor> {
    if x.shape.len() != 4 { return Err(mathverse_core::error::MathError::InvalidArgument("global_avg_pool requires 4D input")); }
    let (batch, ch, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let area = (h * w) as f64;
    let mut out_data = vec![0.0; batch * ch];
    for b in 0..batch {
        for c in 0..ch {
            let mut sum = 0.0;
            for i in 0..h {
                for j in 0..w {
                    sum += x.data[b * ch * h * w + c * h * w + i * w + j];
                }
            }
            out_data[b * ch + c] = sum / area;
        }
    }
    Ok(Tensor { shape: vec![batch, ch], data: out_data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_test() {
        let layer = Linear::new(4, 3);
        let x = Tensor::zeros(&[2, 4]);
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(layer.num_params(), 15);
    }

    #[test]
    fn layer_norm_test() {
        let ln = LayerNorm::new(4, 1e-5);
        let x = Tensor::new(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = ln.forward(&x);
        let m: f64 = (0..4).map(|j| out.data[j]).sum::<f64>() / 4.0;
        assert!(m.abs() < 1e-5);
    }

    #[test]
    fn conv1d_test() {
        let x = Tensor::zeros(&[1, 1, 10]);
        let w = Tensor::zeros(&[1, 1, 3]);
        let out = conv1d(&x, &w, None, 1, 0).unwrap();
        assert_eq!(out.shape, vec![1, 1, 8]);
    }

    #[test]
    fn conv2d_test() {
        let x = Tensor::zeros(&[1, 1, 8, 8]);
        let w = Tensor::zeros(&[1, 1, 3, 3]);
        let out = conv2d(&x, &w, None, 1, 0).unwrap();
        assert_eq!(out.shape, vec![1, 1, 6, 6]);
    }

    #[test]
    fn dropout_test() {
        let drop = Dropout::new(0.0);
        let x = Tensor::ones(&[10]);
        let out = drop.forward(&x, false);
        assert_eq!(out, x);
    }

    #[test]
    fn max_pool1d_test() {
        let x = Tensor::new(&[1, 1, 8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = max_pool1d(&x, 2, 2).unwrap();
        assert_eq!(out.shape, vec![1, 1, 4]);
        assert!((out.data[0] - 2.0).abs() < 1e-9);
    }
}
