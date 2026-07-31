//! Transposed convolution (deconvolution) for 1D and 2D tensors.

use crate::tensor::Tensor;
use mathverse_core::error::{MathError, MathResult};

/// 1-D transposed convolution.
/// `x`: [batch, in_channels, length]
/// `weight`: [in_channels, out_channels, kernel_size]
/// `bias`: [out_channels] optional
pub fn conv_transpose1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> MathResult<Tensor> {
    if x.shape.len() != 3 {
        return Err(MathError::InvalidArgument("x must be 3-D [batch, in_ch, length]"));
    }
    if weight.shape.len() != 3 {
        return Err(MathError::InvalidArgument("weight must be 3-D [in_ch, out_ch, kernel]"));
    }
    let batch = x.shape[0];
    let in_ch = x.shape[1];
    let length = x.shape[2];
    let out_ch = weight.shape[1];
    let kernel = weight.shape[2];
    if weight.shape[0] != in_ch {
        return Err(MathError::DimensionMismatch);
    }

    let out_len_i = (length as i64 - 1) * stride as i64 - 2 * padding as i64 + kernel as i64;
    if out_len_i <= 0 {
        return Err(MathError::InvalidArgument("output length must be positive"));
    }
    let out_len = out_len_i as usize;
    let mut out = vec![0.0; batch * out_ch * out_len];

    for b in 0..batch {
        for ic in 0..in_ch {
            for oc in 0..out_ch {
                for pos in 0..length {
                    let input_val = x.data[b * in_ch * length + ic * length + pos];
                    for k in 0..kernel {
                        let out_pos = pos * stride + k;
                        if out_pos >= padding && out_pos - padding < out_len {
                            let idx = out_pos - padding;
                            let w_val = weight.data[ic * out_ch * kernel + oc * kernel + k];
                            out[b * out_ch * out_len + oc * out_len + idx] += input_val * w_val;
                        }
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        for b in 0..batch {
            for oc in 0..out_ch {
                for idx in 0..out_len {
                    out[b * out_ch * out_len + oc * out_len + idx] += bias.data[oc];
                }
            }
        }
    }

    Tensor::from_vec(&[batch, out_ch, out_len], out)
}

/// 2-D transposed convolution.
/// `x`: [batch, in_channels, height, width]
/// `weight`: [in_channels, out_channels, kH, kW]
/// `bias`: [out_channels] optional
pub fn conv_transpose2d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> MathResult<Tensor> {
    if x.shape.len() != 4 {
        return Err(MathError::InvalidArgument("x must be 4-D [batch, in_ch, H, W]"));
    }
    if weight.shape.len() != 4 {
        return Err(MathError::InvalidArgument("weight must be 4-D [in_ch, out_ch, kH, kW]"));
    }
    let batch = x.shape[0];
    let in_ch = x.shape[1];
    let h_in = x.shape[2];
    let w_in = x.shape[3];
    let out_ch = weight.shape[1];
    let kh = weight.shape[2];
    let kw = weight.shape[3];
    if weight.shape[0] != in_ch {
        return Err(MathError::DimensionMismatch);
    }

    let h_out_i = (h_in as i64 - 1) * stride as i64 - 2 * padding as i64 + kh as i64;
    let w_out_i = (w_in as i64 - 1) * stride as i64 - 2 * padding as i64 + kw as i64;
    if h_out_i <= 0 || w_out_i <= 0 {
        return Err(MathError::InvalidArgument("output dimensions must be positive"));
    }
    let h_out = h_out_i as usize;
    let w_out = w_out_i as usize;
    let mut out = vec![0.0; batch * out_ch * h_out * w_out];

    for b in 0..batch {
        for ic in 0..in_ch {
            for oc in 0..out_ch {
                for hi in 0..h_in {
                    for wi in 0..w_in {
                        let input_val = x.data[b * in_ch * h_in * w_in + ic * h_in * w_in + hi * w_in + wi];
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let oh = hi * stride + kh_i;
                                let ow = wi * stride + kw_i;
                                if oh >= padding && oh - padding < h_out && ow >= padding && ow - padding < w_out {
                                    let oh_idx = oh - padding;
                                    let ow_idx = ow - padding;
                                    let w_val = weight.data[ic * out_ch * kh * kw + oc * kh * kw + kh_i * kw + kw_i];
                                    out[b * out_ch * h_out * w_out + oc * h_out * w_out + oh_idx * w_out + ow_idx] += input_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        for b in 0..batch {
            for oc in 0..out_ch {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        out[b * out_ch * h_out * w_out + oc * h_out * w_out + oh * w_out + ow] += bias.data[oc];
                    }
                }
            }
        }
    }

    Tensor::from_vec(&[batch, out_ch, h_out, w_out], out)
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-9;

    #[test]
    fn conv_transpose1d_identity() {
        // stride=1, padding=0, kernel=1, weight=1 → identity-like
        let x = Tensor::new(&[1, 1, 3], &[1.0, 2.0, 3.0]).unwrap();
        let w = Tensor::new(&[1, 1, 1], &[1.0]).unwrap();
        let y = conv_transpose1d(&x, &w, None, 1, 0).unwrap();
        assert_eq!(y.shape, vec![1, 1, 3]);
        for i in 0..3 {
            assert!((y.data[i] - x.data[i]).abs() < E);
        }
    }

    #[test]
    fn conv_transpose1d_with_bias() {
        let x = Tensor::new(&[1, 1, 2], &[1.0, 1.0]).unwrap();
        let w = Tensor::new(&[1, 1, 3], &[1.0, 1.0, 1.0]).unwrap();
        let b = Tensor::new(&[1], &[0.5]).unwrap();
        let y = conv_transpose1d(&x, &w, Some(&b), 1, 1).unwrap();
        assert_eq!(y.shape, vec![1, 1, 2]);
        // pos 0 gets contributions from x[0]*w[1] + x[1]*w[0] = 1+1 = 2, plus bias 0.5 = 2.5
        assert!((y.data[0] - 2.5).abs() < E);
    }

    #[test]
    fn conv_transpose2d_basic() {
        let x = Tensor::new(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let w = Tensor::new(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let y = conv_transpose2d(&x, &w, None, 1, 0).unwrap();
        assert_eq!(y.shape, vec![1, 1, 3, 3]);
        // Top-left: x[0,0]*w[0,0] = 1
        assert!((y.data[0] - 1.0).abs() < E);
    }

    #[test]
    fn conv_transpose2d_batch() {
        let x = Tensor::new(&[2, 1, 1, 1], &[1.0, 2.0]).unwrap();
        let w = Tensor::new(&[1, 1, 2, 2], &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let y = conv_transpose2d(&x, &w, None, 1, 0).unwrap();
        assert_eq!(y.shape, vec![2, 1, 2, 2]);
        // batch 0: 1.0 scattered into 2x2
        assert!((y.data[0] - 1.0).abs() < E);
        // batch 1: 2.0 scattered
        assert!((y.data[4] - 2.0).abs() < E);
    }

    #[test]
    fn conv_transpose2d_with_bias() {
        let x = Tensor::new(&[1, 1, 1, 1], &[1.0]).unwrap();
        let w = Tensor::new(&[1, 2, 2, 2], &[1.0; 8]).unwrap();
        let b = Tensor::new(&[2], &[0.1, 0.2]).unwrap();
        let y = conv_transpose2d(&x, &w, Some(&b), 1, 0).unwrap();
        assert_eq!(y.shape, vec![1, 2, 2, 2]);
        // bias added to every position of each output channel
        assert!((y.data[0] - 1.1).abs() < E);
        assert!((y.data[4] - 1.2).abs() < E);
    }
}
