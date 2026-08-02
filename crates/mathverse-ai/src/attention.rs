//! Attention math: QKV projection, scaled dot-product attention,
//! multi-head attention, sinusoidal positional encoding, rotary embeddings (RoPE).

use crate::tensor::Tensor;
use mathverse_core::error::MathResult;

/// QKV projection: splits X into Q, K, V via weight matrices.
/// `x`: [batch, seq_len, d_model]
/// `wq`: [d_model, d_k], `wk`: [d_model, d_k], `wv`: [d_model, d_v]
/// Returns (Q, K, V) each [batch, seq_len, d_k] or [batch, seq_len, d_v].
pub fn qkv_projection(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
) -> MathResult<(Tensor, Tensor, Tensor)> {
    // x @ W for each: flatten batch, matmul, reshape back
    let batch = x.shape[0];
    let seq_len = x.shape[1];
    let d_model = x.shape[2];
    let d_k = wq.shape[1];
    let d_v = wv.shape[1];

    // Reshape x to [batch*seq_len, d_model]
    let x_flat = Tensor::new(&[batch * seq_len, d_model], &x.data)?;
    let q = x_flat.matmul(wq)?.reshape(&[batch, seq_len, d_k])?;
    let k = x_flat.matmul(wk)?.reshape(&[batch, seq_len, d_k])?;
    let v = x_flat.matmul(wv)?.reshape(&[batch, seq_len, d_v])?;
    Ok((q, k, v))
}

/// Scaled dot-product attention.
/// `q`: [batch, seq_q, d_k], `k`: [batch, seq_kv, d_k], `v`: [batch, seq_kv, d_v]
/// Returns (output [batch, seq_q, d_v], weights [batch, seq_q, seq_kv]).
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Option<f64>,
    mask: Option<&Tensor>,
) -> MathResult<(Tensor, Tensor)> {
    let d_k = q.shape[2] as f64;
    let s = scale.unwrap_or(1.0 / d_k.sqrt());

    // scores = Q @ K^T * scale → [batch, seq_q, seq_kv]
    // K^T: [batch, d_k, seq_kv]
    let batch = q.shape[0];
    let seq_q = q.shape[1];
    let seq_kv = k.shape[1];
    let d_v = v.shape[2];

    let mut scores_data = vec![0.0; batch * seq_q * seq_kv];
    for bi in 0..batch {
        for i in 0..seq_q {
            for j in 0..seq_kv {
                let mut dot = 0.0;
                for d in 0..q.shape[2] {
                    dot += q.data[bi * seq_q * q.shape[2] + i * q.shape[2] + d]
                         * k.data[bi * seq_kv * k.shape[2] + j * k.shape[2] + d];
                }
                scores_data[bi * seq_q * seq_kv + i * seq_kv + j] = dot * s;
            }
        }
    }

    // Apply mask (set masked positions to -inf)
    if let Some(m) = mask {
        #[allow(clippy::needless_range_loop)]
        for idx in 0..scores_data.len() {
            if m.data[idx] < -1e10 {
                scores_data[idx] = f64::NEG_INFINITY;
            }
        }
    }

    // Softmax along last axis
    let scores = Tensor::new(&[batch, seq_q, seq_kv], &scores_data)?;
    let weights = crate::activations::softmax(&scores, 2)?;

    // output = weights @ V → [batch, seq_q, d_v]
    let mut out_data = vec![0.0; batch * seq_q * d_v];
    for bi in 0..batch {
        for i in 0..seq_q {
            for d in 0..d_v {
                let mut val = 0.0;
                for j in 0..seq_kv {
                    val += weights.data[bi * seq_q * seq_kv + i * seq_kv + j]
                         * v.data[bi * seq_kv * d_v + j * d_v + d];
                }
                out_data[bi * seq_q * d_v + i * d_v + d] = val;
            }
        }
    }

    let output = Tensor::new(&[batch, seq_q, d_v], &out_data)?;
    Ok((output, weights))
}

/// Multi-head attention.
/// `x`: [batch, seq_len, d_model]
/// `wq`: [d_model, num_heads*d_k], `wk`: [d_model, num_heads*d_k], `wv`: [d_model, num_heads*d_v]
/// `wo`: [num_heads*d_v, d_model]
pub fn multi_head_attention(
    x: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    num_heads: usize,
    mask: Option<&Tensor>,
) -> MathResult<Tensor> {
    let batch = x.shape[0];
    let seq_len = x.shape[1];
    let d_model = x.shape[2];
    let d_k = wq.shape[1] / num_heads;
    let d_v = wv.shape[1] / num_heads;

    // QKV projection
    let (q, k, v) = qkv_projection(x, wq, wk, wv)?;

    // Reshape to [batch, num_heads, seq_len, d_k/d_v]
    let q = q.reshape(&[batch, seq_len, num_heads, d_k])?.permute(&[0, 2, 1, 3])?;
    let k = k.reshape(&[batch, seq_len, num_heads, d_k])?.permute(&[0, 2, 1, 3])?;
    let v = v.reshape(&[batch, seq_len, num_heads, d_v])?.permute(&[0, 2, 1, 3])?;

    // Compute attention per head
    // q: [batch, num_heads, seq_len, d_k]
    let mut head_outputs = Vec::new();
    let mut all_weights = Vec::new();
#[allow(clippy::needless_range_loop)]    for h in 0..num_heads {
        // Extract head h: [batch, seq_len, d_k]
        let qh = extract_head(&q, h)?;
        let kh = extract_head(&k, h)?;
        let vh = extract_head(&v, h)?;
        let (out, weights) = scaled_dot_product_attention(&qh, &kh, &vh, None, mask)?;
        head_outputs.push(out);
        all_weights.push(weights);
    }

    // Concatenate heads: [batch, seq_len, num_heads*d_v]
    let mut concat_data = vec![0.0; batch * seq_len * num_heads * d_v];
#[allow(clippy::needless_range_loop)]    for h in 0..num_heads {
        for bi in 0..batch {
            for s in 0..seq_len {
                for d in 0..d_v {
                    let src_idx = bi * seq_len * d_v + s * d_v + d;
                    let dst_idx = bi * seq_len * (num_heads * d_v) + s * (num_heads * d_v) + h * d_v + d;
                    concat_data[dst_idx] = head_outputs[h].data[src_idx];
                }
            }
        }
    }
    let concat = Tensor::new(&[batch, seq_len, num_heads * d_v], &concat_data)?;

    // Output projection: [batch, seq_len, num_heads*d_v] @ [num_heads*d_v, d_model]
    let flat = concat.reshape(&[batch * seq_len, num_heads * d_v])?;
    let out = flat.matmul(wo)?.reshape(&[batch, seq_len, d_model])?;
    Ok(out)
}

/// Extract a single head from [batch, num_heads, seq_len, d].
fn extract_head(t: &Tensor, head: usize) -> MathResult<Tensor> {
    let (batch, _nh, seq, d) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
    let mut data = vec![0.0; batch * seq * d];
    for bi in 0..batch {
        for s in 0..seq {
            for di in 0..d {
                let src = bi * t.shape[1] * seq * d + head * seq * d + s * d + di;
                let dst = bi * seq * d + s * d + di;
                data[dst] = t.data[src];
            }
        }
    }
    Ok(Tensor { shape: vec![batch, seq, d], data })
}

/// Sinusoidal positional encoding (additive).
/// Returns tensor of shape [1, seq_len, d_model].
pub fn sinusoidal_encoding(seq_len: usize, d_model: usize, max_len: usize) -> Tensor {
    let mut data = vec![0.0; seq_len * d_model];
    for pos in 0..seq_len {
        for i in 0..d_model / 2 {
            let angle = pos as f64 / (max_len as f64).powf(2.0 * i as f64 / d_model as f64);
            data[pos * d_model + 2 * i] = angle.sin();
            data[pos * d_model + 2 * i + 1] = angle.cos();
        }
    }
    Tensor { shape: vec![1, seq_len, d_model], data }
}

/// Compute RoPE frequency tensor: [seq_len, d_head].
pub fn rope_freqs(d_head: usize, seq_len: usize, base: f64) -> Tensor {
    let mut data = vec![0.0; seq_len * d_head];
    for pos in 0..seq_len {
        for i in 0..d_head / 2 {
            let freq = 1.0 / base.powf(2.0 * i as f64 / d_head as f64);
            data[pos * d_head + 2 * i] = pos as f64 * freq;
            data[pos * d_head + 2 * i + 1] = pos as f64 * freq;
        }
    }
    Tensor { shape: vec![seq_len, d_head], data }
}

/// Apply rotary positional embeddings to a tensor.
/// `x`: [batch, seq_len, d_head] — applies cos/sin rotation to each pair.
pub fn apply_rope(x: &Tensor, seq_len: usize, base: f64) -> Tensor {
    let d_head = x.shape[x.shape.len() - 1];
    let freqs = rope_freqs(d_head, seq_len, base);
    let mut out = x.data.clone();
    let nd = x.shape.len();
    let outer: usize = x.shape[..nd - 2].iter().product();
    let total_pairs = outer * seq_len * (d_head / 2);

    // Iterate over all (position, head_dim_pair) combos
    for flat in 0..total_pairs {
        // Decompose flat index into (outer_idx, pos, pair_idx)
        let pair_idx = flat % (d_head / 2);
        let remaining = flat / (d_head / 2);
        let pos = remaining % seq_len;
        let outer_idx = remaining / seq_len;

        let theta = freqs.data[pos * d_head + 2 * pair_idx];
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        let base_offset = outer_idx * seq_len * d_head + pos * d_head;
        let i0 = base_offset + 2 * pair_idx;
        let i1 = base_offset + 2 * pair_idx + 1;

        let x0 = x.data[i0];
        let x1 = x.data[i1];
        out[i0] = x0 * cos_t - x1 * sin_t;
        out[i1] = x0 * sin_t + x1 * cos_t;
    }

    Tensor { shape: x.shape.clone(), data: out }
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-5;

    #[test]
    fn scaled_dot_product_attention_test() {
        // Simple case: batch=1, seq_q=2, seq_kv=2, d_k=3
        let q = Tensor::new(&[1, 2, 3], &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]).unwrap();
        let k = Tensor::new(&[1, 2, 3], &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]).unwrap();
        let v = Tensor::new(&[1, 2, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();
        let (out, weights) = scaled_dot_product_attention(&q, &k, &v, None, None).unwrap();
        assert_eq!(out.shape, vec![1, 2, 3]);
        assert_eq!(weights.shape, vec![1, 2, 2]);
        // Weights should sum to 1 along last dim
        let w_sum: f64 = weights.data[0..2].iter().sum();
        assert!((w_sum - 1.0).abs() < E);
    }

    #[test]
    fn attention_mask_test() {
        // q = k = identity, v = [1,2; 3,4], mask blocks pos 0 from attending to pos 1
        let q = Tensor::new(&[1, 2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = Tensor::new(&[1, 2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let v = Tensor::new(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        // mask: pos 0 can only attend to pos 0 (pos 1 masked), pos 1 can attend to both
        let mask = Tensor::new(&[1, 2, 2], &[
            0.0, f64::NEG_INFINITY,
            0.0, 0.0,
        ]).unwrap();
        let (out, _) = scaled_dot_product_attention(&q, &k, &v, None, Some(&mask)).unwrap();
        // Position 0 attends only to v[0]=[1,2] → output[0] = [1,2]
        assert!((out.data[0] - 1.0).abs() < E);
        assert!((out.data[1] - 2.0).abs() < E);
    }

    #[test]
    fn sinusoidal_encoding_test() {
        let enc = sinusoidal_encoding(10, 8, 10000);
        assert_eq!(enc.shape, vec![1, 10, 8]);
        // Position 0: sin(0)=0, cos(0)=1
        assert!(enc.data[0].abs() < E);
        assert!((enc.data[1] - 1.0).abs() < E);
    }

    #[test]
    fn rope_test() {
        let freqs = rope_freqs(4, 5, 10000.0);
        assert_eq!(freqs.shape, vec![5, 4]);
        // Position 0: all zeros (0 * freq = 0)
        for i in 0..4 {
            assert!(freqs.data[i].abs() < E);
        }
    }

    #[test]
    fn apply_rope_preserves_norm() {
        let x = Tensor::new(&[1, 4, 4], &(0..16).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
        let norm_before: f64 = x.data.iter().map(|v| v * v).sum();
        let y = apply_rope(&x, 4, 10000.0);
        let norm_after: f64 = y.data.iter().map(|v| v * v).sum();
        assert!((norm_before - norm_after).abs() < E);
    }

    #[test]
    fn multi_head_attention_shapes() {
        let batch = 2;
        let seq = 4;
        let d_model = 8;
        let num_heads = 2;
        let d_k = 4;
        let x = Tensor::randn(&[batch, seq, d_model]);
        let wq = Tensor::randn(&[d_model, num_heads * d_k]);
        let wk = Tensor::randn(&[d_model, num_heads * d_k]);
        let wv = Tensor::randn(&[d_model, num_heads * d_k]);
        let wo = Tensor::randn(&[num_heads * d_k, d_model]);
        let out = multi_head_attention(&x, &wq, &wk, &wv, &wo, num_heads, None).unwrap();
        assert_eq!(out.shape, vec![batch, seq, d_model]);
    }
}










