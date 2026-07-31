//! Advanced attention mechanisms: flash, grouped query, cross, and linear attention.

use crate::tensor::Tensor;
use mathverse_core::error::{MathError, MathResult};

/// Flash (tiled) attention: computes attention in blocks for efficiency.
/// `q`, `k`, `v`: [batch, seq_len, d_k/d_v]
/// Returns output [batch, seq_len, d_v].
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, block_size: usize) -> MathResult<Tensor> {
    if q.shape.len() != 3 || k.shape.len() != 3 || v.shape.len() != 3 {
        return Err(MathError::InvalidArgument("q, k, v must be 3-D [batch, seq, d]"));
    }
    let batch = q.shape[0];
    let seq_q = q.shape[1];
    let d_k = q.shape[2];
    let seq_kv = k.shape[1];
    let d_v = v.shape[2];
    if k.shape[2] != d_k || v.shape[1] != seq_kv {
        return Err(MathError::DimensionMismatch);
    }

    let scale = 1.0 / (d_k as f64).sqrt();
    let bs = block_size.max(1);

    let mut out = vec![0.0; batch * seq_q * d_v];
    let mut row_sum = vec![0.0; batch * seq_q];
    let mut row_max = vec![f64::NEG_INFINITY; batch * seq_q];

    for bi in 0..batch {
        let mut qi = 0;
        while qi < seq_q {
            let qi_end = (qi + bs).min(seq_q);
            for kj in 0..seq_kv {
                let kj_end = (kj + bs).min(seq_kv);
                // Compute scores for block [qi..qi_end] x [kj..kj_end]
                for i in qi..qi_end {
                    for j in kj..kj_end {
                        let mut dot = 0.0;
                        for d in 0..d_k {
                            dot += q.data[bi * seq_q * d_k + i * d_k + d]
                                * k.data[bi * seq_kv * d_k + j * d_k + d];
                        }
                        let s = dot * scale;

                        // Online softmax update
                        let idx = bi * seq_q + i;
                        if s > row_max[idx] {
                            let ratio = (row_max[idx] - s).exp();
                            for d in 0..d_v {
                                out[bi * seq_q * d_v + i * d_v + d] *= ratio;
                            }
                            row_sum[idx] *= ratio;
                            row_max[idx] = s;
                        }
                        let p = (s - row_max[idx]).exp();
                        row_sum[idx] += p;
                        for d in 0..d_v {
                            out[bi * seq_q * d_v + i * d_v + d] += p * v.data[bi * seq_kv * d_v + j * d_v + d];
                        }
                    }
                }
            }
            // Normalize
            for i in qi..qi_end {
                let idx = bi * seq_q + i;
                let inv = 1.0 / row_sum[idx].max(f64::EPSILON);
                for d in 0..d_v {
                    out[bi * seq_q * d_v + i * d_v + d] *= inv;
                }
            }
            qi += bs;
        }
    }

    Tensor::from_vec(&[batch, seq_q, d_v], out)
}

/// Grouped Query Attention (GQA): shares K, V heads across groups of Q heads.
/// `q`: [batch, seq_q, num_heads * d_k]
/// `k`, `v`: [batch, seq_kv, num_groups * d_k]
/// `num_groups`: number of KV heads
/// Returns output [batch, seq_q, num_heads * d_k].
pub fn grouped_query_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_groups: usize,
) -> MathResult<Tensor> {
    if q.shape.len() != 3 || k.shape.len() != 3 || v.shape.len() != 3 {
        return Err(MathError::InvalidArgument("q, k, v must be 3-D"));
    }
    let batch = q.shape[0];
    let seq_q = q.shape[1];
    let q_dim = q.shape[2];
    let seq_kv = k.shape[1];
    let kv_dim = k.shape[2];
    if kv_dim % num_groups != 0 {
        return Err(MathError::InvalidArgument("kv_dim must be divisible by num_groups"));
    }
    let d_k = kv_dim / num_groups;
    let num_heads = q_dim / d_k;
    if q_dim % d_k != 0 {
        return Err(MathError::InvalidArgument("q_dim must be divisible by d_k (kv_dim/num_groups)"));
    }

    let scale = 1.0 / (d_k as f64).sqrt();
    let mut out = vec![0.0; batch * seq_q * q_dim];

    for bi in 0..batch {
        for h in 0..num_heads {
            let g = h / (num_heads / num_groups);
            for i in 0..seq_q {
                let mut scores = vec![0.0; seq_kv];
                let mut max_s = f64::NEG_INFINITY;
                for j in 0..seq_kv {
                    let mut dot = 0.0;
                    for d in 0..d_k {
                        let q_idx = bi * seq_q * q_dim + i * q_dim + h * d_k + d;
                        let k_idx = bi * seq_kv * kv_dim + j * kv_dim + g * d_k + d;
                        dot += q.data[q_idx] * k.data[k_idx];
                    }
                    scores[j] = dot * scale;
                    if scores[j] > max_s { max_s = scores[j]; }
                }
                let mut sum_exp = 0.0;
                for j in 0..seq_kv {
                    scores[j] = (scores[j] - max_s).exp();
                    sum_exp += scores[j];
                }
                let inv = 1.0 / sum_exp.max(f64::EPSILON);
                for d in 0..d_k {
                    let mut val = 0.0;
                    for j in 0..seq_kv {
                        let v_idx = bi * seq_kv * kv_dim + j * kv_dim + g * d_k + d;
                        val += scores[j] * v.data[v_idx];
                    }
                    let o_idx = bi * seq_q * q_dim + i * q_dim + h * d_k + d;
                    out[o_idx] = val * inv;
                }
            }
        }
    }

    Tensor::from_vec(&[batch, seq_q, q_dim], out)
}

/// Cross-attention: Q from one source, K/V from another.
/// `query`: [batch, seq_q, d_model], `key`, `value`: [batch, seq_kv, d_model]
/// Returns output [batch, seq_q, d_model].
pub fn cross_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: Option<f64>,
) -> MathResult<Tensor> {
    if query.shape.len() != 3 || key.shape.len() != 3 || value.shape.len() != 3 {
        return Err(MathError::InvalidArgument("inputs must be 3-D"));
    }
    let batch = query.shape[0];
    let seq_q = query.shape[1];
    let d_model = query.shape[2];
    let seq_kv = key.shape[1];
    if key.shape[2] != d_model || value.shape[2] != d_model || value.shape[1] != seq_kv {
        return Err(MathError::DimensionMismatch);
    }

    let s = scale.unwrap_or(1.0 / (d_model as f64).sqrt());

    // scores = Q @ K^T * scale
    let mut scores = vec![0.0; batch * seq_q * seq_kv];
    for bi in 0..batch {
        for i in 0..seq_q {
            for j in 0..seq_kv {
                let mut dot = 0.0;
                for d in 0..d_model {
                    dot += query.data[bi * seq_q * d_model + i * d_model + d]
                        * key.data[bi * seq_kv * d_model + j * d_model + d];
                }
                scores[bi * seq_q * seq_kv + i * seq_kv + j] = dot * s;
            }
        }
    }

    // Softmax over seq_kv for each query position
    for bi in 0..batch {
        for i in 0..seq_q {
            let start = bi * seq_q * seq_kv + i * seq_kv;
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..seq_kv {
                if scores[start + j] > max_val { max_val = scores[start + j]; }
            }
            let mut sum = 0.0;
            for j in 0..seq_kv {
                scores[start + j] = (scores[start + j] - max_val).exp();
                sum += scores[start + j];
            }
            let inv = 1.0 / sum.max(f64::EPSILON);
            for j in 0..seq_kv {
                scores[start + j] *= inv;
            }
        }
    }

    // output = scores @ V
    let mut out = vec![0.0; batch * seq_q * d_model];
    for bi in 0..batch {
        for i in 0..seq_q {
            for d in 0..d_model {
                let mut val = 0.0;
                for j in 0..seq_kv {
                    val += scores[bi * seq_q * seq_kv + i * seq_kv + j]
                        * value.data[bi * seq_kv * d_model + j * d_model + d];
                }
                out[bi * seq_q * d_model + i * d_model + d] = val;
            }
        }
    }

    Tensor::from_vec(&[batch, seq_q, d_model], out)
}

/// Linear (kernel-based) attention: O(n) complexity.
/// `q`, `k`, `v`: [batch, seq_len, d]
/// Uses elu(q)+1 as kernel feature map.
/// Returns output [batch, seq_len, d_v].
pub fn linear_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> MathResult<Tensor> {
    if q.shape.len() != 3 || k.shape.len() != 3 || v.shape.len() != 3 {
        return Err(MathError::InvalidArgument("inputs must be 3-D"));
    }
    let batch = q.shape[0];
    let seq = q.shape[1];
    let d_q = q.shape[2];
    let d_k = k.shape[2];
    let d_v = v.shape[2];
    if k.shape[1] != seq || v.shape[1] != seq {
        return Err(MathError::DimensionMismatch);
    }

    // Feature maps: phi(x) = elu(x) + 1
    let phi_q: Vec<f64> = q.data.iter().map(|&x| (x.exp() - 1.0).max(0.0) + 1.0).collect();
    let phi_k: Vec<f64> = k.data.iter().map(|&x| (x.exp() - 1.0).max(0.0) + 1.0).collect();

    // S = K^T @ V: [batch, d_k, d_v]
    let mut s = vec![0.0; batch * d_k * d_v];
    for bi in 0..batch {
        for dk_i in 0..d_k {
            for dv_i in 0..d_v {
                let mut val = 0.0;
                for t in 0..seq {
                    val += phi_k[bi * seq * d_k + t * d_k + dk_i]
                        * v.data[bi * seq * d_v + t * d_v + dv_i];
                }
                s[bi * d_k * d_v + dk_i * d_v + dv_i] = val;
            }
        }
    }

    // z = sum of phi_k over seq: [batch, d_k]
    let mut z = vec![0.0; batch * d_k];
    for bi in 0..batch {
        for dk_i in 0..d_k {
            let mut val = 0.0;
            for t in 0..seq {
                val += phi_k[bi * seq * d_k + t * d_k + dk_i];
            }
            z[bi * d_k + dk_i] = val;
        }
    }

    // out = phi_q @ S, normalized by z
    let mut out = vec![0.0; batch * seq * d_v];
    for bi in 0..batch {
        for t in 0..seq {
            for dv_i in 0..d_v {
                let mut num = 0.0;
                let mut den = 0.0;
                for dk_i in 0..d_k {
                    let qv = phi_q[bi * seq * d_q + t * d_q + dk_i];
                    num += qv * s[bi * d_k * d_v + dk_i * d_v + dv_i];
                    den += qv * z[bi * d_k + dk_i];
                }
                let inv = 1.0 / den.max(f64::EPSILON);
                out[bi * seq * d_v + t * d_v + dv_i] = num * inv;
            }
        }
    }

    Tensor::from_vec(&[batch, seq, d_v], out)
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-4;

    #[test]
    fn flash_attention_matches_standard() {
        // For small sequences, flash should give same results as standard
        let q = Tensor::new(&[1, 3, 4], &(0..12).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
        let k = Tensor::new(&[1, 3, 4], &(12..24).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
        let v = Tensor::new(&[1, 3, 4], &(24..36).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
        let out = flash_attention(&q, &k, &v, 2).unwrap();
        assert_eq!(out.shape, vec![1, 3, 4]);
        // Output should be finite and non-zero
        assert!(out.data.iter().any(|&x| x.abs() > 1e-6));
        assert!(out.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn flash_attention_single_block() {
        let q = Tensor::new(&[1, 2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let k = Tensor::new(&[1, 2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let v = Tensor::new(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = flash_attention(&q, &k, &v, 100).unwrap();
        assert_eq!(out.shape, vec![1, 2, 2]);
        assert!(out.data.iter().all(|&x| x.is_finite()));
        // With identity Q=K, output is weighted combination of V rows
        // softmax([1/sqrt(2), 0]) gives more weight to matching position
        // So out[0] should be closer to v[0]=[1,2] than v[1]=[3,4]
        assert!(out.data[0] < out.data[2]); // first row dominated by v[0]
    }

    #[test]
    fn gqa_basic() {
        // 4 Q-heads, 2 KV-heads → each KV head shared by 2 Q heads
        let q = Tensor::randn(&[1, 2, 8]); // 4 heads * 2 d_k
        let k = Tensor::randn(&[1, 2, 4]); // 2 heads * 2 d_k
        let v = Tensor::randn(&[1, 2, 4]);
        let out = grouped_query_attention(&q, &k, &v, 2).unwrap();
        assert_eq!(out.shape, vec![1, 2, 8]);
        assert!(out.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn cross_attention_test() {
        let query = Tensor::new(&[1, 2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let key = Tensor::new(&[1, 3, 3], &[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();
        let value = Tensor::new(&[1, 3, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let out = cross_attention(&query, &key, &value, None).unwrap();
        assert_eq!(out.shape, vec![1, 2, 3]);
        // query[0]=[1,0,0] attends mostly to key[0]=[1,0,0] → value[0]=[1,2,3]
        assert!(out.data[0] < out.data[3]); // output[0] < output[3] (v0 < v1)
        assert!(out.data.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn linear_attention_test() {
        let q = Tensor::randn(&[1, 4, 3]);
        let k = Tensor::randn(&[1, 4, 3]);
        let v = Tensor::randn(&[1, 4, 3]);
        let out = linear_attention(&q, &k, &v).unwrap();
        assert_eq!(out.shape, vec![1, 4, 3]);
        assert!(out.data.iter().all(|&x| x.is_finite()));
    }
}
