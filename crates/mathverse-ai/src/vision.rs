//! Vision Transformer (ViT) components: patch embedding, CLS token, positional embeddings.

use crate::tensor::Tensor;
use mathverse_core::error::{MathError, MathResult};

/// Split image into patches and project to d_model dimensions.
/// `x`: [batch, channels, height, width]
/// `patch_size`: spatial size of each patch
/// `d_model`: embedding dimension
/// Returns [batch, num_patches, d_model].
pub fn patch_embedding(x: &Tensor, patch_size: usize, d_model: usize) -> MathResult<Tensor> {
    if x.shape.len() != 4 {
        return Err(MathError::InvalidArgument("x must be [batch, channels, H, W]"));
    }
    let batch = x.shape[0];
    let channels = x.shape[1];
    let h = x.shape[2];
    let w = x.shape[3];
    if !h.is_multiple_of(patch_size) || !w.is_multiple_of(patch_size) {
        return Err(MathError::InvalidArgument("H and W must be divisible by patch_size"));
    }
    let patches_h = h / patch_size;
    let patches_w = w / patch_size;
    let num_patches = patches_h * patches_w;
    let patch_dim = channels * patch_size * patch_size;

    // Extract patches: [batch, num_patches, patch_dim]
    let mut patches = vec![0.0; batch * num_patches * patch_dim];
    for b in 0..batch {
        for ph in 0..patches_h {
            for pw in 0..patches_w {
                let patch_idx = ph * patches_w + pw;
                let mut dim = 0;
                for c in 0..channels {
                    for pi in 0..patch_size {
                        for pj in 0..patch_size {
                            let ih = ph * patch_size + pi;
                            let iw = pw * patch_size + pj;
                            patches[b * num_patches * patch_dim + patch_idx * patch_dim + dim] =
                                x.data[b * channels * h * w + c * h * w + ih * w + iw];
                            dim += 1;
                        }
                    }
                }
            }
        }
    }

    // Random projection weights (deterministic seeded)
    let proj = Tensor::randn(&[patch_dim, d_model]);

    // Project: [batch * num_patches, patch_dim] @ [patch_dim, d_model]
    let patches_t = Tensor::from_vec(&[batch * num_patches, patch_dim], patches)?;
    let projected = patches_t.matmul(&proj)?;
    projected.reshape(&[batch, num_patches, d_model])
}

/// Prepend CLS token to sequence.
/// `x`: [batch, seq_len, d_model], `cls_token`: [1, 1, d_model]
/// Returns [batch, 1 + seq_len, d_model].
pub fn cls_token_prepend(x: &Tensor, cls_token: &Tensor) -> MathResult<Tensor> {
    if x.shape.len() != 3 {
        return Err(MathError::InvalidArgument("x must be [batch, seq_len, d_model]"));
    }
    let batch = x.shape[0];
    let seq_len = x.shape[1];
    let d_model = x.shape[2];

    // Expand cls_token to match batch
    let cls_expanded = cls_token.broadcast_to(&[batch, 1, d_model])?;

    // Concat along axis 1
    let mut out = vec![0.0; batch * (1 + seq_len) * d_model];
    for b in 0..batch {
        // CLS token
        for d in 0..d_model {
            out[b * (1 + seq_len) * d_model + d] = cls_expanded.data[b * d_model + d];
        }
        // Original sequence
        for s in 0..seq_len {
            for d in 0..d_model {
                out[b * (1 + seq_len) * d_model + (1 + s) * d_model + d] =
                    x.data[b * seq_len * d_model + s * d_model + d];
            }
        }
    }

    Tensor::from_vec(&[batch, 1 + seq_len, d_model], out)
}

/// Learnable position embeddings (random init, meant to be trained).
/// Returns [1, seq_len, d_model].
pub fn position_embedding(seq_len: usize, d_model: usize) -> Tensor {
    Tensor::randn(&[1, seq_len, d_model])
}

/// ViT forward pass through multiple transformer layers.
/// `patches`: [batch, seq_len, d_model] (already embedded, CLS prepended)
/// `w_q`, `w_k`, `w_v`: [d_model, d_model], `w_o`: [d_model, d_model]
/// Returns output [batch, seq_len, d_model].
pub fn vit_forward(
    patches: &Tensor,
    w_q: &Tensor,
    w_k: &Tensor,
    w_v: &Tensor,
    w_o: &Tensor,
    num_heads: usize,
    num_layers: usize,
) -> MathResult<Tensor> {
    if patches.shape.len() != 3 {
        return Err(MathError::InvalidArgument("patches must be [batch, seq, d_model]"));
    }
    let batch = patches.shape[0];
    let seq_len = patches.shape[1];
    let d_model = patches.shape[2];
    if w_q.shape != [d_model, d_model] || w_k.shape != [d_model, d_model]
        || w_v.shape != [d_model, d_model] || w_o.shape != [d_model, d_model]
    {
        return Err(MathError::DimensionMismatch);
    }

    let d_k = d_model / num_heads;
    let mut x = patches.clone();

    for _layer in 0..num_layers {
        // Layer norm
        let x_normed = x.layer_norm(1e-5);

        // Q, K, V projections
        let flat = x_normed.reshape(&[batch * seq_len, d_model])?;
        let q = flat.matmul(w_q)?.reshape(&[batch, seq_len, d_model])?;
        let k = flat.matmul(w_k)?.reshape(&[batch, seq_len, d_model])?;
        let v = flat.matmul(w_v)?.reshape(&[batch, seq_len, d_model])?;

        // Multi-head attention
        let mut attn_out = vec![0.0; batch * seq_len * d_model];
        for bi in 0..batch {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    // Compute scores for this head
                    let mut scores = vec![0.0; seq_len];
                    let mut max_s = f64::NEG_INFINITY;
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..seq_len {
                        let mut dot = 0.0;
                        for d in 0..d_k {
                            dot += q.data[bi * seq_len * d_model + i * d_model + h * d_k + d]
                                * k.data[bi * seq_len * d_model + j * d_model + h * d_k + d];
                        }
                        scores[j] = dot / (d_k as f64).sqrt();
                        if scores[j] > max_s { max_s = scores[j]; }
                    }
                    let mut sum_exp = 0.0;
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..seq_len {
                        scores[j] = (scores[j] - max_s).exp();
                        sum_exp += scores[j];
                    }
                    let inv = 1.0 / sum_exp.max(f64::EPSILON);
                    for d in 0..d_k {
                        let mut val = 0.0;
                        #[allow(clippy::needless_range_loop)]
                    for j in 0..seq_len {
                            val += scores[j] * v.data[bi * seq_len * d_model + j * d_model + h * d_k + d];
                        }
                        attn_out[bi * seq_len * d_model + i * d_model + h * d_k + d] = val * inv;
                    }
                }
            }
        }

        // Output projection
        let attn = Tensor::from_vec(&[batch * seq_len, d_model], attn_out)?;
        let projected = attn.matmul(w_o)?.reshape(&[batch, seq_len, d_model])?;

        // Residual connection
        x = x.add(&projected)?;
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-4;

    #[test]
    fn patch_embedding_test() {
        // 4x4 image, patch_size=2 → 4 patches of 2x2
        let x = Tensor::new(&[1, 1, 4, 4], &(1..=16).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
        let out = patch_embedding(&x, 2, 8).unwrap();
        assert_eq!(out.shape, vec![1, 4, 8]);
        // Each patch is projected, should be non-zero
        assert!(out.data.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn patch_embedding_batch() {
        let x = Tensor::new(&[2, 3, 6, 6], &(0..216).map(|i| i as f64).collect::<Vec<_>>()).unwrap();
        let out = patch_embedding(&x, 3, 4).unwrap();
        assert_eq!(out.shape, vec![2, 4, 4]);
    }

    #[test]
    fn cls_token_prepend_test() {
        let x = Tensor::new(&[1, 3, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let cls = Tensor::new(&[1, 1, 4], &[0.1, 0.2, 0.3, 0.4]).unwrap();
        let out = cls_token_prepend(&x, &cls).unwrap();
        assert_eq!(out.shape, vec![1, 4, 4]);
        // First token should be the CLS token
        assert!((out.data[0] - 0.1).abs() < E);
        assert!((out.data[1] - 0.2).abs() < E);
        // Second token should be x[0]
        assert!((out.data[4] - 1.0).abs() < E);
    }

    #[test]
    fn position_embedding_test() {
        let pe = position_embedding(10, 16);
        assert_eq!(pe.shape, vec![1, 10, 16]);
        assert_eq!(pe.numel(), 160);
    }

    #[test]
    fn vit_forward_test() {
        let batch = 1;
        let seq = 5; // CLS + 4 patches
        let d_model = 8;
        let num_heads = 2;
        let x = Tensor::randn(&[batch, seq, d_model]);
        let w_q = Tensor::randn(&[d_model, d_model]);
        let w_k = Tensor::randn(&[d_model, d_model]);
        let w_v = Tensor::randn(&[d_model, d_model]);
        let w_o = Tensor::randn(&[d_model, d_model]);
        let out = vit_forward(&x, &w_q, &w_k, &w_v, &w_o, num_heads, 2).unwrap();
        assert_eq!(out.shape, vec![batch, seq, d_model]);
        assert!(out.data.iter().all(|&v| v.is_finite()));
    }
}












