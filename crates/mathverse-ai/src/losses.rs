//! Loss functions: regression (MSE, MAE, Huber) and classification (cross-entropy, etc.).

use crate::tensor::Tensor;
use mathverse_core::error::{MathError, MathResult};

// ---------------------------------------------------------------------------
// Regression losses
// ---------------------------------------------------------------------------

/// Mean squared error: mean((pred - target)²).
pub fn mse(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    let n = pred.numel() as f64;
    let result = pred.data.iter().zip(&target.data)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / n;
    if result.is_nan() || result.is_infinite() {
        return Err(MathError::NumericalFailure("NaN/Inf detected in MSE computation".into()));
    }
    Ok(result)
}

/// MSE gradient w.r.t. pred: 2 * (pred - target) / n.
pub fn mse_grad(pred: &Tensor, target: &Tensor) -> MathResult<Tensor> {
    let n = pred.numel() as f64;
    let data: Vec<f64> = pred.data.iter().zip(&target.data)
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect();
    Ok(Tensor { shape: pred.shape.clone(), data })
}

/// Mean absolute error: mean(|pred - target|).
pub fn mae(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    let n = pred.numel() as f64;
    Ok(pred.data.iter().zip(&target.data)
        .map(|(p, t)| (p - t).abs())
        .sum::<f64>() / n)
}

/// Huber loss (smooth L1): 0.5 * (p-t)² if |p-t| <= delta, else delta * (|p-t| - 0.5*delta).
pub fn huber(pred: &Tensor, target: &Tensor, delta: f64) -> MathResult<f64> {
    let n = pred.numel() as f64;
    let sum: f64 = pred.data.iter().zip(&target.data).map(|(p, t)| {
        let e = (p - t).abs();
        if e <= delta { 0.5 * e * e } else { delta * (e - 0.5 * delta) }
    }).sum();
    let result = sum / n;
    if result.is_nan() || result.is_infinite() {
        return Err(MathError::NumericalFailure("NaN/Inf detected in Huber loss computation".into()));
    }
    Ok(result)
}

/// Smooth L1 (Huber with delta=1).
pub fn smooth_l1(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    huber(pred, target, 1.0)
}

// ---------------------------------------------------------------------------
// Classification losses
// ---------------------------------------------------------------------------

/// Cross-entropy loss for multiclass.
/// `logits`: [batch, classes], `targets`: [batch] with class indices.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> MathResult<f64> {
    if logits.shape.len() != 2 {
        return Err(MathError::InvalidArgument("cross_entropy requires 2-D logits"));
    }
    let (batch, classes) = (logits.shape[0], logits.shape[1]);
    let n = batch as f64;
    let mut loss = 0.0;
    for i in 0..batch {
        let t = targets.data[i] as usize;
        if t >= classes { return Err(MathError::OutOfRange); }
        // Log-sum-exp for stability
        let row = &logits.data[i * classes..(i + 1) * classes];
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse: f64 = row.iter().map(|&x| (x - max_val).exp()).sum::<f64>().ln() + max_val;
        // NaN/Inf protection: if lse or row[t] is NaN/Inf, clamp to stable values
        if lse.is_nan() || lse.is_infinite() || row[t].is_nan() || row[t].is_infinite() {
            return Err(MathError::NumericalFailure("NaN/Inf detected in cross_entropy computation".into()));
        }
        loss += row[t] - lse;
    }
    Ok(-loss / n)
}

/// Cross-entropy gradient w.r.t. logits: (softmax(logits) - one_hot(targets)) / batch.
pub fn cross_entropy_grad(logits: &Tensor, targets: &Tensor) -> MathResult<Tensor> {
    if logits.shape.len() != 2 {
        return Err(MathError::InvalidArgument("cross_entropy requires 2-D logits"));
    }
    let (batch, classes) = (logits.shape[0], logits.shape[1]);
    let n = batch as f64;
    let mut grad = vec![0.0; logits.numel()];
    for i in 0..batch {
        let t = targets.data[i] as usize;
        if t >= classes { return Err(MathError::OutOfRange); }
        // Softmax
        let row = &logits.data[i * classes..(i + 1) * classes];
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for j in 0..classes {
            let sm = exps[j] / sum;
            grad[i * classes + j] = if j == t { (sm - 1.0) / n } else { sm / n };
        }
    }
    Ok(Tensor { shape: logits.shape.clone(), data: grad })
}

/// Binary cross-entropy: -mean(t * log(p) + (1-t) * log(1-p)).
pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    let n = pred.numel() as f64;
    let eps = 1e-7;
    let sum: f64 = pred.data.iter().zip(&target.data).map(|(p, t)| {
        let p = p.clamp(eps, 1.0 - eps);
        -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
    }).sum();
    Ok(sum / n)
}

/// Binary cross-entropy with logits (numerically stable).
/// loss = mean(max(logit, 0) - logit * target + log(1 + exp(-|logit|))).
pub fn binary_cross_entropy_with_logits(logits: &Tensor, target: &Tensor) -> MathResult<f64> {
    let n = logits.numel() as f64;
    let sum: f64 = logits.data.iter().zip(&target.data).map(|(&l, &t)| {
        // numerically stable: max(l,0) - l*t + log(1 + exp(-|l|))
        l.max(0.0) - l * t + (1.0 + (-l.abs()).exp()).ln()
    }).sum();
    Ok(sum / n)
}

/// KL divergence: sum(p * (log(p) - log(q))).
pub fn kl_divergence(p: &Tensor, q: &Tensor) -> MathResult<f64> {
    let eps = 1e-10;
    let sum: f64 = p.data.iter().zip(&q.data).map(|(p, q)| {
        let p = p.max(eps);
        let q = q.max(eps);
        p * (p / q).ln()
    }).sum();
    Ok(sum)
}

/// Hinge loss: mean(max(0, 1 - pred * target)). target should be ±1.
pub fn hinge_loss(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    let n = pred.numel() as f64;
    let sum: f64 = pred.data.iter().zip(&target.data)
        .map(|(p, t)| (1.0 - p * t).max(0.0))
        .sum();
    Ok(sum / n)
}

/// Cosine embedding loss: per-sample cosine similarity.
/// `a`, `b`: [batch, dim], `target`: [batch] with 1.0 (similar) or -1.0 (dissimilar).
pub fn cosine_embedding_loss(a: &Tensor, b: &Tensor, target: &Tensor, margin: f64) -> MathResult<f64> {
    if a.shape != b.shape { return Err(MathError::DimensionMismatch); }
    if a.shape.len() != 2 { return Err(MathError::InvalidArgument("cosine_embedding_loss requires 2-D tensors")); }
    let batch = a.shape[0];
    let dim = a.shape[1] as f64;
    let mut loss = 0.0;
    for i in 0..batch {
        let a_slice = &a.data[i * dim as usize..(i + 1) * dim as usize];
        let b_slice = &b.data[i * dim as usize..(i + 1) * dim as usize];
        let dot: f64 = a_slice.iter().zip(b_slice).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a_slice.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
        let norm_b: f64 = b_slice.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
        let cos_sim = dot / (norm_a * norm_b);
        let t = target.data[i];
        let base = 1.0 - cos_sim;
        loss += if t > 0.0 { base.max(0.0) } else { (base - margin).max(0.0) };
    }
    Ok(loss / batch as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-6;

    #[test]
    fn mse_test() {
        let pred = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
        let target = Tensor::new(&[3], &[1.0, 2.0, 5.0]).unwrap();
        assert!((mse(&pred, &target).unwrap() - 4.0 / 3.0).abs() < E);
    }

    #[test]
    fn cosine_embedding_loss_zero_vector_finite() {
        // Zero vectors previously caused silent sample skips that skewed the
        // mean; every sample must now contribute and the result stay finite.
        let a = Tensor::new(&[2, 2], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let b = Tensor::new(&[2, 2], &[1.0, 0.0, 1.0, 0.0]).unwrap();
        let t = Tensor::new(&[2], &[1.0, 1.0]).unwrap();
        let loss = cosine_embedding_loss(&a, &b, &t, 0.5).unwrap();
        assert!(loss.is_finite(), "loss must be finite with zero-vector inputs");
    }

    #[test]
    fn mse_grad_test() {
        let pred = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
        let target = Tensor::new(&[3], &[1.0, 2.0, 5.0]).unwrap();
        let g = mse_grad(&pred, &target).unwrap();
        assert!((g.data[2] + 4.0 / 3.0).abs() < E); // 2*(3-5)/3 = -4/3
    }

    #[test]
    fn mae_test() {
        let pred = Tensor::new(&[3], &[1.0, 2.0, 3.0]).unwrap();
        let target = Tensor::new(&[3], &[1.0, 2.0, 5.0]).unwrap();
        assert!((mae(&pred, &target).unwrap() - 2.0 / 3.0).abs() < E);
    }

    #[test]
    fn huber_test() {
        let pred = Tensor::new(&[2], &[2.0, 0.0]).unwrap();
        let target = Tensor::new(&[2], &[0.0, 0.0]).unwrap();
        // |2-0|=2 > delta=1 → 1*(2-0.5) = 1.5, |0-0|=0 → 0.5*0=0
        let h = huber(&pred, &target, 1.0).unwrap();
        assert!((h - 0.75).abs() < E); // (1.5 + 0) / 2
    }

    #[test]
    fn cross_entropy_perfect() {
        let logits = Tensor::new(&[2, 3], &[10.0, 1.0, 1.0, 1.0, 10.0, 1.0]).unwrap();
        let targets = Tensor::new(&[2], &[0.0, 1.0]).unwrap();
        let loss = cross_entropy(&logits, &targets).unwrap();
        assert!(loss < 0.01); // near 0 for perfect predictions
    }

    #[test]
    fn binary_cross_entropy_stable() {
        let logits = Tensor::new(&[3], &[10.0, -10.0, 0.0]).unwrap();
        let target = Tensor::new(&[3], &[1.0, 0.0, 0.5]).unwrap();
        let loss = binary_cross_entropy_with_logits(&logits, &target).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn binary_cross_entropy_zero_loss() {
        // Perfect prediction: p=1, t=1 → BCE = 0
        let pred = Tensor::new(&[3], &[10.0, 10.0, 10.0]).unwrap();
        let target = Tensor::new(&[3], &[1.0, 1.0, 1.0]).unwrap();
        let loss = binary_cross_entropy(&pred, &target).unwrap();
        assert!(loss < E);
    }

    #[test]
    fn binary_cross_entropy_one_minus_loss() {
        // Perfect prediction: p=0, t=0 → BCE = 0
        let pred = Tensor::new(&[3], &[-10.0, -10.0, -10.0]).unwrap();
        let target = Tensor::new(&[3], &[0.0, 0.0, 0.0]).unwrap();
        let loss = binary_cross_entropy(&pred, &target).unwrap();
        assert!(loss < E);
    }

    #[test]
    fn hinge_margin_test() {
        // target=1, pred should be >= 1 for zero loss
        let pred = Tensor::new(&[3], &[2.0, 1.0, 0.5]).unwrap();
        let target = Tensor::new(&[3], &[1.0, 1.0, 1.0]).unwrap();
        let h = hinge_loss(&pred, &target).unwrap();
        // 1-2=-1→max(0,-1)=0, 1-1=0→max(0,0)=0, 1-0.5=0.5→max(0,0.5)=0.5 → mean=0.5/3
        assert!((h - 0.5 / 3.0).abs() < E);
    }

    #[test]
    fn cosine_embedding_similar_test() {
        // Similar items (target=1) should have low loss
        let a = Tensor::new(&[2, 3], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let b = Tensor::new(&[2, 3], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let target = Tensor::new(&[2], &[1.0, 1.0]).unwrap();
        let loss = cosine_embedding_loss(&a, &b, &target, 0.0).unwrap();
        assert!(loss < E);
    }

    #[test]
    fn kl_divergence_identical_test() {
        let p = Tensor::new(&[4], &[0.25, 0.25, 0.25, 0.25]).unwrap();
        let q = Tensor::new(&[4], &[0.25, 0.25, 0.25, 0.25]).unwrap();
        let d = kl_divergence(&p, &q).unwrap();
        assert!(d.abs() < E);
    }
}
