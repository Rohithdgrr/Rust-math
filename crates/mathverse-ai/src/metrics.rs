//! Evaluation metrics: classification (accuracy, precision, recall, F1, ROC AUC)
//! and regression (MSE, MAE, R², explained variance).

use crate::tensor::Tensor;
use mathverse_core::error::{MathError, MathResult};

// ---------------------------------------------------------------------------
// Classification metrics
// ---------------------------------------------------------------------------

/// Accuracy: fraction of correct predictions.
pub fn accuracy(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    if pred.numel() != target.numel() { return Err(MathError::DimensionMismatch); }
    let correct = pred.data.iter().zip(&target.data)
        .filter(|(p, t)| (*p - *t).abs() < 1e-9)
        .count();
    Ok(correct as f64 / pred.numel() as f64)
}

/// Per-class precision: tp / (tp + fp).
pub fn precision(pred: &Tensor, target: &Tensor, num_classes: usize) -> MathResult<Vec<f64>> {
    let cm = confusion_matrix(pred, target, num_classes)?;
    let mut result = Vec::with_capacity(num_classes);
    #[allow(clippy::needless_range_loop)]
    for c in 0..num_classes {
        let tp = cm[c][c];
        let col_sum: f64 = (0..num_classes).map(|r| cm[r][c]).sum();
        result.push(if col_sum > 0.0 { tp / col_sum } else { 0.0 });
    }
    Ok(result)
}

/// Per-class recall: tp / (tp + fn).
pub fn recall(pred: &Tensor, target: &Tensor, num_classes: usize) -> MathResult<Vec<f64>> {
    let cm = confusion_matrix(pred, target, num_classes)?;
    let mut result = Vec::with_capacity(num_classes);
    #[allow(clippy::needless_range_loop)]
    for c in 0..num_classes {
        let tp = cm[c][c];
        let row_sum: f64 = cm[c].iter().sum();
        result.push(if row_sum > 0.0 { tp / row_sum } else { 0.0 });
    }
    Ok(result)
}

/// Per-class F1: 2 * precision * recall / (precision + recall).
pub fn f1(pred: &Tensor, target: &Tensor, num_classes: usize) -> MathResult<Vec<f64>> {
    let p = precision(pred, target, num_classes)?;
    let r = recall(pred, target, num_classes)?;
    Ok(p.iter().zip(&r).map(|(p, r)| {
        if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
    }).collect())
}

/// Confusion matrix [num_classes × num_classes].
///
/// # Errors
///
/// Returns `MathError::OutOfRange` if any prediction or target label is
/// outside `[0, num_classes)`.
pub fn confusion_matrix(pred: &Tensor, target: &Tensor, num_classes: usize) -> MathResult<Vec<Vec<f64>>> {
    if pred.numel() != target.numel() { return Err(MathError::DimensionMismatch); }
    let mut cm = vec![vec![0.0f64; num_classes]; num_classes];
    for (p, t) in pred.data.iter().zip(&target.data) {
        let pi = *p as usize;
        let ti = *t as usize;
        if pi >= num_classes || ti >= num_classes || *p < 0.0 || *t < 0.0 || (*p - pi as f64).abs() > 1e-9 || (*t - ti as f64).abs() > 1e-9 {
            return Err(MathError::OutOfRange);
        }
        cm[ti][pi] += 1.0;
    }
    Ok(cm)
}

/// ROC AUC via trapezoidal integration.
/// `scores`: prediction scores (higher = more positive).
/// `labels`: binary labels (0 or 1).
pub fn roc_auc(scores: &Tensor, labels: &Tensor) -> MathResult<f64> {
    if scores.numel() != labels.numel() { return Err(MathError::DimensionMismatch); }
    let n = scores.numel();
    let mut pairs: Vec<(f64, f64)> = scores.data.iter().zip(&labels.data)
        .map(|(&s, &l)| (s, l))
        .collect();
    // Sort by score descending (highest first)
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos: f64 = labels.data.iter().sum();
    let total_neg = n as f64 - total_pos;
    if total_pos == 0.0 || total_neg == 0.0 { return Ok(0.5); }

    let mut tp = 0.0f64;
    let mut fp = 0.0f64;
    let mut prev_fpr = 0.0f64;
    let mut prev_tpr = 0.0f64;
    let mut auc = 0.0;

    for &(_, l) in &pairs {
        if l > 0.5 { tp += 1.0; } else { fp += 1.0; }
        let tpr = tp / total_pos;
        let fpr = fp / total_neg;
        auc += 0.5 * (prev_tpr + tpr) * (fpr - prev_fpr);
        prev_fpr = fpr;
        prev_tpr = tpr;
    }
    Ok(auc)
}

// ---------------------------------------------------------------------------
// Regression metrics
// ---------------------------------------------------------------------------

/// R² (coefficient of determination): 1 - SS_res / SS_tot.
pub fn r_squared(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    let n = pred.numel() as f64;
    let mean: f64 = target.data.iter().sum::<f64>() / n;
    let ss_tot: f64 = target.data.iter().map(|t| (t - mean).powi(2)).sum();
    let ss_res: f64 = pred.data.iter().zip(&target.data).map(|(p, t)| (p - t).powi(2)).sum();
    if ss_tot == 0.0 { return Ok(1.0); }
    Ok(1.0 - ss_res / ss_tot)
}

/// Explained variance: 1 - Var(target - pred) / Var(target).
pub fn explained_variance(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    let n = pred.numel() as f64;
    let mean_t: f64 = target.data.iter().sum::<f64>() / n;
    let var_t: f64 = target.data.iter().map(|t| (t - mean_t).powi(2)).sum::<f64>() / n;
    let diffs: Vec<f64> = pred.data.iter().zip(&target.data).map(|(p, t)| t - p).collect();
    let mean_d: f64 = diffs.iter().sum::<f64>() / n;
    let var_d: f64 = diffs.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>() / n;
    if var_t == 0.0 { return Ok(1.0); }
    Ok(1.0 - var_d / var_t)
}

/// MSE (re-exported for metric context).
pub fn mse(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    crate::losses::mse(pred, target)
}

/// MAE (re-exported for metric context).
pub fn mae(pred: &Tensor, target: &Tensor) -> MathResult<f64> {
    crate::losses::mae(pred, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    const E: f64 = 1e-6;

    #[test]
    fn accuracy_test() {
        let pred = Tensor::new(&[5], &[0.0, 1.0, 2.0, 1.0, 0.0]).unwrap();
        let target = Tensor::new(&[5], &[0.0, 2.0, 2.0, 1.0, 1.0]).unwrap();
        assert!((accuracy(&pred, &target).unwrap() - 0.6).abs() < E);
    }

    #[test]
    fn confusion_matrix_test() {
        let pred = Tensor::new(&[4], &[0.0, 1.0, 1.0, 0.0]).unwrap();
        let target = Tensor::new(&[4], &[0.0, 1.0, 0.0, 1.0]).unwrap();
        let cm = confusion_matrix(&pred, &target, 2).unwrap();
        assert!((cm[0][0] - 1.0).abs() < E); // true 0, pred 0
        assert!((cm[0][1] - 1.0).abs() < E); // true 0, pred 1
        assert!((cm[1][0] - 1.0).abs() < E); // true 1, pred 0
        assert!((cm[1][1] - 1.0).abs() < E); // true 1, pred 1
    }

    #[test]
    fn precision_recall_f1_test() {
        let pred = Tensor::new(&[6], &[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
        let target = Tensor::new(&[6], &[0.0, 1.0, 1.0, 1.0, 0.0, 0.0]).unwrap();
        let p = precision(&pred, &target, 2).unwrap();
        let r = recall(&pred, &target, 2).unwrap();
        let f = f1(&pred, &target, 2).unwrap();
        // cm: [[2,1],[1,2]] — class 0: tp=2, col_sum=3, row_sum=3
        assert!((p[0] - 2.0 / 3.0).abs() < E);
        assert!((r[0] - 2.0 / 3.0).abs() < E);
        assert!((f[0] - 2.0 / 3.0).abs() < E);
    }

    #[test]
    fn confusion_matrix_rejects_out_of_range_labels() {
        let pred = Tensor::new(&[3], &[0.0, 1.0, 5.0]).unwrap();
        let target = Tensor::new(&[3], &[0.0, 1.0, 0.0]).unwrap();
        assert!(confusion_matrix(&pred, &target, 2).is_err());
    }

    #[test]
    fn roc_auc_test() {
        let scores = Tensor::new(&[5], &[0.1, 0.4, 0.35, 0.8, 0.9]).unwrap();
        let labels = Tensor::new(&[5], &[0.0, 0.0, 0.0, 1.0, 1.0]).unwrap();
        let auc = roc_auc(&scores, &labels).unwrap();
        assert!((auc - 1.0).abs() < E); // perfect separation
    }

    #[test]
    fn r_squared_test() {
        let pred = Tensor::new(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let target = Tensor::new(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((r_squared(&pred, &target).unwrap() - 1.0).abs() < E);
    }

    #[test]
    fn explained_variance_test() {
        let pred = Tensor::new(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let target = Tensor::new(&[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((explained_variance(&pred, &target).unwrap() - 1.0).abs() < E);
    }
}








