use mathverse_core::error::{MathError, MathResult};
use std::f64;

pub fn matthews_correlation(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut tn = 0.0;
    let mut fn_ = 0.0;

    for (p, t) in pred.iter().zip(target.iter()) {
        let p_bool = *p > 0.5;
        let t_bool = *t > 0.5;
        match (p_bool, t_bool) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, false) => tn += 1.0,
            (false, true) => fn_ += 1.0,
        }
    }

    let denom_val: f64 = (tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_);
    let denom = denom_val.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (tp * tn - fp * fn_) / denom
}

pub fn cohen_kappa(pred: &[f64], target: &[f64]) -> f64 {
    let n = pred.len() as f64;
    let mut agreement = 0.0;
    let mut po = 0.0; // observed agreement

    for (p, t) in pred.iter().zip(target.iter()) {
        if (*p > 0.5) == (*t > 0.5) {
            po += 1.0;
        }
    }
    po /= n;

    // Expected agreement by chance
    let p11: f64 = pred.iter().filter(|&&p| p > 0.5).count() as f64 / n;
    let t11: f64 = target.iter().filter(|&&t| t > 0.5).count() as f64 / n;
    let pe = p11 * t11 + (1.0 - p11) * (1.0 - t11);

    if (1.0 - pe).abs() < 1e-12 {
        return 0.0;
    }
    (po - pe) / (1.0 - pe)
}

pub fn log_loss(pred_proba: &[Vec<f64>], target: &[f64]) -> f64 {
    let eps = 1e-15;
    let mut loss = 0.0;
    for (probs, &t) in pred_proba.iter().zip(target.iter()) {
        let class = t as usize;
        if class < probs.len() {
            let p = probs[class].clamp(eps, 1.0 - eps);
            loss -= p.ln();
        }
    }
    loss / target.len() as f64
}

pub fn brier_score(pred_proba: &[Vec<f64>], target: &[f64]) -> f64 {
    let mut score = 0.0;
    for (probs, &t) in pred_proba.iter().zip(target.iter()) {
        let class = t as usize;
        for (i, &p) in probs.iter().enumerate() {
            let actual = if i == class { 1.0 } else { 0.0 };
            score += (p - actual).powi(2);
        }
    }
    score / target.len() as f64
}

pub fn calibration_curve(
    pred_proba: &[Vec<f64>],
    target: &[f64],
    n_bins: usize,
) -> Vec<(f64, f64, usize)> {
    let mut bins: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); n_bins];

    for (probs, &t) in pred_proba.iter().zip(target.iter()) {
        let prob = if !probs.is_empty() { probs[1.min(probs.len() - 1)] } else { 0.0 };
        let bin_idx = ((prob * n_bins as f64) as usize).min(n_bins - 1);
        bins[bin_idx].0 += prob;
        bins[bin_idx].1 += t;
        bins[bin_idx].2 += 1;
    }

    bins.into_iter()
        .map(|(sum_prob, sum_target, count)| {
            if count == 0 {
                (0.0, 0.0, 0)
            } else {
                (sum_prob / count as f64, sum_target / count as f64, count)
            }
        })
        .collect()
}

pub fn precision_at_k(scores: &[f64], labels: &[f64], k: usize) -> f64 {
    let k = k.min(scores.len());
    let mut indexed: Vec<(f64, f64)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let relevant_in_top_k: usize = indexed[..k]
        .iter()
        .filter(|(_, l)| *l > 0.5)
        .count();
    relevant_in_top_k as f64 / k as f64
}

pub fn ndcg(scores: &[f64], labels: &[f64], k: usize) -> f64 {
    let k = k.min(scores.len());
    let mut indexed: Vec<(f64, f64)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // DCG
    let dcg: f64 = indexed[..k]
        .iter()
        .enumerate()
        .map(|(i, (_, l))| l / ((i + 2) as f64).log2())
        .sum();

    // Ideal DCG
    let mut ideal_labels: Vec<f64> = labels.to_vec();
    ideal_labels.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let idcg: f64 = ideal_labels[..k]
        .iter()
        .enumerate()
        .map(|(i, &l)| l / ((i + 2) as f64).log2())
        .sum();

    if idcg < 1e-12 {
        0.0
    } else {
        dcg / idcg
    }
}

pub fn mean_absolute_percentage_error(pred: &[f64], target: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for (p, t) in pred.iter().zip(target.iter()) {
        if t.abs() > 1e-12 {
            sum += ((p - t) / t).abs();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64 * 100.0
    }
}

pub fn median_absolute_error(pred: &[f64], target: &[f64]) -> f64 {
    let mut errors: Vec<f64> = pred
        .iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t).abs())
        .collect();
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = errors.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (errors[n / 2 - 1] + errors[n / 2]) / 2.0
    } else {
        errors[n / 2]
    }
}

pub fn max_error(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t).abs())
        .fold(0.0, f64::max)
}

pub fn tweedie_deviance(pred: &[f64], target: &[f64], power: f64) -> f64 {
    let mut dev = 0.0;
    for (p, t) in pred.iter().zip(target.iter()) {
        let p_pos = p.max(1e-10);
        let t_pos = t.max(1e-10);
        dev += 2.0
            * (t_pos.powf(2.0 - power) / ((2.0 - power) * (1.0 - power))
                - t_pos * p_pos.powf(1.0 - power) / (1.0 - power)
                + p_pos.powf(2.0 - power) / (2.0 - power));
    }
    dev / target.len() as f64
}

#[derive(Debug, Clone)]
pub struct ConfusionMatrixResult {
    pub tp: Vec<usize>,
    pub fp: Vec<usize>,
    pub fn_: Vec<usize>,
    pub tn: Vec<usize>,
}

pub fn confusion_matrix_detailed(pred: &[f64], target: &[f64], num_classes: usize) -> ConfusionMatrixResult {
    let mut tp = vec![0usize; num_classes];
    let mut fp = vec![0usize; num_classes];
    let mut fn_ = vec![0usize; num_classes];
    let mut tn = vec![0usize; num_classes];

    for (p, t) in pred.iter().zip(target.iter()) {
        let p_class = (*p as usize).min(num_classes - 1);
        let t_class = (*t as usize).min(num_classes - 1);

        for c in 0..num_classes {
            if p_class == c && t_class == c {
                tp[c] += 1;
            } else if p_class == c && t_class != c {
                fp[c] += 1;
            } else if p_class != c && t_class == c {
                fn_[c] += 1;
            } else {
                tn[c] += 1;
            }
        }
    }

    ConfusionMatrixResult { tp, fp, fn_, tn }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matthews() {
        let pred = vec![1.0, 1.0, 0.0, 0.0];
        let target = vec![1.0, 1.0, 0.0, 1.0];
        let mcc = matthews_correlation(&pred, &target);
        assert!((mcc - 2.0 / 3.0_f64.sqrt()).abs() < 0.01 || (mcc - 0.577).abs() < 0.01, "mcc={mcc}");
    }

    #[test]
    fn test_matthews_perfect() {
        let pred = vec![1.0, 0.0, 1.0, 0.0];
        let target = vec![1.0, 0.0, 1.0, 0.0];
        let mcc = matthews_correlation(&pred, &target);
        assert!((mcc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cohen_kappa() {
        let pred = vec![1.0, 1.0, 0.0, 0.0];
        let target = vec![1.0, 0.0, 0.0, 1.0];
        let kappa = cohen_kappa(&pred, &target);
        assert!(kappa > -1.0 && kappa <= 1.0);
    }

    #[test]
    fn test_log_loss() {
        let pred_proba = vec![vec![0.1, 0.9], vec![0.8, 0.2]];
        let target = vec![1.0, 0.0];
        let ll = log_loss(&pred_proba, &target);
        assert!(ll > 0.0);
    }

    #[test]
    fn test_brier_score() {
        let pred_proba = vec![vec![0.1, 0.9], vec![0.8, 0.2]];
        let target = vec![1.0, 0.0];
        let bs = brier_score(&pred_proba, &target);
        assert!(bs >= 0.0 && bs <= 1.0);
    }

    #[test]
    fn test_calibration_curve() {
        let pred_proba = vec![
            vec![0.1, 0.9],
            vec![0.2, 0.8],
            vec![0.7, 0.3],
            vec![0.8, 0.2],
        ];
        let target = vec![1.0, 1.0, 0.0, 0.0];
        let curve = calibration_curve(&pred_proba, &target, 3);
        assert_eq!(curve.len(), 3);
    }

    #[test]
    fn test_precision_at_k() {
        let scores = vec![0.9, 0.8, 0.7, 0.1, 0.05];
        let labels = vec![1.0, 0.0, 1.0, 0.0, 0.0];
        let p_at_3 = precision_at_k(&scores, &labels, 3);
        assert!((p_at_3 - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ndcg() {
        let scores = vec![0.9, 0.8, 0.1];
        let labels = vec![1.0, 0.0, 1.0];
        let ndcg_val = ndcg(&scores, &labels, 3);
        assert!(ndcg_val >= 0.0 && ndcg_val <= 1.0);
    }

    #[test]
    fn test_mape() {
        let pred = vec![2.0, 4.0];
        let target = vec![1.0, 4.0];
        let mape = mean_absolute_percentage_error(&pred, &target);
        assert!((mape - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_ae() {
        let pred = vec![1.0, 3.0, 5.0];
        let target = vec![2.0, 3.0, 4.0];
        let mae = median_absolute_error(&pred, &target);
        assert!((mae - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_error() {
        let pred = vec![1.0, 5.0, 3.0];
        let target = vec![2.0, 3.0, 4.0];
        let me = max_error(&pred, &target);
        assert!((me - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_tweedie() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.1, 2.1, 2.9];
        let dev = tweedie_deviance(&pred, &target, 1.5);
        assert!(dev >= 0.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let pred = vec![1.0, 0.0, 1.0, 0.0];
        let target = vec![1.0, 1.0, 0.0, 0.0];
        let cm = confusion_matrix_detailed(&pred, &target, 2);
        assert_eq!(cm.tp[0], 1);
        assert_eq!(cm.tp[1], 1);
        assert_eq!(cm.fp[0], 1);
        assert_eq!(cm.fn_[1], 1);
    }
}
