//! Model selection: train/test split, k-fold cross-validation, evaluation metrics.

type SplitData = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

/// Split data into train/test sets.
#[must_use]
pub fn train_test_split(x: &[Vec<f64>], y: &[f64], test_ratio: f64, seed: u64) -> SplitData {
    let n = x.len();
    let n_test = (n as f64 * test_ratio).round() as usize;
    let indices = shuffled_indices(n, seed);
    let test_idx = &indices[..n_test];
    let train_idx = &indices[n_test..];
    let x_test: Vec<Vec<f64>> = test_idx.iter().map(|&i| x[i].clone()).collect();
    let x_train: Vec<Vec<f64>> = train_idx.iter().map(|&i| x[i].clone()).collect();
    let y_test: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();
    let y_train: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
    (x_train, x_test, y_train, y_test)
}

/// K-fold cross-validation for classification (returns accuracy per fold).
#[must_use]
pub fn k_fold_cv<F>(x: &[Vec<f64>], y: &[f64], k: usize, seed: u64, predict_fn: F) -> Vec<f64>
where
    F: Fn(&[Vec<f64>], &[f64], &[Vec<f64>]) -> Vec<f64>,
{
    let n = x.len();
    let indices = shuffled_indices(n, seed);
    let fold_size = n / k;
    let mut accuracies = Vec::with_capacity(k);

    for fold in 0..k {
        let start = fold * fold_size;
        let end = if fold == k - 1 { n } else { start + fold_size };
        let test_idx: Vec<usize> = indices[start..end].to_vec();
        let train_idx: Vec<usize> = indices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < start || *i >= end)
            .map(|(_, &i)| i)
            .collect();
        let x_train: Vec<Vec<f64>> = train_idx.iter().map(|&i| x[i].clone()).collect();
        let y_train: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let x_test: Vec<Vec<f64>> = test_idx.iter().map(|&i| x[i].clone()).collect();
        let y_test: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();
        let preds = predict_fn(&x_train, &y_train, &x_test);
        let correct = preds
            .iter()
            .zip(&y_test)
            .filter(|(p, t)| (**p - **t).abs() < 0.5)
            .count();
        accuracies.push(correct as f64 / y_test.len() as f64);
    }
    accuracies
}

/// Compute accuracy from predictions and labels.
#[must_use]
pub fn accuracy(pred: &[f64], target: &[f64]) -> f64 {
    let correct = pred
        .iter()
        .zip(target)
        .filter(|(p, t)| (**p - **t).abs() < 0.5)
        .count();
    correct as f64 / pred.len() as f64
}

/// Confusion matrix.
#[must_use]
pub fn confusion_matrix(pred: &[f64], target: &[f64], num_classes: usize) -> Vec<Vec<f64>> {
    let mut cm = vec![vec![0.0; num_classes]; num_classes];
    for (p, t) in pred.iter().zip(target) {
        let pi = *p as usize;
        let ti = *t as usize;
        if pi < num_classes && ti < num_classes {
            cm[ti][pi] += 1.0;
        }
    }
    cm
}

/// ROC curve points (FPR, TPR).
#[must_use]
pub fn roc_curve(scores: &[f64], labels: &[f64]) -> Vec<(f64, f64)> {
    let mut pairs: Vec<(f64, f64)> = scores.iter().zip(labels).map(|(&s, &l)| (s, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let total_pos: f64 = labels.iter().sum();
    let total_neg = labels.len() as f64 - total_pos;
    if total_pos == 0.0 || total_neg == 0.0 {
        return vec![(0.0, 0.0), (1.0, 1.0)];
    }
    let mut points = vec![(0.0, 0.0)];
    let mut tp = 0.0;
    let mut fp = 0.0;
    for &(_, l) in &pairs {
        if l > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        points.push((fp / total_neg, tp / total_pos));
    }
    points
}

/// AUC from ROC curve points.
#[must_use]
pub fn auc(points: &[(f64, f64)]) -> f64 {
    let mut area = 0.0;
    for w in points.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        area += 0.5 * (y0 + y1) * (x1 - x0);
    }
    area
}

fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut state = seed as u32;
    for i in (1..n).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_test_split_sizes() {
        let x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let (xt, xte, yt, yte) = train_test_split(&x, &y, 0.2, 42);
        assert_eq!(xt.len() + xte.len(), 100);
        assert_eq!(yt.len() + yte.len(), 100);
        assert!((xte.len() as f64 - 20.0).abs() < 2.0);
    }

    #[test]
    fn accuracy_test() {
        let pred = vec![0.0, 1.0, 1.0, 0.0];
        let target = vec![0.0, 1.0, 0.0, 0.0];
        assert!((accuracy(&pred, &target) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn roc_auc_perfect() {
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let points = roc_curve(&scores, &labels);
        let a = auc(&points);
        assert!((a - 1.0).abs() < 1e-6);
    }

    #[test]
    fn k_fold_cv_runs() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 1.0 }).collect();
        let accs = k_fold_cv(&x, &y, 5, 42, |_xtr, _ytr, xte| {
            xte.iter()
                .map(|row| if row[0] < 10.0 { 0.0 } else { 1.0 })
                .collect()
        });
        assert_eq!(accs.len(), 5);
        let mean: f64 = accs.iter().sum::<f64>() / 5.0;
        assert!(mean > 0.8);
    }
}
