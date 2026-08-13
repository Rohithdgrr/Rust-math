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

/// Stratified train/test split preserving per-class proportions.
///
/// Samples are grouped by class, each group is shuffled with the same seed,
/// and the `test_ratio` fraction is taken from every class.
#[must_use]
pub fn stratified_train_test_split(
    x: &[Vec<f64>],
    y: &[f64],
    test_ratio: f64,
    seed: u64,
) -> SplitData {
    // A test ratio outside [0, 1] is meaningless; clamp so the slice below
    // can never go out of bounds. Non-finite targets (NaN != NaN) cannot be
    // grouped by class, so they are excluded from both splits rather than
    // silently dropped from only one side.
    let ratio = test_ratio.clamp(0.0, 1.0);
    let mut classes: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();

    let mut test_idx = Vec::new();
    let mut train_idx = Vec::new();
    for &c in &classes {
        let group: Vec<usize> = y
            .iter()
            .enumerate()
            .filter(|(_, &yi)| yi == c)
            .map(|(i, _)| i)
            .collect();
        let perm = shuffled_indices(group.len(), seed);
        let n_test = ((group.len() as f64 * ratio).round() as usize).min(group.len());
        test_idx.extend(perm[..n_test].iter().map(|&k| group[k]));
        train_idx.extend(perm[n_test..].iter().map(|&k| group[k]));
    }
    let x_test: Vec<Vec<f64>> = test_idx.iter().map(|&i| x[i].clone()).collect();
    let x_train: Vec<Vec<f64>> = train_idx.iter().map(|&i| x[i].clone()).collect();
    let y_test: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();
    let y_train: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
    (x_train, x_test, y_train, y_test)
}

/// Deterministic shuffled k-fold split indices as `(train, test)` pairs.
///
/// These are raw column indices into `x`/`y`; every sample appears in exactly
/// one test fold.
#[must_use]
pub fn k_fold_indices(n: usize, k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    // k == 0 would divide by zero; k > n makes `fold_size` truncate to 0 and
    // every fold but the last would have an empty test set (and the last an
    // empty train set), which later turns into NaN accuracies. Return no folds
    // for invalid k so callers see an empty result instead of a panic/NaN.
    if k == 0 || k > n || n == 0 {
        return Vec::new();
    }
    let indices = shuffled_indices(n, seed);
    let fold_size = n / k;
    (0..k)
        .map(|fold| {
            let start = fold * fold_size;
            let end = if fold == k - 1 { n } else { start + fold_size };
            let test: Vec<usize> = indices[start..end].to_vec();
            let mut train: Vec<usize> = indices[..start].to_vec();
            train.extend_from_slice(&indices[end..]);
            (train, test)
        })
        .collect()
}

/// K-fold cross-validation for classification (returns accuracy per fold).
#[must_use]
pub fn k_fold_cv<F>(x: &[Vec<f64>], y: &[f64], k: usize, seed: u64, predict_fn: F) -> Vec<f64>
where
    F: Fn(&[Vec<f64>], &[f64], &[Vec<f64>]) -> Vec<f64>,
{
    let folds = k_fold_indices(x.len(), k, seed);
    let mut accuracies = Vec::with_capacity(k);

    for (train_idx, test_idx) in folds {
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

/// Deterministic time-series split indices as `(train, test)` pairs.
///
/// The train set is always a strict prefix of the test set (no future
/// information leaks). Fold `i` trains on `[0, bound_i)` and tests on
/// `[bound_i, bound_{i+1})`.
#[must_use]
pub fn time_series_split_indices(n: usize, n_splits: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    // n_splits == 0 yields no bounds; n_splits > n makes consecutive bounds
    // equal (integer truncation), producing empty test folds that would break
    // downstream scoring. Return no splits for either case.
    if n_splits == 0 || n_splits > n || n == 0 {
        return Vec::new();
    }
    let bounds: Vec<usize> = (1..=n_splits)
        .map(|i| (i as f64 * n as f64 / n_splits as f64) as usize)
        .collect();
    let mut prev = 0usize;
    bounds
        .into_iter()
        .map(|end| {
            let split = ((0..prev).collect(), (prev..end).collect());
            prev = end;
            split
        })
        .collect()
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

/// Per-class precision, recall, F1, and support from a confusion matrix.
///
/// Class `c` support is the number of true labels in that class. A metric is
/// `0.0` when its denominator is zero (no predictions / no real samples).
#[must_use]
pub fn precision_recall_f1(
    pred: &[f64],
    target: &[f64],
    num_classes: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let cm = confusion_matrix(pred, target, num_classes);
    let mut precision = vec![0.0; num_classes];
    let mut recall = vec![0.0; num_classes];
    let mut f1 = vec![0.0; num_classes];
    let mut support = vec![0usize; num_classes];

    for c in 0..num_classes {
        let tp: f64 = cm[c][c];
        let fp: f64 = (0..num_classes).map(|r| cm[r][c]).sum::<f64>() - tp;
        let fn_: f64 = (0..num_classes).map(|cc| cm[c][cc]).sum::<f64>() - tp;
        support[c] = cm[c].iter().sum::<f64>() as usize;
        precision[c] = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        recall[c] = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        f1[c] = if precision[c] + recall[c] > 0.0 {
            2.0 * precision[c] * recall[c] / (precision[c] + recall[c])
        } else {
            0.0
        };
    }
    (precision, recall, f1, support)
}

/// A full classification report: per-class metrics plus macro/weighted averages.
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    /// Per-class precision.
    pub precision: Vec<f64>,
    /// Per-class recall.
    pub recall: Vec<f64>,
    /// Per-class F1.
    pub f1: Vec<f64>,
    /// Per-class support (true sample counts).
    pub support: Vec<usize>,
    /// Overall accuracy.
    pub accuracy: f64,
    /// Unweighted mean F1 across classes.
    pub macro_f1: f64,
    /// Support-weighted mean F1 across classes.
    pub weighted_f1: f64,
}

/// Build a multi-class classification report from predictions and targets.
#[must_use]
pub fn classification_report(pred: &[f64], target: &[f64], num_classes: usize) -> ClassificationReport {
    let (precision, recall, f1, support) = precision_recall_f1(pred, target, num_classes);
    let total: usize = support.iter().sum();
    let accuracy = accuracy(pred, target);
    let classes_with_data = support.iter().filter(|&&s| s > 0).count().max(1);
    let macro_f1 = f1.iter().sum::<f64>() / classes_with_data as f64;
    let weighted_f1 = if total > 0 {
        f1.iter()
            .zip(&support)
            .map(|(f, &s)| f * s as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };
    ClassificationReport {
        precision,
        recall,
        f1,
        support,
        accuracy,
        macro_f1,
        weighted_f1,
    }
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

    #[test]
    fn k_fold_indices_partition() {
        let n = 21;
        let folds = k_fold_indices(n, 4, 7);
        assert_eq!(folds.len(), 4);
        let mut seen = std::collections::HashSet::new();
        let mut total_test = 0usize;
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), n);
            // Test folds must partition the data (each sample in exactly one).
            for &i in test {
                assert!(!seen.contains(&i), "sample {i} appears in two test folds");
                seen.insert(i);
            }
            // Train fold must not overlap its own test fold.
            for &i in train {
                assert!(!test.contains(&i));
            }
            total_test += test.len();
        }
        assert_eq!(total_test, n);
        assert_eq!(seen.len(), n);
    }

    #[test]
    fn time_series_split_monotonic() {
        let folds = time_series_split_indices(15, 3);
        assert_eq!(folds.len(), 3);
        for (i, (train, test)) in folds.iter().enumerate() {
            assert!(!test.is_empty(), "fold {i} has empty test");
            if let Some(&last_train) = train.last() {
                assert!(last_train < test[0], "train leaks into future");
            }
        }
    }

    #[test]
    fn precision_recall_f1_known() {
        let pred = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let target = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let (p, r, f1, s) = precision_recall_f1(&pred, &target, 2);
        // Class 1: tp=2 (idx1,2), fp=1 (idx4), fn=1 (idx0) -> p=2/3, r=2/3, f1=2/3
        assert!((p[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((r[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((f1[1] - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(s, vec![3, 3]);
        // Class 0: tp=2 (idx3,5), fp=1 (idx0), fn=1 (idx4) -> p=r=f1=2/3
        assert!((p[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((r[0] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn classification_report_sums() {
        let pred = vec![0.0, 1.0, 2.0, 0.0, 1.0];
        let target = vec![0.0, 1.0, 2.0, 0.0, 1.0];
        let rep = classification_report(&pred, &target, 3);
        assert!((rep.accuracy - 1.0).abs() < 1e-12);
        let total: usize = rep.support.iter().sum();
        assert_eq!(total, 5);
        assert!((rep.macro_f1 - 1.0).abs() < 1e-12);
        assert!((rep.weighted_f1 - 1.0).abs() < 1e-12);
    }
}
