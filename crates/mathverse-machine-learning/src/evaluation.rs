/// Performs k-fold cross-validation and returns negative MSE scores.
#[must_use]
pub fn cross_val_score<F>(x: &[Vec<f64>], y: &[f64], k: usize, predict_fn: F) -> Vec<f64>
where
    F: Fn(&[Vec<f64>], &[f64], &[Vec<f64>]) -> Vec<f64>,
{
    let n = x.len();
    let fold_size = n / k;
    let mut scores = Vec::with_capacity(k);

    for i in 0..k {
        let test_start = i * fold_size;
        let test_end = if i == k - 1 { n } else { (i + 1) * fold_size };

        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();

        for (j, xi) in x.iter().enumerate() {
            if j >= test_start && j < test_end {
                test_x.push(xi.clone());
            } else {
                train_x.push(xi.clone());
                train_y.push(y[j]);
            }
        }

        let preds = predict_fn(&train_x, &train_y, &test_x);
        let mse: f64 = preds
            .iter()
            .zip(y[test_start..test_end].iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / preds.len() as f64;
        scores.push(-mse);
    }

    scores
}

/// Splits data into k stratified folds preserving class distribution.
#[must_use]
pub fn stratified_k_fold(y: &[f64], k: usize, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = y.len();
    let mut indices: Vec<usize> = (0..n).collect();

    // Deterministic shuffle
    let mut rng_state = seed;
    for i in (1..n).rev() {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }

    // Group by class
    let mut class_groups: std::collections::HashMap<i64, Vec<usize>> =
        std::collections::HashMap::new();
    for &idx in &indices {
        let class = (y[idx] * 1000.0).round() as i64;
        class_groups.entry(class).or_default().push(idx);
    }

    // Distribute into folds
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k];
    for group in class_groups.values_mut() {
        for (i, &idx) in group.iter().enumerate() {
            folds[i % k].push(idx);
        }
    }

    // Build train/test pairs
    (0..k)
        .map(|i| {
            let test = folds[i].clone();
            let train: Vec<usize> = (0..k)
                .filter(|&j| j != i)
                .flat_map(|j| folds[j].clone())
                .collect();
            (train, test)
        })
        .collect()
}

/// Computes learning curve showing train/validation scores at various dataset sizes.
#[must_use]
pub fn learning_curve<F>(
    x: &[Vec<f64>],
    y: &[f64],
    train_sizes: &[usize],
    predict_fn: F,
) -> Vec<(usize, f64, f64)>
where
    F: Fn(&[Vec<f64>], &[f64], &[Vec<f64>]) -> Vec<f64>,
{
    let n = x.len();
    let val_size = n / 5;
    let val_x = &x[n - val_size..];
    let val_y = &y[n - val_size..];
    let train_x_full = &x[..n - val_size];
    let train_y_full = &y[..n - val_size];

    train_sizes
        .iter()
        .map(|&size| {
            let size = size.min(train_x_full.len());
            let train_x = &train_x_full[..size];
            let train_y = &train_y_full[..size];

            let train_preds = predict_fn(train_x, train_y, train_x);
            let train_score: f64 = train_preds
                .iter()
                .zip(train_y.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / train_preds.len() as f64;

            let val_preds = predict_fn(train_x, train_y, val_x);
            let val_score: f64 = val_preds
                .iter()
                .zip(val_y.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / val_preds.len() as f64;

            (size, -train_score, -val_score)
        })
        .collect()
}

/// Estimates model performance via bootstrap resampling, returning (mean, std).
#[must_use]
pub fn bootstrap_score<F>(
    x: &[Vec<f64>],
    y: &[f64],
    n_bootstrap: usize,
    seed: u64,
    predict_fn: F,
) -> (f64, f64)
where
    F: Fn(&[Vec<f64>], &[f64], &[Vec<f64>]) -> Vec<f64>,
{
    let n = x.len();
    let mut rng_state = seed;
    let mut scores = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let mut train_indices = Vec::with_capacity(n);
        let mut test_indices = Vec::with_capacity(n);

        for _ in 0..n {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (rng_state >> 33) as usize % n;
            train_indices.push(idx);
        }

        let mut seen = std::collections::HashSet::new();
        for &idx in &train_indices {
            if !seen.contains(&idx) {
                seen.insert(idx);
                test_indices.push(idx);
            }
        }

        if test_indices.is_empty() {
            continue;
        }

        let train_x: Vec<Vec<f64>> = train_indices.iter().map(|&i| x[i].clone()).collect();
        let train_y: Vec<f64> = train_indices.iter().map(|&i| y[i]).collect();
        let test_x: Vec<Vec<f64>> = test_indices.iter().map(|&i| x[i].clone()).collect();
        let test_y: Vec<f64> = test_indices.iter().map(|&i| y[i]).collect();

        let preds = predict_fn(&train_x, &train_y, &test_x);
        let score: f64 = preds
            .iter()
            .zip(test_y.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / preds.len() as f64;
        scores.push(-score);
    }

    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
    (mean, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_predict(train_x: &[Vec<f64>], train_y: &[f64], test_x: &[Vec<f64>]) -> Vec<f64> {
        let mean_y: f64 = train_y.iter().sum::<f64>() / train_y.len() as f64;
        let mean_x: Vec<f64> = if !train_x.is_empty() {
            (0..train_x[0].len())
                .map(|j| train_x.iter().map(|r| r[j]).sum::<f64>() / train_x.len() as f64)
                .collect()
        } else {
            vec![0.0]
        };

        test_x
            .iter()
            .map(|xi| {
                let mut pred = mean_y;
                for (j, xij) in xi.iter().enumerate() {
                    pred += 0.1 * (xij - mean_x[j]);
                }
                pred
            })
            .collect()
    }

    #[test]
    fn test_cross_val_score() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64 * 2.0).collect();
        let scores = cross_val_score(&x, &y, 4, simple_predict);
        assert_eq!(scores.len(), 4);
        for s in &scores {
            assert!(*s < 0.0, "MSE should be negative R2-like: {s}");
        }
    }

    #[test]
    fn test_stratified_k_fold() {
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let folds = stratified_k_fold(&y, 3, 42);
        assert_eq!(folds.len(), 3);
        for (train, test) in &folds {
            assert!(!train.is_empty());
            assert!(!test.is_empty());
            let total = train.len() + test.len();
            assert_eq!(total, 6);
        }
    }

    #[test]
    fn test_learning_curve() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let curve = learning_curve(&x, &y, &[2, 5, 10], simple_predict);
        assert_eq!(curve.len(), 3);
        for (size, train_s, val_s) in &curve {
            assert!(*size > 0);
            assert!(*train_s <= 0.0);
            assert!(*val_s <= 0.0);
        }
    }

    #[test]
    fn test_bootstrap_score() {
        let x: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (mean, std) = bootstrap_score(&x, &y, 10, 42, simple_predict);
        assert!(mean < 0.0);
        assert!(std >= 0.0);
    }
}
