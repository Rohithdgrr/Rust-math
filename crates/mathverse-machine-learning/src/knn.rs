//! k-Nearest Neighbors for classification and regression.

use mathverse_core::error::MathResult;

/// KNN classifier. Predicts the majority class among k nearest neighbors.
#[must_use]
pub fn classify(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_test: &[Vec<f64>],
    k: usize,
) -> MathResult<Vec<f64>> {
    assert_eq!(x_train.len(), y_train.len());
    let mut results = Vec::with_capacity(x_test.len());
    for query in x_test {
        let mut dists: Vec<(f64, f64)> = x_train
            .iter()
            .zip(y_train)
            .map(|(x, &y)| (euclidean(query, x), y))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut counts = std::collections::HashMap::new();
        for &(_, label) in dists.iter().take(k) {
            // Normalize -0.0 to 0.0 to avoid hash key collision
            let normalized = if label == 0.0 { 0.0 } else { label };
            *counts.entry(normalized.to_bits()).or_insert(0) += 1;
        }
        let best = counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(&v, _)| f64::from_bits(v))
            .unwrap_or(0.0);
        results.push(best);
    }
    Ok(results)
}

/// KNN regressor. Predicts the mean of k nearest neighbors.
#[must_use]
pub fn regress(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_test: &[Vec<f64>],
    k: usize,
) -> MathResult<Vec<f64>> {
    assert_eq!(x_train.len(), y_train.len());
    let mut results = Vec::with_capacity(x_test.len());
    for query in x_test {
        let mut dists: Vec<(f64, f64)> = x_train
            .iter()
            .zip(y_train)
            .map(|(x, &y)| (euclidean(query, x), y))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let sum: f64 = dists.iter().take(k).map(|(_, y)| y).sum();
        results.push(sum / k as f64);
    }
    Ok(results)
}

/// Euclidean distance between two vectors.
#[must_use]
#[inline]
pub(crate) fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_simple() {
        let x_train = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        let y_train = vec![0.0, 0.0, 1.0, 1.0];
        let x_test = vec![vec![0.5], vec![10.5]];
        let preds = classify(&x_train, &y_train, &x_test, 1).unwrap();
        assert_eq!(preds, vec![0.0, 1.0]);
    }

    #[test]
    fn regress_simple() {
        let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y_train = vec![0.0, 10.0, 20.0];
        let x_test = vec![vec![1.5]];
        let preds = regress(&x_train, &y_train, &x_test, 2).unwrap();
        assert!((preds[0] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn classify_multiclass() {
        let x_train = vec![vec![0.0], vec![1.0], vec![5.0], vec![6.0], vec![10.0]];
        let y_train = vec![0.0, 0.0, 1.0, 1.0, 2.0];
        let x_test = vec![vec![5.5]];
        let preds = classify(&x_train, &y_train, &x_test, 3).unwrap();
        assert_eq!(preds, vec![1.0]);
    }
}
