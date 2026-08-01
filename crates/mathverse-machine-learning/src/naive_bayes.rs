//! Gaussian Naive Bayes classifier.

use mathverse_core::error::MathResult;

/// Fitted Naive Bayes model.
#[derive(Debug, Clone)]
pub struct NaiveBayesModel {
    /// Unique class labels.
    pub classes: Vec<f64>,
    /// Prior probability of each class.
    pub class_priors: Vec<f64>,
    /// Per-class per-feature means.
    pub means: Vec<Vec<f64>>,
    /// Per-class per-feature variances.
    pub variances: Vec<Vec<f64>>,
}

/// Fit Gaussian Naive Bayes.
#[must_use]
pub fn fit(x: &[Vec<f64>], y: &[f64]) -> MathResult<NaiveBayesModel> {
    let n = y.len();
    let p = x[0].len();
    // Find unique classes
    let mut classes: Vec<f64> = y.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    classes.dedup();
    let nc = classes.len();
    let mut counts = vec![0usize; nc];
    let mut sums = vec![vec![0.0; p]; nc];
    let mut sum_sqs = vec![vec![0.0; p]; nc];
    for i in 0..n {
        let ci = classes
            .iter()
            .position(|&c| (c - y[i]).abs() < 1e-10)
            .unwrap();
        counts[ci] += 1;
        for j in 0..p {
            sums[ci][j] += x[i][j];
            sum_sqs[ci][j] += x[i][j] * x[i][j];
        }
    }
    let mut means = vec![vec![0.0; p]; nc];
    let mut variances = vec![vec![0.0; p]; nc];
    let mut class_priors = vec![0.0; nc];
    for c in 0..nc {
        class_priors[c] = counts[c] as f64 / n as f64;
        for j in 0..p {
            let n_c = counts[c] as f64;
            means[c][j] = sums[c][j] / n_c;
            variances[c][j] = (sum_sqs[c][j] / n_c - means[c][j].powi(2)).max(1e-9);
        }
    }
    Ok(NaiveBayesModel {
        classes,
        class_priors,
        means,
        variances,
    })
}

/// Predict class labels.
#[must_use]
pub fn predict(model: &NaiveBayesModel, x: &[Vec<f64>]) -> MathResult<Vec<f64>> {
    x.iter().map(|row| predict_one(model, row)).collect()
}

/// Predict probabilities for each class.
#[must_use]
pub fn predict_proba(model: &NaiveBayesModel, x: &[Vec<f64>]) -> MathResult<Vec<Vec<f64>>> {
    x.iter().map(|row| predict_one_proba(model, row)).collect()
}

fn predict_one(model: &NaiveBayesModel, x: &[f64]) -> MathResult<f64> {
    let probs = predict_one_proba(model, x)?;
    let best = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    Ok(model.classes[best])
}

fn predict_one_proba(model: &NaiveBayesModel, x: &[f64]) -> MathResult<Vec<f64>> {
    let nc = model.classes.len();
    let mut log_probs = vec![0.0; nc];
    for c in 0..nc {
        log_probs[c] = model.class_priors[c].ln();
        for j in 0..x.len() {
            let m = model.means[c][j];
            let v = model.variances[c][j];
            log_probs[c] += -0.5 * ((2.0 * std::f64::consts::PI * v).ln() + (x[j] - m).powi(2) / v);
        }
    }
    let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = log_probs.iter().map(|lp| (lp - max_log).exp()).collect();
    let sum: f64 = exps.iter().sum();
    Ok(exps.iter().map(|e| e / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_predict_iris_like() {
        // Simple 2-class, 2-feature data
        let x = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![1.2, 2.2], // class 0
            vec![5.0, 6.0],
            vec![5.5, 5.8],
            vec![4.8, 6.2], // class 1
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let model = fit(&x, &y).unwrap();
        assert_eq!(model.classes, vec![0.0, 1.0]);
        let preds = predict(&model, &x).unwrap();
        let correct = preds
            .iter()
            .zip(&y)
            .filter(|(&p, &t)| (p - t).abs() < 0.5)
            .count();
        assert!(correct >= 5); // at least 5/6 correct
    }

    #[test]
    fn predict_proba_sums_to_one() {
        let x = vec![vec![1.0, 2.0], vec![5.0, 6.0]];
        let y = vec![0.0, 1.0];
        let model = fit(&x, &y).unwrap();
        let probs = predict_proba(&model, &x).unwrap();
        for row in &probs {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
