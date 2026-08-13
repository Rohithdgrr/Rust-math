//! Logistic regression via gradient descent.

use mathverse_core::error::{MathError, MathResult};

/// Logistic regression result.
#[derive(Debug, Clone)]
pub struct LogisticResult {
    /// Regression coefficients for each feature.
    pub coefficients: Vec<f64>,
    /// Intercept (bias) term.
    pub intercept: f64,
    /// Number of iterations performed before convergence.
    pub n_iters: usize,
    /// L2 regularization parameter used.
    pub c: f64,
}

/// Sigmoid function.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Fit logistic regression via gradient descent with L2 regularization.
/// `x`: \[n_samples × n_features\], `y`: \[n_samples\] binary labels (0 or 1).
/// `c`: Inverse regularization strength (smaller = stronger regularization). Use `f64::INFINITY` for no regularization.
#[must_use]
pub fn fit(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    max_iters: usize,
    tol: f64,
    c: f64,
) -> MathResult<LogisticResult> {
    let p = crate::validate::validate_xy(x, y)?;
    if !lr.is_finite() || lr <= 0.0 {
        return Err(MathError::InvalidArgument("learning rate must be positive"));
    }
    if max_iters == 0 {
        return Err(MathError::InvalidArgument("max_iters must be at least 1"));
    }
    if !tol.is_finite() || tol < 0.0 {
        return Err(MathError::InvalidArgument("tol must be non-negative"));
    }
    if !(c > 0.0 || c.is_infinite()) {
        return Err(MathError::InvalidArgument("c must be positive or infinite"));
    }
    for &yi in y {
        if yi != 0.0 && yi != 1.0 {
            return Err(MathError::InvalidArgument(
                "logistic regression labels must be 0 or 1",
            ));
        }
    }
    let n = y.len();
    let mut coef = vec![0.0; p];
    let mut intercept = 0.0;
    let inv_c = if c.is_infinite() { 0.0 } else { 1.0 / c };

    for iter in 0..max_iters {
        let mut grad_coef = vec![0.0; p];
        let mut grad_intercept = 0.0;
        let mut max_change = 0.0;
        for i in 0..n {
            let z = intercept + x[i].iter().zip(&coef).map(|(xi, ci)| xi * ci).sum::<f64>();
            let pred = sigmoid(z);
            let err = pred - y[i];
            grad_intercept += err;
            for j in 0..p {
                grad_coef[j] += x[i][j] * err;
            }
        }
        intercept -= lr * grad_intercept / n as f64;
        for j in 0..p {
            let old = coef[j];
            // L2 regularization: gradient += inv_c * coef[j]
            coef[j] -= lr * (grad_coef[j] / n as f64 + inv_c * coef[j]);
            let change = (coef[j] - old).abs();
            if change > max_change {
                max_change = change;
            }
        }
        if max_change < tol {
            return Ok(LogisticResult {
                coefficients: coef,
                intercept,
                n_iters: iter + 1,
                c,
            });
        }
    }
    Ok(LogisticResult {
        coefficients: coef,
        intercept,
        n_iters: max_iters,
        c,
    })
}

/// Predict probabilities.
#[must_use]
#[inline]
pub fn predict_proba(x: &[Vec<f64>], coefficients: &[f64], intercept: f64) -> Vec<f64> {
    x.iter()
        .map(|row| {
            let z = intercept
                + row
                    .iter()
                    .zip(coefficients)
                    .map(|(xi, ci)| xi * ci)
                    .sum::<f64>();
            sigmoid(z)
        })
        .collect()
}

/// Predict class labels (threshold at 0.5).
#[must_use]
#[inline]
pub fn predict(x: &[Vec<f64>], coefficients: &[f64], intercept: f64) -> Vec<f64> {
    predict_proba(x, coefficients, intercept)
        .iter()
        .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
        .collect()
}

/// Binary cross-entropy loss.
#[must_use]
pub fn cross_entropy(x: &[Vec<f64>], y: &[f64], coefficients: &[f64], intercept: f64) -> f64 {
    let probs = predict_proba(x, coefficients, intercept);
    let n = y.len() as f64;
    -probs
        .iter()
        .zip(y)
        .map(|(p, &yi)| {
            let p = p.clamp(1e-10, 1.0 - 1e-10);
            yi * p.ln() + (1.0 - yi) * (1.0 - p).ln()
        })
        .sum::<f64>()
        / n
}

/// Multiclass logistic regression fitted one-vs-rest.
///
/// One binary [`LogisticResult`] is trained per class against the rest; the
/// probability estimate for a sample normalizes the per-class sigmoid scores
/// to sum to one (a valid probability distribution even though it is not a
/// true softmax over raw logits).
#[derive(Debug, Clone)]
pub struct OvRResult {
    /// One binary logistic model per class, aligned with `classes`.
    pub models: Vec<LogisticResult>,
    /// Distinct target classes, sorted ascending.
    pub classes: Vec<f64>,
}

/// Fit one-vs-rest logistic regression for `>= 2` distinct target classes.
///
/// The integer class labels do not need to be 0/1; any distinct `f64` values
/// work. Hyperparameter meaning matches [`fit`].
#[must_use]
pub fn fit_ovr(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    max_iters: usize,
    tol: f64,
    c: f64,
) -> MathResult<OvRResult> {
    crate::validate::validate_xy(x, y)?;
    let mut classes = y.to_vec();
    classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    classes.dedup();
    if classes.len() < 2 {
        return Err(MathError::InvalidArgument(
            "one-vs-rest logistic needs at least 2 classes",
        ));
    }
    let mut models = Vec::with_capacity(classes.len());
    for &class in &classes {
        let binary: Vec<f64> = y.iter().map(|&yi| if yi == class { 1.0 } else { 0.0 }).collect();
        models.push(fit(x, &binary, lr, max_iters, tol, c)?);
    }
    Ok(OvRResult { models, classes })
}

/// Predict per-class probabilities (softmax-normalized) for a fitted `OvR` model.
#[must_use]
pub fn predict_proba_ovr(x: &[Vec<f64>], model: &OvRResult) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            let scores: Vec<f64> = model
                .models
                .iter()
                .map(|m| {
                    let z = m.intercept + row.iter().zip(&m.coefficients).map(|(xi, ci)| xi * ci).sum::<f64>();
                    sigmoid(z)
                })
                .collect();
            let sum: f64 = scores.iter().sum();
            if sum.abs() < 1e-12 {
                vec![1.0 / scores.len() as f64; scores.len()]
            } else {
                scores.into_iter().map(|s| s / sum).collect()
            }
        })
        .collect()
}

/// Predict class labels for a fitted `OvR` model.
#[must_use]
pub fn predict_ovr(x: &[Vec<f64>], model: &OvRResult) -> Vec<f64> {
    predict_proba_ovr(x, model)
        .iter()
        .map(|probs| {
            let (argmax, _) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            model.classes[argmax]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(dead_code)]
    const E: f64 = 0.05;

    fn separable_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        // Class 0: x < 0, Class 1: x > 0
        let x: Vec<Vec<f64>> = (-10..10).map(|i| vec![i as f64]).collect();
        let y: Vec<f64> = (-10..10).map(|i| if i > 0 { 1.0 } else { 0.0 }).collect();
        (x, y)
    }

    #[test]
    fn fit_converges() {
        let (x, y) = separable_data();
        let r = fit(&x, &y, 0.1, 1000, 1e-8, f64::INFINITY).unwrap();
        assert!(r.coefficients[0] > 0.5); // positive weight
        let preds = predict(&x, &r.coefficients, r.intercept);
        let correct = preds
            .iter()
            .zip(&y)
            .filter(|(&p, &t)| (p - t).abs() < 0.5)
            .count();
        assert!(correct >= 18); // at least 90% accuracy
    }

    #[test]
    fn fit_with_regularization() {
        let (x, y) = separable_data();
        let r = fit(&x, &y, 0.1, 1000, 1e-8, 1.0).unwrap();
        // With regularization, coefficient should be smaller than without
        let r_no_reg = fit(&x, &y, 0.1, 1000, 1e-8, f64::INFINITY).unwrap();
        assert!(r.coefficients[0].abs() < r_no_reg.coefficients[0].abs());
    }

    #[test]
    fn predict_proba_range() {
        let (x, y) = separable_data();
        let r = fit(&x, &y, 0.1, 500, 1e-6, f64::INFINITY).unwrap();
        let probs = predict_proba(&x, &r.coefficients, r.intercept);
        assert!(probs.iter().all(|p| *p >= 0.0 && *p <= 1.0));
    }

    #[test]
    fn cross_entropy_perfect() {
        let x = vec![vec![10.0], vec![-10.0]];
        let y = vec![1.0, 0.0];
        let coef = vec![1.0];
        let ce = cross_entropy(&x, &y, &coef, 0.0);
        assert!(ce < 0.1);
    }

    #[test]
    fn empty_input_errors() {
        assert!(fit(&[], &[], 0.1, 100, 1e-6, f64::INFINITY).is_err());
    }

    #[test]
    fn invalid_labels_error() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![0.5, 1.0];
        assert!(fit(&x, &y, 0.1, 100, 1e-6, f64::INFINITY).is_err());
    }

    #[test]
    fn invalid_hyperparams_error() {
        let (x, y) = separable_data();
        assert!(fit(&x, &y, 0.0, 100, 1e-6, f64::INFINITY).is_err());
        assert!(fit(&x, &y, 0.1, 0, 1e-6, f64::INFINITY).is_err());
        assert!(fit(&x, &y, 0.1, 100, 1e-6, -1.0).is_err());
    }

    #[test]
    fn ovr_multiclass() {
        // Three well-separated clusters in 2D.
        let x: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1],
            vec![5.0, 0.0], vec![5.1, 0.0], vec![5.0, 0.1],
            vec![0.0, 5.0], vec![0.1, 5.0], vec![0.0, 5.1],
        ];
        let y: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let model = fit_ovr(&x, &y, 0.5, 2000, 1e-8, 1.0).unwrap();
        assert_eq!(model.classes, vec![0.0, 1.0, 2.0]);
        let preds = predict_ovr(&x, &model);
        let correct = preds
            .iter()
            .zip(&y)
            .filter(|(p, t)| (*p - *t).abs() < 0.5)
            .count();
        assert!(correct >= 8, "only {correct}/9 correct");
        let probs = predict_proba_ovr(&x, &model);
        for row in &probs {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
        assert!(fit_ovr(&x, &[0.0; 9], 0.5, 2000, 1e-8, 1.0).is_err());
    }
}
