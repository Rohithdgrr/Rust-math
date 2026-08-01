//! Logistic regression via gradient descent.

use mathverse_core::error::MathResult;

/// Logistic regression result.
#[derive(Debug, Clone)]
pub struct LogisticResult {
    /// Regression coefficients for each feature.
    pub coefficients: Vec<f64>,
    /// Intercept (bias) term.
    pub intercept: f64,
    /// Number of iterations performed before convergence.
    pub n_iters: usize,
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

/// Fit logistic regression via gradient descent.
/// `x`: \[n_samples × n_features\], `y`: \[n_samples\] binary labels (0 or 1).
#[must_use]
pub fn fit(
    x: &[Vec<f64>],
    y: &[f64],
    lr: f64,
    max_iters: usize,
    tol: f64,
) -> MathResult<LogisticResult> {
    let n = y.len();
    let p = x[0].len();
    let mut coef = vec![0.0; p];
    let mut intercept = 0.0;
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
            coef[j] -= lr * grad_coef[j] / n as f64;
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
            });
        }
    }
    Ok(LogisticResult {
        coefficients: coef,
        intercept,
        n_iters: max_iters,
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
        let r = fit(&x, &y, 0.1, 1000, 1e-8).unwrap();
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
    fn predict_proba_range() {
        let (x, y) = separable_data();
        let r = fit(&x, &y, 0.1, 500, 1e-6).unwrap();
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
}
