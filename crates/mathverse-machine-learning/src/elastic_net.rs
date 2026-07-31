//! Elastic Net regression: combined L1 + L2 penalty.

use mathverse_core::error::MathResult;

/// Elastic Net result.
pub struct ElasticNetResult {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub r_squared: f64,
    pub n_iters: usize,
}

/// Fit Elastic Net via coordinate descent.
pub fn fit_elastic_net(
    x: &[Vec<f64>], y: &[f64], alpha: f64, l1_ratio: f64, max_iters: usize, tol: f64,
) -> MathResult<ElasticNetResult> {
    let n = x.len();
    let p = x[0].len();

    // Compute column means and center
    let col_mean: Vec<f64> = (0..p).map(|j| x.iter().map(|xi| xi[j]).sum::<f64>() / n as f64).collect();
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let x_centered: Vec<Vec<f64>> = x.iter()
        .map(|xi| xi.iter().zip(&col_mean).map(|(v, m)| v - m).collect())
        .collect();
    let y_centered: Vec<f64> = y.iter().map(|v| v - y_mean).collect();

    // Column norms squared
    let col_norm2: Vec<f64> = (0..p).map(|j| {
        x_centered.iter().map(|xi| xi[j] * xi[j]).sum::<f64>()
    }).collect();

    let l1 = alpha * l1_ratio;
    let l2 = alpha * (1.0 - l1_ratio) * 2.0; // factor 2 for quadratic
    let mut beta = vec![0.0; p];

    for _iter in 0..max_iters {
        let mut max_change = 0.0_f64;
        for j in 0..p {
            if col_norm2[j] < 1e-15 { continue; }
            // Partial residual (everything except j)
            let rj: f64 = (0..n).map(|i| {
                x_centered[i][j] * (y_centered[i] - (0..p).filter(|&k| k != j).map(|k| x_centered[i][k] * beta[k]).sum::<f64>())
            }).sum();

            let soft = rj / col_norm2[j];
            let beta_new = if soft > l1 / col_norm2[j] {
                (rj - l1) / (col_norm2[j] + l2)
            } else if soft < -l1 / col_norm2[j] {
                (rj + l1) / (col_norm2[j] + l2)
            } else {
                0.0
            };
            max_change = max_change.max((beta_new - beta[j]).abs());
            beta[j] = beta_new;
        }
        if max_change < tol { break; }
    }

    let intercept: f64 = y_mean - beta.iter().zip(&col_mean).map(|(b, m)| b * m).sum::<f64>();
    let y_mean_val = y_mean;
    let ss_tot: f64 = y.iter().map(|v| (v - y_mean_val).powi(2)).sum();
    let ss_res: f64 = x.iter().zip(y).map(|(xi, &yi)| {
        let pred: f64 = xi.iter().zip(&beta).map(|(v, c)| v * c).sum::<f64>() + intercept;
        (yi - pred).powi(2)
    }).sum();
    let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 1.0 };
    Ok(ElasticNetResult { coefficients: beta, intercept, r_squared, n_iters: max_iters })
}

/// Predict using fitted coefficients.
pub fn predict(x: &[Vec<f64>], coefficients: &[f64], intercept: f64) -> Vec<f64> {
    x.iter().map(|xi| {
        xi.iter().zip(coefficients).map(|(v, c)| v * c).sum::<f64>() + intercept
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_net_perfect() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi[0] + 1.0).collect();
        let result = fit_elastic_net(&x, &y, 0.1, 0.5, 1000, 1e-6).unwrap();
        assert!(result.r_squared > 0.99);
    }

    #[test]
    fn elastic_net_predict_test() {
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let pred = predict(&x, &[2.0, 3.0], 1.0);
        assert!((pred[0] - 9.0).abs() < 1e-9);
        assert!((pred[1] - 19.0).abs() < 1e-9);
    }
}
