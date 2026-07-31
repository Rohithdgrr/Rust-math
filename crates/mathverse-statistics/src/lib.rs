//! Statistics: descriptive measures, distributions, hypothesis tests,
//! regression, multivariate analysis, and inference.

pub mod descriptive;
pub mod distributions;
pub mod hypothesis_tests;
pub mod regression;
pub mod matrix;
pub mod inference;

// Re-export key functions from each module
pub use descriptive::{mean, median, mode, variance_sample, variance_pop, std_dev_sample, std_dev_pop,
    quartiles, percentile, quantile, weighted_mean, geometric_mean, harmonic_mean,
    trimmed_mean, winsorized_mean, skewness, kurtosis, mad, iqr, range, coefficient_of_variation,
    standard_error, describe, Summary, covariance, pearson, linear_regression, mean_ci, z_test};

pub use distributions::{normal_pdf, normal_cdf, normal_ppf, Normal,
    student_t_pdf, student_t_cdf, student_t_ppf, StudentT,
    chi_squared_pdf, chi_squared_cdf, chi_squared_ppf, ChiSquared,
    f_pdf, f_cdf, f_ppf, FDist,
    binomial_pmf, binomial_cdf, Binomial,
    poisson_pmf, poisson_cdf, Poisson};

pub use hypothesis_tests::{t_test_two_sample, welch_t_test, paired_t_test, one_sample_t_test,
    f_test_variance, anova_with_p, one_way_anova, chi_squared_gof, chi_squared_independence,
    binomial_test, mann_whitney_u, wilcoxon_signed_rank};

pub use regression::{polynomial_regression, multiple_regression, weighted_least_squares,
    logistic_regression, sigmoid, predict, predict_poly, r_squared, residuals, mse, rmse, mae};

pub use matrix::{covariance_matrix, correlation_matrix, pca, pca_transform,
    mahalanobis, cholesky_inverse, precision_matrix, PCA};

pub use inference::{bootstrap_ci, cohens_d, hedges_g, eta_squared, omega_squared,
    bonferroni, sidak, holm_bonferroni, benjamini_hochberg,
    power_two_sample, sample_size_two_sample, power_one_sample, effect_size_from_t};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptive() {
        let xs: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        assert_eq!(mean(&xs), 5.0);
        assert_eq!(median(&xs), 5.0);
        assert_eq!(quartiles(&xs), (3.0, 5.0, 7.0));
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((variance_pop(&v) - 2.0).abs() < 1e-12);
        assert!((variance_sample(&v) - 2.5).abs() < 1e-12);
        assert_eq!(mode(&[1.0, 1.0, 2.0, 3.0]), Some(1.0));
        assert_eq!(mode(&[1.0, 2.0, 3.0]), None);
    }

    #[test]
    fn correlation_and_regression() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [3.0, 5.0, 7.0, 9.0];
        let (s, i, r2) = linear_regression(&xs, &ys);
        assert!((s - 2.0).abs() < 1e-12);
        assert!((i - 1.0).abs() < 1e-12);
        assert!((r2 - 1.0).abs() < 1e-12);
        assert!((pearson(&xs, &ys) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inference() {
        let xs: Vec<f64> = (1..=9).map(|i| i as f64).collect();
        let (lo, hi) = mean_ci(&xs, 1.96);
        assert!(lo < 5.0 && hi > 5.0);
        assert_eq!(z_test(1.0, 1.0, 10, 1.0, 1.0, 10), 0.0);
        let g1 = [1.0, 3.0, 5.0];
        let g2 = [2.0, 3.0, 4.0];
        let (f, _p) = one_way_anova(&[&g1, &g2]);
        assert!((f - 0.0).abs() < 1e-12);
    }
}
