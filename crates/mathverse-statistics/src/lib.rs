//! Statistics: descriptive measures, distributions, hypothesis tests,
//! regression, multivariate analysis, and inference.
//!
//! ## Quick start
//!
//! ```
//! use mathverse_statistics::*;
//!
//! let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
//! let s = describe(&data);
//! println!("n={}, mean={:.1}", s.n, s.mean);
//!
//! let xs = [1.0, 2.0, 3.0, 4.0];
//! let ys = [3.0, 5.0, 7.0, 9.0];
//! let (slope, intercept, r2) = linear_regression(&xs, &ys).unwrap();
//! println!("y = {slope:.1}x + {intercept:.1}, R² = {r2:.3}");
//! ```
//!
//! All modules re-export their public API at the crate root for convenience.
//! Each module also stands alone, e.g. `use mathverse_statistics::distributions::Normal;`.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::approx_constant)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::float_cmp)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

pub mod descriptive;
pub mod distributions;
pub mod hypothesis_tests;
pub mod matrix;
pub mod inference;
pub mod regression;

pub use descriptive::{
    coefficient_of_variation, covariance, describe, geometric_mean, harmonic_mean, iqr,
    kurtosis, linear_regression, mad, mean, mean_ci, median, mode, percentile, pearson,
    quantile, quartiles, range, skewness, standard_error, std_dev_pop, std_dev_sample,
    Summary, trimmed_mean, variance_pop, variance_sample, weighted_mean, winsorized_mean,
    z_test,
};

pub use distributions::{
    binomial_cdf, binomial_pmf, chi_squared_cdf, chi_squared_pdf, chi_squared_ppf, f_cdf, f_pdf,
    f_ppf, normal_cdf, normal_pdf, normal_ppf, poisson_cdf, poisson_pmf, student_t_cdf,
    student_t_pdf, student_t_ppf, Binomial, ChiSquared, FDist, Normal, Poisson, StudentT,
};

pub use hypothesis_tests::{
    binomial_test, chi_squared_gof, chi_squared_independence, f_test_variance,
    mann_whitney_u, one_sample_t_test, one_way_anova, paired_t_test, t_test_two_sample,
    welch_t_test, wilcoxon_signed_rank,
};

pub use regression::{
    logistic_regression, mae, mse, multiple_regression, predict, predict_poly,
    polynomial_regression, r_squared, residuals, rmse, sigmoid, weighted_least_squares,
};

pub use matrix::{cholesky_inverse, correlation_matrix, covariance_matrix, mahalanobis, pca,
    pca_transform, precision_matrix, PCA};

pub use inference::{
    benjamini_hochberg, bonferroni, bootstrap_ci, cohens_d, effect_size_from_t, eta_squared,
    hedges_g, holm_bonferroni, omega_squared, power_one_sample, power_two_sample,
    sample_size_two_sample, sidak,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptive() {
        let xs: Vec<f64> = (1..=9).map(f64::from).collect();
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
        let (s, i, r2) = linear_regression(&xs, &ys).unwrap();
        assert!((s - 2.0).abs() < 1e-12);
        assert!((i - 1.0).abs() < 1e-12);
        assert!((r2 - 1.0).abs() < 1e-12);
        assert!((pearson(&xs, &ys) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inference() {
        let xs: Vec<f64> = (1..=9).map(f64::from).collect();
        let (lo, hi) = mean_ci(&xs, 1.96);
        assert!(lo < 5.0 && hi > 5.0);
        assert_eq!(z_test(1.0, 1.0, 10, 1.0, 1.0, 10), 0.0);
        let g1 = [1.0, 3.0, 5.0];
        let g2 = [2.0, 3.0, 4.0];
        let (f, _p) = one_way_anova(&[&g1, &g2]).unwrap();
        assert!((f - 0.0).abs() < 1e-12);
    }
}
