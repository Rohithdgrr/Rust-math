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
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::manual_range_contains)]
#![allow(unused_assignments)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::needless_for_each)]
#![allow(clippy::double_must_use)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::manual_midpoint)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::cast_lossless)]

extern crate alloc;

pub mod density;
pub mod descriptive;
pub mod distributions;
pub mod error;
pub mod hypothesis_tests;
pub mod inference;
pub mod matrix;
pub mod regression;

pub use error::{MathError, MathResult};

pub use descriptive::{
    coefficient_of_variation, covariance, describe, fd_rule, geometric_mean, harmonic_mean, iqr,
    kurtosis, linear_regression, mad, mean, mean_ci, median, mode, pearson, percentile, quantile,
    quartiles, range, scott_rule, skewness, sqrt_rule, standard_error, std_dev_pop, std_dev_sample,
    sturges_rule, trimmed_mean, variance_pop, variance_sample, weighted_mean, winsorized_mean,
    z_test, Summary,
};

pub use density::{
    kernel_density, kernel_density_curve, resolve_bandwidth, scott_bandwidth, silverman_bandwidth,
    Bandwidth,
};

pub use distributions::{
    binomial_cdf, binomial_pmf, chi_squared_cdf, chi_squared_pdf, chi_squared_ppf, f_cdf, f_pdf,
    f_ppf, normal_cdf, normal_pdf, normal_ppf, poisson_cdf, poisson_pmf, student_t_cdf,
    student_t_pdf, student_t_ppf, Binomial, ChiSquared, FDist, Normal, Poisson, StudentT,
};

pub use hypothesis_tests::{
    binomial_test, chi_squared_gof, chi_squared_independence, f_test_variance, mann_whitney_u,
    one_sample_t_test, one_way_anova, paired_t_test, t_test_two_sample, welch_t_test,
    wilcoxon_signed_rank,
};

pub use regression::{
    logistic_regression, mae, mse, multiple_regression, polynomial_regression, predict,
    predict_poly, r_squared, residuals, rmse, sigmoid, weighted_least_squares,
};

pub use matrix::{
    cholesky_inverse, correlation_matrix, covariance_matrix, mahalanobis, pca, pca_transform,
    precision_matrix, PCA,
};

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
        let (q1, q2, q3) = quartiles(&xs);
        assert!((q1 - 3.0).abs() < 0.1);
        assert_eq!(q2, 5.0);
        assert!((q3 - 7.0).abs() < 0.1);
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((variance_pop(&v) - 2.0).abs() < 1e-12);
        assert!((variance_sample(&v) - 2.5).abs() < 1e-12);
        assert_eq!(mode(&[1.0, 1.0, 2.0, 3.0]), Some(1.0));
        assert_eq!(mode(&[1.0, 2.0, 3.0]), None);
        // Test variance_sample with n=1 returns NaN
        assert!(variance_sample(&[1.0]).is_nan());
    }

    #[test]
    fn correlation_and_regression() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let ys = [3.0, 5.0, 7.0, 9.0];
        let (s, i, r2) = linear_regression(&xs, &ys).unwrap();
        assert!((s - 2.0).abs() < 1e-12);
        assert!((i - 1.0).abs() < 1e-12);
        assert!((r2 - 1.0).abs() < 1e-12);
        assert!((pearson(&xs, &ys).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inference() {
        let xs: Vec<f64> = (1..=9).map(f64::from).collect();
        let (lo, hi) = mean_ci(&xs, 1.96);
        assert!(lo < 5.0 && hi > 5.0);
        assert_eq!(z_test(1.0, 1.0, 10, 1.0, 1.0, 10), 0.0);
        let g1 = [1.0, 3.0, 5.0];
        let g2 = [2.0, 3.0, 4.0];
        let (f, _p) = one_way_anova(&[&g1, &g2]);
        assert!((f - 0.0).abs() < 1e-12);
    }
}
