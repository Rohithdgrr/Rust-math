# MathVerse Statistics

[![Crates.io](https://img.shields.io/crates/v/mathverse-statistics.svg)](https://crates.io/crates/mathverse-statistics)
[![docs.rs](https://docs.rs/mathverse-statistics/badge.svg)](https://docs.rs/mathverse-statistics)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Statistical analysis toolkit: descriptive statistics, distributions, hypothesis testing, regression, multivariate analysis, and inference. Zero external dependencies.

---

## Features

- **Descriptive statistics** — mean, median, mode, variance, std dev, skewness, kurtosis, percentiles, weighted/geometric/harmonic means
- **Probability distributions** — Normal, Student-t, Chi-squared, F, Binomial, Poisson (PDF/CDF/PPF)
- **Hypothesis tests** — t-tests (one-sample, two-sample, paired, Welch's), F-test, ANOVA, χ², Mann-Whitney U, Wilcoxon
- **Regression** — polynomial (OLS), multiple linear, weighted least squares, logistic (IRLS), with R²/MSE/RMSE/MAE
- **Multivariate analysis** — covariance/correlation matrices, PCA, Mahalanobis distance
- **Inference** — bootstrap CI, effect sizes (Cohen's d, Hedges' g, η²), multiple comparison corrections (Bonferroni, BH)

## Module Overview

| Module | Purpose |
|--------|---------|
| `descriptive` | Summary statistics, percentiles, correlation, regression |
| `distributions` | Normal, t, χ², F, Binomial, Poisson (PDF/CDF/PPF) |
| `hypothesis_tests` | Parametric and non-parametric hypothesis tests |
| `regression` | Polynomial, multiple, WLS, logistic regression |
| `matrix` | Covariance, correlation, PCA, Mahalanobis, Cholesky |
| `inference` | Bootstrap, effect sizes, multiple comparison, power |

## Installation

```toml
[dependencies]
mathverse-statistics = "0.1"
```

## Quick Start

```rust
use mathverse_statistics::*;

fn main() {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    // Full summary
    let s = describe(&data);
    println!("n={}, mean={:.1}, std={:.2}, skew={:.2}", s.n, s.mean, s.std_dev, s.skewness);

    // Linear regression
    let xs = [1.0, 2.0, 3.0, 4.0];
    let ys = [3.0, 5.0, 7.0, 9.0];
    let (slope, intercept, r2) = linear_regression(&xs, &ys).unwrap();
    println!("y = {slope:.1}x + {intercept:.1}, R² = {r2:.4}");

    // Two-sample t-test
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [6.0, 7.0, 8.0, 9.0, 10.0];
    let (t, p) = t_test_two_sample(&a, &b);
    println!("t = {t:.3}, p = {p:.4}");
}
```

---

## Per-Module Documentation

### Descriptive Statistics

```rust
use mathverse_statistics::*;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
assert_eq!(percentile(&data, 50.0), 5.0);   // median
let (q1, q2, q3) = quartiles(&data);        // (3, 5, 7)

let xs = [1.0, 2.0, 3.0];
let w = [3.0, 2.0, 1.0];
println!("Weighted mean: {}", weighted_mean(&xs, &w)); // 1.667

let xs = [1.0, 2.0, 3.0, 4.0];
let ys = [3.0, 5.0, 7.0, 9.0];
println!("Pearson r = {}", pearson(&xs, &ys).unwrap()); // 1.0
```

### Distributions

```rust
use mathverse_statistics::{normal_pdf, normal_cdf, normal_ppf};

println!("φ(0) = {:.4}", normal_pdf(0.0));          // 0.3989
println!("Φ(1.96) = {:.4}", normal_cdf(1.96));      // 0.9750
println!("Φ⁻¹(0.975) = {:.4}", normal_ppf(0.975)); // 1.9600
```

### Hypothesis Tests

```rust
use mathverse_statistics::*;

// One-way ANOVA
let g1 = [1.0, 3.0, 5.0];
let g2 = [2.0, 3.0, 4.0];
let g3 = [7.0, 8.0, 9.0];
let (f, p) = one_way_anova(&[&g1, &g2, &g3]);

// Non-parametric: Mann-Whitney U
let a = [1.0, 2.0, 3.0];
let b = [4.0, 5.0, 6.0];
let (u, p) = mann_whitney_u(&a, &b);
```

### Regression

```rust
use mathverse_statistics::{polynomial_regression, logistic_regression};

// Polynomial regression
let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
let ys = [1.0, 2.0, 4.0, 8.0, 16.0];
let coeffs = polynomial_regression(&xs, &ys, 2);

// Logistic regression
let xs: Vec<&[f64]> = vec![&[1.0], &[2.0], &[3.0], &[4.0], &[5.0]];
let ys = [0.0, 0.0, 1.0, 1.0, 1.0];
let coeffs = logistic_regression(&xs, &ys, 200, 1e-8).unwrap();
```

### Multivariate Analysis

```rust
use mathverse_statistics::{pca, covariance_matrix};

let data: Vec<&[f64]> = vec![
    &[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0],
    &[4.0, 8.0], &[5.0, 10.0],
];
let result = pca(&data);
// result.explained_variance_ratio: captures max-variance directions
```

### Inference

```rust
use mathverse_statistics::{cohens_d, benjamini_hochberg, bootstrap_ci, sample_size_two_sample};

println!("Cohen's d = {:.2}", cohens_d(&a, &b));

let pvals = vec![0.01, 0.04, 0.03, 0.20];
let adjusted = benjamini_hochberg(&pvals);

let n = sample_size_two_sample(0.8, 0.05, 0.8);
println!("Need n={} per group", n); // ≈ 26
```

---

## Future Scope

- Robust regression (M-estimators, RANSAC)
- Generalized linear models (GLM)
- Mixed effects models
- Time series analysis (ARIMA, exponential smoothing)
- Permutation tests
- Multivariate hypothesis tests (Hotelling's T², MANOVA)

## License

MIT — see [LICENSE](LICENSE).
