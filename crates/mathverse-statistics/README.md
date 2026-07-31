# mathverse-statistics

Statistical analysis toolkit: descriptive statistics, probability distributions, hypothesis testing, regression, multivariate analysis, and statistical inference. Zero external dependencies.

## Features

- **Descriptive statistics**: mean, median, mode, variance, std dev, skewness, kurtosis, percentiles, weighted/geometric/harmonic means, trimmed/winsorized means
- **Probability distributions**: Normal, Student-t, Chi-squared, F, Binomial, Poisson with PDF, CDF, and PPF (inverse CDF)
- **Hypothesis tests**: t-tests (one-sample, two-sample, paired, Welch's), F-test, ANOVA, χ² goodness-of-fit, χ² independence, binomial test, Mann-Whitney U, Wilcoxon signed-rank
- **Regression**: polynomial (OLS), multiple linear, weighted least squares, logistic (IRLS), with R², MSE, RMSE, MAE
- **Multivariate analysis**: covariance/correlation matrices, PCA, Mahalanobis distance, precision matrix
- **Statistical inference**: bootstrap CI, effect sizes (Cohen's d, Hedges' g, η², ω²), multiple comparison corrections (Bonferroni, Šidák, Holm, BH), power analysis, sample size calculation

## Module Overview

| Module               | Purpose                                                  |
|----------------------|----------------------------------------------------------|
| `descriptive`        | Summary statistics, percentiles, correlation, regression  |
| `distributions`      | Normal, t, χ², F, Binomial, Poisson (PDF/CDF/PPF)       |
| `hypothesis_tests`   | Parametric and non-parametric hypothesis tests            |
| `regression`         | Polynomial, multiple, WLS, logistic regression            |
| `matrix`             | Covariance, correlation, PCA, Mahalanobis, Cholesky      |
| `inference`          | Bootstrap, effect sizes, multiple comparison, power       |

## Installation

```toml
[dependencies]
mathverse-statistics = { path = "../mathverse-statistics" }
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
    let (slope, intercept, r2) = linear_regression(&xs, &ys);
    println!("y = {slope:.1}x + {intercept:.1}, R² = {r2:.4}");

    // Two-sample t-test
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [6.0, 7.0, 8.0, 9.0, 10.0];
    let (t, p) = t_test_two_sample(&a, &b);
    println!("t = {t:.3}, p = {p:.4}");
}
```

---

## `descriptive` — Summary Statistics

```
         ┌─────────────────────────────────────────┐
         │           Box-and-Whisker Plot           │
         │                                          │
         │  min    Q1    median    Q3    max        │
         │  ├──────┤──────┼──────┤──────┤          │
         │  ├──────┤══════════════┤──────┤          │
         │              IQR                          │
         └─────────────────────────────────────────┘
```

### Core Functions

| Function              | Formula / Description                              |
|-----------------------|----------------------------------------------------|
| `mean(xs)`            | Σxᵢ / n                                           |
| `median(xs)`          | Middle value (or average of two middle)            |
| `mode(xs)`            | Most frequent value                                |
| `variance_sample(xs)` | Σ(xᵢ - x̄)² / (n-1)                              |
| `variance_pop(xs)`    | Σ(xᵢ - x̄)² / n                                  |
| `std_dev_sample(xs)`  | √variance_sample                                   |
| `std_dev_pop(xs)`     | √variance_pop                                      |
| `skewness(xs)`        | Adjusted Fisher-Pearson G1                         |
| `kurtosis(xs)`        | Excess kurtosis (Fisher G2)                        |
| `standard_error(xs)`  | s / √n                                             |
| `range(xs)`           | max - min                                           |
| `iqr(xs)`             | Q3 - Q1                                             |
| `mad(xs)`             | Median absolute deviation                           |
| `coefficient_of_variation(xs)` | s / |x̄|                                    |

### Percentiles & Quantiles

```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
assert_eq!(percentile(&data, 50.0), 5.0);   // median
assert_eq!(quantile(&data, 0.25), 3.0);     // Q1
let (q1, q2, q3) = quartiles(&data);        // (3, 5, 7)
```

### Weighted & Robust Means

```rust
let xs = [1.0, 2.0, 3.0];
let w = [3.0, 2.0, 1.0];
println!("Weighted mean: {}", weighted_mean(&xs, &w)); // 1.667

println!("Geometric mean: {}", geometric_mean(&[2.0, 8.0])); // 4.0
println!("Harmonic mean: {}", harmonic_mean(&[2.0, 8.0]));   // 3.2

let data = vec![1.0, 1.0, 2.0, 3.0, 100.0];
println!("Trimmed mean (20%): {}", trimmed_mean(&data, 0.2)); // 2.0
println!("Winsorized mean: {}", winsorized_mean(&data, 0.2)); // 2.0
```

### Correlation & Regression

```rust
let xs = [1.0, 2.0, 3.0, 4.0];
let ys = [3.0, 5.0, 7.0, 9.0];

println!("Pearson r = {}", pearson(&xs, &ys));          // 1.0
println!("Covariance = {}", covariance(&xs, &ys));      // 4.0

let (slope, intercept, r2) = linear_regression(&xs, &ys);
// slope=2, intercept=1, r²=1.0 (perfect fit)
```

### `describe()` — Full Summary

```rust
let s = describe(&data);
// s.n, s.mean, s.std_dev, s.min, s.q1, s.median, s.q3, s.max, s.skewness, s.kurtosis
```

---

## `distributions` — Probability Distributions

```
  Normal Distribution     Student-t Distribution

      ╱████╲                 ╱████╲
    ╱████████╲             ╱██████████╲
  ╱████████████╲         ╱██████████████╲
╱████████████████╲     ╱████████████████████╲
───────────────────   ──────────────────────────
  thin tails            heavier tails
  approaches t as df→∞
```

### Normal Distribution

| Method      | Description                           |
|-------------|---------------------------------------|
| `pdf(x)`    | φ(x) = e^(-x²/2) / √(2π)            |
| `cdf(x)`    | Φ(x) via Abramowitz-Stegun erf       |
| `ppf(p)`    | Φ⁻¹(p) via rational approximation    |

```rust
use mathverse_statistics::{normal_pdf, normal_cdf, normal_ppf};

println!("φ(0) = {:.4}", normal_pdf(0.0));      // 0.3989
println!("Φ(1.96) = {:.4}", normal_cdf(1.96));   // 0.9750
println!("Φ⁻¹(0.975) = {:.4}", normal_ppf(0.975)); // 1.9600
```

### Student-t Distribution

```rust
use mathverse_statistics::{student_t_cdf, student_t_ppf};

// CDF at 0 is always 0.5
assert!((student_t_cdf(0.0, 10.0) - 0.5).abs() < 1e-6);

// t with large df ≈ normal
assert!((student_t_cdf(1.96, 1000.0) - normal_cdf(1.96)).abs() < 0.05);
```

### All Distributions

| Distribution   | Functions                                  |
|----------------|---------------------------------------------|
| `Normal`       | `pdf`, `cdf`, `ppf`                        |
| `StudentT`     | `pdf(t, df)`, `cdf(t, df)`, `ppf(p, df)`  |
| `ChiSquared`   | `pdf(x, k)`, `cdf(x, k)`, `ppf(p, k)`    |
| `FDist`        | `pdf(x, d1, d2)`, `cdf`, `ppf`            |
| `Binomial`     | `pmf(k, n, p)`, `cdf(k, n, p)`, `mean`, `variance` |
| `Poisson`      | `pmf(k, λ)`, `cdf(k, λ)`, `mean`, `variance` |

---

## `hypothesis_tests` — Statistical Tests

```
┌────────────────────────────────────────────────────────────┐
│                    Hypothesis Testing Flow                  │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  State   │    │  Compute │    │ Compare  │             │
│  │   H₀     │───▶│  stat    │───▶│ to α     │             │
│  │   H₁     │    │  & p     │    │          │             │
│  └──────────┘    └──────────┘    └────┬─────┘             │
│                                        │                    │
│                              ┌─────────┴─────────┐        │
│                              │                     │        │
│                         p < α                 p ≥ α       │
│                         REJECT                RETAIN       │
│                         H₀                    H₀          │
└────────────────────────────────────────────────────────────┘
```

### Parametric Tests

| Test                      | Function                     | Returns              |
|---------------------------|------------------------------|----------------------|
| Two-sample t-test         | `t_test_two_sample(a, b)`    | (t, p)               |
| Welch's t-test            | `welch_t_test(a, b)`         | (t, df, p)           |
| Paired t-test             | `paired_t_test(a, b)`        | (t, p)               |
| One-sample t-test         | `one_sample_t_test(xs, μ₀)`  | (t, p)               |
| F-test (variance)         | `f_test_variance(a, b)`      | (F, p)               |
| One-way ANOVA             | `one_way_anova(groups)`      | (F, p)               |
| χ² goodness-of-fit        | `chi_squared_gof(obs, exp)`  | (χ², p)              |
| χ² independence           | `chi_squared_independence(T)`| (χ², p, df)          |
| Binomial test             | `binomial_test(k, n, p₀)`    | p                    |

```rust
// Two-sample t-test
let a = [1.0, 2.0, 3.0, 4.0, 5.0];
let b = [6.0, 7.0, 8.0, 9.0, 10.0];
let (t, p) = t_test_two_sample(&a, &b);
// t ≈ -5, p < 0.01 → strongly different means

// One-way ANOVA
let g1 = [1.0, 3.0, 5.0];
let g2 = [2.0, 3.0, 4.0];
let g3 = [7.0, 8.0, 9.0];
let (f, p) = one_way_anova(&[&g1, &g2, &g3]);
```

### Non-Parametric Tests

| Test                   | Function                       | Returns     |
|------------------------|--------------------------------|-------------|
| Mann-Whitney U         | `mann_whitney_u(a, b)`         | (U, p)      |
| Wilcoxon signed-rank   | `wilcoxon_signed_rank(a, b)`   | (W, p)      |

```rust
let a = [1.0, 2.0, 3.0];
let b = [4.0, 5.0, 6.0];
let (u, p) = mann_whitney_u(&a, &b);
// Non-parametric alternative to t-test
```

---

## `regression` — Curve Fitting

### Polynomial Regression

```
  y
  ▲     ●
  │   ●   ●         fit: y = β₀ + β₁x + β₂x² + ...
  │ ●       ●
  │●         ●
  └──────────────▶ x
```

```rust
use mathverse_statistics::{polynomial_regression, predict_poly};

let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
let ys = [1.0, 2.0, 4.0, 8.0, 16.0]; // y = 2^x ≈ quadratic-ish
let coeffs = polynomial_regression(&xs, &ys, 2);
// coeffs = [β₀, β₁, β₂]
let y_pred = predict_poly(2.5, &coeffs);
```

### Multiple Linear Regression

```rust
use mathverse_statistics::multiple_regression;

let xs: Vec<&[f64]> = vec![&[1.0, 2.0], &[2.0, 1.0], &[3.0, 4.0]];
let ys = [7.0, 8.0, 17.0]; // y = 3x₁ + 2x₂
let coeffs = multiple_regression(&xs, &ys);
// coeffs[1] ≈ 3.0, coeffs[2] ≈ 2.0
```

### Logistic Regression

```rust
use mathverse_statistics::{logistic_regression, sigmoid};

let xs: Vec<&[f64]> = vec![&[1.0], &[2.0], &[3.0], &[4.0], &[5.0]];
let ys = [0.0, 0.0, 1.0, 1.0, 1.0];
let coeffs = logistic_regression(&xs, &ys, 200, 0.1);
let prob = sigmoid(coeffs[0] + coeffs[1] * 3.5); // P(y=1 | x=3.5)
```

### Model Evaluation

| Function       | Formula                        |
|----------------|--------------------------------|
| `r_squared`    | 1 - SSR/SST                    |
| `mse`          | Σ(yᵢ - ŷᵢ)² / n              |
| `rmse`         | √MSE                           |
| `mae`          | Σ|yᵢ - ŷᵢ| / n               |
| `residuals`    | yᵢ - ŷᵢ                       |

---

## `matrix` — Multivariate Analysis

### Covariance & Correlation

```rust
use mathverse_statistics::{covariance_matrix, correlation_matrix};

let data: Vec<&[f64]> = vec![&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]];
let cov = covariance_matrix(&data);    // p×p covariance matrix
let corr = correlation_matrix(&data);  // p×p correlation matrix
```

### Principal Component Analysis

```
  PCA: Find directions of maximum variance

  ●  ●          PC1 ═══════════▶  (max variance)
  ●  ●   ●
     ●  ●        PC2 ════▶        (orthogonal)
       ●

  data → center → covariance → eigendecomposition → project
```

```rust
use mathverse_statistics::{pca, pca_transform};

let data: Vec<&[f64]> = vec![
    &[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0],
    &[4.0, 8.0], &[5.0, 10.0],
];
let result = pca(&data);
// result.components: eigenvectors
// result.explained_variance_ratio: [0.999, 0.001]
// First PC captures 99.9% of variance (data is nearly collinear)
```

### Mahalanobis Distance

```rust
use mathverse_statistics::{mahalanobis, cholesky_inverse};

let mean = vec![0.0, 0.0];
let cov_inv = cholesky_inverse(&covariance_matrix(&data)).unwrap();
let d = mahalanobis(&[3.0, 4.0], &mean, &cov_inv);
```

---

## `inference` — Effect Sizes, Power, Corrections

### Effect Sizes

| Measure          | Function           | Interpretation            |
|------------------|--------------------|---------------------------|
| Cohen's d        | `cohens_d(a, b)`   | 0.2=small, 0.5=med, 0.8=large |
| Hedges' g        | `hedges_g(a, b)`   | Bias-corrected d          |
| Eta-squared      | `eta_squared(g)`   | Proportion of variance (ANOVA) |
| Omega-squared    | `omega_squared(g)` | Less biased η²            |

```rust
let a = [1.0, 2.0, 3.0];
let b = [4.0, 5.0, 6.0];
println!("Cohen's d = {:.2}", cohens_d(&a, &b)); // ≈ -3.67 (large)
```

### Multiple Comparison Corrections

| Method              | Function                | Controls        |
|---------------------|-------------------------|-----------------|
| Bonferroni          | `bonferroni(α, m)`      | FWER            |
| Šidák               | `sidak(α, m)`           | FWER            |
| Holm-Bonferroni     | `holm_bonferroni(pvals)` | FWER (step-down) |
| Benjamini-Hochberg   | `benjamini_hochberg(pvals)` | FDR          |

```rust
let pvals = vec![0.01, 0.04, 0.03, 0.20];
let adjusted = benjamini_hochberg(&pvals);
// adjusted[i] are FDR-adjusted p-values
```

### Power Analysis

```rust
use mathverse_statistics::{power_two_sample, sample_size_two_sample};

// Power of two-sample t-test with d=0.8, n=30 per group
let p = power_two_sample(0.8, 30, 0.05);
println!("Power: {:.1}%", p * 100.0); // ≈ 86%

// Sample size needed for 80% power
let n = sample_size_two_sample(0.8, 0.05, 0.8);
println!("Need n={} per group", n); // ≈ 26
```

### Bootstrap Confidence Intervals

```rust
use mathverse_statistics::bootstrap_ci;

let data = [1.0, 2.0, 3.0, 4.0, 5.0];
let (lo, hi) = bootstrap_ci(&data, |d| mean(d), 1000, 0.05);
// 95% CI for the mean
```

---

## Hypothesis Testing Decision Tree

```
                    ┌───────────────┐
                    │ Normal data?  │
                    └───────┬───────┘
                      yes   │   no
                ┌───────────┴──────────┐
                │                      │
         ┌──────┴──────┐        ┌──────┴──────┐
         │ Known σ?    │        │ Paired?     │
         └──────┬──────┘        └──────┬──────┘
           yes  │  no             yes  │  no
         ┌──────┴──────┐        ┌──────┴──────┐
         │ Z-test      │        │ Wilcoxon    │
         └─────────────┘        │ signed-rank │
                │               └─────────────┘
         ┌──────┴──────┐              │
         │ Equal var?  │              │
         └──────┬──────┘              │
           yes  │  no                 │
         ┌──────┴──────┐              │
         │ t-test      │              │
         │ (pooled)    │              │
         └─────────────┘              │
                │               ┌──────┴──────┐
         ┌──────┴──────┐       │ Mann-Whitney │
         │ >2 groups?  │       └─────────────┘
         └──────┬──────┘
           yes  │  no
         ┌──────┴──────┐
         │ ANOVA       │
         └─────────────┘
```

---

## Future Scope

- Robust regression (M-estimators, RANSAC)
- Generalized linear models (GLM)
- Mixed effects models
- Bayesian estimation and posterior sampling
- Time series analysis (ARIMA, exponential smoothing)
- Non-parametric density estimation (KDE)
- Permutation tests
- Sequential analysis
- Multivariate hypothesis tests (Hotelling's T², MANOVA)
- Structural equation modeling
- Causal inference tools

## License

MIT OR Apache-2.0
