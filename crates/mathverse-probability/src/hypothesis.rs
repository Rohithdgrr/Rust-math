//! Hypothesis testing: Type I/II errors, power analysis, p-values, likelihood ratio tests, multiple testing.

use crate::distributions::ContinuousDist;

/// Hypothesis test result.
#[must_use]
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub reject_null: bool,
    pub confidence_level: f64,
}

impl TestResult {
    #[must_use]
    pub fn new(test_statistic: f64, p_value: f64, alpha: f64) -> Self {
        TestResult {
            test_statistic,
            p_value,
            reject_null: p_value < alpha,
            confidence_level: 1.0 - alpha,
        }
    }
}

/// Z-test for known population variance.
#[must_use]
pub struct ZTest;

impl ZTest {
    /// One-sample Z-test.
    #[must_use]
    pub fn one_sample(
        sample_mean: f64,
        population_mean: f64,
        population_std: f64,
        sample_size: usize,
        alternative: AlternativeHypothesis,
    ) -> TestResult {
        let z = (sample_mean - population_mean) / (population_std / (sample_size as f64).sqrt());
        let p_value = Self::p_value(z, alternative);

        TestResult::new(z, p_value, 0.05)
    }

    /// Two-sample Z-test.
    #[must_use]
    pub fn two_sample(
        mean1: f64,
        mean2: f64,
        std1: f64,
        std2: f64,
        n1: usize,
        n2: usize,
        alternative: AlternativeHypothesis,
    ) -> TestResult {
        let se = (std1 * std1 / n1 as f64 + std2 * std2 / n2 as f64).sqrt();
        let z = (mean1 - mean2) / se;
        let p_value = Self::p_value(z, alternative);

        TestResult::new(z, p_value, 0.05)
    }

    fn p_value(z: f64, alternative: AlternativeHypothesis) -> f64 {
        let normal_cdf = |x: f64| -> f64 {
            0.5 * (1.0 + crate::distributions::erf(x / core::f64::consts::SQRT_2))
        };

        match alternative {
            AlternativeHypothesis::TwoSided => 2.0 * (1.0 - normal_cdf(z.abs())),
            AlternativeHypothesis::Greater => 1.0 - normal_cdf(z),
            AlternativeHypothesis::Less => normal_cdf(z),
        }
    }
}

/// T-test for unknown population variance.
#[must_use]
pub struct TTest;

impl TTest {
    /// One-sample T-test.
    #[must_use]
    pub fn one_sample(
        sample_mean: f64,
        sample_std: f64,
        population_mean: f64,
        sample_size: usize,
        alternative: AlternativeHypothesis,
    ) -> TestResult {
        let t = (sample_mean - population_mean) / (sample_std / (sample_size as f64).sqrt());
        let df = sample_size - 1;
        let p_value = Self::p_value(t, df, alternative);

        TestResult::new(t, p_value, 0.05)
    }

    /// Two-sample T-test (equal variances).
    #[must_use]
    pub fn two_sample_equal_var(
        mean1: f64,
        mean2: f64,
        std1: f64,
        std2: f64,
        n1: usize,
        n2: usize,
        alternative: AlternativeHypothesis,
    ) -> TestResult {
        let pooled_std =
            ((n1 - 1) as f64 * std1 * std1 + (n2 - 1) as f64 * std2 * std2) / (n1 + n2 - 2) as f64;
        let se = pooled_std.sqrt() * (1.0 / n1 as f64 + 1.0 / n2 as f64).sqrt();
        let t = (mean1 - mean2) / se;
        let df = n1 + n2 - 2;
        let p_value = Self::p_value(t, df, alternative);

        TestResult::new(t, p_value, 0.05)
    }

    /// Two-sample T-test (unequal variances, Welch's t-test).
    #[must_use]
    pub fn two_sample_unequal_var(
        mean1: f64,
        mean2: f64,
        std1: f64,
        std2: f64,
        n1: usize,
        n2: usize,
        alternative: AlternativeHypothesis,
    ) -> TestResult {
        let se = (std1 * std1 / n1 as f64 + std2 * std2 / n2 as f64).sqrt();
        let t = (mean1 - mean2) / se;

        // Welch-Satterthwaite degrees of freedom
        let df = (std1 * std1 / n1 as f64 + std2 * std2 / n2 as f64).powi(2)
            / ((std1 * std1 / n1 as f64).powi(2) / (n1 - 1) as f64
                + (std2 * std2 / n2 as f64).powi(2) / (n2 - 1) as f64);

        let p_value = Self::p_value(t, df as usize, alternative);

        TestResult::new(t, p_value, 0.05)
    }

    fn p_value(t: f64, df: usize, alternative: AlternativeHypothesis) -> f64 {
        // Approximate t-distribution CDF
        let t_cdf = |x: f64, df: f64| -> f64 {
            if df > 100.0 {
                // Use normal approximation for large df
                0.5 * (1.0 + crate::distributions::erf(x / core::f64::consts::SQRT_2))
            } else {
                // Simplified approximation
                let beta = crate::distributions::Beta {
                    alpha: df / 2.0,
                    beta: 0.5,
                };
                let z = df / (df + x * x);
                if x >= 0.0 {
                    1.0 - 0.5 * beta.cdf(z)
                } else {
                    0.5 * beta.cdf(z)
                }
            }
        };

        match alternative {
            AlternativeHypothesis::TwoSided => 2.0 * (1.0 - t_cdf(t.abs(), df as f64)),
            AlternativeHypothesis::Greater => 1.0 - t_cdf(t, df as f64),
            AlternativeHypothesis::Less => t_cdf(t, df as f64),
        }
    }
}

/// Chi-squared test.
#[must_use]
pub struct ChiSquaredTest;

impl ChiSquaredTest {
    /// Chi-squared goodness of fit test.
    #[must_use]
    pub fn goodness_of_fit(observed: &[f64], expected: &[f64]) -> TestResult {
        if observed.len() != expected.len() {
            return TestResult::new(f64::NAN, 1.0, 0.05);
        }

        let mut chi_sq = 0.0;
        for (&o, &e) in observed.iter().zip(expected.iter()) {
            if e > 0.0 {
                chi_sq += (o - e) * (o - e) / e;
            }
        }

        let df = observed.len() - 1;
        let p_value = Self::p_value(chi_sq, df);

        TestResult::new(chi_sq, p_value, 0.05)
    }

    /// Chi-squared test of independence.
    #[must_use]
    pub fn independence(contingency: &[Vec<f64>]) -> TestResult {
        let n_rows = contingency.len();
        let n_cols = contingency[0].len();

        // Compute row and column totals
        let mut row_totals = vec![0.0; n_rows];
        let mut col_totals = vec![0.0; n_cols];
        let mut total = 0.0;

        for i in 0..n_rows {
            for j in 0..n_cols {
                row_totals[i] += contingency[i][j];
                col_totals[j] += contingency[i][j];
                total += contingency[i][j];
            }
        }

        // Compute chi-squared statistic
        let mut chi_sq = 0.0;
        for i in 0..n_rows {
            for j in 0..n_cols {
                let expected = row_totals[i] * col_totals[j] / total;
                if expected > 0.0 {
                    chi_sq += (contingency[i][j] - expected).powi(2) / expected;
                }
            }
        }

        let df = (n_rows - 1) * (n_cols - 1);
        let p_value = Self::p_value(chi_sq, df);

        TestResult::new(chi_sq, p_value, 0.05)
    }

    fn p_value(chi_sq: f64, df: usize) -> f64 {
        // Approximate chi-squared CDF
        let chi_sq_cdf = |x: f64, df: f64| -> f64 {
            let k = df / 2.0;
            let gamma = crate::distributions::Gamma {
                shape: k,
                rate: 0.5,
            };
            gamma.cdf(x)
        };

        1.0 - chi_sq_cdf(chi_sq, df as f64)
    }
}

/// F-test.
#[must_use]
pub struct FTest;

impl FTest {
    /// F-test for equality of variances.
    #[must_use]
    pub fn variance_equality(var1: f64, var2: f64, n1: usize, n2: usize) -> TestResult {
        let f = if var1 > var2 {
            var1 / var2
        } else {
            var2 / var1
        };
        let df1 = n1 - 1;
        let df2 = n2 - 1;
        let p_value = Self::p_value(f, df1, df2);

        TestResult::new(f, p_value, 0.05)
    }

    fn p_value(f: f64, df1: usize, df2: usize) -> f64 {
        // Approximate F-distribution CDF
        let f_cdf = |x: f64, d1: f64, d2: f64| -> f64 {
            let beta = crate::distributions::Beta {
                alpha: d1 / 2.0,
                beta: d2 / 2.0,
            };
            let z = d1 * x / (d1 * x + d2);
            beta.cdf(z)
        };

        1.0_f64 - f_cdf(f, df1 as f64, df2 as f64)
    }
}

/// Likelihood ratio test.
#[must_use]
pub struct LikelihoodRatioTest;

impl LikelihoodRatioTest {
    /// Likelihood ratio test statistic.
    #[must_use]
    pub fn test_statistic(
        log_likelihood_null: f64,
        log_likelihood_alt: f64,
        df: usize,
    ) -> TestResult {
        let lr_stat = 2.0 * (log_likelihood_alt - log_likelihood_null);
        let p_value = ChiSquaredTest::p_value(lr_stat, df);

        TestResult::new(lr_stat, p_value, 0.05)
    }
}

/// Alternative hypothesis type.
pub enum AlternativeHypothesis {
    TwoSided,
    Greater,
    Less,
}

/// Power analysis.
#[must_use]
pub struct PowerAnalysis;

impl PowerAnalysis {
    /// Calculate power of a test.
    #[must_use]
    pub fn calculate_power(
        effect_size: f64,
        sample_size: usize,
        alpha: f64,
        test_type: TestType,
    ) -> f64 {
        match test_type {
            TestType::ZTest => {
                let z_alpha = Self::critical_z(alpha);
                let z_beta = effect_size * (sample_size as f64).sqrt() - z_alpha;
                let normal_cdf = |x: f64| -> f64 {
                    0.5 * (1.0 + crate::distributions::erf(x / core::f64::consts::SQRT_2))
                };
                normal_cdf(z_beta)
            }
            TestType::TTest => {
                // Simplified approximation
                let z_alpha = Self::critical_z(alpha);
                let z_beta = effect_size * (sample_size as f64).sqrt() - z_alpha;
                let normal_cdf = |x: f64| -> f64 {
                    0.5 * (1.0 + crate::distributions::erf(x / core::f64::consts::SQRT_2))
                };
                normal_cdf(z_beta)
            }
        }
    }

    /// Required sample size for desired power.
    #[must_use]
    pub fn required_sample_size(
        effect_size: f64,
        alpha: f64,
        desired_power: f64,
        _test_type: TestType,
    ) -> usize {
        let z_alpha = Self::critical_z(alpha);
        let z_beta = Self::critical_z(1.0 - desired_power);
        let n = ((z_alpha + z_beta) / effect_size).powi(2).ceil() as usize;
        n.max(2)
    }

    fn critical_z(alpha: f64) -> f64 {
        // Approximate critical z-value for two-tailed test
        let normal_cdf = |x: f64| -> f64 {
            0.5 * (1.0 + crate::distributions::erf(x / core::f64::consts::SQRT_2))
        };

        // Binary search for critical value
        let mut low = 0.0;
        let mut high = 5.0;
        for _ in 0..50 {
            let mid = (low + high) / 2.0;
            if 1.0 - normal_cdf(mid) < alpha / 2.0 {
                low = mid;
            } else {
                high = mid;
            }
        }
        (low + high) / 2.0
    }
}

/// Test type for power analysis.
pub enum TestType {
    ZTest,
    TTest,
}

/// Multiple testing correction.
#[must_use]
pub struct MultipleTesting;

impl MultipleTesting {
    /// Bonferroni correction.
    #[must_use]
    pub fn bonferroni(p_values: &[f64], alpha: f64) -> Vec<bool> {
        let corrected_alpha = alpha / p_values.len() as f64;
        p_values.iter().map(|&p| p < corrected_alpha).collect()
    }

    /// Holm-Bonferroni correction (step-down).
    #[must_use]
    pub fn holm_bonferroni(p_values: &[f64], alpha: f64) -> Vec<bool> {
        let n = p_values.len();
        let mut indexed: Vec<(usize, f64)> = p_values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut rejected = vec![false; n];
        let mut alpha_i = alpha;

        for (i, (idx, p)) in indexed.iter().enumerate() {
            if *p < alpha_i {
                rejected[*idx] = true;
            } else {
                break;
            }
            alpha_i = alpha / (n - i - 1) as f64;
        }

        rejected
    }

    /// Benjamini-Hochberg procedure (FDR control).
    #[must_use]
    pub fn benjamini_hochberg(p_values: &[f64], q: f64) -> Vec<bool> {
        let n = p_values.len();
        let mut indexed: Vec<(usize, f64)> = p_values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut rejected = vec![false; n];
        let mut max_idx = None;

        for (i, (_idx, p)) in indexed.iter().enumerate() {
            let threshold = (i + 1) as f64 * q / n as f64;
            if *p <= threshold {
                max_idx = Some(i);
            }
        }

        if let Some(max_i) = max_idx {
            for (idx, _) in indexed.iter().take(max_i + 1) {
                rejected[*idx] = true;
            }
        }

        rejected
    }

    /// False discovery rate estimation.
    #[must_use]
    pub fn estimate_fdr(p_values: &[f64], threshold: f64) -> f64 {
        let n = p_values.len();
        let significant = p_values.iter().filter(|&&p| p < threshold).count();

        if significant == 0 {
            return 0.0;
        }

        let expected_false = threshold * n as f64;
        expected_false / significant as f64
    }
}

/// Neyman-Pearson lemma.
#[must_use]
pub struct NeymanPearson;

impl NeymanPearson {
    /// Most powerful test for simple hypotheses.
    #[must_use]
    pub fn most_powerful_test(likelihood_ratio: f64, threshold: f64) -> bool {
        likelihood_ratio > threshold
    }

    /// Likelihood ratio for simple hypotheses.
    #[must_use]
    pub fn likelihood_ratio(
        x: f64,
        null_pdf: impl Fn(f64) -> f64,
        alt_pdf: impl Fn(f64) -> f64,
    ) -> f64 {
        let p_null = null_pdf(x);
        let p_alt = alt_pdf(x);

        if p_null > 0.0 {
            p_alt / p_null
        } else {
            f64::INFINITY
        }
    }
}

/// Sequential probability ratio test (SPRT).
#[must_use]
pub struct SPRT;

impl SPRT {
    /// Wald's SPRT.
    #[must_use]
    pub fn test(log_likelihood_ratio: f64, alpha: f64, beta: f64) -> SPRTDecision {
        let a = (beta / (1.0 - alpha)).ln();
        let b = ((1.0 - beta) / alpha).ln();

        if log_likelihood_ratio >= b {
            SPRTDecision::RejectNull
        } else if log_likelihood_ratio <= a {
            SPRTDecision::AcceptNull
        } else {
            SPRTDecision::Continue
        }
    }
}

/// SPRT decision.
pub enum SPRTDecision {
    AcceptNull,
    RejectNull,
    Continue,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z_test() {
        let result = ZTest::one_sample(5.0, 4.0, 1.0, 100, AlternativeHypothesis::Greater);
        assert!(result.reject_null);
    }

    #[test]
    fn test_t_test() {
        let result = TTest::one_sample(5.0, 1.0, 4.0, 10, AlternativeHypothesis::TwoSided);
        assert!(result.test_statistic > 0.0);
    }

    #[test]
    fn test_chi_squared_test() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![15.0, 15.0, 30.0];
        let result = ChiSquaredTest::goodness_of_fit(&observed, &expected);
        assert!(result.test_statistic >= 0.0);
    }

    #[test]
    fn test_bonferroni() {
        let p_values = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let rejected = MultipleTesting::bonferroni(&p_values, 0.05);
        assert_eq!(rejected.len(), 5);
    }

    #[test]
    fn test_power_analysis() {
        let power = PowerAnalysis::calculate_power(0.5, 100, 0.05, TestType::ZTest);
        assert!(power > 0.5);
    }

    #[test]
    fn test_sprt() {
        let decision = SPRT::test(3.5, 0.05, 0.1);
        assert!(matches!(decision, SPRTDecision::RejectNull));
    }
}
