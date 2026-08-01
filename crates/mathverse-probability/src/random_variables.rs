//! Random variable operations: sum, product, ratio, functions of random variables, convolution, mixture distributions.

use crate::rng::Rng;

/// Sum of random variables.
#[must_use]
pub struct SumOfRandomVariables;

impl SumOfRandomVariables {
    /// Convolution of two discrete distributions.
    #[must_use]
    pub fn discrete_convolution(pmf1: &[f64], pmf2: &[f64]) -> Vec<f64> {
        let n1 = pmf1.len();
        let n2 = pmf2.len();
        let mut result = vec![0.0; n1 + n2 - 1];

        for i in 0..n1 {
            for j in 0..n2 {
                result[i + j] += pmf1[i] * pmf2[j];
            }
        }

        result
    }

    /// Convolution of two continuous distributions (numerical).
    #[must_use]
    pub fn continuous_convolution(
        pdf1: impl Fn(f64) -> f64,
        pdf2: impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        n: usize,
    ) -> Vec<f64> {
        let dx = (b - a) / n as f64;
        let mut result = vec![0.0; 2 * n];

        for i in 0..2 * n {
            let x = a + i as f64 * dx;
            let mut integral = 0.0;

            for j in 0..=i {
                let y = a + j as f64 * dx;
                integral += pdf1(y) * pdf2(x - y) * dx;
            }

            result[i] = integral;
        }

        result
    }

    /// Sum of independent normal random variables.
    #[must_use]
    pub fn normal_sum(mean1: f64, var1: f64, mean2: f64, var2: f64) -> (f64, f64) {
        (mean1 + mean2, var1 + var2)
    }

    /// Sum of independent Poisson random variables.
    #[must_use]
    pub fn poisson_sum(lambda1: f64, lambda2: f64) -> f64 {
        lambda1 + lambda2
    }

    /// Sum of independent binomial random variables (same p).
    #[must_use]
    pub fn binomial_sum(n1: u64, n2: u64, p: f64) -> (u64, f64) {
        (n1 + n2, p)
    }
}

/// Product of random variables.
#[must_use]
pub struct ProductOfRandomVariables;

impl ProductOfRandomVariables {
    /// Product of independent log-normal random variables.
    #[must_use]
    pub fn log_normal_product(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> (f64, f64) {
        (mu1 + mu2, (sigma1 * sigma1 + sigma2 * sigma2).sqrt())
    }

    /// Product of independent random variables (log-transform method).
    #[must_use]
    pub fn general_product(mean1: f64, var1: f64, mean2: f64, var2: f64) -> (f64, f64) {
        // Delta method approximation
        let mean_product = mean1 * mean2;
        let var_product = mean2 * mean2 * var1 + mean1 * mean1 * var2;

        (mean_product, var_product)
    }

    /// Product of independent uniform random variables.
    #[must_use]
    pub fn uniform_product(a1: f64, b1: f64, a2: f64, b2: f64) -> (f64, f64) {
        let mean1 = (a1 + b1) / 2.0;
        let mean2 = (a2 + b2) / 2.0;
        let var1 = (b1 - a1).powi(2) / 12.0;
        let var2 = (b2 - a2).powi(2) / 12.0;

        Self::general_product(mean1, var1, mean2, var2)
    }
}

/// Ratio of random variables.
#[must_use]
pub struct RatioOfRandomVariables;

impl RatioOfRandomVariables {
    /// Ratio of independent normal random variables (Cauchy-like).
    #[must_use]
    pub fn normal_ratio(mean1: f64, sigma1: f64, mean2: f64, sigma2: f64) -> (f64, f64) {
        if mean2 != 0.0 {
            let mean_ratio = mean1 / mean2;
            let var_ratio =
                sigma1 * sigma1 / mean2.powi(2) + mean1 * mean1 * sigma2 * sigma2 / mean2.powi(4);
            (mean_ratio, var_ratio)
        } else {
            (0.0, f64::INFINITY)
        }
    }

    /// Ratio of independent random variables (delta method).
    #[must_use]
    pub fn general_ratio(mean1: f64, var1: f64, mean2: f64, var2: f64) -> (f64, f64) {
        if mean2 != 0.0 {
            let mean_ratio = mean1 / mean2;
            let var_ratio = var1 / mean2.powi(2) + mean1 * mean1 * var2 / mean2.powi(4);
            (mean_ratio, var_ratio)
        } else {
            (0.0, f64::INFINITY)
        }
    }
}

/// Functions of random variables.
#[must_use]
pub struct FunctionsOfRandomVariables;

impl FunctionsOfRandomVariables {
    /// Linear transformation: Y = aX + b.
    #[must_use]
    pub fn linear(mean: f64, variance: f64, a: f64, b: f64) -> (f64, f64) {
        (a * mean + b, a * a * variance)
    }

    /// Square transformation: Y = X².
    #[must_use]
    pub fn square(mean: f64, variance: f64) -> (f64, f64) {
        let mean_y = mean * mean + variance;
        let var_y = 4.0 * mean * mean * variance + 2.0 * variance * variance;
        (mean_y, var_y)
    }

    /// Exponential transformation: Y = exp(X).
    #[must_use]
    pub fn exponential(mean: f64, variance: f64) -> (f64, f64) {
        let mean_y = (mean + 0.5 * variance).exp();
        let var_y = (2.0 * mean + variance).exp() * ((2.0 * mean + variance).exp() - 1.0);
        (mean_y, var_y)
    }

    /// Log transformation: Y = ln(X).
    #[must_use]
    pub fn logarithm(mean: f64, variance: f64) -> (f64, f64) {
        // Delta method approximation
        let mean_y = mean.ln();
        let var_y = variance / (mean * mean);
        (mean_y, var_y)
    }

    /// General transformation using delta method.
    #[must_use]
    pub fn delta_method(
        mean: f64,
        variance: f64,
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
    ) -> (f64, f64) {
        let mean_y = f(mean);
        let var_y = df(mean) * df(mean) * variance;
        (mean_y, var_y)
    }
}

/// Mixture distributions.
#[must_use]
pub struct MixtureDistribution;

impl MixtureDistribution {
    /// Mixture of two distributions.
    #[must_use]
    pub fn two_component(weight: f64, mean1: f64, var1: f64, mean2: f64, var2: f64) -> (f64, f64) {
        let mean = weight * mean1 + (1.0 - weight) * mean2;
        let variance =
            weight * (var1 + mean1 * mean1) + (1.0 - weight) * (var2 + mean2 * mean2) - mean * mean;
        (mean, variance)
    }

    /// Mixture of multiple distributions.
    #[must_use]
    pub fn multiple_component(weights: &[f64], means: &[f64], variances: &[f64]) -> (f64, f64) {
        let mean: f64 = weights.iter().zip(means.iter()).map(|(&w, &m)| w * m).sum();

        let variance: f64 = weights
            .iter()
            .zip(means.iter())
            .zip(variances.iter())
            .map(|((&w, &m), &v)| w * (v + m * m))
            .sum::<f64>()
            - mean * mean;

        (mean, variance)
    }

    /// PDF of mixture distribution.
    #[must_use]
    pub fn pdf(x: f64, weights: &[f64], pdfs: &[Box<dyn Fn(f64) -> f64>]) -> f64 {
        weights
            .iter()
            .zip(pdfs.iter())
            .map(|(&w, pdf)| w * pdf(x))
            .sum()
    }

    /// CDF of mixture distribution.
    #[must_use]
    pub fn cdf(x: f64, weights: &[f64], cdfs: &[Box<dyn Fn(f64) -> f64>]) -> f64 {
        weights
            .iter()
            .zip(cdfs.iter())
            .map(|(&w, cdf)| w * cdf(x))
            .sum()
    }
}

/// Order statistics.
#[must_use]
pub struct OrderStatistics;

impl OrderStatistics {
    /// Distribution of k-th order statistic (simplified).
    #[must_use]
    pub fn kth_order_statistic(
        n: usize,
        k: usize,
        pdf: impl Fn(f64) -> f64 + 'static,
        cdf: impl Fn(f64) -> f64 + 'static,
    ) -> Box<dyn Fn(f64) -> f64> {
        Box::new(move |x| {
            let binom = mathverse_core::algorithms::binomial(n as u64, k as u64) as f64;
            binom * cdf(x).powi(k as i32) * (1.0 - cdf(x)).powi((n - k) as i32) * pdf(x)
        })
    }

    /// Expected value of minimum.
    #[must_use]
    pub fn expected_minimum(cdf: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let n_f = n as f64;
        let integrand = |x: f64| -> f64 {
            x * n_f * (1.0 - cdf(x)).powi((n - 1) as i32) * 1.0 // Simplified PDF
        };

        // Numerical integration
        let steps = 1000;
        let dx = (b - a) / steps as f64;
        let mut integral = 0.0;

        for i in 0..steps {
            let x = a + (i as f64 + 0.5) * dx;
            integral += integrand(x) * dx;
        }

        integral
    }

    /// Expected value of maximum.
    #[must_use]
    pub fn expected_maximum(cdf: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let n_f = n as f64;
        let integrand = |x: f64| -> f64 {
            x * n_f * cdf(x).powi((n - 1) as i32) * 1.0 // Simplified PDF
        };

        // Numerical integration
        let steps = 1000;
        let dx = (b - a) / steps as f64;
        let mut integral = 0.0;

        for i in 0..steps {
            let x = a + (i as f64 + 0.5) * dx;
            integral += integrand(x) * dx;
        }

        integral
    }

    /// Range of sample (max - min).
    #[must_use]
    pub fn sample_range(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max - min
    }

    /// Interquartile range.
    #[must_use]
    pub fn iqr(data: &mut [f64]) -> f64 {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = data.len();

        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;

        data[q3_idx] - data[q1_idx]
    }
}

/// Compound distributions.
#[must_use]
pub struct CompoundDistribution;

impl CompoundDistribution {
    /// Compound Poisson distribution (Poisson sum of i.i.d. random variables).
    #[must_use]
    pub fn poisson_compound(
        poisson_lambda: f64,
        component_mean: f64,
        component_variance: f64,
    ) -> (f64, f64) {
        let mean = poisson_lambda * component_mean;
        let variance = poisson_lambda * (component_variance + component_mean * component_mean);
        (mean, variance)
    }

    /// Compound distribution simulation.
    #[must_use]
    pub fn simulate_compound(
        n_count_distribution: impl Fn(&mut Rng) -> i64,
        component_sampler: impl Fn(&mut Rng) -> f64,
        n_compounds: usize,
        rng: &mut Rng,
    ) -> Vec<f64> {
        let mut results = Vec::new();

        for _ in 0..n_compounds {
            let n = n_count_distribution(rng);
            let mut sum = 0.0;

            for _ in 0..n.max(0) {
                sum += component_sampler(rng);
            }

            results.push(sum);
        }

        results
    }
}

/// Random variable algebra.
#[must_use]
pub struct RandomVariableAlgebra;

impl RandomVariableAlgebra {
    /// Addition of independent random variables.
    #[must_use]
    pub fn add(mean1: f64, var1: f64, mean2: f64, var2: f64) -> (f64, f64) {
        (mean1 + mean2, var1 + var2)
    }

    /// Subtraction of independent random variables.
    #[must_use]
    pub fn subtract(mean1: f64, var1: f64, mean2: f64, var2: f64) -> (f64, f64) {
        (mean1 - mean2, var1 + var2)
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scalar_multiply(mean: f64, variance: f64, scalar: f64) -> (f64, f64) {
        (scalar * mean, scalar * scalar * variance)
    }

    /// Weighted sum of independent random variables.
    #[must_use]
    pub fn weighted_sum(means: &[f64], variances: &[f64], weights: &[f64]) -> (f64, f64) {
        let mean: f64 = means.iter().zip(weights.iter()).map(|(&m, &w)| w * m).sum();

        let variance: f64 = variances
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| w * w * v)
            .sum();

        (mean, variance)
    }
}

/// Transformation of random variables.
#[must_use]
pub struct RandomVariableTransform;

impl RandomVariableTransform {
    /// Box-Cox transformation.
    #[must_use]
    pub fn box_cox(x: f64, lambda: f64) -> f64 {
        if lambda.abs() < 1e-10 {
            x.ln()
        } else {
            (x.powf(lambda) - 1.0) / lambda
        }
    }

    /// Inverse Box-Cox transformation.
    #[must_use]
    pub fn inverse_box_cox(y: f64, lambda: f64) -> f64 {
        if lambda.abs() < 1e-10 {
            y.exp()
        } else {
            (lambda * y + 1.0).powf(1.0 / lambda)
        }
    }

    /// Probability integral transform (to uniform).
    #[must_use]
    pub fn probability_integral_transform(cdf: impl Fn(f64) -> f64, x: f64) -> f64 {
        cdf(x)
    }

    /// Inverse probability integral transform (from uniform).
    #[must_use]
    pub fn inverse_probability_integral_transform(quantile: impl Fn(f64) -> f64, u: f64) -> f64 {
        quantile(u)
    }

    /// Jacobian transformation for 1D.
    #[must_use]
    pub fn jacobian_1d(
        x: f64,
        _transform: impl Fn(f64) -> f64,
        derivative: impl Fn(f64) -> f64,
    ) -> f64 {
        derivative(x).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_convolution() {
        let pmf1 = vec![0.5, 0.5];
        let pmf2 = vec![0.3, 0.7];
        let result = SumOfRandomVariables::discrete_convolution(&pmf1, &pmf2);

        assert!((result.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_normal_sum() {
        let (mean, var) = SumOfRandomVariables::normal_sum(1.0, 2.0, 3.0, 4.0);
        assert!((mean - 4.0).abs() < 1e-10);
        assert!((var - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_transformation() {
        let (mean, var) = FunctionsOfRandomVariables::linear(5.0, 2.0, 2.0, 3.0);
        assert!((mean - 13.0).abs() < 1e-10);
        assert!((var - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_mixture_distribution() {
        let (mean, var) = MixtureDistribution::two_component(0.5, 0.0, 1.0, 10.0, 4.0);
        assert!((mean - 5.0).abs() < 1e-10);
        assert!(var > 0.0);
    }

    #[test]
    fn test_random_variable_algebra() {
        let (mean, var) = RandomVariableAlgebra::add(1.0, 2.0, 3.0, 4.0);
        assert!((mean - 4.0).abs() < 1e-10);
        assert!((var - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_box_cox() {
        let transformed = RandomVariableTransform::box_cox(2.0, 0.5);
        assert!(transformed > 0.0);
    }

    #[test]
    fn test_sample_range() {
        let data = vec![1.0, 5.0, 3.0, 9.0, 2.0];
        let range = OrderStatistics::sample_range(&data);
        assert!((range - 8.0).abs() < 1e-10);
    }
}
