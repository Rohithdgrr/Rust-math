//! Probability inequalities: Chebyshev, Chernoff, Hoeffding, Markov, Jensen, Azuma, concentration.

/// Markov's inequality: P(X ≥ a) ≤ E[X] / a for non-negative X.
pub struct MarkovInequality;

impl MarkovInequality {
    /// Upper bound on tail probability.
    pub fn bound(expected_value: f64, threshold: f64) -> f64 {
        if threshold <= 0.0 {
            return 1.0;
        }
        (expected_value / threshold).min(1.0)
    }

    /// Generalized Markov inequality: P(|X| ≥ a) ≤ E[|X|^p] / a^p.
    pub fn generalized_bound(p_moment: f64, threshold: f64, p: f64) -> f64 {
        if threshold <= 0.0 || p <= 0.0 {
            return 1.0;
        }
        (p_moment / threshold.powf(p)).min(1.0)
    }
}

/// Chebyshev's inequality: P(|X - μ| ≥ kσ) ≤ 1/k².
pub struct ChebyshevInequality;

impl ChebyshevInequality {
    /// Upper bound on deviation from mean.
    pub fn bound(variance: f64, k: f64) -> f64 {
        if k <= 0.0 {
            return 1.0;
        }
        (variance / (k * k)).min(1.0)
    }

    /// One-sided Chebyshev (Cantelli's inequality).
    pub fn one_sided_bound(variance: f64, deviation: f64) -> f64 {
        if deviation <= 0.0 {
            return 1.0;
        }
        let t = deviation / variance.sqrt();
        (1.0 / (1.0 + t * t)).min(1.0)
    }

    /// Chebyshev for sample mean.
    pub fn sample_mean_bound(variance: f64, n: usize, k: f64) -> f64 {
        let sample_variance = variance / n as f64;
        Self::bound(sample_variance, k)
    }
}

/// Chernoff bounds: exponential concentration inequalities.
pub struct ChernoffBound;

impl ChernoffBound {
    /// Chernoff bound for sum of Bernoulli random variables.
    pub fn bernoulli_bound(p: f64, n: usize, delta: f64) -> f64 {
        let mu = p * n as f64;
        let x = (1.0 + delta) * mu;
        
        if delta <= 0.0 {
            return 1.0;
        }
        
        // Upper tail: P(X ≥ (1+δ)μ) ≤ exp(-μδ²/(2+δ))
        let bound = (-mu * delta * delta / (2.0 + delta)).exp();
        bound.min(1.0)
    }

    /// Lower tail Chernoff bound.
    pub fn bernoulli_lower_bound(p: f64, n: usize, delta: f64) -> f64 {
        let mu = p * n as f64;
        
        if delta <= 0.0 || delta >= 1.0 {
            return 1.0;
        }
        
        // Lower tail: P(X ≤ (1-δ)μ) ≤ exp(-μδ²/2)
        let bound = (-mu * delta * delta / 2.0).exp();
        bound.min(1.0)
    }

    /// General Chernoff bound using moment generating function.
    pub fn general_bound(
        mgf: impl Fn(f64) -> f64,
        threshold: f64,
        t: f64,
    ) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let bound = mgf(t) * (-t * threshold).exp();
        bound.min(1.0)
    }
}

/// Hoeffding's inequality: concentration for bounded random variables.
pub struct HoeffdingInequality;

impl HoeffdingInequality {
    /// Hoeffding bound for sum of bounded random variables.
    pub fn bound(n: usize, range: f64, epsilon: f64) -> f64 {
        if epsilon <= 0.0 {
            return 1.0;
        }
        let bound = (-2.0 * n as f64 * epsilon * epsilon / (range * range)).exp();
        bound.min(1.0)
    }

    /// Hoeffding bound for sample mean.
    pub fn sample_mean_bound(n: usize, range: f64, epsilon: f64) -> f64 {
        if epsilon <= 0.0 {
            return 1.0;
        }
        let bound = (-2.0 * n as f64 * epsilon * epsilon / (range * range)).exp();
        bound.min(1.0)
    }

    /// Two-sided Hoeffding bound.
    pub fn two_sided_bound(n: usize, range: f64, epsilon: f64) -> f64 {
        2.0 * Self::bound(n, range, epsilon).min(1.0)
    }
}

/// Azuma's inequality: concentration for martingales with bounded differences.
pub struct AzumaInequality;

impl AzumaInequality {
    /// Azuma-Hoeffding inequality.
    pub fn bound(n: usize, max_diff: f64, epsilon: f64) -> f64 {
        if epsilon <= 0.0 {
            return 1.0;
        }
        let bound = (-2.0 * epsilon * epsilon / (n as f64 * max_diff * max_diff)).exp();
        bound.min(1.0)
    }

    /// Azuma inequality for martingale difference sequence.
    pub fn martingale_bound(
        differences: &[f64],
        epsilon: f64,
    ) -> f64 {
        let c_sq: f64 = differences.iter()
            .map(|&d| d * d)
            .sum();
        
        if epsilon <= 0.0 || c_sq == 0.0 {
            return 1.0;
        }
        
        let bound = (-2.0 * epsilon * epsilon / c_sq).exp();
        bound.min(1.0)
    }
}

/// McDiarmid's inequality: concentration for functions with bounded differences.
pub struct McDiarmidInequality;

impl McDiarmidInequality {
    /// McDiarmid's bounded differences inequality.
    pub fn bound(
        c_i: &[f64],
        epsilon: f64,
    ) -> f64 {
        if epsilon <= 0.0 {
            return 1.0;
        }
        
        let c_sq_sum: f64 = c_i.iter()
            .map(|&c| c * c)
            .sum();
        
        let bound = (-2.0 * epsilon * epsilon / c_sq_sum).exp();
        bound.min(1.0)
    }

    /// McDiarmid for functions with uniform bounded differences.
    pub fn uniform_bound(n: usize, c: f64, epsilon: f64) -> f64 {
        Self::bound(&vec![c; n], epsilon)
    }
}

/// Bennett's inequality: refinement of Bernstein and Chernoff.
pub struct BennettInequality;

impl BennettInequality {
    /// Bennett's inequality.
    pub fn bound(
        n: usize,
        variance: f64,
        max_diff: f64,
        epsilon: f64,
    ) -> f64 {
        if epsilon <= 0.0 || max_diff <= 0.0 {
            return 1.0;
        }
        
        let sigma_sq = variance;
        let b = max_diff;
        let t = epsilon;
        
        // Bennett's bound: exp(-n * sigma²/b² * h(bt/σ²))
        // where h(u) = (1+u)ln(1+u) - u
        let u = b * t / sigma_sq;
        let h = if u > 0.0 {
            (1.0 + u) * (1.0 + u).ln() - u
        } else {
            0.0
        };
        
        let bound = (-n as f64 * sigma_sq / (b * b) * h).exp();
        bound.min(1.0)
    }
}

/// Bernstein's inequality: combines variance and range information.
pub struct BernsteinInequality;

impl BernsteinInequality {
    /// Bernstein's inequality.
    pub fn bound(
        n: usize,
        variance: f64,
        max_diff: f64,
        epsilon: f64,
    ) -> f64 {
        if epsilon <= 0.0 {
            return 1.0;
        }
        
        let sigma_sq = variance;
        let b = max_diff;
        let t = epsilon;
        
        // Bernstein's bound: exp(-nt²/(2σ² + 2bt/3))
        let bound = (-n as f64 * t * t / (2.0 * sigma_sq + 2.0 * b * t / 3.0)).exp();
        bound.min(1.0)
    }
}

/// Jensen's inequality: for convex functions.
pub struct JensenInequality;

impl JensenInequality {
    /// Check Jensen's inequality for a convex function.
    pub fn check_convex(
        values: &[f64],
        weights: &[f64],
        f: impl Fn(f64) -> f64,
    ) -> bool {
        if values.len() != weights.len() || values.is_empty() {
            return false;
        }
        
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum == 0.0 {
            return false;
        }
        
        // E[f(X)] ≥ f(E[X])
        let expected_x: f64 = values.iter()
            .zip(weights.iter())
            .map(|(&x, &w)| x * w)
            .sum() / weight_sum;
        
        let expected_f_x: f64 = values.iter()
            .zip(weights.iter())
            .map(|(&x, &w)| f(x) * w)
            .sum() / weight_sum;
        
        expected_f_x >= f(expected_x) - 1e-10
    }

    /// Jensen's inequality for concave functions (reverse).
    pub fn check_concave(
        values: &[f64],
        weights: &[f64],
        f: impl Fn(f64) -> f64,
    ) -> bool {
        if values.len() != weights.len() || values.is_empty() {
            return false;
        }
        
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum == 0.0 {
            return false;
        }
        
        // E[f(X)] ≤ f(E[X]) for concave f
        let expected_x: f64 = values.iter()
            .zip(weights.iter())
            .map(|(&x, &w)| x * w)
            .sum() / weight_sum;
        
        let expected_f_x: f64 = values.iter()
            .zip(weights.iter())
            .map(|(&x, &w)| f(x) * w)
            .sum() / weight_sum;
        
        expected_f_x <= f(expected_x) + 1e-10
    }
}

/// Kolmogorov's inequality: maximal inequality for martingales.
pub struct KolmogorovInequality;

impl KolmogorovInequality {
    /// Kolmogorov's inequality for sum of independent random variables.
    pub fn bound(variance_sum: f64, epsilon: f64) -> f64 {
        if epsilon <= 0.0 {
            return 1.0;
        }
        (variance_sum / (epsilon * epsilon)).min(1.0)
    }

    /// Kolmogorov's maximal inequality.
    pub fn maximal_bound(
        variances: &[f64],
        epsilon: f64,
    ) -> f64 {
        let variance_sum: f64 = variances.iter().sum();
        Self::bound(variance_sum, epsilon)
    }
}

/// Doob's martingale inequalities.
pub struct DoobInequality;

impl DoobInequality {
    /// Doob's maximal inequality for non-negative submartingales.
    pub fn maximal_bound(p: f64, n: usize) -> f64 {
        if p <= 1.0 {
            return 1.0;
        }
        (p / (p - 1.0)).powf(p) / n as f64.powf(p - 1.0)
    }

    /// Doob's Lp inequality.
    pub fn lp_bound(p: f64) -> f64 {
        if p <= 1.0 {
            return 1.0;
        }
        (p / (p - 1.0)).powf(p)
    }
}

/// Union bound (Boole's inequality).
pub struct UnionBound;

impl UnionBound {
    /// Union bound: P(∪A_i) ≤ ΣP(A_i).
    pub fn bound(probabilities: &[f64]) -> f64 {
        probabilities.iter().sum::<f64>().min(1.0)
    }

    /// Bonferroni correction for multiple testing.
    pub fn bonferroni(alpha: f64, n_tests: usize) -> f64 {
        alpha / n_tests as f64
    }
}

/// Concentration of measure.
pub struct ConcentrationOfMeasure;

impl ConcentrationOfMeasure {
    /// Levy's lemma for Lipschitz functions on the sphere.
    pub fn levy_bound(
        dimension: usize,
        lipschitz_constant: f64,
        epsilon: f64,
    ) -> f64 {
        if epsilon <= 0.0 {
            return 1.0;
        }
        let bound = 2.0 * (-dimension as f64 * epsilon * epsilon / (2.0 * lipschitz_constant * lipschitz_constant)).exp();
        bound.min(1.0)
    }

    /// Gaussian concentration (isoperimetric inequality).
    pub fn gaussian_bound(
        lipschitz_constant: f64,
        epsilon: f64,
    ) -> f64 {
        if epsilon <= 0.0 {
            return 1.0;
        }
        let bound = 2.0 * (-epsilon * epsilon / (2.0 * lipschitz_constant * lipschitz_constant)).exp();
        bound.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markov_inequality() {
        let bound = MarkovInequality::bound(1.0, 2.0);
        assert!((bound - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_inequality() {
        let bound = ChebyshevInequality::bound(1.0, 2.0);
        assert!((bound - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_hoeffding_inequality() {
        let bound = HoeffdingInequality::bound(100, 1.0, 0.1);
        assert!(bound > 0.0 && bound < 1.0);
    }

    #[test]
    fn test_chernoff_bound() {
        let bound = ChernoffBound::bernoulli_bound(0.5, 100, 0.1);
        assert!(bound > 0.0 && bound < 1.0);
    }

    #[test]
    fn test_jensen_inequality() {
        let values = vec![0.0, 1.0, 2.0];
        let weights = vec![1.0, 1.0, 1.0];
        let f = |x: f64| x * x;  // convex function
        
        assert!(JensenInequality::check_convex(&values, &weights, f));
    }

    #[test]
    fn test_union_bound() {
        let probabilities = vec![0.1, 0.2, 0.3];
        let bound = UnionBound::bound(&probabilities);
        assert!((bound - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_bonferroni() {
        let corrected = UnionBound::bonferroni(0.05, 10);
        assert!((corrected - 0.005).abs() < 1e-10);
    }
}
