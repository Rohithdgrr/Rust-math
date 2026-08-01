//! Conditional probability: Bayes' theorem, conditional distributions, conditional expectation, martingales.

/// Basic conditional probability: P(A|B) = P(A ∩ B) / P(B).
pub struct ConditionalProbability;

impl ConditionalProbability {
    /// Compute conditional probability.
    pub fn compute(p_a_and_b: f64, p_b: f64) -> f64 {
        if p_b > 0.0 {
            p_a_and_b / p_b
        } else {
            f64::NAN
        }
    }

    /// Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B).
    pub fn bayes_theorem(p_b_given_a: f64, p_a: f64, p_b: f64) -> f64 {
        if p_b > 0.0 {
            p_b_given_a * p_a / p_b
        } else {
            f64::NAN
        }
    }

    /// Extended Bayes' theorem with multiple hypotheses.
    pub fn extended_bayes(p_b_given_a: &[f64], p_a: &[f64]) -> Vec<f64> {
        let n = p_b_given_a.len();
        if n != p_a.len() || n == 0 {
            return vec![f64::NAN; n];
        }

        // Compute P(B) using law of total probability
        let p_b: f64 = p_b_given_a
            .iter()
            .zip(p_a.iter())
            .map(|(&pba, &pa)| pba * pa)
            .sum();

        if p_b == 0.0 {
            return vec![f64::NAN; n];
        }

        // Compute P(A_i|B) for each hypothesis
        p_b_given_a
            .iter()
            .zip(p_a.iter())
            .map(|(&pba, &pa)| pba * pa / p_b)
            .collect()
    }

    /// Law of total probability: P(B) = Σ P(B|A_i) * P(A_i).
    pub fn total_probability(p_b_given_a: &[f64], p_a: &[f64]) -> f64 {
        p_b_given_a
            .iter()
            .zip(p_a.iter())
            .map(|(&pba, &pa)| pba * pa)
            .sum()
    }
}

/// Conditional distributions.
pub struct ConditionalDistributions;

impl ConditionalDistributions {
    /// Conditional PMF: P(X=x|Y=y).
    pub fn conditional_pmf(joint_pmf: &[Vec<f64>], x: usize, y: usize) -> f64 {
        let p_xy = joint_pmf
            .get(x)
            .and_then(|row| row.get(y))
            .copied()
            .unwrap_or(0.0);

        let p_y: f64 = joint_pmf
            .iter()
            .map(|row| row.get(y).copied().unwrap_or(0.0))
            .sum();

        if p_y > 0.0 {
            p_xy / p_y
        } else {
            0.0
        }
    }

    /// Conditional PDF (numerical approximation).
    pub fn conditional_pdf(
        joint_pdf: impl Fn(f64, f64) -> f64,
        marginal_y: impl Fn(f64) -> f64,
        x: f64,
        y: f64,
    ) -> f64 {
        let p_xy = joint_pdf(x, y);
        let p_y = marginal_y(y);

        if p_y > 0.0 {
            p_xy / p_y
        } else {
            0.0
        }
    }

    /// Conditional CDF.
    pub fn conditional_cdf(
        joint_cdf: impl Fn(f64, f64) -> f64,
        marginal_y_cdf: impl Fn(f64) -> f64,
        x: f64,
        y: f64,
    ) -> f64 {
        let f_xy = joint_cdf(x, y);
        let f_y = marginal_y_cdf(y);

        if f_y > 0.0 {
            f_xy / f_y
        } else {
            0.0
        }
    }

    /// Conditional distribution of X given Y for bivariate normal.
    pub fn bivariate_normal_conditional(
        mu_x: f64,
        mu_y: f64,
        sigma_x: f64,
        sigma_y: f64,
        rho: f64,
        y: f64,
    ) -> (f64, f64) {
        let mean = mu_x + rho * (sigma_x / sigma_y) * (y - mu_y);
        let variance = sigma_x * sigma_x * (1.0 - rho * rho);
        (mean, variance)
    }
}

/// Conditional expectation.
pub struct ConditionalExpectation;

impl ConditionalExpectation {
    /// Conditional expectation E[X|Y=y] for discrete case.
    pub fn discrete(joint_pmf: &[Vec<f64>], y: usize) -> f64 {
        let p_y: f64 = joint_pmf
            .iter()
            .map(|row| row.get(y).copied().unwrap_or(0.0))
            .sum();

        if p_y == 0.0 {
            return 0.0;
        }

        let mut expectation = 0.0;
        for (x, row) in joint_pmf.iter().enumerate() {
            let p_xy = row.get(y).copied().unwrap_or(0.0);
            expectation += x as f64 * p_xy / p_y;
        }

        expectation
    }

    /// Conditional expectation for continuous case (numerical).
    pub fn continuous(
        joint_pdf: impl Fn(f64, f64) -> f64,
        marginal_y: impl Fn(f64) -> f64,
        y: f64,
        a: f64,
        b: f64,
        n: usize,
    ) -> f64 {
        let dx = (b - a) / n as f64;
        let mut integral = 0.0;
        let mut normalization = 0.0;

        for i in 0..n {
            let x = a + (i as f64 + 0.5) * dx;
            let p_xy = joint_pdf(x, y);
            let p_y = marginal_y(y);

            if p_y > 0.0 {
                let conditional = p_xy / p_y;
                integral += x * conditional * dx;
                normalization += conditional * dx;
            }
        }

        if normalization > 0.0 {
            integral / normalization
        } else {
            0.0
        }
    }

    /// Tower property: E\[E\[X|Y\]\] = E\[X].
    pub fn tower_property(
        joint_pmf: &[Vec<f64>],
        conditional_expectation: impl Fn(usize) -> f64,
    ) -> f64 {
        let n_y = joint_pmf[0].len();
        let mut expectation = 0.0;

        for y in 0..n_y {
            let p_y: f64 = joint_pmf
                .iter()
                .map(|row| row.get(y).copied().unwrap_or(0.0))
                .sum();
            expectation += conditional_expectation(y) * p_y;
        }

        expectation
    }

    /// Conditional variance: Var(X|Y) = E[X²|Y] - (E[X|Y])².
    pub fn variance(joint_pmf: &[Vec<f64>], y: usize) -> f64 {
        let p_y: f64 = joint_pmf
            .iter()
            .map(|row| row.get(y).copied().unwrap_or(0.0))
            .sum();

        if p_y == 0.0 {
            return 0.0;
        }

        let mut e_x = 0.0;
        let mut e_x2 = 0.0;

        for (x, row) in joint_pmf.iter().enumerate() {
            let p_xy = row.get(y).copied().unwrap_or(0.0);
            let conditional = p_xy / p_y;
            e_x += x as f64 * conditional;
            e_x2 += (x as f64) * (x as f64) * conditional;
        }

        e_x2 - e_x * e_x
    }

    /// Law of total variance: Var(X) = E[Var(X|Y)] + Var(E[X|Y]).
    pub fn total_variance(joint_pmf: &[Vec<f64>]) -> f64 {
        let n_y = joint_pmf[0].len();
        let mut e_var = 0.0;
        let mut expectations = Vec::new();

        for y in 0..n_y {
            let p_y: f64 = joint_pmf
                .iter()
                .map(|row| row.get(y).copied().unwrap_or(0.0))
                .sum();

            let var = Self::variance(joint_pmf, y);
            let exp = Self::discrete(joint_pmf, y);

            e_var += var * p_y;
            expectations.push((exp, p_y));
        }

        let e_exp = expectations.iter().map(|&(e, _)| e).sum::<f64>();
        let var_exp: f64 = expectations
            .iter()
            .map(|&(e, p)| (e - e_exp).powi(2) * p)
            .sum();

        e_var + var_exp
    }
}

/// Independence testing.
pub struct IndependenceTesting;

impl IndependenceTesting {
    /// Check if events are independent: P(A ∩ B) = P(A) * P(B).
    pub fn events_independent(p_a: f64, p_b: f64, p_a_and_b: f64, tolerance: f64) -> bool {
        (p_a_and_b - p_a * p_b).abs() < tolerance
    }

    /// Check if random variables are independent (discrete).
    pub fn variables_independent_discrete(joint_pmf: &[Vec<f64>], tolerance: f64) -> bool {
        let n_x = joint_pmf.len();
        let n_y = joint_pmf[0].len();

        // Compute marginal distributions
        let mut marginal_x = vec![0.0; n_x];
        let mut marginal_y = vec![0.0; n_y];

        for i in 0..n_x {
            for j in 0..n_y {
                marginal_x[i] += joint_pmf[i][j];
                marginal_y[j] += joint_pmf[i][j];
            }
        }

        // Check independence condition
        for i in 0..n_x {
            for j in 0..n_y {
                let expected = marginal_x[i] * marginal_y[j];
                if (joint_pmf[i][j] - expected).abs() > tolerance {
                    return false;
                }
            }
        }

        true
    }

    /// Mutual information as independence test.
    pub fn mutual_information_test(joint_pmf: &[Vec<f64>], threshold: f64) -> bool {
        let mi = crate::information::MutualInformation::from_joint(joint_pmf);
        mi < threshold
    }
}

/// Conditional independence.
pub struct ConditionalIndependence;

impl ConditionalIndependence {
    /// Check conditional independence: P(X,Y|Z) = P(X|Z) * P(Y|Z).
    pub fn check(joint_xyz: &[Vec<Vec<f64>>], z: usize, tolerance: f64) -> bool {
        let n_x = joint_xyz.len();
        let n_y = joint_xyz[0].len();
        let n_z = joint_xyz[0][0].len();

        if z >= n_z {
            return false;
        }

        // Compute conditional distributions
        let mut p_x_given_z = vec![0.0; n_x];
        let mut p_y_given_z = vec![0.0; n_y];

        for i in 0..n_x {
            for j in 0..n_y {
                p_x_given_z[i] += joint_xyz[i][j][z];
                p_y_given_z[j] += joint_xyz[i][j][z];
            }
        }

        // Check conditional independence
        for i in 0..n_x {
            for j in 0..n_y {
                let expected = p_x_given_z[i] * p_y_given_z[j];
                if (joint_xyz[i][j][z] - expected).abs() > tolerance {
                    return false;
                }
            }
        }

        true
    }

    /// Markov property: X ⊥ Y | Z if P(X|Y,Z) = P(X|Z).
    pub fn markov_property(
        joint_xyz: &[Vec<Vec<f64>>],
        y: usize,
        z: usize,
        tolerance: f64,
    ) -> bool {
        let n_x = joint_xyz.len();
        let n_y = joint_xyz[0].len();
        let n_z = joint_xyz[0][0].len();

        if y >= n_y || z >= n_z {
            return false;
        }

        // Compute P(X|Z)
        let mut p_x_given_z = vec![0.0; n_x];
        for i in 0..n_x {
            for j in 0..n_y {
                p_x_given_z[i] += joint_xyz[i][j][z];
            }
        }

        // Compute P(X|Y,Z) and compare
        for i in 0..n_x {
            let p_x_given_yz = joint_xyz[i][y][z];
            if (p_x_given_yz - p_x_given_z[i]).abs() > tolerance {
                return false;
            }
        }

        true
    }
}

/// Martingales (conditional expectation property).
pub struct Martingales;

impl Martingales {
    /// Check if sequence is a martingale: E[X_{n+1} | X_1,...,X_n] = X_n.
    pub fn is_martingale(
        transition_probs: &[Vec<f64>],
        current_state: usize,
        tolerance: f64,
    ) -> bool {
        let n = transition_probs.len();
        if current_state >= n {
            return false;
        }

        // Compute expected next value
        let mut expected = 0.0;
        for (next_state, &prob) in transition_probs[current_state].iter().enumerate() {
            expected += next_state as f64 * prob;
        }

        (expected - current_state as f64).abs() < tolerance
    }

    /// Check if sequence is a submartingale: E[X_{n+1} | ...] ≥ X_n.
    pub fn is_submartingale(transition_probs: &[Vec<f64>], current_state: usize) -> bool {
        let n = transition_probs.len();
        if current_state >= n {
            return false;
        }

        let mut expected = 0.0;
        for (next_state, &prob) in transition_probs[current_state].iter().enumerate() {
            expected += next_state as f64 * prob;
        }

        expected >= current_state as f64
    }

    /// Check if sequence is a supermartingale: E[X_{n+1} | ...] ≤ X_n.
    pub fn is_supermartingale(transition_probs: &[Vec<f64>], current_state: usize) -> bool {
        let n = transition_probs.len();
        if current_state >= n {
            return false;
        }

        let mut expected = 0.0;
        for (next_state, &prob) in transition_probs[current_state].iter().enumerate() {
            expected += next_state as f64 * prob;
        }

        expected <= current_state as f64
    }

    /// Doob's optional stopping theorem (simplified check).
    pub fn optional_stopping(
        stopping_times: &[usize],
        expectations: &[f64],
        initial_expectation: f64,
        tolerance: f64,
    ) -> bool {
        for (i, &_t) in stopping_times.iter().enumerate() {
            if (expectations[i] - initial_expectation).abs() > tolerance {
                return false;
            }
        }
        true
    }
}

/// Bayesian networks (conditional probability structures).
pub struct BayesianNetwork;

impl BayesianNetwork {
    /// Compute joint probability from conditional probability tables.
    pub fn joint_probability(variables: &[usize], cpts: &[Box<dyn Fn(&[usize]) -> f64>]) -> f64 {
        let mut probability = 1.0;

        for (i, cpt) in cpts.iter().enumerate() {
            let parents = &variables[..i];
            probability *= cpt(parents);
        }

        probability
    }

    /// Compute marginal probability by summing out variables.
    pub fn marginal_probability(
        target_var: usize,
        target_value: usize,
        cpts: &[Box<dyn Fn(&[usize]) -> f64>],
        n_values: usize,
    ) -> f64 {
        let mut probability = 0.0;

        for assignment in 0..n_values {
            let mut variables = vec![assignment; cpts.len()];
            variables[target_var] = target_value;
            probability += Self::joint_probability(&variables, cpts);
        }

        probability
    }

    /// Variable elimination algorithm (simplified).
    pub fn variable_elimination(
        _query_var: usize,
        evidence: &[(usize, usize)],
        cpts: &[Box<dyn Fn(&[usize]) -> f64>],
        _elimination_order: &[usize],
    ) -> f64 {
        // Simplified implementation
        // Apply evidence
        let mut result = 1.0;
        for cpt in cpts {
            let mut assignment = vec![0; evidence.len()];
            for (i, &(_var, val)) in evidence.iter().enumerate() {
                assignment[i] = val;
            }
            result *= cpt(&assignment);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conditional_probability() {
        let cp = ConditionalProbability::compute(0.25, 0.5);
        assert!((cp - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_theorem() {
        let posterior = ConditionalProbability::bayes_theorem(0.9, 0.01, 0.05);
        assert!(posterior > 0.0);
    }

    #[test]
    fn test_total_probability() {
        let p_b_given_a = vec![0.5, 0.3];
        let p_a = vec![0.6, 0.4];
        let p_b = ConditionalProbability::total_probability(&p_b_given_a, &p_a);
        assert!((p_b - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_conditional_expectation() {
        let joint = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let ce = ConditionalExpectation::discrete(&joint, 0);
        assert!(ce > 0.0);
    }

    #[test]
    fn test_events_independent() {
        let independent = IndependenceTesting::events_independent(0.5, 0.5, 0.25, 1e-10);
        assert!(independent);
    }

    #[test]
    fn test_bivariate_normal_conditional() {
        let (mean, var) =
            ConditionalDistributions::bivariate_normal_conditional(0.0, 0.0, 1.0, 1.0, 0.5, 1.0);
        assert!((mean - 0.5).abs() < 1e-10);
        assert!((var - 0.75).abs() < 1e-10);
    }
}
