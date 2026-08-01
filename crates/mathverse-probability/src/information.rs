//! Information theory: entropy, mutual information, KL divergence, cross-entropy, channel capacity.

/// Shannon entropy for discrete distributions.
#[must_use]
pub struct Entropy;

impl Entropy {
    /// Shannon entropy H(X) = -Σ p(x) log₂ p(x).
    #[must_use]
    pub fn shannon(probabilities: &[f64]) -> f64 {
        let mut entropy = 0.0;
        for &p in probabilities {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Natural entropy (using natural log).
    #[must_use]
    pub fn natural(probabilities: &[f64]) -> f64 {
        let mut entropy = 0.0;
        for &p in probabilities {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Differential entropy for continuous distributions.
    #[must_use]
    pub fn differential(pdf: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
        let dx = (b - a) / n as f64;
        let mut entropy = 0.0;

        for i in 0..n {
            let x = a + (i as f64 + 0.5) * dx;
            let p = pdf(x);
            if p > 0.0 {
                entropy -= p * p.ln() * dx;
            }
        }

        entropy
    }

    /// Joint entropy H(X,Y) = -Σ p(x,y) log p(x,y).
    #[must_use]
    pub fn joint(joint_probabilities: &[Vec<f64>]) -> f64 {
        let mut entropy = 0.0;
        for row in joint_probabilities {
            for &p in row {
                if p > 0.0 {
                    entropy -= p * p.log2();
                }
            }
        }
        entropy
    }

    /// Conditional entropy H(X|Y) = H(X,Y) - H(Y).
    #[must_use]
    pub fn conditional(joint_probabilities: &[Vec<f64>], marginal_y: &[f64]) -> f64 {
        let joint_entropy = Self::joint(joint_probabilities);
        let marginal_entropy = Self::shannon(marginal_y);
        joint_entropy - marginal_entropy
    }

    /// Rényi entropy of order α.
    #[must_use]
    pub fn renyi(probabilities: &[f64], alpha: f64) -> f64 {
        if alpha == 1.0 {
            return Self::shannon(probabilities);
        }

        let sum: f64 = probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p.powf(alpha))
            .sum();

        (1.0 / (1.0 - alpha)) * sum.log2()
    }

    /// Hartley entropy (max-entropy).
    #[must_use]
    pub fn hartley(n_outcomes: usize) -> f64 {
        (n_outcomes as f64).log2()
    }
}

/// Kullback-Leibler divergence.
#[must_use]
pub struct KLDivergence;

impl KLDivergence {
    /// KL divergence D_KL(P || Q) = Σ p(x) log(p(x)/q(x)).
    #[must_use]
    pub fn discrete(p: &[f64], q: &[f64]) -> f64 {
        if p.len() != q.len() {
            return f64::NAN;
        }

        let mut divergence = 0.0;
        for i in 0..p.len() {
            if p[i] > 0.0 {
                if q[i] > 0.0 {
                    divergence += p[i] * (p[i] / q[i]).log2();
                } else {
                    return f64::INFINITY;
                }
            }
        }
        divergence
    }

    /// Symmetric KL divergence (Jeffreys divergence).
    #[must_use]
    pub fn symmetric(p: &[f64], q: &[f64]) -> f64 {
        Self::discrete(p, q) + Self::discrete(q, p)
    }

    /// Continuous KL divergence (numerical integration).
    #[must_use]
    pub fn continuous(
        p_pdf: impl Fn(f64) -> f64,
        q_pdf: impl Fn(f64) -> f64,
        a: f64,
        b: f64,
        n: usize,
    ) -> f64 {
        let dx = (b - a) / n as f64;
        let mut divergence = 0.0;

        for i in 0..n {
            let x = a + (i as f64 + 0.5) * dx;
            let p = p_pdf(x);
            let q = q_pdf(x);

            if p > 0.0 {
                if q > 0.0 {
                    divergence += p * (p / q).ln() * dx;
                } else {
                    return f64::INFINITY;
                }
            }
        }

        divergence
    }

    /// Jensen-Shannon divergence.
    #[must_use]
    pub fn jensen_shannon(p: &[f64], q: &[f64]) -> f64 {
        let n = p.len();
        let mut m = vec![0.0; n];

        for i in 0..n {
            m[i] = (p[i] + q[i]) / 2.0;
        }

        0.5 * Self::discrete(p, &m) + 0.5 * Self::discrete(q, &m)
    }
}

/// Mutual information.
#[must_use]
pub struct MutualInformation;

impl MutualInformation {
    /// Mutual information I(X;Y) = H(X) - H(X|Y).
    #[must_use]
    pub fn from_entropy(h_x: f64, h_x_given_y: f64) -> f64 {
        h_x - h_x_given_y
    }

    /// Mutual information from joint distribution.
    #[must_use]
    pub fn from_joint(joint_probabilities: &[Vec<f64>]) -> f64 {
        let n_rows = joint_probabilities.len();
        let n_cols = joint_probabilities[0].len();

        // Compute marginals
        let mut marginal_x = vec![0.0; n_rows];
        let mut marginal_y = vec![0.0; n_cols];

        for i in 0..n_rows {
            for j in 0..n_cols {
                marginal_x[i] += joint_probabilities[i][j];
                marginal_y[j] += joint_probabilities[i][j];
            }
        }

        // Compute mutual information
        let mut mi = 0.0;
        for i in 0..n_rows {
            for j in 0..n_cols {
                let p_xy = joint_probabilities[i][j];
                if p_xy > 0.0 {
                    let p_x = marginal_x[i];
                    let p_y = marginal_y[j];
                    if p_x > 0.0 && p_y > 0.0 {
                        mi += p_xy * (p_xy / (p_x * p_y)).log2();
                    }
                }
            }
        }

        mi
    }

    /// Normalized mutual information.
    #[must_use]
    pub fn normalized(mi: f64, h_x: f64, h_y: f64) -> f64 {
        let denom = h_x.max(h_y);
        if denom > 0.0 {
            mi / denom
        } else {
            0.0
        }
    }

    /// Pointwise mutual information.
    #[must_use]
    pub fn pointwise(p_xy: f64, p_x: f64, p_y: f64) -> f64 {
        if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
            (p_xy / (p_x * p_y)).log2()
        } else {
            0.0
        }
    }
}

/// Cross-entropy.
#[must_use]
pub struct CrossEntropy;

impl CrossEntropy {
    /// Cross-entropy H(P,Q) = -Σ p(x) log q(x).
    #[must_use]
    pub fn discrete(p: &[f64], q: &[f64]) -> f64 {
        if p.len() != q.len() {
            return f64::NAN;
        }

        let mut ce = 0.0;
        for i in 0..p.len() {
            if p[i] > 0.0 {
                if q[i] > 0.0 {
                    ce -= p[i] * q[i].log2();
                } else {
                    return f64::INFINITY;
                }
            }
        }
        ce
    }

    /// Cross-entropy loss for classification.
    #[must_use]
    pub fn loss(predictions: &[f64], targets: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (&p, &t) in predictions.iter().zip(targets.iter()) {
            if t > 0.0 {
                if p > 0.0 {
                    loss -= t * p.ln();
                } else {
                    return f64::INFINITY;
                }
            }
        }
        loss
    }

    /// Binary cross-entropy.
    #[must_use]
    pub fn binary(predictions: &[f64], targets: &[f64]) -> f64 {
        let mut loss = 0.0;
        for (&p, &t) in predictions.iter().zip(targets.iter()) {
            let p = p.clamp(1e-15, 1.0 - 1e-15);
            loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
        }
        loss / predictions.len() as f64
    }
}

/// Channel capacity.
#[must_use]
pub struct ChannelCapacity;

impl ChannelCapacity {
    /// Channel capacity C = max_{p(x)} I(X;Y).
    #[must_use]
    pub fn discrete(channel: &[Vec<f64>], tolerance: f64) -> f64 {
        let n_inputs = channel.len();
        let n_outputs = channel[0].len();

        // Start with uniform input distribution
        let mut p_x = vec![1.0 / n_inputs as f64; n_inputs];
        let mut capacity = 0.0;

        for _ in 0..1000 {
            // Compute output distribution
            let mut p_y = vec![0.0; n_outputs];
            for i in 0..n_inputs {
                for j in 0..n_outputs {
                    p_y[j] += p_x[i] * channel[i][j];
                }
            }

            // Compute mutual information
            let mut mi = 0.0;
            for i in 0..n_inputs {
                for j in 0..n_outputs {
                    let p_xy = p_x[i] * channel[i][j];
                    if p_xy > 0.0 && p_y[j] > 0.0 {
                        mi += p_xy * (p_xy / p_y[j]).log2();
                    }
                }
            }

            // Check convergence
            if (mi - capacity).abs() < tolerance {
                capacity = mi;
                break;
            }

            capacity = mi;

            // Update input distribution using Blahut-Arimoto (simplified)
            let mut new_p_x = vec![0.0; n_inputs];
            for i in 0..n_inputs {
                let mut sum = 0.0;
                for j in 0..n_outputs {
                    if channel[i][j] > 0.0 && p_y[j] > 0.0 {
                        sum += channel[i][j] * (channel[i][j] / p_y[j]).log2();
                    }
                }
                new_p_x[i] = p_x[i] * (2.0_f64).powf(sum);
            }

            let total: f64 = new_p_x.iter().sum();
            if total > 0.0 {
                for p in &mut new_p_x {
                    *p /= total;
                }
            }
            p_x = new_p_x;
        }

        capacity
    }

    /// Binary symmetric channel capacity.
    #[must_use]
    pub fn binary_symmetric(error_probability: f64) -> f64 {
        let p = error_probability;
        1.0 - (-p * p.log2() - (1.0 - p) * (1.0 - p).log2())
    }
}

/// Fisher information.
#[must_use]
pub struct FisherInformation;

impl FisherInformation {
    /// Fisher information I(θ) = E[(∂/∂θ log f(X;θ))²].
    #[must_use]
    pub fn discrete(
        log_likelihood: impl Fn(f64, f64) -> f64,
        theta: f64,
        support: &[f64],
        probabilities: &[f64],
    ) -> f64 {
        let epsilon = 1e-6;
        let mut fisher = 0.0;

        for (&x, &p) in support.iter().zip(probabilities.iter()) {
            let ll_plus = log_likelihood(x, theta + epsilon);
            let ll_minus = log_likelihood(x, theta - epsilon);
            let gradient = (ll_plus - ll_minus) / (2.0 * epsilon);
            fisher += p * gradient * gradient;
        }

        fisher
    }

    /// Fisher information for normal distribution.
    #[must_use]
    pub fn normal(variance: f64) -> f64 {
        1.0 / variance
    }

    /// Fisher information for Bernoulli distribution.
    #[must_use]
    pub fn bernoulli(p: f64) -> f64 {
        1.0 / (p * (1.0 - p))
    }

    /// Fisher information matrix (multivariate).
    #[must_use]
    pub fn matrix(
        log_likelihood: impl Fn(&[f64], &[f64]) -> f64,
        theta: &[f64],
        epsilon: f64,
    ) -> Vec<Vec<f64>> {
        let n = theta.len();
        let mut fisher_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let mut theta_plus = theta.to_vec();
                let mut theta_minus = theta.to_vec();
                theta_plus[i] += epsilon;
                theta_minus[i] -= epsilon;

                let mut theta_plus_j = theta_plus.to_vec();
                let mut theta_minus_j = theta_minus.to_vec();
                theta_plus_j[j] += epsilon;
                theta_minus_j[j] -= epsilon;

                let ll_pp = log_likelihood(&theta_plus_j, &[0.0]);
                let ll_pm = log_likelihood(&theta_plus, &[0.0]);
                let ll_mp = log_likelihood(&theta_minus, &[0.0]);
                let ll_mm = log_likelihood(&theta_minus_j, &[0.0]);

                let hessian = (ll_pp - ll_pm - ll_mp + ll_mm) / (4.0 * epsilon * epsilon);
                fisher_matrix[i][j] = -hessian;
                fisher_matrix[j][i] = fisher_matrix[i][j];
            }
        }

        fisher_matrix
    }
}

/// Information bottleneck method.
#[must_use]
pub struct InformationBottleneck;

impl InformationBottleneck {
    /// Information bottleneck optimization (simplified).
    #[must_use]
    pub fn optimize(
        joint_xy: &[Vec<f64>],
        beta: f64,
        n_clusters: usize,
        iterations: usize,
    ) -> (Vec<Vec<f64>>, f64) {
        let n_x = joint_xy.len();
        let n_y = joint_xy[0].len();

        // Initialize clustering
        let mut p_t_given_x = vec![vec![0.0; n_clusters]; n_x];
        for i in 0..n_x {
            for k in 0..n_clusters {
                p_t_given_x[i][k] = 1.0 / n_clusters as f64;
            }
        }

        let mut ib_value = 0.0;

        for _ in 0..iterations {
            // Compute p(t)
            let mut p_t = vec![0.0; n_clusters];
            let p_x: Vec<f64> = (0..n_x).map(|i| joint_xy[i].iter().sum()).collect();

            for i in 0..n_x {
                for k in 0..n_clusters {
                    p_t[k] += p_x[i] * p_t_given_x[i][k];
                }
            }

            // Compute p(y|t)
            let mut p_y_given_t = vec![vec![0.0; n_y]; n_clusters];
            for k in 0..n_clusters {
                for j in 0..n_y {
                    for i in 0..n_x {
                        p_y_given_t[k][j] += joint_xy[i][j] * p_t_given_x[i][k] / p_t[k];
                    }
                }
            }

            // Compute information bottleneck value
            let mut i_xt = 0.0;
            let mut i_ty = 0.0;

            for i in 0..n_x {
                for k in 0..n_clusters {
                    if p_t_given_x[i][k] > 0.0 && p_t[k] > 0.0 {
                        i_xt += p_x[i] * p_t_given_x[i][k] * (p_t_given_x[i][k] / p_t[k]).log2();
                    }
                }
            }

            for k in 0..n_clusters {
                for j in 0..n_y {
                    let p_y: f64 = (0..n_x).map(|i| joint_xy[i][j]).sum();
                    if p_y_given_t[k][j] > 0.0 && p_y > 0.0 {
                        i_ty += p_t[k] * p_y_given_t[k][j] * (p_y_given_t[k][j] / p_y).log2();
                    }
                }
            }

            ib_value = i_xt - beta * i_ty;

            // Update p(t|x) (simplified)
            for i in 0..n_x {
                let mut weights = vec![0.0; n_clusters];
                for k in 0..n_clusters {
                    let mut log_weight = 0.0;
                    for j in 0..n_y {
                        if joint_xy[i][j] > 0.0 && p_y_given_t[k][j] > 0.0 {
                            log_weight += joint_xy[i][j] * (p_y_given_t[k][j]).log2();
                        }
                    }
                    weights[k] = log_weight;
                }

                // Softmax
                let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_weights: Vec<f64> =
                    weights.iter().map(|&w| (w - max_weight).exp()).collect();
                let sum: f64 = exp_weights.iter().sum();

                for k in 0..n_clusters {
                    p_t_given_x[i][k] = exp_weights[k] / sum;
                }
            }
        }

        (p_t_given_x, ib_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy() {
        let uniform = vec![0.5, 0.5];
        let entropy = Entropy::shannon(&uniform);
        assert!((entropy - 1.0).abs() < 1e-10);

        let certain = vec![1.0, 0.0];
        let entropy_certain = Entropy::shannon(&certain);
        assert!((entropy_certain - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = KLDivergence::discrete(&p, &q);
        assert!((kl - 0.0).abs() < 1e-10);

        let q2 = vec![1.0, 0.0];
        let kl2 = KLDivergence::discrete(&p, &q2);
        assert!(kl2 > 0.0);
    }

    #[test]
    fn test_mutual_information() {
        let independent = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        let mi = MutualInformation::from_joint(&independent);
        assert!((mi - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        let p = vec![1.0, 0.0];
        let q = vec![1.0, 0.0];
        let ce = CrossEntropy::discrete(&p, &q);
        assert!((ce - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_binary_symmetric_channel() {
        let capacity = ChannelCapacity::binary_symmetric(0.1);
        assert!(capacity > 0.0 && capacity < 1.0);
    }

    #[test]
    fn test_fisher_information() {
        let fisher = FisherInformation::normal(1.0);
        assert!((fisher - 1.0).abs() < 1e-10);
    }
}
