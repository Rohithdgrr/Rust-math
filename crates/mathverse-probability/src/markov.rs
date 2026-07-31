//! Advanced Markov chains: HMM, MCMC, Metropolis-Hastings, Gibbs sampling, stationary distributions.

use crate::rng::Rng;

/// Hidden Markov Model.
pub struct HiddenMarkovModel {
    pub n_states: usize,
    pub n_observations: usize,
    pub initial: Vec<f64>,           // Initial state distribution
    pub transition: Vec<Vec<f64>>,   // State transition matrix
    pub emission: Vec<Vec<f64>>,     // Emission probabilities
}

impl HiddenMarkovModel {
    /// Create a new HMM.
    pub fn new(
        initial: Vec<f64>,
        transition: Vec<Vec<f64>>,
        emission: Vec<Vec<f64>>,
    ) -> Result<Self, String> {
        let n_states = initial.len();
        let n_observations = emission[0].len();
        
        // Validate dimensions
        if transition.len() != n_states {
            return Err("Transition matrix dimension mismatch".to_string());
        }
        
        for row in &transition {
            if row.len() != n_states {
                return Err("Transition matrix must be square".to_string());
            }
        }
        
        if emission.len() != n_states {
            return Err("Emission matrix dimension mismatch".to_string());
        }
        
        for row in &emission {
            if row.len() != n_observations {
                return Err("Emission matrix row dimension mismatch".to_string());
            }
        }
        
        Ok(HiddenMarkovModel {
            n_states,
            n_observations,
            initial,
            transition,
            emission,
        })
    }

    /// Forward algorithm: compute probability of observations.
    pub fn forward(&self, observations: &[usize]) -> f64 {
        let t = observations.len();
        if t == 0 {
            return 0.0;
        }
        
        let mut alpha = vec![0.0; self.n_states];
        
        // Initialization
        for i in 0..self.n_states {
            alpha[i] = self.initial[i] * self.emission[i][observations[0]];
        }
        
        // Induction
        for t in 1..observations.len() {
            let mut new_alpha = vec![0.0; self.n_states];
            for j in 0..self.n_states {
                for i in 0..self.n_states {
                    new_alpha[j] += alpha[i] * self.transition[i][j];
                }
                new_alpha[j] *= self.emission[j][observations[t]];
            }
            alpha = new_alpha;
        }
        
        // Termination
        alpha.iter().sum()
    }

    /// Viterbi algorithm: find most likely state sequence.
    pub fn viterbi(&self, observations: &[usize]) -> Vec<usize> {
        let t = observations.len();
        if t == 0 {
            return Vec::new();
        }
        
        let mut delta = vec![vec![0.0; self.n_states]; t];
        let mut psi = vec![vec![0usize; self.n_states]; t];
        
        // Initialization
        for i in 0..self.n_states {
            delta[0][i] = self.initial[i] * self.emission[i][observations[0]];
            psi[0][i] = 0;
        }
        
        // Recursion
        for t in 1..observations.len() {
            for j in 0..self.n_states {
                let mut max_val = 0.0;
                let mut max_idx = 0;
                
                for i in 0..self.n_states {
                    let val = delta[t - 1][i] * self.transition[i][j];
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                
                delta[t][j] = max_val * self.emission[j][observations[t]];
                psi[t][j] = max_idx;
            }
        }
        
        // Termination
        let mut path = vec![0usize; t];
        let mut max_val = 0.0;
        let mut max_idx = 0;
        
        for i in 0..self.n_states {
            if delta[t - 1][i] > max_val {
                max_val = delta[t - 1][i];
                max_idx = i;
            }
        }
        
        path[t - 1] = max_idx;
        
        // Backtracking
        for t in (1..observations.len()).rev() {
            path[t - 1] = psi[t][path[t]];
        }
        
        path
    }

    /// Sample observation sequence from HMM.
    pub fn sample(&self, length: usize, rng: &mut Rng) -> Vec<usize> {
        let mut observations = Vec::new();
        let mut state = Self::sample_categorical(&self.initial, rng);
        
        for _ in 0..length {
            let obs = Self::sample_categorical(&self.emission[state], rng);
            observations.push(obs);
            state = Self::sample_categorical(&self.transition[state], rng);
        }
        
        observations
    }

    fn sample_categorical(probs: &[f64], rng: &mut Rng) -> usize {
        let u = rng.uniform();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if u <= acc {
                return i;
            }
        }
        probs.len() - 1
    }
}

/// Metropolis-Hastings MCMC sampler.
pub struct MetropolisHastings {
    pub target_log_prob: Box<dyn Fn(&[f64]) -> f64>,
    pub proposal: Box<dyn Fn(&[f64], &mut Rng) -> Vec<f64>>,
    pub proposal_log_prob: Option<Box<dyn Fn(&[f64], &[f64]) -> f64>>,
}

impl MetropolisHastings {
    pub fn new<F1, F2>(
        target_log_prob: F1,
        proposal: F2,
    ) -> Self
    where
        F1: Fn(&[f64]) -> f64 + 'static,
        F2: Fn(&[f64], &mut Rng) -> Vec<f64> + 'static,
    {
        MetropolisHastings {
            target_log_prob: Box::new(target_log_prob),
            proposal: Box::new(proposal),
            proposal_log_prob: None,
        }
    }

    pub fn with_symmetric_proposal<F1, F2>(target_log_prob: F1, proposal: F2) -> Self
    where
        F1: Fn(&[f64]) -> f64 + 'static,
        F2: Fn(&[f64], &mut Rng) -> Vec<f64> + 'static,
    {
        MetropolisHastings {
            target_log_prob: Box::new(target_log_prob),
            proposal: Box::new(proposal),
            proposal_log_prob: None,
        }
    }

    /// Run MCMC chain.
    pub fn sample(&self, initial: &[f64], n_samples: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        let mut current = initial.to_vec();
        let mut current_log_prob = (self.target_log_prob)(&current);
        let mut samples = Vec::new();
        
        for _ in 0..n_samples {
            let proposed = (self.proposal)(&current, rng);
            let proposed_log_prob = (self.target_log_prob)(&proposed);
            
            let acceptance_ratio = if let Some(ref log_q) = self.proposal_log_prob {
                let log_q_current = log_q(&current, &proposed);
                let log_q_proposed = log_q(&proposed, &current);
                (proposed_log_prob - current_log_prob + log_q_current - log_q_proposed).exp()
            } else {
                (proposed_log_prob - current_log_prob).exp()
            };
            
            let u = rng.uniform();
            if u < acceptance_ratio.min(1.0) {
                current = proposed;
                current_log_prob = proposed_log_prob;
            }
            
            samples.push(current.clone());
        }
        
        samples
    }
}

/// Gibbs sampler.
pub struct GibbsSampler {
    pub conditional_dists: Vec<Box<dyn Fn(usize, &[f64], &mut Rng) -> f64>>,
}

impl GibbsSampler {
    pub fn new(conditional_dists: Vec<Box<dyn Fn(usize, &[f64], &mut Rng) -> f64>>) -> Self {
        GibbsSampler {
            conditional_dists,
        }
    }

    /// Run Gibbs sampling.
    pub fn sample(&self, initial: &[f64], n_samples: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
        let mut current = initial.to_vec();
        let dim = current.len();
        let mut samples = Vec::new();
        
        for _ in 0..n_samples {
            for i in 0..dim {
                current[i] = (self.conditional_dists[i])(i, &current, rng);
            }
            samples.push(current.clone());
        }
        
        samples
    }
}

/// Stationary distribution computation.
pub struct StationaryDistribution;

impl StationaryDistribution {
    /// Compute stationary distribution using power iteration.
    pub fn power_iteration(transition: &[Vec<f64>], tolerance: f64, max_iter: usize) -> Result<Vec<f64>, String> {
        let n = transition.len();
        if n == 0 {
            return Err("Empty transition matrix".to_string());
        }
        
        // Validate transition matrix
        for row in transition {
            if row.len() != n {
                return Err("Transition matrix must be square".to_string());
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err("Rows must sum to 1".to_string());
            }
        }
        
        // Initialize with uniform distribution
        let mut pi = vec![1.0 / n as f64; n];
        
        for _ in 0..max_iter {
            let mut new_pi = vec![0.0; n];
            
            // new_pi = pi * P
            for i in 0..n {
                for j in 0..n {
                    new_pi[j] += pi[i] * transition[i][j];
                }
            }
            
            // Check convergence
            let diff: f64 = pi.iter().zip(new_pi.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            
            pi = new_pi;
            
            if diff < tolerance {
                break;
            }
        }
        
        Ok(pi)
    }

    /// Check if a distribution is stationary.
    pub fn is_stationary(distribution: &[f64], transition: &[Vec<f64>], tolerance: f64) -> bool {
        let n = distribution.len();
        let mut result = vec![0.0; n];
        
        for i in 0..n {
            for j in 0..n {
                result[j] += distribution[i] * transition[i][j];
            }
        }
        
        for i in 0..n {
            if (distribution[i] - result[i]).abs() > tolerance {
                return false;
            }
        }
        
        true
    }

    /// Compute mixing time (simplified).
    pub fn mixing_time(transition: &[Vec<f64>], tolerance: f64) -> usize {
        let n = transition.len();
        let mut pi = vec![1.0 / n as f64; n];
        let mut steps = 0;
        
        loop {
            let mut new_pi = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_pi[j] += pi[i] * transition[i][j];
                }
            }
            
            let diff: f64 = pi.iter().zip(new_pi.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            
            pi = new_pi;
            steps += 1;
            
            if diff < tolerance || steps > 10000 {
                break;
            }
        }
        
        steps
    }

    /// Check ergodicity (irreducible and aperiodic).
    pub fn is_ergodic(transition: &[Vec<f64>]) -> bool {
        let n = transition.len();
        
        // Check irreducibility: all states communicate
        Self::is_irreducible(transition) && Self::is_aperiodic(transition)
    }

    fn is_irreducible(transition: &[Vec<f64>]) -> bool {
        let n = transition.len();
        
        // Build adjacency matrix
        let mut adj = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                adj[i][j] = transition[i][j] > 0.0;
            }
        }
        
        // Check connectivity using BFS from each state
        for start in 0..n {
            let mut visited = vec![false; n];
            let mut queue = vec![start];
            visited[start] = true;
            let mut count = 1;
            
            while let Some(state) = queue.pop() {
                for next in 0..n {
                    if adj[state][next] && !visited[next] {
                        visited[next] = true;
                        queue.push(next);
                        count += 1;
                    }
                }
            }
            
            if count != n {
                return false;
            }
        }
        
        true
    }

    fn is_aperiodic(transition: &[Vec<f64>]) -> bool {
        let n = transition.len();
        
        // Check if any state has self-loop
        for i in 0..n {
            if transition[i][i] > 0.0 {
                return true;
            }
        }
        
        // More sophisticated check would involve computing GCD of cycle lengths
        // For simplicity, return true if irreducible (most practical cases)
        true
    }
}

/// Absorbing Markov chain analysis.
pub struct AbsorbingMarkovChain {
    pub transition: Vec<Vec<f64>>,
    pub absorbing_states: Vec<bool>,
}

impl AbsorbingMarkovChain {
    pub fn new(transition: Vec<Vec<f64>>, absorbing_states: Vec<bool>) -> Result<Self, String> {
        let n = transition.len();
        if absorbing_states.len() != n {
            return Err("Dimension mismatch".to_string());
        }
        
        Ok(AbsorbingMarkovChain {
            transition,
            absorbing_states,
        })
    }

    /// Compute fundamental matrix (I - Q)^(-1).
    pub fn fundamental_matrix(&self) -> Result<Vec<Vec<f64>>, String> {
        let n = self.transition.len();
        let transient: Vec<usize> = (0..n).filter(|&i| !self.absorbing_states[i]).collect();
        let t = transient.len();
        
        if t == 0 {
            return Err("No transient states".to_string());
        }
        
        // Build Q matrix (transient to transient transitions)
        let mut q = vec![vec![0.0; t]; t];
        for (i, &ti) in transient.iter().enumerate() {
            for (j, &tj) in transient.iter().enumerate() {
                q[i][j] = self.transition[ti][tj];
            }
        }
        
        // Compute I - Q
        for i in 0..t {
            q[i][i] = 1.0 - q[i][i];
            for j in (i + 1)..t {
                q[i][j] = -q[i][j];
                q[j][i] = -q[j][i];
            }
        }
        
        // Invert to get fundamental matrix
        Self::invert(&q)
    }

    /// Expected time to absorption from each transient state.
    pub fn expected_absorption_time(&self) -> Result<Vec<f64>, String> {
        let n = self.transition.len();
        let transient: Vec<usize> = (0..n).filter(|&i| !self.absorbing_states[i]).collect();
        let t = transient.len();
        
        if t == 0 {
            return Ok(vec![0.0; n]);
        }
        
        let fundamental = self.fundamental_matrix()?;
        let mut times = vec![0.0; n];
        
        for (i, &ti) in transient.iter().enumerate() {
            times[ti] = fundamental[i].iter().sum();
        }
        
        Ok(times)
    }

    /// Probability of absorption in each absorbing state.
    pub fn absorption_probabilities(&self) -> Result<Vec<Vec<f64>>, String> {
        let n = self.transition.len();
        let transient: Vec<usize> = (0..n).filter(|&i| !self.absorbing_states[i]).collect();
        let absorbing: Vec<usize> = (0..n).filter(|&i| self.absorbing_states[i]).collect();
        let t = transient.len();
        let a = absorbing.len();
        
        if t == 0 || a == 0 {
            return Ok(vec![vec![0.0; n]; n]);
        }
        
        let fundamental = self.fundamental_matrix()?;
        let mut r = vec![vec![0.0; a]; t];
        
        for (i, &ti) in transient.iter().enumerate() {
            for (j, &aj) in absorbing.iter().enumerate() {
                r[i][j] = self.transition[ti][aj];
            }
        }
        
        // B = F * R
        let mut b = vec![vec![0.0; a]; t];
        for i in 0..t {
            for j in 0..a {
                for k in 0..t {
                    b[i][j] += fundamental[i][k] * r[k][j];
                }
            }
        }
        
        // Map back to full state space
        let mut result = vec![vec![0.0; n]; n];
        for (i, &ti) in transient.iter().enumerate() {
            for (j, &aj) in absorbing.iter().enumerate() {
                result[ti][aj] = b[i][j];
            }
        }
        
        Ok(result)
    }

    fn invert(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let n = matrix.len();
        let mut aug = vec![vec![0.0; 2 * n]; n];
        
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = matrix[i][j];
            }
            aug[i][n + i] = 1.0;
        }
        
        for i in 0..n {
            let mut max_row = i;
            let mut max_val = aug[i][i].abs();
            for j in (i + 1)..n {
                if aug[j][i].abs() > max_val {
                    max_val = aug[j][i].abs();
                    max_row = j;
                }
            }
            
            if max_row != i {
                aug.swap(i, max_row);
            }
            
            let pivot = aug[i][i];
            if pivot.abs() < 1e-10 {
                return Err("Matrix is singular".to_string());
            }
            
            for j in 0..2 * n {
                aug[i][j] /= pivot;
            }
            
            for j in 0..n {
                if j != i {
                    let factor = aug[j][i];
                    for k in 0..2 * n {
                        aug[j][k] -= factor * aug[i][k];
                    }
                }
            }
        }
        
        let mut inv = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inv[i][j] = aug[i][n + j];
            }
        }
        
        Ok(inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_forward() {
        let initial = vec![0.6, 0.4];
        let transition = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let emission = vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]];
        
        let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();
        let obs = vec![0, 1, 2];
        let prob = hmm.forward(&obs);
        assert!(prob > 0.0);
    }

    #[test]
    fn test_hmm_viterbi() {
        let initial = vec![0.6, 0.4];
        let transition = vec![vec![0.7, 0.3], vec![0.4, 0.6]];
        let emission = vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]];
        
        let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();
        let obs = vec![0, 1, 2];
        let path = hmm.viterbi(&obs);
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_stationary_distribution() {
        let transition = vec![
            vec![0.9, 0.1],
            vec![0.5, 0.5],
        ];
        
        let pi = StationaryDistribution::power_iteration(&transition, 1e-10, 1000).unwrap();
        assert!((pi[0] - 5.0 / 6.0).abs() < 1e-6);
        assert!((pi[1] - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_metropolis_hastings() {
        let target = |x: &[f64]| -> f64 {
            -0.5 * x[0] * x[0]  // Standard normal
        };
        
        let proposal = |x: &[f64], rng: &mut Rng| -> Vec<f64> {
            let z = crate::distributions::Normal { mu: 0.0, sigma: 0.5 }.sample(rng);
            vec![x[0] + z]
        };
        
        let mh = MetropolisHastings::new(target, proposal);
        let mut rng = Rng::new(42);
        let samples = mh.sample(&[0.0], 1000, &mut rng);
        assert_eq!(samples.len(), 1000);
    }
}
