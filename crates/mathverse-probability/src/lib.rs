//! # mathverse-probability
//!
//! Probability and stochastic processes: distributions, Bayesian inference,
//! Monte Carlo methods, Markov chains, queueing theory, extreme value theory,
//! hypothesis testing, estimation, and more.
//!
//! Dependency-free deterministic RNG (xorshift64\*); seed for reproducibility.
//!
//! # Quick start
//!
//! ```rust
//! use mathverse_probability::{Rng, bayes, Normal};
//!
//! // Bayes' theorem
//! let posterior = bayes(0.01, 0.9, 0.01 * 0.9 + 0.99 * 0.1);
//!
//! // Sample from a normal distribution
//! let mut rng = Rng::new(42);
//! let n = Normal { mu: 0.0, sigma: 1.0 };
//! let sample: f64 = n.sample(&mut rng);
//! ```

#![allow(
    unstable_name_collisions,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::double_must_use
)]

pub mod bayesian;
pub mod conditional;
pub mod distributions;
pub mod estimation;
pub mod extreme_value;
pub mod generating_functions;
pub mod hypothesis;
pub mod inequalities;
pub mod information;
pub mod limit_theorems;
pub mod markov;
pub mod multivariate;
pub mod properties;
pub mod queueing;
pub mod random_variables;
pub mod reliability;
pub mod rng;
pub mod sampling;
pub mod simulation;
pub mod special;
pub mod stochastic;

pub trait F64Ext {
    fn gamma(self) -> f64;
}

impl F64Ext for f64 {
    fn gamma(self) -> f64 {
        special::gamma_fn(self)
    }
}

pub use distributions::{
    Bernoulli, Binomial, ContinuousDist, DiscreteDist, Distribution, Normal, Poisson, Uniform,
};
pub use rng::Rng;

/// Bayes' theorem: `P(A|B) = P(A)·P(B|A) / P(B)`.
///
/// ```
/// use mathverse_probability::bayes;
/// // 0.4·0.6 / 0.3 = 0.8
/// assert!((bayes(0.4, 0.6, 0.3) - 0.8).abs() < 1e-12);
/// ```
pub fn bayes(prior: f64, likelihood: f64, evidence: f64) -> f64 {
    prior * likelihood / evidence
}

/// Monte Carlo estimate of `∫_a^b f(x) dx`: `(estimate, standard error)`.
///
/// ```
/// use mathverse_probability::{mc_integrate, Rng};
/// let mut rng = Rng::new(7);
/// let (est, err) = mc_integrate(&f64::sin, 0.0, core::f64::consts::PI, 100_000, &mut rng);
/// assert!((est - 2.0).abs() < 5.0 * err, "est {est} err {err}");
/// ```
pub fn mc_integrate(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    b: f64,
    samples: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    let n = samples as f64;
    let (mut sum, mut sumsq) = (0.0, 0.0);
    for _ in 0..samples {
        let y = f(a + (b - a) * rng.uniform());
        sum += y;
        sumsq += y * y;
    }
    let m = sum / n;
    let var = (sumsq / n - m * m).max(0.0);
    let scale = b - a;
    (m * scale, (var / n).sqrt() * scale)
}

/// One step of a Markov chain: next state from row `state` of the
/// transition matrix (rows must sum to 1).
pub fn markov_step(transition: &[&[f64]], state: usize, rng: &mut Rng) -> usize {
    let row = transition[state];
    let u = rng.uniform();
    let mut acc = 0.0;
    for (i, &p) in row.iter().enumerate() {
        acc += p;
        if u <= acc {
            return i;
        }
    }
    row.len() - 1
}

/// Distribution after `steps` transitions from `start` (power iteration).
///
/// ```
/// use mathverse_probability::markov_distribution;
/// // Stationary distribution of [[0.9, 0.1], [0.5, 0.5]] is (5/6, 1/6).
/// let p = markov_distribution(&[&[0.9, 0.1], &[0.5, 0.5]], &[1.0, 0.0], 500);
/// assert!((p[0] - 5.0 / 6.0).abs() < 1e-9 && (p[1] - 1.0 / 6.0).abs() < 1e-9);
/// ```
pub fn markov_distribution(transition: &[&[f64]], start: &[f64], steps: usize) -> Vec<f64> {
    let mut p = start.to_vec();
    for _ in 0..steps {
        let mut np = vec![0.0; p.len()];
        for (i, &pi) in p.iter().enumerate() {
            for (j, &t) in transition[i].iter().enumerate() {
                np[j] += pi * t;
            }
        }
        p = np;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monte_carlo_converges() {
        let mut rng = Rng::new(99);
        let (est, err) = mc_integrate(&|x| x * x, 0.0, 1.0, 100_000, &mut rng);
        assert!((est - 1.0 / 3.0).abs() < 5.0 * err);
    }

    #[test]
    fn markov_chain() {
        let t: &[&[f64]] = &[&[0.9, 0.1], &[0.5, 0.5]];
        let mut rng = Rng::new(3);
        let mut state = 0usize;
        for _ in 0..200 {
            state = markov_step(t, state, &mut rng);
        }
        // After burn-in, mostly state 0 (stationary mass 5/6).
        assert_eq!(state, 0);
    }
}
