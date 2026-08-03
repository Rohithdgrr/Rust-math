# MathVerse Probability

[![Crates.io](https://img.shields.io/crates/v/mathverse-probability.svg)](https://crates.io/crates/mathverse-probability)
[![docs.rs](https://docs.rs/mathverse-probability/badge.svg)](https://docs.rs/mathverse-probability)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Comprehensive probability and stochastic processes library with 20+ distributions, Bayesian inference, Markov chains, Monte Carlo simulation, and information theory.

---

## Features

- **20+ distributions** — PMF/PDF, CDF, sampling, moments, and properties
- **Bayesian inference** — conjugate priors, credible intervals, Bayes factors, hierarchical models
- **Markov chains** — HMM (forward/Viterbi), MCMC (Metropolis-Hastings, Gibbs), stationary distributions
- **Stochastic processes** — Brownian motion, Poisson, Gaussian processes, Ornstein-Uhlenbeck
- **Sampling** — rejection, importance, stratified, bootstrap, Latin hypercube, quasi-Monte Carlo
- **Information theory** — entropy, KL divergence, mutual information, Fisher information
- **Queueing theory** — M/M/1, M/M/c, M/G/1, birth-death, Jackson networks
- **Extreme value theory** — GEV, Gumbel, Fréchet, Weibull EVD, return periods, POT
- **Hypothesis testing** — Z/T/F/χ² tests, likelihood ratio, SPRT, multiple testing corrections
- **Estimation** — MLE, method of moments, Cramér-Rao bound, confidence intervals
- Deterministic seeded RNG for reproducible simulations

## Module Overview

| Module | Purpose |
|--------|---------|
| `distributions` | All distribution structs with moments, PMF/PDF, CDF, sample |
| `rng` | Deterministic xorshift64\* PRNG |
| `bayesian` | Conjugate priors, credible intervals, Bayes factors |
| `markov` | HMM, MCMC, Gibbs sampler, stationary distributions |
| `stochastic` | Random walks, Brownian motion, Poisson/Gaussian processes |
| `sampling` | Rejection, importance, stratified, bootstrap, particle filter |
| `information` | Entropy, KL divergence, mutual information, Fisher info |
| `queueing` | M/M/1, M/M/c, M/G/1, birth-death, Jackson networks |
| `extreme_value` | GEV, Gumbel, Fréchet, Weibull EVD, return periods, POT |
| `hypothesis` | Z/T/F/χ² tests, likelihood ratio, SPRT, multiple testing |
| `estimation` | MLE, method of moments, Cramér-Rao, confidence intervals |
| `inequalities` | Markov, Chebyshev, Chernoff, Hoeffding, Azuma, Jensen |
| `limit_theorems` | LLN, CLT, Berry-Esseen, large deviations, renewal theory |
| `random_variables` | Convolution, mixture, order statistics, transformations |
| `generating_functions` | PGF, MGF, characteristic functions, Laplace/Z-transforms |
| `conditional` | Conditional probability, expectation, martingales |
| `simulation` | Monte Carlo, event-driven, quasi-random sequences |
| `special` | Gamma, erf, lower/upper incomplete gamma, beta functions |

## Installation

```toml
[dependencies]
mathverse-probability = "0.1"
```

## Quick Start

```rust
use mathverse_probability::{Rng, Normal, mc_integrate, bayes, markov_distribution};

fn main() {
    // Monte Carlo integration of sin(x) from 0 to π
    let mut rng = Rng::new(42);
    let (est, err) = mc_integrate(&f64::sin, 0.0, core::f64::consts::PI, 100_000, &mut rng);
    println!("∫sin(x)dx ≈ {est:.6} ± {err:.6}"); // ≈ 2.0

    // Bayes' theorem: P(disease|positive)
    let posterior = bayes(0.01, 0.95, 0.0585).unwrap();
    println!("P(disease|pos) = {posterior:.4}"); // ≈ 0.1624

    // Sample from normal distribution
    let n = Normal { mu: 0.0, sigma: 1.0 };
    let sample = n.sample(&mut rng);
    println!("Normal sample: {sample:.4}");
}
```

---

## Per-Module Documentation

### Distributions

#### Discrete Distributions

| Distribution | Parameters | Sampling Method |
|--------------|------------|-----------------|
| `Bernoulli` | p | Inverse CDF |
| `Binomial` | n, p | Inverse CDF |
| `Poisson` | λ | Recurrence |
| `Geometric` | p | Inverse CDF |
| `NegativeBinomial` | r, p | Via gamma/Poisson |
| `Hypergeometric` | N, K, n | Inverse CDF |

#### Continuous Distributions

| Distribution | Parameters | Sampling Method |
|--------------|------------|-----------------|
| `Uniform` | a, b | Direct |
| `Normal` | μ, σ | Box-Muller |
| `Exponential` | λ | Inverse transform |
| `LogNormal` | μ, σ | Via normal |
| `Weibull` | shape, scale | Inverse transform |
| `ChiSquared` | k | Sum of k normal² |
| `StudentsT` | ν | Via normal/chi-squared |
| `Beta` | α, β | Via gamma |
| `Gamma` | shape, rate | Marsaglia-Tsang |
| `Cauchy` | x₀, γ | tan(π(U-0.5)) |
| `Gumbel` | μ, β | -ln(-ln(U)) |
| `Pareto` | xₘ, α | Inverse transform |

### Bayesian Inference

```rust
use mathverse_probability::bayesian::{BetaPrior, CredibleInterval};

// Beta-Bernoulli: start with Beta(1,1), observe 5 heads, 5 tails
let prior = BetaPrior::new(1.0, 1.0);
let posterior = prior.posterior(5, 5);
println!("Posterior: Beta({}, {})", posterior.alpha, posterior.beta); // Beta(6, 6)
```

### Markov Chains & MCMC

```rust
use mathverse_probability::markov::HiddenMarkovModel;

let hmm = HiddenMarkovModel::new(
    vec![0.6, 0.4],
    vec![vec![0.7, 0.3], vec![0.4, 0.6]],
    vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]],
).unwrap();

let prob = hmm.forward(&[0, 1, 2]);
let path = hmm.viterbi(&[0, 1, 2]);
```

### Stochastic Processes

```rust
use mathverse_probability::stochastic::BrownianMotion;

let mut bm = BrownianMotion::new(0.01, 0.0);
let path = bm.generate(1000, &mut rng);
```

### Information Theory

```rust
use mathverse_probability::information::{Entropy, KLDivergence};

let uniform = vec![0.25, 0.25, 0.25, 0.25];
println!("H = {} bits", Entropy::shannon(&uniform)); // 2.0
```

### Queueing Theory

```rust
use mathverse_probability::queueing::MM1Queue;

let q = MM1Queue::new(2.0, 5.0).unwrap();
println!("Utilization: {:.1}%", q.utilization() * 100.0);
```

---

## Future Scope

- HMM Baum-Welch (EM) training
- Particle MCMC and Hamiltonian Monte Carlo
- Copula models (Clayton, Frank, Gumbel)
- Non-parametric density estimation (KDE)
- Stochastic differential equations (SDE solvers)
- Variational inference methods

## License

MIT — see [LICENSE](LICENSE).
