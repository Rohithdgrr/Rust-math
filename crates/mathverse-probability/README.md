# mathverse-probability

Comprehensive probability and stochastic processes library. Distributions, Bayesian inference, Markov chains, Monte Carlo simulation, information theory, queueing theory, and extreme value analysis — all dependency-free with a deterministic xorshift64\* RNG.

## Features

- **20+ distributions** with PMF/PDF, CDF, sampling, moments, and properties
- **Bayesian inference**: conjugate priors, credible intervals, Bayes factors, hierarchical models
- **Markov chains**: HMM (forward/Viterbi), MCMC (Metropolis-Hastings, Gibbs), stationary distributions
- **Stochastic processes**: Brownian motion, Poisson, Gaussian processes, Ornstein-Uhlenbeck, jump diffusion
- **Sampling**: rejection, importance, stratified, bootstrap, Latin hypercube, quasi-Monte Carlo
- **Limit theorems**: LLN, CLT, Berry-Esseen, large deviations, renewal theory
- **Inequalities**: Chebyshev, Chernoff, Hoeffding, Azuma, Bennett, Bernstein, Jensen
- **Information theory**: entropy, KL divergence, mutual information, Fisher information, channel capacity
- **Reliability**: survival/hazard functions, MTTF/MTBF, Kaplan-Meier, warranty analysis
- **Queueing theory**: M/M/1, M/M/c, M/G/1, G/G/1, birth-death, Jackson networks
- **Extreme value theory**: GEV, Gumbel, Fréchet, Weibull EVD, return periods, POT
- **Hypothesis testing**: Z/T/F/χ² tests, likelihood ratio, SPRT, multiple testing corrections
- **Estimation**: MLE, method of moments, Cramér-Rao bound, robust estimation, confidence intervals
- **Random variable algebra**: convolution, mixture distributions, order statistics, transformations
- **Generating functions**: PGF, MGF, characteristic functions, Laplace/Z-transforms, cumulants
- **Conditional probability**: Bayes, conditional distributions/expectation, martingales
- **Simulation**: Monte Carlo, event-driven, variance reduction, quasi-random sequences
- Deterministic seeded RNG for reproducible simulations

## Module Overview

| Module              | Purpose                                                    |
|---------------------|------------------------------------------------------------|
| `distributions`     | All distribution structs with moments, PMF/PDF, CDF, sample |
| `rng`               | Deterministic xorshift64\* PRNG                            |
| `properties`        | Extended moments, quantiles, skewness, kurtosis, MGF       |
| `multivariate`      | Covariance/correlation matrices, multivariate normal, copulas |
| `stochastic`        | Random walks, Brownian motion, Poisson/Gaussian processes   |
| `markov`            | HMM, MCMC, Gibbs sampler, stationary distributions         |
| `bayesian`          | Conjugate priors, credible intervals, Bayes factors        |
| `sampling`          | Rejection, importance, stratified, bootstrap, particle filter |
| `limit_theorems`    | LLN, CLT, convergence types, large deviations, Berry-Esseen |
| `inequalities`      | Markov, Chebyshev, Chernoff, Hoeffding, Azuma, Jensen      |
| `information`       | Entropy, KL divergence, mutual information, Fisher info     |
| `reliability`       | Survival, hazard, MTTF, MTBF, warranty analysis            |
| `queueing`          | M/M/1, M/M/c, M/G/1, G/G/1, birth-death, Jackson networks |
| `extreme_value`     | GEV, Gumbel, Fréchet, Weibull EVD, return periods, POT     |
| `hypothesis`        | Z/T/F/χ² tests, likelihood ratio, SPRT, multiple testing   |
| `estimation`        | MLE, method of moments, Cramér-Rao, confidence intervals   |
| `random_variables`  | Convolution, mixture, order statistics, transformations    |
| `generating_functions` | PGF, MGF, characteristic functions, Laplace/Z-transforms |
| `conditional`       | Conditional probability, expectation, martingales          |
| `simulation`        | Monte Carlo, event-driven, quasi-random sequences          |
| `special`           | Gamma, erf, lower/upper incomplete gamma, beta functions   |

## Installation

```toml
[dependencies]
mathverse-probability = { path = "../mathverse-probability" }
```

## Quick Start

```rust
use mathverse_probability::{Rng, Normal, mc_integrate, bayes, markov_distribution};

fn main() {
    // Monte Carlo integration of sin(x) from 0 to π
    let mut rng = Rng::new(42);
    let (est, err) = mc_integrate(&f64::sin, 0.0, core::f64::consts::PI, 100_000, &mut rng);
    println!("∫sin(x)dx ≈ {est:.6} ± {err:.6}"); // ≈ 2.0

    // Bayes' theorem: P(disease|positive) = P(pos|disease)·P(disease) / P(positive)
    let posterior = bayes(0.01, 0.95, 0.0585);
    println!("P(disease|pos) = {posterior:.4}"); // ≈ 0.1624

    // Sample from normal distribution
    let n = Normal { mu: 0.0, sigma: 1.0 };
    let sample = n.sample(&mut rng);
    println!("Normal sample: {sample:.4}");
}
```

---

## `distributions` — Probability Distributions

### Discrete Distributions

#### Bernoulli(p)

```
P(X=1) = p

  ▐█
  ▐█          p = 0.7
  ▐█   ▐█
  ▐█   ▐█
  ▐█   ▐█
  ▐█   ▐█
  ▐█   ▐█
  0     1
```

| Statistic | Formula          |
|-----------|------------------|
| Mean      | p                |
| Variance  | p(1-p)           |

```rust
use mathverse_probability::{Bernoulli, DiscreteDist};

let b = Bernoulli { p: 0.7 };
println!("P(X=1) = {}", b.pmf(1));   // 0.7
println!("P(X≤0) = {}", b.cdf(0));   // 0.3
let sample = b.sample(&mut rng);      // 0 or 1
```

#### Binomial(n, p)

```
    ▐██▌
    ▐██▌
  ▐███▌▐███▌
  ▐███▌▐███▌
▐████▌▐████▌▐███▌
▐████▌▐████▌▐███▌
▐████▌▐████▌▐███▌▐██▌▐██▌▐█▌▐█▌
 0  1  2  3  4  5  6  7  8  9 10
           n=10, p=0.5
```

| Statistic | Formula        |
|-----------|----------------|
| Mean      | np             |
| Variance  | np(1-p)        |

```rust
use mathverse_probability::{Binomial, DiscreteDist};

let b = Binomial { n: 10, p: 0.5 };
println!("P(X=5) = {}", b.pmf(5));   // ≈ 0.2461
println!("P(X≤5) = {}", b.cdf(5));   // ≈ 0.6230
```

#### Poisson(λ)

```
  ▐██▌
  ▐███▌
  ▐████▌
  ▐█████▌
▐███████▌▐███▌
▐███████▌▐████▌▐███▌
▐███████▌▐████▌▐████▌▐███▌▐██▌▐█▌
 0  1  2  3  4  5  6  7  8  9 10
         λ=3
```

| Statistic | Formula |
|-----------|---------|
| Mean      | λ       |
| Variance  | λ       |

```rust
use mathverse_probability::{Poisson, DiscreteDist};

let p = Poisson { lambda: 3.0 };
println!("P(X=2) = {}", p.pmf(2));   // ≈ 0.2240
```

### Continuous Distributions

#### Normal(μ, σ) — Bell Curve

```
                    ▄▄████▄▄
                ▄██████████████▄
            ▄██████████████████████▄
         ▄████████████████████████████▄
       ▄████████████████████████████████▄
     ▄██████████████████████████████████████▄
  ▄██████████████████████████████████████████████▄
▁██████████████████████████████████████████████████████▁
──────────────────────────────────────────────────────────
  μ-3σ   μ-2σ    μ-σ     μ     μ+σ    μ+2σ   μ+3σ

    68.2% within μ ± σ
    95.4% within μ ± 2σ
    99.7% within μ ± 3σ
```

```rust
use mathverse_probability::{Normal, ContinuousDist};

let n = Normal { mu: 0.0, sigma: 1.0 };
println!("φ(0) = {}", n.pdf(0.0));          // ≈ 0.3989
println!("Φ(1.96) = {}", n.cdf(1.96));      // ≈ 0.9750
let sample = n.sample(&mut rng);             // Box-Muller
```

#### Exponential(λ)

```
▐███▌
▐███▌
▐██▌
▐██▌
▐█▌
▐█▌
▐▌▐▌
▐▌▐▌
▐▌▐▌
▐▌▐▌▐▌
──────▶ t
  1/λ (mean)
```

```rust
let e = Exponential { lambda: 2.0 };
println!("Mean = {}", e.mean());   // 0.5
let sample = e.sample(&mut rng);   // -ln(U)/λ
```

### All Distributions

| Distribution       | Parameters    | Sampling Method           |
|--------------------|---------------|---------------------------|
| `Bernoulli`        | p             | Inverse CDF               |
| `Binomial`         | n, p          | Inverse CDF               |
| `Poisson`          | λ             | Recurrence                |
| `Uniform`          | a, b          | Direct                    |
| `Normal`           | μ, σ          | Box-Muller                |
| `Exponential`      | λ             | Inverse transform         |
| `LogNormal`        | μ, σ          | Via normal                |
| `Weibull`          | shape, scale  | Inverse transform         |
| `ChiSquared`       | k             | Sum of k normal²         |
| `StudentsT`        | ν             | Via normal/chi-squared    |
| `FDistribution`    | d₁, d₂        | Via chi-squared            |
| `Geometric`        | p             | Inverse CDF               |
| `NegativeBinomial` | r, p          | Via gamma/Poisson          |
| `Hypergeometric`   | N, K, n       | Inverse CDF               |
| `Cauchy`           | x₀, γ         | tan(π(U-0.5))            |
| `Laplace`          | μ, b          | Difference of exponentials |
| `Gumbel`           | μ, β          | -ln(-ln(U))               |
| `Pareto`           | xₘ, α         | Inverse transform         |
| `Triangular`       | a, b, c       | Inverse CDF               |
| `Beta`             | α, β          | Via gamma                 |
| `Gamma`            | shape, rate   | Marsaglia-Tsang           |

---

## `bayesian` — Bayesian Inference

```
┌───────────────────────────────────────────────────────┐
│                   Bayes' Theorem                       │
│                                                        │
│         P(θ|data) = P(data|θ) · P(θ)                 │
│                        ─────────────                  │
│                           P(data)                     │
│                                                        │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐            │
│  │  Prior   │ × │Likelihood│ = │Posterior │            │
│  │  P(θ)    │   │P(data|θ)│   │ P(θ|data)│            │
│  └─────────┘   └─────────┘   └─────────┘            │
└───────────────────────────────────────────────────────┘
```

### Conjugate Pairs

| Likelihood     | Prior         | Posterior       |
|----------------|---------------|-----------------|
| Bernoulli      | Beta          | Beta            |
| Normal (known σ) | Normal      | Normal          |
| Poisson        | Gamma         | Gamma           |
| Multinomial    | Dirichlet     | Dirichlet       |

```rust
use mathverse_probability::bayesian::{BetaPrior, NormalPrior, CredibleInterval};

// Beta-Bernoulli: start with Beta(1,1), observe 5 heads, 5 tails
let prior = BetaPrior::new(1.0, 1.0);
let posterior = prior.posterior(5, 5);
println!("Posterior: Beta({}, {})", posterior.alpha, posterior.beta); // Beta(6, 6)
println!("E[p] = {}", posterior.predictive());                        // 0.5

// Normal-Normal: known variance
let prior = NormalPrior::new(0.0, 1.0);
let posterior = prior.posterior_known_variance(&[1.0, 2.0, 3.0], 1.0);
println!("Posterior: N({}, {}²)", posterior.mu, posterior.sigma);
```

### Credible Intervals

```rust
let samples = vec![/* ... MCMC samples ... */];
let (lo, hi) = CredibleInterval::from_samples(&samples, 0.05); // 95% CI
let (lo_hpd, hi_hpd) = CredibleInterval::hpd(&samples, 0.05);  // HPD interval
```

---

## `markov` — Markov Chains & MCMC

```
    ┌──────┐   0.7   ┌──────┐
    │State │────────▶│State │
    │  0   │◀────────│  1   │
    └──────┘   0.4   └──────┘
      │ 0.3            │ 0.6
      ▼                ▼
    (self)           (self)
```

### Hidden Markov Model

```rust
use mathverse_probability::markov::HiddenMarkovModel;

let hmm = HiddenMarkovModel::new(
    vec![0.6, 0.4],                                    // initial
    vec![vec![0.7, 0.3], vec![0.4, 0.6]],             // transition
    vec![vec![0.1, 0.4, 0.5], vec![0.6, 0.3, 0.1]],   // emission
).unwrap();

let prob = hmm.forward(&[0, 1, 2]);       // P(observations)
let path = hmm.viterbi(&[0, 1, 2]);       // Most likely state sequence
let obs = hmm.sample(10, &mut rng);       // Generate observation sequence
```

### MCMC Sampling

```rust
use mathverse_probability::markov::MetropolisHastings;

let target = |x: &[f64]| -> f64 { -0.5 * x[0] * x[0] }; // Standard normal
let proposal = |x: &[f64], rng: &mut Rng| -> Vec<f64> {
    vec![x[0] + Normal { mu: 0.0, sigma: 0.5 }.sample(rng)]
};

let mh = MetropolisHastings::new(target, proposal);
let samples = mh.sample(&[0.0], 10_000, &mut rng);
```

---

## `stochastic` — Stochastic Processes

### Brownian Motion

```
     ▲ W(t)
     │    ╱╲
     │   ╱  ╲    ╱╲
     │──╱────╲──╱──╲────
     │ ╱      ╲╱    ╲  ╱
     │╱              ╲╱
     └────────────────────▶ t
```

```rust
use mathverse_probability::stochastic::BrownianMotion;

let mut bm = BrownianMotion::new(0.01, 0.0);
let path = bm.generate(1000, &mut rng);

// Geometric Brownian Motion (Black-Scholes)
let price = bm.geometric_step(0.05, 0.2, &mut rng);
```

### Poisson Process

```rust
use mathverse_probability::stochastic::PoissonProcess;

let mut pp = PoissonProcess::new(2.0); // λ=2 events/sec
let events = pp.simulate(10.0, &mut rng); // Simulate for 10 seconds
println!("{} events occurred", events.len());
```

### Ornstein-Uhlenbeck (Mean-Reverting)

```rust
let mut ou = OrnsteinUhlenbeck::new(0.5, 0.0, 0.1, 1.0, 0.01);
let path = ou.generate(1000, &mut rng);
```

---

## `sampling` — Advanced Sampling

| Method                  | Use Case                                  |
|-------------------------|-------------------------------------------|
| `RejectionSampling`     | Sample from any target PDF                |
| `ImportanceSampling`    | Estimate integrals with known proposal    |
| `StratifiedSampling`    | Reduce variance in integration            |
| `Resampling::bootstrap` | Confidence intervals without distributional assumptions |
| `ParticleFilter`        | Sequential Monte Carlo / state estimation |

```rust
use mathverse_probability::sampling::{Resampling, StratifiedSampling};

// Bootstrap confidence interval for the mean
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let (lo, hi) = Resampling::bootstrap_ci(&data, |d| d.iter().sum::<f64>() / d.len() as f64, 0.05, 1000, &mut rng);

// Latin Hypercube Sampling
let samples = StratifiedSampling::latin_hypercube(3, 100, &mut rng);
```

---

## `information` — Information Theory

### Entropy & Divergence

```
H(X) = -Σ p(x) log₂ p(x)

         ┌──────────────────────────┐
Uniform  │████████████████████████████│  H = log₂(n) (max)
         ├──────────────────────────┤
Skewed   │████████████████████        │  H < log₂(n)
         ├──────────────────────────┤
Determin │████████████████████████████████│  H = 0 (min)
         └──────────────────────────┘
```

```rust
use mathverse_probability::information::{Entropy, KLDivergence, MutualInformation};

// Shannon entropy
let uniform = vec![0.25, 0.25, 0.25, 0.25];
println!("H = {} bits", Entropy::shannon(&uniform)); // 2.0

// KL divergence D_KL(P || Q)
let p = vec![0.5, 0.5];
let q = vec![0.9, 0.1];
println!("D_KL = {}", KLDivergence::discrete(&p, &q));

// Mutual information from joint distribution
let joint = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
println!("I(X;Y) = {}", MutualInformation::from_joint(&joint)); // 0 (independent)
```

---

## `queueing` — Queueing Theory

### M/M/1 Queue

```
                    ┌───────────┐
  λ ───▶ ────────▶ │  Server   │ ────────▶ μ
  arrival           │  (μ)      │        departure
                    └───────────┘
  
  ρ = λ/μ < 1 (stability condition)
  L = ρ/(1-ρ)     (avg customers in system)
  W = 1/(μ-λ)     (avg time in system)
```

```rust
use mathverse_probability::queueing::MM1Queue;

let q = MM1Queue::new(2.0, 5.0).unwrap();
println!("Utilization: {:.1}%", q.utilization() * 100.0);       // 40%
println!("Avg in system: {:.2}", q.average_number_in_system()); // 0.667
println!("Avg wait: {:.4} sec", q.average_waiting_time());
```

---

## `extreme_value` — Extreme Value Theory

```
  GEV Distribution Shapes

  ξ > 0 (Fréchet)    ξ = 0 (Gumbel)    ξ < 0 (Weibull)
      ╱╲                  ╱╲                  ╱╲
     ╱  ╲                ╱  ╲                ╱  ╲
    ╱    ╲              ╱    ╲              ╱    ╲
───╱──────╲───      ───╱──────╲───      ───╱──────╲───
  heavy tail         light tail          bounded upper
```

```rust
use mathverse_probability::extreme_value::{GEVDistribution, ReturnPeriod};

let gev = GEVDistribution::new(0.0, 1.0, 0.1).unwrap();
println!("100-year return level: {:.2}", ReturnPeriod::return_level(&gev, 100.0));
```

---

## `hypothesis` — Hypothesis Testing

```
┌─────────────────────────────────────────────────────┐
│  H₀: no effect     vs     H₁: effect exists       │
│                                                      │
│  ┌──────────┐         ┌──────────┐                  │
│  │ Retain H₀│         │ Reject H₀│                  │
│  │ (p > α)  │         │ (p < α)  │                  │
│  └──────────┘         └──────────┘                  │
│                                                      │
│  Type I error (α): reject H₀ when H₀ true          │
│  Type II error (β): retain H₀ when H₁ true         │
│  Power = 1 - β: correctly reject H₀                 │
└─────────────────────────────────────────────────────┘
```

```rust
use mathverse_probability::hypothesis::{ZTest, TTest, ChiSquaredTest, AlternativeHypothesis};

// One-sample Z-test
let result = ZTest::one_sample(5.0, 4.0, 1.0, 100, AlternativeHypothesis::Greater);
println!("Z = {:.3}, p = {:.4}", result.test_statistic, result.p_value);

// Two-sample t-test
let result = TTest::two_sample_equal_var(5.0, 4.0, 1.0, 1.2, 30, 30, AlternativeHypothesis::TwoSided);

// Chi-squared goodness of fit
let observed = vec![10.0, 20.0, 30.0];
let expected = vec![15.0, 15.0, 30.0];
let result = ChiSquaredTest::goodness_of_fit(&observed, &expected);
```

---

## Future Scope

- Hidden Markov Model Baum-Welch (EM) training
- Particle MCMC and Hamiltonian Monte Carlo
- Copula models (Clayton, Frank, Gumbel)
- Non-parametric density estimation (KDE)
- Survival analysis with Cox proportional hazards
- Network reliability and graph-based models
- Stochastic differential equations (SDE solvers)
- Variational inference methods
- Sequential hypothesis testing extensions
- Parallel RNG with jump-ahead support

## License

MIT OR Apache-2.0
