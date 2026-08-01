# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-01

### Added
- Probability distributions: `Normal`, `Bernoulli`, `Binomial`, `Poisson`, `Exponential`, `Uniform`, `Cauchy`, `Gamma`
- Bayesian inference: `BetaPrior`, `NormalPrior`, `GammaPrior`, `DirichletPrior`, `BayesFactor`
- Monte Carlo integration and simulation (`mc_integrate`, `MonteCarloSimulation`)
- Markov chains: discrete, hidden (HMM), Metropolis-Hastings, Gibbs sampler
- Queueing theory: M/M/1, M/M/c, M/G/1, G/G/1, birth-death, Little's Law
- Extreme value theory: Gumbel, Fréchet, Weibull, Generalized Extreme Value (GEV)
- Hypothesis testing: z-test, t-test, chi-squared, SPRT, power analysis, Bonferroni correction
- Estimation: MLE, method of moments, confidence intervals, bootstrap
- Stochastic processes: random walks, Brownian motion, Poisson processes, Gaussian processes, Ornstein-Uhlenbeck
- Random variables: mixture distributions, transformations, convolution
- Generating functions: MGF, PGF, CGF, characteristic function, Z-transform
- Conditional probability: tower property, total probability, Bayes' theorem
- Inequalities: Chebyshev, Chernoff, Hoeffding, Markov, Jensen, Azuma
- Information theory: Shannon entropy, KL divergence, mutual information, Fisher information
- Reliability analysis: Kaplan-Meier, MTTF, MTBF, Weibull hazard, censored data
- Sampling methods: rejection sampling, importance sampling, stratified sampling, Latin hypercube, bootstrap, jackknife, antithetic variates, control variates
- Deterministic xorshift64 RNG with seed control
- `#[must_use]` annotations on all public types and pure functions

### Fixed
- Fixed 63 compile errors: missing imports, type inference, lifetime bounds, closure traits
- Fixed 4 failing tests: `test_gumbel_evd`, `test_sprt`, `test_iqr`, `test_birth_death_queue`
- Fixed doc warnings (broken intra-doc links with unescaped brackets)
- Deduplication: `MonteCarloSimulation::integrate` delegates to `mc_integrate`

### Changed
- MSRV requirement: 1.87 (edition 2021)
- Crate documentation improved with quick-start examples

## [Unreleased]

### Added
- None yet

### Fixed
- None yet

### Changed
- None yet
