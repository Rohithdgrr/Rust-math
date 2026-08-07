# Changelog

All notable changes to `mathverse-calculus` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Integration tests covering cross-module workflows (`tests/integration.rs`)
- Property-based tests using proptest (`tests/properties.rs`)
- Criterion benchmarks for all major functions (`benches/calculus.rs`)
- Basic and advanced usage examples (`examples/basic.rs`, `examples/advanced.rs`)
- Crate-level README with quick start and usage guide
- `prelude` module for convenient imports
- `std` / `no_std` feature flags

### Changed
- All fallible functions now return `MathResult<T>` (alias for `Result<T, MathError>`) instead of `Result<T, &'static str>`
- `runge_kutta_4_system` now returns `MathResult<Vec<(f64, Vec<f64>)>>` for consistency
- Improved crate-level documentation with module overview and Python parity table

### Fixed
- Removed duplicate code in `ode.rs` (midpoint function)
- `runge_kutta_4_system` now properly handles `steps == 0` case

## [0.1.1] - 2026-08-07

### Added
- Initial production release
- Numerical derivatives: central differences, partial, nth-order, discrete gradient
- Numerical integration: trapezoid, Simpson, adaptive, Gaussian quadrature, Romberg, 2D
- ODE solvers: Euler, midpoint, RK4 with builder API
- Vector calculus: gradient, divergence, curl, Laplacian, Jacobian, Hessian
- Root finding: Newton-Raphson with auto-differentiation, critical point finding

[Unreleased]: https://github.com/Rohithdgrr/Rust-math/compare/mathverse-calculus-v0.1.1...HEAD
[0.1.1]: https://github.com/Rohithdgrr/Rust-math/releases/tag/mathverse-calculus-v0.1.1
