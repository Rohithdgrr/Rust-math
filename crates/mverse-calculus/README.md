# mathverse-calculus

Calculus operations including derivatives, integrals, and vector calculus for the MathVerse ecosystem.

## Overview

`mathverse-calculus` provides comprehensive calculus functionality including numerical differentiation, integration, and vector calculus operations. This crate enables mathematical analysis of continuous functions and systems.

## Features

- **Numerical Derivatives**: 
  - First and higher-order derivatives
  - Partial derivatives
  - Gradient computation
- **Integration**:
  - Definite and indefinite integrals
  - Numerical integration methods (Riemann sums, trapezoidal rule, Simpson's rule)
  - Multiple integrals
- **Vector Calculus**:
  - Divergence
  - Curl
  - Gradient fields
  - Line integrals
  - Surface integrals
- **Limits**: Computation of function limits
- **Series**: Taylor and Maclaurin series expansions

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
mathverse-calculus = "0.1.0"
```

## Usage

```rust
use mathverse_calculus::prelude::*;

// Example: Numerical differentiation
let f = |x: f64| x.powi(2);
let derivative = differentiate(f, 2.0); // derivative at x=2

// Example: Numerical integration
let f = |x: f64| x.sin();
let integral = integrate(f, 0.0, std::f64::consts::PI);

// Example: Gradient computation
let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
let gradient = compute_gradient(f, &[1.0, 2.0]);
```

## Dependencies

- `mathverse-core`: Core numeric traits and utilities

## License

MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Status

This crate is currently in early development. API stability is not guaranteed.