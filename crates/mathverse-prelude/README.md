# MathVerse Prelude

[![Crates.io](https://img.shields.io/crates/v/mathverse-prelude.svg)](https://crates.io/crates/mathverse-prelude)
[![docs.rs](https://docs.rs/mathverse-prelude/badge.svg)](https://docs.rs/mathverse-prelude)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Convenience prelude that re-exports the entire MathVerse ecosystem through a single `use` statement.

---

## Features

- **Single import** — `use mathverse_prelude::prelude::*;` brings in every crate
- **27 crates** — core, arithmetic, algebra, trigonometry, calculus, statistics, physics, finance, and more
- **Zero overhead** — re-exports only, no additional logic

## Installation

```toml
[dependencies]
mathverse-prelude = "0.1"
```

## Quick Start

```rust
use mathverse_prelude::prelude::*;

fn main() {
    // From mathverse-algebra
    let p = mathverse_algebra::Polynomial::from_coeffs(&[6.0, -5.0, 1.0]);
    println!("Roots: {:?}", p.roots());

    // From mathverse-physics
    let ke = mathverse_physics::mechanics::kinetic_energy(2.0, 3.0);
    println!("KE = {ke} J");

    // From mathverse-finance
    let fv = mathverse_finance::tvm::future_value(1000.0, 0.05, 10);
    println!("FV = ${fv:.2}");

    // From mathverse-symbolic
    let x = mathverse_symbolic::Expr::v("x");
    let expr = x.clone().pow(mathverse_symbolic::Expr::c(2.0));
    println!("d/dx x² = {}", mathverse_symbolic::derivative::differentiate(&expr, "x"));
}
```

## Re-exported Crates

| Crate | Description |
|---|---|
| `mathverse_core` | Traits, numeric abstractions, errors, constants |
| `mathverse_arithmetic` | Basic ops, powers, roots, logs, rounding |
| `mathverse_algebra` | Polynomials, equation solving, factorization |
| `mathverse_trigonometry` | Trig, hyperbolic, inverse, angle conversions |
| `mathverse_geometry` | 2D/3D shapes, area, volume, transforms |
| `mathverse_linear_algebra` | Matrix, vector, tensor, decompositions |
| `mathverse_matrix` | Matrix specializations |
| `mathverse_vector` | Vector specializations |
| `mathverse_calculus` | Derivatives, integrals, vector calculus |
| `mathverse_complex` | Complex numbers |
| `mathverse_probability` | Distributions, Bayes, Monte Carlo |
| `mathverse_statistics` | Descriptive + inferential statistics |
| `mathverse_number_theory` | Primes, GCD/LCM, modular arithmetic |
| `mathverse_combinatorics` | Combinatorial math |
| `mathverse_graph` | Graph algorithms |
| `mathverse_optimization` | Gradient descent, SGD, genetic algorithms |
| `mathverse_numerical` | Root finding, Runge-Kutta, interpolation |
| `mathverse_equations` | Equation solving |
| `mathverse_transforms` | FFT, DCT, wavelets |
| `mathverse_signal` | Filters, convolution, correlation |
| `mathverse_ai` | Activations, losses, metrics, attention |
| `mathverse_machine_learning` | Linear, logistic, KNN, trees, clustering |
| `mathverse_vision` | Camera, homography, features, optical flow |
| `mathverse_physics` | Mechanics, E&M, optics, thermo, waves |
| `mathverse_finance` | TVM, investment, risk, options, portfolio |
| `mathverse_symbolic` | Expression trees, symbolic derivatives, LaTeX |
| `mathverse_units` | SI/imperial, compile-time dimensional analysis |
| `mathverse_plot` | SVG, HTML, terminal plotting |

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE) for details.
