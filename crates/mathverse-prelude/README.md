# MathVerse Prelude

[![Crates.io](https://img.shields.io/crates/v/mathverse-prelude.svg)](https://crates.io/crates/mathverse-prelude)
[![docs.rs](https://docs.rs/mathverse-prelude/badge.svg)](https://docs.rs/mathverse-prelude)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Convenience prelude that re-exports the entire MathVerse ecosystem through a single `use` statement.

---

## Features

- **Namespaced imports** — every crate lives under its own module (`prelude::ai`, `prelude::matrix`, ...), eliminating cross-crate name collisions
- **Curated flat prelude** — `use mathverse_prelude::prelude::*;` brings in the most common, collision-free types (`Tensor`, `Matrix`, `Vector`, `Complex`, `Polynomial`, `MathError`, ...)
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
    // Curated prelude types
    let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]);
    println!("Roots: {:?}", p.roots());

    let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    println!("det = {}", m.det().unwrap());

    // Namespaced access to any ecosystem crate
    let t = mathverse_prelude::ai::Tensor::zeros(&[2, 3]);
    println!("shape = {:?}", t.shape());
}
```

## Namespace Modules

| Module | Crate | Description |
|---|---|---|
| `core` | `mathverse_core` | Traits, numeric abstractions, errors, constants |
| `algebra` | `mathverse_algebra` | Polynomials, equation solving, factorization |
| `ai` | `mathverse_ai` | Tensors, activations, losses, optimizers, autograd |
| `ml` | `mathverse_machine_learning` | Linear/logistic models, KNN, trees, clustering |
| `matrix` | `mathverse_matrix` | Dense/sparse matrices, decompositions |
| `linear_algebra` | `mathverse_linear_algebra` | Decompositions, solvers, norms |
| `vector` | `mathverse_vector` | Vector operations |
| `complex` | `mathverse_complex` | Complex numbers |
| `calculus` | `mathverse_calculus` | Derivatives, integrals, vector calculus |
| `probability` | `mathverse_probability` | Distributions, Bayes, Monte Carlo |
| `statistics` | `mathverse_statistics` | Descriptive + inferential statistics |
| `number_theory` | `mathverse_number_theory` | Primes, GCD/LCM, modular arithmetic |
| `combinatorics` | `mathverse_combinatorics` | Combinatorial math |
| `graph` | `mathverse_graph` | Graph algorithms |
| `numerical` | `mathverse_numerical` | Root finding, Runge-Kutta, interpolation |
| `equations` | `mathverse_equations` | Equation solving |
| `transforms` | `mathverse_transforms` | FFT, DCT, wavelets |
| `signal` | `mathverse_signal` | Filters, convolution, correlation |
| `trigonometry` | `mathverse_trigonometry` | Trig and hyperbolic functions |
| `special` | `mathverse_special` | Special functions |
| `graphics` | `mathverse_graphics` | Graphics math |
| `vision` | `mathverse_vision` | Computer vision primitives |
| `ndarray_interop` | `mathverse_ndarray_interop` | `ndarray` conversions |
| `parallel` | `mathverse_parallel` | Parallel matrix ops |
| `views` | `mathverse_views` | Zero-copy views |
| `plot` (feature `plot`) | `mathverse_plot` | SVG/HTML/terminal plotting |

> Note: crates without a row above that are part of the workspace may not be
> re-exported here; depend on them directly for the trimmed dependency graph.

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE) for details.
