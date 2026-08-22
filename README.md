# MathVerse

[![Crates.io](https://img.shields.io/crates/v/mathverse-core.svg)](https://crates.io/crates/mathverse-core)
[![docs.rs](https://docs.rs/mathverse-core/badge.svg)](https://docs.rs/mathverse-core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

A production-grade Rust mathematical computing ecosystem — from elementary arithmetic to advanced scientific computing through one consistent, modular API.

---

## Overview

MathVerse aspires to be for Rust what NumPy, SciPy, SymPy, scikit-learn, OpenCV math, CGAL, Eigen, and parts of MATLAB collectively provide for other ecosystems. It is:

- **Unified** — one workspace, one error type (`MathError`), one trait hierarchy, one prelude
- **Modular** — 37 independent crates, each usable standalone
- **Safe** — `#![forbid(unsafe_code)]` workspace-wide, `Result`-based error handling
- **Tested** — 1,000+ unit tests, property-based tests, numerical accuracy tests
- **Documented** — every public item has `///` doc comments with runnable examples

---

## Workspace Layout

| Crate | Domain | Key Capabilities |
|-------|--------|-----------------|
| `mathverse-core` | Foundation | Traits (`Num`, `Signed`, `Field`, `Real`), scalar ops, constants, error type |
| `mathverse-arithmetic` | Arithmetic | Powers, roots, logs, rounding, GCD/LCM |
| `mathverse-algebra` | Algebra | Polynomials, root finding, factorization, systems |
| `mathverse-trigonometry` | Trigonometry | Trig, hyperbolic, inverse, angle conversions |
| `mathverse-geometry` | Geometry | 2D/3D shapes, transforms, intersection, spatial indexing |
| `mathverse-vector` | Vectors | Vector types and operations |
| `mathverse-matrix` | Matrices | Dense/sparse matrices, decompositions |
| `mathverse-linear-algebra` | Linear Algebra | Eigenvalues, SVD, QR, Cholesky, least squares |
| `mathverse-calculus` | Calculus | Derivatives, integrals, vector calculus |
| `mathverse-complex` | Complex | Complex number arithmetic |
| `mathverse-special` | Special Functions | Bessel, error function, gamma, zeta |
| `mathverse-probability` | Probability | Distributions, random variables, Bayes, Monte Carlo |
| `mathverse-statistics` | Statistics | Descriptive/inferential stats, regression |
| `mathverse-number-theory` | Number Theory | Primes, modular arithmetic, RSA helpers |
| `mathverse-combinatorics` | Combinatorics | Factorials, Catalan, Stirling, partitions |
| `mathverse-graph` | Graphs | Graph algorithms, traversal |
| `mathverse-numerical` | Numerical Methods | Root finding, ODE solvers, interpolation, quadrature |
| `mathverse-equations` | Equations | Equation solving |
| `mathverse-transforms` | Transforms | FFT, DCT, wavelets |
| `mathverse-signal` | Signal Processing | Filters, convolution, correlation |
| `mathverse-image` | Image Processing | Kernels, edge detection, morphology |
| `mathverse-vision` | Computer Vision | Camera models, homography, optical flow |
| `mathverse-ai` | AI/ML Primitives | Tensors, activations, losses, optimizers, attention |
| `mathverse-machine-learning` | Machine Learning | Linear/logistic regression, KNN, trees, clustering |
| `mathverse-physics` | Physics | Mechanics, E&M, optics, thermodynamics, waves |
| `mathverse-finance` | Finance | TVM, Black-Scholes, portfolio optimization |
| `mathverse-symbolic` | Symbolic Math | Expression trees, symbolic derivatives, LaTeX |
| `mathverse-units` | Units | SI/imperial conversion, dimensional analysis |
| `mathverse-optimization` | Optimization | Gradient descent, genetic algorithms, constrained optimization |
| `mathverse-graphics` | Graphics | Quaternions, meshes, transforms |
| `mathverse-plot` | Plotting | SVG, HTML, terminal visualization |
| `mathverse-prelude` | Prelude | Namespaced re-exports plus a curated collision-free prelude |
| `mathverse-ndarray-interop` | Interop | Conversions to/from `ndarray::Array2` |
| `mathverse-parallel` | Parallelism | Multi-threaded matrix operations |
| `mathverse-views` | Views | Zero-copy matrix/vector view types |
| `mathverse-dataframe` | Data Frames | Tabular data helpers |
| `mathverse-gpu` | GPU Backends | GPU-accelerated operations (experimental) |
| `mathverse-benches` | Benchmarks | Criterion benchmark suites |

---

## Quick Start

```toml
[dependencies]
mathverse-core = "0.1"
```

```rust
use mathverse_core::prelude::*;

fn main() {
    // Linear interpolation
    let x = lerp(0.0, 100.0, 0.3);              // 30.0

    // Float comparison with tolerance
    let eq = almost_eq(0.1 + 0.2, 0.3, 1e-6);  // true

    // Constants
    let area = PI * 5.0_f64.powi(2);             // 78.5398...

    // GCD and LCM
    let g = gcd(48, 18);                         // 6
    let l = lcm(4, 6);                           // 12

    // Prime checking
    assert!(is_prime(97));

    println!("lerp(0, 100, 0.3) = {x}");
    println!("pi x 5² = {area:.4}");
    println!("gcd(48, 18) = {g}");
}
```

For the full ecosystem:

```toml
[dependencies]
mathverse-prelude = "0.1"
```

Every crate is re-exported under its own namespace module (avoiding name
collisions), and a curated flat prelude covers the most common types:

```rust
use mathverse_prelude::prelude::*;

fn main() {
    // Curated prelude: Polynomial, Matrix, Vector, Tensor, Complex, ...
    let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]);
    println!("Roots: {:?}", p.roots());

    // Namespaced access to any crate
    let x_train = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
    let y_train = vec![0.0, 0.0, 1.0, 1.0];
    let preds = mathverse_prelude::ml::knn::classify(
        &x_train, &y_train, &vec![vec![0.5], vec![10.5]], 1
    ).unwrap();
    println!("KNN predictions: {preds:?}");
}
```

---

## Design Principles

| Principle | Description |
|-----------|-------------|
| **Zero-cost abstractions** | Generic over numeric traits, monomorphized at compile time |
| **SIMD-ready** | Hot paths designed for vectorization via `std::simd` |
| **`no_std` compatible** | Core crates work without `std` (feature flag) |
| **`#[forbid(unsafe_code)]`** | All safety guarantees enforced by the compiler |
| **`#[must_use]`** | Every pure function annotated to prevent silent discard |
| **Consistent API** | Same verbs across domains: `solve()`, `evaluate()`, `transform()` |

---

## MSRV

The minimum supported Rust version is **1.87** (stable channel).

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — workspace layout, dependencies, performance strategy
- [Feature Inventory](docs/FEATURES.md) — complete feature list by crate
- [API Guidelines](docs/API-GUIDELINES.md) — naming, error handling, documentation standards
- [MSRV Policy](docs/MSRV.md) — minimum supported version policy
- [Roadmap](docs/ROADMAP.md) — release phases and milestones
- [Contributing](CONTRIBUTING.md) — how to contribute

---

## License

Licensed under the [MIT License](LICENSE).
