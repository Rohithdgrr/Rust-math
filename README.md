# MathVerse

A production-grade Rust mathematical computing ecosystem: elementary arithmetic to advanced scientific computing through one consistent, modular API.

Mission: become for Rust what NumPy, SciPy, SymPy, scikit-learn utilities, OpenCV math, CGAL, Eigen, and parts of MATLAB collectively provide for other ecosystems.

## Objectives

- One unified mathematical ecosystem
- Production-ready, high performance, safe APIs
- Generic numeric types
- Well documented, educational, research friendly
- GPU ready, WASM compatible, embedded compatible

## Core Principles

- Modular, zero-cost abstractions, generic, extensible
- SIMD optimized, parallel execution
- Excellent documentation, 100% tested, stable API

## Workspace Layout

| Crate | Area |
|---|---|
| mathverse-core | Traits, numeric abstractions, errors, constants, generic ops, precision |
| mathverse-arithmetic | Basic ops, powers, roots, logs, rounding |
| mathverse-algebra | Polynomials, equation solving, factorization |
| mathverse-trigonometry | Trig, hyperbolic, inverse, angle conversions |
| mathverse-geometry | 2D/3D shapes, area, volume, transforms, intersection |
| mathverse-linear-algebra | Matrix, vector, tensor, sparse, eigen, LU/QR/SVD/Cholesky |
| mathverse-matrix / mathverse-vector | Matrix & vector specializations |
| mathverse-calculus | Derivatives, integrals, vector calculus |
| mathverse-complex | Complex numbers |
| mathverse-probability | Random variables, Bayes, distributions, Monte Carlo |
| mathverse-statistics | Descriptive + inferential statistics, regression |
| mathverse-discrete | Logic, sets, relations, graphs, trees, boolean algebra |
| mathverse-number-theory | Primes, GCD/LCM, modular arithmetic, RSA helpers |
| mathverse-combinatorics | Combinatorial math |
| mathverse-graph | Graph algorithms |
| mathverse-optimization | Gradient descent, SGD, Adam, annealing, genetic |
| mathverse-numerical | Root finding, Runge-Kutta, interpolation |
| mathverse-equations | Equation solving |
| mathverse-transforms | FFT, DCT, wavelets |
| mathverse-signal | Filters, convolution, correlation |
| mathverse-image | Kernels, blur, Sobel, Canny, morphology, transforms |
| mathverse-vision | Camera matrix, homography, epipolar, features, optical flow |
| mathverse-graphics | 2D/3D transforms, quaternions, Bezier, projection |
| mathverse-ai | Activations, losses, metrics, tensor ops, optimizers, attention |
| mathverse-physics | Mechanics, E&M, optics, thermo, fluids, quantum basics |
| mathverse-finance | Interest, EMI, NPV, IRR, Black-Scholes, risk |
| mathverse-symbolic | Expression trees, symbolic derivatives, LaTeX |
| mathverse-units | SI/imperial/currency, compile-time dimensional analysis |
| mathverse-plot | Plot, scatter, histogram, heatmap, SVG/HTML/terminal |
| mathverse-prelude | Re-exports of the whole ecosystem |

## Phases

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — workspace layout, dependencies, performance, testing
- [docs/FEATURES.md](docs/FEATURES.md) — full feature inventory by crate
- [docs/PHASEWISE-PLAN.md](docs/PHASEWISE-PLAN.md) — phase-by-phase execution plan
- [docs/ROADMAP.md](docs/ROADMAP.md) — phases, releases, developer experience
- [docs/API-GUIDELINES.md](docs/API-GUIDELINES.md) — API conventions, errors, docs & test standards

## Quick Start

```rust
use mathverse::prelude::*;

fn main() {
    // TBD — available after v0.1
}
```

## Release Roadmap

- **v0.1** — core, arithmetic, algebra
- **v0.2** — geometry, trigonometry
- **v0.3** — linear algebra
- **v0.4** — calculus
- **v0.5** — probability & statistics
- **v0.6** — numerical methods & optimization
- **v0.7** — signal processing
- **v0.8** — image processing & computer vision
- **v0.9** — AI/ML mathematics
- **v1.0** — production-ready scientific computing ecosystem

## Long-Term Vision (v2+)

Automatic differentiation, tensor engine, GPU acceleration (CUDA/ROCm/Metal/Vulkan), distributed computing, symbolic engine, PDE solvers, FEM, scientific simulation, robotics math, quantum computing math, educational examples, plugin architecture.
