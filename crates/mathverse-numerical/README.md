# MathVerse Numerical

[![Crates.io](https://img.shields.io/crates/v/mathverse-numerical.svg)](https://crates.io/crates/mathverse-numerical)
[![docs.rs](https://docs.rs/mathverse-numerical/badge.svg)](https://docs.rs/mathverse-numerical)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Comprehensive numerical methods library for Rust — root finding, ODE integration, interpolation, optimization, numerical linear algebra, and eigenvalue solvers.

---

## Features

- **Root finding** — 12 methods: Bisection, Newton-Raphson, Secant, Brent, Halley, Muller, and more
- **ODE integration** — 6 solvers: RK4, RKF45 (adaptive), Dormand-Prince, Adams-Bashforth, Backward Euler, Crank-Nicolson
- **Interpolation** — 7 methods: Cubic spline, Hermite, Barycentric, RBF, Multilinear, Chebyshev, Nearest neighbor
- **Optimization** — 6 methods: Gradient descent, BFGS, Simulated annealing, Genetic algorithm, Nelder-Mead, Particle swarm
- **Integration** — 9 quadrature methods: Gaussian-Legendre, Romberg, Adaptive Simpson, Monte Carlo, and more
- **Linear solvers** — 8 iterative solvers: Jacobi, Gauss-Seidel, SOR, Conjugate Gradient, GMRES, BiCGSTAB
- **Eigenvalue methods** — 7 solvers: Power method, QR algorithm, Lanczos, Jacobi eigenvalue, and more

## Module Overview

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `root` | Derivative-free and derivative-based root finders | `secant`, `brent`, `halley`, `muller` |
| `ode` | Adaptive and multistep ODE integrators | `RKF45`, `DormandPrince`, `AdamsBashforth` |
| `interpolation` | 1D and multi-dimensional interpolation | `CubicSpline`, `BarycentricInterpolation`, `RBFInterpolation` |
| `optimization` | Global and local optimization | `BFGS`, `SimulatedAnnealing`, `ParticleSwarm` |
| `integration` | Numerical quadrature and Monte Carlo | `GaussianQuadrature`, `RombergIntegration` |
| `linear_solvers` | Iterative solvers for large sparse systems | `ConjugateGradient`, `GMRES`, `BiCGSTAB` |
| `eigenvalue` | Eigenvalue and eigenvector computation | `QRAlgorithm`, `Lanczos`, `JacobiEigenvalue` |

## Installation

```toml
[dependencies]
mathverse-numerical = { path = "crates/mathverse-numerical" }
```

## Quick Start

```rust
use mathverse_numerical::*;

fn main() {
    // Find root of x^2 - 2 = 0 using Newton-Raphson
    let root = newton_raphson(
        &|x| x * x - 2.0,
        &|x| 2.0 * x,
        1.5, 1e-12, 100,
    ).unwrap();
    println!("sqrt(2) ≈ {}", root); // 1.4142135623730951

    // Solve dy/dt = y, y(0) = 1 over [0, 1] using RK4
    let sol = rk4(&|_, y| y, 1.0, 0.0, 1.0, 100);
    println!("e ≈ {}", sol.last().unwrap().1); // 2.71828...

    // Integrate x^2 from 0 to 1 using Gaussian quadrature
    let area = GaussianQuadrature::integrate(&|x| x * x, 0.0, 1.0, 5);
    println!("∫₀¹ x² dx ≈ {}", area); // 0.33333...
}
```

---

## Module: `root` — Root Finding

### Newton's Method Iteration

```
  x_{n+1} = x_n - f(x_n) / f'(x_n)

  Iteration    x_n              f(x_n)
  ─────────────────────────────────────
       0      1.5000000000     0.2500000000
       1      1.4166666667     0.0069444444
       2      1.4142156863     0.0000060073
       3      1.4142135624     0.0000000000
  ─────────────────────────────────────
  Converged to √2 ≈ 1.4142135623730951
```

### Available Methods

| Method | Signature | Requires Derivative | Convergence |
|--------|-----------|:-------------------:|-------------|
| `bisection` | `(f, a, b, tol)` | No | Linear |
| `newton_raphson` | `(f, fp, x0, tol, max_iters)` | Yes | Quadratic |
| `secant` | `(f, x0, x1, tol, max_iters)` | No | ~1.618 |
| `false_position` | `(f, a, b, tol, max_iters)` | No | Linear |
| `brent` | `(f, a, b, tol, max_iters)` | No | Superlinear |
| `muller` | `(f, x0, x1, x2, tol, max_iters)` | No | ~1.618 |
| `illinois` | `(f, a, b, tol, max_iters)` | No | Linear+ |
| `steffensen` | `(f, x0, tol, max_iters)` | No | Quadratic |
| `halley` | `(f, fp, fpp, x0, tol, max_iters)` | Yes (1st+2nd) | Cubic |
| `householder` | `(f, derivatives, x0, order, tol, max_iters)` | Yes | Order+1 |
| `fixed_point` | `(g, x0, tol, max_iters)` | No | Linear |

### Usage

```rust
use mathverse_numerical::{bisection, brent, secant};

// Bisection — guaranteed convergence with bracket
let root = bisection(&|x| x.powi(3) - x - 2.0, 1.0, 2.0, 1e-10).unwrap();

// Brent — fastest bracket method
let root = brent(&|x| x.cos() - x, 0.0, 1.0, 1e-12, 100).unwrap();

// Secant — derivative-free Newton
let root = secant(&|x| x.ln() - 1.0, 2.0, 3.0, 1e-10, 100).unwrap();
```

---

## Module: `ode` — Ordinary Differential Equations

### Available Integrators

| Method | Type | Order | Stiff-Capable | Adaptive |
|--------|------|:-----:|:-------------:|:--------:|
| `rk4` | Explicit RK | 4 | No | No |
| `RKF45` | Explicit RK | 4(5) | No | Yes |
| `DormandPrince` | Explicit RK | 4(5) | No | Yes |
| `AdamsBashforth` | Multistep | 2-5 | No | No |
| `BackwardEuler` | Implicit | 1 | Yes | No |
| `CrankNicolson` | Implicit | 2 | Yes | No |

### Usage

```rust
use mathverse_numerical::{rk4, RKF45, BackwardEuler};

// Simple RK4 integration: dy/dt = -2y, y(0) = 1
let sol = rk4(&|_t, y| -2.0 * y, 1.0, 0.0, 2.0, 200);
let final_y = sol.last().unwrap().1; // ≈ e^(-4) ≈ 0.0183

// Adaptive RKF45 for stiff problem
let rkf = RKF45::new(1e-8, 1.0, 1e-10, 1e-10);
let f = |_: f64, y: &[f64]| vec![-50.0 * y[0]];
let result = rkf.integrate(&f, 0.0, &[1.0], 1.0).unwrap();
```

---

## Module: `interpolation` — Interpolation

### Available Methods

| Method | Input | Continuity | Multi-Dim |
|--------|-------|:----------:|:---------:|
| `CubicSpline` | Data points | C² | No |
| `HermiteInterpolation` | Points + derivatives | C¹ | No |
| `BarycentricInterpolation` | Data points | Polynomial | No |
| `RBFInterpolation` | Scattered data | Smooth | Yes |
| `MultilinearInterpolation` | Regular grid | C⁰ | Yes |
| `ChebyshevInterpolation` | Function | Polynomial | No |
| `NearestNeighbor` | Data points | C⁰ | No |

### Usage

```rust
use mathverse_numerical::{CubicSpline, BarycentricInterpolation};

// Cubic spline through y = x²
let xs = vec![0.0, 1.0, 2.0, 3.0];
let ys = vec![0.0, 1.0, 4.0, 9.0];
let spline = CubicSpline::new(xs, ys).unwrap();
println!("S(1.5) = {}", spline.evaluate(1.5)); // ≈ 2.25
```

---

## Module: `optimization` — Optimization

### Available Methods

| Method | Gradient Required | Global/Local | Derivative-Free |
|--------|:-----------------:|:------------:|:---------------:|
| `GradientDescent` | Yes | Local | No |
| `BFGS` | Yes | Local | No |
| `NelderMead` | No | Local | Yes |
| `SimulatedAnnealing` | No | Global | Yes |
| `GeneticAlgorithm` | No | Global | Yes |
| `ParticleSwarm` | No | Global | Yes |

### Usage

```rust
use mathverse_numerical::{BFGS, SimulatedAnnealing};

// BFGS — minimize f(x,y) = x² + y²
let bfgs = BFGS::new(100, 1e-10);
let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
let (x, fval, _) = bfgs.minimize(&f, &grad, &[3.0, 4.0]).unwrap();
// x ≈ [0.0, 0.0], fval ≈ 0.0
```

---

## Module: `integration` — Numerical Integration

### Available Methods

| Method | Exact For Polynomials Up To | Adaptive | Multi-Dim |
|--------|:---------------------------:|:--------:|:---------:|
| `MidpointRule` | Degree 1 | No | No |
| `SimpsonRule` | Degree 3 | No | No |
| `BooleRule` | Degree 5 | No | No |
| `GaussianQuadrature` | Degree 2n-1 | No | 2D |
| `RombergIntegration` | High | Yes | No |
| `AdaptiveSimpson` | — | Yes | No |
| `MonteCarloIntegration` | — | No | 2D |

### Usage

```rust
use mathverse_numerical::{GaussianQuadrature, RombergIntegration, MonteCarloIntegration};

// Gaussian quadrature (exact for degree ≤ 2n-1 polynomials)
let area = GaussianQuadrature::integrate(&|x| x.powi(4), 0.0, 1.0, 5);
// = 1/5 = 0.2

// Romberg integration (Richardson extrapolation)
let area = RombergIntegration::integrate(&|x| x.sin(), 0.0, std::f64::consts::PI, 10);
// ≈ 2.0
```

---

## Module: `linear_solvers` — Iterative Linear Solvers

### Available Methods

| Method | Matrix Type | Requires Preconditioner |
|--------|-------------|:-----------------------:|
| `Jacobi` | Diagonally dominant | No |
| `GaussSeidel` | Diagonally dominant | No |
| `SOR` | Diagonally dominant | No |
| `ConjugateGradient` | Symmetric positive definite | No |
| `PreconditionedCG` | Symmetric positive definite | Diagonal |
| `GMRES` | General | No |
| `BiCGSTAB` | General | No |
| `ILUPreconditioner` | Any | — |

### Usage

```rust
use mathverse_numerical::{ConjugateGradient, GMRES};

// Solve 4x + y = 1, x + 3y = 2
let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
let b = vec![1.0, 2.0];

let cg = ConjugateGradient::new(100, 1e-10);
let x = cg.solve(&a, &b, None).unwrap();
// x ≈ [0.142857, 0.619048]
```

---

## Module: `eigenvalue` — Eigenvalue Solvers

### Available Methods

| Method | Matrix Type | Computes |
|--------|-------------|----------|
| `PowerMethod` | Any | Dominant eigenvalue |
| `InversePowerMethod` | Any | Eigenvalue nearest shift |
| `RayleighQuotientIteration` | Any | Single eigenvalue |
| `QRAlgorithm` | Any | All eigenvalues |
| `Lanczos` | Symmetric | k eigenvalues |
| `SubspaceIteration` | Any | k eigenvalues |
| `JacobiEigenvalue` | Symmetric | All eigenvalues |

### Usage

```rust
use mathverse_numerical::{PowerMethod, QRAlgorithm, JacobiEigenvalue};

// Power method — dominant eigenvalue of [2,1;1,2] is 3
let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
let pm = PowerMethod::new(1000, 1e-10);
let (lambda, v) = pm.compute(&a, None).unwrap();
// lambda ≈ 3.0
```

---

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE).
