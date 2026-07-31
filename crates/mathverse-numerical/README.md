# mathverse-numerical

> Comprehensive numerical methods library for Rust — root finding, ODE integration, interpolation, optimization, numerical linear algebra, and eigenvalue solvers.

```
mathverse-numerical
├── root          Root finding (Newton, Brent, Halley, Secant, ...)
├── ode           ODE integrators (RKF45, Dormand-Prince, Adams-Bashforth, ...)
├── interpolation Spline, Hermite, Barycentric, RBF, Chebyshev, ...
├── optimization  Gradient descent, BFGS, SA, GA, Nelder-Mead, PSO
├── integration   Gaussian, Romberg, Simpson, Monte Carlo, ...
├── linear_solvers Jacobi, Gauss-Seidel, SOR, CG, GMRES, BiCGSTAB
└── eigenvalue    Power method, QR, Lanczos, Jacobi eigenvalue, ...
```

## Features

- **Root finding**: Bisection, Newton-Raphson, Secant, False Position, Brent, Muller, Illinois, Steffensen, Halley, Householder, Fixed Point, Aitken delta-squared
- **ODE integration**: RK4, RKF45 (adaptive), Dormand-Prince (adaptive), Adams-Bashforth (multistep), Backward Euler (implicit/stiff), Crank-Nicolson (implicit)
- **Interpolation**: Cubic spline, Hermite, Barycentric, RBF (multiquadric), Multilinear, Chebyshev, Nearest neighbor
- **Optimization**: Gradient descent (+ momentum), BFGS, Simulated annealing, Genetic algorithm, Nelder-Mead, Particle swarm
- **Integration**: Gaussian-Legendre, Romberg, Adaptive Simpson, Monte Carlo, Simpson, Midpoint, Boole, Clenshaw-Curtis, Double exponential
- **Linear solvers**: Jacobi, Gauss-Seidel, SOR, Conjugate Gradient, Preconditioned CG, GMRES, BiCGSTAB, ILU preconditioner
- **Eigenvalue methods**: Power method, Inverse power method, Rayleigh quotient iteration, QR algorithm, Lanczos, Subspace iteration, Jacobi eigenvalue

## Module Overview

| Module | Description | Key Types |
|--------|-------------|-----------|
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

### Bisection Method Steps

```
  f(x) = x² - 2,  bracket [0, 2]

  Step   a        b        m        f(m)
  ──────────────────────────────────────
   1    0.0000   2.0000   1.0000  -1.0000
   2    1.0000   2.0000   1.5000   0.2500
   3    1.0000   1.5000   1.2500  -0.4375
   4    1.2500   1.5000   1.3750  -0.1094
   5    1.3750   1.5000   1.4375   0.0664
   ...
   40   1.4142   1.4143   1.4142   0.0000
  ──────────────────────────────────────
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

### Use Cases

- Equation solving in scientific computing
- Finding equilibrium points in physical systems
- Inverting nonlinear models

---

## Module: `ode` — Ordinary Differential Equations

### Adaptive Step Control (RKF45)

```
  Error = ||y_4th - y_5th|| / (abs_tol + rel_tol * |y_5th|)

  If error ≤ 1.0:  Accept step, adjust h ↑
  If error > 1.0:  Reject step, adjust h ↓

  h_new = 0.9 * h * (1/error)^0.2
```

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

// Backward Euler for stiff: dy/dt = -1000y
let be = BackwardEuler::new(100, 1e-10);
let f = |_: f64, y: &[f64]| vec![-1000.0 * y[0]];
let jac = |_: f64, _: &[f64]| vec![vec![-1000.0]];
let result = be.integrate(&f, &jac, 0.0, &[1.0], 1.0, 200).unwrap();
```

### Use Cases

- Simulating dynamical systems (physics, biology)
- Circuit simulation (stiff ODEs)
- Chemical kinetics

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
println!("S'(1.5) = {}", spline.derivative(1.5)); // ≈ 3.0

// Barycentric — numerically stable polynomial interpolation
let bary = BarycentricInterpolation::new(
    vec![0.0, 1.0, 2.0, 3.0],
    vec![0.0, 1.0, 4.0, 9.0],
).unwrap();
println!("P(1.5) = {}", bary.evaluate(1.5)); // 2.25
```

### Use Cases

- Curve fitting in CAD/graphics
- Resampling time series data
- Surface reconstruction from point clouds

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

// Simulated annealing — minimize Rastrigin function (many local minima)
let sa = SimulatedAnnealing::new(100.0, 0.95, 0.01, 5000);
let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
let bounds = [(-5.0, 5.0), (-5.0, 5.0)];
let mut rng = rand::thread_rng();
let (x, fval) = sa.minimize(&f, &[0.0, 0.0], &bounds, &mut rng);
```

### Use Cases

- Machine learning model training
- Engineering design optimization
- Portfolio optimization in finance

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
| `ClenshawCurtis` | — | No | No |
| `DoubleExponential` | — | No | No |
| `MonteCarloIntegration` | — | No | 2D |

### Usage

```rust
use mathverse_numerical::{
    GaussianQuadrature, RombergIntegration, AdaptiveSimpson, MonteCarloIntegration
};

// Gaussian quadrature (exact for degree ≤ 2n-1 polynomials)
let area = GaussianQuadrature::integrate(&|x| x.powi(4), 0.0, 1.0, 5);
// = 1/5 = 0.2

// Romberg integration (Richardson extrapolation)
let area = RombergIntegration::integrate(&|x| x.sin(), 0.0, std::f64::consts::PI, 10);
// ≈ 2.0

// Adaptive Simpson
let area = AdaptiveSimpson::integrate(&|x| (-x * x).exp(), -10.0, 10.0, 1e-10, 30);

// Monte Carlo with error estimate
let (area, err) = MonteCarloIntegration::integrate(&|x| x * x, 0.0, 1.0, 100_000);
```

### Use Cases

- Computing areas, volumes, and moments
- Probabilistic integration of high-dimensional integrals
- Physics simulations (path integrals)

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

let gmres = GMRES::new(100, 1e-10, 50);
let x = gmres.solve(&a, &b, None).unwrap();
```

### Use Cases

- Finite element method (FEM) systems
- Computational fluid dynamics (CFD)
- Large sparse systems from PDE discretization

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

// QR algorithm — all eigenvalues
let qr = QRAlgorithm::new(1000, 1e-10);
let eigenvalues = qr.compute(&a).unwrap();
// eigenvalues ≈ [1.0, 3.0]

// Jacobi — for symmetric matrices
let jacobi = JacobiEigenvalue::new(1000, 1e-10);
let eigenvalues = jacobi.compute(&a).unwrap();
// eigenvalues ≈ [1.0, 3.0]
```

### Use Cases

- Principal component analysis (PCA)
- Vibrational analysis in structural mechanics
- Quantum mechanics (energy eigenvalues)

---

## Feature Comparison with Other Crates

| Feature | mathverse-numerical | ndarray | nalgebra |
|---------|:-------------------:|:-------:|:--------:|
| Root finding | ✅ 12 methods | ❌ | ❌ |
| ODE integration | ✅ 6 solvers | ❌ | ❌ |
| Interpolation | ✅ 7 methods | ❌ | ❌ |
| Optimization | ✅ 6 methods | ❌ | ❌ |
| Integration | ✅ 9 methods | ❌ | ❌ |
| Linear solvers | ✅ 8 methods | Partial | Partial |
| Eigenvalue | ✅ 7 methods | ❌ | Partial |

## Future Scope

- [ ] Boundary value problem (BVP) solvers
- [ ] PDE solvers (FDM, FEM, FVM)
- [ ] Automatic differentiation
- [ ] Interval arithmetic
- [ ] Sparse matrix support with explicit CSR/CSC formats
- [ ] GPU acceleration for large-scale problems
- [ ] Parallel ODE solvers for coupled systems
- [ ] More optimization: L-BFGS-B, trust-region methods

## License

MIT OR Apache-2.0
