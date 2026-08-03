# MathVerse Calculus

[![Crates.io](https://img.shields.io/crates/v/mathverse-calculus.svg)](https://crates.io/crates/mathverse-calculus)
[![docs.rs](https://docs.rs/mathverse-calculus/badge.svg)](https://docs.rs/mathverse-calculus)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Numerical calculus: derivatives, integration, vector calculus, ODEs, and root finding.

---

## Features

- **Numerical derivatives** — first, second, nth (central differences), partial derivatives
- **Integration** — adaptive Simpson, Gaussian quadrature, composite trapezoid/Simpson
- **Vector calculus** — gradient, divergence, curl, Laplacian, Jacobian, Hessian, directional derivatives
- **ODE solvers** — Euler, Midpoint, RK4 for scalar ODEs and systems
- **Root finding** — Bisection, Newton-Raphson, Secant, False Position, auto-derivative Newton
- All functions accept closures for natural composition

## Module Overview

| Module | Purpose |
|--------|---------|
| `derivative` | First, second, nth derivatives; partial derivatives (central differences) |
| `integrate` | Trapezoid, Simpson, adaptive Simpson, Gaussian quadrature |
| `vector_calculus` | Gradient, divergence, curl, Laplacian, Jacobian, Hessian, directional derivative |
| `ode` | Euler, Midpoint, RK4 for scalar ODEs; RK4 for systems |
| `root_finding` | Bisection, Newton-Raphson, Secant, False Position, auto-derivative Newton |

## Installation

```toml
[dependencies]
mathverse-calculus = "0.1"
```

## Quick Start

```rust
use mathverse_calculus::derivative::derivative;
use mathverse_calculus::integrate::integrate;
use mathverse_calculus::ode::runge_kutta_4;
use mathverse_calculus::root_finding::newton_raphson;

fn main() {
    // Derivative of sin(x) at x=0
    let slope = derivative(&f64::sin, 0.0);
    println!("sin'(0) = {:.6}", slope);  // 1.000000

    // Integrate sin(x) from 0 to π → should be 2.0
    let area = integrate(&f64::sin, 0.0, std::f64::consts::PI, 1e-10);
    println!("∫₀^π sin(x) dx = {:.6}", area);  // 2.000000

    // Solve dy/dt = y, y(0)=1 → y = e^t
    let result = runge_kutta_4(&|_t, y| y, 0.0, 1.0, 1.0, 10);
    let y_final = result.last().unwrap().1;
    println!("y(1) = {:.6}", y_final);  // ≈ 2.718282

    // Find root of x² - 4 = 0 starting from 3
    let root = newton_raphson(
        &|x| x * x - 4.0,
        &|x| 2.0 * x,
        3.0, 1e-10, 100,
    ).unwrap();
    println!("√4 = {:.6}", root);  // 2.000000
}
```

---

## Per-Module Documentation

### Derivative (`derivative`)

Central-difference numerical derivatives with adaptive step size.

```rust
use mathverse_calculus::derivative::*;

// First derivative: f'(x)
let slope = derivative(&f64::sin, 0.0);

// Second derivative: f''(x)
let concavity = second_derivative(&|x| x * x * x, 2.0);

// Partial derivative: ∂f/∂x_i
let f = |x: &[f64]| x[0] * x[0] * x[1];
let df_dx0 = partial_derivative(&f, &[2.0, 3.0], 0);

// nth derivative: f⁽ⁿ⁾(x)
let d3 = nth_derivative(&|x| x * x * x, 2.0, 3);
```

### Integration (`integrate`)

```rust
use mathverse_calculus::integrate::*;

// Composite trapezoid rule
let area = trapezoid(&f64::sin, 0.0, PI, 1024);

// Composite Simpson's rule (n must be even)
let area = simpson(&f64::sin, 0.0, PI, 64);

// Adaptive Simpson (auto-refines to tolerance)
let area = integrate(&f64::sin, 0.0, PI, 1e-10);

// Gaussian quadrature (n points, exact for degree ≤ 2n-1)
let area = gaussian_quadrature(&|x| x * x, 0.0, 1.0, 3);
```

### Vector Calculus (`vector_calculus`)

```rust
use mathverse_calculus::vector_calculus::*;

// Gradient: ∇f = [∂f/∂x₀, ∂f/∂x₁, ...]
let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
let g = gradient(&f, &[1.0, 2.0]); // [2.0, 4.0]

// Laplacian: ∇²f
let lap = laplacian(&f, &[1.0, 2.0]); // 4.0

// Jacobian, Hessian, divergence, curl also available
```

### ODE Solvers (`ode`)

```rust
use mathverse_calculus::ode::*;

// RK4: 4th-order, excellent accuracy
let result = runge_kutta_4(&|_t, y| y, 0.0, 1.0, 1.0, 10);
// y(1) ≈ 2.718282 (error < 1e-6 with just 10 steps)

// System of ODEs: harmonic oscillator d²x/dt² = -x
let f = |t: f64, y: &[f64]| vec![y[1], -y[0]];
let result = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 2.0 * PI, 100);
```

### Root Finding (`root_finding`)

```rust
use mathverse_calculus::root_finding::*;

// Newton-Raphson: quadratic convergence, needs f'
let root = newton_raphson(
    &|x| x * x - 4.0,
    &|x| 2.0 * x,
    3.0, 1e-10, 100,
).unwrap();

// Secant: superlinear, no derivative needed
let root = secant(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
```

---

## Future Scope

- Symbolic differentiation (integration with `mathverse-symbolic`)
- Multi-dimensional integration (Monte Carlo, cubature)
- Stochastic ODE solvers (Euler-Maruyama, Milstein)
- Boundary value problem solvers (shooting method, finite differences)
- Automatic differentiation (forward/reverse mode)
- Adaptive step-size ODE solvers (Dormand-Prince, Adams-Bashforth)

## License

MIT — see [LICENSE](LICENSE).
