# mathverse-calculus

[![Crates.io](https://img.shields.io/crates/v/mathverse-calculus.svg)](https://crates.io/crates/mathverse-calculus)
[![docs.rs](https://docs.rs/mathverse-calculus/badge.svg)](https://docs.rs/mathverse-calculus)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Numerical calculus for Rust: derivatives, integration, vector calculus, and ODE solvers.

Part of the [MathVerse](https://github.com/Rohithdgrr/Rust-math) ecosystem.

## Features

- **Derivatives**: Central differences, partial derivatives, nth-order, discrete gradient
- **Integration**: Trapezoid, Simpson, adaptive, Gaussian quadrature, Romberg, 2D
- **ODE Solvers**: Euler, midpoint, RK4 with builder API
- **Vector Calculus**: Gradient, divergence, curl, Laplacian, Jacobian, Hessian
- **Root Finding**: Newton-Raphson with auto-differentiation, critical point finding
- `#![forbid(unsafe_code)]` — fully safe Rust
- `no_std` compatible (with feature flag)

## Installation

```toml
[dependencies]
mathverse-calculus = "0.1"
```

## Quick Start

```rust
use mathverse_calculus::prelude::*;

fn main() {
    // Derivative of sin at 0 ≈ 1
    let d = derivative(&f64::sin, 0.0);
    assert!((d - 1.0).abs() < 1e-8);

    // Integral of sin from 0 to π ≈ 2
    let i = integrate(&f64::sin, 0.0, core::f64::consts::PI, 1e-10);
    assert!((i - 2.0).abs() < 1e-8);

    // Solve dy/dt = y, y(0) = 1 → y = e^t
    let sol = OdeProblem::new(&|_, y| y, (0.0, 1.0), 1.0).solve().unwrap();
    let y_final = sol.last().unwrap().1;
    assert!((y_final - 1.0_f64.exp()).abs() < 1e-6);
}
```

## Usage

### Derivatives

```rust
use mathverse_calculus::derivative::*;

// First derivative: f'(x)
let d = derivative(&|x| x * x * x, 2.0);  // ≈ 12.0

// Second derivative: f''(x)
let d2 = second_derivative(&f64::sin, 0.0);  // ≈ 0.0

// Nth derivative with error estimate
let (val, err) = nth_derivative(&|x| x.powi(5), 1.0, 5);  // ≈ 120.0

// Partial derivative: ∂f/∂x_i
let pd = partial_derivative(&|x| x[0] * x[1], &[2.0, 3.0], 0);  // ≈ 3.0
```

### Integration

```rust
use mathverse_calculus::integrate::*;

// Trapezoid rule
let t = trapezoid(&f64::sin, 0.0, PI, 1000).unwrap();

// Simpson's rule
let s = simpson(&f64::sin, 0.0, PI, 100).unwrap();

// Adaptive Simpson (recommended for most uses)
let a = integrate(&f64::sin, 0.0, PI, 1e-10);

// Gaussian quadrature (exact for polynomials up to degree 2n-1)
let g = gaussian_quadrature(&|x| x * x, 0.0, 1.0, 5).unwrap();

// Romberg integration
let r = romberg(&f64::sin, 0.0, PI, 10, 1e-12).unwrap();

// 2D integration
let i2d = integrate_2d(&|x, y| x * y, 0.0, 1.0, 0.0, 1.0, 5).unwrap();
```

### ODE Solvers

```rust
use mathverse_calculus::ode::*;

// Direct function calls
let sol = runge_kutta_4(&|_, y| -y, 0.0, 1.0, 1.0, 100).unwrap();

// Builder API (scipy-like)
let sol = OdeProblem::new(&|_, y| y, (0.0, 1.0), 1.0)
    .method(OdeMethod::Rk4)
    .steps(1000)
    .solve()
    .unwrap();

// Systems of ODEs (harmonic oscillator)
let osc = runge_kutta_4_system(
    &|_, y| vec![y[1], -y[0]],
    0.0, &[1.0, 0.0], 2.0 * PI, 1000,
).unwrap();
```

### Vector Calculus

```rust
use mathverse_calculus::vector_calculus::*;

let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];

// Gradient: ∇f
let g = gradient(&f, &[1.0, 2.0]);  // ≈ [2.0, 4.0]

// Laplacian: ∇²f
let l = laplacian(&f, &[1.0, 2.0]);  // ≈ 4.0

// Directional derivative: ∇f · v
let d = directional_derivative(&f, &[1.0, 2.0], &[1.0, 0.0]).unwrap();  // ≈ 2.0
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `std` | Yes | Enable standard library; disabling gives `no_std` |

## Performance

- All algorithms are cache-friendly with minimal allocations
- Hot paths designed for future SIMD acceleration
- Benchmarks available via `cargo bench`

## Python Parity

| Rust function | SciPy/NumPy equivalent |
|---|---|
| `nth_derivative` | `scipy.misc.derivative` |
| `gaussian_quadrature` | `scipy.integrate.fixed_quad` |
| `romberg` | `scipy.integrate.romberg` |
| `integrate_2d` | `scipy.integrate.dblquad` |
| `OdeProblem` | `scipy.integrate.solve_ivp` |
| `discrete_gradient` | `numpy.gradient` |

## License

Licensed under [MIT](../LICENSE) OR [Apache-2.0](../LICENSE-APACHE).

## Contributing

See the [MathVerse contributing guide](../CONTRIBUTING.md).
