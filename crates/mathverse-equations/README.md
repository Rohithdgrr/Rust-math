# MathVerse Equations

[![Crates.io](https://img.shields.io/crates/v/mathverse-equations.svg)](https://crates.io/crates/mathverse-equations)
[![docs.rs](https://docs.rs/mathverse-equations/badge.svg)](https://docs.rs/mathverse-equations)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Equation solving toolkit — polynomial roots, linear systems, nonlinear solvers, differential equations, matrix operations, and discrete dynamical systems.

---

## Features

- **Polynomial solvers** — Closed-form solutions for linear, quadratic, cubic, and quartic equations
- **Linear systems** — Direct solvers for 2×2, 3×3, and arbitrary n×n systems via Gaussian elimination
- **Nonlinear solvers** — Newton-Raphson, secant, bisection, and Newton's method for systems
- **Differential equations** — Euler and Runge-Kutta 4th order for scalar and vector ODEs
- **Optimization** — 1D minimization via golden section, ternary search, Fibonacci search, and Brent's method
- **Matrix operations** — Gauss elimination, matrix inverse, determinant, rank computation
- **Dynamical systems** — Fixed point iteration, orbit computation, Lyapunov exponents, period detection

## Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `polynomial` | Closed-form polynomial root finders | `solve_quadratic`, `solve_cubic`, `solve_quartic` |
| `linear_system` | Direct linear system solvers | `solve_2x2`, `solve_3x3`, `gaussian_elimination` |
| `nonlinear` | Iterative nonlinear solvers | `newton`, `secant`, `bisection`, `newton_system` |
| `differential` | ODE initial value solvers | `euler`, `runge_kutta4`, `rk4_system` |
| `optimization` | 1D function minimization | `golden_section`, `ternary_search`, `brent_min` |
| `matrix_eq` | Matrix algebra operations | `solve_gauss`, `matrix_inverse`, `determinant`, `rank` |
| `dynamical` | Discrete dynamical systems | `fixed_point`, `iterate_map`, `lyapunov_exponent` |

## Installation

```toml
[dependencies]
mathverse-equations = { path = "crates/mathverse-equations" }
```

## Quick Start

```rust
use mathverse_equations::*;

fn main() {
    // Solve x² - 3x + 2 = 0
    let roots = solve_quadratic(1.0, -3.0, 2.0);
    println!("Roots: {:?}", roots); // [1.0, 2.0]

    // Solve linear system: 2x + y = 5, x + 3y = 7
    let x = solve_linear_system(
        &vec![vec![2.0, 1.0], vec![1.0, 3.0]],
        &vec![5.0, 7.0],
    ).unwrap();
    println!("x = {}, y = {}", x[0], x[1]); // x=1.6, y=1.8

    // Solve dy/dt = y, y(0)=1
    let sol = runge_kutta4(|_t, y| y, 1.0, 0.0, 1.0, 0.01);
    println!("e ≈ {}", sol.last().unwrap().1); // 2.71828...
}
```

---

## Module: `polynomial` — Polynomial Root Finding

### Available Functions

| Function | Degree | Returns |
|----------|:------:|---------|
| `solve_linear(a, b)` | 1 | `Vec<f64>` |
| `solve_quadratic(a, b, c)` | 2 | `Vec<f64>` |
| `solve_cubic(a, b, c, d)` | 3 | `Vec<f64>` |
| `solve_quartic(a, b, c, d, e)` | 4 | `Vec<f64>` |
| `polynomial_eval(coeffs, x)` | any | `f64` |

### Usage

```rust
use mathverse_equations::{solve_quadratic, solve_cubic, polynomial_eval};

// Quadratic: x² - 5x + 6 = 0 → roots: 2, 3
let roots = solve_quadratic(1.0, -5.0, 6.0);
assert_eq!(roots.len(), 2);

// Cubic: x³ - 6x² + 11x - 6 = 0 → roots: 1, 2, 3
let roots = solve_cubic(1.0, -6.0, 11.0, -6.0);
assert_eq!(roots.len(), 3);
```

---

## Module: `linear_system` — Linear Systems

### Available Functions

| Function | Input | Returns |
|----------|-------|---------|
| `solve_2x2(a, b)` | 2×2 matrix + vector | `Option<[f64; 2]>` |
| `solve_3x3(a, b)` | 3×3 matrix + vector | `Option<[f64; 3]>` |
| `gaussian_elimination(matrix)` | Augmented matrix | `bool` (in-place) |

### Usage

```rust
use mathverse_equations::{solve_2x2, solve_3x3, gaussian_elimination};

// 2x + y = 5, x + 3y = 7
let x = solve_2x2([[2.0, 1.0], [1.0, 3.0]], [5.0, 7.0]).unwrap();
println!("x = {}, y = {}", x[0], x[1]); // [1.6, 1.8]
```

---

## Module: `nonlinear` — Nonlinear Solvers

### Available Functions

| Function | Method | Requires Derivative | System? |
|----------|--------|:-------------------:|:-------:|
| `newton` | Newton-Raphson | Yes | No |
| `secant` | Secant | No | No |
| `bisection` | Bisection | No | No |
| `newton_system` | Newton for systems | Yes (Jacobian) | Yes |

### Usage

```rust
use mathverse_equations::{newton, secant, bisection};

// Newton: find √2 (root of x² - 2)
let x = newton(|x| x * x - 2.0, |x| 2.0 * x, 1.0, 1e-15, 100).unwrap();
// x ≈ 1.4142135623730951
```

---

## Module: `differential` — Differential Equations

### Available Functions

| Function | Method | System? |
|----------|--------|:-------:|
| `euler(f, y0, t0, tf, h)` | Forward Euler | No |
| `runge_kutta4(f, y0, t0, tf, h)` | RK4 | No |
| `euler_system(f, y0, t0, tf, h)` | Forward Euler | Yes |
| `rk4_system(f, y0, t0, tf, h)` | RK4 | Yes |

### Usage

```rust
use mathverse_equations::{euler, runge_kutta4, rk4_system};

// Scalar: dy/dt = y, y(0) = 1 → y(t) = e^t
let sol = runge_kutta4(|_t, y| y, 1.0, 0.0, 1.0, 0.01);
let y_final = sol.last().unwrap().1;
assert!((y_final - std::f64::consts::E).abs() < 0.0001);
```

---

## Module: `optimization` — 1D Optimization

### Available Functions

| Function | Method | Complexity |
|----------|--------|:----------:|
| `golden_section(f, a, b, tol)` | Golden section | O(log(1/ε)) |
| `ternary_search(f, a, b, tol)` | Ternary search | O(log(1/ε)) |
| `fibonacci_search(f, a, b, n)` | Fibonacci search | O(n) |
| `brent_min(f, a, b, tol)` | Brent's method | O(log(1/ε)) |

### Usage

```rust
use mathverse_equations::{golden_section, ternary_search, brent_min};

// Minimize (x - 2)² on [0, 5]
let x = golden_section(|x| (x - 2.0).powi(2), 0.0, 5.0, 1e-10);
assert!((x - 2.0).abs() < 1e-8);
```

---

## Module: `matrix_eq` — Matrix Operations

### Available Functions

| Function | Description | Complexity |
|----------|-------------|:----------:|
| `solve_gauss(a, b)` | Solve Ax = b | O(n³) |
| `matrix_inverse(a)` | Compute A⁻¹ | O(n³) |
| `determinant(a)` | Compute det(A) | O(n³) |
| `rank(a)` | Compute rank(A) | O(n³) |

### Usage

```rust
use mathverse_equations::{solve_gauss, matrix_inverse, determinant, rank};

let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
let det = determinant(&a).unwrap(); // 5.0
let inv = matrix_inverse(&a).unwrap();
```

---

## Module: `dynamical` — Discrete Dynamical Systems

### Available Functions

| Function | Description |
|----------|-------------|
| `fixed_point(g, x0, tol, max_iter)` | Find fixed point x* = g(x*) |
| `iterate_map(f, x0, n)` | Iterate xₙ₊₁ = f(xₙ) n times |
| `cobweb(g, x0, n)` | Generate cobweb diagram points |
| `lyapunov_exponent(f, df, x0, n)` | Compute Lyapunov exponent |
| `orbit(g, x0, n)` | Compute orbit trajectory |
| `period(g, x0, tol, max_period)` | Detect periodicity |

### Usage

```rust
use mathverse_equations::{fixed_point, lyapunov_exponent, period};

// Fixed point: √2 via x_{n+1} = (x_n + 2/x_n) / 2
let x = fixed_point(|x| (x + 2.0 / x) / 2.0, 1.0, 1e-15, 100).unwrap();
assert!((x - 2.0_f64.sqrt()).abs() < 1e-12);
```

---

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE).
