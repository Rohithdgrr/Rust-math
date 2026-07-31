# mathverse-equations

> Equation solving toolkit — polynomial roots, linear systems, nonlinear solvers, differential equations, matrix operations, and discrete dynamical systems.

```
mathverse-equations
├── polynomial      Closed-form solvers for degrees 1–4
├── linear_system   2×2, 3×3, and Gaussian elimination
├── nonlinear       Newton, secant, bisection, Newton system
├── differential    Euler, RK4, system solvers
├── optimization    Golden section, ternary search, Brent minimizer
├── matrix_eq       Gauss solve, inverse, determinant, rank
└── dynamical       Fixed points, orbits, Lyapunov exponents
```

## Features

- **Polynomial solvers**: Closed-form solutions for linear, quadratic, cubic, and quartic equations
- **Linear systems**: Direct solvers for 2×2, 3×3, and arbitrary n×n systems via Gaussian elimination
- **Nonlinear solvers**: Newton-Raphson, secant, bisection, and Newton's method for systems
- **Differential equations**: Euler and Runge-Kutta 4th order for scalar and vector ODEs
- **Optimization**: 1D minimization via golden section, ternary search, Fibonacci search, and Brent's method
- **Matrix operations**: Gauss elimination, matrix inverse, determinant, rank computation
- **Dynamical systems**: Fixed point iteration, orbit computation, Lyapunov exponents, period detection

## Module Overview

| Module | Description | Key Functions |
|--------|-------------|---------------|
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

### Formulas

**Quadratic** (degree 2): `ax² + bx + c = 0`
```
  discriminant = b² - 4ac
  x = (-b ± √discriminant) / (2a)
```

**Cubic** (degree 3): `ax³ + bx² + cx + d = 0` (Cardano's method)
```
  p = c/a - b²/(3a²)
  q = 2b³/(27a³) - bc/(3a²) + d/a
  disc = q²/4 + p³/27
```
Three cases based on discriminant sign (one real root, repeated, or three distinct real roots via trigonometric form).

**Quartic** (degree 4): `ax⁴ + bx³ + cx² + dx + e = 0` (Ferrari's method)
Solved by reducing to a resolvent cubic, then factoring into quadratics.

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

// Evaluate polynomial 1 - 6x + 11x² - x³ at x = 2
let val = polynomial_eval(&[-6.0, 11.0, -6.0, 1.0], 2.0);
assert!(val.abs() < 1e-10); // ≈ 0
```

### Use Cases

- Signal processing (filter design)
- Control systems (pole placement)
- Geometry (curve intersection)

---

## Module: `linear_system` — Linear Systems

### Formulas

**Cramer's Rule** (2×2):
```
  det = a₁₁a₂₂ - a₁₂a₂₁
  x = (b₁a₂₂ - b₂a₁₂) / det
  y = (a₁₁b₂ - a₂₁b₁) / det
```

**Cramer's Rule** (3×3):
```
  det = a₁₁(a₂₂a₃₃ - a₂₃a₃₂) - a₁₂(a₂₁a₃₃ - a₂₃a₃₁) + a₁₃(a₂₁a₃₂ - a₂₂a₃₁)
```

**Gaussian Elimination**: Forward elimination with partial pivoting → back substitution.

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

// 3×3 system
let x = solve_3x3(
    [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]],
    [8.0, -11.0, -3.0],
).unwrap();
// x ≈ [2.0, 3.0, -1.0]

// In-place Gaussian elimination on augmented matrix
let mut m = vec![
    vec![2.0, 1.0, -1.0, 8.0],
    vec![-3.0, -1.0, 2.0, -11.0],
    vec![-2.0, 1.0, 2.0, -3.0],
];
gaussian_elimination(&mut m); // m is now in row echelon form
```

### Use Cases

- Circuit analysis (nodal analysis)
- Structural engineering (force equilibrium)
- Computer graphics (coordinate transforms)

---

## Module: `nonlinear` — Nonlinear Solvers

### Newton's Method Flow

```
  ┌─────────────────────────┐
  │   Choose initial x₀     │
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐
  │   Compute f(xₙ)        │
  └───────────┬─────────────┘
              ▼
  ┌─────────────────────────┐     Yes
  │   |f(xₙ)| < tol?       │──────────▶  Return xₙ
  └───────────┬─────────────┘
              │ No
              ▼
  ┌─────────────────────────┐     Yes
  │   |f'(xₙ)| ≈ 0?        │──────────▶  Return None
  └───────────┬─────────────┘
              │ No
              ▼
  ┌─────────────────────────┐
  │   xₙ₊₁ = xₙ - f(xₙ)   │
  │              / f'(xₙ)   │
  └───────────┬─────────────┘
              │
              └──────▶ Loop
```

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
let x = newton(
    |x| x * x - 2.0,
    |x| 2.0 * x,
    1.0, 1e-15, 100,
).unwrap();
// x ≈ 1.4142135623730951

// Secant: no derivative needed
let x = secant(|x| x * x - 2.0, 1.0, 2.0, 1e-15, 100).unwrap();

// Bisection: guaranteed with bracket
let x = bisection(|x| x * x - 2.0, 0.0, 2.0, 1e-12).unwrap();
```

### Use Cases

- Inverting nonlinear models
- Finding equilibrium in chemical reactions
- Solving transcendental equations (cos(x) = x)

---

## Module: `differential` — Differential Equations

### RK4 Stage Computation

```
  k₁ = f(tₙ, yₙ)
  k₂ = f(tₙ + h/2, yₙ + h·k₁/2)
  k₃ = f(tₙ + h/2, yₙ + h·k₂/2)
  k₄ = f(tₙ + h, yₙ + h·k₃)

  yₙ₊₁ = yₙ + h/6 · (k₁ + 2k₂ + 2k₃ + k₄)
```

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

// System: Lotka-Volterra (predator-prey)
let f = |_t, y: &[f64]| vec![
    y[0] * (1.0 - y[1]),    // dx/dt = x(1-y)
    -y[1] * (1.0 - y[0]),   // dy/dt = -y(1-x)
];
let sol = rk4_system(&f, &[0.5, 0.5], 0.0, 20.0, 0.01);
```

### Use Cases

- Physics simulations (projectile motion, orbital mechanics)
- Population dynamics (epidemiology, ecology)
- Chemical kinetics (reaction rates)

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

// Ternary search
let x = ternary_search(|x| (x - 3.0).powi(2), 0.0, 6.0, 1e-10);
assert!((x - 3.0).abs() < 1e-8);

// Brent's method (combines golden section with parabolic interpolation)
let x = brent_min(|x| (x - 1.5).powi(2), 0.0, 3.0, 1e-10);
assert!((x - 1.5).abs() < 1e-8);
```

### Use Cases

- Finding optimal parameters in unimodal functions
- Line search in multivariate optimization
- Engineering design (single-variable optimization)

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

// Solve Ax = b
let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
let b = vec![5.0, 7.0];
let x = solve_gauss(&a, &b).unwrap();
// x ≈ [1.6, 1.8]

// Matrix inverse
let inv = matrix_inverse(&a).unwrap();
// inv[0][0] ≈ 0.6, inv[0][1] ≈ -0.2

// Determinant
let det = determinant(&a).unwrap();
// det = 5.0

// Rank
let r = rank(&a);
// r = 2
```

### Use Cases

- Computer graphics (transformation matrices)
- Control theory (system analysis)
- Data science (covariance matrix operations)

---

## Module: `dynamical` — Discrete Dynamical Systems

### Cobweb Diagram

```
  y
  │      y = g(x)
  │     ╱
  │    ╱   ●─────●
  │   ╱   ╱│    ╱
  │  ╱   ╱ │   ╱
  │ ╱   ╱  │  ╱
  │╱   ╱   │ ╱
  ┼───╱────│╱───────── x
  │  ╱     │
  │ ╱      ●  y = x
  │╱
  
  Orbit: x₀ → g(x₀) → g²(x₀) → ... → x*
```

### Available Functions

| Function | Description |
|----------|-------------|
| `fixed_point(g, x0, tol, max_iter)` | Find fixed point x* = g(x*) |
| `iterate_map(f, x0, n)` | Iterate xₙ₊₁ = f(xₙ) n times |
| `cobweb(g, x0, n)` | Generate cobweb diagram points |
| `lyapunov_exponent(f, df, x0, n)` | Compute Lyapunov exponent |
| `basin_of_attraction(g, x0, tol, max_iter)` | Find attractor basin |
| `orbit(g, x0, n)` | Compute orbit trajectory |
| `period(g, x0, tol, max_period)` | Detect periodicity |

### Usage

```rust
use mathverse_equations::{fixed_point, lyapunov_exponent, period};

// Fixed point: √2 via x_{n+1} = (x_n + 2/x_n) / 2
let x = fixed_point(|x| (x + 2.0 / x) / 2.0, 1.0, 1e-15, 100).unwrap();
assert!((x - 2.0_f64.sqrt()).abs() < 1e-12);

// Lyapunov exponent of logistic map r*x*(1-x), r=2
let l = lyapunov_exponent(
    |x| 2.0 * x * (1.0 - x),
    |x| 2.0 - 4.0 * x,
    0.3, 1000,
);
// l > 0 indicates chaos

// Period detection
let p = period(|x| -1.5 * x * x + 1.5, 0.1, 1e-10, 10);
// p = Some(period)
```

### Use Cases

- Chaos theory analysis
- Iterative method convergence analysis
- Bifurcation studies

---

## Future Scope

- [ ] Root finding for complex polynomials
- [ ] Resultant and GCD of polynomials
- [ ] Symbolic polynomial manipulation
- [ ] Sparse linear system solvers
- [ ] BVP (boundary value problem) solvers
- [ ] Chaos analysis tools (bifurcation diagrams)
- [ ] Stochastic differential equation solvers
- [ ] Multi-precision arithmetic support

## License

MIT OR Apache-2.0
