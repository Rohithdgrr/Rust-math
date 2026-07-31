# mathverse-calculus

**Numerical calculus: derivatives, integration, vector calculus, ODEs, and root finding.**

`mathverse-calculus` provides pure-Rust numerical methods for differentiation,
integration, vector calculus operations, ODE solving, and root finding. All
functions accept closures, making composition natural.

## Features

- Numerical derivatives: first, second, nth (central differences)
- Partial derivatives of multivariate functions
- Adaptive Simpson integration
- Gaussian quadrature (Legendre nodes, up to degree 2n-1 exact)
- Composite trapezoid and Simpson rules
- Vector calculus: gradient, divergence, curl, Laplacian, Jacobian, Hessian
- Directional derivatives
- ODE solvers: Euler, Midpoint, RK4 (scalar and systems)
- Root finding: Bisection, Newton-Raphson, Secant, False Position
- Auto-differentiation via central differences

## Module Overview

| Module | Description |
|---|---|
| `derivative` | First, second, nth derivatives; partial derivatives (central diff) |
| `integrate` | Trapezoid, Simpson, adaptive Simpson, Gaussian quadrature |
| `vector_calculus` | Gradient, divergence, curl, Laplacian, Jacobian, Hessian, directional derivative |
| `ode` | Euler, Midpoint, RK4 for scalar ODEs; RK4 for systems |
| `root_finding` | Bisection, Newton-Raphson, Secant, False Position, auto-derivative Newton |

## ASCII Art: Derivative & Integral Visualization

```
Derivative: Central Difference
==============================

        f(x+h)
         *
        /|
       / |
      /  |  f(x+h) - f(x-h)
     /   |  ───────────────
    /    |     2h = slope
   *─────*─────→
  f(x-h) f(x)

  f'(x) ≈ [f(x+h) - f(x-h)] / 2h     Error: O(h²)


Second Derivative: Central Difference
======================================

  f(x+h)        *
      \        /|
       \      / |
        \    /  |  f(x+h) - 2f(x) + f(x-h)
         \  /   |  ─────────────────────
          \/    |        h²
  f(x) ----*----*--------→
           /\
          /  \
         /    \
  f(x-h)*

  f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²


Integration: Simpson's Rule
============================

    f(x)
    │    *
    │   / \         S = h/3 [f(a) + 4f(mid) + f(b)]
    │  /   \
    │ *─────*       Error: O(h⁴)
    │╱  mid  ╲
    └──────────→ x
     a        b

    Composite: n subintervals, n even
    ┌───┬───┬───┬───┐
    │ 4 │ 2 │ 4 │ 2 │ ... pattern: 4,2,4,2,...,4
    └───┴───┴───┴───┘
    × h/3


Gaussian Quadrature (3 points)
===============================

    Legendre nodes on [-1, 1]:

    *         ●         *
    ├────●────┼────●────┤
   -0.774  0.0   0.774

    Weights: [0.556, 0.889, 0.556]
    Exact for polynomials of degree ≤ 5
```

## Installation

```toml
[dependencies]
mathverse-calculus = { path = "../mathverse-calculus" }
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

## Per-Module Documentation

### Derivative (`derivative`)

Central-difference numerical derivatives with adaptive step size.

```rust
use mathverse_calculus::derivative::*;

// First derivative: f'(x)
let slope = derivative(&f64::sin, 0.0);
// sin'(0) = cos(0) = 1.0

// Second derivative: f''(x)
let concavity = second_derivative(&|x| x * x * x, 2.0);
// (x³)'' = 6x → 12.0

// Partial derivative: ∂f/∂x_i
let f = |x: &[f64]| x[0] * x[0] * x[1];
let df_dx0 = partial_derivative(&f, &[2.0, 3.0], 0);
// ∂f/∂x₀ = 2·x₀·x₁ = 12.0

let df_dx1 = partial_derivative(&f, &[2.0, 3.0], 1);
// ∂f/∂x₁ = x₀² = 4.0

// nth derivative: f⁽ⁿ⁾(x)
let d3 = nth_derivative(&|x| x * x * x, 2.0, 3);
// (x³)''' = 6.0
```

**Formulas:**

```
f'(x)   ≈ [f(x+h) - f(x-h)] / 2h          O(h²)
f''(x)  ≈ [f(x+h) - 2f(x) + f(x-h)] / h²  O(h²)
f⁽ⁿ⁾(x) ≈ Σ_{k=0}^{n} (-1)^(n-k) C(n,k) f(x + (k-n/2)h) / hⁿ

h scales with |x| for relative accuracy away from zero.
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
// → 2.0

// Gaussian quadrature (n points, exact for degree ≤ 2n-1)
let area = gaussian_quadrature(&|x| x * x, 0.0, 1.0, 3);
// → 1/3

// Gaussian quadrature for smooth functions
let gaussian = gaussian_quadrature(
    &|x| (-x * x).exp(),
    -3.0, 3.0, 5,
);
```

**Comparison:**

```
┌──────────────────┬────────────┬───────────┬──────────────────┐
│ Method           │ Error      │ Exact for │ Cost per eval    │
├──────────────────┼────────────┼───────────┼──────────────────┤
│ Trapezoid        │ O(h²)      │ linear    │ 1 f eval / step  │
│ Simpson          │ O(h⁴)      │ cubic     │ 3 f eval / 2 step│
│ Adaptive Simpson │ O(tol)     │ adaptive  │ recursive        │
│ Gaussian (n)     │ O(h^2n)    │ deg ≤2n-1 │ n nodes × weights│
└──────────────────┴────────────┴───────────┴──────────────────┘
```

### Vector Calculus (`vector_calculus`)

```rust
use mathverse_calculus::vector_calculus::*;

// Gradient: ∇f = [∂f/∂x₀, ∂f/∂x₁, ...]
let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
let g = gradient(&f, &[1.0, 2.0]);
// [2.0, 4.0]

// Divergence: ∇·F = Σ ∂F_i/∂x_i
let f = |x: &[f64]| vec![x[0], x[1], x[2]];
let div = divergence(&f, &[1.0, 2.0, 3.0]);
// 3.0

// Curl: ∇×F (3D only)
let f = |x: &[f64]| vec![x[1], x[2], x[0]];
let c = curl(&f, &[1.0, 2.0, 3.0]).unwrap();
// [-1.0, -1.0, -1.0]

// Laplacian: ∇²f = Σ ∂²f/∂x_i²
let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
let lap = laplacian(&f, &[1.0, 2.0, 3.0]);
// 6.0

// Jacobian: J_ij = ∂F_i/∂x_j
let f = |x: &[f64]| vec![x[0] * x[1], x[0] + x[1]];
let j = jacobian(&f, &[2.0, 3.0]);
// [[3, 2], [1, 1]]  (flattened)

// Hessian: H_ij = ∂²f/∂x_i∂x_j
let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
let h = hessian(&f, &[1.0, 2.0]);
// [[2, 0], [0, 2]]

// Directional derivative: ∇f · v̂
let d = directional_derivative(&f, &[1.0, 2.0], &[1.0, 0.0]);
// 2.0
```

**Formulas:**

```
∇f   = [∂f/∂x₀, ∂f/∂x₁, ..., ∂f/∂xₙ]
∇·F  = Σᵢ ∂Fᵢ/∂xᵢ
∇×F  = [∂F₃/∂x₂ - ∂F₂/∂x₃,  ∂F₁/∂x₃ - ∂F₃/∂x₁,  ∂F₂/∂x₁ - ∂F₁/∂x₂]
∇²f  = Σᵢ ∂²f/∂xᵢ²
J_ij = ∂Fᵢ/∂xⱼ
H_ij = ∂²f/∂xᵢ∂xⱼ
D_vf = ∇f · (v / ‖v‖)
```

### ODE Solvers (`ode`)

```rust
use mathverse_calculus::ode::*;

// Forward Euler: y_{n+1} = y_n + h f(t_n, y_n)
let result = euler(&|_t, y| y, 0.0, 1.0, 1.0, 100);
// y(1) ≈ e¹ ≈ 2.718

// Midpoint (RK2): more accurate than Euler
let result = midpoint(&|_t, y| y, 0.0, 1.0, 1.0, 100);
// y(1) ≈ 2.718282

// RK4: 4th-order, excellent accuracy
let result = runge_kutta_4(&|_t, y| y, 0.0, 1.0, 1.0, 10);
// y(1) ≈ 2.718282 (error < 1e-6 with just 10 steps)

// System of ODEs: harmonic oscillator d²x/dt² = -x
let f = |t: f64, y: &[f64]| vec![y[1], -y[0]];
let result = runge_kutta_4_system(&f, 0.0, &[1.0, 0.0], 2.0 * PI, 100);
// After one period: y ≈ [1.0, 0.0]
```

**ODE Solution Flow:**

```
dy/dt = f(t, y),  y(t₀) = y₀

  ┌──────────────────────────────────────────┐
  │  Input: f, t₀, y₀, t_end, steps (N)     │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────┐
  │  h = (t_end - t₀) / N                    │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  ┌──── For each step k = 0..N-1 ────────────┐
  │                                          │
  │  Euler:  y ← y + h f(t, y)              │
  │                                          │
  │  RK2:    k₁ = f(t, y)                   │
  │          k₂ = f(t+h/2, y+hk₁/2)         │
  │          y ← y + h k₂                   │
  │                                          │
  │  RK4:    k₁ = f(t, y)                   │
  │          k₂ = f(t+h/2, y+hk₁/2)         │
  │          k₃ = f(t+h/2, y+hk₂/2)         │
  │          k₄ = f(t+h, y+hk₃)             │
  │          y ← y + h(k₁+2k₂+2k₃+k₄)/6    │
  │                                          │
  │  t ← t + h                               │
  └──────────────┬───────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────────┐
  │  Output: [(t₀,y₀), (t₁,y₁), ..., (tₙ,yₙ)] │
  └──────────────────────────────────────────┘

Accuracy comparison (dy/dt=y, y(0)=1, y(1)=e):
  Euler(100):     error ≈ 1.4e-2
  Midpoint(100):  error ≈ 8.3e-5
  RK4(10):        error < 1.0e-6
```

### Root Finding (`root_finding`)

```rust
use mathverse_calculus::root_finding::*;

// Bisection: guaranteed convergence, slow
let root = bisection(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
// 2.0

// Newton-Raphson: quadratic convergence, needs f'
let root = newton_raphson(
    &|x| x * x - 4.0,
    &|x| 2.0 * x,
    3.0, 1e-10, 100,
).unwrap();
// 2.0

// Secant: superlinear, no derivative needed
let root = secant(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
// 2.0

// False position: bracketed, like secant
let root = false_position(&|x| x * x - 4.0, 1.0, 3.0, 1e-10, 100).unwrap();
// 2.0

// Auto-derivative Newton: uses central diff for f'
let root = newton_raphson_auto(&|x| x * x - 4.0, 3.0, 1e-10, 100).unwrap();
// 2.0
```

**Comparison:**

```
┌──────────────────┬─────────────┬──────────────┬──────────────┐
│ Method           │ Convergence │ Needs f'?    │ Bracketing?  │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ Bisection        │ linear      │ no           │ yes          │
│ Newton-Raphson   │ quadratic   │ yes          │ no           │
│ Secant           │ ~1.618      │ no           │ no           │
│ False position   │ ~1.618      │ no           │ yes          │
│ Auto Newton      │ quadratic   │ numerical    │ no           │
└──────────────────┴─────────────┴──────────────┴──────────────┘
```

## Future Scope

- Symbolic differentiation (integration with `mathverse-symbolic`)
- Multi-dimensional integration (Monte Carlo, cubature)
- Stochastic ODE solvers (Euler-Maruyama, Milstein)
- Boundary value problem solvers (shooting method, finite differences)
- Automatic differentiation (forward/reverse mode)
- Chebyshev spectral methods for ODEs
- Adaptive step-size ODE solvers (Dormand-Prince, Adams-Bashforth)
- Polynomial root finding (Jenkins-Traub, companion matrix)

## License

MIT OR Apache-2.0
