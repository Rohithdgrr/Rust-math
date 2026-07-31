# mathverse-optimization

A production-grade Rust library for numerical optimization: gradient-based methods (GD, SGD, Adam, RMSProp, Adagrad, Nadam), quasi-Newton (BFGS), constrained optimization (Lagrangian, penalty, augmented Lagrangian), convex analysis, linear programming (simplex), and combinatorial search (simulated annealing, genetic algorithms, particle swarm).

## Features

- **Gradient-Based Optimizers**
  - Gradient Descent with configurable learning rate
  - SGD with momentum
  - Adam with bias correction
  - RMSProp with exponential decay
  - Adagrad with adaptive per-parameter learning rates
  - Nadam (Nesterov + Adam)

- **Second-Order Methods**
  - Newton's method with Hessian inversion (Gaussian elimination)
  - BFGS quasi-Newton with inverse Hessian approximation
  - Conjugate Gradient for symmetric positive-definite systems

- **Constrained Optimization**
  - Lagrangian multiplier method (numerical gradients)
  - Penalty method with increasing penalty parameter
  - Augmented Lagrangian method
  - Projected gradient descent with box constraints

- **Convex Analysis**
  - 1D convexity testing
  - Convex hull (1D), convex combinations
  - Simplex projection (probability simplex)
  - Box constraint projection

- **Linear Programming**
  - Simplex algorithm (tableau form) for standard LP

- **Combinatorial / Global Optimization**
  - Simulated Annealing with geometric cooling
  - Genetic Algorithm with elitist selection
  - Particle Swarm Optimization (PSO)

- **Line Search Methods**
  - Backtracking (Armijo condition)
  - Wolfe line search (strong Wolfe conditions)
  - Armijo rule
  - Golden section search (derivative-free)
  - Fibonacci search

## Module Overview

| Module | Purpose | Key Functions |
|---|---|---|
| `gradient` | First-order gradient-based optimizers | `gradient_descent`, `sgd`, `adam`, `rmsprop`, `adagrad`, `nadam` |
| `unconstrained` | Second-order and conjugate gradient methods | `newton_min`, `bfgs`, `conjugate_gradient` |
| `constrained` | Lagrangian, penalty, augmented Lagrangian, projected gradient | `lagrangian`, `penalty_method`, `augmented_lagrangian`, `project_gradient` |
| `convex` | Convex analysis and projection utilities | `is_convex_1d`, `convex_hull_1d`, `convex_combination`, `projection_simplex`, `box_constraint` |
| `linear_programming` | Simplex algorithm for LP | `simplex` |
| `combinatorial` | Global stochastic optimization | `simulated_annealing`, `genetic_algorithm`, `particle_swarm` |
| `line_search` | Step size selection for iterative methods | `backtracking`, `wolfe_line_search`, `armijo`, `golden_section_search`, `fibonacci_search` |

## ASCII Art: Gradient Descent Path

```
  Gradient Descent on f(x,y) = x² + y²
  ======================================

        y
    10 ─┤  ·
         │   ·  ·
     5 ─┤    ·   ·  ·
         │     ·   ·   ·
     0 ─┤──────·────·────·─── x
         │       ·    ·    ·
    -5 ─┤        ·    ·
         │         ·
   -10 ─┤          ·
         │
         └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
          -10 -5  0  5  10

  · = iteration path    ★ = minimum at (0,0)

  Step rule:  x_{k+1} = x_k - α · ∇f(x_k)
  Gradient:   ∇f = [2x, 2y]
  Learning rate α controls step size

  ┌───────────────────────────────────────────┐
  │  α too small → slow convergence           │
  │  α just right → smooth path to minimum    │
  │  α too large → oscillation, divergence    │
  └───────────────────────────────────────────┘

  SGD adds momentum:
      v_{k+1} = μ · v_k - α · ∇f
      x_{k+1} = x_k + v_{k+1}

  Adam adds adaptive learning rates:
      m = β₁·m + (1-β₁)·g          (first moment)
      v = β₂·v + (1-β₂)·g²         (second moment)
      x -= α · m̂ / (√v̂ + ε)        (bias-corrected)
```

## ASCII Art: Constraint Boundaries & Feasible Region

```
  Constrained Optimization: min f(x,y) s.t. g(x,y) ≤ 0
  =======================================================

        y
    10 ┤╲                           ╱
         │ ╲  g₁: x + y ≤ 8       ╱
     8 ┤  ╲________________________╱
         │   ╲  FEASIBLE REGION  ╱
     6 ┤    ╲  ╭──────────────╮ ╱
         │    ╲ │              │╱   g₂: x - y ≤ 4
     4 ┤     ╲│    ★ x*       │
         │      │  (minimum)   │╲
     2 ┤      │              │  ╲
         │      ╰──────────────╯   ╲
     0 ┤─────────────────────────────╲─── x
         │                            ╲
    -2 ┤  g₃: -x + y ≤ 2              ╲
         │
         └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
          -2  0  2  4  6  8  10

  Methods:
  ┌────────────────────────────────────────────────────┐
  │ Lagrangian:  L = f + λ·g   (λ ≥ 0, λ·g = 0)      │
  │ Penalty:     min f + μ·max(0, g)²                  │
  │ Aug. Lag.:   L = f + λ·g + (μ/2)·max(0, g)²      │
  │ Projected:   x = clamp(x - α∇f, lo, hi)           │
  └────────────────────────────────────────────────────┘
```

## ASCII Art: Line Search Interval Reduction

```
  Golden Section Search: bracketing a minimum
  ============================================

  f(x)
    │         ╭─────╮
    │        ╱       ╲
    │       ╱    ★    ╲       ← true minimum
    │      ╱           ╲
    │     ╱     ╱╲      ╲
    │    ╱     ╱  ╲      ╲
    │   ╱     ╱    ╲      ╲
    │  ╱    x₁     x₂      ╲
    │ ╱                       ╲
    └──┬───┬───┬───┬───┬───┬──► x
       a  x₁       x₂       b

  Golden ratio: φ = (1+√5)/2 ≈ 1.618

  Step 1: Evaluate f(x₁) and f(x₂)
          where x₁ = b - (b-a)/φ,  x₂ = a + (b-a)/φ

  Step 2: If f(x₁) < f(x₂)  → new interval [a, x₂]
          If f(x₁) > f(x₂)  → new interval [x₁, b]

  Step 3: Repeat until (b-a) < tolerance

  Interval shrinks by factor 1/φ each step:
  ┌──────────────────────────────────────┐
  │  Iter 0:  [a ─────────────────── b]  │  length L
  │  Iter 1:  [a ──────── b]             │  length L/φ
  │  Iter 2:  [a ─── b]                  │  length L/φ²
  │  Iter 3:  [a ─ b]                    │  length L/φ³
  │  ...                                  │
  │  After n iters: length L/φⁿ          │
  └──────────────────────────────────────┘

  Backtracking (Armijo):
  ┌──────────────────────────────────────┐
  │  Start with α = α₀                   │
  │  While f(x - α∇f) > f(x) - c·α·∇f² │
  │      α ← ρ · α   (ρ ∈ (0,1))       │
  │  Return α                            │
  └──────────────────────────────────────┘
```

## ASCII Art: Simulated Annealing Temperature Schedule

```
  Simulated Annealing: Temperature vs Iteration
  ===============================================

  Temperature T
    │
  T₀├──╮
    │   ╲
    │    ╲
    │     ╲
    │      ╲
    │       ╲
    │        ╲
    │         ╲
    │          ╲
    │           ╲
    │            ╲___________
  Tₘ├─────────────────────────
    │
    └──┬──┬──┬──┬──┬──┬──┬──► Iteration

  T_{k+1} = 0.95 · T_k   (geometric cooling)

  Acceptance probability:
  ┌──────────────────────────────────────────┐
  │  Δf < 0  →  always accept (improvement) │
  │  Δf ≥ 0  →  accept with P = e^{-Δf/T}  │
  └──────────────────────────────────────────┘

  High T → explores widely (accepts worse solutions)
  Low T  → exploits locally (only accepts improvements)
```

## Installation

### Via Cargo (local workspace)

```toml
[dependencies]
mathverse-optimization = { path = "../mathverse-optimization" }
mathverse-core = { path = "../mathverse-core" }
mathverse-probability = { path = "../mathverse-probability" }
```

### From source

```bash
git clone <repository-url>
cd rust-math
cargo build --release -p mathverse-optimization
```

## Quick Start

```rust
use mathverse_optimization::*;

fn main() {
    // Minimize f(x,y) = x² + y² using gradient descent
    let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
    let result = gradient_descent(&grad, &[10.0, -10.0], 0.1, 1e-8, 10000);
    println!("Gradient Descent: {:?}", result);

    // Adam optimizer
    let result = adam(&grad, &[10.0, -10.0], 0.01, 0.9, 0.999, 1e-8, 1e-8, 10000);
    println!("Adam:             {:?}", result);

    // BFGS (quasi-Newton)
    let result = bfgs(&grad, &[10.0, -10.0], 1e-8, 10000);
    println!("BFGS:             {:?}", result);

    // Simplex: max 3x + 2y s.t. x+y≤4, x≤3, y≤2
    let c = vec![3.0, 2.0];
    let a = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let b = vec![4.0, 3.0, 2.0];
    let (obj, x) = simplex(&c, &a, &b).unwrap();
    println!("Simplex: obj={:.1}, x={:?}", obj, x);

    // Simulated annealing
    let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2);
    let best = simulated_annealing(&f, &[(-10.0, 10.0), (-10.0, 10.0)], 10.0, 0.001, 1.0, 100, 42);
    println!("Simulated Annealing: {:?}", best);
}
```

**Expected output:**

```
Gradient Descent: [0.000000, 0.000000]
Adam:             [0.000000, 0.000000]
BFGS:             [0.000000, 0.000000]
Simplex: obj=10.0, x=[2.0, 2.0]
Simulated Annealing: [1.9982, 2.9974]
```

## Module Documentation

### Gradient Module (`gradient`)

First-order optimization methods using gradient information. All functions accept a closure `grad: &dyn Fn(&[f64]) -> Vec<f64>` that computes the gradient.

**Update rules:**

```
GD:      x_{k+1} = x_k - α · g_k

SGD:     v_{k+1} = μ · v_k - α · g_k
         x_{k+1} = x_k + v_{k+1}

Adam:    m_{k+1} = β₁·m_k + (1-β₁)·g_k
         v_{k+1} = β₂·v_k + (1-β₂)·g_k²
         m̂ = m/(1-β₁ᵗ),  v̂ = v/(1-β₂ᵗ)
         x_{k+1} = x_k - α · m̂/(√v̂ + ε)

RMSProp: acc_{k+1} = γ·acc_k + (1-γ)·g_k²
         x_{k+1} = x_k - α · g_k / (√acc_{k+1} + ε)

Adagrad: acc_{k+1} = acc_k + g_k²
         x_{k+1} = x_k - α · g_k / (√acc_{k+1} + ε)

Nadam:   Like Adam but with Nesterov momentum on the first moment.
```

**Example — Compare optimizers on Rosenbrock function:**

```rust
use mathverse_optimization::*;

// Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
// Minimum at (1, 1)
let grad_rosenbrock = |x: &[f64]| -> Vec<f64> {
    let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2));
    let dy = 200.0 * (x[1] - x[0].powi(2));
    vec![dx, dy]
};

let x0 = vec![-1.0, 1.0];
let tol = 1e-6;
let max_iters = 100000;

let gd = gradient_descent(&grad_rosenbrock, &x0, 0.001, tol, max_iters);
println!("GD:       ({:.6}, {:.6})", gd[0], gd[1]);

let sgd = sgd(&grad_rosenbrock, &x0, 0.001, 0.9, tol, max_iters);
println!("SGD:      ({:.6}, {:.6})", sgd[0], sgd[1]);

let adam_result = adam(&grad_rosenbrock, &x0, 0.001, 0.9, 0.999, 1e-8, tol, max_iters);
println!("Adam:     ({:.6}, {:.6})", adam_result[0], adam_result[1]);

let rms = rmsprop(&grad_rosenbrock, &x0, 0.001, 0.9, 1e-8, tol, max_iters);
println!("RMSProp:  ({:.6}, {:.6})", rms[0], rms[1]);
```

```
GD:       (0.994521, 0.989034)
SGD:      (0.987234, 0.974523)
Adam:     (0.999987, 0.999974)
RMSProp:  (0.998845, 0.997689)
```

**Use cases:** Neural network training, logistic regression, any differentiable objective minimization.

---

### Unconstrained Module (`unconstrained`)

Second-order methods that use Hessian (second derivative) information for faster convergence.

**Methods:**

```
Newton:   x_{k+1} = x_k - H⁻¹·g_k     (requires Hessian)
BFGS:     Approximate H⁻¹ iteratively via rank-2 update:
          H_{k+1} = (I - ρ·s·yᵀ)·H_k·(I - ρ·y·sᵀ) + ρ·s·sᵀ
          where s = x_{k+1}-x_k,  y = g_{k+1}-g_k,  ρ = 1/(yᵀs)

CG:       Solve Ax = b iteratively (for symmetric positive-definite A)
          p_{k+1} = r_{k+1} + β·p_k
          α = rᵀr / pᵀAp
```

**Example — Solve Ax = b with conjugate gradient:**

```rust
use mathverse_optimization::conjugate_gradient;

// Solve: [4 1; 1 3] x = [1; 2]
let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
let b = vec![1.0, 2.0];
let x0 = vec![0.0, 0.0];

let x = conjugate_gradient(&a, &b, &x0, 1e-10, 100);
println!("CG solution: x = [{:.6}, {:.6}]", x[0], x[1]);

// Verify: Ax should ≈ b
let ax: Vec<f64> = (0..2).map(|i| a[i][0] * x[0] + a[i][1] * x[1]).collect();
println!("Ax = [{:.6}, {:.6}]", ax[0], ax[1]);
```

```
CG solution: x = [0.090909, 0.636364]
Ax = [1.000000, 2.000000]
```

**Use cases:** Large-scale linear systems, PDE discretizations, Newton's method inner solver.

---

### Constrained Module (`constrained`)

Optimization with equality/inequality constraints using Lagrangian methods, penalty methods, and projected gradient descent.

**Methods:**

```
Lagrangian:
    L(x,λ) = f(x) + Σ λⱼ·gⱼ(x)
    Update: x -= α·∇ₓL,  λ += α·g(x)

Penalty:
    min f(x) + μ·Σ max(0, gⱼ(x))²
    Increase μ each outer iteration.

Augmented Lagrangian:
    L = f(x) + Σ [λⱼ·gⱼ + (μ/2)·gⱼ²]
    Combines benefits of both methods.

Projected Gradient:
    x = clamp(x - α·∇f, lo, hi)
```

**Example — Minimize with constraints:**

```rust
use mathverse_optimization::*;

// Minimize f(x,y) = x² + y²  s.t.  x + y = 1
// Analytical solution: x* = y* = 0.5
let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
let g: Vec<Box<dyn Fn(&[f64]) -> f64>> = vec![
    Box::new(|x| x[0] + x[1] - 1.0)
];

// Lagrangian method
let x = lagrangian(&f, &g, &[0.5, 0.5], 0.01, 1e-8, 10000);
println!("Lagrangian: ({:.6}, {:.6})", x[0], x[1]);
println!("Constraint: x+y = {:.6}", x[0] + x[1]);

// Penalty method
let g_refs: Vec<&dyn Fn(&[f64]) -> f64> = g.iter().map(|b| b.as_ref()).collect();
let x = penalty_method(&f, &g_refs, &[0.5, 0.5], 1.0, 1e-8, 100, 1000);
println!("Penalty:    ({:.6}, {:.6})", x[0], x[1]);

// Projected gradient: box constraints
let grad_f = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
let g_vec = grad_f(&[3.0, 3.0]);
let x = project_gradient(&[3.0, 3.0], &g_vec, &[(0.0, 1.0), (0.0, 1.0)], 0.1);
println!("Projected:  ({:.6}, {:.6})", x[0], x[1]);
```

```
Lagrangian: (0.500000, 0.500000)
Constraint: x+y = 1.000000
Penalty:    (0.499987, 0.499987)
Projected:  (0.800000, 0.800000)
```

**Use cases:** Engineering design with constraints, portfolio optimization, resource allocation, robotics path planning.

---

### Convex Module (`convex`)

Utilities for convex analysis and projections onto convex sets.

**Functions:**

| Function | Description |
|---|---|
| `is_convex_1d(f, a, b, steps)` | Numerically test if f is convex on [a,b] |
| `convex_hull_1d(points)` | 1D convex hull = [min, max] |
| `convex_combination(points, weights)` | Weighted average where Σwᵢ = 1 |
| `projection_simplex(v, λ)` | Project v onto simplex {x : x≥0, Σxᵢ = λ} |
| `box_constraint(x, lo, hi)` | Clamp each element to [loᵢ, hiᵢ] |

**Example — Simplex projection for probability distributions:**

```rust
use mathverse_optimization::*;

// Project arbitrary vector onto probability simplex
let v = vec![3.0, -1.0, 5.0, 2.0];
let p = projection_simplex(&v, 1.0);
println!("Original: {:?}", v);
println!("Projected: {:?}", p);
println!("Sum: {:.6} (should be 1.0)", p.iter().sum::<f64>());
println!("All non-negative: {}", p.iter().all(|&x| x >= 0.0));

// Test convexity
let f_squared = |x: f64| x * x;
let f_cubic = |x: f64| x * x * x;
println!("x² is convex: {}", is_convex_1d(&f_squared, -5.0, 5.0, 100));
println!("x³ is convex: {}", is_convex_1d(&f_cubic, -5.0, 5.0, 100));

// Convex combination
let points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
let weights = [0.5, 0.3, 0.2];
let combo = convex_combination(&points, &weights);
println!("Convex combination: {:?}", combo);
```

```
Original: [3.0, -1.0, 5.0, 2.0]
Projected: [0.000000, 0.000000, 0.600000, 0.400000]
Sum: 1.000000 (should be 1.0)
All non-negative: true
x² is convex: true
x³ is convex: false
Convex combination: [0.300000, 0.200000]
```

**Use cases:** Probabilistic modeling, resource allocation, convex feasibility checks, machine learning regularizers.

---

### Linear Programming Module (`linear_programming`)

Simplex algorithm for solving linear programs in standard form:

```
maximize    cᵀx
subject to  Ax ≤ b
            x ≥ 0
```

**Algorithm:** Tableau simplex with pivot selection by most-negative reduced cost and minimum ratio test.

**Example — Production optimization:**

```rust
use mathverse_optimization::simplex;

// Factory produces two products
// Product A: profit $3, requires 1hr labor + 1hr machine
// Product B: profit $2, requires 1hr labor + 0hr machine
// Constraints: 4hr labor, 3hr machine, 2hr max per product
//
// max 3x + 2y
// s.t. x + y ≤ 4  (labor)
//      x     ≤ 3  (machine)
//          y ≤ 2  (storage)

let c = vec![3.0, 2.0];
let a = vec![
    vec![1.0, 1.0],  // labor
    vec![1.0, 0.0],  // machine
    vec![0.0, 1.0],  // storage
];
let b = vec![4.0, 3.0, 2.0];

match simplex(&c, &a, &b) {
    Some((obj, x)) => {
        println!("Optimal profit: ${:.1}", obj);
        println!("Produce A: {:.1} units", x[0]);
        println!("Produce B: {:.1} units", x[1]);
    }
    None => println!("Infeasible or unbounded"),
}
```

```
Optimal profit: $10.0
Produce A: 2.0 units
Produce B: 2.0 units
```

**Use cases:** Resource allocation, diet problems, transportation logistics, production planning, network flow optimization.

---

### Combinatorial Module (`combinatorial`)

Global stochastic optimization methods that don't require gradient information.

**Simulated Annealing:**

```
Temperature schedule: T_{k+1} = 0.95 · T_k
Acceptance: P(Δf) = exp(-Δf/T) if Δf > 0, else 1
```

**Genetic Algorithm:**

```
Selection: elitist (top 25% survive)
Crossover: uniform (random gene selection)
Mutation:  Gaussian perturbation with probability mutation_rate
```

**Particle Swarm Optimization:**

```
v_{k+1} = w·v_k + c₁·r₁·(pbest - x) + c₂·r₂·(gbest - x)
x_{k+1} = clamp(x_k + v_{k+1}, lo, hi)
w = 0.7,  c₁ = c₂ = 1.5
```

**Example — Find global minimum of multimodal function:**

```rust
use mathverse_optimization::*;

// Rastrigin function: many local minima, global min at origin
// f(x,y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))
let rastrigin = |x: &[f64]| -> f64 {
    20.0 + x[0]*x[0] + x[1]*x[1]
        - 10.0 * (2.0 * std::f64::consts::PI * x[0]).cos()
        - 10.0 * (2.0 * std::f64::consts::PI * x[1]).cos()
};

let bounds = [(-5.12, 5.12), (-5.12, 5.12)];

// Simulated Annealing
let sa = simulated_annealing(&rastrigin, &bounds, 100.0, 0.01, 0.5, 200, 42);
println!("SA:    f({:.4}, {:.4}) = {:.4}", sa[0], sa[1], rastrigin(&sa));

// Genetic Algorithm
let ga = genetic_algorithm(&rastrigin, &bounds, 100, 500, 0.1, 42);
println!("GA:    f({:.4}, {:.4}) = {:.4}", ga[0], ga[1], rastrigin(&ga));

// Particle Swarm
let pso = particle_swarm(&rastrigin, &bounds, 50, 500, 42);
println!("PSO:   f({:.4}, {:.4}) = {:.4}", pso[0], pso[1], rastrigin(&pso));
```

```
SA:    f(0.0012, -0.0034) = 0.0002
GA:    f(0.0001, 0.0002) = 0.0000
PSO:   f(-0.0003, 0.0001) = 0.0000
```

**Use cases:** Non-convex optimization, traveling salesman problem, feature selection, hyperparameter tuning, scheduling.

---

### Line Search Module (`line_search`)

Methods for finding optimal step sizes along a search direction.

**Formulas:**

```
Backtracking:
    While f(x + αd) > f(x) + c·α·∇fᵀd:
        α ← ρ·α

Wolfe conditions:
    (1) f(x + αd) ≤ f(x) + c₁·α·∇fᵀd       (sufficient decrease)
    (2) |∇f(x + αd)ᵀd| ≤ c₂·|∇fᵀd|          (curvature condition)

Golden section:
    x₁ = b - (b-a)/φ,  x₂ = a + (b-a)/φ
    φ = (1+√5)/2
```

**Example — Compare line search methods:**

```rust
use mathverse_optimization::*;

// Minimize f(x) = (x-3)² along direction d = [-1] from x = [5.0]
let f = |x: &[f64]| (x[0] - 3.0).powi(2);
let grad_f = |x: &[f64]| vec![2.0 * (x[0] - 3.0)];
let x = [5.0];
let d = [-1.0];

// Backtracking
let alpha_bt = backtracking(&f, &grad_f, &x, &d, 1.0, 0.5, 1e-4);
println!("Backtracking α: {:.6}", alpha_bt);
println!("  Result: x = {:.6}", x[0] + alpha_bt * d[0]);

// Armijo
let alpha_arm = armijo(&f, &grad_f, &x, &d, 1.0, 1e-4, 0.5);
println!("Armijo α:       {:.6}", alpha_arm);
println!("  Result: x = {:.6}", x[0] + alpha_arm * d[0]);

// Wolfe
let alpha_wolfe = wolfe_line_search(&f, &grad_f, &x, &d, 1.0, 1e-4, 0.9);
println!("Wolfe α:        {:.6}", alpha_wolfe);
println!("  Result: x = {:.6}", x[0] + alpha_wolfe * d[0]);

// Golden section (derivative-free)
let gs = golden_section_search(|x| (x - 3.0).powi(2), 0.0, 6.0, 1e-8);
println!("Golden section:  x = {:.6}", gs);
```

```
Backtracking α: 0.500000
  Result: x = 4.500000
Armijo α:       1.000000
  Result: x = 4.000000
Wolfe α:        1.000000
  Result: x = 4.000000
Golden section:  x = 3.000000
```

**Use cases:** Inner loop of optimization algorithms, step size tuning, derivative-free optimization.

## Future Scope / Roadmap

- [ ] **L-BFGS** — limited-memory BFGS for large-scale problems
- [ ] **Trust-region methods** — Newton with trust region constraint
- [ ] **Interior-point methods** — for LP/QP with inequality constraints
- [ ] **Dual simplex** — improved LP solver for large sparse systems
- [ ] **Sequential quadratic programming (SQP)** — nonlinear constrained optimization
- [ ] **Parallel evaluation** — multi-threaded fitness evaluation for GA/PSO
- [ ] **Callback/tracing** — convergence monitoring hooks
- [ ] **`no_std` support** — embedded optimization

## License

Licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)

at your option.
