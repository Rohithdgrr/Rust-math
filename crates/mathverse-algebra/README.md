# MathVerse Algebra

**Polynomials, roots, factorization, symmetric polynomials, determinants, and linear systems — a complete algebraic toolkit built on `mathverse-core`.**

[![Rust](https://img.shields.io/badge/Rust-2021-EA5727?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Features

- **Polynomial** — `Polynomial<T>` with lowest-degree-first storage, arithmetic, division, evaluation
- **Root finding** — Linear, quadratic, cubic (Cardano), quartic (Ferrari) solvers
- **Factorization** — Synthetic division, long division, GCD, rational root candidates
- **Identities** — `(a+b)²`, `(a-b)²`, `(a+b)(a-b)`, `(a+b)³`, `(a-b)³`, `(a²-b²)`, `(a³±b³)`
- **Rational expressions** — Simplify, partial fractions, rationalize denominators
- **Sequences** — Arithmetic & geometric nth-term, sum, explicit ↔ recursive conversion
- **Interpolation** — Lagrange polynomial from arbitrary points
- **Symmetric polynomials** — Elementary symmetric, power sums → elementary conversion
- **Composition** — `f(g(x))` for polynomials
- **Determinants** — 2×2, 3×3, Cramer's rule
- **Exponent rules** — Power of power, product, quotient, negative & fractional exponents
- **Linear systems** — 2×2 and 3×3 Gaussian elimination with partial pivoting

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `polynomial` | `Polynomial<T>` — `new`, `eval`, `derivative`, `integral`, `add`, `sub`, `mul`, `div`, `roots` |
| `roots` | `solve_linear`, `solve_quadratic`, `solve_cubic` (Cardano), `solve_quartic` (Ferrari), discriminants |
| `factor` | `synthetic_division`, `divide` (long division), `polynomial_gcd`, `rational_root_candidates` |
| `identities` | Algebraic identity formulas as functions returning `Polynomial` |
| `rational` | `RationalExpression` — `simplify`, `partial_fractions`, `rationalize_denominator` |
| `sequences` | `arithmetic_nth_term`, `geometric_nth_term`, `arithmetic_sum`, `geometric_sum`, convert explicit ↔ recursive |
| `interpolate` | `lagrange_interpolation` — polynomial through n points |
| `symmetric` | `elementary_symmetric`, `power_sum_to_elementary`, `is_symmetric` |
| `compose` | `compose(f, g)` — polynomial composition f(g(x)) |
| `determinant` | `det2`, `det3`, `inverse2`, `inverse3`, `cramer2`, `cramer3` |
| `exponents` | `power_of_power`, `product_of_powers`, `quotient_of_powers`, `negative_exponent`, `fractional_exponent` |
| `systems` | `solve_2x2`, `solve_3x3` — Gaussian elimination with partial pivoting |

---

## Module Dependency Graph

```
                    mathverse-core
                         │
                ┌────────┴────────┐
                │                 │
         ┌──────┴──────┐   ┌─────┴─────┐
         │ Polynomial  │   │  Sequences │
         │  <T> type   │   │  Arith/Geo│
         └──────┬──────┘   └───────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───┴───┐  ┌───┴───┐  ┌───┴───┐
│ Roots │  │ Factor│  │Compose│
│Linear │  │Synth. │  │ f∘g   │
│Quad/C │  │ Long  │  └───────┘
│Cubic/Q│  │  GCD  │
└───┬───┘  └───────┘
    │
    ├──────────┐
    │          │
┌───┴──────┐ ┌┴───────────┐
│Determinant│ │  Systems   │
│ 2×2/3×3  │ │  2×2 / 3×3 │
│  Cramer  │ │  Gaussian  │
└──────────┘ └────────────┘
    │
┌───┴──────┐  ┌──────────┐
│  Sym.    │  │Rational  │
│Poly.     │  │Expr.     │
│ Elem.    │  │Simplify  │
│ PowerSum │  │ Partial  │
└──────────┘  └──────────┘
```

---

## Installation

```toml
[dependencies]
mathverse-algebra = { path = "../mathverse-algebra" }
```

For `no_std` environments:

```toml
mathverse-algebra = { path = "../mathverse-algebra", default-features = false }
```

---

## Quick Start

```rust
use mathverse_algebra::prelude::*;
use mathverse_core::prelude::*;

fn main() {
    // Polynomial: x² - 5x + 6 = (x-2)(x-3)
    let p = Polynomial::new(vec![6.0, -5.0, 1.0]); // lowest-degree first
    let roots = p.roots();
    // [2.0, 3.0]

    // Quadratic solver: x² + 2x - 8 = 0 → x = 2, x = -4
    let q_roots = solve_quadratic(1.0, 2.0, -8.0);
    // [2.0, -4.0]

    // Lagrange interpolation through (0,1), (1,3), (2,7)
    let points = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
    let interp = lagrange_interpolation(&points);
    // 2x² + x + 1

    // Determinant
    let d = det3(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
    );
    // -3.0

    // Cramer's rule: solve [1 2; 3 4] × [x; y] = [5; 11]
    let (x, y) = cramer2(1.0, 2.0, 3.0, 4.0, 5.0, 11.0);
    // x = 1.0, y = 2.0

    println!("Roots of x²-5x+6: {roots:?}");
    println!("Quadratic roots: {q_roots:?}");
    println!("Interpolated polynomial: {interp}");
    println!("det3 = {d}");
    println!("Cramer: x={x}, y={y}");
}
```

**Output:**
```
Roots of x²-5x+6: [2.0, 3.0]
Quadratic roots: [2.0, -4.0]
Interpolated polynomial: 2x^2 + x + 1
det3 = -3
Cramer: x=1, y=2
```

---

## Module Reference

### `polynomial` — Polynomial Type

Storage is **lowest-degree first**: `[a₀, a₁, a₂, …]` represents `a₀ + a₁x + a₂x² + …`

| Method | Description |
|--------|-------------|
| `Polynomial::new(coeffs)` | Create from coefficient vector |
| `eval(x)` | Evaluate at x (Horner's method) |
| `derivative()` | First derivative |
| `integral()` | Antiderivative (constant = 0) |
| `degree()` | Highest power with non-zero coeff |
| `is_zero()` | All coefficients ≈ 0 |
| `leading_coeff()` | Coefficient of highest term |
| `roots()` | Find all real roots (if degree ≤ 4, exact; else numerical) |

**Operators:** `+`, `-`, `*`, `/`, `==`

```rust
let p = Polynomial::new(vec![-6.0, 11.0, -6.0, 1.0]); // x³ - 6x² + 11x - 6
let q = p.derivative();          // 3x² - 12x + 11
let r = p.integral();            // -6x + 11x²/2 - 6x³/3 + x⁴/4
let v = p.eval(2.0);             // 0.0 (x=2 is a root)
```

---

### `roots` — Root Solvers

**Quadratic Formula**

```
                -b ± √(b² - 4ac)
        x = ─────────────────────
                      2a

    discriminant = b² - 4ac
        Δ > 0  →  2 distinct real roots
        Δ = 0  →  1 repeated root
        Δ < 0  →  no real roots
```

**Cardano's Method (Cubic)**

```
    x³ + px + q = 0     (depressed cubic)

        Δ = -(4p³ + 27q²)

        u = ∛(-q/2 + √(-Δ/108))
        v = ∛(-q/2 - √(-Δ/108))

        x₁ = u + v
        x₂ = -(u+v)/2 + i√3(u-v)/2
        x₃ = -(u+v)/2 - i√3(u-v)/2
```

**Ferrari's Method (Quartic)**

```
    x⁴ + ax³ + bx² + cx + d = 0

    → Reduce to depressed quartic
    → Solve resolvent cubic
    → Factor into two quadratics
    → Solve each quadratic
```

| Function | Degree | Method |
|----------|--------|--------|
| `solve_linear(a, b)` | 1 | Direct formula |
| `solve_quadratic(a, b, c)` | 2 | Quadratic formula |
| `solve_cubic(a, b, c, d)` | 3 | Cardano's formula |
| `solve_quartic(a, b, c, d, e)` | 4 | Ferrari's formula |

```rust
// x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3)
let roots = solve_cubic(1.0, -6.0, 11.0, -6.0);
// [1.0, 2.0, 3.0]
```

---

### `factor` — Factorization

| Function | Description |
|----------|-------------|
| `synthetic_division(dividend, root)` | Divide by (x - root), return (quotient, remainder) |
| `divide(dividend, divisor)` | Polynomial long division, return (quotient, remainder) |
| `polynomial_gcd(a, b)` | GCD via Euclidean algorithm |
| `rational_root_candidates(p)` | ±(factors of a₀) / (factors of aₙ) |

**Synthetic division steps:**

```
    Dividing x³ - 6x² + 11x - 6  by  (x - 2)

    Coefficients:  1   -6    11   -6
                   │    2   -8    6
                   ─────────────────
                   1   -4    3    0

    Quotient: x² - 4x + 3 = (x-1)(x-3)
    Remainder: 0 → x=2 is a root
```

```rust
let (q, rem) = synthetic_division(&[−6.0, 11.0, −6.0, 1.0], 2.0);
// q = [3.0, -4.0, 1.0]  (x² - 4x + 3)
// rem = 0.0

let candidates = rational_root_candidates(&[−6.0, 11.0, −6.0, 1.0]);
// [1, -1, 2, -2, 3, -3, 6, -6]
```

---

### `identities` — Algebraic Identities

All functions return `Polynomial<i64>`.

| Function | Identity |
|----------|----------|
| `square_sum(a, b)` | `(a + b)² = a² + 2ab + b²` |
| `square_diff(a, b)` | `(a - b)² = a² - 2ab + b²` |
| `diff_of_squares(a, b)` | `(a + b)(a - b) = a² - b²` |
| `cube_sum(a, b)` | `(a + b)³ = a³ + 3a²b + 3ab² + b³` |
| `cube_diff(a, b)` | `(a - b)³ = a³ - 3a²b + 3ab² - b³` |
| `sum_of_cubes(a, b)` | `a³ + b³ = (a + b)(a² - ab + b²)` |
| `diff_of_cubes(a, b)` | `a³ - b³ = (a - b)(a² + ab + b²)` |

```rust
let s = square_sum(3, 4);
// Polynomial representing 9 + 24 + 16 = 49

let d = diff_of_squares(5, 3);
// Polynomial representing 25 - 9 = 16
```

---

### `rational` — Rational Expressions

| Function | Description |
|----------|-------------|
| `RationalExpression::new(num, den)` | Create numerator/denominator pair |
| `simplify()` | Cancel common polynomial factors |
| `partial_fractions()` | Decompose into partial fractions |
| `rationalize_denominator()` | Eliminate radicals from denominator |

```
    Partial Fractions:
        2x + 3          A         B
    ───────────── = ───────── + ─────────
    (x+1)(x+2)      (x+1)      (x+2)

    Solve: 2x + 3 = A(x+2) + B(x+1)
           x = -1  →  1 = A(1)  →  A = 1
           x = -2  →  -1 = B(-1) → B = 1
```

---

### `sequences` — Arithmetic & Geometric Sequences

| Function | Formula |
|----------|---------|
| `arithmetic_nth_term(a₁, d, n)` | `aₙ = a₁ + (n-1)d` |
| `arithmetic_sum(a₁, d, n)` | `Sₙ = n/2 × (2a₁ + (n-1)d)` |
| `geometric_nth_term(a₁, r, n)` | `aₙ = a₁ · r^(n-1)` |
| `geometric_sum(a₁, r, n)` | `Sₙ = a₁(1 - rⁿ)/(1 - r)` |
| `explicit_to_recursive(a₁, d)` | Convert `a₁ + (n-1)d` to recursive form |
| `recursive_to_explicit(a₁, d)` | Convert recursive to closed form |

---

### `interpolate` — Lagrange Interpolation

Given n points `(x₀, y₀), (x₁, y₁), …, (xₙ₋₁, yₙ₋₁)`:

```
            n-1
    P(x) =  Σ   yᵢ × Lᵢ(x)
            i=0

                 ┌     (x - xⱼ)
    Lᵢ(x) =     │ ∏   ─────────   (Lagrange basis polynomial)
                 │j≠i  (xᵢ - xⱼ)
```

```rust
let points = vec![(0.0, 1.0), (1.0, 3.0), (2.0, 7.0)];
let p = lagrange_interpolation(&points);
// 2x² + x + 1

assert_eq!(p.eval(0.0), 1.0);
assert_eq!(p.eval(1.0), 3.0);
assert_eq!(p.eval(2.0), 7.0);
```

---

### `symmetric` — Symmetric Polynomials

| Function | Description |
|----------|-------------|
| `elementary_symmetric(degree)` | Elementary symmetric polynomial `eₖ(x₁, …, xₙ)` |
| `power_sum_to_elementary(power_sums)` | Newton's identity: power sums → elementary |
| `is_symmetric(poly)` | Check if polynomial is symmetric |

**Elementary symmetric polynomials for 3 variables:**

```
    e₀ = 1
    e₁ = x₁ + x₂ + x₃
    e₂ = x₁x₂ + x₁x₃ + x₂x₃
    e₃ = x₁x₂x₃
```

---

### `compose` — Polynomial Composition

```
    f(x) = x² + 1
    g(x) = 2x + 3

    f∘g = f(g(x)) = (2x+3)² + 1 = 4x² + 12x + 10
```

```rust
let f = Polynomial::new(vec![1.0, 0.0, 1.0]); // x² + 1
let g = Polynomial::new(vec![3.0, 2.0]);       // 2x + 3
let h = compose(&f, &g);                       // 4x² + 12x + 10
```

---

### `determinant` — Determinants & Cramer's Rule

**2×2 Determinant:**

```
    | a  b |
    | c  d |  = ad - bc
```

**3×3 Determinant (Sarrus):**

```
    | a  b  c |         aei + bfg + cdh - ceg - bdi - afh
    | d  e  f |
    | g  h  i |
```

**Cramer's Rule (2×2):**

```
    ax + by = e        | e  b |        | a  e |
    cx + dy = f    x = | f  d | / det  y = | c  f | / det
                       | a  b |            | a  b |
                       | c  d |            | c  d |
```

| Function | Description |
|----------|-------------|
| `det2(a, b, c, d)` | 2×2 determinant |
| `det3(a..i)` | 3×3 determinant |
| `inverse2(a, b, c, d)` | 2×2 matrix inverse (Option) |
| `inverse3(a..i)` | 3×3 matrix inverse (Option) |
| `cramer2(a, b, c, d, e, f)` | Solve 2×2 system via Cramer |
| `cramer3(a..i, b..i, c..i)` | Solve 3×3 system via Cramer |

```rust
let d = det2(2.0, 3.0, 1.0, 4.0);
// 2×4 - 3×1 = 5.0

let inv = inverse2(2.0, 3.0, 1.0, 4.0);
// Some([0.8, -0.6, -0.2, 0.4])
```

---

### `exponents` — Exponent Rules

| Function | Rule | Example |
|----------|------|---------|
| `power_of_power(a, m, n)` | `(aᵐ)ⁿ = aᵐⁿ` | (2³)² = 2⁶ = 64 |
| `product_of_powers(a, m, n)` | `aᵐ · aⁿ = aᵐ⁺ⁿ` | 2³ · 2² = 2⁵ = 32 |
| `quotient_of_powers(a, m, n)` | `aᵐ / aⁿ = aᵐ⁻ⁿ` | 2⁵ / 2³ = 2² = 4 |
| `negative_exponent(a, n)` | `a⁻ⁿ = 1/aⁿ` | 2⁻³ = 1/8 = 0.125 |
| `fractional_exponent(a, m, n)` | `a^(m/n) = ⁿ√(aᵐ)` | 8^(2/3) = 4.0 |

---

### `systems` — Linear Systems

**Gaussian Elimination with Partial Pivoting:**

```
    [a₁₁  a₁₂  a₁₃ | b₁]       Forward elimination
    [a₂₁  a₂₂  a₂₃ | b₂]  →    Upper triangular form
    [a₃₁  a₃₂  a₃₃ | b₃]       → Back substitution

    Partial pivoting: swap rows to put largest |aₖⱼ| on diagonal
    Prevents division by zero and reduces numerical error
```

| Function | Input | Returns |
|----------|-------|---------|
| `solve_2x2(a, b, c, d, e, f)` | 2×2 system | `Option<(f64, f64)>` |
| `solve_3x3(a..i, b..i, c..i)` | 3×3 system | `Option<(f64, f64, f64)>` |

```rust
// x + y = 3
// 2x - y = 0
let (x, y) = solve_2x2(1.0, 1.0, 2.0, -1.0, 3.0, 0.0);
// x = 1.0, y = 2.0
```

---

## Roadmap

| Phase | What |
|-------|------|
| 0.3.0 | `PolynomialRing` for modular arithmetic |
| 0.3.0 | Symbolic differentiation of rational expressions |
| 0.4.0 | Galois theory helpers (splitting fields, automorphisms) |
| 0.4.0 | Multivariate polynomial system solver (Gröbner bases) |
| 0.5.0 | `no_std` support (currently needs `alloc`) |
| 0.5.0 | Matrix trait and n×n Gaussian elimination |

---

## License

MIT — see [LICENSE](LICENSE).
