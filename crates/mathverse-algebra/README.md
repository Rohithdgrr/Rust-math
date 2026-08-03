# MathVerse Algebra

[![Crates.io](https://img.shields.io/crates/v/mathverse-algebra.svg)](https://crates.io/crates/mathverse-algebra)
[![docs.rs](https://docs.rs/mathverse-algebra/badge.svg)](https://docs.rs/mathverse-algebra)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Polynomials, roots, factorization, symmetric polynomials, determinants, and linear systems — a complete algebraic toolkit built on `mathverse-core`.

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
- **Solvability** — Galois-flavored classification: solvable by radicals, Eisenstein, palindromic
- **LaTeX rendering** — Polynomials, roots, equations, factors in LaTeX format

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
| `solvability` | Solvability-by-radicals classification — `solvable_by_radicals`, `degree`, `integer_root`, `eisenstein_irreducible`, palindromic/reciprocal helpers |
| `latex` | LaTeX rendering — `polynomial_latex`, `roots_latex`, `equation_solution_latex`, `factors_latex` |
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
mathverse-algebra = "0.1"
```

For `no_std` environments:

```toml
mathverse-algebra = { version = "0.1", default-features = false }
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

---

### `roots` — Root Solvers

| Function | Degree | Method |
|----------|--------|--------|
| `solve_linear(a, b)` | 1 | Direct formula |
| `solve_quadratic(a, b, c)` | 2 | Quadratic formula |
| `solve_cubic(a, b, c, d)` | 3 | Cardano's formula |
| `solve_quartic(a, b, c, d, e)` | 4 | Ferrari's formula |

---

### `factor` — Factorization

| Function | Description |
|----------|-------------|
| `synthetic_division(dividend, root)` | Divide by (x - root), return (quotient, remainder) |
| `divide(dividend, divisor)` | Polynomial long division, return (quotient, remainder) |
| `polynomial_gcd(a, b)` | GCD via Euclidean algorithm |
| `rational_root_candidates(p)` | ±(factors of a₀) / (factors of aₙ) |

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

---

### `determinant` — Determinants & Cramer's Rule

| Function | Description |
|----------|-------------|
| `det2(a, b, c, d)` | 2×2 determinant |
| `det3(a..i)` | 3×3 determinant |
| `inverse2(a, b, c, d)` | 2×2 matrix inverse (Option) |
| `inverse3(a..i)` | 3×3 matrix inverse (Option) |
| `cramer2(a, b, c, d, e, f)` | Solve 2×2 system via Cramer |
| `cramer3(a..i, b..i, c..i)` | Solve 3×3 system via Cramer |

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

| Function | Input | Returns |
|----------|-------|---------|
| `solve_2x2(a, b, c, d, e, f)` | 2×2 system | `Option<(f64, f64)>` |
| `solve_3x3(a..i, b..i, c..i)` | 3×3 system | `Option<(f64, f64, f64)>` |

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
