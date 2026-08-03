# MathVerse Arithmetic

[![Crates.io](https://img.shields.io/crates/v/mathverse-arithmetic.svg)](https://crates.io/crates/mathverse-arithmetic)
[![docs.rs](https://docs.rs/mathverse-arithmetic/badge.svg)](https://docs.rs/mathverse-arithmetic)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Percentage, powers, roots, scientific notation, complex numbers, and beyond — a complete numeric toolbox built on `mathverse-core`.

---

## Features

- **Percentage** — compute, increase, decrease, find difference, reverse percentages
- **Powers & roots** — integer powers, arbitrary nth roots, nth_roots with sign handling
- **Scientific notation** — `ScientificNotation` struct, engineering notation, scientific rounding
- **Complex numbers** — `Complex<f64>` with arithmetic, polar, transcendental & hyperbolic functions
- **Rational arithmetic** — `Rational` exact fractions, gcd-reduction, mixed numbers
- **Continued fractions** — simple & generalized, convergents, precision control
- **Special functions** — Gamma, Beta, Erf, Bessel J₀/J₁, Zeta, Lambert W
- **Numerical methods** — integration, differentiation, root-finding, optimization
- **Sequences & series** — Fibonacci, Lucas, prime gaps, harmonic, sum/product
- **Approximation** — Taylor series, Padé approximants, minimax, interpolation-based
- **Interpolation** — polynomial, linear, nearest-neighbor on arbitrary data
- **Number theory** — modular exponentiation, primality, totient, partitions
- **Rounding** — floor/ceil/trunc/nearest, banker's, fixed decimal, engineering
- **Comparison** — total ordering, clamp, sort helpers, epsilon-based fuzzy compare
- **Finance** — future value, present value, annuities, perpetuities, Rule of 72
- **Checked/saturating ops** — overflow-safe arithmetic, approximate equality, remapping

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `percentage` | Percent calculations: `find_percent`, `increase_by_percent`, `decrease_by_percent`, `find_reverse_percent` |
| `power` | `power(base, exp)`, `power_with_sign`, `fast_power_mod`, `power_series` |
| `root` | `nth_root(x, n)`, `sqrt`, `cbrt`, `sqrt_with_sign`, `nth_roots` |
| `modulus` | `modulus(a, b)` with flooring semantics, `mod_inverse`, `mod_pow` |
| `number_theory` | `is_prime`, `sieve_of_eratosthenes`, `prime_factors`, `divisor_count`, `divisor_sum` |
| `scientific` | `ScientificNotation` struct, `from_f64`, `to_f64`, `engineering_notation`, rounding |
| `rounding` | `round_to`, `bankers_rounding`, `fixed_decimal`, `engineering_rounding`, `truncate_to` |
| `comparison` | `total_compare`, `fuzzy_compare`, `clamp_val`, `sort_ascending`, `sort_descending` |
| `complex` | `Complex { re, im }` — arithmetic, polar form, `phase`, `abs`, trig, log, exp |
| `rational` | `Rational { numer, denom }` — exact fraction arithmetic, continued fraction conversion |
| `continued_fraction` | `SimpleContinuedFraction`, `GeneralizedContinuedFraction`, convergents, approximations |
| `approximation` | Taylor, Padé, minimax polynomial, interpolation-based polynomial approximations |
| `interpolation` | Lagrange polynomial, linear interpolation, nearest-neighbor, `InterpolatedFunction` |
| `special_functions` | `gamma`, `beta`, `erf`, `erfc`, `bessel_j0`, `bessel_j1`, `zeta`, `lambert_w` |
| `numerical` | Integration, differentiation, root-finding (bisection, Newton), golden-section optimization |
| `sequence` | Fibonacci, Lucas, prime gaps, harmonic, sum, product, memoized sequences |
| `finance` | `future_value`, `present_value`, `annuity_future_value`, `rule_of_72`, `continuous_compound` |
| `checked_ops` | `checked_add`, `saturating_mul`, `approx_eq`, `inverse_lerp`, `remap` |

---

## Dependency Graph

```
                    mathverse-core
                         │
                ┌────────┴────────┐
                │                 │
         ┌──────┴──────┐   ┌─────┴─────┐
         │  Percentage  │   │   Power   │
         │  Root, Mod   │   │  Sci, Err │
         └──────┬──────┘   └─────┬─────┘
                │                 │
                └────────┬────────┘
                         │
                ┌────────┴────────┐
                │                 │
          ┌─────┴─────┐   ┌──────┴──────┐
          │  Complex   │   │  Rational   │
          │  Numbers   │   │  Fractions  │
          └─────┬─────┘   └──────┬──────┘
                │                 │
                └────────┬────────┘
                         │
                ┌────────┴────────┐
                │  Continued Frac │
                │  Special Funcs  │
                │  Numerical      │
                └─────────────────┘
```

---

## Installation

```toml
[dependencies]
mathverse-arithmetic = "0.1"
```

For `no_std` environments:

```toml
mathverse-arithmetic = { version = "0.1", default-features = false }
```

---

## Quick Start

```rust
use mathverse_arithmetic::prelude::*;

fn main() {
    // Percentage
    let discounted = decrease_by_percent(200.0, 15.0);
    // 200 - (200 × 0.15) = 170.0

    // Powers
    let cube = power(3.0, 3);        // 27.0
    let root = nth_root(27.0, 3);    // 3.0

    // Scientific notation
    let sci = ScientificNotation::from_f64(123456.0);
    // 1.23456 × 10⁵

    // Complex arithmetic
    let z1 = Complex::new(1.0, 2.0);
    let z2 = Complex::new(3.0, -1.0);
    let sum = z1 + z2;   // 4.0 + 1.0i
    let prod = z1 * z2;  // 5.0 + 1.0i

    // Exact rational arithmetic
    let r = Rational::new(1, 3) + Rational::new(1, 6);
    // 1/2 (exact, no floating-point error)

    // Special functions
    let gamma_5 = gamma(5.0);        // 24.0 (= 4!)
    let erf_0 = erf(0.0);            // 0.0

    // Numerical integration: ∫₀¹ x² dx = 1/3
    let integral = trapezoidal(|x| x * x, 0.0, 1.0, 1000);
    // ≈ 0.3333...

    println!("Discounted: {discounted}");
    println!("3³ = {cube}");
    println!("∛27 = {root}");
    println!("123456 in sci: {sci}");
    println!("z1 + z2 = {sum}");
    println!("z1 × z2 = {prod}");
    println!("1/3 + 1/6 = {r}");
    println!("Γ(5) = {gamma_5}");
    println!("∫₀¹ x² dx ≈ {integral}");
}
```

---

## Module Reference

### `percentage` — Percent Calculations

| Function | Formula | Used for |
|----------|---------|----------|
| `find_percent(value, total)` | `(value / total) × 100` | Share of total |
| `increase_by_percent(value, pct)` | `value × (1 + pct/100)` | Price increases |
| `decrease_by_percent(value, pct)` | `value × (1 - pct/100)` | Discounts, tax |
| `find_percent_difference(a, b)` | `|a - b| / ((a + b)/2) × 100` | Symmetric comparison |
| `find_reverse_percent(total, part)` | `(part / total) × 100` | Find % of a part |

```rust
let tax = increase_by_percent(89.99, 8.25);
// 89.99 × 1.0825 ≈ 97.41
```

---

### `power` — Powers & Exponents

| Function | Formula | Used for |
|----------|---------|----------|
| `power(base, exp)` | `base^exp` | General integer exponent |
| `power_with_sign(base, exp)` | `sign(base^exp) × |base|^exp` | Negative bases |
| `fast_power_mod(base, exp, m)` | `base^exp mod m` | Cryptography, hashing |
| `power_series(x, terms)` | `Σ x^n / n!` | Taylor approximation of eˣ |

---

### `root` — Roots

| Function | Formula | Used for |
|----------|---------|----------|
| `nth_root(x, n)` | `x^(1/n)` for `x ≥ 0` | General root |
| `sqrt(x)` | `x^(1/2)` | Standard square root |
| `cbrt(x)` | `x^(1/3)` | Cube root (works for negative) |
| `sqrt_with_sign(x)` | `sign(x) × √|x|` | Preserves sign information |

---

### `complex` — Complex Numbers

```rust
let z = Complex::new(3.0, 4.0);

z.abs()              // 5.0  (magnitude)
z.phase()            // 0.927... rad (atan2(4, 3))
z.conjugate()        // 3 - 4i
z.norm_squared()     // 25.0

// Polar form
let p = Complex::from_polar(5.0, PI / 4.0);
// 3.5355... + 3.5355...i

// Transcendental
z.sqrt()             // 2.0 + 1.0i
z.exp()              // -11.317... + 2.857...i
z.ln()               // 1.609... + 0.927...i
z.sin()              // 3.853... + (-27.016...)i
```

---

### `rational` — Exact Fraction Arithmetic

```rust
let a = Rational::new(1, 3);   // 1/3
let b = Rational::new(2, 5);   // 2/5

let sum = a + b;    // 11/15 (exact)
let prod = a * b;   // 2/15
let inv = b.inv();   // 5/2

let as_f64: f64 = sum.into(); // 0.7333...
```

The `Rational` type automatically reduces via GCD at each operation.

---

### `special_functions` — Mathematical Special Functions

| Function | Formula | Domain |
|----------|---------|--------|
| `gamma(x)` | Γ(x) = ∫₀∞ t^(x-1) e^(-t) dt | x > 0 |
| `beta(a, b)` | B(a,b) = Γ(a)Γ(b)/Γ(a+b) | a, b > 0 |
| `erf(x)` | (2/√π) ∫₀ˣ e^(-t²) dt | All real x |
| `erfc(x)` | 1 - erf(x) | All real x |
| `bessel_j0(x)` | J₀(x) | All real x |
| `bessel_j1(x)` | J₁(x) | All real x |
| `zeta(s)` | Σ 1/n^s | s > 1 |
| `lambert_w(x)` | W where W·e^W = x | x ≥ -1/e |

---

### `numerical` — Numerical Methods

```rust
// ∫₀¹ eˣ dx = e - 1 ≈ 1.71828
let result = trapezoidal(|x| x.exp(), 0.0, 1.0, 10000);

// Find root of x² - 2 = 0
let root = newton_raphson(
    |x| x * x - 2.0,
    |x| 2.0 * x,
    1.0, 1e-10
);
// ≈ 1.41421356237 (√2)
```

---

## Roadmap

| Phase | What |
|-------|------|
| 0.3.0 | `Quaternion<T>` type |
| 0.3.0 | Adaptive integration (Romberg, Gauss-Kronrod) |
| 0.4.0 | `BigInt` / `BigRational` for arbitrary precision |
| 0.4.0 | Sparse polynomial support |
| 0.5.0 | GPU-accelerated batch operations |
| 0.5.0 | WASM build with web-target optimizations |

---

## License

MIT — see [LICENSE](LICENSE).
