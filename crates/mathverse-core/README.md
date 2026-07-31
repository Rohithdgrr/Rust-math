# MathVerse Core

**Shared substrate for the MathVerse ecosystem — traits, scalar operations, precision, constants, and algorithms.**

[![Rust](https://img.shields.io/badge/Rust-2021-EA5727?style=flat-square&logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Features

- **Zero-dependency** — runs on `no_std`, embedded-friendly
- **Generic math traits** — `Num`, `Signed`, `Field`, `Real` with blanket impls for `f32`/`f64`
- **30+ scalar operations** — lerp, smoothstep, nth_root, hypot, wrap, angle conversions
- **Float precision** — ULP comparison, relative/absolute tolerances, safe division
- **50+ math constants** — π, τ, e, φ, √2, and their reciprocals
- **Number theory** — GCD, LCM, primes (sieve), factorials, Fibonacci, Bell numbers, Euler φ
- **Prelude** — one-import access to the most-used items

---

## Module Overview

| Module | What it does |
|--------|-------------|
| `traits` | Core numeric trait hierarchy (`Num → Signed → Field → Real`) |
| `ops` | Scalar math operations: lerp, smoothstep, hypot, nth_root, wrap, trig |
| `precision` | Float comparison: `almost_eq`, ULP, significant figures, safe division |
| `constants` | 50+ math constants (`PI`, `TAU`, `E`, `PHI`, `SQRT_*`, …) |
| `algorithms` | GCD, LCM, primes, factorials, Fibonacci, binomial, Catalan, Bell |
| `error` | `MathError` enum and `MathResult<T>` type alias |
| `prelude` | Re-exports of the most commonly used items |

---

## Trait Hierarchy

```
                        ┌─────────┐
                        │   Num   │   Integer + Float
                        └────┬────┘
                             │
                    ┌────────┴────────┐
                    │                 │
               ┌────┴────┐     ┌─────┴─────┐
               │ Signed  │     │ Unsigned* │  (*marker, not enforced)
               └────┬────┘     └───────────┘
                    │
           ┌────────┴────────┐
           │                 │
      ┌────┴────┐     ┌──────┴──────┐
      │  Field  │     │   (marker)  │
      └────┬────┘     └─────────────┘
           │
      ┌────┴────┐
      │  Real   │   f32 / f64 blanket impl
      └─────────┘
```

All traits are implemented for `f32` and `f64` via blanket impls, so generic code works out of the box.

---

## Installation

```toml
[dependencies]
mathverse-core = { path = "../mathverse-core" }
```

For `no_std` environments, disable the `std` feature:

```toml
mathverse-core = { path = "../mathverse-core", default-features = false }
```

---

## Quick Start

```rust
use mathverse_core::prelude::*;

fn main() {
    // Linear interpolation
    let x = lerp(0.0, 100.0, 0.3);        // 30.0

    // Smooth easing (0→1 with ease-in-out curve)
    let y = smoothstep(0.0, 10.0, 5.0);   // 0.5

    // Float comparison with tolerance
    let eq = almost_eq(0.1 + 0.2, 0.3, 1e-6); // true

    // Use constants directly
    let area = PI * 5.0_f64.powi(2);       // 78.5398…

    // GCD and LCM
    let g = gcd(48, 18);                   // 6
    let l = lcm(4, 6);                     // 12

    println!("lerp(0, 100, 0.3) = {x}");
    println!("smoothstep(0, 10, 5) = {y}");
    println!("0.1 + 0.2 ≈ 0.3? {eq}");
    println!("π × 5² = {area:.4}");
    println!("gcd(48, 18) = {g}");
    println!("lcm(4, 6) = {l}");
}
```

**Output:**
```
lerp(0, 100, 0.3) = 30
smoothstep(0, 10, 5) = 0.5
0.1 + 0.2 ≈ 0.3? true
π × 5² = 78.5398
gcd(48, 18) = 6
lcm(4, 6) = 12
```

---

## Module Reference

### `traits` — Numeric Trait Hierarchy

| Trait | Description | Implemented for |
|-------|-------------|-----------------|
| `Num` | Scalar + Copy + Clone + Debug + PartialOrd + PartialEq | f32, f64, i8–i128, u8–u128 |
| `Signed` | Adds `abs()`, `signum()` | f32, f64, i8–i128 |
| `Field` | Adds `/`, inv(), `powi`, `from_f64` | f32, f64 |
| `Real` | Adds `floor()`, `ceil()`, `sqrt()`, `sin()`, `cos()`, `ln()`, `exp()`, `from_f64` | f32, f64 |

```rust
fn midpoint<T: Real>(a: T, b: T) -> T {
    (a + b) / T::from_f64(2.0)
}
```

---

### `ops` — Scalar Operations

| Function | Formula | Used for |
|----------|---------|----------|
| `lerp(a, b, t)` | `a + (b - a) * t` | Animation, interpolation |
| `smoothstep(edge0, edge1, x)` | `3t² - 2t³` where `t = clamp((x - e0) / (e1 - e0))` | Smooth easing |
| `nth_root(x, n)` | `x^(1/n)` | General root computation |
| `hypot(a, b)` | `√(a² + b²)` | Euclidean distance |
| `wrap(val, min, max)` | `min + ((val - min) % (max - min))` | Angle wrapping, cycling |
| `map_range(val, a, b, c, d)` | Linear map from `[a,b]` → `[c,d]` | Coordinate transforms |
| `to_degrees(rad)` | `rad × 180/π` | Angle conversion |
| `to_radians(deg)` | `deg × π/180` | Angle conversion |
| `angle_between(v1, v2)` | `atan2(cross, dot)` | Signed angle |

```rust
let distance = hypot(3.0, 4.0);       // 5.0
let wrapped = wrap(370.0, 0.0, 360.0); // 10.0
```

---

### `precision` — Float Comparison

| Function | Comparison | Used when |
|----------|-----------|-----------|
| `almost_eq(a, b, epsilon)` | `|a - b| < ε` | General-purpose, coarse tolerance |
| `almost_eq_rel(a, b, rel)` | `|a - b| / max(|a|, |b|) < rel` | Unknown magnitude |
| `is_close(a, b, rel, abs)` | Combined relative + absolute | Robust, combined tolerance |
| `almost_eq_ulp(a, b, n)` | ULP difference ≤ n | IEEE-754 bitwise precision |
| `round_to(x, decimals)` | Round to n decimal places | Display formatting |
| `significant_figures(x, n)` | Round to n significant figures | Scientific reporting |
| `safe_div(a, b, fallback)` | Returns `fallback` when `b ≈ 0` | Avoid NaN/Inf |

**Flow: choosing the right comparator**

```
                    ┌─────────────────┐
                    │  Are you comparing│
                    │  near-zero values? │
                    └───────┬─────────┘
                            │
                  yes ◄─────┴─────► no
                  │                 │
         use abs epsilon    use relative tolerance
         (almost_eq)        (almost_eq_rel / is_close)
                                    │
                            need IEEE exactness?
                           yes ◄────┴────► no
                           │              │
                     use ULP         is_close is fine
                    (almost_eq_ulp)
```

```rust
// General-purpose: quick check with absolute tolerance
assert!(almost_eq(0.1 + 0.2, 0.3, 1e-9));

// For large/small values with unknown scale
assert!(almost_eq_rel(1e15, 1e15 + 1.0, 1e-6));

// Safe division — never NaN
let val = safe_div(1.0, 0.0, 0.0); // 0.0, not NaN
```

---

### `constants` — Math Constants

All constants are `f64` with `f32` variants via `_F32` suffix.

| Constant | Value | Notes |
|----------|-------|-------|
| `PI` / `PI_F32` | 3.14159265… | Half-circle |
| `TAU` / `TAU_F32` | 6.28318530… | Full circle (2π) |
| `E` / `E_F32` | 2.71828182… | Euler's number |
| `PHI` / `PHI_F32` | 1.61803398… | Golden ratio |
| `SQRT_2` | 1.41421356… | √2 |
| `SQRT_3` | 1.73205080… | √3 |
| `LN_2` | 0.69314718… | Natural log of 2 |
| `LOG_2_E` | 1.44269504… | log₂(e) |
| `LOG_10_E` | 0.43429448… | log₁₀(e) |

```rust
let circumference = TAU * 5.0;     // 31.4159…
let reciprocal_pi = FRAC_1_PI;     // 0.31830…
```

---

### `algorithms` — Number Theory & Combinatorics

| Function | Formula | Used for |
|----------|---------|----------|
| `gcd(a, b)` | Euclidean algorithm | Reducing fractions |
| `lcm(a, b)` | `|a·b| / gcd(a,b)` | Synchronization periods |
| `extended_gcd(a, b)` | Bézout coefficients | Modular inverse |
| `is_prime(n)` | Trial division | Primality check |
| `primes_up_to(n)` | Sieve of Eratosthenes | Generating primes |
| `factorial(n)` | n! | Permutations |
| `fibonacci(n)` | F(n) = F(n-1) + F(n-2) | Growth models |
| `binomial(n, k)` | C(n,k) = n! / (k!·(n-k)!) | Combinations |
| `catalan(n)` | C(n) = C(2n,n)/(n+1) | Parenthesizations |
| `bell_number(n)` | Total partitions of {1..n} | Set partitioning |
| `stirling(n)` | √(2πn)·(n/e)ⁿ | Factorial approximation |
| `euler_phi(n)` | Count of coprimes ≤ n | Euler's theorem |
| `mobius(n)` | Möbius function | Inclusion-exclusion |

```rust
let primes: Vec<u64> = primes_up_to(50);
// [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

let b10 = bell_number(10); // 115975
let c5  = catalan(5);      // 42
let f10 = fibonacci(10);   // 55
```

---

### `error` — Error Handling

```rust
pub enum MathError {
    Domain(String),          // Value outside function domain
    DivisionByZero(String),  // Division by zero
    InvalidArgument(String), // Bad function argument
    NotConverged(String),    // Iterative method didn't converge
    Overflow(String),        // Arithmetic overflow
    Underflow(String),       // Arithmetic underflow
    PrecisionLoss(String),   // Loss of precision
}

pub type MathResult<T> = Result<T, MathError>;
```

```rust
use mathverse_core::error::MathResult;

fn safe_sqrt(x: f64) -> MathResult<f64> {
    if x < 0.0 {
        Err(MathError::Domain(
            format!("sqrt({x}) is undefined for negative x")
        ))
    } else {
        Ok(x.sqrt())
    }
}
```

---

## Roadmap

| Phase | What |
|-------|------|
| 0.3.0 | Fixed-point arithmetic type for embedded |
| 0.3.0 | SIMD-accelerated trait impls for `f32x4` / `f64x2` |
| 0.4.0 | Generic `Complex<T>` in core (currently in arithmetic) |
| 0.4.0 | `Matrix` and `Vector` traits |
| 0.5.0 | `const fn` equivalents for all constants |
| 0.5.0 | `no_std` + `alloc` dual support |

---

## License

MIT — see [LICENSE](LICENSE).
