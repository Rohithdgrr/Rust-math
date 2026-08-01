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
- **50+ math constants** — pi, tau, e, phi, sqrt(2), and their reciprocals
- **Number theory** — GCD, LCM, primes (sieve, Miller-Rabin), factorials, Fibonacci, Bell numbers, Euler phi
- **Combinatorics** — Catalan, Stirling, partitions, Pascal's triangle, derangements
- **Prelude** — one-import access to the most-used items
- **`#[must_use]`** on all pure functions — prevent silent discard bugs
- **`#[inline]`** on trivial leaf functions — optimizer hints for generic code
- **Comprehensive doc-tests** — every public function has runnable examples
- **Property-based tests** — mathematical invariants verified via `proptest`
- **Benchmarks** — `criterion` benchmarks for hot paths

---

## Module Overview

| Module | What it does |
|--------|-------------|
| `traits` | Core numeric trait hierarchy (`Num -> Signed -> Field -> Real`) |
| `ops` | Scalar math operations: lerp, smoothstep, hypot, nth_root, wrap, trig |
| `precision` | Float comparison: `almost_eq`, ULP, significant figures, safe division |
| `constants` | 50+ math constants (`PI`, `TAU`, `E`, `PHI`, `SQRT_*`, ...) |
| `algorithms` | GCD, LCM, primes, factorials, Fibonacci, binomial, Catalan, Bell |
| `error` | `MathError` enum, `MathResult<T>` type alias, `#[non_exhaustive]` |
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

    // Smooth easing (0->1 with ease-in-out curve)
    let y = smoothstep(0.5);               // 0.5

    // Float comparison with tolerance
    let eq = almost_eq(0.1 + 0.2, 0.3, 1e-6); // true

    // Use constants directly
    let area = PI * 5.0_f64.powi(2);       // 78.5398...

    // GCD and LCM
    let g = gcd(48, 18);                   // 6
    let l = lcm(4, 6);                     // 12

    // Prime checking
    assert!(is_prime(97));

    // Combinatorics
    let c5 = catalan_number(5);            // 42

    println!("lerp(0, 100, 0.3) = {x}");
    println!("smoothstep(0.5) = {y}");
    println!("0.1 + 0.2 ≈ 0.3? {eq}");
    println!("pi x 5² = {area:.4}");
    println!("gcd(48, 18) = {g}");
    println!("lcm(4, 6) = {l}");
    println!("catalan(5) = {c5}");
}
```

---

## Error Handling

```rust
use mathverse_core::error::{MathError, MathResult};

fn safe_sqrt(x: f64) -> MathResult<f64> {
    if x < 0.0 {
        Err(MathError::Domain)
    } else {
        Ok(x.sqrt())
    }
}

// Access the error message:
let err = MathError::DivisionByZero;
assert_eq!(err.msg(), "division by zero");
assert_eq!(err.to_string(), "division by zero");

// Convert to String:
let s: String = MathError::Overflow.into();
```

---

## MSRV

The minimum supported Rust version is **1.87**.

---

## Roadmap

| Phase | What |
|-------|------|
| 0.3.0 | Statistics module (mean, median, variance, correlation) |
| 0.3.0 | Numerical methods (root finding, integration, differentiation) |
| 0.4.0 | Polynomial arithmetic and interpolation |
| 0.4.0 | Special functions (gamma, beta, erf, bessel) |
| 0.5.0 | `Complex<T>` and `Matrix` types |
| 0.5.0 | SIMD-accelerated trait impls for `f32x4` / `f64x2` |

---

## License

MIT — see [LICENSE](LICENSE).
