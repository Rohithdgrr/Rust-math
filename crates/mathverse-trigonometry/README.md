# MathVerse Trigonometry

[![Crates.io](https://img.shields.io/crates/v/mathverse-trigonometry.svg)](https://crates.io/crates/mathverse-trigonometry)
[![docs.rs](https://docs.rs/mathverse-trigonometry/badge.svg)](https://docs.rs/mathverse-trigonometry)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Complete trigonometry toolkit — circular, hyperbolic, and inverse functions, plus identities, laws, special functions, coordinate conversions, batched evaluation, and exact special-angle values.

---

## Features

- **6 circular trig functions** (sin, cos, tan, cot, sec, csc) + all inverses
- **6 hyperbolic trig functions** (sinh, cosh, tanh, coth, sech, csch) + all inverses
- **Degree variants** for every function (`sin_deg`, `acos_deg`, `sinh_deg`, …)
- **Angle conversions** — radians, degrees, gradians, turns
- **Coordinate systems** — polar/cartesian, spherical (physics + math), cylindrical
- **Trigonometric identities** — double/half angle, sum/difference, product-to-sum, power reduction
- **Geometric laws** — law of sines, law of cosines, Heron's formula, haversine distance
- **Batched operations** (`batched`) — map/sum `sin`/`cos` over slices, additive synthesis, no allocation
- **Exact special angles** (`exact`) — closed-form `sin`/`cos`/`tan` for multiples of 30°/45°
- Generic over `Real` trait — zero-cost `f32`/`f64` support
- **`no_std`** — disable `std` and enable `libm` for embedded targets

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `lib.rs` | Core trig functions (radian + degree): `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`, `asin`, `acos`, `atan`, `atan2`, `acot`, `asec`, `acsc`, `asinh`, `acosh`, `atanh`, `acoth`, `asech`, `acsch` + all `*_deg` variants |
| `conversions` | Angle normalization + coordinate transforms: `wrap_angle`, `turns_to_radians`, `rad_to_grad`, `polar_to_cartesian`, `spherical_to_cartesian`, `cylindrical_to_cartesian` |
| `identities` | Trigonometric identities & formulas: `sin_cos`, `sin_double`, `cos_double`, `sin_half`, `sin_sum`, `cos_sum`, `sin_squared`, product-to-sum, sum-to-product |
| `laws` | Triangle & spherical geometry laws: `law_of_sines_side`, `law_of_cosines_side`, `heron`, `triangle_area_sas`, `bearing`, `haversine_distance` |
| `special` | Special trigonometric functions: `sinc`, `versine`, `haversine`, `exsecant`, `gudermannian`, `chebyshev_first`, `chebyshev_second`, `sin_power`, `cos_power` |
| `batched` | Slice-based batch trig (DSP/audio/graphics): `map_sin`, `map_cos`, `map_sin_cos`, `sin_inplace`, `sum_sin`, `sum_cos`, `accumulate_sine` |
| `exact` | Exact values for special angles: `sin_exact_deg`, `cos_exact_deg`, `tan_exact_deg`, `ExactValue` |

---

## Installation

```toml
[dependencies]
mathverse-trigonometry = "0.1"
```

`std` is the default feature. For embedded / `no_std` targets:

```toml
mathverse-trigonometry = { version = "0.1", default-features = false, features = ["libm"] }
```

---

## Quick Start

```rust
use mathverse_trigonometry::{sin_deg, cos_deg, heron, haversine_distance, polar_to_cartesian};

fn main() {
    // Trig in degrees — no conversion needed
    let s = sin_deg(30.0);
    let c = cos_deg(60.0);
    println!("sin(30°) = {s}, cos(60°) = {c}");
    // sin(30°) = 0.5, cos(60°) = 0.5

    // Triangle area from 3 sides (Heron's formula)
    let area = heron(3.0, 4.0, 5.0);
    println!("3-4-5 triangle area = {area}");
    // 3-4-5 triangle area = 6

    // Polar → Cartesian
    let (x, y) = polar_to_cartesian(1.0, std::f64::consts::FRAC_PI_4);
    println!("Polar (1, π/4) → Cartesian ({x:.4}, {y:.4})");
    // Polar (1, π/4) → Cartesian (0.7071, 0.7071)

    // Haversine distance (lat/lon in radians)
    let d = haversine_distance(0.0, 0.0, 0.0, std::f64::consts::PI, 6_371_000.0);
    println!("Equator half-circumference = {d:.0} m");
    // Equator half-circumference = 20015087 m
}
```

---

## Module Documentation

### Circular & Hyperbolic Functions (`lib`)

All functions accept `T: Real` (radians for circular, degrees for `*_deg` variants).

```rust
use mathverse_trigonometry::{sin, cos, tan, cot, sec, csc, sinh, cosh, tanh};

// Circular (radians)
assert!((sin(0.5) - 0.4794).abs() < 0.001);
assert!((cos(0.5) - 0.8776).abs() < 0.001);
assert!((tan(0.5) - 0.5463).abs() < 0.001);

// Hyperbolic
assert!((sinh(1.0) - 1.1752).abs() < 0.001);
assert!((cosh(1.0) - 1.5431).abs() < 0.001);

// Inverses
assert!((asin(0.5) - std::f64::consts::FRAC_PI_6).abs() < 1e-12);
```

**Degree variants:** `sin_deg(30.0) == 0.5`, `acos_deg(0.5) == 60.0`

---

### Angle Conversions (`conversions`)

| Function | Formula |
|---|---|
| `wrap_angle(x)` | Normalize to `[-π, π)` |
| `wrap_angle_positive(x)` | Normalize to `[0, 2π)` |
| `turns_to_radians(t)` | `t × 2π` |
| `radians_to_turns(r)` | `r / 2π` |
| `rad_to_grad(r)` | `r × 200/π` |
| `grad_to_rad(g)` | `g × π/200` |

**Coordinate systems:** polar, spherical (physics + math), cylindrical — all with bidirectional conversions.

---

### Trigonometric Identities (`identities`)

| Identity | Formula |
|---|---|
| Double angle | `sin(2x) = 2 sin(x) cos(x)` |
| Double angle | `cos(2x) = 2cos²(x) - 1` |
| Half angle | `sin(x/2) = ±√((1 - cos(x))/2)` |
| Sum | `sin(a+b) = sin(a)cos(b) + cos(a)sin(b)` |
| Difference | `cos(a-b) = cos(a)cos(b) + sin(a)sin(b)` |
| Power reduction | `sin²(x) = (1 - cos(2x))/2` |
| Product-to-sum | `sin(a)sin(b) = [cos(a-b) - cos(a+b)] / 2` |
| Sum-to-product | `sin(a)+sin(b) = 2 sin((a+b)/2) cos((a-b)/2)` |

---

### Triangle & Spherical Laws (`laws`)

| Law | Formula |
|---|---|
| Law of sines | `a/sin(A) = b/sin(B) = c/sin(C)` |
| Law of cosines | `c² = a² + b² - 2ab·cos(C)` |
| Heron's formula | `A = √(s(s-a)(s-b)(s-c))` where `s = (a+b+c)/2` |
| SAS area | `A = ½ab·sin(C)` |
| Haversine | `d = 2r · arcsin(√(sin²(Δlat/2) + cos(lat₁)cos(lat₂)sin²(Δlon/2)))` |

---

### Special Functions (`special`)

| Function | Formula | Use |
|---|---|---|
| `sinc(x)` | `sin(πx)/(πx)` | Signal processing, interpolation |
| `versine(x)` | `1 - cos(x)` | Navigation (haversine formula) |
| `haversine(x)` | `(1 - cos(x))/2` | Great-circle distance |
| `exsecant(x)` | `sec(x) - 1` | Surveying |
| `gudermannian(x)` | `2·arctan(eˣ) - π/2` | Mercator projection |
| `chebyshev_first(n,x)` | `Tₙ(x) = cos(n·acos(x))` | Polynomial approximation |
| `chebyshev_second(n,x)` | `Uₙ(x) = sin((n+1)·acos(x))/sin(acos(x))` | Polynomial approximation |

---

### Batched Operations (`batched`)

Slice-based helpers for DSP, audio synthesis, and graphics tight loops. No allocation required — `no_std` compatible.

| Function | Behavior |
|---|---|
| `map_sin(xs, out)` / `map_cos(xs, out)` | `out[i] = sin(xs[i])` / `cos(xs[i])` |
| `map_sin_cos(xs, sin_out, cos_out)` | `(sin(xᵢ), cos(xᵢ))` in one pass via `sin_cos()` |
| `sin_inplace(xs)` | Replace each element with its sine |
| `sum_sin(xs)` / `sum_cos(xs)` | `Σ sin(xᵢ)` / `Σ cos(xᵢ)` |
| `accumulate_sine(freq, phases, amps, out)` | Additive synthesis: `out[i] = Σₖ amps[k]·sin(freq·i + phases[k])` |

---

### Exact Special Angles (`exact`)

Closed-form values for angles that are multiples of 30° or 45°; all other angles return `None`. Values are represented by `ExactValue` — integers, halves, and `c·√r / d` — so they render symbolically (`"√3/2"`) and convert exactly via `to_f64()`.

```rust
use mathverse_trigonometry::{sin_exact_deg, cos_exact_deg, ExactValue};

assert_eq!(sin_exact_deg(30), Some(ExactValue::Half(1)));        // 1/2
assert_eq!(sin_exact_deg(30).unwrap().to_string(), "1/2");
assert_eq!(cos_exact_deg(45), Some(ExactValue::Root { coeff: 1, radicand: 2, denom: 2 }));
assert_eq!(cos_exact_deg(45).unwrap().to_string(), "√2/2");
assert_eq!(sin_exact_deg(15), None);                              // not a special angle
```

---

## Roadmap

- [ ] Inverse trig identities (arcsum, arcdiff)
- [ ] Angle addition for hyperbolic functions
- [ ] Spherical trigonometry (napier's rules, spherical excess)
- [ ] SIMD-vectorized batch evaluations
- [ ] Exact values for additional angles (e.g. 15°, 18°) and hyperbolic functions
- [ ] Compile-time angle evaluation via `const fn`

---

## License

MIT — see [LICENSE](LICENSE).
