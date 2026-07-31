# mathverse-trigonometry

> Complete trigonometry toolkit — circular, hyperbolic, and inverse functions, plus identities, laws, special functions, and coordinate conversions. Generic over `Real`, works with `f32` and `f64`.

[![MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE)

## Features

- **6 circular trig functions** (sin, cos, tan, cot, sec, csc) + all inverses
- **6 hyperbolic trig functions** (sinh, cosh, tanh, coth, sech, csch) + all inverses
- **Degree variants** for every function (`sin_deg`, `acos_deg`, `sinh_deg`, …)
- **Angle conversions** — radians, degrees, gradians, turns
- **Coordinate systems** — polar/cartesian, spherical (physics + math), cylindrical
- **Trigonometric identities** — double/half angle, sum/difference, product-to-sum, power reduction
- **Geometric laws** — law of sines, law of cosines, Heron's formula, haversine distance
- **Special functions** — sinc, versine family, Gudermannian, Chebyshev polynomials, sin/cos powers
- Generic over `Real` trait — zero-cost `f32`/`f64` support

## Module Overview

| Module | Purpose | Key Functions |
|---|---|---|
| `lib.rs` | Core trig functions (radian + degree) | `sin`, `cos`, `tan`, `cot`, `sec`, `csc`, `sinh`, `cosh`, `tanh`, `coth`, `sech`, `csch`, `asin`, `acos`, `atan`, `atan2`, `acot`, `asec`, `acsc`, `asinh`, `acosh`, `atanh`, `acoth`, `asech`, `acsch` + all `*_deg` variants |
| `conversions` | Angle normalization + coordinate transforms | `wrap_angle`, `wrap_angle_positive`, `turns_to_radians`, `radians_to_turns`, `rad_to_grad`, `grad_to_rad`, `polar_to_cartesian`, `cartesian_to_polar`, `spherical_to_cartesian`, `cartesian_to_spherical`, `cylindrical_to_cartesian`, `cartesian_to_cylindrical` |
| `identities` | Trigonometric identities & formulas | `sin_cos`, `sin_double`, `cos_double`, `tan_double`, `sin_half`, `cos_half`, `tan_half`, `sin_sum`, `sin_diff`, `cos_sum`, `cos_diff`, `tan_sum`, `tan_diff`, `sin_squared`, `cos_squared`, `tan_squared` + product-to-sum / sum-to-product |
| `laws` | Triangle & spherical geometry laws | `law_of_sines_side`, `law_of_sines_angle`, `law_of_cosines_side`, `law_of_cosines_angle`, `heron`, `triangle_area_sas`, `triangle_area_base_height`, `bearing`, `haversine_distance` |
| `special` | Special trigonometric functions | `sinc`, `sinc_unnorm`, `versine`, `coversine`, `vercosine`, `covercosine`, `haversine`, `havercosine`, `hacoversine`, `hacovercosine`, `exsecant`, `excosecant`, `gudermannian`, `gudermannian_inv`, `gudermannian_alt`, `chebyshev_first`, `chebyshev_second`, `sin_power`, `cos_power` |

## Installation

```toml
[dependencies]
mathverse-trigonometry = { path = "../mathverse-trigonometry" }
```

Or add via workspace:

```toml
[dependencies]
mathverse-trigonometry.workspace = true
```

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

## Module Documentation

### Circular & Hyperbolic Functions (`lib`)

All functions accept `T: Real` (radians for circular, degrees for `*_deg` variants).

```
        ╱│
       ╱ │  sin(θ) = opposite / hypotenuse
  hyp ╱  │  cos(θ) = adjacent / hypotenuse
     ╱   │  tan(θ) = opposite / adjacent
    ╱ θ  │
   ──────
   adj
```

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

**Use cases:** physics simulations, signal processing, any domain involving angles.

---

### Angle Conversions (`conversions`)

```
  Turns ─────► Radians ─────► Degrees
   0.5          π              180°
               200 ────────────► Gradians
```

| Function | Formula |
|---|---|
| `wrap_angle(x)` | Normalize to `[-π, π)` |
| `wrap_angle_positive(x)` | Normalize to `[0, 2π)` |
| `turns_to_radians(t)` | `t × 2π` |
| `radians_to_turns(r)` | `r / 2π` |
| `rad_to_grad(r)` | `r × 200/π` |
| `grad_to_rad(g)` | `g × π/200` |

```rust
use mathverse_trigonometry::conversions::{wrap_angle, polar_to_cartesian};

let angle = wrap_angle(3.0 * std::f64::consts::PI); // wraps to -π
let (x, y) = polar_to_cartesian(2.0, std::f64::consts::FRAC_PI_3);
// x ≈ 1.0, y ≈ 1.7321
```

**Coordinate systems:**

```
  Polar (r, θ)          Spherical (r, θ, φ)          Cylindrical (r, θ, z)
       y                     z                            z
       │   ╱ (r,θ)           │  ╱ (r,θ,φ)                │  ╱ (r,θ,z)
       │  ╱                  │ ╱                          │ ╱
       │ ╱                   │╱                          │╱
  ─────┼──── x          ────┼──── y                 ────┼──── y
       │                     │                           │
                          x ─┘                        x ─┘
```

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

```rust
use mathverse_trigonometry::identities::{sin_double, cos_double, sin_sum};

assert!((sin_double(0.5) - (2.0 * 0.5f64.sin() * 0.5f64.cos())).abs() < 1e-12);
assert!((sin_sum(0.3, 0.7) - (0.3 + 0.7).sin()).abs() < 1e-12);
```

---

### Triangle & Spherical Laws (`laws`)

```
         A
        /\
   b   /  \   a
      /    \
     / B  C \
    ─────────
        a
```

| Law | Formula |
|---|---|
| Law of sines | `a/sin(A) = b/sin(B) = c/sin(C)` |
| Law of cosines | `c² = a² + b² - 2ab·cos(C)` |
| Heron's formula | `A = √(s(s-a)(s-b)(s-c))` where `s = (a+b+c)/2` |
| SAS area | `A = ½ab·sin(C)` |
| Haversine | `d = 2r · arcsin(√(sin²(Δlat/2) + cos(lat₁)cos(lat₂)sin²(Δlon/2)))` |

```rust
use mathverse_trigonometry::laws::{law_of_cosines_side, heron, haversine_distance};

// Find side c given a=3, b=4, C=90°
let c = law_of_cosines_side(3.0, 4.0, std::f64::consts::FRAC_PI_2);
assert!((c - 5.0).abs() < 1e-10);

// Heron's formula: 3-4-5 triangle
assert!((heron(3.0, 4.0, 5.0) - 6.0).abs() < 1e-10);
```

**Use cases:** surveying, navigation, geographic computations.

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
| `sin_power(n,x)` | `sinⁿ(x)` via Chebyshev | Power integrals |

```rust
use mathverse_trigonometry::special::{sinc, haversine, gudermannian};

assert!((sinc(0.0) - 1.0).abs() < 1e-12);   // sinc(0) = 1
assert!((sinc(1.0)).abs() < 1e-12);          // sinc(1) = 0
assert!((haversine(std::f64::consts::PI) - 1.0).abs() < 1e-12);
```

## Future Scope

- [ ] Inverse trig identities (arcsum, arcdiff)
- [ ] Angle addition for hyperbolic functions
- [ ] Spherical trigonometry (napier's rules, spherical excess)
- [ ] `no_std` support with feature flag
- [ ] SIMD-optimized batch trig evaluations
- [ ] Compile-time angle evaluation via `const fn`

## License

MIT OR Apache-2.0
