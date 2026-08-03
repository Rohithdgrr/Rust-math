# MathVerse Complex

[![Crates.io](https://img.shields.io/crates/v/mathverse-complex.svg)](https://crates.io/crates/mathverse-complex)
[![docs.rs](https://docs.rs/mathverse-complex/badge.svg)](https://docs.rs/mathverse-complex)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Complex number arithmetic, analysis, special functions, and matrix algebra over `f64`.

---

## Features

- **Full `Complex` type** — arithmetic operators, polar form, principal roots, exponentiation
- **Transcendental functions** — trigonometric, hyperbolic, inverse trigonometric, logarithmic
- **Complex analysis** — contour integration, residue calculus, Laurent series, conformal mappings
- **Special functions** — gamma, zeta, polylogarithm, Bessel, Airy, Fresnel over complex domain
- **Complex matrix algebra** — LU decomposition, determinant, inverse, eigenvalues
- Zero external dependencies

## Module Overview

| Module | Purpose |
|--------|---------|
| `Complex` (root) | Core type, arithmetic, transcendental functions |
| `analysis` | Residues, contour integrals, conformal mappings, Cauchy-Riemann checks |
| `special_functions` | Gamma, zeta, polylog, erf, Bessel, Airy, Fresnel |
| `matrix` | Complex matrix algebra and decompositions |

## Installation

```toml
[dependencies]
mathverse-complex = "0.1"
```

## Quick Start

```rust
use mathverse_complex::Complex;

fn main() {
    let z = Complex::new(3.0, 4.0);
    println!("|z| = {}", z.norm());       // 5
    println!("z² = {}", z * z);          // (-7, 24)
    println!("√z = {}", z.sqrt());       // (2, 1)
}
```

---

## Per-Module Documentation

### `Complex` — Core Type

#### Construction

```rust
let a = Complex::new(1.0, 2.0);       // 1 + 2i
let b = Complex::real(5.0);            // 5 + 0i
let c = Complex::i();                  // 0 + 1i
let d = Complex::polar(2.0, PI / 2.0); // 0 + 2i  (r·e^(iθ))
let e: Complex = 3.0.into();           // From<f64>
let f: Complex = (1.0, 2.0).into();   // From<(f64, f64)>
```

#### Properties

| Method | Formula |
|--------|---------|
| `norm()` | √(re² + im²) |
| `norm_sq()` | re² + im² |
| `arg()` | atan2(im, re) ∈ (-π, π] |
| `conjugate()` | re - im·i |
| `signum()` | z / \|z\| (unit magnitude) |
| `is_zero()` | re == 0 && im == 0 |

#### Functions

| Method | Formula / Notes |
|--------|-----------------|
| `sqrt()` | Principal square root via polar form |
| `exp()` | e^z = e^re(cos im + i·sin im) |
| `ln()` | Principal: ln\|z\| + i·arg(z) |
| `pow(p)` | e^(p·ln z), principal branch |
| `sin/cos/tan` | Complex trigonometric functions |
| `sinh/cosh/tanh` | Complex hyperbolic functions |
| `asin/acos/atan` | Inverse trigonometric functions |

#### Arithmetic

Implements `Add`, `Sub`, `Mul`, `Div`, `Neg`:

```rust
let z1 = Complex::new(1.0, 2.0);
let z2 = Complex::new(3.0, 4.0);
let sum  = z1 + z2;   // (4, 6)
let prod = z1 * z2;   // (-5, 10)
let quot = z1 / z2;   // (0.44, 0.08)
```

---

### `analysis` — Complex Analysis

#### Residues

```rust
use mathverse_complex::{Complex, analysis::ComplexAnalysis};

// f(z) = 1/(z-1), residue at z=1 is 1
let f = |z: Complex| Complex::one() / (z - Complex::real(1.0));
let residue = ComplexAnalysis::residue_simple_pole(&f, Complex::real(1.0), 0.001);
```

#### Contour Integration

```rust
// ∮ z dz around unit circle = 0
let f = |z: Complex| z;
let result = ComplexAnalysis::contour_integral_circle(&f, Complex::zero(), 1.0, 1000);
```

#### Analyticity & Conformal Mappings

- `is_analytic(&f, z, tol)` — checks Cauchy-Riemann equations
- `mobius_transform(z,a,b,c,d)` — Möbius transform w = (az+b)/(cz+d)
- `argument_principle(f,z₀,r,n)` — zeros minus poles count
- `laurent_series_coefficients(...)` — positive and negative power coefficients

---

### `special_functions` — Complex Special Functions

#### Gamma Function

Lanczos approximation with reflection formula for Re(z) < 0.5.

```rust
use mathverse_complex::special_functions::ComplexSpecialFunctions;

let g = ComplexSpecialFunctions::gamma(Complex::real(5.0));
// Γ(5) = 24
```

#### Error Functions & Bessel

```rust
let e = ComplexSpecialFunctions::erf(Complex::real(1.0), 50);
// erf(1) ≈ 0.8427

let j0 = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(1.0), 50);
// J₀(1) ≈ 0.7652
```

---

### `matrix` — Complex Matrix Algebra

```rust
use mathverse_complex::{Complex, matrix::ComplexMatrix};

let mut a = ComplexMatrix::new(2, 2);
a.set(0, 0, Complex::real(2.0));
a.set(0, 1, Complex::real(1.0));
a.set(1, 0, Complex::real(1.0));
a.set(1, 1, Complex::real(1.0));

let b = vec![Complex::real(3.0), Complex::real(2.0)];
let x = a.solve(&b).unwrap();
// x ≈ [1.0, 1.0]
```

Supports LU decomposition, determinant, inverse, power, matrix exp/ln, Hermitian/unitary checks.

---

## Future Scope

- Eigenvalue decomposition (QR algorithm)
- SVD (singular value decomposition)
- Matrix exponential via Padé approximation
- Fast Fourier Transform on complex arrays
- Quaternion support
- SIMD-accelerated operations

## License

MIT — see [LICENSE](LICENSE).
