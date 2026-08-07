# MathVerse Complex

[![Crates.io](https://img.shields.io/crates/v/mathverse-complex.svg)](https://crates.io/crates/mathverse-complex)
[![docs.rs](https://docs.rs/mathverse-complex/badge.svg)](https://docs.rs/mathverse-complex)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE.md)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Complex number arithmetic, analysis, special functions, and matrix algebra —
generic over any `RealFull` type (`f64` by default, with `f32` support).

---

## Features

- **Generic `Complex<T>`** — one implementation over `f64` (default), `f32`, or any
  [`RealFull`] type; `C32`/`C64` convenience aliases
- **Full arithmetic** — `Add`/`Sub`/`Mul`/`Div`/`Neg`, overflow-safe division
  (Smith's algorithm), scalar operations, `Index`-style accessors
- **Transcendental functions** — trigonometric, hyperbolic, inverse trigonometric,
  logarithmic, principal powers and roots (`0⁰ = 1` convention)
- **Python parity** — `cmath`/`numpy`-named methods: `phase`, `to_polar`/`polar`,
  `rect`, `is_close`, `real()`/`imag()`
- **Complex analysis** — contour integration, Cauchy integral/derivative formulas,
  complex-step derivative, residues, Laurent series, conformal mappings
- **Special functions** — gamma, digamma, zeta, polylogarithm, erf/erfc, Bessel,
  Airy, Fresnel, exponential integral over the complex domain
- **Complex matrix algebra** — LU decomposition, solve, inverse, determinant,
  QR decomposition, **eigenvalues** (Wilkinson-shift QR), matrix exp/ln
- **FFT** — radix-2 Cooley–Tukey `fft`/`ifft` with round-trip identity
- **Polynomial roots** — Durand–Kerner (Weierstrass) method for all complex roots
- **Mandelbrot** — iteration and smooth-coloring helpers
- No external runtime dependencies (only `mathverse-core`)

## Module Overview

| Module | Purpose |
|--------|---------|
| [`Complex`] (root) | Core type, arithmetic, transcendental functions, Mandelbrot |
| [`analysis`] | Contour integrals, Cauchy formulas, residues, conformal mappings |
| [`special_functions`] | Gamma, zeta, polylog, erf, Bessel, Airy, Fresnel |
| [`matrix`] | Complex matrix algebra, decompositions, eigenvalues |
| [`fft`] | Radix-2 fast Fourier transform |
| [`polynomial`] | Polynomial evaluation and root finding |

## Installation

```toml
[dependencies]
mathverse-complex = "0.2"
```

## Quick Start

```rust
use mathverse_complex::Complex;

fn main() {
    let z = Complex::new(3.0, 4.0);
    println!("|z| = {}", z.norm());       // 5
    println!("z² = {}", z * z);          // (-7, 24)
    println!("√z = {}", z.sqrt());       // (2, 1)

    // f32 works with the same code
    let w: mathverse_complex::C32 = Complex::new(1.0f32, -1.0);
    println!("arg(w) = {}", w.arg());    // -π/4
}
```

---

## Per-Module Documentation

### `Complex` — Core Type

#### Construction

```rust
let a = Complex::new(1.0, 2.0);          // 1 + 2i
let b = Complex::real(5.0);              // 5 + 0i
let c = Complex::i();                    // 0 + 1i
let d = Complex::polar(2.0, PI / 2.0);   // 0 + 2i  (r·e^(iθ))
let e: Complex = 3.0.into();             // From<f64>
let f: Complex = (1.0, 2.0).into();      // From<(f64, f64)>
```

#### Properties

| Method | Formula |
|--------|---------|
| `norm()` | √(re² + im²) |
| `norm_sq()` | re² + im² |
| `arg()` / `phase()` | atan2(im, re) ∈ (-π, π] |
| `to_polar()` / `polar()` | (norm, arg) — `cmath.polar` parity |
| `rect(r, θ)` | r·(cos θ + i·sin θ) — `cmath.rect` parity |
| `conjugate()` | re − im·i |
| `signum()` | z / \|z\| (unit magnitude) |
| `is_zero()` | re == 0 && im == 0 |
| `is_close(other, rel, abs)` | `cmath.isclose` semantics |
| `re` / `im` fields | Direct component access (like `numpy.real(z)` / `numpy.imag(z)`) |

#### Functions

| Method | Formula / Notes |
|--------|-----------------|
| `sqrt()` | Algebraic formula (Numerical Recipes §5.4), full precision near the real axis; branch cut on (−∞, 0] |
| `exp()` | e^z = e^re(cos im + i·sin im) |
| `ln()` | Principal: ln\|z\| + i·arg(z) |
| `pow(p)` | e^(p·ln z), principal branch; `0⁰ = 1`, `0^p = 0` for Re(p) > 0 |
| `sin/cos/tan` | Complex trigonometric functions |
| `sinh/cosh/tanh` | Complex hyperbolic functions |
| `asin/acos/atan` | Inverse trigonometric functions |
| `acosh/asinh/atanh` | Inverse hyperbolic functions (documented branch cuts) |

#### Arithmetic

Implements `Add`, `Sub`, `Mul`, `Div`, `Neg`, plus scalar `Complex<T> ± T`,
`Complex<T> · T`, `Complex<T> / T`:

```rust
let z1 = Complex::new(1.0, 2.0);
let z2 = Complex::new(3.0, 4.0);
let sum  = z1 + z2;   // (4, 6)
let prod = z1 * z2;   // (-5, 10)
let quot = z1 / z2;   // (0.44, 0.08)
```

Division uses Smith's overflow-safe algorithm: `(1e300 + 1e300i) / (1e300 + 1e300i)`
is exactly `1 + 0i`, where the naive `a·d/|d|²` formula overflows to `NaN`.

### `fft` — Fast Fourier Transform

```rust
use mathverse_complex::{fft, ifft, Complex};

// 1024-sample sine wave → spectrum → round-trip
let n = 1024;
let signal: Vec<Complex> = (0..n)
    .map(|k| Complex::new((2.0 * std::f64::consts::PI * k as f64 / n as f64 * 8.0).sin(), 0.0))
    .collect();
let spectrum = fft(&signal);
let back = ifft(&spectrum);
assert!((back[0] - signal[0]).norm() < 1e-10);
```

Input length must be a power of two (asserted).

### `polynomial` — Root Finding

```rust
use mathverse_complex::{polynomial_roots, Complex};

// z² + 1 = 0  →  ±i
let roots = polynomial_roots(&[Complex::one(), Complex::zero(), Complex::one()], 1000, 1e-12);
assert!(roots.len() == 2);
```

Coefficients are lowest-order first: `coeffs[0] + coeffs[1]·z + coeffs[2]·z² + …`.
Uses the Durand–Kerner (Weierstrass) iteration, globally convergent for any
polynomial degree.

### `analysis` — Complex Analysis

```rust
use mathverse_complex::{Complex, analysis::ComplexAnalysis};
```

- `derivative` / `derivative_complex_step` / `derivative_cauchy` — three ways to
  differentiate an analytic function
- `contour_integral_circle`, `cauchy_integral_formula`, `cauchy_derivative_formula`
- `residue_simple_pole` / `residue_pole_order_n`
- `is_analytic` — Cauchy–Riemann check
- `laurent_series_coefficients`, `mobius_transform`, `schwarz_christoffel`,
  `argument_principle`, `rouches_theorem`

### `special_functions` — Complex Special Functions

```rust
use mathverse_complex::special_functions::ComplexSpecialFunctions;

let g = ComplexSpecialFunctions::gamma(Complex::real(5.0));  // Γ(5) = 24
let z = ComplexSpecialFunctions::zeta(Complex::real(2.0), 1000); // ζ(2) ≈ π²/6
let e = ComplexSpecialFunctions::erf(Complex::real(1.0), 50); // erf(1) ≈ 0.8427
let j0 = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(1.0), 50); // J₀(1) ≈ 0.7652
```

Lanczos gamma with reflection; Euler–Maclaurin zeta with the functional equation
for Re(s) < 0; erf/erfc switch to an optimally-truncated asymptotic expansion
for `|z| ≥ 3`; Bessel series with early-exit convergence detection.

### `matrix` — Complex Matrix Algebra

```rust
use mathverse_complex::{Complex, matrix::ComplexMatrix};

let mut a = ComplexMatrix::new(2, 2);
a.set(0, 0, Complex::real(2.0));
a.set(0, 1, Complex::real(1.0));
a.set(1, 0, Complex::real(1.0));
a.set(1, 1, Complex::real(1.0));

let b = vec![Complex::real(3.0), Complex::real(2.0)];
let x = a.solve(&b).unwrap();  // x ≈ [1.0, 1.0]
```

Supports LU decomposition (relative pivot threshold), solve, inverse,
determinant, power, matrix exp/ln, QR decomposition, Hermitian/unitary checks,
and **eigenvalues** via the QR algorithm with Wilkinson shifts:

```rust
let e = a.eigenvalues(500, 1e-10)?;      // Vec<Complex>
use mathverse_complex::matrix::linalg;
let ev = linalg::eig(&a)?;               // scipy.linalg.eig parity
```

Dimension mismatches and out-of-bounds access return `MathResult`/`Option`
instead of panicking (`add`, `mul`, `try_get`, `try_set`); `get`/`set` follow the
slice-indexing convention for hot paths.

## Testing & Benchmarks

```bash
cargo test        # 80+ unit tests + doctests
cargo bench       # Criterion: FFT vs naive DFT, arithmetic, matrix ops, roots
```

## Rust Version & Safety

- Minimum supported Rust version: **1.87** (workspace policy)
- `#![forbid(unsafe_code)]` — no `unsafe` anywhere
- Division by zero, `ln(0)`, etc. yield `NaN`/`inf` components mirroring
  std float semantics — never panic

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT OR Apache-2.0 — see [LICENSE-MIT](LICENSE-MIT) and [LICENSE.md](LICENSE.md) (Apache-2.0).
