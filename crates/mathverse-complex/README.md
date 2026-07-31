# mathverse-complex

Complex number arithmetic, analysis, special functions, and matrix algebra over `f64`.

## Features

- Full `Complex` type with arithmetic operators (`+`, `-`, `*`, `/`, unary `-`)
- Trigonometric, hyperbolic, inverse trigonometric, and logarithmic functions
- Polar form conversion, principal roots, and exponentiation
- Contour integration, residue calculus, and Laurent series
- Complex gamma, zeta, polylogarithm, Bessel, Airy, and Fresnel functions
- Complex matrix operations: LU decomposition, determinant, inverse, eigenvalues
- Hermitian and unitary matrix checks
- Zero external dependencies

## Module Overview

| Module               | Purpose                                           |
|----------------------|---------------------------------------------------|
| `Complex` (root)     | Core type, arithmetic, transcendental functions    |
| `analysis`           | Residues, contour integrals, conformal mappings    |
| `special_functions`  | Gamma, zeta, polylog, erf, Bessel, Airy, Fresnel  |
| `matrix`             | Complex matrix algebra and decompositions          |

## Installation

```toml
[dependencies]
mathverse-complex = { path = "../mathverse-complex" }
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

## `Complex` — Core Type

### Construction

```rust
let a = Complex::new(1.0, 2.0);       // 1 + 2i
let b = Complex::real(5.0);            // 5 + 0i
let c = Complex::i();                  // 0 + 1i
let d = Complex::polar(2.0, PI / 2.0); // 0 + 2i  (r·e^(iθ))
let e: Complex = 3.0.into();           // From<f64>
let f: Complex = (1.0, 2.0).into();   // From<(f64, f64)>
```

### Properties

| Method          | Formula                           |
|-----------------|-----------------------------------|
| `norm()`        | √(re² + im²)                     |
| `norm_sq()`     | re² + im²                        |
| `arg()`         | atan2(im, re) ∈ (-π, π]          |
| `conjugate()`   | re - im·i                         |
| `signum()`      | z / \|z\| (unit magnitude)        |
| `is_zero()`     | re == 0 && im == 0               |
| `is_nan()`      | either component is NaN           |
| `is_infinite()` | either component is infinite      |
| `is_finite()`   | both components are finite        |

### Functions

| Method     | Formula / Notes                              |
|------------|----------------------------------------------|
| `sqrt()`   | Principal square root via polar form         |
| `cbrt()`   | Cube root via powf(1/3)                      |
| `root(n)`  | Principal nth root                           |
| `exp()`    | e^z = e^re(cos im + i·sin im)               |
| `ln()`     | Principal: ln\|z\| + i·arg(z)               |
| `log10()`  | log₁₀(z) = ln(z) / ln(10)                   |
| `log2()`   | log₂(z) = ln(z) / ln(2)                     |
| `pow(p)`   | e^(p·ln z), principal branch                |
| `powf(p)`  | Real exponent shorthand                      |
| `recip()`  | 1/z = z̄/|z|²                               |
| `sin/cos/tan` | Complex trigonometric functions           |
| `sinh/cosh/tanh` | Complex hyperbolic functions           |
| `asin/acos/atan` | Inverse trigonometric functions        |
| `asinh/acosh/atanh` | Inverse hyperbolic functions        |

### Arithmetic

Implements `Add`, `Sub`, `Mul`, `Div`, `Neg`:

```rust
let z1 = Complex::new(1.0, 2.0);
let z2 = Complex::new(3.0, 4.0);
let sum  = z1 + z2;   // (4, 6)
let diff = z1 - z2;   // (-2, -2)
let prod = z1 * z2;   // (-5, 10)
let quot = z1 / z2;   // (0.44, 0.08)
let neg  = -z1;        // (-1, -2)
```

---

## `analysis` — Complex Analysis

```
                 ┌─────────────────────┐
                 │   Complex Plane     │
                 │      Im             │
                 │       |             │
                 │   ┌───●───┐         │
                 │   │ Contour│         │
                 │   │  ∮f(z)dz│        │
                 │   └───────┘         │
                 │       |      Re     │
                 └─────────────────────┘
```

### Residues

```rust
use mathverse_complex::{Complex, analysis::ComplexAnalysis};

// f(z) = 1/(z-1), residue at z=1 is 1
let f = |z: Complex| Complex::one() / (z - Complex::real(1.0));
let residue = ComplexAnalysis::residue_simple_pole(&f, Complex::real(1.0), 0.001);
// residue ≈ (1, 0)
```

| Method                     | Description                                        |
|----------------------------|----------------------------------------------------|
| `residue_simple_pole(f,z,h)` | lim_{z→z₀} (z-z₀)·f(z) using finite h          |
| `residue_pole_order_n(f,z,n,h)` | Residue at pole of order n                    |
| `derivative(f,z,h)`        | Numerical first derivative via central differences |
| `nth_derivative(f,z,n,h)`  | Numerical nth derivative                           |

### Contour Integration

```rust
// ∮ z dz around unit circle = 0
let f = |z: Complex| z;
let result = ComplexAnalysis::contour_integral_circle(&f, Complex::zero(), 1.0, 1000);
```

| Method                              | Description                           |
|--------------------------------------|---------------------------------------|
| `contour_integral_circle(f,z₀,r,n)` | Trapezoidal rule on circle of radius r |
| `cauchy_integral_formula(f,z₀,r,n)` | f(z₀) = 1/(2πi) ∮ f(z)/(z-z₀) dz  |
| `cauchy_derivative_formula(f,z₀,n,r,contour_n)` | f⁽ⁿ⁾(z₀) via contour integral |

### Analyticity

```rust
let f = |z: Complex| z * z;
assert!(ComplexAnalysis::is_analytic(&f, Complex::new(1.0, 1.0), 1e-6));
```

Checks Cauchy-Riemann equations numerically: ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x.

### Laurent Series

```rust
let (positive, negative) = ComplexAnalysis::laurent_series_coefficients(
    &f, z0, max_positive, max_negative, radius, n_points,
);
```

Returns `(a₀, a₁, ..., aₙ)` for positive powers and `(a₋₁, a₋₂, ..., a₋ₘ)` for negative.

### Conformal Mappings

| Method                      | Formula                       |
|-----------------------------|-------------------------------|
| `mobius_transform(z,a,b,c,d)` | w = (az+b)/(cz+d)          |
| `mobius_inverse(w,a,b,c,d)` | Inverse Möbius transform      |
| `schwarz_christoffel(z,verts,angles,f)` | Maps upper half-plane to polygon |

### Theorems

| Method                 | Description                                      |
|------------------------|--------------------------------------------------|
| `argument_principle(f,z₀,r,n)` | (1/2πi)∮ f'/f dz = Z - P (zeros - poles) |
| `rouches_theorem(f,g,z₀,r,n)` | Tests |f| > |g| on contour C             |

---

## `special_functions` — Complex Special Functions

```
  Γ(z)    ζ(s)    Li_s(z)    erf(z)    J_v(z)    Ai(z)
   │       │        │          │         │         │
   ▼       ▼        ▼          ▼         ▼         ▼
┌──────────────────────────────────────────────────────┐
│            Complex Special Functions                  │
│  Lanczos Γ  │  Dirichlet ζ  │  Series expansions    │
│  Bessel J,Y │  Airy Ai,Bi   │  Fresnel C,S          │
└──────────────────────────────────────────────────────┘
```

### Gamma Function

Lanczos approximation with reflection formula for Re(z) < 0.5.

```rust
use mathverse_complex::special_functions::ComplexSpecialFunctions;

let g = ComplexSpecialFunctions::gamma(Complex::real(5.0));
// Γ(5) = 24
```

| Method                        | Description                                  |
|-------------------------------|----------------------------------------------|
| `gamma(z)`                    | Γ(z) via Lanczos approximation              |
| `digamma(z)`                  | ψ(z) = d/dz ln Γ(z)                         |
| `zeta(z, iterations)`         | Riemann ζ(s), Dirichlet series for Re(s) > 1 |
| `polylog(s, z, iterations)`   | Li_s(z) = Σ zⁿ/nˢ                          |

### Error Functions

```rust
let e = ComplexSpecialFunctions::erf(Complex::real(1.0), 50);
// erf(1) ≈ 0.8427
```

| Method                     | Description                    |
|----------------------------|--------------------------------|
| `erf(z, iterations)`       | Complex error function         |
| `erfc(z, iterations)`      | Complementary error function   |
| `exponential_integral(z, n)` | Ei(z)                        |

### Bessel Functions

```rust
let j0 = ComplexSpecialFunctions::bessel_j(Complex::zero(), Complex::real(1.0), 50);
// J₀(1) ≈ 0.7652
```

| Method                   | Description                          |
|--------------------------|--------------------------------------|
| `bessel_j(v, z, n)`      | J_v(z) — first kind                 |
| `bessel_y(v, z, n)`      | Y_v(z) — second kind                |

### Airy and Fresnel

| Method                   | Description                          |
|--------------------------|--------------------------------------|
| `airy(z, n)`             | Returns (Ai(z), Bi(z))              |
| `fresnel(z, n)`          | Returns (C(z), S(z))                |

---

## `matrix` — Complex Matrix Algebra

```
┌──────────────────────────────────┐
│  ComplexMatrix (m × n)           │
│  data: Vec<Complex>  (row-major) │
│                                  │
│  ┌─────┬─────┬─────┐            │
│  │ z₁₁ │ z₁₂ │ z₁₃ │  ← row 0 │
│  ├─────┼─────┼─────┤            │
│  │ z₂₁ │ z₂₂ │ z₂₃ │  ← row 1 │
│  └─────┴─────┴─────┘            │
└──────────────────────────────────┘
```

### Construction

```rust
use mathverse_complex::matrix::ComplexMatrix;

let mut m = ComplexMatrix::new(2, 2);
m.set(0, 0, Complex::real(1.0));
m.set(0, 1, Complex::new(0.0, 1.0));
m.set(1, 0, Complex::new(0.0, -1.0));
m.set(1, 1, Complex::real(1.0));

let i = ComplexMatrix::identity(3);
let z = ComplexMatrix::zeros(4, 4);
```

### Operations

| Method             | Description                      |
|--------------------|----------------------------------|
| `add(&other)`      | Element-wise addition            |
| `sub(&other)`      | Element-wise subtraction         |
| `mul(&other)`      | Matrix multiplication            |
| `scale(scalar)`    | Scalar multiplication            |
| `transpose()`      | Matrix transpose Aᵀ             |
| `hermitian()`      | Conjugate transpose Aᴴ          |
| `trace()`          | Sum of diagonal elements         |
| `frobenius_norm()` | √(Σ|aᵢⱼ|²)                     |

### Decompositions

```rust
let (l, u, pivot) = m.lu_decomposition().unwrap();
// L·U = P·A (with partial pivoting)
```

| Method                | Returns                            |
|-----------------------|------------------------------------|
| `lu_decomposition()`  | `(L, U, pivot)` or `None` if singular |
| `determinant()`       | det(A) via LU for n > 3            |
| `inverse()`           | A⁻¹ or `None` if singular         |
| `solve(b)`            | Solution to Ax = b                 |

### Properties

```rust
assert!(m.is_hermitian(1e-10));   // A = Aᴴ
assert!(m.is_unitary(1e-10));     // A·Aᴴ = I
```

### Matrix Functions

| Method             | Description                          |
|--------------------|--------------------------------------|
| `power(n)`         | Aⁿ via exponentiation by squaring   |
| `exp(iterations)`  | eᴬ via Taylor series                |
| `ln(iterations)`   | ln(A) via Taylor series              |

### Example: Solving a Linear System

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

---

## Future Scope

- Eigenvalue decomposition (QR algorithm)
- SVD (singular value decomposition)
- Matrix exponential via Padé approximation
- Fast Fourier Transform on complex arrays
- Quaternion support
- SIMD-accelerated operations
- Serde serialization for `Complex` and `ComplexMatrix`

## License

MIT OR Apache-2.0
