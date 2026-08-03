# MathVerse Special

[![Crates.io](https://img.shields.io/crates/v/mathverse-special.svg)](https://crates.io/crates/mathverse-special)
[![docs.rs](https://docs.rs/mathverse-special/badge.svg)](https://docs.rs/mathverse-special)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Real-valued special functions: gamma, error functions, Bessel functions, and Riemann zeta.

---

## Features

- **Gamma family** — `gamma`, `log_gamma`, `digamma`, `beta`, incomplete gamma (`gamma_p` / `gamma_q`)
- **Error functions** — `erf`, `erfc`
- **Bessel functions** — `bessel_j0`, `bessel_j1`, `bessel_jn`, `bessel_y0`, `bessel_y1`, `bessel_i0`, `bessel_i1`, `bessel_k0`, `bessel_k1`
- **Riemann zeta** — `zeta`
- Complex-domain counterparts in `mathverse-complex::special_functions`

## Module Overview

| Module | Purpose |
|--------|---------|
| `gamma` | Gamma function, log-gamma, digamma, beta, incomplete gamma |
| `erf` | Error function erf(z), complementary error function erfc(z) |
| `bessel` | Bessel functions of the first and second kind (J, Y, I, K) |
| `zeta` | Riemann zeta function ζ(s) |

## Installation

```toml
[dependencies]
mathverse-special = "0.1"
```

## Quick Start

```rust
use mathverse_special::{gamma, erf, bessel_j0, zeta};

fn main() {
    // Γ(5) = 4! = 24
    println!("Γ(5) = {}", gamma(5.0));

    // erf(1) ≈ 0.8427
    println!("erf(1) = {}", erf(1.0));

    // J₀(1) ≈ 0.7652
    println!("J₀(1) = {}", bessel_j0(1.0));

    // ζ(2) = π²/6 ≈ 1.6449
    println!("ζ(2) = {}", zeta(2.0, 50));
}
```

---

## Future Scope

- Polygamma functions ψ⁽ⁿ⁾(z)
- Spherical Bessel functions
- Legendre polynomials and associated functions
- Chebyshev polynomials
- Hypergeometric functions

## License

MIT — see [LICENSE](LICENSE).
