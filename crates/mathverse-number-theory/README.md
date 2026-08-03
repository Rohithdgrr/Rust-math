# MathVerse Number Theory

[![Crates.io](https://img.shields.io/crates/v/mathverse-number-theory.svg)](https://crates.io/crates/mathverse-number-theory)
[![docs.rs](https://docs.rs/mathverse-number-theory/badge.svg)](https://docs.rs/mathverse-number-theory)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Comprehensive number theory: prime detection, factorization, modular arithmetic, Euler's totient, quadratic residues, Diophantine equations, and continued fractions.

---

## Features

- **Prime algorithms** — Sieve of Eratosthenes, primality testing, twin primes, Goldbach conjecture
- **Factorization** — prime factorization, divisors, divisor functions, Möbius and Liouville functions
- **Modular arithmetic** — modular exponentiation, inverses, CRT, basic operations
- **Euler's totient** — totient function, Carmichael function, primitive roots, multiplicative order
- **Quadratic residues** — Legendre/Jacobi symbols, Tonelli-Shanks, residue enumeration
- **Diophantine equations** — extended GCD, linear Diophantine solver, Pell equation solver
- **Continued fractions** — conversion, convergents, special constants (π, e, φ)
- **Advanced functions** — prime factorization trees, radicals, highly composite numbers

## Module Overview

| Module | Purpose |
|--------|---------|
| `primes` | Prime detection, sieve, nth prime, twin primes, Goldbach |
| `factorization` | Prime factors, divisors, divisor functions, arithmetic functions |
| `modular` | Modular exponentiation, inverses, CRT, basic operations |
| `totient` | Euler's totient, Carmichael, coprimality, primitive roots |
| `quadratic_residue` | Legendre/Jacobi symbols, Tonelli-Shanks, residue enumeration |
| `diophantine` | Extended GCD, linear Diophantine, Pell equation, square-free |
| `continued_fraction` | CF conversion, convergents, special constants, approximation |
| `advanced` | Prime factorization, radical, highly composite, perfect powers |

## Installation

```toml
[dependencies]
mathverse-number-theory = "0.1"
```

## Quick Start

```rust
use mathverse_number_theory::*;

fn main() {
    // Check if a number is prime
    assert!(is_prime(97));
    assert!(!is_prime(15));

    // Find primes up to 100
    let primes = sieve(100);
    println!("Primes ≤ 100: {:?}", primes);

    // Factorize a number
    let factors = prime_factors(84);
    println!("Prime factors of 84: {:?}", factors); // [2, 2, 3, 7]

    // Modular exponentiation
    let result = mod_pow(2, 10, 1000);
    println!("2^10 mod 1000 = {}", result); // 24

    // Euler's totient
    let phi = euler_totient(12);
    println!("φ(12) = {}", phi); // 4
}
```

---

## Per-Module Documentation

### Primes

```rust
use mathverse_number_theory::*;

let (p, q) = goldbach(100).unwrap();
println!("100 = {} + {}", p, q); // 100 = 3 + 97

let twins = twin_primes(50);
println!("Twin primes under 50: {:?}", twins);
```

### Factorization

```rust
let factors = prime_factors(84);
println!("84 = {:?}", factors); // [2, 2, 3, 7]

let (base, exp) = prime_factorization(84);
// 84 = 2² × 3¹ × 7¹

assert!(is_perfect_number(6));  // 1+2+3 = 6
assert!(is_perfect_number(28)); // 1+2+4+7+14 = 28
```

### Modular Arithmetic

```rust
assert_eq!(mod_pow(2, 10, 1000), 24);
assert_eq!(mod_inverse(3, 11), Some(4)); // 3*4 ≡ 1 mod 11

// Chinese Remainder Theorem: x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x ≡ 8 (mod 15)
assert_eq!(crt(&[2, 3], &[3, 5]), Some(8));
```

### Totient & Primitive Roots

```rust
assert_eq!(euler_totient(10), 4);  // {1, 3, 7, 9}
assert_eq!(euler_totient(97), 96); // 97 is prime

if let Some(g) = primitive_root(7) {
    println!("Primitive root of 7: {}", g); // 3 or 5
}
```

### Quadratic Residues

```rust
let r = tonelli_shanks(2, 7).unwrap();
assert_eq!((r * r) % 7, 2); // x² ≡ 2 (mod 7)
```

### Diophantine Equations

```rust
let (x, y) = solve_linear_diophantine(6, 10, 14).unwrap();
assert_eq!(6 * x + 10 * y, 14);

let (x, y) = pell_fundamental(2).unwrap();
assert_eq!(x * x - 2 * y * y, 1); // (3, 2)
```

### Continued Fractions

```rust
let cf = golden_ratio_cf(10);
let value = cf_to_value(&cf);
let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
assert!((value - phi).abs() < 1e-6);
```

---

## Future Scope

- Miller-Rabin and AKS primality testing
- Pollard's rho and quadratic sieve factorization
- Cipolla's algorithm for modular square roots
- Elliptic curve arithmetic and ECDH
- Parallel sieves and arbitrary precision support

## License

MIT — see [LICENSE](LICENSE).
