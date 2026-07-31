# mathverse-number-theory

A comprehensive Rust library for number theory computations including prime detection, factorization, modular arithmetic, Euler's totient, quadratic residues, Diophantine equations, continued fractions, and advanced arithmetic functions.

## Features

- **Prime number algorithms**: Sieve of Eratosthenes, primality testing, twin primes, Goldbach conjecture
- **Factorization**: Prime factorization, divisors, divisor functions, Möbius and Liouville functions
- **Modular arithmetic**: Modular exponentiation, inverses, CRT, basic operations
- **Euler's totient**: Totient function, Carmichael function, primitive roots, multiplicative order
- **Quadratic residues**: Legendre/Jacobi symbols, Tonelli-Shanks, quadratic residue enumeration
- **Diophantine equations**: Extended GCD, linear Diophantine solver, Pell equation solver
- **Continued fractions**: Conversion, convergents, special constants (π, e, φ)
- **Advanced functions**: Prime factorization trees, radicals, highly composite numbers

## Module Overview

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `primes` | Prime detection, sieve, nth prime, twin primes, Goldbach | `is_prime`, `sieve`, `nth_prime`, `twin_primes`, `goldbach` |
| `factorization` | Prime factors, divisors, divisor functions, arithmetic functions | `prime_factors`, `divisors`, `divisor_count`, `sigma_k`, `mobius` |
| `modular` | Modular exponentiation, inverses, CRT, basic operations | `mod_pow`, `mod_inverse`, `crt`, `mod_add/sub/mul/div` |
| `totient` | Euler's totient, Carmichael, coprimality, primitive roots | `euler_totient`, `carmichael`, `primitive_root`, `multiplicative_order` |
| `quadratic_residue` | Legendre/Jacobi symbols, Tonelli-Shanks, residue enumeration | `legendre`, `jacobi`, `tonelli_shanks`, `quadratic_residues` |
| `diophantine` | Extended GCD, linear Diophantine, Pell equation, square-free | `extended_gcd`, `solve_linear_diophantine`, `pell_fundamental` |
| `continued_fraction` | CF conversion, convergents, special constants, approximation | `continued_fraction`, `convergents`, `golden_ratio_cf`, `pi_cf` |
| `advanced` | Prime factorization, radical, highly composite, perfect powers | `prime_factorization`, `radical`, `highly_composite`, `perfect_power` |

## ASCII Diagram: Sieve of Eratosthenes

```
Finding primes up to 30:
Initial: [2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
  p=2:   [2, 3, -, 5, -, 7, -, 9, -,11, -,13, -,15, -,17, -,19, -,21, -,23, -,25, -,27, -,29, -]
  p=3:   [2, 3, -, 5, -, 7, -, -, -,11, -,13, -, -, -,17, -,19, -, -, -,23, -,25, -, -, -,29, -]
  p=5:   [2, 3, -, 5, -, 7, -, -, -,11, -,13, -, -, -,17, -,19, -, -, -,23, -, -, -, -, -,29, -]

Result: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mathverse-number-theory = { path = "../mathverse-number-theory" }
```

Or from the workspace root:

```toml
[dependencies]
mathverse-number-theory = { path = "crates/mathverse-number-theory" }
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

## Per-Module Documentation

### primes Module

Core prime number algorithms.

**Functions:**

- `is_prime(n: u64) -> bool` — Test primality using trial division
- `sieve(limit: usize) -> Vec<u64>` — Sieve of Eratosthenes
- `nth_prime(n: usize) -> u64` — Find the nth prime
- `prime_between(a: u64, b: u64) -> Vec<u64>` — Primes in range
- `twin_primes(limit: u64) -> Vec<(u64, u64)>` — Twin prime pairs
- `goldbach(n: u64) -> Option<(u64, u64)>` — Goldbach decomposition
- `mersenne_prime(p: u64) -> Option<u64>` — Mersenne prime test
- `prime_gap(n: u64) -> u64` — Gap to next prime

**Example: Goldbach's Conjecture**
```rust
let (p, q) = goldbach(100).unwrap();
println!("100 = {} + {}", p, q); // 100 = 3 + 97
```

**Example: Twin Primes**
```rust
let twins = twin_primes(50);
println!("Twin primes under 50: {:?}", twins);
// [(3, 5), (5, 7), (11, 13), (17, 19), (29, 31), (41, 43)]
```

**Use Cases:** Cryptography, prime sieves, number theory research, algorithm analysis.

### factorization Module

Integer factorization and divisor functions.

**Formulas:**
- Divisor count: `d(n) = (a₁+1)(a₂+1)...(aₖ+1)` for `n = p₁^a₁ · p₂^a₂ · ... · pₖ^aₖ`
- Divisor sum: `σ(n) = ∏ (p^(a+1) - 1) / (p - 1)`
- Möbius: `μ(n) = (-1)^k` if square-free, `0` otherwise

**Functions:**

- `prime_factors(n: u64) -> Vec<u64>` — List prime factors with repetition
- `divisors(n: u64) -> Vec<u64>` — All divisors of n
- `divisor_count(n: u64) -> u64` — Number of divisors τ(n)
- `divisor_sum(n: u64) -> u64` — Sum of divisors σ(n)
- `sigma_k(n: u64, k: u32) -> u64` — Sum of k-th powers of divisors
- `mobius(n: u64) -> i64` — Möbius function
- `liouville(n: u64) -> i64` — Liouville function
- `is_perfect_number(n: u64) -> bool` — Perfect number test
- `is_abundant(n: u64) -> bool` — Abundant number test
- `is_deficient(n: u64) -> bool` — Deficient number test

**Example: Prime Factorization Tree**
```
        84
       /  \
      2    42
          / \
         2   21
             / \
            3   7
```

```rust
let factors = prime_factors(84);
println!("84 = {:?}", factors); // [2, 2, 3, 7]

let (base, exp) = prime_factorization(84);
println!("84 = {:?}", base.iter().zip(exp.iter()).collect::<Vec<_>>());
// [(2, 2), (3, 1), (7, 1)]
```

**Example: Perfect Numbers**
```rust
assert!(is_perfect_number(6));   // 1+2+3 = 6
assert!(is_perfect_number(28));  // 1+2+4+7+14 = 28
assert!(is_perfect_number(496));
```

**Use Cases:** RSA encryption, divisor function analysis, arithmetic function theory.

### modular Module

Modular arithmetic operations and Chinese Remainder Theorem.

**Formulas:**
- Modular exponentiation: `a^b mod m` via square-and-multiply
- Modular inverse: `a⁻¹ mod m` via extended Euclidean algorithm
- CRT: `x ≡ a₁ (mod m₁), x ≡ a₂ (mod m₂) → x ≡ a (mod lcm(m₁,m₂))`

**Functions:**

- `mod_pow(base: u64, exp: u64, m: u64) -> u64` — Fast modular exponentiation
- `mod_inverse(a: u64, m: u64) -> Option<u64>` — Modular inverse (if exists)
- `mod_add(a: u64, b: u64, m: u64) -> u64` — Modular addition
- `mod_sub(a: u64, b: u64, m: u64) -> u64` — Modular subtraction
- `mod_mul(a: u64, b: u64, m: u64) -> u64` — Modular multiplication
- `mod_div(a: u64, b: u64, m: u64) -> Option<u64>` — Modular division
- `crt(rems: &[u64], mods: &[u64]) -> Option<u64>` — Chinese Remainder Theorem

**Example: Modular Arithmetic**
```rust
// 2^10 mod 1000 = 1024 mod 1000 = 24
assert_eq!(mod_pow(2, 10, 1000), 24);

// Inverse of 3 mod 11 = 4 (since 3*4 = 12 ≡ 1 mod 11)
assert_eq!(mod_inverse(3, 11), Some(4));
```

**Example: Chinese Remainder Theorem**
```rust
// x ≡ 2 (mod 3)
// x ≡ 3 (mod 5)
// Solution: x ≡ 8 (mod 15)
assert_eq!(crt(&[2, 3], &[3, 5]), Some(8));
```

**Modular Arithmetic Visualization:**
```
  Z/7Z Addition Table:     Z/7Z Multiplication Table:
  + | 0 1 2 3 4 5 6        * | 0 1 2 3 4 5 6
  --+-----------            --+-----------
  0 | 0 1 2 3 4 5 6        0 | 0 0 0 0 0 0 0
  1 | 1 2 3 4 5 6 0        1 | 0 1 2 3 4 5 6
  2 | 2 3 4 5 6 0 1        2 | 0 2 4 6 1 3 5
  3 | 3 4 5 6 0 1 2        3 | 0 3 6 2 5 1 4
  4 | 4 5 6 0 1 2 3        4 | 0 4 1 5 2 6 3
  5 | 5 6 0 1 2 3 4        5 | 0 5 3 1 6 4 2
  6 | 6 0 1 2 3 4 5        6 | 0 6 5 4 3 2 1
```

**Use Cases:** Cryptography (RSA, Diffie-Hellman), hashing, competitive programming.

### totient Module

Euler's totient function and related arithmetic functions.

**Formulas:**
- Euler's totient: `φ(n) = n · ∏ (1 - 1/pᵢ)` for prime factors pᵢ
- Carmichael: `λ(n) = lcm(λ(p₁^a₁), ..., λ(pₖ^aₖ))` where `λ(p^a) = φ(p^a)` except `λ(2^a) = 2^(a-2)` for a ≥ 3
- Multiplicative order: `ord_n(a) = min{k : a^k ≡ 1 (mod n)}`

**Functions:**

- `euler_totient(n: u64) -> u64` — Euler's totient φ(n)
- `euler_totient_sum(limit: u64) -> Vec<u64>` — Precompute totients up to limit
- `carmichael(n: u64) -> u64` — Carmichael function λ(n)
- `is_coprime(a: u64, b: u64) -> bool` — Coprimality test
- `coprimes_up_to(n: u64) -> Vec<u64>` — Numbers coprime to n
- `primitive_root(n: u64) -> Option<u64>` — Find primitive root modulo n
- `multiplicative_order(a: u64, n: u64) -> Option<u64>` — Multiplicative order

**Example: Euler's Totient**
```rust
// φ(12) = 4 (coprimes: 1, 5, 7, 11)
assert_eq!(euler_totient(10), 4);  // {1, 3, 7, 9}
assert_eq!(euler_totient(97), 96); // 97 is prime
```

**Example: Primitive Roots**
```rust
if let Some(g) = primitive_root(7) {
    println!("Primitive root of 7: {}", g); // 3 or 5
}
```

**Use Cases:** RSA key generation, primitive root finding, cyclic group theory.

### quadratic_residue Module

Quadratic residues and modular square roots.

**Formulas:**
- Legendre symbol: `(a/p) = a^((p-1)/2) mod p`
- Quadratic reciprocity: `(p/q)(q/p) = (-1)^((p-1)(q-1)/4)`

**Functions:**

- `legendre(a: u64, p: u64) -> i64` — Legendre symbol (for prime p)
- `jacobi(a: u64, n: u64) -> i64` — Jacobi symbol (for odd n)
- `tonelli_shanks(n: u64, p: u64) -> Option<u64>` — Modular square root
- `quadratic_residues(p: u64) -> Vec<u64>` — List all quadratic residues mod p
- `is_quadratic_residue(a: u64, p: u64) -> bool` — Test if a is QR mod p

**Example: Tonelli-Shanks**
```rust
// Find x such that x² ≡ 2 (mod 7)
let r = tonelli_shanks(2, 7).unwrap();
assert_eq!((r * r) % 7, 2); // r = 3 or 4
```

**Quadratic Residues Mod 7:**
```
x  | x² mod 7
---+---------
0  | 0
1  | 1
2  | 4
3  | 2
4  | 2
5  | 4
6  | 1

QR(7) = {0, 1, 2, 4}
```

**Use Cases:** Cryptographic protocols, quadratic congruences, algebraic number theory.

### diophantine Module

Linear Diophantine equations and related functions.

**Functions:**

- `extended_gcd(a: i64, b: i64) -> (i64, i64, i64)` — Extended GCD: returns (g, x, y) where ax + by = g
- `solve_linear_diophantine(a: i64, b: i64, c: i64) -> Option<(i64, i64)>` — Solve ax + by = c
- `is_square_free(n: u64) -> bool` — Square-free number test
- `kronecker(a: u64, b: u64) -> i64` — Kronecker symbol
- `quadratic_reciprocity(p: u64, q: u64) -> i64` — Quadratic reciprocity
- `pell_fundamental(d: u64) -> Option<(u64, u64)>` — Fundamental solution to Pell's equation x² - dy² = 1

**Example: Linear Diophantine**
```rust
// Solve 6x + 10y = 14
let (x, y) = solve_linear_diophantine(6, 10, 14).unwrap();
assert_eq!(6 * x + 10 * y, 14);
```

**Example: Pell's Equation**
```rust
// x² - 2y² = 1
let (x, y) = pell_fundamental(2).unwrap();
assert_eq!(x * x - 2 * y * y, 1); // (3, 2)
```

**Use Cases:** Integer programming, continued fractions, algebraic number theory.

### continued_fraction Module

Continued fraction representation and approximation.

**Functions:**

- `continued_fraction(n: u64, d: u64) -> Vec<u64>` — Compute CF of rational n/d
- `convergents(cf: &[u64]) -> Vec<(u128, u128)>` — Compute convergents h_n/k_n
- `approximant(n: u64, d: u64, terms: usize) -> f64` — Approximate rational with limited terms
- `golden_ratio_cf(terms: usize) -> Vec<u64>` — CF for φ = [1; 1, 1, 1, ...]
- `e_cf(terms: usize) -> Vec<u64>` — CF for e = [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]
- `pi_cf(terms: usize) -> Vec<u64>` — CF for π = [3; 7, 15, 1, 292, ...]
- `cf_to_value(cf: &[u64]) -> f64` — Convert CF to decimal
- `cf_to_fraction(cf: &[u64]) -> (u128, u128)` — Convert CF to rational

**Continued Fraction Visualization:**
```
Golden Ratio: φ = [1; 1, 1, 1, 1, ...]
Convergents:  1/1, 2/1, 3/2, 5/3, 8/5, 13/8, ...
              (Fibonacci ratios converge to φ)

e = [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]
Convergents:  2/1, 3/1, 8/3, 11/4, 19/7, 87/32, ...
```

**Example: Golden Ratio Approximation**
```rust
let cf = golden_ratio_cf(10);
let value = cf_to_value(&cf);
let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
assert!((value - phi).abs() < 1e-6);
```

**Use Cases:** Diophantine approximation, irrational number analysis, rational approximation algorithms.

### advanced Module

Higher-level arithmetic functions.

**Functions:**

- `next_prime(n: u64) -> u64` — Next prime after n
- `prev_prime(n: u64) -> Option<u64>` — Previous prime before n
- `prime_factorization(n: u64) -> Vec<(u64, u32)>` — Factorization as (prime, exponent) pairs
- `sum_of_divisors(n: u64) -> u64` — Alias for divisor_sum
- `number_of_divisors(n: u64) -> u64` — Alias for divisor_count
- `radical(n: u64) -> u64` — Product of distinct prime factors
- `kiuchi(n: u64) -> u64` — Count of primes ≤ n
- `highly_composite(limit: u64) -> Vec<u64>` — Highly composite numbers ≤ limit
- `perfect_power(n: u64) -> Option<(u64, u32)>` — Express n as b^e

**Example: Prime Factorization**
```rust
let factors = prime_factorization(84);
println!("84 = {:?}", factors); // [(2, 2), (3, 1), (7, 1)]
// Meaning: 84 = 2² × 3¹ × 7¹
```

**Example: Highly Composite Numbers**
```rust
let hcn = highly_composite(100);
println!("Highly composite ≤ 100: {:?}", hcn);
// [1, 2, 4, 6, 12, 24, 36, 48, 60]
```

**Use Cases:** Number theory research, arithmetic function analysis, perfect power detection.

## Future Scope

- **Primality testing**: Miller-Rabin, AKS algorithms
- **Factorization algorithms**: Pollard's rho, quadratic sieve, elliptic curve method
- **Modular square roots**: Cipolla's algorithm, Adleman-Manders-Miller
- **Algebraic number theory**: Class number computation, ideal arithmetic
- **Elliptic curves**: Point arithmetic, group order, ECDH
- **Transcendental number theory**: Continued fractions for algebraic irrationals
- **Performance**: Parallel sieves, cache-friendly algorithms, arbitrary precision support

## License

This project is dual-licensed under **MIT** and **Apache-2.0** licenses. You may choose either license for your use.

- MIT License: See [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0: See [LICENSE-APACHE](LICENSE-APACHE)
