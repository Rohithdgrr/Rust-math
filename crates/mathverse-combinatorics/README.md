# MathVerse Combinatorics

[![Crates.io](https://img.shields.io/crates/v/mathverse-combinatorics.svg)](https://crates.io/crates/mathverse-combinatorics)
[![docs.rs](https://docs.rs/mathverse-combinatorics/badge.svg)](https://docs.rs/mathverse-combinatorics)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Combinatorial mathematics: permutations, combinations, sequences, partitions, Stirling numbers, inclusion-exclusion, and factorial variants.

---

## Features

- **Counting functions** — combinations, permutations, with/without repetition, falling/rising factorials
- **Sequences** — Fibonacci, Lucas, Catalan, Tribonacci, Tetranacci, Collatz
- **Partitions** — integer partitions, restricted partitions, Euler's pentagonal formula
- **Stirling numbers** — first and second kind, Bell numbers, derangements, Lah numbers
- **Subsets** — power set, k-subsets, Cartesian products, permutations with index
- **Inclusion-exclusion** — 2/3-set formulas, birthday paradox, coupon collector
- **Factorial variants** — double, super, hyper, primorial, subfactorial, tetration

## Module Overview

| Module | Purpose |
|--------|---------|
| `counting` | Combinations, permutations, factorials |
| `sequences` | Fibonacci, Lucas, Catalan, Tribonacci |
| `partitions` | Integer partitions, restricted partitions |
| `stirling` | Stirling numbers, Bell, derangements, Lah |
| `subsets` | Power set, k-subsets, Cartesian product |
| `inclusion_exclusion` | IE formulas, birthday, coupon collector |
| `factorials` | Factorial variants, tetration |

## Installation

```toml
[dependencies]
mathverse-combinatorics = "0.1"
```

## Quick Start

```rust
use mathverse_combinatorics::*;

fn main() {
    // Combinations: C(10, 3) = 120
    println!("C(10,3) = {}", combinations(10, 3));

    // Fibonacci numbers
    println!("F(20) = {}", fibonacci(20)); // 6765

    // Partitions of 5
    println!("p(5) = {}", partition(5)); // 7

    // Stirling numbers of the second kind
    println!("S(5,3) = {}", stirling2(5, 3)); // 25

    // Factorial
    println!("10! = {}", factorial(10)); // 3628800
}
```

---

## Per-Module Documentation

### Counting

```rust
assert_eq!(combinations(10, 3), 120);
assert_eq!(permutations(5, 3), 60);
assert_eq!(combinations_with_repetition(3, 2), 6); // ice cream: 2 flavors from 3

// Falling and rising factorials
assert_eq!(falling_factorial(5, 3), 60); // 5·4·3
assert_eq!(rising_factorial(3, 3), 60);  // 3·4·5
```

### Sequences

```rust
let fibs: Vec<u128> = (0..10).map(fibonacci).collect();
// [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

let cats: Vec<u128> = (0..6).map(catalan).collect();
// [1, 1, 2, 5, 14, 42] — valid parenthesizations
```

### Partitions

```rust
assert_eq!(partition(5), 7);          // 7 partitions of 5
assert_eq!(partition_k(5, 2), 2);     // (4,1), (3,2)
assert_eq!(partition_into_distinct(5), 3); // 5, 4+1, 3+2
```

### Stirling Numbers

```rust
assert_eq!(stirling2(5, 3), 25); // ways to partition 5 elements into 3 non-empty subsets
assert_eq!(bell(4), 15);          // total partitions of 4 elements
assert_eq!(derangements(4), 9);   // !4 = 9
```

### Subsets

```rust
let ps = power_set(3);
// [], [0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]

let perms = permutation_index(3);
// [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]
```

### Inclusion-Exclusion

```rust
assert!(birthday_probability(23, 365) > 0.5); // >50% chance of collision

let expected = coupon_collector_expected(6);
println!("Expected rolls to see all 6 faces: {:.1}", expected); // ~14.7
```

### Factorial Variants

```rust
assert_eq!(factorial(5), 120);         // 5! = 120
assert_eq!(double_factorial(5), 15);   // 5!! = 5·3·1 = 15
assert_eq!(super_factorial(3), 12);    // 1!·2!·3! = 12
assert_eq!(hyper_factorial(3), 108);   // 1¹·2²·3³ = 108
assert_eq!(primorial(10), 210);        // 2·3·5·7 = 210
```

---

## Future Scope

- Generating functions (formal power series, convolution)
- Lattice paths (Dyck paths, Motzkin numbers)
- Young tableaux and Robinson-Schensted correspondence
- Polya enumeration (counting under symmetry)
- q-analogues (q-binomial coefficients, q-factorials)

## License

MIT — see [LICENSE](LICENSE).
