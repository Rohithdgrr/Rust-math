# mathverse-combinatorics

A comprehensive Rust library for combinatorics including permutations, combinations, sequences, partitions, Stirling numbers, subsets, inclusion-exclusion, and factorial variants.

## Features

- **Counting functions**: Combinations, permutations, with/without repetition, falling/rising factorials
- **Sequences**: Fibonacci, Lucas, Catalan, Tribonacci, Tetranacci, Collatz
- **Partitions**: Integer partitions, restricted partitions, Euler's pentagonal formula
- **Stirling numbers**: First and second kind, Bell numbers, derangements, Lah numbers
- **Subsets**: Power set, k-subsets, Cartesian products, permutations with index
- **Inclusion-exclusion**: 2/3-set formulas, birthday paradox, coupon collector
- **Factorial variants**: Double, super, hyper, primorial, subfactorial, tetration

## Module Overview

| Module | Description | Key Functions |
|--------|-------------|---------------|
| `counting` | Combinations, permutations, factorials | `combinations`, `permutations`, `falling_factorial`, `rising_factorial` |
| `sequences` | Fibonacci, Lucas, Catalan, Tribonacci | `fibonacci`, `lucas`, `catalan`, `tribonacci`, `collatz_steps` |
| `partitions` | Integer partitions, restricted partitions | `partition`, `partition_k`, `partition_into_distinct`, `euler_partition_formula` |
| `stirling` | Stirling numbers, Bell, derangements, Lah | `stirling1_unsigned`, `stirling2`, `bell`, `derangements`, `Lah_number` |
| `subsets` | Power set, k-subsets, Cartesian product | `power_set`, `subsets_of_size`, `cartesian_product`, `composition` |
| `inclusion_exclusion` | IE formulas, birthday, coupon collector | `inclusion_exclusion_2`, `inclusion_exclusion_3`, `birthday_probability` |
| `factorials` | Factorial variants, tetration | `factorial`, `double_factorial`, `super_factorial`, `hyper_factorial`, `primorial` |

## ASCII Diagram: Factorial Growth

```
Factorial Growth (log scale):
n    | n!          | Digits
-----+-------------+-------
 0   | 1           | 1
 1   | 1           | 1
 2   | 2           | 1
 3   | 6           | 1
 4   | 24          | 2
 5   | 120         | 3
 6   | 720         | 3
 7   | 5,040       | 4
 8   | 40,320      | 5
 9   | 362,880     | 6
10   | 3,628,800   | 7
15   | 1,307,674,368,000 | 13
20   | 2,432,902,008,176,640,000 | 19
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
mathverse-combinatorics = { path = "../mathverse-combinatorics" }
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

## Per-Module Documentation

### counting Module

Fundamental counting functions for combinatorics.

**Formulas:**
- Combinations: `C(n,k) = n! / (k!(n-k)!)`
- Permutations: `P(n,k) = n! / (n-k)!`
- With repetition: `C(n+k-1, k)`
- Falling factorial: `(n)_k = n(n-1)...(n-k+1)`
- Rising factorial: `n^(k) = n(n+1)...(n+k-1)`

**Functions:**

- `combinations(n: u64, k: u64) -> u128` — Binomial coefficient C(n,k)
- `permutations(n: u64, k: u64) -> u128` — P(n,k) = n!/(n-k)!
- `permutations_with_repetition(n: u64, k: u64) -> u128` — n^k
- `combinations_with_repetition(n: u64, k: u64) -> u128` — C(n+k-1, k)
- `falling_factorial(n: u64, k: u64) -> u128` — (n)_k
- `rising_factorial(n: u64, k: u64) -> u128` — n^(k)
- `multichoose(n: u64, k: u64) -> u128` — Alias for combinations_with_repetition
- `arrangements(n: u64, k: u64) -> u128` — Alias for permutations

**Example: Combinations**
```rust
// How many ways to choose 3 items from 10?
assert_eq!(combinations(10, 3), 120);

// With repetition: choosing 2 flavors from 3
assert_eq!(combinations_with_repetition(3, 2), 6);
```

**Pascal's Triangle:**
```
C(n,k) values (Pascal's Triangle):
n\k | 0   1   2   3   4   5   6
----+---------------------------
  0 | 1
  1 | 1   1
  2 | 1   2   1
  3 | 1   3   3   1
  4 | 1   4   6   4   1
  5 | 1   5  10  10   5   1
  6 | 1   6  15  20  15   6   1
```

**Use Cases:** Probability, binomial distributions, algorithm analysis.

### sequences Module

Classic mathematical sequences.

**Formulas:**
- Fibonacci: `F(n) = F(n-1) + F(n-2)`, F(0)=0, F(1)=1
- Lucas: `L(n) = L(n-1) + L(n-2)`, L(0)=2, L(1)=1
- Catalan: `C(n) = C(2n,n) / (n+1)`
- Binet's formula: `F(n) = φⁿ / √5`

**Functions:**

- `fibonacci(n: u64) -> u128` — nth Fibonacci number
- `lucas(n: u64) -> u128` — nth Lucas number
- `catalan(n: u64) -> u128` — nth Catalan number
- `tribonacci(n: u64) -> u128` — nth Tribonacci number
- `tetranacci(n: u64) -> u128` — nth Tetranacci number
- `fibonacci_binet(n: u64) -> f64` — Fibonacci via Binet's formula
- `nth_fibonacci_mod(n: u64, m: u64) -> u64` — F(n) mod m
- `collatz_steps(n: u64) -> u64` — Steps to reach 1 in Collatz sequence

**Fibonacci Spiral Visualization:**
```
Fibonacci Spiral (approximate):
    ┌─────────┐
    │         │
    │    ┌────┤
    │    │    │
    │  ┌─┤  ┌─┤
    │  │ │  │ │
    │  │ └──┤ │
    │  │    │ │
    └──┴────┴─┘

Square sizes: 1, 1, 2, 3, 5, 8, 13, 21, ...
```

**Example: Fibonacci Numbers**
```rust
let fibs: Vec<u128> = (0..10).map(fibonacci).collect();
println!("F(0..10): {:?}", fibs);
// [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

**Example: Catalan Numbers**
```rust
// Number of valid parenthesizations of n pairs
let cats: Vec<u128> = (0..6).map(catalan).collect();
println!("Catalan(0..5): {:?}", cats);
// [1, 1, 2, 5, 14, 42]
```

**Use Cases:** Counting binary trees, parenthesizations, Dyck paths, grid paths.

### partitions Module

Integer partition functions.

**Formulas:**
- Partition function: `p(n)` = number of partitions of n
- Pentagonal number: `g(k) = k(3k-1)/2`
- Euler's formula: `p(n) = Σ (-1)^(k+1) · p(n - g(k))`

**Functions:**

- `partition(n: u64) -> u128` — Number of partitions of n
- `partition_k(n: u64, k: u64) -> u128` — Partitions of n into exactly k parts
- `partition_into_distinct(n: u64) -> u128` — Partitions into distinct parts
- `partition_count_even_parts(n: u64) -> u128` — Partitions with all even parts
- `partition_count_odd_parts(n: u64) -> u128` — Partitions with all odd parts
- `partitions_leq(n: u64, max_part: u64) -> u128` — Partitions with parts ≤ max_part
- `pentagonal(n: i64) -> i64` — nth pentagonal number
- `euler_partition_formula(n: u64) -> u128` — Partition via Euler's formula

**Partition Diagrams:**
```
Partitions of 5 (7 total):
5       = 5
4 + 1   = 5
3 + 2   = 5
3 + 1 + 1 = 5
2 + 2 + 1 = 5
2 + 1 + 1 + 1 = 5
1 + 1 + 1 + 1 + 1 = 5

Partitions into distinct parts (3 total):
5
4 + 1
3 + 2
```

**Example: Partitions**
```rust
// p(5) = 7
assert_eq!(partition(5), 7);

// Partitions of 5 into 2 parts: (4,1), (3,2)
assert_eq!(partition_k(5, 2), 2);

// Distinct parts: 5, 4+1, 3+2
assert_eq!(partition_into_distinct(5), 3);
```

**Use Cases:** Number theory, combinatorial analysis, partition asymptotics.

### stirling Module

Stirling numbers and related combinatorial numbers.

**Formulas:**
- Stirling 1st kind: `s(n,k) = s(n-1,k-1) - (n-1)·s(n-1,k)`
- Stirling 2nd kind: `S(n,k) = k·S(n-1,k) + S(n-1,k-1)`
- Bell number: `B(n) = Σ S(n,k)` for k=0..n
- Lah number: `L(n,k) = (n-1)! · C(n,k) · k!`

**Functions:**

- `stirling1_unsigned(n: u64, k: u64) -> u128` — Unsigned Stirling of the first kind
- `stirling2(n: u64, k: u64) -> u128` — Stirling of the second kind
- `bell(n: u64) -> u128` — nth Bell number
- `derangements(n: u64) -> u128` — Number of derangements (subfactorial)
- `Lah_number(n: u64, k: u64) -> u128` — Lah number L(n,k)
- `eulerian_number(n: u64, k: u64) -> u128` — Eulerian number ⟨n,k⟩

**Stirling Triangle (Second Kind):**
```
S(n,k) values:
n\k | 0   1   2   3   4   5
----+------------------------
  0 | 1
  1 | 0   1
  2 | 0   1   1
  3 | 0   1   3   1
  4 | 0   1   7   6   1
  5 | 0   1  15  25  10   1
```

**Example: Stirling Numbers**
```rust
// S(5,3) = 25 (ways to partition 5 elements into 3 non-empty subsets)
assert_eq!(stirling2(5, 3), 25);

// Bell(4) = 15 (total partitions of 4 elements)
assert_eq!(bell(4), 15);

// Derangements of 4 elements = 9
assert_eq!(derangements(4), 9);
```

**Use Cases:** Set partitions, permutation statistics, combinatorial identities.

### subsets Module

Subset generation and combinatorial structures.

**Functions:**

- `power_set(n: usize) -> Vec<Vec<usize>>` — All subsets of {0,...,n-1}
- `subsets_of_size(n: usize, k: usize) -> Vec<Vec<usize>>` — All k-subsets
- `cartesian_product(a: &[usize], b: &[usize]) -> Vec<(usize, usize)>` — A × B
- `permutation_index(n: usize) -> Vec<Vec<usize>>` — All permutations with index
- `composition(n: u64, k: u64) -> u128` — Compositions of n into k parts
- `stars_and_bars(n: u64, k: u64) -> u128` — Stars and bars: C(n+k-1, k-1)

**Example: Power Set**
```rust
let ps = power_set(3);
println!("Power set of {{0,1,2}}:");
for s in &ps {
    println!("  {:?}", s);
}
// [], [0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]
```

**Example: Permutations**
```rust
let perms = permutation_index(3);
println!("Permutations of {{0,1,2}}:");
for p in &perms {
    println!("  {:?}", p);
}
// [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]
```

**Use Cases:** Enumeration, brute-force algorithms, combinatorial optimization.

### inclusion_exclusion Module

Inclusion-exclusion principle and related probabilities.

**Formulas:**
- 2 sets: `|A∪B| = |A| + |B| - |A∩B|`
- 3 sets: `|A∪B∪C| = |A|+|B|+|C| - |A∩B|-|A∩C|-|B∩C| + |A∩B∩C|`
- Birthday: `P(collision) = 1 - (365/365)·(364/365)·...·((365-n+1)/365)`
- Coupon collector: `E = n·H_n` where `H_n = 1 + 1/2 + ... + 1/n`

**Functions:**

- `inclusion_exclusion_2(a, b, ab: u128) -> u128` — Union of 2 sets
- `inclusion_exclusion_3(a, b, c, ab, ac, bc, abc: u128) -> u128` — Union of 3 sets
- `union_count(set_sizes, intersections: &[u128]) -> u128` — General IE
- `derangement_count(n: usize) -> u128` — Number of derangements
- `birthday_probability(n_people, n_days: usize) -> f64` — Birthday paradox
- `coupon_collector_expected(n: usize) -> f64` — Expected coupons to collect all n
- `pigeonhole_min(n_items, n_holes: usize) -> usize` — Pigeonhole principle

**Birthday Paradox Visualization:**
```
People | P(collision)
-------+-------------
  5    |   2.7%
 10    |  11.7%
 15    |  25.3%
 20    |  41.1%
 23    |  50.7%  ← >50%!
 30    |  70.6%
 50    |  97.0%
```

**Example: Birthday Paradox**
```rust
// Probability of shared birthday in 23 people
let p = birthday_probability(23, 365);
assert!(p > 0.5 && p < 0.6);
```

**Example: Coupon Collector**
```rust
// Expected rolls to see all 6 faces of a die
let expected = coupon_collector_expected(6);
println!("Expected rolls: {:.1}", expected); // ~14.7
```

**Use Cases:** Probability puzzles, hashing analysis, expected value computations.

### factorials Module

Extended factorial functions and hyperoperations.

**Formulas:**
- Double factorial: `n!! = n·(n-2)·(n-4)·...`
- Super factorial: `sf(n) = 1!·2!·3!·...·n!`
- Hyper factorial: `hf(n) = 1¹·2²·3³·...·nⁿ`
- Primorial: `n# = product of primes ≤ n`
- Subfactorial: `!n = n!·Σ (-1)^k / k!`
- Tetration: `ⁿa = a^a^...^a` (n times)

**Functions:**

- `factorial(n: u64) -> u128` — Standard factorial n!
- `double_factorial(n: u64) -> u128` — Double factorial n!!
- `super_factorial(n: u64) -> u128` — Product of factorials
- `hyper_factorial(n: u64) -> u128` — Product i^i
- `primorial(n: u64) -> u128` — Product of primes ≤ n
- `subfactorial(n: u64) -> u128` — Derangement count
- `tetration(a: u64, n: u64) -> u128` — Power tower a↑↑n

**Factorial Growth Visualization:**
```
n! Growth Comparison:
  n  |    n!    |   n!!   |  sf(n)  |  hf(n)
-----+----------+---------+---------+---------
  1  |        1 |       1 |       1 |       1
  2  |        2 |       2 |       2 |       4
  3  |        6 |       3 |      12 |     108
  4  |       24 |       8 |     288 |   4,1472
  5  |      120 |      15 |  34,560 | 86,400,000
  6  |      720 |      48 | 29,030,400 | ~1.2×10^12
```

**Example: Factorial Variants**
```rust
assert_eq!(factorial(5), 120);         // 5! = 120
assert_eq!(double_factorial(5), 15);   // 5!! = 5·3·1 = 15
assert_eq!(double_factorial(6), 48);   // 6!! = 6·4·2 = 48
assert_eq!(super_factorial(3), 12);    // 1!·2!·3! = 12
assert_eq!(hyper_factorial(3), 108);   // 1¹·2²·3³ = 108
assert_eq!(primorial(10), 210);        // 2·3·5·7 = 210
```

**Use Cases:** Algorithm analysis, asymptotic approximations, number theory.

## Future Scope

- **Generating functions**: Formal power series, convolution
- **Lattice paths**: Dyck paths, Motzkin numbers, ballot problems
- **Young tableaux**: Hook length formula, Robinson-Schensted correspondence
- **Polya enumeration**: Counting under symmetry
- **q-analogues**: q-binomial coefficients, q-factorials
- **Combinatorial optimization**: Matching, covering, packing
- **Formal languages**: Enumeration of words, automata

## License

This project is dual-licensed under **MIT** and **Apache-2.0** licenses. You may choose either license for your use.

- MIT License: See [LICENSE-MIT](LICENSE-MIT)
- Apache License 2.0: See [LICENSE-APACHE](LICENSE-APACHE)
