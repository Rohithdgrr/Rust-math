# mathverse-discrete

> Discrete mathematics toolkit — boolean logic, set operations, combinatorics, graph theory, number theory, and recurrence relations.

```
mathverse-discrete
├── combinatorics   Permutations, combinations, partitions, special numbers
├── graph           Directed & undirected graphs, traversals, shortest paths
├── number_theory   GCD, primes, modular arithmetic, CRT, Diophantine
└── recurrence      Linear recurrences, named sequences, Z-transform
```

## Features

- **Boolean logic**: `implies`, `iff`, `xor`, `nand`, `nor`, truth table generation
- **Set operations**: Union, intersection, difference, subset check (O(n) on sorted sets)
- **Functions**: Composition, injectivity check
- **Combinatorics**: Factorial, binomial, permutations, Stirling/Bell/Catalan/Fibonacci/Lucas numbers, partitions, derangements, inclusion-exclusion, stars and bars
- **Graph theory**: Directed and undirected graphs, BFS, DFS, topological sort, shortest path, cycle detection, SCCs, MST, bipartite check
- **Number theory**: GCD/LCM, modular exponentiation/inverse, primality (deterministic + Miller-Rabin), sieve, factorization, Euler totient, Carmichael function, CRT, Legendre/Jacobi symbols, Diophantine equations
- **Recurrence relations**: Linear homogeneous/non-homogeneous, named sequences (Fibonacci, Lucas, Tribonacci, Padovan, Perrin), characteristic equation solver, convolution, Z-transform

## Module Overview

| Module | Description | Key Types |
|--------|-------------|-----------|
| `combinatorics` | Counting and enumeration | `Combinatorics` |
| `graph` | Graph data structures and algorithms | `DirectedGraph`, `UndirectedGraph` |
| `number_theory` | Arithmetic and algebraic number theory | `NumberTheory` |
| `recurrence` | Recurrence relations and sequences | `RecurrenceRelations` |

## Installation

```toml
[dependencies]
mathverse-discrete = { path = "crates/mathverse-discrete" }
```

## Quick Start

```rust
use mathverse_discrete::*;

fn main() {
    // Combinatorics
    println!("10! = {}", Combinatorics::factorial(10));           // 3628800
    println!("C(10,3) = {}", Combinatorics::binomial(10, 3));    // 120
    println!("Fib(10) = {}", Combinatorics::fibonacci(10));      // 55
    println!("Bell(4) = {}", Combinatorics::bell(4));            // 15

    // Number theory
    println!("gcd(48,18) = {}", NumberTheory::gcd(48, 18));      // 6
    println!("is_prime(97) = {}", NumberTheory::is_prime(97));   // true
    let primes = NumberTheory::sieve_of_eratosthenes(30);
    println!("primes ≤ 30: {:?}", primes);                       // [2,3,5,7,11,13,17,19,23,29]

    // Graph
    let mut g = DirectedGraph::new();
    g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(2, 0);
    println!("has_cycle: {}", g.has_cycle());                     // true
}
```

---

## Module: `combinatorics` — Counting & Enumeration

### Key Formulas

```
  Factorial:        n! = 1·2·3·...·n

  Binomial:         C(n,k) = n! / (k! · (n-k)!)

  Permutations:     P(n,k) = n! / (n-k)!

  Multinomial:      n! / (k₁!·k₂!·...·kₘ!)

  Stirling (2nd):   S(n,k) = k·S(n-1,k) + S(n-1,k-1)

  Bell:             B(n) = Σ_{k=0}^{n} S(n,k)

  Catalan:          Cₙ = C(2n,n) / (n+1)

  Derangements:     D(n) = (n-1)·(D(n-1) + D(n-2))

  Stars and Bars:   C(n+k-1, k-1)
```

### Special Number Sequences

```
  n:    0   1   2   3   4   5   6   7    8    9   10
  ──────────────────────────────────────────────────────
  n!:   1   1   2   6  24 120 720 5040 40320 ...
  C(n): 1   1   2   5  14  42 132  429 1430 4862 ...
  Fib:  0   1   1   2   3   5   8  13  21   34   55
  Luc:  2   1   3   4   7  11  18  29  47   76  123
  Bell: 1   1   2   5  15  52 203 877 4140 ...
  ──────────────────────────────────────────────────────
```

### Available Functions

| Function | Description | Complexity |
|----------|-------------|:----------:|
| `factorial(n)` | n! | O(n) |
| `binomial(n, k)` | C(n,k) | O(k) |
| `permutations(n, k)` | P(n,k) | O(k) |
| `multinomial(n, ks)` | Multinomial coefficient | O(n) |
| `stirling_first(n, k)` | Signed Stirling 1st kind | O(nk) |
| `stirling_second(n, k)` | Stirling 2nd kind | O(nk) |
| `bell(n)` | Bell number | O(n²) |
| `catalan(n)` | Catalan number | O(n) |
| `fibonacci(n)` | Fibonacci number | O(n) |
| `lucas(n)` | Lucas number | O(n) |
| `partition(n)` | Integer partitions | O(n²) |
| `derangement(n)` | Derangements | O(n) |
| `generate_permutations(n)` | All n! permutations | O(n!) |
| `generate_combinations(n, k)` | All C(n,k) combos | O(C(n,k)) |
| `stars_and_bars(n, k)` | Distribute n into k | O(k) |
| `harmonic(n)` | Harmonic number H_n | O(n) |

### Usage

```rust
use mathverse_discrete::Combinatorics;

// Basic counting
assert_eq!(Combinatorics::factorial(5), 120);
assert_eq!(Combinatorics::binomial(10, 3), 120);
assert_eq!(Combinatorics::permutations(5, 3), 60);

// Special numbers
assert_eq!(Combinatorics::catalan(5), 42);
assert_eq!(Combinatorics::fibonacci(10), 55);
assert_eq!(Combinatorics::bell(4), 15);

// Partitions
assert_eq!(Combinatorics::partition(5), 7);  // 5=5,4+1,3+2,3+1+1,2+2+1,2+1+1+1,1+1+1+1+1
assert_eq!(Combinatorics::derangement(4), 9);

// Generate all C(4,2) = 6 combinations
let combos = Combinatorics::generate_combinations(4, 2);
assert_eq!(combos.len(), 6);
```

### Use Cases

- Probability calculations
- Algorithm analysis (counting complexity)
- Cryptographic protocol design
- Chemical isomer counting

---

## Module: `graph` — Graph Theory

### Graph Structures

```
  Directed Graph              Undirected Graph
  ┌───┐     ┌───┐            ┌───┐     ┌───┐
  │ 0 │────▶│ 1 │            │ 0 │─────│ 1 │
  └───┘     └─┬─┘            └───┘     └───┘
    │         │                │  ╲    ╱  │
    │         ▼                │   ╲  ╱   │
    │     ┌───┐                │    ╲╱    │
    └────▶│ 2 │                │    ╱╲    │
          └───┘                │   ╱  ╲   │
                               │  ╱    ╲  │
  Adjacency list:              └───┘     └───┘
  0: [1, 2]
  1: [2]                     Adjacency list:
  2: []                      0: [1, 2]
                             1: [0]
                             2: [0]
```

### Available Types

**DirectedGraph**:
| Method | Description |
|--------|-------------|
| `add_vertex(v)` | Add vertex |
| `add_edge(u, v)` | Add directed edge u→v |
| `neighbors(v)` | Get outgoing neighbors |
| `has_edge(u, v)` | Check edge existence |
| `in_degree(v)` / `out_degree(v)` | Degree queries |
| `bfs(start)` | Breadth-first search |
| `dfs(start)` | Depth-first search |
| `topological_sort()` | Topological ordering (DAG) |
| `shortest_path_bfs(start, end)` | Shortest path (unweighted) |
| `has_cycle()` | Cycle detection |
| `strongly_connected_components()` | Kosaraju's SCC |
| `is_connected()` | Connectivity check |

**UndirectedGraph**:
| Method | Description |
|--------|-------------|
| `add_vertex(v)` / `add_edge(u, v)` | Build graph |
| `degree(v)` | Vertex degree |
| `is_connected()` | Connectivity check |
| `has_cycle()` | Cycle detection |
| `mst_prim()` | Minimum spanning tree |
| `is_bipartite()` | Bipartiteness check |

### Traversal Order

```
  Graph:       0 → 1 → 3
               0 → 2 → 3

  BFS from 0:  [0, 1, 2, 3]    (level order)
  DFS from 0:  [0, 1, 3, 2]    (depth first)
```

### Usage

```rust
use mathverse_discrete::DirectedGraph;

// Build a DAG
let mut g = DirectedGraph::new();
g.add_edge(0, 1);
g.add_edge(0, 2);
g.add_edge(1, 3);
g.add_edge(2, 3);

// Traversals
assert_eq!(g.bfs(0), vec![0, 1, 2, 3]);

// Topological sort (valid ordering for DAG)
let order = g.topological_sort().unwrap();
// 0 appears before 1 and 2; 1, 2 before 3

// Shortest path
let path = g.shortest_path_bfs(0, 3).unwrap();
// path = [0, 1, 3] or [0, 2, 3]

// Cycle detection
assert!(!g.has_cycle());
g.add_edge(3, 0);
assert!(g.has_cycle());

// Undirected: bipartite check
let mut ug = mathverse_discrete::UndirectedGraph::new();
ug.add_edge(0, 1); ug.add_edge(1, 2); ug.add_edge(2, 3);
assert!(ug.is_bipartite());  // path graph is bipartite
ug.add_edge(0, 2);
assert!(!ug.is_bipartite()); // triangle is not bipartite
```

### Use Cases

- Dependency resolution (topological sort)
- Social network analysis (connectivity, SCC)
- Routing algorithms (shortest path)
- Compiler design (DAG for expression evaluation)

---

## Module: `number_theory` — Number Theory

### Key Algorithms

**Euclidean Algorithm** (GCD):
```
  gcd(a, b) = gcd(b, a mod b)
  gcd(a, 0) = a
```

**Extended Euclidean** (Bezout's identity):
```
  ax + by = gcd(a, b)
```

**Modular Exponentiation** (square-and-multiply):
```
  a^b mod m:
  result = 1
  while b > 0:
      if b odd: result = result * a mod m
      a = a * a mod m
      b = b / 2
```

**Miller-Rabin** (probabilistic primality):
```
  Write n-1 = 2^r · d
  For random witness a:
      x = a^d mod n
      if x = 1 or x = n-1: probably prime
      repeat r-1 times:
          x = x² mod n
          if x = n-1: probably prime
      composite
```

### Available Functions

| Function | Description | Complexity |
|----------|-------------|:----------:|
| `gcd(a, b)` | Greatest common divisor | O(log(min(a,b))) |
| `extended_gcd(a, b)` | Bezout coefficients | O(log(min(a,b))) |
| `lcm(a, b)` | Least common multiple | O(log(min(a,b))) |
| `mod_pow(base, exp, m)` | Modular exponentiation | O(log(exp)) |
| `mod_inverse(a, m)` | Modular inverse | O(log(m)) |
| `is_prime(n)` | Deterministic primality | O(√n) |
| `miller_rabin(n, k)` | Probabilistic primality | O(k·log²(n)) |
| `sieve_of_eratosthenes(n)` | All primes ≤ n | O(n·log(log(n))) |
| `prime_factorization(n)` | Prime factors | O(√n) |
| `euler_totient(n)` | φ(n) | O(√n) |
| `carmichael(n)` | λ(n) | O(√n) |
| `chinese_remainder(a, m)` | CRT solver | O(k²·log(max(m))) |
| `legendre_symbol(a, p)` | Quadratic residue | O(log(p)) |
| `jacobi_symbol(a, n)` | Generalized residue | O(log(n)) |
| `diophantine(a, b, c)` | Linear Diophantine | O(log(min(a,b))) |
| `divisor_count(n)` | Number of divisors | O(√n) |
| `divisor_sum(n)` | Sum of divisors | O(√n) |
| `mobius(n)` | Möbius function | O(√n) |

### Sieve Visualization

```
  Sieve of Eratosthenes (n=30):

  Start:  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  p=2:    2  3  ×  5  ×  7  ×  9  × 11  × 13  × 15  × 17  × 19  × 21  × 23  × 25  × 27  × 29  ×
  p=3:    2  3     5     7     ×    11    13     ×    17    19     ×    23    25     ×    29
  p=5:    2  3     5     7          11    13          17    19          23    ×          29
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  Result: 2  3     5     7          11    13          17    19          23              29
```

### Usage

```rust
use mathverse_discrete::NumberTheory;

// GCD and LCM
assert_eq!(NumberTheory::gcd(48, 18), 6);
assert_eq!(NumberTheory::lcm(12, 18), 36);

// Extended GCD: 48x + 18y = 6
let (g, x, y) = NumberTheory::extended_gcd(48, 18);
assert_eq!(48 * x + 18 * y, g);

// Modular arithmetic
assert_eq!(NumberTheory::mod_pow(2, 10, 1000), 24);  // 2^10 = 1024 ≡ 24 mod 1000
assert_eq!(NumberTheory::mod_inverse(3, 7), Some(5));  // 3·5 = 15 ≡ 1 mod 7

// Primes
assert!(NumberTheory::is_prime(97));
let primes = NumberTheory::sieve_of_eratosthenes(20);
assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);

// Factorization
let factors = NumberTheory::prime_factorization(360);
// [(2,3), (3,2), (5,1)] → 2³·3²·5 = 360

// Euler totient
assert_eq!(NumberTheory::euler_totient(12), 4);  // {1,5,7,11}

// Chinese Remainder Theorem: x ≡ 2 mod 3, x ≡ 3 mod 5
let x = NumberTheory::chinese_remainder(&[2, 3], &[3, 5]).unwrap();
assert_eq!(x, 8);  // 8 ≡ 2 mod 3, 8 ≡ 3 mod 5

// Diophantine: 6x + 15y = 9
let (g, x, y) = NumberTheory::diophantine(6, 15, 9).unwrap();
assert_eq!(6 * x + 15 * y, 9);
```

### Use Cases

- Cryptography (RSA, Diffie-Hellman)
- Hash function design
- Pseudorandom number generation
- Error correcting codes

---

## Module: `recurrence` — Recurrence Relations

### Named Sequence Visualization

```
  Fibonacci:  0  1  1  2  3  5  8  13  21  34  55 ...
              └──┘
                Fₙ = Fₙ₋₁ + Fₙ₋₂

  Lucas:      2  1  3  4  7  11 18  29  47  76 123 ...
              └──┘
                Lₙ = Lₙ₋₁ + Lₙ₋₂,  L₀=2, L₁=1

  Tribonacci: 0  1  1  2  4  7  13 24  44  81 149 ...
              └──┘
                Tₙ = Tₙ₋₁ + Tₙ₋₂ + Tₙ₋₃

  Padovan:    1  1  1  2  2  3  4  5  7  9  12 ...
              └──┘
                Pₙ = Pₙ₋₂ + Pₙ₋₃

  Perrin:     3  0  2  3  2  5  5  7  10 12 17 ...
              └──┘
                Pₙ = Pₙ₋₂ + Pₙ₋₃,  P₀=3, P₁=0, P₂=2
```

### Characteristic Equation

For `aₙ = c₁·aₙ₋₁ + c₂·aₙ₋₂`:
```
  Characteristic equation: r² - c₁r - c₂ = 0

  Roots: r = (c₁ ± √(c₁² + 4c₂)) / 2

  Distinct real roots:   aₙ = α·r₁ⁿ + β·r₂ⁿ
  Repeated root:         aₙ = (α + βn)·rⁿ
  Complex roots:         aₙ = ρⁿ(α·cos(nθ) + β·sin(nθ))
```

### Available Functions

| Function | Description |
|----------|-------------|
| `linear_homogeneous(coeffs, initial, n)` | Solve aₙ = c₁aₙ₋₁ + ... + cₖaₙ₋ₖ |
| `linear_nonhomogeneous(coeffs, f, initial, n)` | Solve with driving function f(n) |
| `fibonacci(n)` / `lucas(n)` / `tribonacci(n)` | Named sequences |
| `padovan(n)` / `perrin(n)` | Named sequences |
| `arithmetic(a0, d, n)` | aₙ = a₀ + n·d |
| `geometric(a0, r, n)` | aₙ = a₀ · rⁿ |
| `characteristic_solution(coeffs)` | Find roots of characteristic eq |
| `closed_form(r1, r2, c1, c2, n)` | Generate from closed form |
| `convolution(a, b)` | Sequence convolution |
| `satisfies_recurrence(seq, coeffs)` | Verify a sequence |
| `find_recurrence(seq, order)` | Discover coefficients from data |
| `z_transform(seq)` / `inverse_z_transform(coeffs, n)` | Z-transform pair |

### Usage

```rust
use mathverse_discrete::RecurrenceRelations;

// Fibonacci via recurrence
let fib = RecurrenceRelations::linear_homogeneous(&[1.0, 1.0], &[0.0, 1.0], 10);
// [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

assert_eq!(RecurrenceRelations::fibonacci(10), 55);
assert_eq!(RecurrenceRelations::lucas(5), 11);
assert_eq!(RecurrenceRelations::tribonacci(5), 7);

// Arithmetic: 1, 3, 5, 7, 9
let arith = RecurrenceRelations::arithmetic(1.0, 2.0, 5);
assert_eq!(arith, vec![1.0, 3.0, 5.0, 7.0, 9.0]);

// Geometric: 1, 2, 4, 8, 16
let geo = RecurrenceRelations::geometric(1.0, 2.0, 5);
assert_eq!(geo, vec![1.0, 2.0, 4.0, 8.0, 16.0]);

// Characteristic equation for Fibonacci
let roots = RecurrenceRelations::characteristic_solution(&[1.0, 1.0]);
// roots ≈ [(1+√5)/2, (1-√5)/2] (golden ratio and conjugate)

// Verify a sequence satisfies a recurrence
let seq = vec![0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0];
assert!(RecurrenceRelations::satisfies_recurrence(&seq, &[1.0, 1.0]));

// Discover recurrence from data
let seq = vec![0.0, 1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0];
let coeffs = RecurrenceRelations::find_recurrence(&seq, 2).unwrap();
// coeffs ≈ [1.0, 1.0] → Fibonacci

// Convolution: [1,2,3] * [1,1] = [1,3,5,3]
let conv = RecurrenceRelations::convolution(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
assert_eq!(conv, vec![1.0, 3.0, 5.0, 3.0]);
```

### Use Cases

- Algorithm analysis (recurrence for recursive algorithms)
- Financial modeling (compound interest, annuities)
- Population growth modeling
- Signal processing (linear filters)

---

## Future Scope

- [ ] Petri net simulation
- [ ] Lattice operations and order theory
- [ ] Formal language recognition (FSM, regex)
- [ ] SAT solver integration
- [ ] Hypergraph algorithms
- [ ] Algebraic structures (groups, rings, fields)
- [ ] Category theory primitives
- [ ] Automated theorem proving hooks

## License

MIT OR Apache-2.0
