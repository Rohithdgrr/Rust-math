# mathverse-combinatorics

Combinatorial mathematics including permutations, combinations, and special sequences for the MathVerse ecosystem.

## Overview

`mathverse-combinatorics` provides comprehensive combinatorial functionality including permutations, combinations, and special number sequences like Catalan and Stirling numbers. This crate is essential for discrete mathematics and probability calculations.

## Features

- **Permutations**: 
  - Calculate permutations (nPr)
  - Generate all permutations of a set
  - Permutations with repetition
- **Combinations**:
  - Calculate combinations (nCr)
  - Generate all combinations of a set
  - Combinations with repetition
- **Special Sequences**:
  - Catalan numbers
  - Stirling numbers (first and second kind)
  - Bell numbers
  - Fibonacci numbers
  - Binomial coefficients
- **Counting Principles**: 
  - Multiplication principle
  - Addition principle
  - Inclusion-exclusion principle
- **Generating Functions**: Tools for sequence analysis

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
mathverse-combinatorics = "0.1.0"
```

## Usage

```rust
use mathverse_combinatorics::prelude::*;

// Example: Calculate combinations
let n = 10;
let r = 3;
let combinations = n_choose_r(n, r); // 120

// Example: Calculate permutations
let permutations = n_permute_r(n, r); // 720

// Example: Catalan numbers
let catalan = catalan_number(5); // 42

// Example: Stirling numbers
let stirling = stirling_second_kind(5, 3); // 25
```

## Dependencies

- `mathverse-core`: Core numeric traits and utilities

## License

MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Status

This crate is currently in early development. API stability is not guaranteed.