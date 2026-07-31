# mathverse-complex

Complex number arithmetic and operations for the MathVerse ecosystem.

## Overview

`mathverse-complex` provides comprehensive complex number functionality including arithmetic operations, powers, roots, and transcendental functions. This crate enables mathematical operations on complex numbers which are essential in many scientific and engineering applications.

## Features

- **Complex Number Arithmetic**: 
  - Addition, subtraction, multiplication, division
  - Conjugate operations
  - Absolute value and magnitude
- **Polar Representation**:
  - Conversion between rectangular and polar forms
  - Argument (phase angle) computation
- **Powers and Roots**:
  - Complex exponentiation
  - Nth roots of complex numbers
  - De Moivre's theorem applications
- **Transcendental Functions**:
  - Complex exponential (e^z)
  - Complex logarithm
  - Complex trigonometric functions (sin, cos, tan)
  - Complex hyperbolic functions (sinh, cosh, tanh)
- **Complex Analysis**: 
  - Analytic function support
  - Series expansions

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
mathverse-complex = "0.1.0"
```

## Usage

```rust
use mathverse_complex::prelude::*;

// Example: Creating complex numbers
let z1 = Complex::new(3.0, 4.0); // 3 + 4i
let z2 = Complex::new(1.0, 2.0); // 1 + 2i

// Example: Arithmetic operations
let sum = z1 + z2;
let product = z1 * z2;
let conjugate = z1.conjugate(); // 3 - 4i

// Example: Polar form
let magnitude = z1.magnitude(); // 5.0
let argument = z1.arg(); // arctan(4/3)

// Example: Transcendental functions
let exponential = z1.exp();
let sine = z1.sin();
```

## Dependencies

This crate has no external dependencies and works standalone.

## License

MIT OR Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Status

This crate is currently in early development. API stability is not guaranteed.