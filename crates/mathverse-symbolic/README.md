# mathverse-symbolic

[![Crates.io](https://img.shields.io/crates/v/mathverse-symbolic.svg)](https://crates.io/crates/mathverse-symbolic)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)

Symbolic computation with expression trees, automatic differentiation, simplification, and LaTeX rendering for the MathVerse ecosystem.

## Features

- **Expression trees** — build symbolic expressions with `Expr::c()`, `Expr::v()`, and arithmetic/trig ops
- **Symbolic differentiation** — first, nth-order, partial derivatives, and gradients
- **Simplification** — algebraic simplification, expansion, and factoring
- **LaTeX rendering** — display, inline, equation, and align modes

## Module Overview

| Module | Items | Description |
|---|---|---|
| `expr` | 1 `Expr` enum + 14 methods | Symbolic expression tree |
| `derivative` | 4 functions | Symbolic differentiation |
| `simplify` | 3 functions | Algebraic simplification |
| `latex` | 5 functions | LaTeX rendering |

## Installation

```toml
[dependencies]
mathverse-symbolic = { path = "../mathverse-symbolic" }
```

## Quick Start

```rust
use mathverse_symbolic::*;
use std::collections::HashMap;

fn main() {
    // Build: x^2 + 2x + 1
    let x = Expr::v("x");
    let expr = x.clone().pow(Expr::c(2.0)) + Expr::c(2.0) * x.clone() + Expr::c(1.0);
    println!("Expression: {expr}");

    // Differentiate: d/dx (x^2 + 2x + 1) = 2x + 2
    let d = derivative::differentiate(&expr, "x");
    println!("Derivative: {d}");

    // Simplify
    let s = simplify::simplify(&d);
    println!("Simplified: {s}");

    // Evaluate at x = 3
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    println!("At x=3: {}", s.evaluate(&vars).unwrap());

    // LaTeX
    println!("LaTeX: {}", latex::to_latex_display(&expr));
}
```

Expected output:

```
Expression: ((x ^ 2) + ((2 * x) + 1))
Derivative: ((2 * x) + 2)
Simplified: (2 * (x + 1))
At x=3: 8
LaTeX: \[ \left( x^{2} + \left( 2 \cdot x + 1 \right) \right) \]
```

## Per-Module Reference

### `expr` — Expression Tree

The `Expr` enum represents a symbolic expression:

```rust
enum Expr {
    Constant(f64),
    Variable(String),
    Add(Rc<Expr>, Rc<Expr>),
    Sub(Rc<Expr>, Rc<Expr>),
    Mul(Rc<Expr>, Rc<Expr>),
    Div(Rc<Expr>, Rc<Expr>),
    Pow(Rc<Expr>, Rc<Expr>),
    Neg(Rc<Expr>),
    Ln(Rc<Expr>),
    Exp(Rc<Expr>),
    Sin(Rc<Expr>),
    Cos(Rc<Expr>),
    Tan(Rc<Expr>),
    Sqrt(Rc<Expr>),
}
```

| Method | Description |
|---|---|
| `Expr::c(value)` | Create constant |
| `Expr::v(name)` | Create variable |
| `.add(other)` | a + b |
| `.sub(other)` | a − b |
| `.mul(other)` | a × b |
| `.div(other)` | a / b |
| `.pow(other)` | a^b |
| `.neg()` | −a |
| `.ln()` | ln(a) |
| `.exp()` | e^a |
| `.sin()` | sin(a) |
| `.cos()` | cos(a) |
| `.tan()` | tan(a) |
| `.sqrt()` | √a |
| `.evaluate(&vars)` | Numerical evaluation |
| `.variables()` | List variable names |

Implements: `Display`, `Clone`, `Debug`, `PartialEq`, `Add`, `Sub`, `Mul`, `Neg`.

### `derivative` — Symbolic Differentiation

| Function | Description |
|---|---|
| `differentiate(expr, var)` | First derivative |
| `nth_derivative(expr, var, n)` | nth-order derivative |
| `partial_derivative(expr, var)` | Alias for `differentiate` |
| `gradient(expr, vars)` | Vector of partial derivatives |

Supported rules: power, product, quotient, chain, trig, exponential, logarithmic.

### `simplify` — Algebraic Simplification

| Function | Description |
|---|---|
| `simplify(expr)` | Full simplification (constant folding, identity elimination) |
| `expand(expr)` | Distribute multiplication over addition |
| `factor(expr)` | Basic factoring of common terms |

### `latex` — LaTeX Rendering

| Function | Description |
|---|---|
| `to_latex(expr)` | Core LaTeX string |
| `to_latex_display(expr)` | Wraps in `\[ ... \]` |
| `to_latex_inline(expr)` | Wraps in `$ ... $` |
| `to_latex_equation(label, expr)` | `equation` environment |
| `to_latex_align(exprs)` | `align` environment for multiple expressions |

## Dependencies

- `mathverse-core`
- `mathverse-calculus`
- `mathverse-algebra`
- `thiserror`

## Future Scope

- Integration (symbolic antiderivatives)
- Series expansion (Taylor, Laurent)
- Equation solving (symbolic roots)
- Substitution and change of variables
- Simplification with trig identities

## License

MIT OR Apache-2.0
