# mathverse-prelude

Umbrella re-export crate for the entire MathVerse workspace. Import one crate, get access to everything.

## What's Included

```toml
[dependencies]
mathverse-prelude = { path = "../mathverse-prelude" }
```

```rust
use mathverse_prelude::*;
```

This gives you access to all modules from every MathVerse crate:

| Crate | What You Get |
|-------|-------------|
| `mathverse_core` | Core error types, shared utilities |
| `mathverse_arithmetic` | Basic arithmetic operations |
| `mathverse_algebra` | Algebraic structures and operations |
| `mathverse_calculus` | Derivatives, integrals, limits |
| `mathverse_trigonometry` | Trig functions and identities |
| `mathverse_statistics` | Descriptive statistics |
| `mathverse_probability` | Distributions, random variables |
| `mathverse_linear_algebra` | Vector spaces, decompositions |
| `mathverse_matrix` | Matrix operations |
| `mathverse_vector` | Vector operations |
| `mathverse_complex` | Complex number arithmetic |
| `mathverse_discrete` | Discrete math utilities |
| `mathverse_number_theory` | Primes, modular arithmetic, GCD |
| `mathverse_combinatorics` | Combinations, permutations |
| `mathverse_graph` | Graph algorithms |
| `mathverse_transforms` | Fourier, Laplace, Z-transforms |
| `mathverse_signal` | Signal processing |
| `mathverse_optimization` | Optimization solvers |
| `mathverse_numerical` | Numerical methods |
| `mathverse_equations` | Equation solvers |
| `mathverse_ai` | Tensors, activations, losses, optimizers, attention |
| `mathverse_machine_learning` | Classical ML algorithms |
| `mathverse_vision` | Computer vision primitives |

## Quick Start

```rust
use mathverse_prelude::*;

fn main() {
    // From mathverse_ai
    let t = Tensor::zeros(&[2, 3]);
    let r = relu(&t);

    // From mathverse_machine_learning
    let x = vec![vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![2.0, 4.0, 6.0];
    let result = linear::fit(&x, &y).unwrap();

    // From mathverse_vision
    let cam = camera::Camera::new(800.0, 600.0, 320.0, 240.0);
    let (u, v) = cam.project(1.0, 2.0, 10.0);
}
```

## Why Use the Prelude?

- **Convenience**: One import instead of 20+
- **Discovery**: See everything available at a glance
- **Exploration**: Great for prototyping and learning

## Why NOT Use the Prelude?

- **Compile time**: Pulls in all crates even if you only need one
- **Namespace pollution**: May conflict with your own types
- **Production code**: Prefer explicit dependencies for clarity

For production use, depend on only the crates you need:

```toml
[dependencies]
mathverse-ai = { path = "../mathverse-ai" }
mathverse-machine-learning = { path = "../mathverse-machine-learning" }
```

## License

MIT OR Apache-2.0
