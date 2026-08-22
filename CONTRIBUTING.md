# Contributing to MathVerse

Thank you for your interest in contributing to MathVerse. This guide covers everything you need to get started.

---

## Prerequisites

- **Rust 1.87+** (workspace MSRV, stable channel)
- **Git**
- A GitHub account

## Setup

```bash
git clone https://github.com/Rohithdgrr/Rust-math.git
cd Rust-math

# Verify the workspace builds
cargo check --workspace

# Run the full test suite
cargo test --workspace

# Run lints
cargo clippy --workspace --all-targets -- -D warnings

# Build documentation
cargo doc --workspace --no-deps --open
```

---

## Repository Structure

```
Rust-math/
├── crates/                    # 32 independent crates
│   ├── mathverse-core/        # Numeric traits, errors, foundation
│   ├── mathverse-algebra/     # Polynomials, equation solving
│   ├── mathverse-ai/          # Neural nets, autograd, attention
│   ├── mathverse-machine-learning/  # Classical ML algorithms
│   └── ...
├── docs/                      # Architecture, features, guidelines
├── scripts/                   # Validation and helper scripts
├── CONTRIBUTING.md            # This file
└── README.md                  # Project overview
```

### Key Crates

| Crate | Purpose | Depends On |
|-------|---------|------------|
| `mathverse-core` | Foundation — all crates depend on this | — |
| `mathverse-prelude` | One-stop import for everything | All others |
| `mathverse-numerical` | Canonical root-finding, ODE, interpolation | core |
| `mathverse-optimization` | Canonical optimization algorithms | core, probability |
| `mathverse-transforms` | Canonical FFT, DCT, wavelets | complex |

---

## How to Contribute

### 1. Find an Issue

Check the [issue tracker](https://github.com/Rohithdgrr/Rust-math/issues) for:

- `good first issue` — ideal for newcomers
- `help wanted` — we would appreciate assistance
- `documentation` — always welcome

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-123
```

### 3. Make Changes

Follow the coding standards outlined below.

### 4. Write Tests

Every new function needs:

- At least one unit test covering happy path and edge cases
- Doc tests in `///` comments showing usage
- Golden tests for numerical algorithms (comparing against known-good outputs)

```rust
/// Compute the factorial of n.
///
/// # Examples
/// ```
/// use mathverse_combinatorics::factorial;
/// assert_eq!(factorial(5), 120);
/// ```
///
/// # Panics
/// Panics if `n > 20` (would overflow `u64`).
pub fn factorial(n: u64) -> u64 {
    // Implementation...
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorial_small() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn test_factorial_overflow() {
        factorial(21);
    }
}
```

### 5. Update Documentation

- Add `///` doc comments to all public items
- Include at least one usage example per function
- Document panics, errors, and mathematical conventions
- Update the crate's `README.md` if adding significant features

### 6. Run the Validation Suite

```bash
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

### 7. Commit and Push

```bash
git add .
git commit -m "feat: add factorial function to combinatorics"
# Conventional commits: feat / fix / docs / refactor / test / chore
git push origin feature/your-feature-name
```

### 8. Open a Pull Request

- Describe what you changed and why
- Link to any related issues (`Fixes #123`)
- Ensure CI passes

---

## Coding Standards

### Rust Style

- Follow `rustfmt` defaults (run `cargo fmt --all`)
- Use `clippy::pedantic` lints (configured workspace-wide)
- **No `unsafe` code** (forbidden at workspace level)
- Prefer explicit error handling (`Result`) over panics

### Documentation

Every public item must include, in order:

1. **Mathematical definition** — what it computes
2. **Formula** — rendered in plain text or LaTeX-compatible notation
3. **Complexity** — Big-O notation
4. **Numerical stability notes** — cancellation, overflow, ill-conditioning
5. **References** — sources (papers, textbooks, standards)
6. **Examples** — runnable `# Examples` doc-tests

Example:

```rust
/// Compute the Fast Fourier Transform (FFT) of a signal.
///
/// Uses the radix-2 Cooley-Tukey algorithm. Input length must be a power of 2.
///
/// # Mathematical Convention
/// Forward transform uses `exp(-2pi*i*k*n/N)` and is un-normalized.
/// Use [`ifft`] for the inverse, which applies the `1/N` scaling factor.
///
/// # Arguments
/// * `signal` — Input signal (length must be power of 2)
///
/// # Returns
/// Complex spectrum with same length as input.
///
/// # Panics
/// Panics if input length is not a power of 2.
///
/// # Examples
/// ```
/// use mathverse_transforms::fft::fft;
///
/// let signal = vec![1.0, 0.0, 0.0, 0.0];
/// let spectrum = fft(&signal);
/// assert_eq!(spectrum.len(), 4);
/// ```
///
/// # See Also
/// - [`ifft`] — Inverse FFT
/// - [`dct`] — Discrete Cosine Transform
pub fn fft(signal: &[f64]) -> Vec<Complex<f64>> {
    // Implementation...
}
```

### Error Handling

Use the workspace-wide error type from `mathverse-core`; do not define
per-crate error enums:

```rust
use mathverse_core::error::{MathError, MathResult};

pub fn invert(a: &Matrix) -> MathResult<Matrix> {
    if !a.is_square() {
        return Err(MathError::DimensionMismatch);
    }
    // ...
}
```

### Testing

Every crate should have:

1. **Unit tests** — test individual functions
2. **Integration tests** — test crate-level APIs (in `tests/`)
3. **Doc tests** — ensure examples compile and run
4. **Property tests** — use `proptest` for algorithmic properties
5. **Golden tests** — for numerical algorithms, compare against known-good outputs

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_abs_diff_eq!(dot(&a, &b), 32.0, epsilon = 1e-10);
    }
}
```

---

## Adding a New Algorithm

1. **Check if it belongs in an existing crate**
   - Root finding: `mathverse-numerical`
   - Matrix operations: `mathverse-matrix`
   - ML algorithms: `mathverse-machine-learning`

2. **Add the function with full documentation**

3. **Write comprehensive tests**
   - At least 3–5 unit tests
   - Golden test against reference implementation if available
   - Edge cases (empty input, singular matrices, etc.)

4. **Add an example** (in `examples/` directory)

5. **Update the crate's `README.md`** if it is a significant addition

6. **Consider performance**
   - Add benchmarks for O(n^2) or slower algorithms
   - Profile with `cargo flamegraph` if performance-critical

---

## Reporting Bugs

### Before Reporting

1. Search existing issues
2. Try with the latest version
3. Minimize the reproduction case

### What to Include

- Rust version (`rustc --version`)
- MathVerse version
- Minimal code reproducing the issue
- Expected vs. actual behavior
- Error messages (full output)

---

## Code Review Process

### What We Look For

1. **Correctness** — does it work as specified?
2. **Tests** — are there sufficient tests?
3. **Documentation** — is it documented clearly?
4. **Style** — does it follow Rust conventions?
5. **Performance** — any obvious inefficiencies?
6. **Breaking changes** — are they necessary and documented?

### Review Timeline

- Small PRs (< 100 lines): 1–2 days
- Medium PRs (100–500 lines): 3–5 days
- Large PRs (> 500 lines): 1–2 weeks

Smaller PRs get reviewed faster.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License, matching the project license.
