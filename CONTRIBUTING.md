# Contributing to MathVerse

Thank you for your interest in contributing to MathVerse! This guide will help you get started.

## 🚀 Quick Start

### Prerequisites
- Rust 1.87+ (workspace MSRV)
- Git
- A GitHub account

### Setup
```bash
# Clone the repository
git clone https://github.com/Rohithdgrr/Rust-math.git
cd Rust-math

# Check everything builds
cargo check --workspace

# Run tests
cargo test --workspace

# Run lints
cargo clippy --workspace --all-targets -- -D warnings

# Build documentation
cargo doc --workspace --no-deps --open
```

---

## 📁 Repository Structure

```
Rust-math/
├── crates/
│   ├── mathverse-core/          # Numeric traits, errors, foundation
│   ├── mathverse-algebra/       # Polynomials, equation solving
│   ├── mathverse-matrix/        # Dense/sparse matrices, decompositions
│   ├── mathverse-ai/            # Neural nets, autograd, attention
│   ├── mathverse-machine-learning/  # Classical ML algorithms
│   └── ...                      # 31 total crates
├── .github/workflows/           # CI/CD pipelines
├── scripts/                     # Validation and helper scripts
├── PRODUCTION_READINESS_STATUS.md  # Current quality status
├── CONSOLIDATION_GUIDE.md       # Algorithm deduplication plan
└── CONTRIBUTING.md              # This file
```

### Key Crates to Know

| Crate | Purpose | Depends On |
|-------|---------|------------|
| **mathverse-core** | Foundation - all crates should use this | - |
| **mathverse-prelude** | One-stop import for everything | All others |
| **mathverse-numerical** | Canonical root-finding, ODE, interpolation | core |
| **mathverse-optimization** | Canonical optimization algorithms | core, probability |
| **mathverse-transforms** | Canonical FFT, DCT, wavelets | complex |

---

## 🎯 How to Contribute

### 1. Find an Issue

Check our [issue tracker](https://github.com/Rohithdgrr/Rust-math/issues) for:
- `good first issue` - Perfect for newcomers
- `help wanted` - We'd love assistance here
- `documentation` - Always appreciated!

Or review [`PRODUCTION_READINESS_STATUS.md`](./PRODUCTION_READINESS_STATUS.md) for known gaps.

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-123
```

### 3. Make Your Changes

Follow our coding standards (see below).

### 4. Write Tests

Every new function needs:
- At least one unit test
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
/// Panics if n > 20 (would overflow u64).
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
- Include examples in doc comments
- Document panics, errors, and mathematical conventions
- Update the crate's README.md if adding significant features

### 6. Run the Validation Suite

```bash
# On Windows (PowerShell)
.\scripts\validate_workspace.ps1

# Or manually:
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

### 7. Commit and Push

```bash
git add .
git commit -m "feat: add factorial function to combinatorics"
# Follow conventional commits: feat/fix/docs/refactor/test/chore
git push origin feature/your-feature-name
```

### 8. Open a Pull Request

- Describe what you changed and why
- Link to any related issues (`Fixes #123`)
- Ensure CI passes

---

## 📝 Coding Standards

### Rust Style
- Follow `rustfmt` defaults (run `cargo fmt --all`)
- Use `clippy::pedantic` lints (already configured workspace-wide)
- **No `unsafe` code** (forbidden at workspace level)
- Prefer explicit error handling (`Result`) over panics

### Documentation
- **Every public item must have `///` documentation**
- Include at least one usage example
- Document panics and errors explicitly
- Explain mathematical conventions (normalization, coordinate systems, etc.)

Example of good documentation:

```rust
/// Compute the Fast Fourier Transform (FFT) of a signal.
///
/// Uses the radix-2 Cooley-Tukey algorithm. Input length must be a power of 2.
///
/// # Mathematical Convention
/// Forward transform uses `exp(-2πi·k·n/N)` and is **un-normalized**.
/// Use [`ifft`] for the inverse, which applies the `1/N` scaling factor.
///
/// # Arguments
/// * `signal` - Input signal (length must be power of 2)
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
/// - [`ifft`] - Inverse FFT
/// - [`dct`] - Discrete Cosine Transform
pub fn fft(signal: &[f64]) -> Vec<Complex<f64>> {
    // Implementation...
}
```

### Error Handling

Use `thiserror` for error types:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LinearAlgebraError {
    #[error("matrix is singular (determinant = 0)")]
    SingularMatrix,
    
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("matrix is not positive definite")]
    NotPositiveDefinite,
}
```

### Testing

Every crate should have:
1. **Unit tests** - Test individual functions
2. **Integration tests** - Test crate-level APIs (in `tests/`)
3. **Doc tests** - Ensure examples compile and run
4. **Property tests** - Use `proptest` for algorithmic properties
5. **Golden tests** - For numerical algorithms, compare against known-good outputs

```rust
// Unit test
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

// Property test (in dev-dependencies: proptest = "1.0")
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_norm_non_negative(v in prop::collection::vec(-1000.0..1000.0, 1..100)) {
            let norm = l2_norm(&v);
            assert!(norm >= 0.0);
        }
    }
}
```

---

## 🏗️ Adding a New Algorithm

1. **Check if it belongs in an existing crate**
   - Root finding → `mathverse-numerical`
   - Matrix operations → `mathverse-matrix`
   - ML algorithms → `mathverse-machine-learning`
   - etc.

2. **Add the function with full documentation**

3. **Write comprehensive tests**
   - At least 3-5 unit tests
   - Golden test against reference implementation if available
   - Edge cases (empty input, singular matrices, etc.)

4. **Add an example** (in `examples/` directory)

5. **Update the crate's README** if it's a significant addition

6. **Consider performance**
   - Add benchmarks for O(n²) or slower algorithms
   - Profile with `cargo flamegraph` if performance-critical

---

## 🐛 Reporting Bugs

### Before Reporting
1. Search existing issues
2. Try with the latest version
3. Minimize the reproduction case

### What to Include
- Rust version (`rustc --version`)
- MathVerse version
- Minimal code reproducing the issue
- Expected vs. actual behavior
- Error messages (full stack trace)

### Example Bug Report

```markdown
## Bug: FFT returns incorrect results for length-8 signals

**Environment:**
- Rust: 1.87.0
- MathVerse: 0.1.0
- OS: Windows 11

**Code:**
\`\`\`rust
use mathverse_transforms::fft::fft;

let signal = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
let spectrum = fft(&signal);
println!("{:?}", spectrum);
\`\`\`

**Expected:** First coefficient should be 4.0+0i (DC component = sum of signal)
**Actual:** Getting 3.8+0.2i

**Additional context:**
Works correctly for length-4 signals. Seems specific to length-8.
```

---

## 🎨 Feature Requests

We welcome feature requests! Please:
1. Check if it's already requested
2. Explain the use case (not just "add feature X", but "I need X to do Y")
3. Suggest which crate it belongs in
4. Provide pseudocode or reference implementations if available

---

## 🔍 Code Review Process

### What We Look For
1. **Correctness** - Does it work as specified?
2. **Tests** - Are there sufficient tests?
3. **Documentation** - Is it documented clearly?
4. **Style** - Does it follow Rust conventions?
5. **Performance** - Any obvious inefficiencies?
6. **Breaking changes** - Are they necessary and documented?

### Review Timeline
- Small PRs (< 100 lines): 1-2 days
- Medium PRs (100-500 lines): 3-5 days
- Large PRs (> 500 lines): 1-2 weeks

**Tip:** Smaller PRs get reviewed faster!

---

## 🏆 Recognition

Contributors will be:
- Listed in the crate's `Cargo.toml` authors (for significant contributions)
- Mentioned in release notes
- Credited in the main README

---

## 📞 Questions?

- **Discord/Chat:** [Coming soon]
- **GitHub Discussions:** Use for questions and architecture discussions
- **Issues:** For bugs and feature requests only

---

## 📜 License

By contributing, you agree that your contributions will be dual-licensed under MIT OR Apache-2.0, matching the project license.

---

## 🙏 Thank You!

Every contribution, no matter how small, makes MathVerse better. We appreciate your time and effort! 

**Happy coding!** 🚀
