# Algorithm Consolidation Guide

## Overview

This guide addresses the primary remaining architectural issue: algorithm duplication across multiple crates. Consolidating these implementations will:

1. **Eliminate maintenance burden** - One implementation to test and debug
2. **Ensure consistency** - Same algorithms, same numerical behavior
3. **Improve discoverability** - Users know where to find each algorithm
4. **Enable better optimization** - Focus optimization effort on canonical implementations

---

## 🎯 Consolidation Targets

### Target 1: Root Finding & ODE Solvers

**Current State:**
- Root finding appears in: `mathverse-calculus`, `mathverse-numerical`, `mathverse-equations`
- ODE solving appears in: `mathverse-calculus`, `mathverse-numerical`

**Canonical Owner:** `mathverse-numerical`
**Reasoning:** Most general-purpose, dedicated to numerical methods

**Implementation Plan:**

#### Step 1: Audit Current Implementations

```bash
# Find all root-finding functions
rg "fn.*root" crates/mathverse-{calculus,numerical,equations}/src/ --type rust

# Find all ODE-solving functions
rg "fn.*(ode|rk4|euler)" crates/mathverse-{calculus,numerical}/src/ --type rust
```

Create a comparison matrix:

| Algorithm | mathverse-calculus | mathverse-numerical | mathverse-equations | Keep In |
|-----------|-------------------|---------------------|---------------------|---------|
| Newton-Raphson | ✓ | ✓ | ✓ | numerical |
| Bisection | ✓ | ✓ | - | numerical |
| Secant | - | ✓ | ✓ | numerical |
| RK4 (ODE) | ✓ | ✓ | - | numerical |
| Euler (ODE) | ✓ | ✓ | - | numerical |
| etc. | ... | ... | ... | ... |

#### Step 2: Enhance mathverse-numerical

Ensure `mathverse-numerical` has the most complete implementations:

```rust
// crates/mathverse-numerical/src/root_finding.rs

/// Newton-Raphson root finding.
///
/// Finds a root of `f(x) = 0` near initial guess `x0` using
/// Newton's iteration: x_{n+1} = x_n - f(x_n) / f'(x_n).
///
/// # Arguments
/// * `f` - Function to find root of
/// * `df` - Derivative of f
/// * `x0` - Initial guess
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// * `Ok(x)` - Root found within tolerance
/// * `Err(...)` - Failed to converge or derivative zero
///
/// # Examples
/// ```
/// use mathverse_numerical::root_finding::newton_raphson;
///
/// let f = |x: f64| x * x - 2.0;
/// let df = |x: f64| 2.0 * x;
/// let root = newton_raphson(f, df, 1.0, 1e-10, 100).unwrap();
/// assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10);
/// ```
pub fn newton_raphson<F, DF>(
    f: F,
    df: DF,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, NumericalError>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    // Implementation...
}
```

Add comprehensive test suite:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_newton_raphson_sqrt() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let root = newton_raphson(f, df, 1.0, 1e-12, 100).unwrap();
        assert_abs_diff_eq!(root, std::f64::consts::SQRT_2, epsilon = 1e-10);
    }

    #[test]
    fn test_newton_raphson_cubic() {
        // Test against known roots...
    }

    #[test]
    fn test_newton_raphson_no_convergence() {
        // Should return Err for oscillating cases...
    }
}
```

#### Step 3: Update mathverse-calculus to Depend on numerical

```toml
# crates/mathverse-calculus/Cargo.toml

[dependencies]
mathverse-core = { path = "../mathverse-core", version = "0.1.0" }
mathverse-numerical = { path = "../mathverse-numerical", version = "0.1.0" }  # ADD THIS
mathverse-algebra = { path = "../mathverse-algebra", version = "0.1.0" }
# ...
```

Replace implementations with re-exports:

```rust
// crates/mathverse-calculus/src/root_finding.rs

//! Root finding for calculus applications.
//!
//! This module re-exports root-finding algorithms from [`mathverse_numerical`],
//! providing convenient access for calculus-specific use cases.
//!
//! For general-purpose root finding, use [`mathverse_numerical::root_finding`] directly.

pub use mathverse_numerical::root_finding::{
    newton_raphson,
    bisection,
    secant,
    NumericalError,
};

/// Convenience wrapper: find root of derivative (critical points).
///
/// # Examples
/// ```
/// use mathverse_calculus::root_finding::find_critical_point;
///
/// let f = |x: f64| x.powi(3) - 3.0 * x;  // Has critical points at x = ±1
/// let critical = find_critical_point(f, 0.5, 1e-10, 100).unwrap();
/// assert!((critical - 1.0).abs() < 1e-8);
/// ```
pub fn find_critical_point<F>(
    f: F,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, NumericalError>
where
    F: Fn(f64) -> f64,
{
    // Use numerical differentiation + newton_raphson on the derivative
    let df = |x: f64| (f(x + tol) - f(x - tol)) / (2.0 * tol);
    let ddf = |x: f64| (df(x + tol) - df(x - tol)) / (2.0 * tol);
    newton_raphson(df, ddf, x0, tol, max_iter)
}
```

#### Step 4: Deprecate Old mathverse-equations Implementations

```rust
// crates/mathverse-equations/src/nonlinear.rs

#[deprecated(since = "0.2.0", note = "Use mathverse_numerical::root_finding instead")]
pub fn newton_root(f: impl Fn(f64) -> f64, df: impl Fn(f64) -> f64, x0: f64) -> f64 {
    // Keep implementation for one minor version to avoid breaking users,
    // but forward to the canonical location internally
    use mathverse_numerical::root_finding::newton_raphson;
    newton_raphson(f, df, x0, 1e-10, 100)
        .unwrap_or_else(|_| panic!("Failed to converge"))
}
```

Add clear migration guide in the deprecation docs:

```rust
/// # Migration Guide
///
/// **Before (mathverse-equations 0.1.x):**
/// ```ignore
/// use mathverse_equations::nonlinear::newton_root;
/// let root = newton_root(f, df, 1.0);
/// ```
///
/// **After (mathverse-equations 0.2.x):**
/// ```ignore
/// use mathverse_numerical::root_finding::newton_raphson;
/// let root = newton_raphson(f, df, 1.0, 1e-10, 100)?;
/// ```
///
/// Note: The new API returns `Result` for proper error handling.
```

#### Step 5: Update CHANGELOG

```markdown
# Changelog

## [0.2.0] - 2026-XX-XX

### Changed (Breaking)
- **mathverse-calculus**: Root finding algorithms now re-export from `mathverse-numerical` instead of providing independent implementations. Behavior is identical but import paths change.
- **mathverse-equations**: Root finding in `nonlinear` module now deprecated; use `mathverse-numerical::root_finding` directly.

### Migration
If you used `mathverse_calculus::root_finding`, no code changes needed (re-export path unchanged).
If you used `mathverse_equations::nonlinear::newton_root`, migrate to:
```rust
use mathverse_numerical::root_finding::newton_raphson;
let root = newton_raphson(f, df, x0, 1e-10, 100)?; // Now returns Result
```

### Added
- **mathverse-numerical**: Comprehensive documentation for all root-finding algorithms
- **mathverse-calculus**: New `find_critical_point` convenience wrapper
```

---

### Target 2: Optimization Algorithms

**Current State:**
- Optimization appears in: `mathverse-optimization` (dedicated), `mathverse-numerical`

**Canonical Owner:** `mathverse-optimization`
**Reasoning:** Dedicated crate with more comprehensive coverage

**Implementation Plan:**

#### Step 1: Move or Re-export

Check what's in `mathverse-numerical/src/optimization.rs`:

```bash
cat crates/mathverse-numerical/src/optimization.rs | head -50
```

If it's basic (gradient descent only), have it depend on and re-export from `mathverse-optimization`:

```rust
// crates/mathverse-numerical/src/optimization.rs

//! Basic optimization for numerical methods.
//!
//! This module re-exports commonly-used optimization algorithms from
//! [`mathverse_optimization`]. For comprehensive optimization (genetic
//! algorithms, simulated annealing, linear programming, etc.), use
//! [`mathverse_optimization`] directly.

pub use mathverse_optimization::gradient::{
    gradient_descent,
    GradientDescentConfig,
};

// Keep any numerical-methods-specific wrappers here if needed
```

Update Cargo.toml:

```toml
# crates/mathverse-numerical/Cargo.toml

[dependencies]
mathverse-optimization = { path = "../mathverse-optimization", version = "0.1.0" }
```

---

### Target 3: FFT & Spectral Analysis

**Current State:**
- FFT implemented in: `mathverse-transforms` (comprehensive)
- Spectral analysis in: `mathverse-signal` (likely calls FFT internally or duplicates)

**Canonical Owner:** `mathverse-transforms`
**Reasoning:** FFT is fundamentally a transform, signal processing should use it as a building block

**Implementation Plan:**

#### Step 1: Verify mathverse-signal Dependency

Check if `mathverse-signal` already depends on `mathverse-transforms`:

```bash
grep -A5 dependencies crates/mathverse-signal/Cargo.toml
```

If not, add it:

```toml
# crates/mathverse-signal/Cargo.toml

[dependencies]
mathverse-core = { path = "../mathverse-core", version = "0.1.0" }
mathverse-transforms = { path = "../mathverse-transforms", version = "0.1.0" }  # ADD THIS
```

#### Step 2: Refactor mathverse-signal to Use Transforms

```rust
// crates/mathverse-signal/src/spectrum.rs

//! Spectral analysis for signal processing.
//!
//! All FFT operations delegate to [`mathverse_transforms::fft`].

use mathverse_transforms::fft::{fft, ifft};

/// Compute power spectral density (PSD) via FFT.
///
/// # Examples
/// ```
/// use mathverse_signal::spectrum::power_spectrum;
///
/// let signal = vec![1.0, 0.5, -0.5, -1.0];
/// let psd = power_spectrum(&signal);
/// assert_eq!(psd.len(), signal.len() / 2 + 1);  // One-sided spectrum
/// ```
pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
    let spectrum = fft(signal);
    
    // Compute one-sided power spectrum: |X[k]|² for k = 0..N/2
    let n = signal.len();
    let nyquist = n / 2 + 1;
    
    spectrum[..nyquist]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im) / (n as f64))
        .collect()
}

/// Compute spectrogram (time-frequency representation) via short-time FFT.
pub fn spectrogram(
    signal: &[f64],
    window_size: usize,
    hop_size: usize,
) -> Vec<Vec<f64>> {
    signal
        .windows(window_size)
        .step_by(hop_size)
        .map(|window| {
            let windowed: Vec<f64> = window
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    // Hann window
                    let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / window_size as f64).cos());
                    x * w
                })
                .collect();
            power_spectrum(&windowed)
        })
        .collect()
}
```

Remove any internal FFT implementation from `mathverse-signal/src/` if one exists.

---

## 📋 Consolidation Checklist

Use this checklist to track consolidation progress:

### Root Finding & ODE
- [ ] Audit all root-finding implementations across 3 crates
- [ ] Create comparison matrix documenting differences
- [ ] Enhance mathverse-numerical with best-of-breed implementations
- [ ] Add comprehensive tests to mathverse-numerical
- [ ] Update mathverse-calculus to depend on mathverse-numerical
- [ ] Replace mathverse-calculus implementations with re-exports
- [ ] Deprecate mathverse-equations nonlinear module
- [ ] Write migration guide
- [ ] Update CHANGELOG for all affected crates
- [ ] Verify all tests pass
- [ ] Update documentation with cross-references

### Optimization
- [ ] Audit mathverse-numerical optimization module
- [ ] Decide: merge into optimization or re-export?
- [ ] Update dependencies
- [ ] Refactor/re-export as appropriate
- [ ] Add cross-reference docs
- [ ] Update CHANGELOG

### FFT & Spectral
- [ ] Verify mathverse-signal doesn't duplicate FFT
- [ ] Add mathverse-transforms dependency to mathverse-signal
- [ ] Refactor spectrum.rs to use transforms::fft
- [ ] Remove any internal FFT code
- [ ] Add tests for spectral analysis
- [ ] Document the layering (transforms → signal)
- [ ] Update CHANGELOG

---

## 🧪 Testing Strategy

After each consolidation, run this full test suite:

```bash
# 1. Check workspace builds
cargo check --workspace

# 2. Run all tests
cargo test --workspace

# 3. Run clippy (should pass with -D warnings)
cargo clippy --workspace --all-targets -- -D warnings

# 4. Build docs (verify cross-references work)
cargo doc --workspace --no-deps

# 5. Test examples still work
cargo run --example root_finding  # mathverse-numerical
cargo run --example optimization   # mathverse-optimization
cargo run --example fft_demo        # mathverse-transforms
```

### Regression Test Suite

Create a new integration test to ensure consolidated algorithms match old behavior:

```rust
// crates/tests/consolidation_regression.rs

//! Regression tests ensuring consolidated algorithms maintain exact numerical behavior.

#[test]
fn test_root_finding_unchanged() {
    // Known input/output pairs from pre-consolidation
    let f = |x: f64| x * x - 2.0;
    let df = |x: f64| 2.0 * x;
    
    let root = mathverse_numerical::root_finding::newton_raphson(
        f, df, 1.0, 1e-12, 100
    ).unwrap();
    
    // This should match pre-consolidation output exactly
    assert_eq!(root, 1.4142135623730951);  // Exact value from old version
}
```

---

## 📅 Phased Rollout

### Phase 1: Week 1-2 (Preparation)
- [ ] Create feature branch: `consolidation/root-finding`
- [ ] Complete audit and comparison matrices
- [ ] Draft migration guides
- [ ] Set up regression test fixtures

### Phase 2: Week 3-4 (Root Finding & ODE)
- [ ] Implement consolidated mathverse-numerical
- [ ] Update dependents
- [ ] Run full test suite
- [ ] Get review/feedback

### Phase 3: Week 5 (Optimization)
- [ ] Implement optimization consolidation
- [ ] Update dependents
- [ ] Run full test suite

### Phase 4: Week 6 (FFT & Spectral)
- [ ] Implement FFT consolidation
- [ ] Update mathverse-signal
- [ ] Run full test suite

### Phase 5: Week 7 (Documentation & Release)
- [ ] Update all CHANGELOGs
- [ ] Update root README with architecture diagram
- [ ] Create migration guide document
- [ ] Tag release: v0.2.0

---

## 🏗️ Architecture Diagram (Post-Consolidation)

```
                    ┌─────────────────────┐
                    │  mathverse-core     │
                    │  (numeric traits,   │
                    │   errors, precision)│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼────────┐ ┌────▼─────────┐ ┌───▼──────────┐
    │ mathverse-       │ │ mathverse-   │ │ mathverse-   │
    │ numerical        │ │ transforms   │ │ optimization │
    │ (CANONICAL for   │ │ (CANONICAL   │ │ (CANONICAL   │
    │  root finding,   │ │  for FFT,    │ │  for gradient│
    │  ODE, interp.)   │ │  DCT, etc.)  │ │  methods, LP)│
    └────────┬─────────┘ └──────┬───────┘ └──────────────┘
             │                  │
    ┌────────▼─────────┐  ┌────▼──────────┐
    │ mathverse-       │  │ mathverse-    │
    │ calculus         │  │ signal        │
    │ (re-exports +    │  │ (builds on    │
    │  calculus-       │  │  transforms)  │
    │  specific utils) │  │               │
    └──────────────────┘  └───────────────┘
```

**Key Principles:**
1. **Single source of truth** - Each algorithm has one canonical implementation
2. **Layered architecture** - Higher-level crates build on lower-level ones
3. **Re-export for convenience** - Domain crates can re-export from canonical sources
4. **Clear documentation** - Every re-export explains where the canonical version lives

---

## ✅ Success Criteria

Consolidation is complete when:

1. ✅ `cargo check --workspace` passes
2. ✅ `cargo test --workspace` passes with no regressions
3. ✅ `cargo clippy --workspace -- -D warnings` passes
4. ✅ Every algorithm has exactly one implementation (canonical)
5. ✅ All re-exports are documented with `/// See [canonical location]`
6. ✅ Migration guide exists for breaking changes
7. ✅ CHANGELOG updated for all affected crates
8. ✅ No user-visible behavioral changes (pure refactor)
9. ✅ CI passes on all targets
10. ✅ Documentation builds with no warnings and cross-references work

---

## 🆘 Rollback Plan

If consolidation introduces issues:

1. **Keep feature branch** - Don't merge to main until fully validated
2. **Tag pre-consolidation state** - `git tag pre-consolidation-v0.1.x`
3. **Incremental rollout** - Do root-finding first, validate, then proceed
4. **Maintain deprecation period** - Keep old APIs for at least one minor version

---

## 📞 Questions?

Common questions and answers:

**Q: Why not just leave duplicate implementations?**
A: Maintenance burden multiplies. Every bug fix needs 3x patches. Numerical behavior diverges over time.

**Q: What about performance differences between implementations?**
A: Consolidate to the *fastest* implementation. Add benchmarks to verify.

**Q: Will this break existing code?**
A: Re-exports mean most code doesn't break. Deprecation warnings guide migration for direct uses.

**Q: How long should we keep deprecated APIs?**
A: At least one minor version (e.g., deprecate in 0.2.0, remove in 0.3.0).

---

**Ready to start?** Begin with Phase 1 (audit) and work through the checklist systematically. Good luck! 🚀
