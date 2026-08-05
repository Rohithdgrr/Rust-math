# MathVerse Crates Audit Reports

## 1. mathverse-core

**Files:** 13 source files, 1 test file (properties.rs, 13 proptest properties), 1 bench file
**Tests:** 169 passed, 0 failed

### Bugs
- None found. All 169 tests pass including property-based tests.

### Security Risks
- No `unsafe` blocks anywhere in the crate.
- No external network or filesystem access.
- Input validation is present (NaN/infinity checks in `ops.rs`, `float.rs`).
- The `almost_eq` function uses relative + absolute tolerance which is the correct approach for floating-point comparison.

### Code Quality
- Excellent. Consistent use of `#[inline]`, `#[must_use]`, `#[allow(clippy::excessive_precision)]`.
- Comprehensive doc comments with `///` and examples on all public APIs.
- Good trait-based design: `Zero`, `One`, `Float` from `num-traits` for generic numeric programming.
- `libm_fallback.rs` provides portable fallbacks for libm functions — good cross-platform consideration.
- `precision.rs` is well-designed with `almost_eq`, `ulp`, `next_float`, `prev_float`, `round_to`, etc.
- `algorithms.rs` contains well-implemented algorithms (Euclidean GCD, Miller-Rabin primality, sieve of Eratosthenes, etc.).

### Over-Engineering
- `libm_fallback.rs` is ~200 lines of fallbacks for libm functions that are available on most platforms. Could be conditionally compiled behind a feature flag instead of always present.
- `constants.rs` hardcodes many constants that are available in `std::f64::consts`. The custom constants (e.g., `EULER_MASCHERONI`, `CATALAN`) are justified, but some duplicates exist.
- The `Float` trait in `traits.rs` re-implements some `num-traits` functionality.

### Hardcoded Data
- `constants.rs` has hardcoded values for Euler-Mascheroni, Catalan, Khinchin, etc. These are well-known mathematical constants and hardcoding them is acceptable, but they should be `const` (they are).
- `LANCZOS_COEF` in gamma.rs (in mathverse-special, not core) — not in this crate.

### Broken Features
- None. All features work correctly.

### Python-Parity Gaps
- Missing: `math.isclose()` equivalent is `almost_eq` but it's not named as such.
- Missing: `math.fsum()` — no Kahan summation or pairwise summation for high-accuracy float addition.
- Missing: `math.gcd()` for multiple arguments (only binary GCD exists).
- Missing: `math.lcm()` for multiple arguments.
- Missing: `math.perm()` and `math.comb()` — `binomial` exists but named differently.
- Missing: `math.isqrt()` — integer square root not provided.
- Missing: `math.factorial()` returns `u128` which overflows at 35! — Python's `math.factorial` uses arbitrary precision.

### Improvement Suggestions
1. Add a `math.isqrt()` function for integer square root.
2. Add Kahan summation (`fsum`) for accurate float addition.
3. Consider making `libm_fallback.rs` conditional on a `no-std` or `libm-fallback` feature flag.
4. Add `math.perm()` and `math.comb()` as aliases for `binomial` with multi-argument support.

### Ratings
- Correctness: 9/10
- Security: 10/10
- Code Quality: 9/10
- Completeness: 7/10
- Python-Parity: 6/10

---

## 2. mathverse-arithmetic

**Files:** 5 source files, 0 test files (only inline `#[cfg(test)]` in `rounding.rs`)
**Tests:** 24 passed (all inline), 0 failed

### Bugs
- None found. All 24 tests pass.

### Security Risks
- No `unsafe` blocks.
- `checked_ops.rs` uses `checked_add`, `checked_sub`, etc. which is good for overflow safety.
- `finance.rs` has no input validation for negative rates or periods — `compound_interest` and `present_value` accept negative `rate` without warning.

### Code Quality
- Good documentation with examples on all public functions.
- `finance.rs` implements standard financial formulas correctly.
- `percentage.rs` is clean and simple.
- `rounding.rs` implements multiple rounding modes (up, down, nearest, zero, bankers) — well done.
- `checked_ops.rs` provides overflow-checked arithmetic — good for safety.

### Over-Engineering
- `finance.rs` includes `amortization_schedule` which returns a `Vec<AmortizationEntry>` — this is a complex return type for a simple calculation. Could just return the total interest paid.
- `percentage.rs` has `percentage_change` which is trivial (just `(new - old) / old * 100.0`) — fine as-is.

### Hardcoded Data
- None significant.

### Broken Features
- None.

### Python-Parity Gaps
- Missing: `decimal.Decimal` equivalent — no arbitrary-precision decimal arithmetic.
- Missing: `fractions.Fraction` equivalent — `rational.rs` is in mathverse-algebra, not here.
- Missing: `math.fsum()` — no Kahan summation.
- Missing: `math.isclose()` — no floating-point comparison utility (that's in core).
- Missing: `math.remainder()` — the IEEE 754 remainder operation.
- Missing: `math.powm()` — modular exponentiation (only in checked_ops as `checked_pow`).

### Improvement Suggestions
1. Add input validation for financial functions (negative rate, zero period).
2. Consider adding arbitrary-precision decimal arithmetic.
3. Add `math.remainder()` as a public function.
4. Add tests for `finance.rs` and `percentage.rs` (currently no test files).

### Ratings
- Correctness: 8/10
- Security: 8/10
- Code Quality: 8/10
- Completeness: 6/10
- Python-Parity: 5/10

---

## 3. mathverse-algebra

**Files:** 15 source files, 0 test files (only inline `#[cfg(test)]`)
**Tests:** 88 passed (all inline doc tests), 0 failed

### Bugs
- None found. All 88 doc tests pass.

### Security Risks
- No `unsafe` blocks.
- `roots.rs` uses Newton's method which can diverge for poor initial guesses — no convergence guard or max-iteration limit in all paths.
- `systems.rs` uses Gaussian elimination without pivoting — can be numerically unstable for ill-conditioned matrices.

### Code Quality
- Excellent breadth of algebraic functionality.
- `polynomial.rs` is well-structured with `Polynomial` type and operations.
- `latex.rs` generates LaTeX output — unique feature not in Python's standard library.
- `interpolate.rs` implements Lagrange and Newton interpolation — good.
- `solvability.rs` implements Galois theory solvability check — impressive.
- `symmetric.rs` implements elementary symmetric polynomials and Newton's identities.
- `determinant.rs` uses Laplace expansion — O(n!) complexity, fine for small matrices but not scalable.

### Over-Engineering
- `latex.rs` is ~200 lines of LaTeX generation — this is a niche feature that adds significant complexity. If the crate's purpose is algebraic computation, LaTeX output is a nice-to-have but shouldn't dominate the codebase.
- `compose.rs` implements function composition — useful but could be a one-liner.
- `determinant.rs` uses recursive Laplace expansion which is O(n!). For n > 10 this becomes impractical. Should use LU decomposition for larger matrices.

### Hardcoded Data
- None significant.

### Broken Features
- None.

### Python-Parity Gaps
- Missing: `sympy`-equivalent symbolic computation — this crate does numeric algebra, not symbolic.
- Missing: `numpy.linalg.solve()` equivalent — `systems.rs` exists but only for small systems.
- Missing: `numpy.polyfit()` — polynomial fitting is not provided.
- Missing: `cmath` module equivalents for complex algebra (that's in mathverse-complex).
- Missing: `math.gcd()` for polynomials — GCD of polynomials not implemented.

### Improvement Suggestions
1. Add pivoting to Gaussian elimination in `systems.rs`.
2. Add a max-iteration/convergence guard to Newton's method in `roots.rs`.
3. Use LU decomposition for determinant computation instead of Laplace expansion.
4. Add polynomial GCD functionality.
5. Add tests for `polynomial.rs`, `roots.rs`, `systems.rs`, and `determinant.rs`.

### Ratings
- Correctness: 8/10
- Security: 7/10
- Code Quality: 8/10
- Completeness: 7/10
- Python-Parity: 6/10

---

## 4. mathverse-complex

**Files:** 4 source files, 0 test files (only inline `#[cfg(test)]` in `lib.rs`)
**Tests:** 2 doc tests passed, 0 failed

### Bugs
- None found. The 2 doc tests pass.

### Security Risks
- No `unsafe` blocks.
- `analysis.rs` implements complex differentiation using finite differences — no guard against step size being too small (catastrophic cancellation).

### Code Quality
- Clean and well-structured.
- `Complex` struct with `new`, `from_polar`, `conjugate`, `abs`, `arg`, `exp`, `log`, `sqrt`, `pow` — covers the essentials.
- `matrix.rs` implements complex matrix operations.
- `special_functions.rs` re-exports from mathverse-special for complex arguments.
- Good use of `num-traits` for generic numeric bounds.

### Over-Engineering
- `special_functions.rs` is mostly a re-export shim — could be simplified to just re-export from `mathverse-special`.
- `matrix.rs` implements basic matrix ops but doesn't implement `Mul` trait for complex matrices — inconsistent with `mathverse-matrix`.

### Hardcoded Data
- None significant.

### Broken Features
- `complex::log` has a branch cut along the negative real axis but doesn't document which branch is used (should use principal branch with `(-π, π]`).
- `complex::sqrt` doesn't handle the branch cut consistently with `log`.

### Python-Parity Gaps
- Missing: `cmath.phase()` — `arg()` exists but isn't named `phase`.
- Missing: `cmath.polar()` — `from_polar` exists but not a `polar()` function returning `(r, θ)`.
- Missing: `cmath.rect()` — no rectangular form constructor from polar.
- Missing: `cmath.exp()`, `cmath.log()`, `cmath.sqrt()` are all present but not named to match Python's `cmath` module.
- Missing: `cmath.isclose()` for complex numbers.
- Missing: `cmath.tan()`, `cmath.sin()`, `cmath.cos()` — trigonometric functions for complex numbers.

### Improvement Suggestions
1. Document the branch cut convention for `log` and `sqrt`.
2. Add `cmath`-compatible function names (`phase`, `polar`, `rect`).
3. Add complex trigonometric functions (`sin`, `cos`, `tan`).
4. Add `Mul` trait implementation for complex matrices.
5. Add tests for `analysis.rs` and `matrix.rs`.

### Ratings
- Correctness: 7/10
- Security: 9/10
- Code Quality: 7/10
- Completeness: 5/10
- Python-Parity: 5/10

---

## 5. mathverse-special

**Files:** 5 source files, 0 test files (all tests are inline `#[cfg(test)]`)
**Tests:** 14 total, 3 FAILED (y0_y1_reference, digamma_values, zeta_known_values)

### Bugs
1. **`bessel_y1(2.0)` tolerance too tight** (`bessel.rs:257`): The assertion `abs() < 1e-5` fails because the actual error is ~1e-5. The tolerance should be relaxed to `1e-4` or the implementation needs higher precision.
2. **`digamma(-0.5)` reflection formula bug** (`gamma.rs:268`): The test expects `digamma(-0.5) ≈ 1.9635100260214`, but the implementation uses `digamma(1-z) - π/tan(πz)`. For `z = -0.5`, this gives `digamma(1.5) - π/tan(-π/2)`. Since `tan(-π/2)` is `-∞`, the second term goes to 0, giving `digamma(1.5) ≈ 0.036489974`. The expected value `1.9635` is `ψ(0.5)`, not `ψ(-0.5)`. The reflection formula implementation appears correct but the test expectation is wrong — or the formula has a sign error in the `tan` term.
3. **`zeta(3.0)` approximation error** (`zeta.rs:79`): The tail estimate `N^(1-s)/(s-1) + ½·N^(-s)` with `N=1000` gives insufficient precision for `s=3`. The error is ~1e-9, exceeding the 1e-10 tolerance. Need more terms in the Euler-Maclaurin expansion or a larger `N`.

### Security Risks
- No `unsafe` blocks.
- `gamma_series` and `gamma_cf` use `FPMIN = 1e-300` to prevent division by zero — good defensive programming.
- `bessel_jn` uses forward recurrence which can overflow for large `n` — no overflow guard.

### Code Quality
- Good documentation with `///` comments and examples on all public functions.
- `gamma.rs` implements Lanczos approximation correctly.
- `erf.rs` uses the well-known A&S 7.1.26 approximation — good choice.
- `bessel.rs` implements power series per DLMF — correct approach.
- `zeta.rs` uses direct summation with Euler-Maclaurin tail — reasonable for `s > 1`.
- Code is well-organized with clear separation of concerns.

### Over-Engineering
- `gamma.rs` is ~290 lines — the incomplete gamma functions (`gamma_p`, `gamma_q`) are complex but well-implemented.
- `bessel.rs` is ~285 lines — the Y₀/Y₁ implementations using derivative-of-series approach are clever but could use standard DLMF 10.8.1 formulas for better numerical stability.
- `zeta.rs` could use a more sophisticated algorithm (e.g., Riemann-Siegel) for better precision at large `s`.

### Hardcoded Data
- `BERNOULLI_EVEN` in `zeta.rs` hardcodes Bernoulli numbers B₂ through B₁₆ — these are exact rational values stored as floats, which introduces rounding for larger indices.
- `LANCZOS_COEF` in `gamma.rs` hardcodes the Lanczos coefficients for g=7 — these are well-known and correct.
- `EULER_GAMMA` constant is hardcoded — fine, it's a mathematical constant.

### Broken Features
- The 3 failing tests indicate broken precision in `bessel_y1`, `digamma` (for negative non-integer arguments), and `zeta` (for odd integers ≥ 3).
- `zeta.rs` returns `NaN` for `s ≤ 1` but the Riemann zeta function is defined for `s < 1` via analytic continuation — only the pole at `s=1` is truly undefined.
- `gamma.rs` `digamma` reflection formula may have a sign error for negative arguments.

### Python-Parity Gaps
- Missing: `scipy.special.gamma()` — `gamma()` exists but `scipy.special.gamma` handles complex arguments.
- Missing: `scipy.special.gammaln()` — `log_gamma()` exists, equivalent.
- Missing: `scipy.special.erfinv()` and `scipy.special.erfcinv()` — inverse error functions not provided.
- Missing: `scipy.special.bessel_yn()` — only Y₀ and Y₁ are provided, not general Yₙ.
- Missing: `scipy.special.hyp1f1()` (confluent hypergeometric) and other higher special functions.
- Missing: `scipy.special.zeta()` for complex arguments — only real `s` is supported.
- Missing: `scipy.special.exp1()` (exponential integral).
- Missing: `scipy.special.ellipj()`, `scipy.special.ellipk()` — elliptic functions.

### Improvement Suggestions
1. Fix the 3 failing tests — relax tolerance for `bessel_y1`, fix `digamma` reflection formula, increase `N` or use Euler-Maclaurin with more terms for `zeta`.
2. Add `erfinv()` and `erfcinv()` — commonly needed in statistics.
3. Add `bessel_yn(n, x)` for general order Yₙ.
4. Add complex-argument support for zeta via analytic continuation.
5. Add overflow guards for `bessel_jn` forward recurrence.
6. Add tests for `zeta.rs`, `gamma.rs`, and `bessel.rs` as separate test files (not just inline).
7. Consider using `num-bigint` or `rug` for higher-precision Bernoulli numbers in zeta computation.

### Ratings
- Correctness: 5/10 (3 failing tests)
- Security: 9/10
- Code Quality: 8/10
- Completeness: 7/10
- Python-Parity: 5/10

---

## Summary

| Crate | Correctness | Security | Code Quality | Completeness | Python-Parity |
|-------|------------|----------|-------------|-------------|---------------|
| mathverse-core | 9/10 | 10/10 | 9/10 | 7/10 | 6/10 |
| mathverse-arithmetic | 8/10 | 8/10 | 8/10 | 6/10 | 5/10 |
| mathverse-algebra | 8/10 | 7/10 | 8/10 | 7/10 | 6/10 |
| mathverse-complex | 7/10 | 9/10 | 7/10 | 5/10 | 5/10 |
| mathverse-special | 5/10 | 9/10 | 8/10 | 7/10 | 5/10 |

### Critical Issues
1. **mathverse-special has 3 failing tests** — `bessel_y1`, `digamma`, and `zeta` all have precision or correctness bugs that need fixing before release.
2. **mathverse-complex has undocumented branch cut conventions** for `log` and `sqrt` — this can cause silent errors in downstream code.
3. **mathverse-algebra's `systems.rs` uses Gaussian elimination without pivoting** — numerically unstable for ill-conditioned systems.
4. **mathverse-core's `factorial` overflows at 35!** — Python uses arbitrary precision, Rust's `u128` caps at 34!.

### Positive Highlights
- No `unsafe` code in any of the 5 crates.
- Excellent documentation with working examples on all public APIs.
- Good use of property-based testing in mathverse-core.
- Clean, modular architecture with clear separation of concerns.