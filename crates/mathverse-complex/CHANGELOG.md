# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-07

### Added

- `Complex` is now generic over `T: RealFull` (defaults to `f64`); new `C32`/`C64` type aliases
- `fft` module: radix-2 Cooley–Tukey `fft`/`ifft` (power-of-two lengths)
- `polynomial` module: `polynomial_roots` (Durand–Kerner) and `eval_polynomial`
- `matrix` module: `eigenvalues` (QR algorithm with Wilkinson shifts),
  `qr_decomposition`, `linalg` submodule (`eig`, `expm`, `logm`)
- `Complex::phase`, `Complex::to_polar`/`polar`, `Complex::rect`, `Complex::is_close`
  (numpy/cmath parity), `real()`/`imag()` aliases
- Scalar arithmetic: `Complex<T> ± T`, `Complex<T> · T`, `Complex<T> / T`
- `mandelbrot_iterate`, `mandelbrot_smooth` helpers
- `analysis`: `derivative_complex_step` (complex-step differentiation),
  `derivative_cauchy` (Cauchy integral formula)
- `ComplexMatrix::try_get`/`try_set` non-panicking accessors
- Integration tests (`tests/`), Criterion benchmarks (`benches/`), runnable examples (`examples/`)

### Changed

- `sqrt` rewritten with the algebraically stable formula; full precision near
  the real axis and a well-defined branch cut on `(−∞, 0]`
- Complex division now uses Smith's overflow-safe algorithm (was:
  `inf/inf = NaN` for magnitudes near `1e154`)
- `erf`/`erfc` use an optimally-truncated asymptotic expansion for `|z| ≥ 3`
- `bessel_j`/`bessel_y` series stop at convergence (early-exit) and fall back to
  a longer series outside the asymptotic sector
- `polylog` restricted to cases where the series/inversion formulas are valid;
  unsupported arguments return `NaN` rather than silently wrong values
- `zeta` test tolerances tightened (functional-equation path verified)
- `ComplexMatrix::add`/`sub`/`mul` now return `MathResult`, `qr_decomposition`
  returns `MathResult`, `lu_decomposition` uses a relative pivot threshold
- `atanh` branch-cut convention corrected to match C99/IEEE principal values
- README rewritten for the new API surface

### Fixed

- Overflow-safe `recip`/`Div` (division no longer returns `NaN` for large magnitudes)
- `0⁰` returns `1` (documented combinatorial convention) instead of `NaN`
- `acosh` single-square-root formula (removed duplicate `sqrt` cancellation path)
- QR eigenvalue iteration handling of singular shifts and dependent columns
- `sqrt` precision for `z = −4 + εi` with tiny `ε`

## [0.1.2] - 2026-08-01

### Added

- Initial `mathverse-complex` release: `Complex` over `f64`, complex analysis,
  special functions, complex matrix algebra.

[0.2.0]: https://github.com/Rohithdgrr/Rust-math/releases/tag/v0.2.0
[0.1.2]: https://github.com/Rohithdgrr/Rust-math/releases/tag/v0.1.2
