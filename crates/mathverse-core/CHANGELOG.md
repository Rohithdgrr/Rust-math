# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-01

### Added
- `#[must_use]` on all pure functions across all modules
- `#[inline]` hints on trivial leaf functions
- `const fn` where possible (`is_power_of_two`, `mersenne_number`, `fermat_number`, `next_power_of_two`, `msg`, `has_context`)
- `#![deny(missing_docs)]` — all public items now have doc comments
- `#![warn(clippy::all, clippy::pedantic, clippy::nursery)]`
- `MathError::msg()` method for human-readable error descriptions
- `MathError::has_context()` to check if variant carries custom message
- `impl From<MathError> for String` for ergonomic error conversion
- `#[non_exhaustive]` on `MathError` for future-proofing
- Comprehensive doc-tests on every public function (100+ examples)
- Property-based tests via `proptest` (17 test strategies)
- `criterion` benchmarks for hot paths (GCD, factorial, sieve, ops)
- MSRV declaration (`rust-version = "1.87"`)
- `CHANGELOG.md`
- CI workflow (`.github/workflows/ci.yml`)
- Updated README with error handling examples, MSRV, roadmap

### Changed
- Module-level doc comments expanded across all files
- `constants.rs` now has per-constant doc comments
- `error.rs` Display impl improved with cleaner messages
- MSRV bumped from 1.70 to 1.87 (required for `u64::midpoint`, `is_multiple_of`, const `&self` methods)
- All `as` casts replaced with `From`/`Into` where infallible
- Replaced manual `% == 0` with `.is_multiple_of()`, `(x+1)/2` with `.div_ceil()`

### Fixed
- `MathError` variants `InvalidArgument` and `NotConverged` now consistently use `&'static str`
- All clippy warnings resolved (0 warnings with `-D warnings`)

## [0.1.0] - 2026-07-15

### Added
- Initial release
- Core numeric traits (`Num`, `Signed`, `Field`, `Real`)
- 30+ scalar operations (lerp, smoothstep, hypot, wrap, angle conversions)
- Float precision utilities (ULP, relative/absolute tolerances)
- 50+ math constants
- Number theory algorithms (GCD, LCM, primes, factorials, Fibonacci, Bell)
- Combinatorics (Catalan, Stirling, partitions, Pascal's triangle)
- Error types (`MathError`, `MathResult<T>`)
- Prelude module
- `no_std` support
