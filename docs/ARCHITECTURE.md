# MathVerse Architecture

## Repository Layout

Single Cargo workspace with one crate per domain (see README crate table). Each
crate is independently usable and versioned; `mathverse-prelude` re-exports the
full public API.

## Dependencies Between Crates

- All crates depend only on `mathverse-core` (never on each other, so domains
  compose without cycles).
- `mathverse-core` has zero MathVerse dependencies and minimal external deps.
- `mathverse-prelude` depends on all crates.

## Core Abstractions (`mathverse-core`)

- **Traits**: numeric abstractions shared across all crates (field, ring, real,
  complex, etc. — built on std `num` conventions where they exist).
- **Error handling**: one error taxonomy with rich messages and
  `std::error::Error` + `no_std` support.
- **Constants**: high-precision mathematical constants.
- **Precision utilities**: epsilon comparison, tolerance-based equality, rounding
  helpers.
- **Common algorithms**: shared kernels used by multiple domains.

## Performance

- SIMD acceleration behind feature flags (`simd`).
- Rayon parallelism behind feature flags (`parallel`).
- Optional GPU backends (long-term).
- Cache-friendly algorithms, zero allocations where possible.
- `no_std` support for applicable modules.

## Feature Flags

Every crate exposes features consistently:

| Flag | Default | Description |
|------|---------|-------------|
| `std` | Yes | Enables std; disabling gives `no_std` |
| `simd` | No | SIMD-accelerated hot paths |
| `parallel` | No | Rayon-based parallelism |
| `gpu` | No | GPU backends (long-term, opt-in) |
| `serde` | No | Serialization derives |

## Testing Strategy

- Unit tests for every function.
- Property tests (`proptest`).
- Numerical accuracy tests against reference values.
- Fuzz testing on parsers and converters.
- Performance benchmarks (`criterion`) with regression tracking.
- Cross-platform CI (Windows, macOS, Linux; `wasm32` target).
- Target coverage: **95%+**.
