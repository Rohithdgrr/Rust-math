# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `sinpi` / `cospi`: exact results at integer and half-integer arguments (numpy/C99 parity).
- `sind` / `cosd` / `tand` / `sin_cos_deg`: degree-based circular functions with argument
  reduction modulo 360 before conversion.
- Domain-checked (returning `Option<T>`) variants of every inverse hyperbolic function:
  `acosh_checked`, `atanh_checked`, `acoth_checked`, `asech_checked`, `acsch_checked`.
- `angle_difference` / `angle_distance` / `unwrap_angles`: signed difference, minimal
  distance, and phase unwrapping utilities.
- `haversine_distance_deg`: haversine distance with degree lat/lon inputs.
- `sin_power` / `cos_power`: power-reduction series using `powi`.

### Changed

- `tan_half` uses a numerically stable branch (`s/(1 + c)` vs `(1 - c)/s`) to avoid
  cancellation near the poles.

### Fixed

- Inverse hyperbolic functions (`acoth`, `asech`, `acsch`) now return `NaN` (and their
  `*_checked` forms return `None`) for inputs outside their domains, instead of
  propagating meaningless values.
- Unblocked the `no_std` build (`cargo check --no-default-features --features libm`) on
  both this crate and `mathverse-core`, which previously failed on custom toolchains
  lacking core float primitives. See `mathverse-core` for the underlying fixes.

### Other

- New dependency-free `tests/properties.rs` suite with 9 property-style integration tests
  (LCG-driven, no proptest dependency).
- `examples/basic.rs` tour of the public API.

## [0.2.0] - Initial published API

### Added

- 24 inverse + forward circular/hyperbolic functions with `*_deg` variants.
- Angle conversions, coordinate systems (polar/spherical/cylindrical), identities, laws,
  special functions, batched operations, and exact special-angle values.
- Bibliographic `Real`-generic API with `f32`/`f64` support and `no_std` plumbing.

<!-- Keep a Changelog placeholder; earlier history predates this file. -->