# Architecture — `mathverse-trigonometry`

## Purpose

A commutative, dependency-light trigonometry crate for the MathVerse workspace.
It exposes forward/inverse circular and hyperbolic functions, degree variants,
identities, geometric laws, special functions, coordinate conversions, batched
(DSP) helpers, and exact special-angle values — all generic over the `Real`
trait so they compile for `f32` and `f64`.

## Module boundaries

```text
lib.rs            re-exports + top-level scalar functions (sin..csch, *_deg, *_checked)
  ├── conversions   angle normalize/wrap, turn/grad scalers, coordinate systems, phase utils
  ├── exact        closed-form sin/cos/tan for multiples of 30°/45° via ExactValue
  ├── identities   double/half/sum/difference, product-to-sum, power reduction
  ├── laws         law of sines/cosines, Heron, bearing, haversine (rad + deg)
  ├── special      sinc, versine family, gudermannian, chebyshev, sin_power/cos_power
  └── batched      slice map/sum, in-place sin, additive synthesis (no alloc)
```

`lib.rs` is the only crate surface you should import from in normal use; the
other modules are `pub mod` but their items are also re-exported at the crate
root. Nothing else is `pub`.

## Data flow

1. A `T: Real + <Trig|Hyperbolic>` scalar enters a free function (e.g. `sin`).
2. It delegates to the `Real`/`Trig`/`Hyperbolic` trait methods, which for
   standard builds use hardware math and for `no_std` route through
   `mathverse_core`'s `libm_fallback` (software libm).
3. Generic functions that need extra numeric care branch on the `Real` ops:
   `tan_half` selects `s/(1+c)` vs `(1-c)/s`; `sind`/`tand` reduce mod 360 first;
   `sinpi`/`cospi` snap to exact values near integers/half-integers.
4. Domain-optional functions (`acosh_checked` etc.) return
   `Option<T>`/`None` or `NaN` on out-of-domain input.

## Dependency graph

```text
mathverse-trigonometry  ->  mathverse-core (traits, ops)      [only dependency]
standalone touches:  mathverse-core  ->  (libm, optional std)
```

No other MathVerse crates are required by this one. Benchmarks live in the
workspace `mathverse-benches` crate and depend on this crate only as a
dev-dependency.

## Error handling

- Functions return `T` and signal invalidity with `NaN`/`±inf` (matching std),
  except the explicit `*_checked` variants that return `Option`.
- No panics on domain errors; division mirrors std (`1/0 = inf`).
- There is no bespoke error enum; this crate deliberately reuses IEEE-754
  signaling to keep the API crate-shaped.

## `no_std` strategy

- `#![cfg_attr(not(feature = "std"), no_std)]` at the crate root.
- Default `std` feature; the `libm` feature pulls `mathverse-core/libm`.
- All intra-crate math goes through the `Real` trait (never raw `f64` methods),
  so the scalar core is `no_std`-clean; element-wise values in `batched` are
  generic over `Real`.
- Verified by CI: `cargo check -p mathverse-trigonometry --no-default-features
  --features libm`.

## Testing

- Unit tests: `#[cfg(test)] mod` per module (`cargo test`).
- Integration: `tests/properties.rs` — dependency-free property-style checks
  (deterministic LCG, no proptest).
- Doc tests: examples in `///` blocks.
- Benchmarks: `mathverse-benches/benches/trig_bench.rs` (Criterion).
- Target: CI runs `fmt --check`, `clippy -- -D warnings`, `test`, `doc`, and
  the `no_std` build.

## Compatibility / stability

- MSRV: Rust 1.87 (workspace-wide).
- The public re-export list is the committed API; treat additions to it as
  semver-minor and removals/renames as semver-major (see `CHANGELOG.md`).
- Because the crate builds on the `Real` trait, changes to `mathverse-core`
  traits can ripple here; coordinate published cuts across both.