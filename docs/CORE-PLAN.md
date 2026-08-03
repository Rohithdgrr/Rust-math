# mathverse-core — Plan & Status

The shared substrate every other crate builds on. No domain math — only traits, errors, constants, and generic utilities.

## 1. Numeric Traits (`traits.rs`)

| Trait | Purpose | Status |
|-------|---------|--------|
| `Num` | zero/one/from_i64 + std ops (basis of everything) | Complete |
| `Signed` | `abs`, `signum`, `is_negative` | Complete |
| `Field` | division, `reciprocal` | Complete |
| `Real` | from_f64, sqrt, powf/powi, exp/ln/log, trig, floor/ceil/round, min/max, is_finite/is_nan | Complete |

All implemented for `f32`, `f64`, `i8..i64`, `u8..u64`, `isize`, `usize` (Real: `f32`/`f64` only).

## 2. Error Handling (`error.rs`)

| Item | Status |
|------|--------|
| `MathError` enum: Domain, DivisionByZero, InvalidArgument, NotConverged, DimensionMismatch, Overflow, Underflow, Singular | Complete |
| `MathResult<T>` alias | Complete |
| `Display` impls with context | Complete |
| `std::error::Error` (feature-gated) | Complete |

## 3. Constants (`constants.rs`)

`PI`, `TAU`, `E`, `PHI`, `SQRT_2`, `SQRT_3`, `LN_2`, `LN_10`, `EULER_GAMMA`, `CATALAN`, `APERY`, `DEG_TO_RAD`, `RAD_TO_DEG` — `f64` and `f32` — Complete.

## 4. Precision Utilities (`precision.rs`)

`almost_eq`, `almost_eq_rel`, `round_to`, `significant_figures`, `ulp`, `EPS`, `F32_EPS` — Complete.

## 5. Generic Operations (`ops.rs`)

`clamp`, `lerp`, `smoothstep`, `fract`, `nth_root`, `hypot2`, `sum`, `product`, `deg_to_rad`, `rad_to_deg`, `grad_to_deg`, `deg_to_grad` — Complete.

## 6. Common Algorithms (`algorithms.rs`)

`gcd`, `lcm`, `mod_pow`, `factorial`, `binomial`, `fibonacci`, `is_prime`, `next_power_of_two`, `is_power_of_two`, `sieve_of_eratosthenes` — Complete.

## 7. Infrastructure

- Workspace root `Cargo.toml` — Complete
- `Cargo.toml`: `no_std`-capable, features `std` (default) — Complete
- `lib.rs`: `#![forbid(unsafe_code)]`, module tree — Complete
- `prelude.rs` — Complete
- Unit tests per module (doc-tests + inline) — Complete
- `proptest` property tests — Pending (when coverage target enforced)
- Benchmarks (`criterion`) — Pending (after v0.1 API stabilizes)

## 8. Explicitly out of core

- Domain math (trig, matrices, etc. live in their crates)
- Allocation-heavy structures (`Vec` only where a list is the return value)
