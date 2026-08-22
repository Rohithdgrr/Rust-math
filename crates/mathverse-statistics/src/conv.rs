//! Internal numeric conversion helpers.
//!
//! Statistical code constantly converts between integer counts (`usize`,
//! `u64`) and `f64`, and back again for indexing. Centralizing every cast in
//! this module gives each conversion a single, documented safety argument:
//!
//! * **Counts → `f64`** (`count`, `count_u64`): exact for values up to
//!   `2^53`. A sample of that size would need exabytes of memory, so no
//!   realistic input loses precision.
//! * **Floats → indices** (`to_index`, `to_index_u64`): callers must already
//!   have floored/ceiled/clamped the value into `[0, len)`; the conversion is
//!   truncation of a known non-negative integral float.

use crate::error::{MathError, MathResult};

/// Converts a sample count to `f64`.
#[inline]
#[must_use]
#[allow(clippy::cast_precision_loss)] // exact below 2^53; unreachable sample size
pub fn count(n: usize) -> f64 {
    n as f64
}

/// Converts a 64-bit count (trials, degrees-of-freedom terms) to `f64`.
#[inline]
#[must_use]
#[allow(clippy::cast_precision_loss)] // exact below 2^53; see module docs
pub fn count_u64(n: u64) -> f64 {
    n as f64
}

/// Converts an already-floored/clamped non-negative float to a slice index.
///
/// # Panics
///
/// Panics in debug builds if `x` is negative or exceeds `usize::MAX`
/// (release builds truncate, matching the pre-helper behavior).
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn to_index(x: f64) -> usize {
    debug_assert!(x >= 0.0 && x <= usize::MAX as f64);
    x as usize
}

/// Converts a reduced `u64` PRNG draw to a slice index.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn to_index_u64(x: u64) -> usize {
    x as usize
}

/// Converts a small non-negative exponent/degree to `i32` for `powi`.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn exponent(n: usize) -> i32 {
    debug_assert!(i32::try_from(n).is_ok());
    n as i32
}

/// Checked variant of [`exponent`] used where the bound is not statically
/// obvious.
#[inline]
pub fn try_exponent(n: u64) -> MathResult<i32> {
    i32::try_from(n).map_err(|_| MathError::Overflow)
}
