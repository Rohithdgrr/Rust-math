//! Internal utility functions for the trigonometry crate.

use mathverse_core::traits::Real;

/// Helper function to map generic Real types through f64 operations.
/// This is used throughout the crate to provide generic implementations
/// that work with both f32 and f64 by converting to f64, computing, and converting back.
pub fn map_real<T: Real>(x: T, f: impl Fn(f64) -> f64) -> T {
    T::from_f64(f(x.to_f64()))
}
