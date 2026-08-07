//! Internal floating-point helpers that work under both `std` and `no_std`.
//!
//! Under `std` these delegate to the standard library's hardware-accelerated
//! methods. Under `no_std` (with the `libm` feature enabled) they delegate to
//! `mathverse-core`'s `libm`-backed fallbacks, which ship with the same API.

#[cfg(feature = "std")]
pub(crate) fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(feature = "std")]
pub(crate) fn floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(feature = "std")]
pub(crate) fn ceil(x: f64) -> f64 {
    x.ceil()
}

#[cfg(not(feature = "std"))]
pub(crate) use mathverse_core::libm_fallback::{ceil, floor, sqrt};
