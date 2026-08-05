//! Float math helpers that work under both `std` and `no_std`.
//!
//! Under `std` these delegate to the hardware-accelerated inherent methods.
//! Under `no_std` (with the `libm` feature) they delegate to the `libm`
//! software implementations.

#[cfg(feature = "std")]
#[inline]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn powi(x: f64, n: i32) -> f64 {
    x.powi(n)
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub fn powi(x: f64, n: i32) -> f64 {
    libm::pow(x, f64::from(n))
}

#[cfg(feature = "std")]
#[inline]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub fn floor(x: f64) -> f64 {
    libm::floor(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub fn ceil(x: f64) -> f64 {
    libm::ceil(x)
}

#[cfg(feature = "std")]
#[inline]
pub fn round(x: f64) -> f64 {
    x.round()
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
pub fn round(x: f64) -> f64 {
    libm::round(x)
}
