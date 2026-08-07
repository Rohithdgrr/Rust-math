//! Floating-point comparison and rounding utilities.
//!
//! This module provides tools for safe floating-point arithmetic, including
//! multiple comparison strategies (absolute, relative, ULP-based) and
//! rounding utilities.

use crate::traits::{Real, Transcendental};

/// Machine epsilon for `f64` (2^-52).
pub const EPS: f64 = 2.220446049250313e-16;
/// Machine epsilon for `f32` (2^-23).
pub const F32_EPS: f32 = 1.1920929e-7;

/// Absolute-tolerance equality: `|a - b| <= eps`.
///
/// Prefer [`almost_eq_rel`] when operands span many orders of magnitude.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::almost_eq;
///
/// assert!(almost_eq(0.1 + 0.2, 0.3, 1e-15));
/// assert!(!almost_eq(0.1 + 0.2, 0.3, 1e-17));
/// ```
#[must_use]
#[inline]
pub fn almost_eq<T: Real>(a: T, b: T, eps: T) -> bool {
    (a - b).abs() <= eps
}

/// Relative-tolerance equality: `|a - b| <= tol * max(|a|, |b|)`.
/// Equal when both are zero.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::almost_eq_rel;
///
/// assert!(almost_eq_rel(1e15, 1e15 + 1.0, 1e-12));
/// assert!(almost_eq_rel(0.0, 0.0, 1e-10));
/// ```
#[must_use]
#[inline]
pub fn almost_eq_rel<T: Real>(a: T, b: T, tol: T) -> bool {
    let m = a.abs().max(b.abs());
    m == T::zero() || (a - b).abs() <= tol * m
}

/// Combined absolute + relative tolerance comparison.
///
/// Returns `true` when `|a - b| <= max(abs_tol, rel_tol * max(|a|, |b|))`.
/// This is the recommended general-purpose float comparison.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::is_close;
///
/// assert!(is_close(1.0, 1.0 + 1e-16, 1e-12, 1e-15));
/// assert!(!is_close(1.0, 2.0, 1e-12, 1e-15));
/// assert!(is_close(0.0, 0.0, 1e-12, 1e-15));
/// ```
#[must_use]
#[inline]
pub fn is_close<T: Real>(a: T, b: T, rel_tol: T, abs_tol: T) -> bool {
    let diff = (a - b).abs();
    let m = a.abs().max(b.abs());
    diff <= abs_tol || diff <= rel_tol * m
}

/// ULP-based floating-point equality for `f64`.
///
/// Returns `true` if `a` and `b` differ by at most `max_ulp` representable
/// values. Handles signed zeros and infinities correctly.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::almost_eq_ulp;
///
/// assert!(almost_eq_ulp(1.0, 1.0, 0));
/// assert!(almost_eq_ulp(f64::INFINITY, f64::INFINITY, 0));
/// assert!(!almost_eq_ulp(f64::NAN, f64::NAN, 0));
/// ```
#[must_use]
pub fn almost_eq_ulp(a: f64, b: f64, max_ulp: i64) -> bool {
    if a == b {
        return true;
    }
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    let bits_a = a.to_bits();
    let bits_b = b.to_bits();
    let diff = (bits_a as i64 - bits_b as i64).abs();
    diff <= max_ulp
}

/// Check if `f64` is subnormal (denormal).
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::is_subnormal;
///
/// assert!(is_subnormal(f64::MIN_POSITIVE / 2.0));
/// assert!(!is_subnormal(1.0));
/// ```
#[must_use]
#[inline]
pub fn is_subnormal(x: f64) -> bool {
    x.is_finite() && x != 0.0 && x.abs() < f64::MIN_POSITIVE
}

/// Check if `f64` is NaN.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::is_nan;
///
/// assert!(is_nan(f64::NAN));
/// assert!(!is_nan(1.0));
/// ```
#[must_use]
#[inline]
pub const fn is_nan(x: f64) -> bool {
    x.is_nan()
}

/// Check if `f64` is infinite (positive or negative).
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::is_infinite;
///
/// assert!(is_infinite(f64::INFINITY));
/// assert!(!is_infinite(1.0));
/// ```
#[must_use]
#[inline]
pub const fn is_infinite(x: f64) -> bool {
    x.is_infinite()
}

/// Machine epsilon for a generic `Real` type.
///
/// Returns the difference between 1.0 and the next representable value.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::{epsilon, EPS, almost_eq};
///
/// assert!(almost_eq(epsilon::<f64>(), EPS, 1e-20));
/// ```
#[must_use]
#[inline]
pub fn epsilon<T: Real>() -> T {
    T::epsilon()
}

/// Next representable `f64` toward positive infinity.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::next_float;
///
/// let next = next_float(1.0);
/// assert!(next > 1.0);
/// ```
#[must_use]
pub fn next_float(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    let bits = x.to_bits();
    if x.is_sign_negative() && x == 0.0 {
        f64::from_bits(1)
    } else if x >= 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Next representable `f64` toward negative infinity.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::{next_float, prev_float};
///
/// let next = next_float(1.0);
/// assert_eq!(prev_float(next), 1.0);
/// ```
#[must_use]
pub fn prev_float(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let bits = x.to_bits();
    if x.is_sign_positive() && x == 0.0 {
        f64::from_bits(0x8000_0000_0000_0001)
    } else if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// Copy the sign of `b` to the magnitude of `a`.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::copysign;
///
/// assert_eq!(copysign(3.0, -1.0), -3.0);
/// assert_eq!(copysign(3.0, 1.0), 3.0);
/// ```
#[must_use]
#[inline]
pub fn copysign<T: Real>(a: T, b: T) -> T {
    if b.is_negative() {
        -a.abs()
    } else {
        a.abs()
    }
}

/// Absolute difference: `|a - b|`.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::abs_diff;
///
/// assert_eq!(abs_diff(5.0, 3.0), 2.0);
/// assert_eq!(abs_diff(3.0, 5.0), 2.0);
/// ```
#[must_use]
#[inline]
pub fn abs_diff<T: Real>(a: T, b: T) -> T {
    (a - b).abs()
}

/// Relative difference: `|a - b| / max(|a|, |b|)`.
///
/// Returns 0 when both are zero.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::relative_diff;
///
/// assert!((relative_diff(10.0_f64, 9.0) - 0.1).abs() < 1e-12);
/// assert_eq!(relative_diff(0.0, 0.0), 0.0);
/// ```
#[must_use]
#[inline]
pub fn relative_diff<T: Real>(a: T, b: T) -> T {
    let diff = (a - b).abs();
    let m = a.abs().max(b.abs());
    if m == T::zero() {
        T::zero()
    } else {
        diff / m
    }
}

/// Round `x` to `decimals` decimal places.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::round_to;
///
/// assert_eq!(round_to(2.345678, 3), 2.346);
/// assert_eq!(round_to(-1.234, 2), -1.23);
/// ```
#[must_use]
pub fn round_to<T: Real>(x: T, decimals: i32) -> T {
    let f = T::from_f64(pow10(decimals));
    (x * f).round() / f
}

fn pow10(n: i32) -> f64 {
    let mut acc = 1.0;
    for _ in 0..n.unsigned_abs() {
        acc *= 10.0;
    }
    if n < 0 { 1.0 / acc } else { acc }
}

/// Round `x` to `n` significant figures.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::significant_figures;
///
/// assert_eq!(significant_figures(0.00234567, 3), 0.00235);
/// assert_eq!(significant_figures(12345.0, 2), 12000.0);
/// ```
#[must_use]
#[allow(clippy::cast_lossless)] // From<i32> for f64 is unavailable in no_std
pub fn significant_figures<T: Real + Transcendental>(x: T, n: i32) -> T {
    if x == T::zero() {
        return T::zero();
    }
    let shift = T::from_f64((n - 1) as f64) - x.abs().log10().floor();
    round_to(x, shift.to_f64() as i32)
}

/// Distance from `x` to the next representable `f64` toward +infinity.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::ulp;
///
/// assert!(ulp(1.0) < 1e-15);
/// assert!(ulp(-1.0) < 1e-15);
/// ```
#[must_use]
pub fn ulp(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return f64::MIN_POSITIVE;
    }
    let b = x.to_bits();
    let b = if x.is_sign_negative() {
        b.wrapping_sub(1)
    } else {
        b.wrapping_add(1)
    };
    f64::from_bits(b) - x
}

/// Round `x` up to the nearest multiple of `m`.
///
/// Returns `x` if `m` is zero.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::ceil_to_multiple;
///
/// assert_eq!(ceil_to_multiple(7.0, 5.0), 10.0);
/// assert_eq!(ceil_to_multiple(10.0, 5.0), 10.0);
/// assert_eq!(ceil_to_multiple(0.0, 3.0), 0.0);
/// ```
#[must_use]
pub fn ceil_to_multiple<T: Real>(x: T, m: T) -> T {
    if m == T::zero() {
        return x;
    }
    (x / m).ceil() * m
}

/// Round `x` down to the nearest multiple of `m`.
///
/// Returns `x` if `m` is zero.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::floor_to_multiple;
///
/// assert_eq!(floor_to_multiple(7.0, 5.0), 5.0);
/// assert_eq!(floor_to_multiple(10.0, 5.0), 10.0);
/// assert_eq!(floor_to_multiple(0.0, 3.0), 0.0);
/// ```
#[must_use]
pub fn floor_to_multiple<T: Real>(x: T, m: T) -> T {
    if m == T::zero() {
        return x;
    }
    (x / m).floor() * m
}

/// Round `x` to the nearest multiple of `m`.
///
/// Returns `x` if `m` is zero.
///
/// # Examples
///
/// ```
/// use mathverse_core::precision::round_to_multiple;
///
/// assert_eq!(round_to_multiple(7.0, 5.0), 5.0);
/// assert_eq!(round_to_multiple(8.0, 5.0), 10.0);
/// assert_eq!(round_to_multiple(0.0, 3.0), 0.0);
/// ```
#[must_use]
pub fn round_to_multiple<T: Real>(x: T, m: T) -> T {
    if m == T::zero() {
        return x;
    }
    (x / m).round() * m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparisons() {
        assert!(almost_eq(0.1 + 0.2, 0.3, 1e-15));
        assert!(!almost_eq(0.1 + 0.2, 0.3, 1e-17));
        assert!(almost_eq_rel(1e15, 1e15 + 1.0, 1e-12));
        assert!(almost_eq_rel(0.0, 0.0, 1e-10));
    }

    #[test]
    fn rounding() {
        assert_eq!(round_to(3.14159, 2), 3.14);
        assert_eq!(significant_figures(12345.0, 2), 12000.0);
        assert_eq!(significant_figures(0.0, 3), 0.0);
    }

    #[test]
    fn ulp_is_small() {
        assert!(ulp(1.0) < 1e-15);
        assert!(ulp(-1.0) < 1e-15);
    }

    #[test]
    fn is_close_combined() {
        assert!(is_close(1.0, 1.0 + 1e-16, 1e-12, 1e-15));
        assert!(!is_close(1.0, 2.0, 1e-12, 1e-15));
        assert!(is_close(0.0, 0.0, 1e-12, 1e-15));
        assert!(is_close(0.0, 1e-20, 1e-12, 1e-15));
    }

    #[test]
    fn ulp_comparison() {
        assert!(almost_eq_ulp(1.0, 1.0, 0));
        assert!(almost_eq_ulp(1.0, 1.0 + 1e-15, 5));
        assert!(!almost_eq_ulp(1.0, 2.0, 100));
        assert!(almost_eq_ulp(f64::INFINITY, f64::INFINITY, 0));
        assert!(!almost_eq_ulp(f64::NAN, f64::NAN, 0));
    }

    #[test]
    fn float_helpers() {
        assert!(is_subnormal(f64::MIN_POSITIVE / 2.0));
        assert!(!is_subnormal(1.0));
        assert!(is_nan(f64::NAN));
        assert!(!is_nan(1.0));
        assert!(is_infinite(f64::INFINITY));
        assert!(!is_infinite(1.0));
        assert!(almost_eq(epsilon::<f64>(), EPS, 1e-20));
    }

    #[test]
    fn next_prev_float() {
        let next = next_float(1.0);
        assert!(next > 1.0);
        assert_eq!(prev_float(next), 1.0);
        assert_eq!(next_float(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn copysign_abs_diff() {
        assert_eq!(copysign(3.0, -1.0), -3.0);
        assert_eq!(copysign(3.0, 1.0), 3.0);
        assert_eq!(abs_diff(5.0, 3.0), 2.0);
        assert_eq!(relative_diff(10.0, 9.0), 0.1);
        assert_eq!(relative_diff(0.0, 0.0), 0.0);
    }

    #[test]
    fn multiple_rounding() {
        assert_eq!(ceil_to_multiple(7.0, 5.0), 10.0);
        assert_eq!(ceil_to_multiple(10.0, 5.0), 10.0);
        assert_eq!(ceil_to_multiple(0.0, 3.0), 0.0);
        assert_eq!(floor_to_multiple(7.0, 5.0), 5.0);
        assert_eq!(floor_to_multiple(10.0, 5.0), 10.0);
        assert_eq!(floor_to_multiple(0.0, 3.0), 0.0);
        assert_eq!(round_to_multiple(7.0, 5.0), 5.0);
        assert_eq!(round_to_multiple(8.0, 5.0), 10.0);
        assert_eq!(round_to_multiple(0.0, 3.0), 0.0);
    }
}
