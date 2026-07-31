//! Floating-point comparison and rounding utilities.

use crate::traits::Real;

/// Machine epsilon for `f64` (2^-52).
pub const EPS: f64 = 2.220446049250313e-16;
/// Machine epsilon for `f32` (2^-23).
pub const F32_EPS: f32 = 1.1920929e-7;

/// Absolute-tolerance equality: `|a - b| <= eps`.
///
/// Prefer [`almost_eq_rel`] when operands span many orders of magnitude.
pub fn almost_eq<T: Real>(a: T, b: T, eps: T) -> bool {
    (a - b).abs() <= eps
}

/// Relative-tolerance equality: `|a - b| <= tol * max(|a|, |b|)`.
/// Equal when both are zero.
pub fn almost_eq_rel<T: Real>(a: T, b: T, tol: T) -> bool {
    let m = a.abs().max(b.abs());
    m == T::zero() || (a - b).abs() <= tol * m
}

/// Combined absolute + relative tolerance comparison.
///
/// Returns `true` when `|a - b| <= max(abs_tol, rel_tol * max(|a|, |b|))`.
/// This is the recommended general-purpose float comparison.
pub fn is_close<T: Real>(a: T, b: T, rel_tol: T, abs_tol: T) -> bool {
    let diff = (a - b).abs();
    let m = a.abs().max(b.abs());
    diff <= abs_tol || diff <= rel_tol * m
}

/// ULP-based floating-point equality for `f64`.
///
/// Returns `true` if `a` and `b` differ by at most `max_ulp` representable
/// values. Handles signed zeros and infinities correctly.
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
pub fn is_subnormal(x: f64) -> bool {
    x.is_finite() && x != 0.0 && x.abs() < f64::MIN_POSITIVE
}

/// Check if `f64` is NaN.
pub fn is_nan(x: f64) -> bool {
    x.is_nan()
}

/// Check if `f64` is infinite (positive or negative).
pub fn is_infinite(x: f64) -> bool {
    x.is_infinite()
}

/// Machine epsilon for a generic `Real` type.
/// Returns the difference between 1.0 and the next representable value.
pub fn epsilon<T: Real>() -> T {
    T::from_f64(EPS)
}

/// Next representable `f64` toward positive infinity.
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
pub fn copysign<T: Real>(a: T, b: T) -> T {
    if b.is_negative() {
        a.abs().neg()
    } else {
        a.abs()
    }
}

/// Absolute difference: `|a - b|`.
pub fn abs_diff<T: Real>(a: T, b: T) -> T {
    (a - b).abs()
}

/// Relative difference: `|a - b| / max(|a|, |b|)`.
/// Returns 0 when both are zero.
pub fn relative_diff<T: Real>(a: T, b: T) -> T {
    let diff = (a - b).abs();
    let m = a.abs().max(b.abs());
    if m == T::zero() {
        T::zero()
    } else {
        diff / m
    }
}

/// Safe division: returns `0` when `b == 0` instead of producing NaN/inf.
pub fn safe_div<T: Real>(a: T, b: T) -> T {
    if b == T::zero() {
        T::zero()
    } else {
        a / b
    }
}

/// Round `x` to `decimals` decimal places.
///
/// ```
/// use mathverse_core::precision::round_to;
/// assert_eq!(round_to(2.345678, 3), 2.346);
/// assert_eq!(round_to(-1.234, 2), -1.23);
/// ```
pub fn round_to<T: Real>(x: T, decimals: i32) -> T {
    let f = T::from_f64(10f64.powi(decimals));
    (x * f).round() / f
}

/// Round `x` to `n` significant figures.
///
/// ```
/// use mathverse_core::precision::significant_figures;
/// assert_eq!(significant_figures(0.00234567, 3), 0.00235);
/// ```
pub fn significant_figures<T: Real>(x: T, n: i32) -> T {
    if x == T::zero() {
        return T::zero();
    }
    let shift = T::from_f64((n - 1) as f64) - x.abs().log10().floor();
    round_to(x, shift.to_f64() as i32)
}

/// Distance from `x` to the next representable `f64` toward +infinity.
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
        assert_eq!(safe_div(1.0, 0.0), 0.0);
        assert_eq!(safe_div(10.0, 2.0), 5.0);
    }
}
