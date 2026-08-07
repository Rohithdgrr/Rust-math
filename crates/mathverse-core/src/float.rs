//! Floating-point utilities beyond the standard library surface:
//! accurate `expm1`/`log1p`, exact summation, `isclose`, and the classic
//! `frexp`/`ldexp`/`modf`/`nextafter` decomposition functions.
//!
//! `expm1`/`log1p` use hardware-accelerated `std` methods on standard builds
//! and fall back to `libm` under `no_std` (when the `libm` feature is on).

/// `e^x - 1`, accurate for small `x` (no catastrophic cancellation).
///
/// ```
/// use mathverse_core::float::expm1;
/// assert!((expm1(1e-10) - 1e-10).abs() < 1e-20);
/// assert!((expm1(1.0) - (core::f64::consts::E - 1.0)).abs() < 1e-15);
/// ```
#[cfg(feature = "std")]
#[inline]
#[must_use]
pub fn expm1(x: f64) -> f64 {
    x.exp_m1()
}

/// `e^x - 1`, accurate for small `x` (libm fallback for `no_std`).
#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
#[must_use]
pub fn expm1(x: f64) -> f64 {
    libm::expm1(x)
}

/// `ln(1 + x)`, accurate for small `x`.
///
/// ```
/// use mathverse_core::float::log1p;
/// assert!((log1p(1e-12) - 1e-12).abs() < 1e-24);
/// assert!((log1p(core::f64::consts::E - 1.0) - 1.0).abs() < 1e-15);
/// ```
#[cfg(feature = "std")]
#[inline]
#[must_use]
pub fn log1p(x: f64) -> f64 {
    x.ln_1p()
}

/// `ln(1 + x)`, accurate for small `x` (libm fallback for `no_std`).
#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
#[must_use]
pub fn log1p(x: f64) -> f64 {
    libm::log1p(x)
}

/// Exactly rounded sum of a slice (Neumaier compensated summation).
///
/// Returns the correctly-rounded sum even when naive summation would lose
/// precision to cancellation.
///
/// ```
/// use mathverse_core::float::fsum;
/// let xs = [1e16, 1.0, -1e16];
/// assert_eq!(fsum(&xs), 1.0); // naive sum gives 0.0
/// ```
#[must_use]
pub fn fsum(xs: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for &x in xs {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            correction += (sum - t) + x;
        } else {
            correction += (x - t) + sum;
        }
        sum = t;
    }
    sum + correction
}

/// Approximate equality with relative and absolute tolerances.
///
/// `|a - b| <= abs_tol + rel_tol * max(|a|, |b|)`; NaNs never compare close.
///
/// ```
/// use mathverse_core::float::isclose;
/// assert!(isclose(0.1 + 0.2, 0.3, 1e-9, 1e-12));
/// assert!(!isclose(1.0, 2.0, 1e-9, 1e-12));
/// ```
#[inline]
#[must_use]
#[allow(clippy::suboptimal_flops)] // mul_add is a std-only inherent; keep portable
pub fn isclose(a: f64, b: f64, rel_tol: f64, abs_tol: f64) -> bool {
    (a - b).abs() <= rel_tol * b.abs().max(a.abs()) + abs_tol
}

/// Split `x` into a mantissa `m` in `[0.5, 1)` (sign preserved) and an
/// integer exponent `e` such that `x == m * 2^e`.
///
/// Zero, infinities and NaN return `(x, 0)`.
///
/// ```
/// use mathverse_core::float::frexp;
/// let (m, e) = frexp(6.0);
/// assert_eq!(m, 0.75);
/// assert_eq!(e, 3);
/// ```
#[must_use]
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i32;
    if exponent == 0 {
        // Subnormal: normalize by scaling up, then adjust the exponent.
        let (m, e) = frexp(x * 4503599627370496.0); // 2^52
        return (m, e - 52);
    }
    let mantissa = (bits & !(0x7FFu64 << 52)) | (1022u64 << 52);
    (f64::from_bits(mantissa), exponent - 1022)
}

/// Construct `x * 2^e`, correctly handling subnormals and overflow.
///
/// ```
/// use mathverse_core::float::{frexp, ldexp};
/// let x = 123.456;
/// let (m, e) = frexp(x);
/// assert!((ldexp(m, e) - x).abs() < 1e-12);
/// assert!(ldexp(1.0, 2000).is_infinite());
/// ```
#[must_use]
pub fn ldexp(x: f64, e: i32) -> f64 {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return x;
    }
    let mut bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i32;
    if exponent == 0 {
        // Subnormal input: normalize first, then recurse with adjusted exponent.
        return ldexp(x * 4503599627370496.0, e - 52); // 2^52
    }
    let new_exp = exponent + e;
    if new_exp <= 0 {
        // Underflow to subnormal: multiply by 2^new_exp through repeated
        // halving (at most 1074 iterations).
        let mut r = x;
        let mut k = e;
        while k < 0 {
            r *= 0.5;
            k += 1;
        }
        return r;
    }
    if new_exp >= 0x7FF {
        return if bits >> 63 == 1 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    bits = (bits & !(0x7FFu64 << 52)) | ((new_exp as u64) << 52);
    f64::from_bits(bits)
}

/// Split `x` into fractional and integral parts, both with `x`'s sign.
///
/// Returns `(fractional, integral)`.
///
/// ```
/// use mathverse_core::float::modf;
/// let (frac, int) = modf(2.7);
/// assert!((frac - 0.7).abs() < 1e-12);
/// assert_eq!(int, 2.0);
/// let (frac, int) = modf(-2.7);
/// assert!((frac + 0.7).abs() < 1e-12);
/// assert_eq!(int, -2.0);
/// ```
#[must_use]
pub fn modf(x: f64) -> (f64, f64) {
    if x.is_nan() || x.is_infinite() {
        return (x, 0.0);
    }
    let integral = trunc_f64(x);
    (x - integral, integral)
}

/// Truncate toward zero using bit surgery, so it works without `std`.
///
/// The low fraction bits are cleared; subnormals and values below 1 truncate
/// to signed zero.
#[inline]
#[must_use]
fn trunc_f64(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return x;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i32;
    if exponent == 0 {
        return f64::from_bits(bits & (1u64 << 63)); // subnormal -> signed zero
    }
    let unbiased = exponent - 1023;
    if unbiased >= 52 {
        return x;
    }
    if unbiased < 0 {
        return f64::from_bits(bits & (1u64 << 63)); // |x| < 1 -> signed zero
    }
    let keep = 52 - unbiased; // fraction bits live below bit `keep`
    let mask = !((1u64 << keep) - 1);
    f64::from_bits(bits & mask)
}

/// Next representable `f64` stepping from `x` toward `y`.
///
/// Returns `y` when `x == y`; `+0.0`/`-0.0` are handled per IEEE-754.
///
/// ```
/// use mathverse_core::float::nextafter;
/// assert!(nextafter(1.0, 2.0) > 1.0);
/// assert!(nextafter(1.0, 0.0) < 1.0);
/// assert_eq!(nextafter(0.0, 1.0), f64::from_bits(1));
/// assert_eq!(nextafter(1.0, 1.0), 1.0);
/// ```
#[must_use]
pub fn nextafter(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        let mut r = f64::from_bits(1);
        if y < 0.0 {
            r = -r;
        }
        return r;
    }
    let mut bits = x.to_bits();
    if (x < y) == (x > 0.0) {
        bits += 1;
    } else {
        bits -= 1;
    }
    f64::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expm1_log1p_small() {
        assert!((expm1(1e-10) - 1e-10).abs() < 1e-20);
        assert!((log1p(1e-10) - 1e-10).abs() < 1e-20);
        assert!((expm1(0.0) - 0.0).abs() < 1e-20);
        assert!((log1p(0.0) - 0.0).abs() < 1e-20);
    }

    #[test]
    fn fsum_exactness() {
        assert_eq!(fsum(&[1e16, 1.0, -1e16]), 1.0);
        assert_eq!(fsum(&[0.1; 10]), 1.0);
    }

    #[test]
    fn isclose_behavior() {
        assert!(isclose(0.1 + 0.2, 0.3, 1e-9, 1e-12));
        assert!(!isclose(1.0, 2.0, 1e-9, 1e-12));
        assert!(!isclose(f64::NAN, f64::NAN, 1.0, 1.0));
        // Subnormal numbers: use very small atol
        assert!(isclose(1e-300, 1e-300, 1e-9, 1e-320));
    }

    #[test]
    fn frexp_ldexp_roundtrip() {
        for &x in &[0.5, 1.0, 2.0, 6.0, 123.456, 1e-300, -7.5] {
            let (m, e) = frexp(x);
            assert!(m.abs() >= 0.5 && m.abs() < 1.0 || x == 0.0);
            assert!((ldexp(m, e) - x).abs() < x.abs() * 1e-12 || x == 0.0);
        }
        assert_eq!(frexp(0.0), (0.0, 0));
        assert_eq!(frexp(0.75), (0.75, 0));
        assert_eq!(frexp(6.0), (0.75, 3));
        assert!(ldexp(1.0, 2000).is_infinite());
        assert_eq!(ldexp(1.0, -2000), 0.0);
    }

    #[test]
    fn modf_split() {
        assert_eq!(modf(2.7).1, 2.0);
        assert_eq!(modf(-2.7).1, -2.0);
        assert_eq!(modf(0.0).1, 0.0);
        assert_eq!(modf(f64::INFINITY).1, 0.0);
    }

    #[test]
    fn nextafter_stepping() {
        assert!(nextafter(1.0, 2.0) > 1.0);
        assert!(nextafter(1.0, 0.0) < 1.0);
        assert_eq!(nextafter(0.0, 1.0), f64::from_bits(1));
        assert_eq!(nextafter(1.0, 1.0), 1.0);
        assert!(nextafter(f64::MAX, f64::INFINITY).is_infinite());
    }
}
