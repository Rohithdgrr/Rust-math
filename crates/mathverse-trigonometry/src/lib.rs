//! Trigonometry: circular and hyperbolic functions (and inverses),
//! generic over [`Real`]. Arguments are radians; `*_deg` variants take degrees.
//!
//! Computation uses the [`Real`] trait, so `f32` and `f64` inputs compute in
//! their native precision. The crate is `no_std`-compatible: disable `std`
//! and enable `libm` to use software floating point (see the `Cargo.toml`
//! feature flags).
//!
//! Asymptotes (e.g. `tan(π/2)`, `cot(0)`) return `±inf` like std.
//!
//! | Module | Description |
//! |---|---|
//! | [`conversions`] | Angle normalization, turns/grads, coordinate systems |
//! | [`identities`] | Double/half angle, sum/difference, product-to-sum |
//! | [`laws`] | Law of sines/cosines, Heron's formula, bearing, haversine |
//! | [`special`] | sinc, versine family, Gudermannian, Chebyshev |
//! | [`batched`] | Slice-based batch trigonometry |
//! | [`exact`] | Exact values for special angles |

#![cfg_attr(not(feature = "std"), no_std)]

pub mod batched;
pub mod conversions;
pub mod exact;
pub mod identities;
pub mod laws;
pub mod special;

use mathverse_core::traits::{Real, Trig, Hyperbolic};

pub use batched::{
    accumulate_sine, map_cos, map_sin, map_sin_cos, sin_inplace, sum_cos, sum_sin, sum_sin_cos,
};
pub use conversions::{wrap_angle, wrap_angle_positive, wrap_angle_f64, rad_to_grad, grad_to_rad,
    turns_to_radians, turns_to_radians_f64, radians_to_turns, radians_to_turns_f64,
    angle_difference, angle_distance, unwrap_angles, polar_to_cartesian, cartesian_to_polar,
    magnitude, phase, spherical_to_cartesian, cartesian_to_spherical, cylindrical_to_cartesian,
    cartesian_to_cylindrical};
pub use exact::{cos_exact_deg, cos_exact_radians, sin_exact_deg, sin_exact_radians, tan_exact_deg,
    tan_exact_radians, ExactValue};
pub use identities::{sin_cos, sin_double, cos_double, tan_double, sin_half, cos_half, tan_half,
    sin_sum, sin_diff, cos_sum, cos_diff, tan_sum, tan_diff,
    sin_sin_product, cos_cos_product, sin_cos_product,
    sin_sum_to_product, sin_diff_to_product, cos_sum_to_product, cos_diff_to_product,
    sin_squared, cos_squared, tan_squared};
pub use laws::{law_of_sines_side, law_of_sines_angle, law_of_sines_angle_both, law_of_cosines_side, law_of_cosines_angle,
    heron, triangle_area_sas, triangle_area_base_height, bearing, haversine_distance, haversine_distance_deg};
pub use special::{sinc, sinc_unnorm, versine, coversine, vercosine, covercosine,
    haversine, havercosine, hacoversine, hacovercosine, exsecant, excosecant,
    gudermannian, gudermannian_inv, gudermannian_alt,
    chebyshev_first, chebyshev_second, sin_power, cos_power};

pub use mathverse_core::ops::{deg_to_grad, deg_to_rad, grad_to_deg, rad_to_deg};

/// Sine.
pub fn sin<T: Real + Trig>(x: T) -> T {
    x.sin()
}
/// Cosine.
pub fn cos<T: Real + Trig>(x: T) -> T {
    x.cos()
}
/// Tangent.
pub fn tan<T: Real + Trig>(x: T) -> T {
    x.tan()
}
/// Cotangent `cos/sin` (1/tan loses precision near asymptotes).
pub fn cot<T: Real + Trig>(x: T) -> T {
    x.cos() / x.sin()
}
/// Secant `1/cos`.
pub fn sec<T: Real + Trig>(x: T) -> T {
    T::one() / x.cos()
}
/// Cosecant `1/sin`.
pub fn csc<T: Real + Trig>(x: T) -> T {
    T::one() / x.sin()
}

/// sin(π·x) with exact results at integer and half-integer x (C99/numpy parity).
pub fn sinpi<T: Real + Trig>(x: T) -> T {
    let xf = x.to_f64();
    let n2 = (xf * 2.0).round(); // nearest integer odd/even multiple
    if (xf * 2.0 - n2).abs() < 1e-6 {
        if (n2 as i64) % 2 == 0 {
            T::zero() // x integer
        } else if (n2 as i64).rem_euclid(4) == 1 {
            T::one() // x = k + 0.5, k even
        } else {
            -T::one() // x = k + 0.5, k odd
        }
    } else {
        (x * T::from_f64(core::f64::consts::PI)).sin()
    }
}

/// cos(π·x) with exact results at integer and half-integer x (numpy parity).
pub fn cospi<T: Real + Trig>(x: T) -> T {
    let xf = x.to_f64();
    let n2 = (xf * 2.0).round();
    if (xf * 2.0 - n2).abs() < 1e-6 {
        match (n2 as i64).rem_euclid(4) {
            0 => T::one(),       // integer, even
            2 => -T::one(),      // integer, odd
            _ => T::zero(),      // half-integer
        }
    } else {
        (x * T::from_f64(core::f64::consts::PI)).cos()
    }
}

/// Hyperbolic sine.
pub fn sinh<T: Real + Hyperbolic>(x: T) -> T {
    x.sinh()
}
/// Hyperbolic cosine.
pub fn cosh<T: Real + Hyperbolic>(x: T) -> T {
    x.cosh()
}
/// Hyperbolic tangent.
pub fn tanh<T: Real + Hyperbolic>(x: T) -> T {
    x.tanh()
}
/// Hyperbolic cotangent.
pub fn coth<T: Real + Hyperbolic>(x: T) -> T {
    x.cosh() / x.sinh()
}
/// Hyperbolic secant.
pub fn sech<T: Real + Hyperbolic>(x: T) -> T {
    T::one() / x.cosh()
}
/// Hyperbolic cosecant.
pub fn csch<T: Real + Hyperbolic>(x: T) -> T {
    T::one() / x.sinh()
}

/// Arc sine.
pub fn asin<T: Real + Trig>(x: T) -> T {
    x.asin()
}
/// Arc cosine.
pub fn acos<T: Real + Trig>(x: T) -> T {
    x.acos()
}
/// Arc tangent.
pub fn atan<T: Real + Trig>(x: T) -> T {
    x.atan()
}
/// Four-quadrant arc tangent: angle of the point `(y, x)`.
pub fn atan2<T: Real + Trig>(y: T, x: T) -> T {
    y.atan2(x)
}
/// Arc cotangent: `π/2 - atan(x)` (valid for all x).
pub fn acot<T: Real + Trig>(x: T) -> T {
    T::from_f64(core::f64::consts::FRAC_PI_2) - x.atan()
}
/// Arc secant: `acos(1/x)`. Domain: |x| >= 1.
pub fn asec<T: Real + Trig>(x: T) -> T {
    if x.abs() < T::one() {
        return T::from_f64(f64::NAN);
    }
    (T::one() / x).acos()
}
/// Arc secant with domain checking: returns `None` for |x| < 1.
pub fn asec_checked<T: Real + Trig>(x: T) -> Option<T> {
    if x.abs() < T::one() {
        None
    } else {
        Some((T::one() / x).acos())
    }
}
/// Arc cosecant: `asin(1/x)`. Domain: |x| >= 1.
pub fn acsc<T: Real + Trig>(x: T) -> T {
    if x.abs() < T::one() {
        return T::from_f64(f64::NAN);
    }
    (T::one() / x).asin()
}
/// Arc cosecant with domain checking: returns `None` for |x| < 1.
pub fn acsc_checked<T: Real + Trig>(x: T) -> Option<T> {
    if x.abs() < T::one() {
        None
    } else {
        Some((T::one() / x).asin())
    }
}

/// Inverse hyperbolic sine.
pub fn asinh<T: Real + Hyperbolic>(x: T) -> T {
    x.asinh()
}
/// Inverse hyperbolic cosine with domain checking: returns `None` for x < 1.
pub fn acosh_checked<T: Real + Hyperbolic>(x: T) -> Option<T> {
    if x < T::one() {
        None
    } else {
        Some(x.acosh())
    }
}
/// Inverse hyperbolic tangent with domain checking: returns `None` for |x| >= 1.
pub fn atanh_checked<T: Real + Hyperbolic>(x: T) -> Option<T> {
    if x.abs() >= T::one() {
        None
    } else {
        Some(x.atanh())
    }
}
/// Inverse hyperbolic cosine (|x| >= 1).
pub fn acosh<T: Real + Hyperbolic>(x: T) -> T {
    x.acosh()
}
/// Inverse hyperbolic tangent (|x| < 1).
pub fn atanh<T: Real + Hyperbolic>(x: T) -> T {
    x.atanh()
}
/// Inverse hyperbolic cotangent. Domain: |x| > 1.
pub fn acoth<T: Real + Hyperbolic>(x: T) -> T {
    if x.abs() <= T::one() {
        return T::from_f64(f64::NAN);
    }
    (T::one() / x).atanh()
}
/// Inverse hyperbolic cotangent with domain checking: returns `None` for |x| <= 1.
pub fn acoth_checked<T: Real + Hyperbolic>(x: T) -> Option<T> {
    if x.abs() <= T::one() {
        None
    } else {
        Some((T::one() / x).atanh())
    }
}
/// Inverse hyperbolic secant. Domain: 0 < x <= 1.
pub fn asech<T: Real + Hyperbolic>(x: T) -> T {
    if x <= T::zero() || x > T::one() {
        return T::from_f64(f64::NAN);
    }
    (T::one() / x).acosh()
}
/// Inverse hyperbolic secant with domain checking: returns `None` for x <= 0 or x > 1.
pub fn asech_checked<T: Real + Hyperbolic>(x: T) -> Option<T> {
    if x <= T::zero() || x > T::one() {
        None
    } else {
        Some((T::one() / x).acosh())
    }
}
/// Inverse hyperbolic cosecant. Domain: x != 0.
pub fn acsch<T: Real + Hyperbolic>(x: T) -> T {
    if x == T::zero() {
        return T::from_f64(f64::NAN);
    }
    (T::one() / x).asinh()
}
/// Inverse hyperbolic cosecant with domain checking: returns `None` at x == 0.
pub fn acsch_checked<T: Real + Hyperbolic>(x: T) -> Option<T> {
    if x == T::zero() {
        None
    } else {
        Some((T::one() / x).asinh())
    }
}

// ---------------------------------------------------------------------------
// Degree variants for circular functions
// ---------------------------------------------------------------------------

/// Sine of an angle in degrees.
pub fn sin_deg<T: Real + Trig>(d: T) -> T {
    sin(deg_to_rad(d))
}
/// Cosine of an angle in degrees.
pub fn cos_deg<T: Real + Trig>(d: T) -> T {
    cos(deg_to_rad(d))
}
/// Tangent of an angle in degrees.
pub fn tan_deg<T: Real + Trig>(d: T) -> T {
    tan(deg_to_rad(d))
}
/// Cotangent of an angle in degrees.
pub fn cot_deg<T: Real + Trig>(d: T) -> T {
    cot(deg_to_rad(d))
}
/// Secant of an angle in degrees.
pub fn sec_deg<T: Real + Trig>(d: T) -> T {
    sec(deg_to_rad(d))
}
/// Cosecant of an angle in degrees.
pub fn csc_deg<T: Real + Trig>(d: T) -> T {
    csc(deg_to_rad(d))
}

/// Compute `(sin, cos)` of an angle in degrees simultaneously.
#[inline]
pub fn sin_cos_deg<T: Real + Trig>(d: T) -> (T, T) {
    let rad = deg_to_rad(d);
    rad.sin_cos()
}

/// Sine of an angle in degrees, with arguments reduced mod 360 for large angles.
pub fn sind<T: Real + Trig>(d: T) -> T {
    sin(deg_to_rad(d % T::from_f64(360.0)))
}

/// Cosine of an angle in degrees, with arguments reduced mod 360 for large angles.
pub fn cosd<T: Real + Trig>(d: T) -> T {
    cos(deg_to_rad(d % T::from_f64(360.0)))
}

/// Tangent of an angle in degrees, with arguments reduced mod 360 for large angles.
pub fn tand<T: Real + Trig>(d: T) -> T {
    tan(deg_to_rad(d % T::from_f64(360.0)))
}

// ---------------------------------------------------------------------------
// Degree variants for inverse circular functions
// ---------------------------------------------------------------------------

/// Arc sine, result in degrees.
pub fn asin_deg<T: Real + Trig>(x: T) -> T {
    rad_to_deg(asin(x))
}
/// Arc cosine, result in degrees.
pub fn acos_deg<T: Real + Trig>(x: T) -> T {
    rad_to_deg(acos(x))
}
/// Arc tangent, result in degrees.
pub fn atan_deg<T: Real + Trig>(x: T) -> T {
    rad_to_deg(atan(x))
}
/// Four-quadrant arc tangent, result in degrees.
pub fn atan2_deg<T: Real + Trig>(y: T, x: T) -> T {
    rad_to_deg(atan2(y, x))
}
/// Arc cotangent, result in degrees.
pub fn acot_deg<T: Real + Trig>(x: T) -> T {
    rad_to_deg(acot(x))
}
/// Arc secant, result in degrees. Domain: |x| >= 1.
pub fn asec_deg<T: Real + Trig>(x: T) -> T {
    rad_to_deg(asec(x))
}
/// Arc cosecant, result in degrees. Domain: |x| >= 1.
pub fn acsc_deg<T: Real + Trig>(x: T) -> T {
    rad_to_deg(acsc(x))
}

// ---------------------------------------------------------------------------
// Degree variants for hyperbolic functions
// ---------------------------------------------------------------------------

/// Hyperbolic sine of an angle in degrees.
pub fn sinh_deg<T: Real + Hyperbolic>(d: T) -> T {
    sinh(deg_to_rad(d))
}
/// Hyperbolic cosine of an angle in degrees.
pub fn cosh_deg<T: Real + Hyperbolic>(d: T) -> T {
    cosh(deg_to_rad(d))
}
/// Hyperbolic tangent of an angle in degrees.
pub fn tanh_deg<T: Real + Hyperbolic>(d: T) -> T {
    tanh(deg_to_rad(d))
}
/// Hyperbolic cotangent of an angle in degrees.
pub fn coth_deg<T: Real + Hyperbolic>(d: T) -> T {
    coth(deg_to_rad(d))
}
/// Hyperbolic secant of an angle in degrees.
pub fn sech_deg<T: Real + Hyperbolic>(d: T) -> T {
    sech(deg_to_rad(d))
}
/// Hyperbolic cosecant of an angle in degrees.
pub fn csch_deg<T: Real + Hyperbolic>(d: T) -> T {
    csch(deg_to_rad(d))
}

// ---------------------------------------------------------------------------
// Degree variants for inverse hyperbolic functions
// ---------------------------------------------------------------------------

/// Inverse hyperbolic sine, result in degrees.
pub fn asinh_deg<T: Real + Hyperbolic>(x: T) -> T {
    rad_to_deg(asinh(x))
}
/// Inverse hyperbolic cosine, result in degrees.
pub fn acosh_deg<T: Real + Hyperbolic>(x: T) -> T {
    rad_to_deg(acosh(x))
}
/// Inverse hyperbolic tangent, result in degrees.
pub fn atanh_deg<T: Real + Hyperbolic>(x: T) -> T {
    rad_to_deg(atanh(x))
}
/// Inverse hyperbolic cotangent, result in degrees.
pub fn acoth_deg<T: Real + Hyperbolic>(x: T) -> T {
    rad_to_deg(acoth(x))
}
/// Inverse hyperbolic secant, result in degrees.
pub fn asech_deg<T: Real + Hyperbolic>(x: T) -> T {
    rad_to_deg(asech(x))
}
/// Inverse hyperbolic cosecant, result in degrees.
pub fn acsch_deg<T: Real + Hyperbolic>(x: T) -> T {
    rad_to_deg(acsch(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::FRAC_PI_4;

    const EPS: f64 = 1e-12;

    #[test]
    fn pythagorean_identity() {
        for x in [-1.0f64, -0.5, 0.0, 0.5, 1.0, 2.0] {
            assert!((sin(x).powi(2) + cos(x).powi(2) - 1.0).abs() < 1e-12, "x={x}");
        }
    }

    #[test]
    fn reciprocal_identities() {
        for x in [0.3f64, 1.0, 2.0] {
            assert!((sec(x) - 1.0 / cos(x)).abs() < 1e-12);
            assert!((csc(x) - 1.0 / sin(x)).abs() < 1e-12);
            assert!((cot(x) - cos(x) / sin(x)).abs() < 1e-12);
            assert!((tan(x) - sin(x) / cos(x)).abs() < 1e-12);
        }
    }

    #[test]
    fn known_values() {
        assert!((sin_deg(30.0f64) - 0.5).abs() < 1e-12);
        assert!((cos_deg(60.0f64) - 0.5).abs() < 1e-12);
        assert!((tan_deg(45.0f64) - 1.0).abs() < 1e-12);
        assert!((sin(0.0f64) - 0.0).abs() < 1e-15);
        assert!((cos(0.0f64) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn hyperbolic() {
        assert_eq!(sinh(0.0), 0.0);
        assert_eq!(cosh(0.0), 1.0);
        assert_eq!(tanh(0.0), 0.0);
        assert!((cosh(1.0f64).powi(2) - sinh(1.0).powi(2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inverse_domain_errors() {
        // Test asec domain: |x| >= 1
        assert!(asec(0.5f64).is_nan());
        assert!(asec(0.0f64).is_nan());
        assert!(asec(-0.5f64).is_nan());
        assert!(!asec(1.0f64).is_nan());
        assert!(!asec(2.0f64).is_nan());
        assert!(!asec(-1.0f64).is_nan());

        // Test acsc domain: |x| >= 1
        assert!(acsc(0.5f64).is_nan());
        assert!(acsc(0.0f64).is_nan());
        assert!(acsc(-0.5f64).is_nan());
        assert!(!acsc(1.0f64).is_nan());
        assert!(!acsc(2.0f64).is_nan());
        assert!(!acsc(-1.0f64).is_nan());
    }

    #[test]
    fn checked_variants() {
        // Test asec_checked returns None for invalid domain
        assert!(asec_checked(0.5).is_none());
        assert!(asec_checked(0.0).is_none());
        assert!(asec_checked(-0.5).is_none());

        // Test asec_checked returns Some for valid domain
        assert!(asec_checked(1.0).is_some());
        assert!(asec_checked(2.0).is_some());
        assert!(asec_checked(-1.0).is_some());

        // Test acsc_checked returns None for invalid domain
        assert!(acsc_checked(0.5).is_none());
        assert!(acsc_checked(0.0).is_none());
        assert!(acsc_checked(-0.5).is_none());

        // Test acsc_checked returns Some for valid domain
        assert!(acsc_checked(1.0).is_some());
        assert!(acsc_checked(2.0).is_some());
        assert!(acsc_checked(-1.0).is_some());

        // Verify checked variants match regular variants for valid inputs
        assert!((asec_checked(2.0f64).unwrap() - asec(2.0)).abs() < 1e-12);
        assert!((acsc_checked(2.0f64).unwrap() - acsc(2.0)).abs() < 1e-12);
    }

    #[test]
    fn hyperbolic_checked_variants() {
        // asech/asech_checked domain: 0 < x <= 1
        assert!(asech(1.0f64).abs() < 1e-12);
        assert!(asech(0.5f64) > 1.0);
        assert!(asech(2.5f64).is_nan());
        assert!(asech(0.0f64).is_nan());
        assert!(asech(-0.5f64).is_nan());
        assert!(asech_checked(0.5).is_some());
        assert!(asech_checked(2.0).is_none());
        assert!(asech_checked(-1.0).is_none());
        assert!(asech_checked(1.0).is_some());

        // acsch/checked domain: x != 0
        assert!((acsch(1.0f64) - 1.0f64.asinh()).abs() < 1e-12);
        assert!(acsch(0.0f64).is_nan());
        assert!(acsch_checked(0.0).is_none());
        assert!(acsch_checked(2.0).is_some());

        // acoth/checked domain: |x| > 1
        assert!((acoth(2.0f64) - 0.5f64.atanh()).abs() < 1e-12);
        assert!(acoth(1.0f64).is_nan());
        assert!(acoth(0.5f64).is_nan());
        assert!(acoth_checked(0.5).is_none());
        assert!(acoth_checked(1.0).is_none());
        assert!(acoth_checked(2.0).is_some());

        // acosh/atanh checked domains
        assert!(acosh_checked(0.5f64).is_none());
        assert!(acosh_checked(1.0).is_some());
        assert!(atanh_checked(1.0f64).is_none());
        assert!(atanh_checked(-1.0).is_none());
        assert!(atanh_checked(0.5).is_some());

        // checked variants agree with unchecked for valid inputs
        assert!((acoth_checked(2.0f64).unwrap() - acoth(2.0)).abs() < 1e-12);
        assert!((asech_checked(1.0f64).unwrap() - asech(1.0)).abs() < 1e-12);
        assert!((acsch_checked(2.0f64).unwrap() - acsch(2.0)).abs() < 1e-12);
    }

    #[test]
    fn nan_inf_propagation() {
        // Test NaN propagation
        assert!(sin(f64::NAN).is_nan());
        assert!(cos(f64::NAN).is_nan());
        assert!(tan(f64::NAN).is_nan());

        // Test Inf propagation
        assert!(sin(f64::INFINITY).is_nan());
        assert!(cos(f64::INFINITY).is_nan());
        assert!(tan(f64::INFINITY).is_nan());

        // Test tan at asymptotes (should diverge to a huge finite value, as
        // f64 cannot represent ±inf exactly at the asymptote)
        let pi_half = core::f64::consts::FRAC_PI_2;
        assert!(tan(pi_half).abs() > 1e14);
        assert!(tan(-pi_half).abs() > 1e14);

        // Test cot at 0 (should return ±inf)
        assert!(cot(0.0f64).is_infinite());
    }

    #[test]
    fn inverses() {
        for x in [-0.9f64, -0.3, 0.0, 0.3, 0.9] {
            assert!((asin(sin(x)) - x).abs() < 1e-12);
            assert!((acos(cos(x)) - x.abs()).abs() < 1e-12);
            assert!((atan(tan(x)) - x).abs() < 1e-12);
            assert!((asinh(sinh(x)) - x).abs() < 1e-12);
            assert!((atanh(tanh(x)) - x).abs() < 1e-12);
        }
        assert!((atan2(1.0f64, 1.0) - FRAC_PI_4).abs() < 1e-12);
        assert!((acot(1.0f64) - FRAC_PI_4).abs() < 1e-12);
        assert!((asec(2.0f64) - acos(0.5)).abs() < 1e-12);
        assert!((acsc(2.0f64) - asin(0.5)).abs() < 1e-12);
    }

    #[test]
    fn inverse_deg_test() {
        assert!((asin_deg(0.5f64) - 30.0).abs() < EPS);
        assert!((acos_deg(0.5f64) - 60.0).abs() < EPS);
        assert!((atan_deg(1.0f64) - 45.0).abs() < EPS);
        assert!((atan2_deg(1.0f64, 1.0) - 45.0).abs() < EPS);
    }

    #[test]
    fn hyperbolic_deg_test() {
        assert!((sinh_deg(0.0f64)).abs() < EPS);
        assert!((cosh_deg(0.0f64) - 1.0).abs() < EPS);
    }

    #[test]
    fn inverse_hyperbolic_deg_test() {
        assert!((asinh_deg(0.0f64)).abs() < EPS);
        assert!((acosh_deg(1.0f64)).abs() < EPS);
        assert!((atanh_deg(0.0f64)).abs() < EPS);
    }

    #[test]
    fn sinpi_cospi_sind() {
        assert_eq!(sinpi(1.0f64), 0.0);
        assert_eq!(sinpi(0.5), 1.0);
        assert_eq!(sinpi(1.5), -1.0);
        assert_eq!(cospi(0.0), 1.0);
        assert_eq!(cospi(1.0), -1.0);
        assert_eq!(cospi(0.5), 0.0);
        assert!((sinpi(0.25) - 2.0f64.sqrt() / 2.0).abs() < 1e-12);

        // Large angles: reduction mod 360 matches computing sin of the reduced angle.
        let big = 1e16f64;
        assert!((sind(big) - sind(big % 360.0)).abs() < 1e-12);
        assert!(sind(360.0f64).abs() < 1e-12);
        assert!((sind(30.0f64) - 0.5).abs() < 1e-12);
        assert!((cosd(60.0f64) - 0.5).abs() < 1e-12);
        assert!((tand(45.0f64) - 1.0).abs() < 1e-12);

        let (s, c) = sin_cos_deg(30.0f64);
        assert!((s - 0.5).abs() < 1e-12 && (c - 0.5 * 3.0f64.sqrt()).abs() < 1e-12);
    }
}
