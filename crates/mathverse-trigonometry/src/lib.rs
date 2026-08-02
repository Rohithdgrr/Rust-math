//! Trigonometry: circular and hyperbolic functions (and inverses),
//! generic over [`Real`]. Arguments are radians; `*_deg` variants take degrees.
//!
//! Computation happens in `f64` internally (via [`Real::to_f64`]) so `f32`
//! inputs get `f32` results with no precision loss.
//!
//! Asymptotes (e.g. `tan(π/2)`, `cot(0)`) return `±inf` like std.

pub mod conversions;
pub mod identities;
pub mod laws;
pub mod special;

use mathverse_core::traits::Real;

pub use conversions::{wrap_angle, wrap_angle_positive, rad_to_grad, grad_to_rad,
    turns_to_radians, radians_to_turns};
pub use identities::{sin_cos, sin_double, cos_double, tan_double, sin_half, cos_half, tan_half,
    sin_sum, sin_diff, cos_sum, cos_diff, tan_sum, tan_diff,
    sin_sin_product, cos_cos_product, sin_cos_product,
    sin_sum_to_product, sin_diff_to_product, cos_sum_to_product, cos_diff_to_product,
    sin_squared, cos_squared, tan_squared};
pub use laws::{law_of_sines_side, law_of_sines_angle, law_of_cosines_side, law_of_cosines_angle,
    heron, triangle_area_sas, triangle_area_base_height, bearing, haversine_distance};
pub use special::{sinc, sinc_unnorm, versine, coversine, vercosine, covercosine,
    haversine, havercosine, hacoversine, hacovercosine, exsecant, excosecant,
    gudermannian, gudermannian_inv, gudermannian_alt,
    chebyshev_first, chebyshev_second, sin_power, cos_power};

pub use mathverse_core::ops::{deg_to_grad, deg_to_rad, grad_to_deg, rad_to_deg};

fn f<T: Real>(x: T, f: impl Fn(f64) -> f64) -> T {
    T::from_f64(f(x.to_f64()))
}

/// Sine.
pub fn sin<T: Real>(x: T) -> T {
    f(x, f64::sin)
}
/// Cosine.
pub fn cos<T: Real>(x: T) -> T {
    f(x, f64::cos)
}
/// Tangent.
pub fn tan<T: Real>(x: T) -> T {
    f(x, f64::tan)
}
/// Cotangent `cos/sin` (1/tan loses precision near asymptotes).
pub fn cot<T: Real>(x: T) -> T {
    f(x, |r| r.cos() / r.sin())
}
/// Secant `1/cos`.
pub fn sec<T: Real>(x: T) -> T {
    f(x, |r| 1.0 / r.cos())
}
/// Cosecant `1/sin`.
pub fn csc<T: Real>(x: T) -> T {
    f(x, |r| 1.0 / r.sin())
}

/// Hyperbolic sine.
pub fn sinh<T: Real>(x: T) -> T {
    f(x, f64::sinh)
}
/// Hyperbolic cosine.
pub fn cosh<T: Real>(x: T) -> T {
    f(x, f64::cosh)
}
/// Hyperbolic tangent.
pub fn tanh<T: Real>(x: T) -> T {
    f(x, f64::tanh)
}
/// Hyperbolic cotangent.
pub fn coth<T: Real>(x: T) -> T {
    f(x, |r| r.cosh() / r.sinh())
}
/// Hyperbolic secant.
pub fn sech<T: Real>(x: T) -> T {
    f(x, |r| 1.0 / r.cosh())
}
/// Hyperbolic cosecant.
pub fn csch<T: Real>(x: T) -> T {
    f(x, |r| 1.0 / r.sinh())
}

/// Arc sine.
pub fn asin<T: Real>(x: T) -> T {
    f(x, f64::asin)
}
/// Arc cosine.
pub fn acos<T: Real>(x: T) -> T {
    f(x, f64::acos)
}
/// Arc tangent.
pub fn atan<T: Real>(x: T) -> T {
    f(x, f64::atan)
}
/// Four-quadrant arc tangent: angle of the point `(y, x)`.
pub fn atan2<T: Real>(y: T, x: T) -> T {
    f(y, |ry| ry.atan2(x.to_f64()))
}
/// Arc cotangent: `π/2 - atan(x)` (valid for all x).
pub fn acot<T: Real>(x: T) -> T {
    f(x, |r| core::f64::consts::FRAC_PI_2 - r.atan())
}
/// Arc secant: `acos(1/x)`. Domain: |x| >= 1.
pub fn asec<T: Real>(x: T) -> T {
    let r = x.to_f64();
    if r.abs() < 1.0 {
        return T::from_f64(f64::NAN);
    }
    f(x, |r| (1.0 / r).acos())
}
/// Arc cosecant: `asin(1/x)`. Domain: |x| >= 1.
pub fn acsc<T: Real>(x: T) -> T {
    let r = x.to_f64();
    if r.abs() < 1.0 {
        return T::from_f64(f64::NAN);
    }
    f(x, |r| (1.0 / r).asin())
}

/// Inverse hyperbolic sine.
pub fn asinh<T: Real>(x: T) -> T {
    f(x, f64::asinh)
}
/// Inverse hyperbolic cosine (|x| >= 1).
pub fn acosh<T: Real>(x: T) -> T {
    f(x, f64::acosh)
}
/// Inverse hyperbolic tangent (|x| < 1).
pub fn atanh<T: Real>(x: T) -> T {
    f(x, f64::atanh)
}
/// Inverse hyperbolic cotangent.
pub fn acoth<T: Real>(x: T) -> T {
    f(x, |r| (1.0 / r).atanh())
}
/// Inverse hyperbolic secant.
pub fn asech<T: Real>(x: T) -> T {
    f(x, |r| (1.0 / r).acosh())
}
/// Inverse hyperbolic cosecant.
pub fn acsch<T: Real>(x: T) -> T {
    f(x, |r| (1.0 / r).asinh())
}

// ---------------------------------------------------------------------------
// Degree variants for circular functions
// ---------------------------------------------------------------------------

/// Sine of an angle in degrees.
pub fn sin_deg<T: Real>(d: T) -> T {
    sin(deg_to_rad(d))
}
/// Cosine of an angle in degrees.
pub fn cos_deg<T: Real>(d: T) -> T {
    cos(deg_to_rad(d))
}
/// Tangent of an angle in degrees.
pub fn tan_deg<T: Real>(d: T) -> T {
    tan(deg_to_rad(d))
}
/// Cotangent of an angle in degrees.
pub fn cot_deg<T: Real>(d: T) -> T {
    cot(deg_to_rad(d))
}
/// Secant of an angle in degrees.
pub fn sec_deg<T: Real>(d: T) -> T {
    sec(deg_to_rad(d))
}
/// Cosecant of an angle in degrees.
pub fn csc_deg<T: Real>(d: T) -> T {
    csc(deg_to_rad(d))
}

// ---------------------------------------------------------------------------
// Degree variants for inverse circular functions
// ---------------------------------------------------------------------------

/// Arc sine, result in degrees.
pub fn asin_deg<T: Real>(x: T) -> T {
    rad_to_deg(asin(x))
}
/// Arc cosine, result in degrees.
pub fn acos_deg<T: Real>(x: T) -> T {
    rad_to_deg(acos(x))
}
/// Arc tangent, result in degrees.
pub fn atan_deg<T: Real>(x: T) -> T {
    rad_to_deg(atan(x))
}
/// Four-quadrant arc tangent, result in degrees.
pub fn atan2_deg<T: Real>(y: T, x: T) -> T {
    rad_to_deg(atan2(y, x))
}
/// Arc cotangent, result in degrees.
pub fn acot_deg<T: Real>(x: T) -> T {
    rad_to_deg(acot(x))
}
/// Arc secant, result in degrees. Domain: |x| >= 1.
pub fn asec_deg<T: Real>(x: T) -> T {
    rad_to_deg(asec(x))
}
/// Arc cosecant, result in degrees. Domain: |x| >= 1.
pub fn acsc_deg<T: Real>(x: T) -> T {
    rad_to_deg(acsc(x))
}

// ---------------------------------------------------------------------------
// Degree variants for hyperbolic functions
// ---------------------------------------------------------------------------

/// Hyperbolic sine of an angle in degrees.
pub fn sinh_deg<T: Real>(d: T) -> T {
    sinh(deg_to_rad(d))
}
/// Hyperbolic cosine of an angle in degrees.
pub fn cosh_deg<T: Real>(d: T) -> T {
    cosh(deg_to_rad(d))
}
/// Hyperbolic tangent of an angle in degrees.
pub fn tanh_deg<T: Real>(d: T) -> T {
    tanh(deg_to_rad(d))
}
/// Hyperbolic cotangent of an angle in degrees.
pub fn coth_deg<T: Real>(d: T) -> T {
    coth(deg_to_rad(d))
}
/// Hyperbolic secant of an angle in degrees.
pub fn sech_deg<T: Real>(d: T) -> T {
    sech(deg_to_rad(d))
}
/// Hyperbolic cosecant of an angle in degrees.
pub fn csch_deg<T: Real>(d: T) -> T {
    csch(deg_to_rad(d))
}

// ---------------------------------------------------------------------------
// Degree variants for inverse hyperbolic functions
// ---------------------------------------------------------------------------

/// Inverse hyperbolic sine, result in degrees.
pub fn asinh_deg<T: Real>(x: T) -> T {
    rad_to_deg(asinh(x))
}
/// Inverse hyperbolic cosine, result in degrees.
pub fn acosh_deg<T: Real>(x: T) -> T {
    rad_to_deg(acosh(x))
}
/// Inverse hyperbolic tangent, result in degrees.
pub fn atanh_deg<T: Real>(x: T) -> T {
    rad_to_deg(atanh(x))
}
/// Inverse hyperbolic cotangent, result in degrees.
pub fn acoth_deg<T: Real>(x: T) -> T {
    rad_to_deg(acoth(x))
}
/// Inverse hyperbolic secant, result in degrees.
pub fn asech_deg<T: Real>(x: T) -> T {
    rad_to_deg(asech(x))
}
/// Inverse hyperbolic cosecant, result in degrees.
pub fn acsch_deg<T: Real>(x: T) -> T {
    rad_to_deg(acsch(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_4, PI};

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
        assert!(asec(0.5).is_nan());
        assert!(asec(0.0).is_nan());
        assert!(asec(-0.5).is_nan());
        assert!(!asec(1.0).is_nan());
        assert!(!asec(2.0).is_nan());
        assert!(!asec(-1.0).is_nan());
        
        // Test acsc domain: |x| >= 1
        assert!(acsc(0.5).is_nan());
        assert!(acsc(0.0).is_nan());
        assert!(acsc(-0.5).is_nan());
        assert!(!acsc(1.0).is_nan());
        assert!(!acsc(2.0).is_nan());
        assert!(!acsc(-1.0).is_nan());
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
        
        // Test tan at asymptotes (should return ±inf)
        let pi_half = core::f64::consts::FRAC_PI_2;
        assert!(tan(pi_half).is_infinite());
        assert!(tan(-pi_half).is_infinite());
        
        // Test cot at 0 (should return ±inf)
        assert!(cot(0.0).is_infinite());
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
}
