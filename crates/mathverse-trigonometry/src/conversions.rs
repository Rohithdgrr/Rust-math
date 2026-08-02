//! Coordinate conversions: polar/cartesian, spherical, angle normalization, turns/grads.

use mathverse_core::traits::Real;
use crate::util::map_real as f;

fn atan2_f<T: Real>(y: T, x: T) -> T {
    T::from_f64(y.to_f64().atan2(x.to_f64()))
}

// ---------------------------------------------------------------------------
// Angle normalization
// ---------------------------------------------------------------------------

/// Wrap angle in radians to `[-π, π)`.
pub fn wrap_angle<T: Real>(x: T) -> T {
    f(x, |r| {
        let pi = core::f64::consts::PI;
        let tau = core::f64::consts::TAU;
        let v = r % tau;
        if v >= pi { v - tau }
        else if v < -pi { v + tau }
        else { v }
    })
}

/// Wrap angle in radians to `[-π, π)` (const fn for f64).
/// Note: This is a simplified version suitable for const context.
/// For full functionality including special float handling, use the generic `wrap_angle`.
pub const fn wrap_angle_f64(x: f64) -> f64 {
    let pi = core::f64::consts::PI;
    let two_pi = pi * 2.0;
    let v = x % two_pi;
    if v >= pi { v - two_pi }
    else if v < -pi { v + two_pi }
    else { v }
}

/// Wrap angle in radians to `[0, 2π)`.
pub fn wrap_angle_positive<T: Real>(x: T) -> T {
    f(x, |r| {
        let tau = core::f64::consts::TAU;
        let v = r % tau;
        if v < 0.0 { v + tau } else { v }
    })
}

// ---------------------------------------------------------------------------
// Turns / Revolutions
// ---------------------------------------------------------------------------

/// Convert turns (revolutions) to radians.
pub fn turns_to_radians<T: Real>(turns: T) -> T {
    turns * T::from_f64(2.0 * core::f64::consts::PI)
}

/// Convert turns (revolutions) to radians (const fn for f64).
pub const fn turns_to_radians_f64(turns: f64) -> f64 {
    turns * 2.0 * core::f64::consts::PI
}

/// Convert radians to turns (revolutions).
pub fn radians_to_turns<T: Real>(radians: T) -> T {
    radians / T::from_f64(2.0 * core::f64::consts::PI)
}

/// Convert radians to turns (revolutions) (const fn for f64).
pub const fn radians_to_turns_f64(radians: f64) -> f64 {
    radians / (2.0 * core::f64::consts::PI)
}

// ---------------------------------------------------------------------------
// Gradians
// ---------------------------------------------------------------------------

/// Convert radians to gradians (400 grad = 2π rad).
pub fn rad_to_grad<T: Real>(radians: T) -> T {
    radians * T::from_f64(200.0 / core::f64::consts::PI)
}

/// Convert gradians to radians.
pub fn grad_to_rad<T: Real>(grads: T) -> T {
    grads * T::from_f64(core::f64::consts::PI / 200.0)
}

// ---------------------------------------------------------------------------
// Polar <-> Cartesian
// ---------------------------------------------------------------------------

/// Polar to Cartesian: `(r, θ)` → `(x, y)`.
pub fn polar_to_cartesian<T: Real>(r: T, theta: T) -> (T, T) {
    (r * f(theta, f64::cos), r * f(theta, f64::sin))
}

/// Cartesian to polar: `(x, y)` → `(r, θ)`.
pub fn cartesian_to_polar<T: Real>(x: T, y: T) -> (T, T) {
    let r = (x * x + y * y).sqrt();
    let theta = atan2_f(y, x);
    (r, theta)
}

/// Magnitude (radius) from cartesian coordinates.
pub fn magnitude<T: Real>(x: T, y: T) -> T {
    (x * x + y * y).sqrt()
}

/// Phase angle from cartesian coordinates.
pub fn phase<T: Real>(x: T, y: T) -> T {
    atan2_f(y, x)
}

// ---------------------------------------------------------------------------
// Spherical <-> Cartesian (physics convention: θ = polar, φ = azimuthal)
// ---------------------------------------------------------------------------

/// Spherical to Cartesian (physics convention).
/// `r` = radius, `theta` = polar angle from z-axis, `phi` = azimuthal angle in xy-plane.
pub fn spherical_to_cartesian<T: Real>(r: T, theta: T, phi: T) -> (T, T, T) {
    let st = f(theta, f64::sin);
    let ct = f(theta, f64::cos);
    let cp = f(phi, f64::cos);
    let sp = f(phi, f64::sin);
    (r * st * cp, r * st * sp, r * ct)
}

/// Cartesian to Spherical (physics convention).
/// Returns `(r, theta, phi)`.
pub fn cartesian_to_spherical<T: Real>(x: T, y: T, z: T) -> (T, T, T) {
    let r = (x * x + y * y + z * z).sqrt();
    if r == T::zero() {
        return (T::zero(), T::zero(), T::zero());
    }
    let theta = (z / r).acos();
    let phi = atan2_f(y, x);
    (r, theta, phi)
}

/// Spherical to Cartesian (math convention: θ = polar from y-axis).
pub fn spherical_to_cartesian_math<T: Real>(r: T, theta: T, phi: T) -> (T, T, T) {
    let st = f(theta, f64::sin);
    let ct = f(theta, f64::cos);
    let cp = f(phi, f64::cos);
    let sp = f(phi, f64::sin);
    (r * st * sp, r * ct, r * st * cp)
}

/// Cartesian to Spherical (math convention).
/// Returns `(r, theta, phi)`.
pub fn cartesian_to_spherical_math<T: Real>(x: T, y: T, z: T) -> (T, T, T) {
    let r = (x * x + y * y + z * z).sqrt();
    if r == T::zero() {
        return (T::zero(), T::zero(), T::zero());
    }
    let theta = (y / r).acos();
    let phi = atan2_f(z, x);
    (r, theta, phi)
}

// ---------------------------------------------------------------------------
// Cylindrical <-> Cartesian
// ---------------------------------------------------------------------------

/// Cylindrical to Cartesian: `(r, θ, z)` → `(x, y, z)`.
pub fn cylindrical_to_cartesian<T: Real>(r: T, theta: T, z: T) -> (T, T, T) {
    (r * f(theta, f64::cos), r * f(theta, f64::sin), z)
}

/// Cartesian to Cylindrical: `(x, y, z)` → `(r, θ, z)`.
pub fn cartesian_to_cylindrical<T: Real>(x: T, y: T, z: T) -> (T, T, T) {
    let r = (x * x + y * y).sqrt();
    let theta = atan2_f(y, x);
    (r, theta, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    const EPS: f64 = 1e-12;

    #[test]
    fn wrap_angle_test() {
        assert!((wrap_angle(3.0 * PI) - (-PI)).abs() < EPS);
        // -3π wraps to -π (boundary of [-π, π))
        assert!((wrap_angle(-3.0 * PI) - (-PI)).abs() < EPS);
        assert!((wrap_angle_positive(3.0 * PI) - PI).abs() < EPS);
        assert!((wrap_angle_positive(-PI) - PI).abs() < EPS);
    }

    #[test]
    fn turns_test() {
        assert!((turns_to_radians(0.5) - PI).abs() < EPS);
        assert!((radians_to_turns(PI) - 0.5).abs() < EPS);
    }

    #[test]
    fn const_fn_tests() {
        // Test const fn variants work correctly
        assert!((wrap_angle_f64(3.0 * PI) - (-PI)).abs() < EPS);
        assert!((wrap_angle_f64(-3.0 * PI) - (-PI)).abs() < EPS);
        assert!((turns_to_radians_f64(0.5) - PI).abs() < EPS);
        assert!((radians_to_turns_f64(PI) - 0.5).abs() < EPS);
        
        // Verify const variants match generic versions
        assert!((wrap_angle_f64(3.0 * PI) - wrap_angle(3.0 * PI)).abs() < EPS);
        assert!((turns_to_radians_f64(0.5) - turns_to_radians(0.5)).abs() < EPS);
        assert!((radians_to_turns_f64(PI) - radians_to_turns(PI)).abs() < EPS);
    }

    #[test]
    fn gradian_test() {
        assert!((rad_to_grad(TAU) - 400.0).abs() < EPS);
        assert!((grad_to_rad(200.0) - PI).abs() < EPS);
    }

    #[test]
    fn polar_cartesian_roundtrip() {
        let (x, y) = polar_to_cartesian(1.0f64, FRAC_PI_4);
        assert!((x - FRAC_PI_4.cos()).abs() < EPS);
        assert!((y - FRAC_PI_4.sin()).abs() < EPS);
        let (r, theta) = cartesian_to_polar(x, y);
        assert!((r - 1.0).abs() < EPS);
        assert!((theta - FRAC_PI_4).abs() < EPS);
    }

    #[test]
    fn spherical_roundtrip() {
        let (x, y, z) = spherical_to_cartesian(1.0f64, FRAC_PI_2, PI);
        assert!((x - (-1.0)).abs() < EPS);
        assert!((y).abs() < EPS);
        assert!((z).abs() < EPS);
        let (r, theta, phi) = cartesian_to_spherical(x, y, z);
        assert!((r - 1.0).abs() < EPS);
        assert!((theta - FRAC_PI_2).abs() < EPS);
        assert!((phi - PI).abs() < EPS);
    }

    #[test]
    fn cylindrical_roundtrip() {
        let (x, y, z) = cylindrical_to_cartesian(2.0f64, PI, 3.0);
        assert!((x - (-2.0)).abs() < EPS);
        assert!(y.abs() < EPS);
        assert!((z - 3.0).abs() < EPS);
        let (r, theta, zz) = cartesian_to_cylindrical(x, y, z);
        assert!((r - 2.0).abs() < EPS);
        assert!((theta - PI).abs() < EPS);
        assert!((zz - 3.0).abs() < EPS);
    }
}
