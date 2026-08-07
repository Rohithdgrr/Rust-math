//! Relativistic mechanics: Lorentz transformations and relativistic kinematics.

use std::f64::consts::PI;
use crate::constants::C;

/// Lorentz factor γ = 1 / √(1 − v²/c²).
///
/// Returns `None` when `|v| >= C` (at or above light speed, no real γ).
pub fn lorentz_factor(v: f64) -> Option<f64> {
    let beta = v / C;
    let gamma_sq = 1.0 - beta * beta;
    if gamma_sq <= 0.0 {
        return None;
    }
    Some(1.0 / gamma_sq.sqrt())
}

/// Special-relativistic time dilation: Δt' = γ · Δt.
///
/// Returns `None` when `|v| >= C`.
pub fn time_dilation(proper_time: f64, v: f64) -> Option<f64> {
    lorentz_factor(v).map(|gamma| gamma * proper_time)
}

/// Special-relativistic length contraction: L' = L / γ.
///
/// Returns `None` when `|v| >= C`.
pub fn length_contraction(proper_length: f64, v: f64) -> Option<f64> {
    lorentz_factor(v).map(|gamma| proper_length / gamma)
}

/// Relativistic kinetic energy: K = (γ − 1) m c².
///
/// Returns `None` when `|v| >= C`.
pub fn relativistic_kinetic_energy(m: f64, v: f64) -> Option<f64> {
    lorentz_factor(v).map(|gamma| (gamma - 1.0) * m * C * C)
}

/// Relativistic momentum: p = γ m v.
///
/// Returns `None` when `|v| >= C`.
pub fn relativistic_momentum(m: f64, v: f64) -> Option<f64> {
    lorentz_factor(v).map(|gamma| gamma * m * v)
}

/// Velocity-addition formula (colinear, same direction):
/// u' = (u + v) / (1 + u·v / c²).
///
/// Returns `None` if either `u` or `v` equals or exceeds `C`.
pub fn velocity_addition(u: f64, v: f64) -> Option<f64> {
    if u.abs() >= C || v.abs() >= C {
        return None;
    }
    Some((u + v) / (1.0 + (u * v) / (C * C)))
}

/// Relativistic Doppler shift for a source moving directly toward/away from
/// the observer (longitudinal):
/// f' = f · √((1 − β) / (1 + β)),  β = v / c.
///
/// `v > 0` means source receding (redshift); `v < 0` means approaching
/// (blueshift). Returns `None` when `|v| >= C`.
pub fn relativistic_doppler(f_source: f64, v: f64) -> Option<f64> {
    let beta = v / C;
    if beta.abs() >= 1.0 {
        return None;
    }
    Some(f_source * ((1.0 - beta) / (1.0 + beta)).sqrt())
}

/// Schwarzschild radius: r_s = 2 G M / c².
pub fn schwarzschild_radius(m: f64) -> f64 {
    2.0 * crate::constants::G * m / (C * C)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_lorentz_factor() {
        assert_relative_eq!(lorentz_factor(0.0).unwrap(), 1.0, epsilon = 1e-15);
        assert_relative_eq!(
            lorentz_factor(0.6 * C).unwrap(),
            1.0 / (1.0 - 0.36).sqrt(),
            epsilon = 1e-9
        );
        assert!(lorentz_factor(C).is_none());
        assert!(lorentz_factor(-C).is_none());
    }

    #[test]
    fn test_time_dilation() {
        // GPS satellites: v ≈ 3.87 km/s → γ ≈ 1 + 8.4e-11
        let td = time_dilation(1.0, 3870.0).unwrap();
        assert!(td > 1.0 && td < 1.000000001);
    }

    #[test]
    fn test_length_contraction() {
        let lc = length_contraction(1.0, 0.8 * C).unwrap();
        assert_relative_eq!(lc, 1.0 / lorentz_factor(0.8 * C).unwrap(), epsilon = 1e-9);
    }

    #[test]
    fn test_relativistic_energy() {
        // At v = 0, kinetic energy ≈ 0
        assert_relative_eq!(
            relativistic_kinetic_energy(1.0, 0.0).unwrap(),
            0.0,
            epsilon = 1e-15
        );
        // At v = C, return None
        assert!(relativistic_kinetic_energy(1.0, C).is_none());
    }

    #[test]
    fn test_velocity_addition() {
        // Classical: u + v = 0.5c + 0.5c = c; relativistic gives < c
        let va = velocity_addition(0.5 * C, 0.5 * C).unwrap();
        assert!(va < C);
    }

    #[test]
    fn test_doppler() {
        // At v = 0, no shift
        assert_relative_eq!(relativistic_doppler(440.0, 0.0).unwrap(), 440.0, epsilon = 1e-12);
        // v = C → None
        assert!(relativistic_doppler(440.0, C).is_none());
    }
}
