//! Fluid dynamics: Reynolds number, Bernoulli, drag force, Stokes drag.

use crate::constants::G_0;

/// Reynolds number: Re = ρ · v · L / μ.
///
/// Characterises whether flow is laminar (`Re < ~2300`) or turbulent
/// (`Re > ~4000`). Returns `None` when viscosity is zero.
pub fn reynolds_number(rho: f64, v: f64, l: f64, mu: f64) -> Option<f64> {
    if mu == 0.0 {
        return None;
    }
    Some(rho * v * l / mu)
}

/// Bernoulli pressure: P + ½ρv² + ρgh = constant.
///
/// Computes the total dynamic pressure (sum of static, dynamic, and
/// hydrostatic components). All quantities are in SI units.
pub fn bernoulli_pressure(p_static: f64, rho: f64, v: f64, h: f64) -> f64 {
    p_static + 0.5 * rho * v * v + rho * G_0 * h
}

/// Stokes drag (creeping flow, Re ≪ 1): F = 6πμrv.
///
/// Returns `None` when the dynamic viscosity is zero.
pub fn stokes_drag(mu: f64, r: f64, v: f64) -> Option<f64> {
    if mu == 0.0 {
        return None;
    }
    Some(6.0 * std::f64::consts::PI * mu * r * v)
}

/// Quadratic drag (turbulent flow): F = ½ ρ v² C_d A.
///
/// `rho` – fluid density (kg/m³), `v` – velocity (m/s),
/// `cd` – drag coefficient, `area` – cross-sectional area (m²).
pub fn drag_force_quadratic(rho: f64, v: f64, cd: f64, area: f64) -> f64 {
    0.5 * rho * v * v * cd * area
}

/// Terminal velocity for a falling sphere in a viscous fluid
/// (Stokes' law): v_t = (2/9) · r² · g · (ρ_s − ρ_f) / μ.
///
/// Returns `None` on division by zero (viscosity or mismatch).
pub fn terminal_velocity_sphere(
    rho_sphere: f64,
    rho_fluid: f64,
    radius: f64,
    viscosity: f64,
) -> Option<f64> {
    if viscosity == 0.0 || radius == 0.0 {
        return None;
    }
    let delta_rho = rho_sphere - rho_fluid;
    if delta_rho == 0.0 {
        return Some(0.0);
    }
    Some((2.0 / 9.0) * radius * radius * G_0 * delta_rho / viscosity)
}

/// Volumetric flow rate (Poiseuille, laminar pipe flow):
/// Q = π r⁴ ΔP / (8 μ L).
///
/// Returns `None` on division by zero.
pub fn poiseuille_flow_rate(
    radius: f64,
    delta_pressure: f64,
    viscosity: f64,
    length: f64,
) -> Option<f64> {
    if viscosity == 0.0 || length == 0.0 || radius == 0.0 {
        return None;
    }
    Some(std::f64::consts::PI * radius.powi(4) * delta_pressure / (8.0 * viscosity * length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_reynolds_number() {
        assert_relative_eq!(
            reynolds_number(1000.0, 1.0, 0.1, 1e-3).unwrap(),
            1e5,
            epsilon = 1e-6
        );
        assert!(reynolds_number(1000.0, 1.0, 0.1, 0.0).is_none());
    }

    #[test]
    fn test_bernoulli() {
        // Stagnation pressure at sea level
        let p = bernoulli_pressure(101_325.0, 1.225, 0.0, 0.0);
        assert_relative_eq!(p, 101_325.0, epsilon = 1e-3);
    }

    #[test]
    fn test_drag() {
        assert_relative_eq!(
            drag_force_quadratic(1.225, 30.0, 0.47, 0.5),
            0.5 * 1.225 * 900.0 * 0.47 * 0.5,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_poiseuille() {
        assert!(
            poiseuille_flow_rate(0.01, 1000.0, 1e-3, 1.0)
                .unwrap()
                .is_finite()
        );
        assert!(poiseuille_flow_rate(0.01, 1000.0, 0.0, 1.0).is_none());
    }
}
