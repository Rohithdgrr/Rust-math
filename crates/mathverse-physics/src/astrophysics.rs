//! Astrophysics and orbital mechanics.

use crate::constants::{AU, C, G};

/// Schwarzschild radius: r_s = 2GM / c².
pub fn schwarzschild_radius(m: f64) -> f64 {
    2.0 * G * m / (C * C)
}

/// Escape velocity from a body of mass `m` and radius `r`: v = √(2GM/r).
///
/// Returns `None` when `r == 0` or `m < 0`.
pub fn escape_velocity(m: f64, r: f64) -> Option<f64> {
    if r == 0.0 || m < 0.0 {
        return None;
    }
    Some((2.0 * G * m / r).sqrt())
}

/// Kepler's third law (semimajor-axis form, small-body approximation):
/// T² = 4π² a³ / (GM).
///
/// Returns `None` on division by zero.
pub fn orbital_period(m: f64, a: f64) -> Option<f64> {
    if m == 0.0 || a == 0.0 {
        return None;
    }
    Some((4.0 * std::f64::consts::PI.powi(2) * a.powi(3) / (G * m)).sqrt())
}

/// Gravitational potential energy between two point masses: U = −GMm/r.
///
/// Returns `None` when `r == 0`.
pub fn gravitational_potential_energy(m1: f64, m2: f64, r: f64) -> Option<f64> {
    if r == 0.0 {
        return None;
    }
    Some(-G * m1 * m2 / r)
}

/// Orbital speed for a circular orbit: v = √(GM / r).
///
/// Returns `None` when `r == 0` or `m < 0`.
pub fn circular_orbit_speed(m: f64, r: f64) -> Option<f64> {
    if r == 0.0 || m < 0.0 {
        return None;
    }
    Some((G * m / r).sqrt())
}

/// Hubble's law (recession velocity from redshift, linear): v = H₀ d.
///
/// Uses H₀ = 70 km/s/Mpc = 2.2683e-18 s⁻¹.
pub fn hubble_flow(distance: f64) -> f64 {
    // H0 in SI: 70 km/s/Mpc
    const H0: f64 = 2.268_3e-18;
    H0 * distance
}

/// Light travel time from a distance in metres: t = d / c.
pub fn light_travel_time(distance: f64) -> f64 {
    distance / C
}

/// Convert astronomical units to metres.
pub fn au_to_m(au: f64) -> f64 {
    au * AU
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_schwarzschild_sun() {
        // Sun: M = 1.989e30 kg → r_s ≈ 2.95 km
        let rs = schwarzschild_radius(1.989e30);
        assert_relative_eq!(rs, 2953.0, epsilon = 10.0);
        assert!(rs > 0.0);
    }

    #[test]
    fn test_escape_velocity() {
        assert_relative_eq!(
            escape_velocity(1.0, 1.0).unwrap(),
            (2.0 * G).sqrt(),
            epsilon = 1e-10
        );
        assert!(escape_velocity(1.0, 0.0).is_none());
        assert!(escape_velocity(-1.0, 1.0).is_none());
    }

    #[test]
    fn test_orbital_period() {
        // Earth: a = 1 AU, M_sun = 1.989e30 kg → T ≈ 365.25 days
        let t = orbital_period(1.989e30, AU).unwrap();
        assert_relative_eq!(t, 365.25 * 24.0 * 3600.0, epsilon = 1e-4);
    }

    #[test]
    fn test_circular_orbit_speed() {
        let v = circular_orbit_speed(1.989e30, AU).unwrap();
        assert!(v > 0.0 && v < C);
    }

    #[test]
    fn test_light_travel_time() {
        assert_relative_eq!(light_travel_time(AU), 499.004_78, epsilon = 1e-3);
    }
}
