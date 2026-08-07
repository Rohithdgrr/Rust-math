//! Electromagnetism

use crate::constants::{EPSILON_0, MU_0};

/// Calculate electric field from point charge
/// 
/// # Arguments
/// * `q` - Charge (C)
/// * `r` - Distance from charge (m)
/// 
/// # Returns
/// Electric field magnitude (N/C or V/m), or `None` if the distance is zero.
pub fn electric_field_point(q: f64, r: f64) -> Option<f64> {
    if r == 0.0 { return None; }
    Some(q / (4.0 * std::f64::consts::PI * EPSILON_0 * r * r))
}

/// Calculate electric force between two charges (Coulomb's law)
/// 
/// # Arguments
/// * `q1` - First charge (C)
/// * `q2` - Second charge (C)
/// * `r` - Distance between charges (m)
/// 
/// # Returns
/// Electric force (N), or `None` if the distance is zero.
pub fn coulomb_force(q1: f64, q2: f64, r: f64) -> Option<f64> {
    if r == 0.0 { return None; }
    Some(q1 * q2 / (4.0 * std::f64::consts::PI * EPSILON_0 * r * r))
}

/// Calculate electric potential from point charge
/// 
/// # Arguments
/// * `q` - Charge (C)
/// * `r` - Distance from charge (m)
/// 
/// # Returns
/// Electric potential (V), or `None` if the distance is zero.
pub fn electric_potential(q: f64, r: f64) -> Option<f64> {
    if r == 0.0 { return None; }
    Some(q / (4.0 * std::f64::consts::PI * EPSILON_0 * r))
}

/// Calculate electric potential energy
/// 
/// # Arguments
/// * `q1` - First charge (C)
/// * `q2` - Second charge (C)
/// * `r` - Distance between charges (m)
/// 
/// # Returns
/// Electric potential energy (J), or `None` if the distance is zero.
pub fn electric_potential_energy(q1: f64, q2: f64, r: f64) -> Option<f64> {
    if r == 0.0 { return None; }
    Some(q1 * q2 / (4.0 * std::f64::consts::PI * EPSILON_0 * r))
}

/// Calculate capacitance of parallel plate capacitor
/// 
/// # Arguments
/// * `epsilon` - Permittivity of dielectric (F/m)
/// * `a` - Area of plates (m²)
/// * `d` - Distance between plates (m)
/// 
/// # Returns
/// Capacitance (F), or `None` if the plate distance is zero.
pub fn capacitance_parallel_plate(epsilon: f64, a: f64, d: f64) -> Option<f64> {
    if d == 0.0 { return None; }
    Some(epsilon * a / d)
}

/// Calculate energy stored in capacitor
/// 
/// # Arguments
/// * `c` - Capacitance (F)
/// * `v` - Voltage (V)
/// 
/// # Returns
/// Energy (J)
pub fn capacitor_energy(c: f64, v: f64) -> f64 {
    0.5 * c * v * v
}

/// Calculate magnetic field from current in straight wire
/// 
/// # Arguments
/// * `i` - Current (A)
/// * `r` - Distance from wire (m)
/// 
/// # Returns
/// Magnetic field magnitude (T), or `None` if the distance from the wire is zero.
pub fn magnetic_field_wire(i: f64, r: f64) -> Option<f64> {
    if r == 0.0 { return None; }
    Some(MU_0 * i / (2.0 * std::f64::consts::PI * r))
}

/// Calculate magnetic force on current-carrying wire
/// 
/// # Arguments
/// * `i` - Current (A)
/// * `l` - Length of wire (m)
/// * `b` - Magnetic field (T)
/// * `theta` - Angle between current and field (radians)
/// 
/// # Returns
/// Magnetic force (N)
pub fn magnetic_force_wire(i: f64, l: f64, b: f64, theta: f64) -> f64 {
    i * l * b * theta.sin()
}

/// Calculate Lorentz force on charged particle
/// 
/// # Arguments
/// * `q` - Charge (C)
/// * `v` - Velocity (m/s)
/// * `b` - Magnetic field (T)
/// * `theta` - Angle between velocity and field (radians)
/// 
/// # Returns
/// Magnetic force (N)
pub fn lorentz_force(q: f64, v: f64, b: f64, theta: f64) -> f64 {
    q * v * b * theta.sin()
}

/// Calculate magnetic flux
/// 
/// # Arguments
/// * `b` - Magnetic field (T)
/// * `a` - Area (m²)
/// * `theta` - Angle between field and normal (radians)
/// 
/// # Returns
/// Magnetic flux (Wb)
pub fn magnetic_flux(b: f64, a: f64, theta: f64) -> f64 {
    b * a * theta.cos()
}

/// Calculate induced EMF (Faraday's law)
/// 
/// # Arguments
/// * `d_phi` - Change in magnetic flux (Wb)
/// * `dt` - Time interval (s)
/// 
/// # Returns
/// Induced EMF (V), or `None` if the time interval is zero.
pub fn induced_emf(d_phi: f64, dt: f64) -> Option<f64> {
    if dt == 0.0 { return None; }
    Some(-d_phi / dt)
}

/// Calculate inductance of solenoid
/// 
/// # Arguments
/// * `mu` - Permeability (H/m)
/// * `n` - Number of turns per unit length (1/m)
/// * `a` - Cross-sectional area (m²)
/// * `l` - Length (m)
/// 
/// # Returns
/// Inductance (H)
pub fn inductance_solenoid(mu: f64, n: f64, a: f64, l: f64) -> f64 {
    mu * n * n * a * l
}

/// Calculate energy stored in inductor
/// 
/// # Arguments
/// * `l` - Inductance (H)
/// * `i` - Current (A)
/// 
/// # Returns
/// Energy (J)
pub fn inductor_energy(l: f64, i: f64) -> f64 {
    0.5 * l * i * i
}

/// Calculate resistance from resistivity
/// 
/// # Arguments
/// * `rho` - Resistivity (Ω·m)
/// * `l` - Length (m)
/// * `a` - Cross-sectional area (m²)
/// 
/// # Returns
/// Resistance (Ω), or `None` if the cross-sectional area is zero.
pub fn resistance_from_resistivity(rho: f64, l: f64, a: f64) -> Option<f64> {
    if a == 0.0 { return None; }
    Some(rho * l / a)
}

/// Calculate power dissipated in resistor
/// 
/// # Arguments
/// * `v` - Voltage (V)
/// * `r` - Resistance (Ω)
/// 
/// # Returns
/// Power (W), or `None` if the resistance is zero.
pub fn resistor_power(v: f64, r: f64) -> Option<f64> {
    if r == 0.0 { return None; }
    Some(v * v / r)
}

/// Calculate drift velocity of electrons
/// 
/// # Arguments
/// * `i` - Current (A)
/// * `n` - Number density of charge carriers (1/m³)
/// * `q` - Charge per carrier (C)
/// * `a` - Cross-sectional area (m²)
/// 
/// # Returns
/// Drift velocity (m/s), or `None` if any of `n`, `q`, or `a` is zero.
pub fn drift_velocity(i: f64, n: f64, q: f64, a: f64) -> Option<f64> {
    if n == 0.0 || q == 0.0 || a == 0.0 { return None; }
    Some(i / (n * q * a))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_resistor_power() {
        assert_relative_eq!(resistor_power(12.0, 4.0).unwrap(), 36.0, epsilon = 1e-6);
        assert!(resistor_power(12.0, 0.0).is_none());
    }

    #[test]
    fn test_capacitor_energy() {
        assert_relative_eq!(capacitor_energy(1e-6, 10.0), 5e-5, epsilon = 1e-10);
    }

    #[test]
    fn test_coulomb_and_field() {
        let q1 = 1e-6;
        let q2 = -1e-6;
        let r = 0.1;
        let k = 1.0 / (4.0 * std::f64::consts::PI * EPSILON_0);
        assert_relative_eq!(coulomb_force(q1, q2, r).unwrap(), k * q1 * q2 / (r * r), epsilon = 1e-9);
        assert_relative_eq!(electric_field_point(q1, r).unwrap(), k * q1 / (r * r), epsilon = 1e-9);
        assert!(coulomb_force(q1, q2, 0.0).is_none());
        assert!(electric_potential(q1, 0.0).is_none());
    }

    #[test]
    fn test_capacitance() {
        assert_relative_eq!(capacitance_parallel_plate(EPSILON_0, 1e-4, 1e-3).unwrap(), EPSILON_0 * 1e-1, epsilon = 1e-12);
        assert!(capacitance_parallel_plate(EPSILON_0, 1e-4, 0.0).is_none());
    }

    #[test]
    fn test_magnetic_and_resistance() {
        assert_relative_eq!(
            magnetic_field_wire(10.0, 0.05).unwrap(),
            MU_0 * 10.0 / (2.0 * std::f64::consts::PI * 0.05),
            epsilon = 1e-9
        );
        assert!(magnetic_field_wire(10.0, 0.0).is_none());
        assert_relative_eq!(resistance_from_resistivity(1.7e-8, 2.0, 1e-6).unwrap(), 3.4e-2, epsilon = 1e-9);
        assert!(drift_velocity(1.0, 0.0, 1.6e-19, 1e-6).is_none());
    }
}
