//! Electromagnetism

use crate::constants::{EPSILON_0, MU_0};

/// Calculate electric field from point charge
/// 
/// # Arguments
/// * `q` - Charge (C)
/// * `r` - Distance from charge (m)
/// 
/// # Returns
/// Electric field magnitude (N/C or V/m)
pub fn electric_field_point(q: f64, r: f64) -> f64 {
    q / (4.0 * std::f64::consts::PI * EPSILON_0 * r * r)
}

/// Calculate electric force between two charges (Coulomb's law)
/// 
/// # Arguments
/// * `q1` - First charge (C)
/// * `q2` - Second charge (C)
/// * `r` - Distance between charges (m)
/// 
/// # Returns
/// Electric force (N)
pub fn coulomb_force(q1: f64, q2: f64, r: f64) -> f64 {
    q1 * q2 / (4.0 * std::f64::consts::PI * EPSILON_0 * r * r)
}

/// Calculate electric potential from point charge
/// 
/// # Arguments
/// * `q` - Charge (C)
/// * `r` - Distance from charge (m)
/// 
/// # Returns
/// Electric potential (V)
pub fn electric_potential(q: f64, r: f64) -> f64 {
    q / (4.0 * std::f64::consts::PI * EPSILON_0 * r)
}

/// Calculate electric potential energy
/// 
/// # Arguments
/// * `q1` - First charge (C)
/// * `q2` - Second charge (C)
/// * `r` - Distance between charges (m)
/// 
/// # Returns
/// Electric potential energy (J)
pub fn electric_potential_energy(q1: f64, q2: f64, r: f64) -> f64 {
    q1 * q2 / (4.0 * std::f64::consts::PI * EPSILON_0 * r)
}

/// Calculate capacitance of parallel plate capacitor
/// 
/// # Arguments
/// * `epsilon` - Permittivity of dielectric (F/m)
/// * `a` - Area of plates (m²)
/// * `d` - Distance between plates (m)
/// 
/// # Returns
/// Capacitance (F)
pub fn capacitance_parallel_plate(epsilon: f64, a: f64, d: f64) -> f64 {
    epsilon * a / d
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
/// Magnetic field magnitude (T)
pub fn magnetic_field_wire(i: f64, r: f64) -> f64 {
    MU_0 * i / (2.0 * std::f64::consts::PI * r)
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
/// Induced EMF (V)
pub fn induced_emf(d_phi: f64, dt: f64) -> f64 {
    -d_phi / dt
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
/// Resistance (Ω)
pub fn resistance_from_resistivity(rho: f64, l: f64, a: f64) -> f64 {
    rho * l / a
}

/// Calculate power dissipated in resistor
/// 
/// # Arguments
/// * `v` - Voltage (V)
/// * `r` - Resistance (Ω)
/// 
/// # Returns
/// Power (W)
pub fn resistor_power(v: f64, r: f64) -> f64 {
    v * v / r
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
/// Drift velocity (m/s)
pub fn drift_velocity(i: f64, n: f64, q: f64, a: f64) -> f64 {
    i / (n * q * a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_resistor_power() {
        assert_relative_eq!(resistor_power(12.0, 4.0), 36.0, epsilon = 1e-6);
    }

    #[test]
    fn test_capacitor_energy() {
        assert_relative_eq!(capacitor_energy(1e-6, 10.0), 5e-5, epsilon = 1e-10);
    }
}
