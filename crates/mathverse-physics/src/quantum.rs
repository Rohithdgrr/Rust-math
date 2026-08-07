//! Quantum mechanics constants and formulas.

use crate::constants::{E, H, H_BAR, M_E};

/// Bohr radius: a₀ = 4πε₀ħ² / (mₑ e²) ≈ 5.29177210903 × 10⁻¹¹ m.
pub fn bohr_radius() -> f64 {
    4.0 * std::f64::consts::PI * crate::constants::EPSILON_0 * H_BAR.powi(2)
        / (M_E * E.powi(2))
}

/// Hydrogen energy levels (non-relativistic Bohr model with reduced mass
/// correction baked into Rydberg): E_n = −R_y / n², R_y ≈ 13.605693 eV.
///
/// Result is in joules; divide by `E` to get eV.
pub fn hydrogen_energy_level_j(n: u32) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    -2.179_872_361_103_2e-18 / (n as f64).powi(2)
}

/// de Broglie wavelength: λ = h / p.
///
/// `momentum` in kg·m/s. Returns `None` for zero momentum.
pub fn de_broglie_wavelength(momentum: f64) -> Option<f64> {
    if momentum == 0.0 {
        return None;
    }
    Some(H / momentum)
}

/// Minimum position uncertainty given momentum uncertainty
/// (Heisenberg relation, using ħ/2 factor):
/// Δx ≥ ħ / (2 Δp).
///
/// Returns the lower bound in metres.
pub fn uncertainty_position_min(momentum_uncertainty: f64) -> Option<f64> {
    if momentum_uncertainty == 0.0 {
        return None;
    }
    Some(H_BAR / (2.0 * momentum_uncertainty))
}

/// Energy of a photon: E = h f = hc / λ.
pub fn photon_energy_freq(frequency: f64) -> f64 {
    H * frequency
}

/// Energy of a photon in joules given wavelength in metres.
pub fn photon_energy_wavelength(wavelength: f64) -> Option<f64> {
    if wavelength == 0.0 {
        return None;
    }
    Some(H * crate::constants::C / wavelength)
}

/// Compton wavelength of electron: λ_c = h / (m_e c).
pub fn compton_wavelength_electron() -> f64 {
    H / (M_E * crate::constants::C)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bohr_radius() {
        assert_relative_eq!(bohr_radius(), 5.291_772_109_03e-11, epsilon = 1e-20);
    }

    #[test]
    fn test_hydrogen_energy() {
        let e1 = hydrogen_energy_level_j(1);
        assert!(e1 < 0.0);
        let e2 = hydrogen_energy_level_j(2);
        // E_n = −R_y/n² ⇒ E₂/E₁ = (1/2²)/(1/1²) = 1/4
        assert_relative_eq!(e2 / e1, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_de_broglie() {
        let p = M_E * 1e6; // electron at 1 Mm/s
        let lam = de_broglie_wavelength(p).unwrap();
        assert_relative_eq!(lam, H / p, epsilon = 1e-45);
        assert!(de_broglie_wavelength(0.0).is_none());
    }

    #[test]
    fn test_photon_energy() {
        assert_relative_eq!(
            photon_energy_freq(5.0e14),
            H * 5.0e14,
            epsilon = 1e-50
        );
        assert!(photon_energy_wavelength(0.0).is_none());
    }
}