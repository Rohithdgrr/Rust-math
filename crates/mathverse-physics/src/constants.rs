//! Physical constants with value and measurement uncertainty.
//!
//! Each constant is represented as a `PhysicalConstant` which exposes the
//! central value (`value`), the standard uncertainty (`uncertainty`), and a
//! human-readable `unit` string. Relative uncertainty is available via
//! `PhysicalConstant::relative_uncertainty()`.

use std::f64::consts::PI;

/// A physical constant with a central value, measurement uncertainty, and unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalConstant {
    /// Central (best-estimate) value in SI units.
    pub value: f64,
    /// Standard combined uncertainty in the same units as `value`.
    pub uncertainty: f64,
    /// Human-readable unit string, e.g. `"m³·kg⁻¹·s⁻²"`.
    pub unit: &'static str,
}

impl PhysicalConstant {
    /// Relative uncertainty: `uncertainty / value`.
    ///
    /// Returns `0.0` for a zero-valued constant (avoids division by zero).
    pub fn relative_uncertainty(&self) -> f64 {
        if self.value == 0.0 {
            return 0.0;
        }
        self.uncertainty / self.value
    }
}

// ---------------------------------------------------------------------------
// CODATA 2018 – selected constants
// ---------------------------------------------------------------------------

/// Speed of light in vacuum: 299 792 458 m/s (exact by SI definition).
pub const C: f64 = 299_792_458.0;
pub const C_CONST: PhysicalConstant = PhysicalConstant {
    value: C,
    uncertainty: 0.0,
    unit: "m/s",
};

/// Gravitational constant: 6.67430(15) × 10⁻¹¹ m³·kg⁻¹·s⁻².
pub const G: f64 = 6.674_30e-11;
pub const G_CONST: PhysicalConstant = PhysicalConstant {
    value: 6.674_30e-11,
    uncertainty: 0.000_15e-11,
    unit: "m³·kg⁻¹·s⁻²",
};

/// Planck constant: 6.626 070 15 × 10⁻³⁴ J·s (exact by SI definition).
pub const H: f64 = 6.626_070_15e-34;
pub const H_CONST: PhysicalConstant = PhysicalConstant {
    value: H,
    uncertainty: 0.0,
    unit: "J·s",
};

/// Reduced Planck constant: ħ = h / (2π).
pub const H_BAR: f64 = H / (2.0 * PI);
pub const H_BAR_CONST: PhysicalConstant = PhysicalConstant {
    value: H_BAR,
    uncertainty: 0.0,
    unit: "J·s",
};

/// Elementary charge: 1.602 176 634 × 10⁻¹⁹ C (exact by SI definition).
pub const E: f64 = 1.602_176_634e-19;
pub const E_CONST: PhysicalConstant = PhysicalConstant {
    value: E,
    uncertainty: 0.0,
    unit: "C",
};

/// Boltzmann constant: 1.380 649 × 10⁻²³ J/K (exact by SI definition).
pub const K_B: f64 = 1.380_649e-23;
pub const K_B_CONST: PhysicalConstant = PhysicalConstant {
    value: K_B,
    uncertainty: 0.0,
    unit: "J/K",
};

/// Avogadro constant: 6.022 140 76 × 10²³ mol⁻¹ (exact by SI definition).
pub const N_A: f64 = 6.022_140_76e23;
pub const N_A_CONST: PhysicalConstant = PhysicalConstant {
    value: N_A,
    uncertainty: 0.0,
    unit: "mol⁻¹",
};

/// Molar gas constant: R = N_A · k_B.
pub const R: f64 = N_A * K_B;
pub const R_CONST: PhysicalConstant = PhysicalConstant {
    value: R,
    uncertainty: 0.0,
    unit: "J·mol⁻¹·K⁻¹",
};

/// Permittivity of free space: 8.854 187 812 8 × 10⁻¹² F/m.
pub const EPSILON_0: f64 = 8.854_187_812_8e-12;
pub const EPSILON_0_CONST: PhysicalConstant = PhysicalConstant {
    value: EPSILON_0,
    uncertainty: 0.000_000_001_3e-12,
    unit: "F/m",
};

/// Permeability of free space: 4π × 10⁻⁷ H/m (exact by SI definition).
pub const MU_0: f64 = 4.0 * PI * 1.0e-7;
pub const MU_0_CONST: PhysicalConstant = PhysicalConstant {
    value: MU_0,
    uncertainty: 0.0,
    unit: "H/m",
};

/// Standard gravitational acceleration: 9.80665 m/s².
pub const G_0: f64 = 9.806_65;
pub const G_0_CONST: PhysicalConstant = PhysicalConstant {
    value: G_0,
    uncertainty: 0.0,
    unit: "m/s²",
};

/// Astronomical unit: 1.495 978 707 × 10¹¹ m.
pub const AU: f64 = 1.495_978_707e11;
pub const AU_CONST: PhysicalConstant = PhysicalConstant {
    value: AU,
    uncertainty: 0.0,
    unit: "m",
};

/// Electron mass: 9.109 383 701 5 × 10⁻³¹ kg.
pub const M_E: f64 = 9.109_383_701_5e-31;
pub const M_E_CONST: PhysicalConstant = PhysicalConstant {
    value: M_E,
    uncertainty: 0.000_000_002_8e-31,
    unit: "kg",
};

/// Proton mass: 1.672 621 923 69 × 10⁻²⁷ kg.
pub const M_P: f64 = 1.672_621_923_69e-27;
pub const M_P_CONST: PhysicalConstant = PhysicalConstant {
    value: M_P,
    uncertainty: 0.000_000_000_83e-27,
    unit: "kg",
};

/// Neutron mass: 1.674 927 498 04 × 10⁻²⁷ kg.
pub const M_N: f64 = 1.674_927_498_04e-27;
pub const M_N_CONST: PhysicalConstant = PhysicalConstant {
    value: M_N,
    uncertainty: 0.000_000_000_95e-27,
    unit: "kg",
};

/// Rydberg constant: 10 973 731.568 160 m⁻¹.
pub const R_INF: f64 = 10_973_731.568_160;
pub const R_INF_CONST: PhysicalConstant = PhysicalConstant {
    value: R_INF,
    uncertainty: 0.000_021,
    unit: "m⁻¹",
};

/// Fine structure constant (dimensionless).
pub const ALPHA: f64 = 7.297_352_569_3e-3;
pub const ALPHA_CONST: PhysicalConstant = PhysicalConstant {
    value: ALPHA,
    uncertainty: 0.000_000_000_001_2e-3,
    unit: "dimensionless",
};

/// Stefan-Boltzmann constant: 5.670 374 419 × 10⁻⁸ W·m⁻²·K⁻⁴.
pub const SIGMA: f64 = 5.670_374_419e-8;
pub const SIGMA_CONST: PhysicalConstant = PhysicalConstant {
    value: SIGMA,
    uncertainty: 0.0,
    unit: "W·m⁻²·K⁻⁴",
};

// ---------------------------------------------------------------------------
// scipy.constants-style conversion helpers
// ---------------------------------------------------------------------------

/// Errors that can occur during constant lookups or conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("constant error: {0}")]
pub struct ConstantError(&'static str);

/// Convert a temperature value between named scales.
///
/// Supported scales: `"Celsius"`, `"Kelvin"`, `"Fahrenheit"`, `"Rankine"`.
pub fn convert_temperature(val: f64, from: &str, to: &str) -> Result<f64, ConstantError> {
    let to_k = |v: f64, s: &str| -> Result<f64, ConstantError> {
        match s {
            "Kelvin" => Ok(v),
            "Celsius" => Ok(v + 273.15),
            "Fahrenheit" => Ok((v - 32.0) * 5.0 / 9.0 + 273.15),
            "Rankine" => Ok(v * 5.0 / 9.0),
            _ => Err(ConstantError("unknown temperature scale")),
        }
    };
    let k = to_k(val, from)?;
    match to {
        "Kelvin" => Ok(k),
        "Celsius" => Ok(k - 273.15),
        "Fahrenheit" => Ok((k - 273.15) * 9.0 / 5.0 + 32.0),
        "Rankine" => Ok(k * 9.0 / 5.0),
        _ => Err(ConstantError("unknown temperature scale")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_constant_relative_uncertainty() {
        assert!((G_CONST.relative_uncertainty() - 0.000_15e-11 / 6.674_30e-11).abs() < 1e-12);
        assert_eq!(C_CONST.relative_uncertainty(), 0.0);
        assert_eq!(PHYSICAL_CONSTANTS.len(), 16);
    }

    #[test]
    fn test_temperature_conversions() {
        assert!((convert_temperature(0.0, "Celsius", "Kelvin").unwrap() - 273.15).abs() < 1e-10);
        assert!((convert_temperature(273.15, "Kelvin", "Celsius").unwrap() - 0.0).abs() < 1e-10);
        assert!((convert_temperature(32.0, "Fahrenheit", "Celsius").unwrap() - 0.0).abs() < 1e-9);
        assert!((convert_temperature(32.0, "Fahrenheit", "Kelvin").unwrap() - 273.15).abs() < 1e-9);
        assert!((convert_temperature(491.67, "Rankine", "Fahrenheit").unwrap() - 32.0).abs() < 1e-9);
        assert!(convert_temperature(0.0, "Banana", "Kelvin").is_err());
    }
}

/// All defined `PhysicalConstant` instances, for iteration in consumers.
pub const PHYSICAL_CONSTANTS: &[PhysicalConstant] = &[
    C_CONST,
    G_CONST,
    H_CONST,
    H_BAR_CONST,
    E_CONST,
    K_B_CONST,
    N_A_CONST,
    R_CONST,
    EPSILON_0_CONST,
    MU_0_CONST,
    G_0_CONST,
    AU_CONST,
    M_E_CONST,
    M_P_CONST,
    M_N_CONST,
    R_INF_CONST,
    ALPHA_CONST,
    SIGMA_CONST,
];
