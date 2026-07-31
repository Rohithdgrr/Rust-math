//! Thermodynamics

use crate::constants::K_B;

/// Calculate ideal gas law: PV = nRT
/// 
/// # Arguments
/// * `n` - Number of moles
/// * `t` - Temperature (K)
/// * `v` - Volume (m³)
/// 
/// # Returns
/// Pressure (Pa)
pub fn ideal_gas_pressure(n: f64, t: f64, v: f64) -> f64 {
    n * 8.314_462_618 * t / v
}

/// Calculate temperature from ideal gas law
/// 
/// # Arguments
/// * `p` - Pressure (Pa)
/// * `v` - Volume (m³)
/// * `n` - Number of moles
/// 
/// # Returns
/// Temperature (K)
pub fn ideal_gas_temperature(p: f64, v: f64, n: f64) -> f64 {
    p * v / (n * 8.314_462_618)
}

/// Calculate change in internal energy (ideal gas)
/// 
/// # Arguments
/// * `n` - Number of moles
/// * `cv` - Molar heat capacity at constant volume (J/mol·K)
/// * `dt` - Temperature change (K)
/// 
/// # Returns
/// Change in internal energy (J)
pub fn internal_energy_change(n: f64, cv: f64, dt: f64) -> f64 {
    n * cv * dt
}

/// Calculate work done by gas at constant pressure
/// 
/// # Arguments
/// * `p` - Pressure (Pa)
/// * `v_initial` - Initial volume (m³)
/// * `v_final` - Final volume (m³)
/// 
/// # Returns
/// Work done (J)
pub fn work_isobaric(p: f64, v_initial: f64, v_final: f64) -> f64 {
    p * (v_final - v_initial)
}

/// Calculate heat added at constant pressure
/// 
/// # Arguments
/// * `n` - Number of moles
/// * `cp` - Molar heat capacity at constant pressure (J/mol·K)
/// * `dt` - Temperature change (K)
/// 
/// # Returns
/// Heat added (J)
pub fn heat_isobaric(n: f64, cp: f64, dt: f64) -> f64 {
    n * cp * dt
}

/// Calculate Carnot efficiency
/// 
/// # Arguments
/// * `t_hot` - Hot reservoir temperature (K)
/// * `t_cold` - Cold reservoir temperature (K)
/// 
/// # Returns
/// Efficiency (0-1)
pub fn carnot_efficiency(t_hot: f64, t_cold: f64) -> f64 {
    1.0 - t_cold / t_hot
}

/// Calculate entropy change
/// 
/// # Arguments
/// * `q` - Heat transferred (J)
/// * `t` - Temperature (K)
/// 
/// # Returns
/// Entropy change (J/K)
pub fn entropy_change(q: f64, t: f64) -> f64 {
    q / t
}

/// Calculate thermal expansion (linear)
/// 
/// # Arguments
/// * `l0` - Initial length (m)
/// * `alpha` - Coefficient of linear expansion (1/K)
/// * `dt` - Temperature change (K)
/// 
/// # Returns
/// Change in length (m)
pub fn linear_expansion(l0: f64, alpha: f64, dt: f64) -> f64 {
    l0 * alpha * dt
}

/// Calculate heat transfer by conduction
/// 
/// # Arguments
/// * `k` - Thermal conductivity (W/m·K)
/// * `a` - Cross-sectional area (m²)
/// * `dt` - Temperature difference (K)
/// * `d` - Thickness (m)
/// 
/// # Returns
/// Heat transfer rate (W)
pub fn heat_conduction(k: f64, a: f64, dt: f64, d: f64) -> f64 {
    k * a * dt / d
}

/// Calculate heat transfer by radiation (Stefan-Boltzmann)
/// 
/// # Arguments
/// * `epsilon` - Emissivity (0-1)
/// * `sigma` - Stefan-Boltzmann constant (W/m²·K⁴)
/// * `a` - Surface area (m²)
/// * `t` - Temperature (K)
/// 
/// # Returns
/// Power radiated (W)
pub fn heat_radiation(epsilon: f64, sigma: f64, a: f64, t: f64) -> f64 {
    epsilon * sigma * a * t.powi(4)
}

/// Calculate specific heat
/// 
/// # Arguments
/// * `q` - Heat added (J)
/// * `m` - Mass (kg)
/// * `dt` - Temperature change (K)
/// 
/// # Returns
/// Specific heat (J/kg·K)
pub fn specific_heat(q: f64, m: f64, dt: f64) -> f64 {
    q / (m * dt)
}

/// Convert Celsius to Kelvin
/// 
/// # Arguments
/// * `c` - Temperature in Celsius
/// 
/// # Returns
/// Temperature in Kelvin
pub fn celsius_to_kelvin(c: f64) -> f64 {
    c + 273.15
}

/// Convert Kelvin to Celsius
/// 
/// # Arguments
/// * `k` - Temperature in Kelvin
/// 
/// # Returns
/// Temperature in Celsius
pub fn kelvin_to_celsius(k: f64) -> f64 {
    k - 273.15
}

/// Convert Fahrenheit to Celsius
/// 
/// # Arguments
/// * `f` - Temperature in Fahrenheit
/// 
/// # Returns
/// Temperature in Celsius
pub fn fahrenheit_to_celsius(f: f64) -> f64 {
    (f - 32.0) * 5.0 / 9.0
}

/// Convert Celsius to Fahrenheit
/// 
/// # Arguments
/// * `c` - Temperature in Celsius
/// 
/// # Returns
/// Temperature in Fahrenheit
pub fn celsius_to_fahrenheit(c: f64) -> f64 {
    c * 9.0 / 5.0 + 32.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_celsius_to_kelvin() {
        assert_relative_eq!(celsius_to_kelvin(0.0), 273.15, epsilon = 1e-6);
        assert_relative_eq!(celsius_to_kelvin(100.0), 373.15, epsilon = 1e-6);
    }

    #[test]
    fn test_kelvin_to_celsius() {
        assert_relative_eq!(kelvin_to_celsius(273.15), 0.0, epsilon = 1e-6);
        assert_relative_eq!(kelvin_to_celsius(373.15), 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_carnot_efficiency() {
        assert_relative_eq!(carnot_efficiency(500.0, 300.0), 0.4, epsilon = 1e-6);
    }
}
