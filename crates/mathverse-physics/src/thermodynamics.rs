//! Thermodynamics

use crate::constants::R;

/// Calculate ideal gas law: PV = nRT
/// 
/// # Arguments
/// * `n` - Number of moles
/// * `t` - Temperature (K)
/// * `v` - Volume (m³)
/// 
/// # Returns
/// Pressure (Pa), or `None` if the volume is zero.
pub fn ideal_gas_pressure(n: f64, t: f64, v: f64) -> Option<f64> {
    if v == 0.0 { return None; }
    Some(n * R * t / v)
}

/// Calculate temperature from ideal gas law
/// 
/// # Arguments
/// * `p` - Pressure (Pa)
/// * `v` - Volume (m³)
/// * `n` - Number of moles
/// 
/// # Returns
/// Temperature (K), or `None` if the number of moles is zero.
pub fn ideal_gas_temperature(p: f64, v: f64, n: f64) -> Option<f64> {
    if n == 0.0 { return None; }
    Some(p * v / (n * R))
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
/// Efficiency (0-1), or `None` if `t_hot <= 0` or `t_cold` exceeds `t_hot`.
pub fn carnot_efficiency(t_hot: f64, t_cold: f64) -> Option<f64> {
    if t_hot <= 0.0 || t_cold < 0.0 || t_cold > t_hot {
        return None;
    }
    Some(1.0 - t_cold / t_hot)
}

/// Calculate entropy change
/// 
/// # Arguments
/// * `q` - Heat transferred (J)
/// * `t` - Temperature (K)
/// 
/// # Returns
/// Entropy change (J/K), or `None` if the temperature is zero.
pub fn entropy_change(q: f64, t: f64) -> Option<f64> {
    if t == 0.0 { return None; }
    Some(q / t)
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
/// Heat transfer rate (W), or `None` if the thickness is zero.
pub fn heat_conduction(k: f64, a: f64, dt: f64, d: f64) -> Option<f64> {
    if d == 0.0 { return None; }
    Some(k * a * dt / d)
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
/// Specific heat (J/kg·K), or `None` if mass or temperature change is zero.
pub fn specific_heat(q: f64, m: f64, dt: f64) -> Option<f64> {
    if m == 0.0 || dt == 0.0 { return None; }
    Some(q / (m * dt))
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
#[allow(clippy::unwrap_used)]
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
        assert_relative_eq!(carnot_efficiency(500.0, 300.0).unwrap(), 0.4, epsilon = 1e-6);
        assert!(carnot_efficiency(300.0, 500.0).is_none());
        assert!(carnot_efficiency(0.0, 300.0).is_none());
    }

    #[test]
    fn test_ideal_gas_uses_constant_r() {
        assert_relative_eq!(
            ideal_gas_pressure(1.0, 273.15, 0.0224).unwrap(),
            crate::constants::R * 273.15 / 0.0224,
            epsilon = 1e-9
        );
        assert!(ideal_gas_pressure(1.0, 273.15, 0.0).is_none());
        assert!(ideal_gas_temperature(1.0, 1.0, 0.0).is_none());
    }

    #[test]
    fn test_entropy_change() {
        assert_relative_eq!(entropy_change(100.0, 300.0).unwrap(), 100.0 / 300.0, epsilon = 1e-9);
        assert!(entropy_change(100.0, 0.0).is_none());
    }
}
