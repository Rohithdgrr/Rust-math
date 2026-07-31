//! Physical constants

use std::f64::consts::PI;

/// Speed of light in vacuum (m/s)
pub const C: f64 = 299_792_458.0;

/// Gravitational constant (m³/kg·s²)
pub const G: f64 = 6.674_30e-11;

/// Planck constant (J·s)
pub const H: f64 = 6.626_070_15e-34;

/// Reduced Planck constant (J·s)
pub const H_BAR: f64 = H / (2.0 * PI);

/// Elementary charge (C)
pub const E: f64 = 1.602_176_634e-19;

/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.380_649e-23;

/// Avogadro constant (mol⁻¹)
pub const N_A: f64 = 6.022_140_76e23;

/// Permittivity of free space (F/m)
pub const EPSILON_0: f64 = 8.854_187_812_8e-12;

/// Permeability of free space (H/m)
pub const MU_0: f64 = 4.0 * PI * 1.0e-7;

/// Standard gravitational acceleration (m/s²)
pub const G_0: f64 = 9.806_65;

/// Astronomical unit (m)
pub const AU: f64 = 1.495_978_707e11;

/// Electron mass (kg)
pub const M_E: f64 = 9.109_383_701_5e-31;

/// Proton mass (kg)
pub const M_P: f64 = 1.672_621_923_69e-27;

/// Neutron mass (kg)
pub const M_N: f64 = 1.674_927_498_04e-27;

/// Rydberg constant (m⁻¹)
pub const R_INF: f64 = 10_973_731.568_160;

/// Fine structure constant
pub const ALPHA: f64 = 7.297_352_569_3e-3;

/// Stefan-Boltzmann constant (W/m²·K⁴)
pub const SIGMA: f64 = 5.670_374_419e-8;
