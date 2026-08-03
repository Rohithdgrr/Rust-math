# MathVerse Physics

[![Crates.io](https://img.shields.io/crates/v/mathverse-physics.svg)](https://crates.io/crates/mathverse-physics)
[![docs.rs](https://docs.rs/mathverse-physics/badge.svg)](https://docs.rs/mathverse-physics)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Rust: 1.87+](https://img.shields.io/badge/Rust-1.87%2B-EA5727?logo=rust)](https://www.rust-lang.org)

Physics domain applications for MathVerse: mechanics, thermodynamics, electromagnetism, waves, and fundamental constants.

---

## Features

- **Mechanics** — kinematics, energy, momentum, forces, rotational dynamics
- **Thermodynamics** — ideal gas, heat transfer, Carnot cycle, temperature conversions
- **Electromagnetism** — Coulomb's law, capacitance, inductance, magnetic fields, circuits
- **Waves & Optics** — wave equations, Doppler effect, Snell's law, diffraction, lenses
- **Constants** — 17 CODATA fundamental physical constants

## Module Overview

| Module | Items | Description |
|---|---|---|
| `constants` | 17 `pub const` | CODATA physical constants (c, G, ℏ, e, k_B, N_A, …) |
| `mechanics` | 16 functions | Kinematics, energy, forces, rotation |
| `thermodynamics` | 15 functions | Ideal gas, heat, Carnot efficiency, temperature |
| `electromagnetism` | 16 functions | Coulomb, capacitance, inductance, circuits |
| `waves` | 20 functions | Wave equations, Doppler, optics, diffraction |

## Installation

```toml
[dependencies]
mathverse-physics = "0.1"
```

## Quick Start

```rust
use mathverse_physics::*;

fn main() {
    // Mechanics: free-fall from 100m
    let t = 10.0_f64.sqrt();
    let v = mechanics::final_velocity(0.0, mechanics::G_0, t);
    println!("Impact velocity: {v:.2} m/s");

    // Thermodynamics: Carnot engine
    let eta = thermodynamics::carnot_efficiency(600.0, 300.0);
    println!("Carnot efficiency: {eta:.1}%");

    // Electromagnetism: capacitor energy
    let e = electromagnetism::capacitor_energy(1e-6, 12.0);
    println!("Capacitor energy: {e:.6} J");

    // Waves: speed of sound at 20°C
    let v = waves::speed_of_sound_air(20.0);
    println!("Speed of sound: {v:.1} m/s");
}
```

Expected output:

```
Impact velocity: 9.90 m/s
Carnot efficiency: 50.0%
Capacitor energy: 0.000072 J
Speed of sound: 343.0 m/s
```

## Per-Module Reference

### `constants` — Fundamental Constants

| Constant | Value | Description |
|---|---|---|
| `C` | 2.998 × 10⁸ m/s | Speed of light in vacuum |
| `G` | 6.674 × 10⁻¹¹ m³/(kg·s²) | Gravitational constant |
| `H` | 6.626 × 10⁻³⁴ J·s | Planck constant |
| `H_BAR` | 1.055 × 10⁻³⁴ J·s | Reduced Planck constant |
| `E` | 1.602 × 10⁻¹⁹ C | Elementary charge |
| `K_B` | 1.381 × 10⁻²³ J/K | Boltzmann constant |
| `N_A` | 6.022 × 10²³ mol⁻¹ | Avogadro constant |
| `EPSILON_0` | 8.854 × 10⁻¹² F/m | Permittivity of free space |
| `MU_0` | 4π × 10⁻⁷ H/m | Permeability of free space |
| `G_0` | 9.80665 m/s² | Standard gravity |
| `AU` | 1.496 × 10¹¹ m | Astronomical unit |
| `M_E` | 9.109 × 10⁻³¹ kg | Electron mass |
| `M_P` | 1.673 × 10⁻²⁷ kg | Proton mass |
| `M_N` | 1.675 × 10⁻²⁷ kg | Neutron mass |
| `R_INF` | 1.097 × 10⁷ m⁻¹ | Rydberg constant |
| `ALPHA` | 7.297 × 10⁻³ | Fine structure constant |
| `SIGMA` | 5.670 × 10⁻⁸ W/(m²·K⁴) | Stefan-Boltzmann constant |

### `mechanics` — Classical Mechanics

| Function | Formula | Description |
|---|---|---|
| `displacement(v0, a, t)` | s = v₀t + ½at² | Displacement |
| `final_velocity(v0, a, t)` | v = v₀ + at | Final velocity |
| `velocity_from_displacement(v0, a, d)` | v = √(v₀² + 2ad) | Velocity from displacement |
| `kinetic_energy(m, v)` | KE = ½mv² | Kinetic energy |
| `potential_energy(m, h, g)` | PE = mgh | Gravitational PE |
| `momentum(m, v)` | p = mv | Linear momentum |
| `force(m, a)` | F = ma | Newton's 2nd law |
| `work(f, d, theta)` | W = Fd cos θ | Work done |
| `power(w, t)` | P = W/t | Power |
| `centripetal_force(m, v, r)` | F = mv²/r | Centripetal force |
| `gravitational_force(m1, m2, r)` | F = Gm₁m₂/r² | Newton's gravitation |
| `pendulum_period(l, g)` | T = 2π√(l/g) | Simple pendulum |
| `spring_force(k, x)` | F = −kx | Hooke's law |
| `angular_velocity(v, r)` | ω = v/r | Angular velocity |
| `moment_of_inertia_cylinder(m, r)` | I = ½mr² | Solid cylinder |
| `moment_of_inertia_sphere(m, r)` | I = ⅖mr² | Solid sphere |

### `thermodynamics` — Heat & Temperature

| Function | Description |
|---|---|
| `ideal_gas_pressure(n, t, v)` | PV = nRT |
| `ideal_gas_temperature(p, v, n)` | T = PV/(nR) |
| `internal_energy_change(n, cv, dt)` | ΔU = nCvΔT |
| `work_isobaric(p, v1, v2)` | W = P(V₂−V₁) |
| `heat_isobaric(n, cp, dt)` | Q = nCpΔT |
| `carnot_efficiency(t_hot, t_cold)` | η = 1 − T_cold/T_hot |
| `entropy_change(q, t)` | ΔS = Q/T |
| `linear_expansion(l0, alpha, dt)` | ΔL = L₀αΔT |
| `heat_conduction(k, a, dt, d)` | P = kAΔT/d |
| `heat_radiation(epsilon, sigma, a, t)` | P = εσAT⁴ |
| `specific_heat(q, m, dt)` | c = Q/(mΔT) |
| `celsius_to_kelvin(c)` | K = °C + 273.15 |
| `kelvin_to_celsius(k)` | °C = K − 273.15 |
| `fahrenheit_to_celsius(f)` | °C = (°F−32) × 5/9 |
| `celsius_to_fahrenheit(c)` | °F = °C × 9/5 + 32 |

### `electromagnetism` — E&M and Circuits

| Function | Description |
|---|---|
| `electric_field_point(q, r)` | E = q/(4πε₀r²) |
| `coulomb_force(q1, q2, r)` | F = q₁q₂/(4πε₀r²) |
| `electric_potential(q, r)` | V = q/(4πε₀r) |
| `electric_potential_energy(q1, q2, r)` | U = q₁q₂/(4πε₀r) |
| `capacitance_parallel_plate(epsilon, a, d)` | C = εA/d |
| `capacitor_energy(c, v)` | E = ½CV² |
| `magnetic_field_wire(i, r)` | B = μ₀I/(2πr) |
| `magnetic_force_wire(i, l, b, theta)` | F = ILB sin θ |
| `lorentz_force(q, v, b, theta)` | F = qvB sin θ |
| `magnetic_flux(b, a, theta)` | Φ = BA cos θ |
| `induced_emf(d_phi, dt)` | EMF = −dΦ/dt |
| `inductance_solenoid(mu, n, a, l)` | L = μn²Al |
| `inductor_energy(l, i)` | E = ½LI² |
| `resistance_from_resistivity(rho, l, a)` | R = ρL/A |
| `resistor_power(v, r)` | P = V²/R |
| `drift_velocity(i, n, q, a)` | v_d = I/(nqA) |

### `waves` — Waves & Optics

| Function | Description |
|---|---|
| `wave_speed(frequency, wavelength)` | v = fλ |
| `frequency_from_wavelength(speed, wavelength)` | f = v/λ |
| `wavelength_from_frequency(speed, frequency)` | λ = v/f |
| `wave_number(wavelength)` | k = 2π/λ |
| `angular_frequency(frequency)` | ω = 2πf |
| `period(frequency)` | T = 1/f |
| `speed_of_sound_air(temperature)` | v = 331 + 0.6T |
| `doppler_source_moving(f, v_s, v_w)` | Doppler (moving source) |
| `doppler_observer_moving(f, v_o, v_w)` | Doppler (moving observer) |
| `beat_frequency(f1, f2)` | \|f₁ − f₂\| |
| `string_wave_speed(tension, linear_density)` | v = √(T/μ) |
| `string_fundamental_frequency(length, tension, linear_density)` | f₁ = v/(2L) |
| `light_speed_medium(refractive_index)` | v = c/n |
| `snells_law(n1, theta1, n2)` | Snell's law |
| `critical_angle(n1, n2)` | θ_c = arcsin(n₂/n₁) |
| `lens_focal_length(n, r1, r2)` | Lens maker equation |
| `thin_lens_equation(f, d_o)` | 1/f = 1/d_o + 1/d_i |
| `magnification(d_i, d_o)` | M = −d_i/d_o |
| `single_slit_diffraction(m, wavelength, slit_width)` | θ = arcsin(mλ/a) |
| `double_slit_interference(m, wavelength, slit_separation)` | θ = arcsin(mλ/d) |

## Dependencies

- `mathverse-core`
- `mathverse-calculus`
- `mathverse-algebra`
- `mathverse-trigonometry`
- `mathverse-vector`
- `mathverse-units`

## Future Scope

- Relativistic mechanics (Lorentz factors, time dilation)
- Quantum mechanics (wave functions, Schrödinger equation)
- Statistical mechanics (Maxwell-Boltzmann, Fermi-Dirac)
- Fluid dynamics (Navier-Stokes, Bernoulli)
- Nuclear physics (decay chains, binding energy)

## License

MIT OR Apache-2.0 — see [LICENSE](LICENSE) for details.
