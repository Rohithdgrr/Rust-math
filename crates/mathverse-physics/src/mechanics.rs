//! Classical mechanics - kinematics and dynamics

use crate::constants::G_0;

/// Calculate displacement with constant acceleration
/// 
/// # Arguments
/// * `v0` - Initial velocity (m/s)
/// * `a` - Acceleration (m/s²)
/// * `t` - Time (s)
/// 
/// # Returns
/// Displacement (m)
pub fn displacement(v0: f64, a: f64, t: f64) -> f64 {
    v0 * t + 0.5 * a * t * t
}

/// Calculate final velocity with constant acceleration
/// 
/// # Arguments
/// * `v0` - Initial velocity (m/s)
/// * `a` - Acceleration (m/s²)
/// * `t` - Time (s)
/// 
/// # Returns
/// Final velocity (m/s)
pub fn final_velocity(v0: f64, a: f64, t: f64) -> f64 {
    v0 + a * t
}

/// Calculate velocity from displacement and acceleration
/// 
/// # Arguments
/// * `v0` - Initial velocity (m/s)
/// * `a` - Acceleration (m/s²)
/// * `d` - Displacement (m)
/// 
/// # Returns
/// Final velocity (m/s)
pub fn velocity_from_displacement(v0: f64, a: f64, d: f64) -> f64 {
    (v0 * v0 + 2.0 * a * d).sqrt()
}

/// Calculate kinetic energy
/// 
/// # Arguments
/// * `m` - Mass (kg)
/// * `v` - Velocity (m/s)
/// 
/// # Returns
/// Kinetic energy (J)
pub fn kinetic_energy(m: f64, v: f64) -> f64 {
    0.5 * m * v * v
}

/// Calculate potential energy (gravitational)
/// 
/// # Arguments
/// * `m` - Mass (kg)
/// * `h` - Height (m)
/// * `g` - Gravitational acceleration (default: 9.80665 m/s²)
/// 
/// # Returns
/// Potential energy (J)
pub fn potential_energy(m: f64, h: f64, g: Option<f64>) -> f64 {
    m * h * g.unwrap_or(G_0)
}

/// Calculate momentum
/// 
/// # Arguments
/// * `m` - Mass (kg)
/// * `v` - Velocity (m/s)
/// 
/// # Returns
/// Momentum (kg·m/s)
pub fn momentum(m: f64, v: f64) -> f64 {
    m * v
}

/// Calculate force from mass and acceleration
/// 
/// # Arguments
/// * `m` - Mass (kg)
/// * `a` - Acceleration (m/s²)
/// 
/// # Returns
/// Force (N)
pub fn force(m: f64, a: f64) -> f64 {
    m * a
}

/// Calculate work done
/// 
/// # Arguments
/// * `f` - Force (N)
/// * `d` - Displacement (m)
/// * `theta` - Angle between force and displacement (radians)
/// 
/// # Returns
/// Work done (J)
pub fn work(f: f64, d: f64, theta: f64) -> f64 {
    f * d * theta.cos()
}

/// Calculate power
/// 
/// # Arguments
/// * `w` - Work done (J)
/// * `t` - Time (s)
/// 
/// # Returns
/// Power (W)
pub fn power(w: f64, t: f64) -> f64 {
    w / t
}

/// Calculate centripetal force
/// 
/// # Arguments
/// * `m` - Mass (kg)
/// * `v` - Velocity (m/s)
/// * `r` - Radius (m)
/// 
/// # Returns
/// Centripetal force (N)
pub fn centripetal_force(m: f64, v: f64, r: f64) -> f64 {
    m * v * v / r
}

/// Calculate gravitational force between two masses
/// 
/// # Arguments
/// * `m1` - First mass (kg)
/// * `m2` - Second mass (kg)
/// * `r` - Distance between masses (m)
/// 
/// # Returns
/// Gravitational force (N)
pub fn gravitational_force(m1: f64, m2: f64, r: f64) -> f64 {
    crate::constants::G * m1 * m2 / (r * r)
}

/// Calculate period of a simple pendulum
/// 
/// # Arguments
/// * `l` - Length of pendulum (m)
/// * `g` - Gravitational acceleration (default: 9.80665 m/s²)
/// 
/// # Returns
/// Period (s)
pub fn pendulum_period(l: f64, g: Option<f64>) -> f64 {
    2.0 * std::f64::consts::PI * (l / g.unwrap_or(G_0)).sqrt()
}

/// Calculate spring force (Hooke's law)
/// 
/// # Arguments
/// * `k` - Spring constant (N/m)
/// * `x` - Displacement from equilibrium (m)
/// 
/// # Returns
/// Spring force (N)
pub fn spring_force(k: f64, x: f64) -> f64 {
    -k * x
}

/// Calculate angular velocity
/// 
/// # Arguments
/// * `v` - Linear velocity (m/s)
/// * `r` - Radius (m)
/// 
/// # Returns
/// Angular velocity (rad/s)
pub fn angular_velocity(v: f64, r: f64) -> f64 {
    v / r
}

/// Calculate moment of inertia for a solid cylinder
/// 
/// # Arguments
/// * `m` - Mass (kg)
/// * `r` - Radius (m)
/// 
/// # Returns
/// Moment of inertia (kg·m²)
pub fn moment_of_inertia_cylinder(m: f64, r: f64) -> f64 {
    0.5 * m * r * r
}

/// Calculate moment of inertia for a solid sphere
/// 
/// # Arguments
/// * `m` - Mass (kg)
/// * `r` - Radius (m)
/// 
/// # Returns
/// Moment of inertia (kg·m²)
pub fn moment_of_inertia_sphere(m: f64, r: f64) -> f64 {
    (2.0 / 5.0) * m * r * r
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_displacement() {
        assert_relative_eq!(displacement(0.0, 9.8, 2.0), 19.6, epsilon = 1e-6);
    }

    #[test]
    fn test_final_velocity() {
        assert_relative_eq!(final_velocity(10.0, 2.0, 5.0), 20.0, epsilon = 1e-6);
    }

    #[test]
    fn test_kinetic_energy() {
        assert_relative_eq!(kinetic_energy(10.0, 5.0), 125.0, epsilon = 1e-6);
    }

    #[test]
    fn test_momentum() {
        assert_relative_eq!(momentum(10.0, 5.0), 50.0, epsilon = 1e-6);
    }

    #[test]
    fn test_force() {
        assert_relative_eq!(force(10.0, 5.0), 50.0, epsilon = 1e-6);
    }
}
