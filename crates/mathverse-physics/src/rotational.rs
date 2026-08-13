//! Rotational dynamics.
//!
//! Type-safe rotational dynamics quantities with compile-time unit checking.
//!
//! # Example
//!
//! ```
//! use mathverse_physics::rotational::{Torque, MomentOfInertia, AngularVelocity, rotational_kinetic_energy};
//! use mathverse_units::si::{Radian, Newton, Kilogram, Watt, Second};
//!
//! let torque = Torque::from_force_at_distance(1.0.m, Force::new(10.0), 0.0);
//! let angular_vel = AngularVelocity::new(2.0.rad_s());
//! let ke = rotational_kinetic_energy(I, angular_vel);
//! ```
//
//! #![allow(unused_imports)] // re-exports pull in all submodules

use mathverse_units::quantity::Quantity;
use mathverse_units::si::{Radian, Meter, Newton, Kilogram, Watt, Second, Joule};
use mathverse_units::dimensions::{AngleDim, LengthDim, MassDim, TimeDim, ForceDim, EnergyDim, PowerDim};
use crate::Force;
use crate::Work;
use crate::Power;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, Div};

/// Angle in radians.
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::Angle;
/// use mathverse_units::si::Radian;
///
/// let angle: Angle = 1.0.radians();
/// let deg = angle.to_degrees();
/// assert!((deg - 57.29578).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Angle(pub Quantity<AngleDim, Radian>);

impl Angle {
    /// Create a new angle from a value in radians.
    pub fn new(value: f64) -> Self {
        Self(Quantity::new(value))
    }

    /// Get the raw value in radians.
    pub fn value(&self) -> f64 {
        self.0.value()
    }

    /// Convert to degrees.
    pub fn to_degrees(self) -> f64 {
        self.0.value * 180.0 / std::f64::consts::PI
    }

    /// Convert from degrees.
    pub fn from_degrees(deg: f64) -> Self {
        Self::new(deg * std::f64::consts::PI / 180.0)
    }
}

impl Default for Angle {
    fn default() -> Self {
        Self(Quantity::new(0.0))
    }
}

impl From<f64> for Angle {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

/// Angular velocity in radians per second.
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::AngularVelocity;
/// use mathverse_units::si::Radian;
///
/// let av: AngularVelocity = 2.0.rad_s();
/// assert_eq!(av.value(), 2.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AngularVelocity(pub Quantity<mathverse_units::dimensions::VelocityDim, Radian>);

impl AngularVelocity {
    /// Create a new angular velocity from a value in rad/s.
    pub fn new(value: f64) -> Self {
        Self(Quantity::new(value))
    }

    /// Get the raw value in radians per second.
    pub fn value(&self) -> f64 {
        self.0.value()
    }

    /// Convert to revolutions per minute (RPM).
    pub fn to_rpm(self) -> f64 {
        self.0.value * 60.0 / std::f64::consts::PI
    }

    /// Check if angular velocity is finite.
    pub fn is_finite(&self) -> bool {
        self.0.value.is_finite()
    }
}

impl Default for AngularVelocity {
    fn default() -> Self {
        Self(Quantity::new(0.0))
    }
}

impl From<f64> for AngularVelocity {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

/// Angular acceleration in radians per second squared.
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::AngularAcceleration;
/// use mathverse_units::si::Radian;
///
/// let aa: AngularAcceleration = 1.0.rad_s2();
/// assert_eq!(aa.value(), 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AngularAcceleration(pub Quantity<mathverse_units::dimensions::AccelerationDim, Radian>);

impl AngularAcceleration {
    /// Create a new angular acceleration from a value in rad/s².
    pub fn new(value: f64) -> Self {
        Self(Quantity::new(value))
    }

    /// Get the raw value in radians per second squared.
    pub fn value(&self) -> f64 {
        self.0.value()
    }

    /// Check if angular acceleration is finite.
    pub fn is_finite(&self) -> bool {
        self.0.value.is_finite()
    }
}

impl Default for AngularAcceleration {
    fn default() -> Self {
        Self(Quantity::new(0.0))
    }
}

impl From<f64> for AngularAcceleration {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

/// Torque in newton-meters (N·m).
///
/// Dimension: ForceDim × LengthDim = M·L²/T² (same as EnergyDim),
/// but physically represents torque (moment of force).
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::Torque;
/// use mathverse_units::si::Newton;
///
/// let torque: Torque = 5.0.nm();
/// assert_eq!(torque.value(), 5.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Torque(pub Quantity<EnergyDim, Newton>); // N·m, but we use Newton for unit branding

impl Torque {
    /// Create a new torque from a value in newton-meters.
    pub fn new(value: f64) -> Self {
        Self(Quantity::new(value))
    }

    /// Get the raw value in newton-meters.
    pub fn value(&self) -> f64 {
        self.0.value()
    }

    /// Convert to mewton-meters (1e-3 N·m).
    pub fn to_mewnmeter(self) -> f64 {
        self.0.value * 1e-3
    }

    /// Check if torque is finite.
    pub fn is_finite(&self) -> bool {
        self.0.value.is_finite()
    }

    /// Compute torque: τ = r · F · sin(θ)
    ///
    /// # Arguments
    /// * `distance` - Distance from pivot (f64 in meters)
    /// * `force` - Force applied
    /// * `angle` - Angle between position vector and force (radians)
    ///
    /// # Returns
    /// Torque (Torque)
    ///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::Torque;
/// use mathverse_physics::dynamics::Force;
/// use mathverse_units::si::Meter;
///
/// let torque = Torque::from_force_at_distance(1.0, Force::new(10.0), std::f64::consts::PI / 2.0);
/// assert_eq!(torque.value(), 10.0);
/// ```
    pub fn from_force_at_distance(
        distance: f64,
        force: Force,
        angle: f64,
    ) -> Self {
        // τ = r · F · sin(θ)
        let t = distance * force.0.value * angle.sin();
        Self::new(t)
    }

    /// Compute torque from moment of inertia and angular acceleration: τ = I · α
    ///
    /// # Arguments
    /// * `moment_of_inertia` - MomentOfInertia
    /// * `angular_acceleration` - AngularAcceleration
    ///
    /// # Returns
    /// Torque (Torque)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::rotational::{Torque, MomentOfInertia, AngularAcceleration};
    ///
    /// let torque = Torque::from_moment_of_inertia(I, aa);
    /// ```
    pub fn from_moment_of_inertia(
        moment_of_inertia: MomentOfInertia,
        angular_acceleration: AngularAcceleration,
    ) -> Self {
        // τ = I · α
        let t = moment_of_inertia.0.value * angular_acceleration.0.value;
        Self::new(t)
    }
}

impl Default for Torque {
    fn default() -> Self {
        Self(Quantity::new(0.0))
    }
}

impl From<f64> for Torque {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

/// Moment of inertia in kilogram-meter² (kg·m²).
///
/// Dimension: MassDim × LengthDim² = M · L²
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::MomentOfInertia;
/// use mathverse_units::si::Kilogram;
///
/// let io: MomentOfInertia = 1.0.kg_m2();
/// assert_eq!(io.value(), 1.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MomentOfInertia(pub Quantity<mathverse_units::dimensions::MassDim, Newton>); // kg·m² branding

impl MomentOfInertia {
    /// Create a new moment of inertia from a value in kg·m².
    pub fn new(value: f64) -> Self {
        Self(Quantity::new(value))
    }

    /// Get the raw value in kg·m².
    pub fn value(&self) -> f64 {
        self.0.value()
    }

    /// Moment of inertia for a solid cylinder about its central axis: I = ½ m r²
    ///
    /// # Arguments
    /// * `mass` - Mass (Quantity<MassDim, Kilogram>)
    /// * `radius` - Radius (f64 in meters)
    ///
    /// # Returns
    /// Moment of inertia (MomentOfInertia)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::rotational::MomentOfInertia;
    /// use mathverse_units::si::Kilogram;
    ///
    /// let io = MomentOfInertia::solid_cylinder(2.0.kg(), 3.0.m());
    /// assert!((io.value() - 9.0).abs() < 1e-6); // 0.5 * 2 * 9 = 9
    /// ```
    pub fn solid_cylinder(
        mass: mathverse_units::quantity::Quantity<mathverse_units::dimensions::MassDim, Kilogram>,
        radius: f64,
    ) -> Self {
        Self::new(0.5 * mass.value * radius.powi(2))
    }

    /// Moment of inertia for a solid sphere about its central axis: I = ²/₅ m r²
    ///
    /// # Arguments
    /// * `mass` - Mass (Quantity<MassDim, Kilogram>)
    /// * `radius` - Radius (f64 in meters)
    ///
    /// # Returns
    /// Moment of inertia (MomentOfInertia)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::rotational::MomentOfInertia;
    /// use mathverse_units::si::Kilogram;
    ///
    /// let io = MomentOfInertia::solid_sphere(2.0.kg(), 3.0.m());
    /// assert!((io.value() - 3.6).abs() < 1e-6); // (2/5) * 2 * 9 = 3.6
    /// ```
    pub fn solid_sphere(
        mass: mathverse_units::quantity::Quantity<mathverse_units::dimensions::MassDim, Kilogram>,
        radius: f64,
    ) -> Self {
        Self::new((2.0 / 5.0) * mass.value * radius.powi(2))
    }

    /// Moment of inertia for a thin hoop about its central axis: I = m r²
    ///
    /// # Arguments
    /// * `mass` - Mass (Quantity<MassDim, Kilogram>)
    /// * `radius` - Radius (f64 in meters)
    ///
    /// # Returns
    /// Moment of inertia (MomentOfInertia)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::rotational::MomentOfInertia;
    /// use mathverse_units::si::Kilogram;
    ///
    /// let io = MomentOfInertia::thin_hoop(2.0.kg(), 3.0.m());
    /// assert_eq!(io.value(), 18.0); // 2 * 9 = 18
    /// ```
    pub fn thin_hoop(
        mass: mathverse_units::quantity::Quantity<mathverse_units::dimensions::MassDim, Kilogram>,
        radius: f64,
    ) -> Self {
        Self::new(mass.value * radius.powi(2))
    }

    /// Moment of inertia for a point mass at distance r: I = m r²
    ///
    /// # Arguments
    /// * `mass` - Mass (Quantity<MassDim, Kilogram>)
    /// * `radius` - Radius (f64 in meters)
    ///
    /// # Returns
    /// Moment of inertia (MomentOfInertia)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::rotational::MomentOfInertia;
    /// use mathverse_units::si::Kilogram;
    ///
    /// let io = MomentOfInertia::point_mass(2.0.kg(), 3.0.m());
    /// assert_eq!(io.value(), 18.0); // 2 * 9 = 18
    /// ```
    pub fn point_mass(
        mass: mathverse_units::quantity::Quantity<mathverse_units::dimensions::MassDim, Kilogram>,
        radius: f64,
    ) -> Self {
        Self::new(mass.value * radius.powi(2))
    }

    /// Rotational kinetic energy: KE_rot = ½ · I · ω²
    ///
    /// # Arguments
    /// * `self` - This moment of inertia
    /// * `angular_velocity` - AngularVelocity
    ///
    /// # Returns
    /// RotationalKineticEnergy
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::rotational::{MomentOfInertia, AngularVelocity, rotational_kinetic_energy};
    ///
    /// let io = MomentOfInertia::new(2.0);
    /// let av = AngularVelocity::new(3.0);
    /// let ke = io.rotational_kinetic_energy(av);
    /// assert_eq!(ke.value(), 9.0); // 0.5 * 2 * 9 = 9
    /// ```
    pub fn rotational_kinetic_energy(
        self,
        angular_velocity: AngularVelocity,
    ) -> RotationalKineticEnergy {
        // KE_rot = ½ · I · ω²
        let ke = 0.5 * self.0.value * angular_velocity.0.value.powi(2);
        RotationalKineticEnergy::new(ke)
    }

    /// Compute torque from force and distance: τ = r · F · sin(θ)
    pub fn from_force_and_distance(
        distance: f64,
        force: Force,
        angle: f64,
    ) -> Self {
        // τ = r · F · sin(θ)
        let t = distance * force.0.value * angle.sin();
        Self::new(t)
    }
}

impl Default for MomentOfInertia {
    fn default() -> Self {
        Self(Quantity::new(0.0))
    }
}

impl From<f64> for MomentOfInertia {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

/// Rotational kinetic energy in joules (J).
///
/// Dimension: EnergyDim (M·L²/T²), same as work/energy.
/// Distinct from translational KE at the type level.
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::RotationalKineticEnergy;
/// use mathverse_units::si::Joule;
///
/// let ke: RotationalKineticEnergy = 42.0.joules();
/// assert_eq!(ke.value(), 42.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RotationalKineticEnergy(pub Quantity<EnergyDim, Joule>);

impl RotationalKineticEnergy {
    /// Create a new rotational kinetic energy from a value in joules.
    pub fn new(value: f64) -> Self {
        Self(Quantity::new(value))
    }

    /// Get the raw value in joules.
    pub fn value(&self) -> f64 {
        self.0.value()
    }

    /// Compute rotational kinetic energy: KE_rot = ½ · I · ω²
    ///
    /// # Arguments
    /// * `moment_of_inertia` - MomentOfInertia
    /// * `angular_velocity` - AngularVelocity
    ///
    /// # Returns
    /// RotationalKineticEnergy
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::rotational::{MomentOfInertia, AngularVelocity, rotational_kinetic_energy};
    ///
    /// let io = MomentOfInertia::new(2.0);
    /// let av = AngularVelocity::new(3.0);
    /// let ke = rotational_kinetic_energy(io, av);
    /// assert_eq!(ke.value(), 9.0); // 0.5 * 2 * 9 = 9
    /// ```
    pub fn from_moment_and_angular_velocity(
        moment_of_inertia: MomentOfInertia,
        angular_velocity: AngularVelocity,
    ) -> Self {
        // KE_rot = ½ · I · ω²
        let ke = 0.5 * moment_of_inertia.0.value * angular_velocity.0.value.powi(2);
        Self::new(ke)
    }

    /// Compute angular velocity from rotational kinetic energy and moment of inertia:
    /// ω = sqrt(2 · KE / I)
    ///
    /// # Arguments
    /// * `energy` - RotationalKineticEnergy
    /// * `moment_of_inertia` - MomentOfInertia
    ///
    /// # Returns
    /// AngularVelocity
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::rotational::{RotationalKineticEnergy, MomentOfInertia, AngularVelocity, angular_velocity_from_energy};
    ///
    /// let energy = RotationalKineticEnergy::new(9.0);
    /// let io = MomentOfInertia::new(2.0);
    /// let av = angular_velocity_from_energy(energy, io);
    /// assert_eq!(av.value(), 3.0); // sqrt(2*9/2) = sqrt(9) = 3
    /// ```
    pub fn angular_velocity_from_energy(
        energy: RotationalKineticEnergy,
        moment_of_inertia: MomentOfInertia,
    ) -> AngularVelocity {
        // ω = sqrt(2 · E / I)
        let omega_sq = 2.0 * energy.0.value / moment_of_inertia.0.value;
        let omega = omega_sq.sqrt();
        AngularVelocity::new(omega)
    }
}

impl Default for RotationalKineticEnergy {
    fn default() -> Self {
        Self(Quantity::new(0.0))
    }
}

impl From<f64> for RotationalKineticEnergy {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

/// Compute torque from moment of inertia and angular acceleration: τ = I · α
///
/// # Arguments
/// * `moment_of_inertia` - MomentOfInertia
/// * `angular_acceleration` - AngularAcceleration
///
/// # Returns
/// Torque (Torque)
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::{compute_torque, MomentOfInertia, AngularAcceleration};
///
/// let torque = compute_torque(I, aa);
/// ```
pub fn compute_torque(
    moment_of_inertia: MomentOfInertia,
    angular_acceleration: AngularAcceleration,
) -> Torque {
    Torque::from_moment_of_inertia(moment_of_inertia, angular_acceleration)
}

/// Compute rotational kinetic energy: KE_rot = ½ · I · ω²
///
/// # Arguments
/// * `moment_of_inertia` - MomentOfInertia
/// * `angular_velocity` - AngularVelocity
///
/// # Returns
/// RotationalKineticEnergy
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::{compute_rotational_kinetic_energy, MomentOfInertia, AngularVelocity};
///
/// let ke = compute_rotational_kinetic_energy(I, av);
/// assert_eq!(ke.value(), 9.0);
/// ```
pub fn compute_rotational_kinetic_energy(
    moment_of_inertia: MomentOfInertia,
    angular_velocity: AngularVelocity,
) -> RotationalKineticEnergy {
    moment_of_inertia.rotational_kinetic_energy(angular_velocity)
}

/// Compute work done by torque (angle in radians): W = τ · θ
///
/// # Arguments
/// * `torque` - Torque
/// * `angle` - Angle in radians (Angle)
///
/// # Returns
/// Work (Work)
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::{compute_work_by_torque, Torque, Angle};
///
/// let work = compute_work_by_torque(torque, angle);
/// ```
pub fn compute_work_by_torque(
    torque: Torque,
    angle: Angle,
) -> Work {
    // W = τ · θ (for constant torque, θ in radians)
    let w = torque.0.value * angle.0.value;
    Work::new(w)
}

/// Compute power from torque and angular velocity: P = τ · ω
///
/// # Arguments
/// * `torque` - Torque
/// * `angular_velocity` - AngularVelocity
///
/// # Returns
/// Power (Power)
///
/// # Example
///
/// ```
/// use mathverse_physics::rotational::{compute_power_from_torque, Torque, AngularVelocity};
///
/// let power = compute_power_from_torque(torque, av);
/// ```
pub fn compute_power_from_torque(
    torque: Torque,
    angular_velocity: AngularVelocity,
) -> Power {
    // P = τ · ω
    let p = torque.0.value * angular_velocity.0.value;
    Power::new(p)
}