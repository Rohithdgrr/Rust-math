//! Dynamics: motion caused by forces.
//!
//! Type-safe dynamics quantities with compile-time unit checking.
//! Enforces physical laws at compile time through dimensional analysis.
//! F = m·a, W = F·d, P = dW/dt, etc.
//!
//! # Example
//!
//! ```
//! use mathverse_physics::dynamics::{Force, Work, Power, compute_force, compute_work};
//! use mathverse_units::si::{Newton, Joule, Watt, Kilogram};
//!
//! let force = Force::from_mass_acceleration(1.0.kg(), 2.0.mps2());
//! let work = Work::from_force_displacement(force, 3.0.m());
//! let power = Work::power_from_time(work, 1.0.s);
//! ```
//
//! #![allow(unused_imports)] // re-exports pull in all submodules

use mathverse_units::quantity::Quantity;
use mathverse_units::si::{Meter, Kilogram, Newton, Joule, Watt};
use mathverse_units::dimensions::{LengthDim, MassDim, TimeDim, ForceDim, EnergyDim, PowerDim, AccelerationDim};
use crate::Position;
use crate::Velocity;
use crate::Time;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, Div};

/// Force in newtons.
///
/// Implements Newton's second law: F = m · a.
/// The dimension is ForceDim (M·L/T²).
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::Force;
/// use mathverse_units::si::{Kilogram, mps2};
///
/// let force = Force::from_mass_acceleration(1.0.kg(), 2.0.mps2());
/// assert_eq!(force.value(), 2.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Force(pub Quantity<ForceDim, Newton>);

impl Force {
    /// Create a new force from a value in newtons.
    pub fn new(value: f64) -> Self {
        Self(Quantity::new(value))
    }

    /// Get the raw value in newtons.
    pub fn value(&self) -> f64 {
        self.0.value()
    }

    /// Compute force from mass and acceleration: F = m · a
    ///
    /// # Arguments
    /// * `mass` - Mass (Quantity<MassDim, Kilogram>)
    /// * `acceleration` - Acceleration
    ///
    /// # Returns
    /// Force (Force)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::Force;
    /// use mathverse_units::si::{Kilogram, mps2};
    ///
    /// let force = Force::from_mass_acceleration(1.0.kg(), 2.0.mps2());
    /// assert_eq!(force.value(), 2.0);
    /// ```
    pub fn from_mass_acceleration(
        mass: mathverse_units::quantity::Quantity<MassDim, Kilogram>,
        acceleration: mathverse_units::quantity::Quantity<AccelerationDim, Meter>,
    ) -> Self {
        // F = m · a
        let f = mass.value * acceleration.value;
        Self::new(f)
    }

    /// Compute work done by this force over a displacement:
    /// W = F · d · cos(θ)
    ///
    /// # Arguments
    /// * `displacement` - Displacement (Position)
    /// * `angle` - Angle between force and displacement (radians)
    ///
    /// # Returns
    /// Work (Work)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Force, Position, Work};
    /// use mathverse_units::si::{Newton, Meter};
    ///
    /// let force = Force::new(10.0);
    /// let displacement = Position::new(3.0);
    /// let work = force.work_over(displacement, 0.0);
    /// assert_eq!(work.value(), 30.0);
    /// ```
    pub fn work_over(
        &self,
        displacement: Position,
        angle: f64,
    ) -> Work {
        // W = F · d · cos(θ)
        let work_val = self.0.value * displacement.0.value * angle.cos();
        Work::new(work_val)
    }

    /// Compute power: P = F · v (dot product, for parallel force and velocity)
    ///
    /// # Arguments
    /// * `velocity` - Velocity
    ///
    /// # Returns
    /// Power (Power)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::Force;
    /// use mathverse_physics::kinematics::Velocity;
    /// use mathverse_units::si::Watt;
    ///
    /// let force = Force::new(10.0);
    /// let velocity = Velocity::new(2.0);
    /// let power = force.power_from_velocity(velocity);
    /// assert_eq!(power.value(), 20.0);
    /// ```
    pub fn power_from_velocity(&self, velocity: Velocity) -> Power {
        // P = F · v (for parallel force and velocity)
        let power_val = self.0.value * velocity.0.value;
        Power::new(power_val)
    }
}

impl Default for Force {
    fn default() -> Self {
        Self(Quantity::new(0.0))
    }
}

impl From<f64> for Force {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

// Arithmetic operations that preserve the Force dimension

impl Add for Force {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Force::new(self.0.value + other.0.value)
    }
}

impl Add<f64> for Force {
    type Output = Self;
    fn add(self, scalar: f64) -> Self::Output {
        Force::new(self.0.value + scalar)
    }
}

impl Add<Quantity<ForceDim, Newton>> for Force {
    type Output = Self;
    fn add(self, other: Quantity<ForceDim, Newton>) -> Self::Output {
        Force::new(self.0.value + other.value)
    }
}

impl Sub for Force {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Force::new(self.0.value - other.0.value)
    }
}

impl Sub<f64> for Force {
    type Output = Self;
    fn sub(self, scalar: f64) -> Self::Output {
        Force::new(self.0.value - scalar)
    }
}

impl Sub<Quantity<ForceDim, Newton>> for Force {
    type Output = Self;
    fn sub(self, other: Quantity<ForceDim, Newton>) -> Self::Output {
        Force::new(self.0.value - other.value)
    }
}

impl Mul<f64> for Force {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self::Output {
        Force::new(self.0.value * scalar)
    }
}

impl Div<f64> for Force {
    type Output = Self;
    fn div(self, scalar: f64) -> Self::Output {
        Force::new(self.0.value / scalar)
    }
}

// Scale force by a dimensionless factor
impl AddAssign for Force {
    fn add_assign(&mut self, other: Self) {
        self.0 = Force::new(self.0.value + other.0.value).0;
    }
}

impl SubAssign for Force {
    fn sub_assign(&mut self, other: Self) {
        self.0 = Force::new(self.0.value - other.0.value).0;
    }
}

/// Work in joules.
///
/// Work done by a force over a distance. The dimension is EnergyDim
/// (M·L²/T²), same as energy.
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::Work;
/// use mathverse_units::si::Joule;
///
/// let work: Work = 100.0.joules();
/// assert_eq!(work.value(), 100.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Work(pub Quantity<EnergyDim, Joule>);

impl Work {
    /// Create a new work from a value in joules.
    pub fn new(value: f64) -> Self {
        Self(Quantity::new(value))
    }

    /// Get the raw value in joules.
    pub fn value(&self) -> f64 {
        self.0.value()
    }

    /// Compute work from force and displacement: W = F · d (parallel)
    ///
    /// # Arguments
    /// * `force` - Force
    /// * `displacement` - Displacement (Position)
    ///
    /// # Returns
    /// Work (Work)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Force, Position, Work};
    /// use mathverse_units::si::{Newton, Meter};
    ///
    /// let force = Force::new(10.0);
    /// let displacement = Position::new(3.0);
    /// let work = Work::from_force_displacement(force, displacement);
    /// assert_eq!(work.value(), 30.0);
    /// ```
    pub fn from_force_displacement(
        force: Force,
        displacement: Position,
    ) -> Self {
        // W = F · d (assuming force parallel to displacement, cos(θ) = 1)
        let w = force.0.value * displacement.0.value;
        Self::new(w)
    }

    /// Compute work from force, displacement, and angle: W = F · d · cos(θ)
    ///
    /// # Arguments
    /// * `force` - Force
    /// * `displacement` - Displacement (Position)
    /// * `theta` - Angle in radians between force and displacement
    ///
    /// # Returns
    /// Work (Work)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Force, Position, Work};
    /// use mathverse_units::si::{Newton, Meter};
    ///
    /// let force = Force::new(10.0);
    /// let displacement = Position::new(3.0);
    /// let work = Work::from_force_displacement_angle(force, displacement, 0.0);
    /// assert_eq!(work.value(), 30.0);
    /// ```
    pub fn from_force_displacement_angle(
        force: Force,
        displacement: Position,
        theta: f64,
    ) -> Self {
        // W = F · d · cos(θ)
        let w = force.0.value * displacement.0.value * theta.cos();
        Self::new(w)
    }

    /// Compute change in kinetic energy (work-energy theorem)
    ///
    /// # Arguments
    /// * `initial_kinetic_energy` - Initial kinetic energy (Work)
    /// * `net_work` - Net work done on the system (Work)
    ///
    /// # Returns
    /// Final kinetic energy (Work)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Work};
    ///
    /// let initial = Work::new(0.0);
    /// let net = Work::new(50.0);
    /// let final = Work::from_kinetic_energy_change(initial, net);
    /// assert_eq!(final.value(), 50.0);
    /// ```
    pub fn from_kinetic_energy_change(
        initial_kinetic_energy: Work,
        net_work: Work,
    ) -> Self {
        // W_net = ΔKE = KE_final - KE_initial
        // KE_final = KE_initial + W_net
        let final_val = initial_kinetic_energy.0.value + net_work.0.value;
        Self::new(final_val)
    }

    /// Compute power from work and time: P = W / t
    ///
    /// # Arguments
    /// * `time` - Time
    ///
    /// # Returns
    /// Power (Power)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Work, Power};
    /// use mathverse_units::si::Watt;
    ///
    /// let work = Work::new(100.0);
    /// let power = Work::power_from_time(work, 10.0);
    /// assert_eq!(power.value(), 10.0);
    /// ```
    pub fn power_from_time(&self, time: Time) -> Power {
        // P = W / t — handle zero time to avoid division by zero
        if time.value == 0.0 {
            return Power::new(0.0);
        }
        let p = self.0.value / time.value;
        Power::new(p)
    }

    /// Compute work from power and time: W = P · t
    ///
    /// # Arguments
    /// * `power` - Power
    /// * `time` - Time
    ///
    /// # Returns
    /// Work (Work)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Power, Work};
    /// use mathverse_units::si::Joule;
    ///
    /// let power = Power::new(10.0);
    /// let work = Power::work_from_time(power, 10.0);
    /// assert_eq!(work.value(), 100.0);
    /// ```
    pub fn work_from_time(
        power: Power,
        time: Time,
    ) -> Self {
        // W = P · t
        let w = power.0.value * time.value;
        Self::new(w)
    }

    /// Compute energy from power and time: E = P · t
    ///
    /// # Arguments
    /// * `power` - Power
    /// * `time` - Time
    ///
    /// # Returns
    /// Energy (Work - same dimension as energy)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Power, Work};
    /// use mathverse_units::si::Joule;
    ///
    /// let power = Power::new(10.0);
    /// let energy = Power::energy_from_time(power, 10.0);
    /// assert_eq!(energy.value(), 100.0);
    /// ```
    pub fn energy_from_time(
        power: Power,
        time: Time,
    ) -> Work {
        // E = P · t (energy has same dimension as work)
        let e = power.0.value * time.value;
        Work::new(e)
    }
}

impl Default for Work {
    fn default() -> Self {
        Self(Quantity::new(0.0))
    }
}

impl From<f64> for Work {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

// Arithmetic operations that preserve the Energy dimension

impl Add for Work {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Work::new(self.0.value + other.0.value)
    }
}

impl Add<f64> for Work {
    type Output = Self;
    fn add(self, scalar: f64) -> Self::Output {
        Work::new(self.0.value + scalar)
    }
}

impl Add<Quantity<EnergyDim, Joule>> for Work {
    type Output = Self;
    fn add(self, other: Quantity<EnergyDim, Joule>) -> Self::Output {
        Work::new(self.0.value + other.value)
    }
}

impl Sub for Work {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Work::new(self.0.value - other.0.value)
    }
}

impl Sub<f64> for Work {
    type Output = Self;
    fn sub(self, scalar: f64) -> Self::Output {
        Work::new(self.0.value - scalar)
    }
}

impl Sub<Quantity<EnergyDim, Joule>> for Work {
    type Output = Self;
    fn sub(self, other: Quantity<EnergyDim, Joule>) -> Self::Output {
        Work::new(self.0.value - other.value)
    }
}

impl Mul<f64> for Work {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self::Output {
        Work::new(self.0.value * scalar)
    }
}

impl Div<f64> for Work {
    type Output = Self;
    fn div(self, scalar: f64) -> Self::Output {
        Work::new(self.0.value / scalar)
    }
}

// Scale work by a dimensionless factor
impl AddAssign for Work {
    fn add_assign(&mut self, other: Self) {
        self.0 = Work::new(self.0.value + other.0.value).0;
    }
}

impl SubAssign for Work {
    fn sub_assign(&mut self, other: Self) {
        self.0 = Work::new(self.0.value - other.0.value).0;
    }
}

/// Power in watts.
///
/// Rate of doing work or transferring energy. The dimension is PowerDim
/// (M·L²/T³).
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::Power;
/// use mathverse_units::si::Watt;
///
/// let power: Power = 60.0.watts();
/// assert_eq!(power.value(), 60.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Power(pub Quantity<PowerDim, Watt>);

impl Power {
    /// Create a new power from a value in watts.
    pub fn new(value: f64) -> Self {
        Self(Quantity::new(value))
    }

    /// Get the raw value in watts.
    pub fn value(&self) -> f64 {
        self.0.value()
    }

    /// Compute power from work and time: P = W / t
    ///
    /// # Arguments
    /// * `work` - Work
    /// * `time` - Time
    ///
    /// # Returns
    /// Power (Power)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Work, Power};
    /// use mathverse_units::si::Watt;
    ///
    /// let work = Work::new(100.0);
    /// let time = 10.0.s();
    /// let power = Power::from_work_time(work, time);
    /// assert_eq!(power.value(), 10.0);
    /// ```
    pub fn from_work_time(
        work: Work,
        time: Time,
    ) -> Self {
        // P = W / t
        let p = work.0.value / time.value;
        Self::new(p)
    }

    /// Compute work from power and time: W = P · t
    ///
    /// # Arguments
    /// * `power` - Power
    /// * `time` - Time
    ///
    /// # Returns
    /// Work (Work)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Power, Work};
    /// use mathverse_units::si::Joule;
    ///
    /// let power = Power::new(10.0);
    /// let time = 10.0.s();
    /// let work = Power::work_from_time(power, time);
    /// assert_eq!(work.value(), 100.0);
    /// ```
    pub fn work_from_time(
        power: Power,
        time: Time,
    ) -> Self {
        // W = P · t
        let w = power.0.value * time.value;
        Self::new(w)
    }

    /// Compute energy from power and time: E = P · t
    ///
    /// # Arguments
    /// * `power` - Power
    /// * `time` - Time
    ///
    /// # Returns
    /// Energy (Work - same dimension as energy)
    ///
    /// # Example
    ///
    /// ```
    /// use mathverse_physics::dynamics::{Power, Work};
    /// use mathverse_units::si::Joule;
    ///
    /// let power = Power::new(10.0);
    /// let time = 10.0.s();
    /// let energy = Power::energy_from_time(power, time);
    /// assert_eq!(energy.value(), 100.0);
    /// ```
    pub fn energy_from_time(
        power: Power,
        time: Time,
    ) -> Work {
        // E = P · t (energy has same dimension as work)
        let e = power.0.value * time.value;
        Work::new(e)
    }
}

impl Default for Power {
    fn default() -> Self {
        Self(Quantity::new(0.0))
    }
}

impl From<f64> for Power {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

// Arithmetic operations that preserve the Power dimension

impl Add for Power {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Power::new(self.0.value + other.0.value)
    }
}

impl Add<f64> for Power {
    type Output = Self;
    fn add(self, scalar: f64) -> Self::Output {
        Power::new(self.0.value + scalar)
    }
}

impl Add<Quantity<PowerDim, Watt>> for Power {
    type Output = Self;
    fn add(self, other: Quantity<PowerDim, Watt>) -> Self::Output {
        Power::new(self.0.value + other.value)
    }
}

impl Sub for Power {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Power::new(self.0.value - other.0.value)
    }
}

impl Sub<f64> for Power {
    type Output = Self;
    fn sub(self, scalar: f64) -> Self::Output {
        Power::new(self.0.value - scalar)
    }
}

impl Sub<Quantity<PowerDim, Watt>> for Power {
    type Output = Self;
    fn sub(self, other: Quantity<PowerDim, Watt>) -> Self::Output {
        Power::new(self.0.value - other.value)
    }
}

impl Mul<f64> for Power {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self::Output {
        Power::new(self.0.value * scalar)
    }
}

impl Div<f64> for Power {
    type Output = Self;
    fn div(self, scalar: f64) -> Self::Output {
        Power::new(self.0.value / scalar)
    }
}

// Scale power by a dimensionless factor
impl AddAssign for Power {
    fn add_assign(&mut self, other: Self) {
        self.0 = Power::new(self.0.value + other.0.value).0;
    }
}

impl SubAssign for Power {
    fn sub_assign(&mut self, other: Self) {
        self.0 = Power::new(self.0.value - other.0.value).0;
    }
}

/// Alias for mass quantity with MassDim dimension.
pub type MassQuant = mathverse_units::quantity::Quantity<mathverse_units::dimensions::MassDim, mathverse_units::si::Kilogram>;

/// Alias for time quantity with TimeDim dimension.
pub type TimeQuant = mathverse_units::quantity::Quantity<mathverse_units::dimensions::TimeDim, mathverse_units::si::Second>;

/// Compute force from mass and acceleration: F = m · a
///
/// # Arguments
/// * `mass` - Mass (MassQuant)
/// * `acceleration` - Acceleration
///
/// # Returns
/// Force (Force)
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::{compute_force, MassQuant, Acceleration};
/// use mathverse_units::si::{Kilogram, mps2};
///
/// let force = compute_force(1.0.kg(), 2.0.mps2());
/// assert_eq!(force.value(), 2.0);
/// ```
pub fn compute_force(
    mass: mathverse_units::quantity::Quantity<MassDim, Kilogram>,
    acceleration: mathverse_units::quantity::Quantity<AccelerationDim, Meter>,
) -> Force {
    Force::from_mass_acceleration(mass, acceleration)
}

/// Compute work from force and displacement: W = F · d
///
/// # Arguments
/// * `force` - Force
/// * `displacement` - Displacement (Position)
/// * `theta` - Angle in radians between force and displacement (default: 0, parallel)
///
/// # Returns
/// Work (Work)
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::{compute_work, Force, Position};
/// use mathverse_units::si::{Newton, Meter};
///
/// let force = Force::new(10.0);
/// let displacement = Position::new(3.0);
/// let work = compute_work(force, displacement, 0.0);
/// assert_eq!(work.value(), 30.0);
/// ```
pub fn compute_work(
    force: Force,
    displacement: Position,
    theta: f64,
) -> Work {
    force.work_over(displacement, theta)
}

/// Compute power from work and time: P = W / t
///
/// # Arguments
/// * `work` - Work
/// * `time` - Time
///
/// # Returns
/// Power (Power)
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::{compute_power, Work};
/// use mathverse_units::si::Watt;
///
/// let work = Work::new(100.0);
/// let time = 10.0.s();
/// let power = compute_power(work, time);
/// assert_eq!(power.value(), 10.0);
/// ```
pub fn compute_power(
    work: Work,
    time: Time,
) -> Power {
    Power::from_work_time(work, time)
}

/// Compute work from power and time: W = P · t
///
/// # Arguments
/// * `power` - Power
/// * `time` - Time
///
/// # Returns
/// Work (Work)
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::{compute_work_from_power, Power};
/// use mathverse_units::si::Joule;
///
/// let power = Power::new(10.0);
/// let work = compute_work_from_power(power, 10.0);
/// assert_eq!(work.value(), 100.0);
/// ```
pub fn compute_work_from_power(
    power: Power,
    time: Time,
) -> Work {
    // W = P · t
    Work::new(power.0.value * time.value)
}

/// Compute kinetic energy: KE = ½ · m · v²
///
/// # Arguments
/// * `mass` - Mass (MassQuant)
/// * `velocity` - Velocity
///
/// # Returns
/// Kinetic energy (Work)
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::{compute_kinetic_energy};
/// use mathverse_units::si::{Kilogram, mps};
///
/// let ke = compute_kinetic_energy(1.0.kg(), 10.0.mps());
/// assert_eq!(ke.value(), 50.0);
/// ```
pub fn compute_kinetic_energy(
    mass: mathverse_units::quantity::Quantity<MassDim, Kilogram>,
    velocity: mathverse_units::quantity::Quantity<mathverse_units::dimensions::VelocityDim, Meter>,
) -> Work {
    // KE = ½ · m · v²
    let ke = 0.5 * mass.value * velocity.value.powi(2);
    Work::new(ke)
}

/// Compute potential energy (gravitational): PE = m · g · h
///
/// # Arguments
/// * `mass` - Mass (MassQuant)
/// * `g` - Gravitational acceleration (Acceleration)
/// * `height` - Height (Position)
///
/// # Returns
/// Potential energy (Work)
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::{compute_potential_energy_gravitational};
/// use mathverse_units::si::{Kilogram, mps2, Meter};
///
/// let pe = compute_potential_energy_gravitational(1.0.kg(), 9.81.mps2(), 5.0.m());
/// assert!((pe.value() - 49.05).abs() < 1e-6);
/// ```
pub fn compute_potential_energy_gravitational(
    mass: mathverse_units::quantity::Quantity<MassDim, Kilogram>,
    g: mathverse_units::quantity::Quantity<AccelerationDim, Meter>,
    height: Position,
) -> Work {
    // PE = m · g · h
    let pe = mass.value * g.value * height.0.value;
    Work::new(pe)
}

/// Compute momentum: p = m · v
///
/// # Arguments
/// * `mass` - Mass (MassQuant)
/// * `velocity` - Velocity
///
/// # Returns
/// Momentum value (f64, kg·m/s)
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::compute_momentum;
/// use mathverse_units::si::{Kilogram, mps};
///
/// let momentum = compute_momentum(1.0.kg(), 10.0.mps());
/// assert_eq!(momentum, 10.0);
/// ```
pub fn compute_momentum(
    mass: mathverse_units::quantity::Quantity<MassDim, Kilogram>,
    velocity: mathverse_units::quantity::Quantity<mathverse_units::dimensions::VelocityDim, Meter>,
) -> f64 {
    mass.value * velocity.value
}

/// Compute impulse: J = F · Δt (impulse-momentum theorem)
///
/// # Arguments
/// * `force` - Force
/// * `dt` - Time interval
///
/// # Returns
/// Impulse value (f64, kg·m/s)
///
/// # Example
///
/// ```
/// use mathverse_physics::dynamics::compute_impulse;
/// use mathverse_units::si::{Newton, s};
///
/// let impulse = compute_impulse(Force::new(10.0), 5.0.s());
/// assert_eq!(impulse, 50.0);
/// ```
pub fn compute_impulse(
    force: Force,
    dt: Time,
) -> f64 {
    force.0.value * dt.value
}

/// Alias for position quantity with LengthDim dimension.
pub type PositionQuant = mathverse_units::quantity::Quantity<mathverse_units::dimensions::LengthDim, mathverse_units::si::Meter>;

/// Alias for velocity quantity with VelocityDim dimension.
pub type VelocityQuant = mathverse_units::quantity::Quantity<mathverse_units::dimensions::VelocityDim, mathverse_units::si::Meter>;

/// Compute force from mass and acceleration: F = m · a
///
/// # Arguments
/// * `mass` - Mass
/// * `acceleration` - Acceleration
///
/// # Returns
/// Force (Force)
pub fn compute_force_generic(
    mass: mathverse_units::quantity::Quantity<MassDim, Kilogram>,
    acceleration: mathverse_units::quantity::Quantity<AccelerationDim, Meter>,
) -> Force {
    Force::from_mass_acceleration(mass, acceleration)
}

/// Compute work from force and displacement: W = F · d
///
/// # Arguments
/// * `force` - Force
/// * `displacement` - Displacement (Position)
/// * `theta` - Angle in radians between force and displacement (default: 0, parallel)
///
/// # Returns
/// Work (Work)
pub fn compute_work_generic(
    force: Force,
    displacement: Position,
    theta: f64,
) -> Work {
    force.work_over(displacement, theta)
}

/// Compute power from work and time: P = W / t
///
/// # Arguments
/// * `work` - Work
/// * `time` - Time
///
/// # Returns
/// Power (Power)
pub fn compute_power_generic(
    work: Work,
    time: Time,
) -> Power {
    Power::from_work_time(work, time)
}

/// Compute work from power and time: W = P · t
///
/// # Arguments
/// * `power` - Power
/// * `time` - Time
///
/// # Returns
/// Work (Work)
pub fn compute_work_from_power_generic(
    power: Power,
    time: Time,
) -> Work {
    // W = P · t
    Work::new(power.0.value * time.value)
}

/// Compute kinetic energy: KE = ½ · m · v²
///
/// # Arguments
/// * `mass` - Mass
/// * `velocity` - Velocity
///
/// # Returns
/// Kinetic energy (Work)
pub fn compute_kinetic_energy_generic(
    mass: mathverse_units::quantity::Quantity<MassDim, Kilogram>,
    velocity: mathverse_units::quantity::Quantity<mathverse_units::dimensions::VelocityDim, Meter>,
) -> Work {
    // KE = ½ · m · v²
    let ke = 0.5 * mass.value * velocity.value.powi(2);
    Work::new(ke)
}

/// Compute potential energy (gravitational): PE = m · g · h
///
/// # Arguments
/// * `mass` - Mass
/// * `g` - Gravitational acceleration
/// * `height` - Height
///
/// # Returns
/// Potential energy (Work)
pub fn compute_potential_energy_gravitational_generic(
    mass: mathverse_units::quantity::Quantity<MassDim, Kilogram>,
    g: mathverse_units::quantity::Quantity<AccelerationDim, Meter>,
    height: mathverse_units::quantity::Quantity<LengthDim, Meter>,
) -> Work {
    // PE = m · g · h
    let pe = mass.value * g.value * height.value;
    Work::new(pe)
}

/// Compute impulse: J = F · Δt (impulse-momentum theorem)
///
/// # Arguments
/// * `force` - Force
/// * `dt` - Time interval
///
/// # Returns
/// Impulse value (f64, kg·m/s)
pub fn compute_impulse_generic(
    force: Force,
    dt: Time,
) -> f64 {
    force.0.value * dt.value
}