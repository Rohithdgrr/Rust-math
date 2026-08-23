//! Fluent SI unit constructors.
//!
//! Enables physics-style literals such as `1.0.kg()`, `9.81.mps2()`,
//! `100.0.joules()` anywhere the trait is in scope. Every constructor returns
//! a [`Quantity`] whose dimension parameter is fixed at compile time, so
//! dimensional mistakes fail to build rather than silently producing wrong
//! numbers.

use mathverse_units::dimensions::{
    AccelerationDim, AngleDim, EnergyDim, ForceDim, LengthDim, MassDim, PowerDim, TimeDim,
    VelocityDim,
};
use mathverse_units::quantity::Quantity;
use mathverse_units::si::{Joule, Kilogram, Meter, Newton, Radian, Second, Watt};

/// Fluent constructors for common SI quantities on `f64` literals.
pub trait FluentPhysicsUnits {
    /// Length in meters.
    #[must_use]
    fn m(self) -> Quantity<LengthDim, Meter>;
    /// Mass in kilograms.
    #[must_use]
    fn kg(self) -> Quantity<MassDim, Kilogram>;
    /// Time in seconds.
    #[must_use]
    fn s(self) -> Quantity<TimeDim, Second>;
    /// Speed in meters per second.
    #[must_use]
    fn mps(self) -> Quantity<VelocityDim, Meter>;
    /// Acceleration in meters per second squared.
    #[must_use]
    fn mps2(self) -> Quantity<AccelerationDim, Meter>;
    /// Plane angle in radians.
    #[must_use]
    fn radians(self) -> Quantity<AngleDim, Radian>;
    /// Force in newtons.
    #[must_use]
    fn newtons(self) -> Quantity<ForceDim, Newton>;
    /// Energy/work in joules.
    #[must_use]
    fn joules(self) -> Quantity<EnergyDim, Joule>;
    /// Power in watts.
    #[must_use]
    fn watts(self) -> Quantity<PowerDim, Watt>;
}

impl FluentPhysicsUnits for f64 {
    #[inline]
    fn m(self) -> Quantity<LengthDim, Meter> {
        Quantity::new(self)
    }
    #[inline]
    fn kg(self) -> Quantity<MassDim, Kilogram> {
        Quantity::new(self)
    }
    #[inline]
    fn s(self) -> Quantity<TimeDim, Second> {
        Quantity::new(self)
    }
    #[inline]
    fn mps(self) -> Quantity<VelocityDim, Meter> {
        Quantity::new(self)
    }
    #[inline]
    fn mps2(self) -> Quantity<AccelerationDim, Meter> {
        Quantity::new(self)
    }
    #[inline]
    fn radians(self) -> Quantity<AngleDim, Radian> {
        Quantity::new(self)
    }
    #[inline]
    fn newtons(self) -> Quantity<ForceDim, Newton> {
        Quantity::new(self)
    }
    #[inline]
    fn joules(self) -> Quantity<EnergyDim, Joule> {
        Quantity::new(self)
    }
    #[inline]
    fn watts(self) -> Quantity<PowerDim, Watt> {
        Quantity::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fluent_constructors_carry_values() {
        assert_eq!(3.5.m().value(), 3.5);
        assert_eq!(2.0.kg().value(), 2.0);
        assert_eq!(10.0.s().value(), 10.0);
        assert_eq!(42.0.joules().value(), 42.0);
    }

    #[test]
    fn dimensions_are_distinct_types() {
        // Type-checking helpers; distinct phantom parameters must never unify.
        fn takes_length(_q: Quantity<LengthDim, Meter>) {}
        fn takes_time(_q: Quantity<TimeDim, Second>) {}

        let length = 1.0.m();
        let time = 1.0.s();
        takes_length(length);
        takes_time(time);
    }

    #[test]
    fn force_from_fluent_parts() {
        let f = crate::dynamics::Force::from_mass_acceleration(1.0.kg(), 2.0.mps2());
        assert_eq!(f.value(), 2.0);
    }
}
