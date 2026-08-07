//! Quantity type with dimension and unit information.

use crate::dimensions::Dimension;
use crate::si::SiUnit;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

/// A quantity with compile-time dimension and unit checking.
///
/// `D` is the dimension type (e.g. `LengthDim`, `MassDim`), `U` is the unit
/// type (e.g. `Meter`, `Kilogram`). Operations between quantities of different
/// dimensions are prevented at compile time: `Quantity<LengthDim, Meter> +
/// Quantity<MassDim, Kilogram>` does not compile.
///
/// # Type Safety
///
/// ```compile_fail
/// use mathverse_units::{Quantity, LengthDim, Meter, MassDim, Kilogram};
/// let length: Quantity<LengthDim, Meter> = Quantity::new(5.0);
/// let mass: Quantity<MassDim, Kilogram> = Quantity::new(10.0);
/// let _ = length + mass; // ERROR: cannot add Length to Mass
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Quantity<D: Dimension, U: SiUnit> {
    pub value: f64,
    _dim: PhantomData<D>,
    _unit: PhantomData<U>,
}

impl<D: Dimension, U: SiUnit> Quantity<D, U> {
    /// Create a new quantity with the given value.
    pub fn new(value: f64) -> Self {
        Self {
            value,
            _dim: PhantomData,
            _unit: PhantomData,
        }
    }

    /// Get the raw numeric value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Convert to a different unit of the same dimension by applying a factor.
    ///
    /// For example: `q.convert::<Centimeter>(100.0)` converts meters to
    /// centimeters.
    pub fn convert<V: SiUnit>(self, factor: f64) -> Quantity<D, V> {
        Quantity::new(self.value * factor)
    }

    /// Return the quantity's absolute value.
    pub fn abs(self) -> Self {
        Quantity::new(self.value.abs())
    }
}

impl<D: Dimension, U: SiUnit> Add for Quantity<D, U> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Quantity::new(self.value + other.value)
    }
}

impl<D: Dimension, U: SiUnit> Sub for Quantity<D, U> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Quantity::new(self.value - other.value)
    }
}

// Scalar multiplication — preserves dimension
impl<D: Dimension, U: SiUnit> Mul<f64> for Quantity<D, U> {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self::Output {
        Quantity::new(self.value * scalar)
    }
}

// Scalar division — preserves dimension
impl<D: Dimension, U: SiUnit> Div<f64> for Quantity<D, U> {
    type Output = Self;
    fn div(self, scalar: f64) -> Self::Output {
        Quantity::new(self.value / scalar)
    }
}

impl<D: Dimension, U: SiUnit> fmt::Display for Quantity<D, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.value, U::SYMBOL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimensions::{LengthDim, MassDim};
    use crate::si::{Kilogram, Meter};

    #[test]
    fn phantoms_are_distinct() {
        // Key test: PhantomData<D> and PhantomData<U> carry real type info.
        // Constructing Quantity<LengthDim, Meter> and Quantity<MassDim, Kilogram>
        // produces different types; they cannot be added together.
        let _length: Quantity<LengthDim, Meter> = Quantity::new(5.0);
        let _mass: Quantity<MassDim, Kilogram> = Quantity::new(10.0);

        // Uncommenting the line below should fail to compile:
        // let _ = _length + _mass;
    }

    #[test]
    fn test_quantity_creation() {
        let q: Quantity<LengthDim, Meter> = Quantity::new(5.0);
        assert_eq!(q.value(), 5.0);
    }

    #[test]
    fn test_quantity_arithmetic() {
        let q1: Quantity<LengthDim, Meter> = Quantity::new(5.0);
        let q2: Quantity<LengthDim, Meter> = Quantity::new(3.0);
        assert_eq!((q1 + q2).value(), 8.0);
        assert_eq!((q1 - q2).value(), 2.0);
        assert_eq!((q1 * 2.0).value(), 10.0);
        assert_eq!((q1 / 2.0).value(), 2.5);
    }

    #[test]
    fn test_display() {
        let q: Quantity<LengthDim, Meter> = Quantity::new(5.0);
        assert_eq!(format!("{}", q), "5.0m");
    }

    #[test]
    fn test_abs() {
        let q: Quantity<LengthDim, Meter> = Quantity::new(-3.0);
        assert_eq!(q.abs().value(), 3.0);
    }
}
