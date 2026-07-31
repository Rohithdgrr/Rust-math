//! Quantity type with dimension and unit information.

use crate::dimensions::Dimension;
use crate::si::SiUnit;
use std::ops::{Add, Sub, Mul, Div};
use std::marker::PhantomData;

/// A quantity with a specific dimension and unit.
#[derive(Debug, Clone, Copy)]
pub struct Quantity<D: Dimension, U: SiUnit> {
    pub value: f64,
    _dim: PhantomData<D>,
    _unit: PhantomData<U>,
}

impl<D: Dimension, U: SiUnit> Quantity<D, U> {
    /// Create a new quantity with the given value.
    pub fn new(value: f64) -> Self {
        Quantity {
            value,
            _dim: PhantomData,
            _unit: PhantomData,
        }
    }

    /// Get the value of the quantity.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Convert to a different unit by applying a conversion factor.
    pub fn convert<U2: SiUnit>(self, factor: f64) -> Quantity<D, U2> {
        Quantity {
            value: self.value * factor,
            _dim: PhantomData,
            _unit: PhantomData,
        }
    }
}

impl<D: Dimension, U: SiUnit> Add for Quantity<D, U> {
    type Output = Quantity<D, U>;

    fn add(self, other: Self) -> Self::Output {
        Quantity {
            value: self.value + other.value,
            _dim: PhantomData,
            _unit: PhantomData,
        }
    }
}

impl<D: Dimension, U: SiUnit> Sub for Quantity<D, U> {
    type Output = Quantity<D, U>;

    fn sub(self, other: Self) -> Self::Output {
        Quantity {
            value: self.value - other.value,
            _dim: PhantomData,
            _unit: PhantomData,
        }
    }
}

impl<D: Dimension, U: SiUnit> Mul<f64> for Quantity<D, U> {
    type Output = Quantity<D, U>;

    fn mul(self, scalar: f64) -> Self::Output {
        Quantity {
            value: self.value * scalar,
            _dim: PhantomData,
            _unit: PhantomData,
        }
    }
}

impl<D: Dimension, U: SiUnit> Div<f64> for Quantity<D, U> {
    type Output = Quantity<D, U>;

    fn div(self, scalar: f64) -> Self::Output {
        Quantity {
            value: self.value / scalar,
            _dim: PhantomData,
            _unit: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimensions::LengthDim;
    use crate::si::Meter;

    #[test]
    fn test_quantity_creation() {
        let q: Quantity<LengthDim, Meter> = Quantity::new(5.0);
        assert_eq!(q.value(), 5.0);
    }

    #[test]
    fn test_quantity_arithmetic() {
        let q1: Quantity<LengthDim, Meter> = Quantity::new(5.0);
        let q2: Quantity<LengthDim, Meter> = Quantity::new(3.0);
        
        let sum = q1 + q2;
        assert_eq!(sum.value(), 8.0);
        
        let diff = q1 - q2;
        assert_eq!(diff.value(), 2.0);
        
        let scaled = q1 * 2.0;
        assert_eq!(scaled.value(), 10.0);
        
        let divided = q1 / 2.0;
        assert_eq!(divided.value(), 2.5);
    }
}
