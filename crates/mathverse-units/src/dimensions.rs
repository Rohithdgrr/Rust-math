//! Dimension types for compile-time dimensional analysis

use typenum::{Integer, Z0, P1, N1, P2, N2, P3, N3};

/// Base dimension markers
pub type Length = P1;
pub type Mass = P1;
pub type Time = P1;
pub type ElectricCurrent = P1;
pub type Temperature = P1;
pub type Amount = P1;
pub type LuminousIntensity = P1;

/// Negative dimension markers
pub type LengthInv = N1;
pub type MassInv = N1;
pub type TimeInv = N1;

/// Dimension trait
pub trait Dimension {
    type L: Integer;
    type M: Integer;
    type T: Integer;
    type I: Integer;
    type Th: Integer;
    type N: Integer;
    type J: Integer;
}

/// Dimensionless quantity
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dimensionless;

impl Dimension for Dimensionless {
    type L = Z0;
    type M = Z0;
    type T = Z0;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Length dimension
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LengthDim;

impl Dimension for LengthDim {
    type L = P1;
    type M = Z0;
    type T = Z0;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Mass dimension
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MassDim;

impl Dimension for MassDim {
    type L = Z0;
    type M = P1;
    type T = Z0;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Time dimension
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeDim;

impl Dimension for TimeDim {
    type L = Z0;
    type M = Z0;
    type T = P1;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Velocity dimension (L/T)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityDim;

impl Dimension for VelocityDim {
    type L = P1;
    type M = Z0;
    type T = N1;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Acceleration dimension (L/T²)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccelerationDim;

impl Dimension for AccelerationDim {
    type L = P1;
    type M = Z0;
    type T = N2;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Force dimension (M·L/T²)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForceDim;

impl Dimension for ForceDim {
    type L = P1;
    type M = P1;
    type T = N2;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Energy dimension (M·L²/T²)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyDim;

impl Dimension for EnergyDim {
    type L = P2;
    type M = P1;
    type T = N2;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Power dimension (M·L²/T³)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerDim;

impl Dimension for PowerDim {
    type L = P2;
    type M = P1;
    type T = N3;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Frequency dimension (1/T)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyDim;

impl Dimension for FrequencyDim {
    type L = Z0;
    type M = Z0;
    type T = N1;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Angle dimension (radians)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngleDim;

impl Dimension for AngleDim {
    type L = Z0;
    type M = Z0;
    type T = Z0;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Area dimension (L²)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaDim;

impl Dimension for AreaDim {
    type L = P2;
    type M = Z0;
    type T = Z0;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Volume dimension (L³)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeDim;

impl Dimension for VolumeDim {
    type L = P3;
    type M = Z0;
    type T = Z0;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Pressure dimension (M/(L·T²))
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureDim;

impl Dimension for PressureDim {
    type L = N1;
    type M = P1;
    type T = N2;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

/// Density dimension (M/L³)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensityDim;

impl Dimension for DensityDim {
    type L = N3;
    type M = P1;
    type T = Z0;
    type I = Z0;
    type Th = Z0;
    type N = Z0;
    type J = Z0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_traits() {
        // Test that dimensions compile correctly
        let _: LengthDim = LengthDim;
        let _: MassDim = MassDim;
        let _: TimeDim = TimeDim;
    }
}
