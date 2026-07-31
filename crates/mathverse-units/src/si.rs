//! SI unit definitions and base units.

use crate::dimensions::*;
use std::marker::PhantomData;

/// SI base unit marker trait
pub trait SiUnit: Copy + Default + std::fmt::Debug {
    const NAME: &'static str;
    const SYMBOL: &'static str;
}

/// Meter - base unit of length
#[derive(Debug, Copy, Clone, Default)]
pub struct Meter;

impl SiUnit for Meter {
    const NAME: &'static str = "meter";
    const SYMBOL: &'static str = "m";
}

/// Kilogram - base unit of mass
#[derive(Debug, Copy, Clone, Default)]
pub struct Kilogram;

impl SiUnit for Kilogram {
    const NAME: &'static str = "kilogram";
    const SYMBOL: &'static str = "kg";
}

/// Second - base unit of time
#[derive(Debug, Copy, Clone, Default)]
pub struct Second;

impl SiUnit for Second {
    const NAME: &'static str = "second";
    const SYMBOL: &'static str = "s";
}

/// Ampere - base unit of electric current
#[derive(Debug, Copy, Clone, Default)]
pub struct Ampere;

impl SiUnit for Ampere {
    const NAME: &'static str = "ampere";
    const SYMBOL: &'static str = "A";
}

/// Kelvin - base unit of temperature
#[derive(Debug, Copy, Clone, Default)]
pub struct Kelvin;

impl SiUnit for Kelvin {
    const NAME: &'static str = "kelvin";
    const SYMBOL: &'static str = "K";
}

/// Mole - base unit of amount of substance
#[derive(Debug, Copy, Clone, Default)]
pub struct Mole;

impl SiUnit for Mole {
    const NAME: &'static str = "mole";
    const SYMBOL: &'static str = "mol";
}

/// Candela - base unit of luminous intensity
#[derive(Debug, Copy, Clone, Default)]
pub struct Candela;

impl SiUnit for Candela {
    const NAME: &'static str = "candela";
    const SYMBOL: &'static str = "cd";
}

/// Derived SI units
#[derive(Debug, Copy, Clone, Default)]
pub struct Newton; // Force: kg·m/s²

impl SiUnit for Newton {
    const NAME: &'static str = "newton";
    const SYMBOL: &'static str = "N";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Joule; // Energy: kg·m²/s²

impl SiUnit for Joule {
    const NAME: &'static str = "joule";
    const SYMBOL: &'static str = "J";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Watt; // Power: kg·m²/s³

impl SiUnit for Watt {
    const NAME: &'static str = "watt";
    const SYMBOL: &'static str = "W";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Pascal; // Pressure: kg/(m·s²)

impl SiUnit for Pascal {
    const NAME: &'static str = "pascal";
    const SYMBOL: &'static str = "Pa";
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Hertz; // Frequency: 1/s

impl SiUnit for Hertz {
    const NAME: &'static str = "hertz";
    const SYMBOL: &'static str = "Hz";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_si_units() {
        let m = Meter;
        assert_eq!(Meter::NAME, "meter");
        assert_eq!(Meter::SYMBOL, "m");
    }
}
