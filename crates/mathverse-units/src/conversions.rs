//! Unit conversion factors and utilities.

use crate::si::*;

/// Conversion factors between common units.
pub struct ConversionFactors;

impl ConversionFactors {
    /// Length conversions (from meters)
    pub const METER_TO_CENTIMETER: f64 = 100.0;
    pub const METER_TO_MILLIMETER: f64 = 1000.0;
    pub const METER_TO_KILOMETER: f64 = 1e-3;
    pub const METER_TO_INCH: f64 = 39.3701;
    pub const METER_TO_FOOT: f64 = 3.28084;
    pub const METER_TO_YARD: f64 = 1.09361;
    pub const METER_TO_MILE: f64 = 6.21371e-4;

    /// Mass conversions (from kilograms)
    pub const KILOGRAM_TO_GRAM: f64 = 1000.0;
    pub const KILOGRAM_TO_MILLIGRAM: f64 = 1e6;
    pub const KILOGRAM_TO_POUND: f64 = 2.20462;
    pub const KILOGRAM_TO_OUNCE: f64 = 35.274;

    /// Time conversions (from seconds)
    pub const SECOND_TO_MILLISECOND: f64 = 1000.0;
    pub const SECOND_TO_MICROSECOND: f64 = 1e6;
    pub const SECOND_TO_NANOSECOND: f64 = 1e9;
    pub const SECOND_TO_MINUTE: f64 = 1.0 / 60.0;
    pub const SECOND_TO_HOUR: f64 = 1.0 / 3600.0;
    pub const SECOND_TO_DAY: f64 = 1.0 / 86400.0;

    /// Temperature conversions
    pub const KELVIN_TO_CELSIUS_OFFSET: f64 = -273.15;
    pub const KELVIN_TO_FAHRENHEIT_OFFSET: f64 = -459.67;
    pub const KELVIN_TO_FAHRENHEIT_SCALE: f64 = 1.8;

    /// Energy conversions (from joules)
    pub const JOULE_TO_CALORIE: f64 = 0.239006;
    pub const JOULE_TO_KILOWATT_HOUR: f64 = 2.77778e-7;
    pub const JOULE_TO_ELECTRONVOLT: f64 = 6.242e18;

    /// Pressure conversions (from pascals)
    pub const PASCAL_TO_BAR: f64 = 1e-5;
    pub const PASCAL_TO_ATMOSPHERE: f64 = 9.86923e-6;
    pub const PASCAL_TO_MMHG: f64 = 0.00750062;
}

/// Convert Celsius to Kelvin
pub fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius + 273.15
}

/// Convert Kelvin to Celsius
pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin - 273.15
}

/// Convert Fahrenheit to Celsius
pub fn fahrenheit_to_celsius(fahrenheit: f64) -> f64 {
    (fahrenheit - 32.0) * 5.0 / 9.0
}

/// Convert Celsius to Fahrenheit
pub fn celsius_to_fahrenheit(celsius: f64) -> f64 {
    celsius * 9.0 / 5.0 + 32.0
}

/// Convert Fahrenheit to Kelvin
pub fn fahrenheit_to_kelvin(fahrenheit: f64) -> f64 {
    celsius_to_kelvin(fahrenheit_to_celsius(fahrenheit))
}

/// Convert Kelvin to Fahrenheit
pub fn kelvin_to_fahrenheit(kelvin: f64) -> f64 {
    celsius_to_fahrenheit(kelvin_to_celsius(kelvin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_conversions() {
        assert!((celsius_to_kelvin(0.0) - 273.15).abs() < 1e-10);
        assert!((kelvin_to_celsius(273.15) - 0.0).abs() < 1e-10);
        assert!((fahrenheit_to_celsius(32.0) - 0.0).abs() < 1e-10);
        assert!((celsius_to_fahrenheit(0.0) - 32.0).abs() < 1e-10);
        assert!((fahrenheit_to_kelvin(32.0) - 273.15).abs() < 1e-10);
        assert!((kelvin_to_fahrenheit(273.15) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_conversion_factors() {
        assert_eq!(ConversionFactors::METER_TO_CENTIMETER, 100.0);
        assert_eq!(ConversionFactors::KILOGRAM_TO_GRAM, 1000.0);
        assert_eq!(ConversionFactors::SECOND_TO_MINUTE, 1.0 / 60.0);
    }
}
