//! Rounding and precision operations: various rounding methods, precision control.

use mathverse_core::error::{MathError, MathResult};

/// Rounding methods.
pub enum RoundingMode {
    RoundHalfUp,
    RoundHalfDown,
    RoundHalfEven,
    RoundHalfAwayFromZero,
    RoundHalfTowardZero,
    RoundUp,
    RoundDown,
    RoundTowardZero,
    RoundAwayFromZero,
}

/// Advanced rounding operations.
pub struct Rounding;

impl Rounding {
    /// Round to specified number of decimal places.
    pub fn to_decimal_places(x: f64, places: u32) -> f64 {
        let factor = 10.0_f64.powi(places as i32);
        (x * factor).round() / factor
    }

    /// Round with specified rounding mode.
    pub fn with_mode(x: f64, places: u32, mode: RoundingMode) -> f64 {
        let factor = 10.0_f64.powi(places as i32);
        let scaled = x * factor;
        let fractional = scaled.fract();
        let integer = scaled.trunc();
        
        let result = match mode {
            RoundingMode::RoundHalfUp => {
                if fractional.abs() >= 0.5 {
                    integer + fractional.signum()
                } else {
                    integer
                }
            }
            RoundingMode::RoundHalfDown => {
                if fractional.abs() > 0.5 {
                    integer + fractional.signum()
                } else {
                    integer
                }
            }
            RoundingMode::RoundHalfEven => {
                if fractional.abs() > 0.5 {
                    integer + fractional.signum()
                } else if fractional.abs() < 0.5 {
                    integer
                } else {
                    // Round to nearest even integer
                    let rounded = integer.round();
                    if rounded % 2.0 == 0.0 {
                        rounded
                    } else {
                        integer
                    }
                }
            }
            RoundingMode::RoundHalfAwayFromZero => {
                if fractional.abs() >= 0.5 {
                    integer + fractional.signum()
                } else {
                    integer
                }
            }
            RoundingMode::RoundHalfTowardZero => {
                if fractional.abs() > 0.5 {
                    integer + fractional.signum()
                } else {
                    integer
                }
            }
            RoundingMode::RoundUp => {
                if x >= 0.0 {
                    scaled.ceil()
                } else {
                    scaled.floor()
                }
            }
            RoundingMode::RoundDown => {
                if x >= 0.0 {
                    scaled.floor()
                } else {
                    scaled.ceil()
                }
            }
            RoundingMode::RoundTowardZero => {
                scaled.trunc()
            }
            RoundingMode::RoundAwayFromZero => {
                if x >= 0.0 {
                    scaled.ceil()
                } else {
                    scaled.floor()
                }
            }
        };
        
        result / factor
    }

    /// Round to nearest integer.
    pub fn to_integer(x: f64) -> i64 {
        x.round() as i64
    }

    /// Round up (ceiling).
    pub fn ceil(x: f64) -> f64 {
        x.ceil()
    }

    /// Round down (floor).
    pub fn floor(x: f64) -> f64 {
        x.floor()
    }

    /// Round toward zero.
    pub fn trunc(x: f64) -> f64 {
        x.trunc()
    }

    /// Round away from zero.
    pub fn away_from_zero(x: f64) -> f64 {
        if x >= 0.0 {
            x.ceil()
        } else {
            x.floor()
        }
    }

    /// Round to significant figures.
    pub fn to_significant_figures(x: f64, sig_figs: u32) -> f64 {
        if x == 0.0 {
            return 0.0;
        }
        
        let abs_x = x.abs();
        let exponent = abs_x.log10().floor() as i32;
        let factor = 10.0_f64.powi(sig_figs as i32 - 1 - exponent);
        
        let rounded = (abs_x * factor).round() / factor;
        
        if x >= 0.0 {
            rounded
        } else {
            -rounded
        }
    }

    /// Round to multiple of a value.
    pub fn to_multiple(x: f64, multiple: f64) -> f64 {
        if multiple == 0.0 {
            return x;
        }
        (x / multiple).round() * multiple
    }

    /// Banker's rounding (round half to even).
    pub fn bankers_round(x: f64) -> f64 {
        Self::with_mode(x, 0, RoundingMode::RoundHalfEven)
    }

    /// Round to specified precision (relative to magnitude).
    pub fn to_precision(x: f64, precision: f64) -> f64 {
        if x == 0.0 {
            return 0.0;
        }
        
        let magnitude = x.abs().log10().floor();
        let factor = 10.0_f64.powi((magnitude - precision.log10().floor()) as i32);
        
        (x * factor).round() / factor
    }
}

/// Precision control.
pub struct Precision;

impl Precision {
    /// Check if two numbers are almost equal within absolute tolerance.
    pub fn almost_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    /// Check if two numbers are almost equal within relative tolerance.
    pub fn almost_equal_relative(a: f64, b: f64, relative_tolerance: f64) -> bool {
        if a == b {
            return true;
        }
        
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        
        diff / max_abs < relative_tolerance
    }

    /// Check if two numbers are almost equal using both absolute and relative tolerance.
    pub fn almost_equal_mixed(
        a: f64,
        b: f64,
        absolute_tolerance: f64,
        relative_tolerance: f64,
    ) -> bool {
        let diff = (a - b).abs();
        
        if diff < absolute_tolerance {
            return true;
        }
        
        let max_abs = a.abs().max(b.abs());
        diff / max_abs < relative_tolerance
    }

    /// Count decimal places in a number.
    pub fn count_decimal_places(x: f64) -> usize {
        if x == 0.0 {
            return 0;
        }
        
        let s = format!("{:.15}", x.abs());
        
        if let Some(pos) = s.find('.') {
            let decimal_part = &s[pos + 1..];
            let trimmed = decimal_part.trim_end_matches('0');
            trimmed.len()
        } else {
            0
        }
    }

    /// Truncate to specified decimal places.
    pub fn truncate(x: f64, places: u32) -> f64 {
        let factor = 10.0_f64.powi(places as i32);
        (x * factor).trunc() / factor
    }

    /// Get machine epsilon for f64.
    pub fn machine_epsilon() -> f64 {
        f64::EPSILON
    }

    /// Get next representable float after x.
    pub fn next_after(x: f64) -> f64 {
        if x == 0.0 {
            return f64::MIN_POSITIVE;
        }
        
        let bits = x.to_bits();
        let next_bits = if x > 0.0 {
            bits + 1
        } else {
            bits - 1
        };
        
        f64::from_bits(next_bits)
    }

    /// Get previous representable float before x.
    pub fn previous_before(x: f64) -> f64 {
        if x == 0.0 {
            return -f64::MIN_POSITIVE;
        }
        
        let bits = x.to_bits();
        let prev_bits = if x > 0.0 {
            bits - 1
        } else {
            bits + 1
        };
        
        f64::from_bits(prev_bits)
    }

    /// Check if a number is within floating-point precision of an integer.
    pub fn is_effectively_integer(x: f64, tolerance: f64) -> bool {
        (x - x.round()).abs() < tolerance
    }

    /// Clamp value to range with precision consideration.
    pub fn clamp_with_precision(x: f64, min: f64, max: f64) -> f64 {
        if x < min {
            min
        } else if x > max {
            max
        } else {
            x
        }
    }

    /// Safe division that handles near-zero denominators.
    pub fn safe_divide(numerator: f64, denominator: f64, tolerance: f64) -> MathResult<f64> {
        if denominator.abs() < tolerance {
            return Err(MathError::InvalidArgument("denominator too close to zero"));
        }
        Ok(numerator / denominator)
    }
}

/// Fixed-point arithmetic.
pub struct FixedPoint;

impl FixedPoint {
    /// Convert f64 to fixed-point representation.
    pub fn to_fixed(x: f64, scale: u32) -> i64 {
        let factor = 10.0_f64.powi(scale as i32);
        (x * factor).round() as i64
    }

    /// Convert fixed-point back to f64.
    pub fn from_fixed(value: i64, scale: u32) -> f64 {
        let factor = 10.0_f64.powi(scale as i32);
        value as f64 / factor
    }

    /// Add two fixed-point numbers.
    pub fn add_fixed(a: i64, b: i64) -> i64 {
        a + b
    }

    /// Subtract two fixed-point numbers.
    pub fn sub_fixed(a: i64, b: i64) -> i64 {
        a - b
    }

    /// Multiply two fixed-point numbers (adjusting for scale).
    pub fn mul_fixed(a: i64, b: i64, scale: u32) -> i64 {
        let scale_factor = 10_i64.pow(scale);
        (a * b) / scale_factor
    }

    /// Divide two fixed-point numbers (adjusting for scale).
    pub fn div_fixed(a: i64, b: i64, scale: u32) -> MathResult<i64> {
        if b == 0 {
            return Err(MathError::DivisionByZero);
        }
        let scale_factor = 10_i64.pow(scale);
        Ok((a * scale_factor) / b)
    }

    /// Round fixed-point to specified precision.
    pub fn round_fixed(value: i64, from_scale: u32, to_scale: u32) -> i64 {
        if to_scale >= from_scale {
            let factor = 10_i64.pow(to_scale - from_scale);
            value * factor
        } else {
            let factor = 10_i64.pow(from_scale - to_scale);
            let rounded = (value as f64 / factor as f64).round() as i64;
            rounded
        }
    }
}

/// Decimal string formatting.
pub struct DecimalFormat;

impl DecimalFormat {
    /// Format number with specified decimal places.
    pub fn format(x: f64, places: u32) -> String {
        format!("{:.1$}", x, places)
    }

    /// Format number with thousands separator.
    pub fn format_with_separator(x: f64, places: u32, separator: char) -> String {
        let formatted = Self::format(x, places);
        
        if let Some(dot_pos) = formatted.find('.') {
            let integer_part = &formatted[..dot_pos];
            let decimal_part = &formatted[dot_pos..];
            
            let mut result = String::new();
            let mut count = 0;
            
            for c in integer_part.chars().rev() {
                if count > 0 && count % 3 == 0 {
                    result.push(separator);
                }
                result.push(c);
                count += 1;
            }
            
            result = result.chars().rev().collect();
            result.push_str(decimal_part);
            
            result
        } else {
            let mut result = String::new();
            let mut count = 0;
            
            for c in formatted.chars().rev() {
                if count > 0 && count % 3 == 0 {
                    result.push(separator);
                }
                result.push(c);
                count += 1;
            }
            
            result.chars().rev().collect()
        }
    }

    /// Format as currency.
    pub fn format_currency(x: f64, symbol: &str, places: u32) -> String {
        let formatted = Self::format_with_separator(x, places, ',');
        format!("{}{}", symbol, formatted)
    }

    /// Format as percentage.
    pub fn format_percentage(x: f64, places: u32) -> String {
        let percentage = x * 100.0;
        format!("{}%", Self::format(percentage, places))
    }

    /// Parse decimal string.
    pub fn parse(s: &str) -> MathResult<f64> {
        let s = s.replace(',', "").replace('_', "");
        s.parse().map_err(|_| MathError::InvalidArgument("invalid decimal string"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounding() {
        assert_eq!(Rounding::to_decimal_places(3.14159, 2), 3.14);
        assert_eq!(Rounding::to_decimal_places(3.15, 1), 3.2);
        assert_eq!(Rounding::to_integer(3.7), 4);
        assert_eq!(Rounding::ceil(3.2), 4.0);
        assert_eq!(Rounding::floor(3.9), 3.0);
    }

    #[test]
    fn test_rounding_modes() {
        assert_eq!(Rounding::with_mode(2.5, 0, RoundingMode::RoundHalfUp), 3.0);
        assert_eq!(Rounding::with_mode(2.5, 0, RoundingMode::RoundHalfDown), 2.0);
        assert_eq!(Rounding::with_mode(2.5, 0, RoundingMode::RoundHalfEven), 2.0);
        assert_eq!(Rounding::with_mode(3.5, 0, RoundingMode::RoundHalfEven), 4.0);
    }

    #[test]
    fn test_significant_figures() {
        assert_eq!(Rounding::to_significant_figures(1234.5, 3), 1230.0);
        assert_eq!(Rounding::to_significant_figures(0.0012345, 3), 0.00123);
    }

    #[test]
    fn test_precision() {
        assert!(Precision::almost_equal(1.0, 1.000001, 0.00001));
        assert!(Precision::almost_equal_relative(1000.0, 1001.0, 0.001));
        assert_eq!(Precision::count_decimal_places(3.14159), 5);
    }

    #[test]
    fn test_fixed_point() {
        let fixed = FixedPoint::to_fixed(3.14159, 2);
        assert_eq!(fixed, 314);
        
        let back = FixedPoint::from_fixed(314, 2);
        assert!((back - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_decimal_format() {
        assert_eq!(DecimalFormat::format(3.14159, 2), "3.14");
        assert_eq!(DecimalFormat::format_with_separator(1234.56, 2, ','), "1,234.56");
        assert_eq!(DecimalFormat::format_currency(1234.56, "$", 2), "$1,234.56");
        assert_eq!(DecimalFormat::format_percentage(0.1234, 2), "12.34%");
    }

    #[test]
    fn test_parse_decimal() {
        assert!((DecimalFormat::parse("1,234.56").unwrap() - 1234.56).abs() < 1e-10);
    }
}
