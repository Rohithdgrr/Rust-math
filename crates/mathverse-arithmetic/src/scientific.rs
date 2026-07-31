//! Scientific notation: conversion, formatting, arithmetic with scientific notation.

use mathverse_core::error::{MathError, MathResult};

/// Scientific notation representation.
#[derive(Debug, Clone, PartialEq)]
pub struct ScientificNotation {
    pub mantissa: f64,
    pub exponent: i32,
}

impl ScientificNotation {
    /// Create scientific notation from a number.
    pub fn from_f64(x: f64) -> Self {
        if x == 0.0 {
            return ScientificNotation {
                mantissa: 0.0,
                exponent: 0,
            };
        }
        
        let abs_x = x.abs();
        let exponent = (abs_x.log10().floor()) as i32;
        let mantissa = x / 10.0_f64.powi(exponent);
        
        // Normalize mantissa to [1, 10)
        let normalized_mantissa = if mantissa.abs() >= 10.0 {
            let new_mantissa = mantissa / 10.0;
            ScientificNotation {
                mantissa: new_mantissa,
                exponent: exponent + 1,
            }
        } else if mantissa.abs() < 1.0 && mantissa.abs() > 0.0 {
            let new_mantissa = mantissa * 10.0;
            ScientificNotation {
                mantissa: new_mantissa,
                exponent: exponent - 1,
            }
        } else {
            ScientificNotation {
                mantissa,
                exponent,
            }
        };
        
        normalized_mantissa
    }

    /// Convert back to f64.
    pub fn to_f64(&self) -> f64 {
        self.mantissa * 10.0_f64.powi(self.exponent)
    }

    /// Create from mantissa and exponent.
    pub fn new(mantissa: f64, exponent: i32) -> Self {
        let mut result = Self::from_f64(mantissa * 10.0_f64.powi(exponent));
        result.exponent += exponent;
        result
    }

    /// Format as string: "m × 10^e".
    pub fn format(&self) -> String {
        format!("{} × 10^{}", self.mantissa, self.exponent)
    }

    /// Format with E notation: "mEe".
    pub fn format_e(&self) -> String {
        format!("{}E{}", self.mantissa, self.exponent)
    }

    /// Format with specified significant figures.
    pub fn format_with_sig_figs(&self, sig_figs: usize) -> String {
        let rounded = self.round_mantissa(sig_figs);
        rounded.format_e()
    }

    /// Round mantissa to specified significant figures.
    pub fn round_mantissa(&self, sig_figs: usize) -> Self {
        let factor = 10.0_f64.powi((sig_figs as i32) - 1);
        let rounded = (self.mantissa / factor).round() * factor;
        
        ScientificNotation {
            mantissa: rounded,
            exponent: self.exponent,
        }
    }

    /// Add two scientific notations.
    pub fn add(&self, other: &Self) -> Self {
        let x = self.to_f64();
        let y = other.to_f64();
        Self::from_f64(x + y)
    }

    /// Subtract two scientific notations.
    pub fn sub(&self, other: &Self) -> Self {
        let x = self.to_f64();
        let y = other.to_f64();
        Self::from_f64(x - y)
    }

    /// Multiply two scientific notations.
    pub fn mul(&self, other: &Self) -> Self {
        ScientificNotation {
            mantissa: self.mantissa * other.mantissa,
            exponent: self.exponent + other.exponent,
        }
    }

    /// Divide two scientific notations.
    pub fn div(&self, other: &Self) -> MathResult<Self> {
        if other.mantissa == 0.0 {
            return Err(MathError::DivisionByZero);
        }
        
        Ok(ScientificNotation {
            mantissa: self.mantissa / other.mantissa,
            exponent: self.exponent - other.exponent,
        })
    }

    /// Power of scientific notation.
    pub fn pow(&self, n: i32) -> Self {
        ScientificNotation {
            mantissa: self.mantissa.powi(n),
            exponent: self.exponent * n,
        }
    }
}

/// Engineering notation (exponent multiple of 3).
#[derive(Debug, Clone, PartialEq)]
pub struct EngineeringNotation {
    pub mantissa: f64,
    pub exponent: i32,
}

impl EngineeringNotation {
    /// Create engineering notation from a number.
    pub fn from_f64(x: f64) -> Self {
        if x == 0.0 {
            return EngineeringNotation {
                mantissa: 0.0,
                exponent: 0,
            };
        }
        
        let sci = ScientificNotation::from_f64(x);
        
        // Adjust exponent to be multiple of 3
        let remainder = sci.exponent % 3;
        let adjusted_exponent = if remainder != 0 {
            sci.exponent - remainder
        } else {
            sci.exponent
        };
        
        let adjustment = sci.exponent - adjusted_exponent;
        let adjusted_mantissa = sci.mantissa * 10.0_f64.powi(adjustment);
        
        EngineeringNotation {
            mantissa: adjusted_mantissa,
            exponent: adjusted_exponent,
        }
    }

    /// Convert back to f64.
    pub fn to_f64(&self) -> f64 {
        self.mantissa * 10.0_f64.powi(self.exponent)
    }

    /// Format as string with SI prefix.
    pub fn format_with_prefix(&self) -> String {
        let prefix = match self.exponent {
            -24 => "y",
            -21 => "z",
            -18 => "a",
            -15 => "f",
            -12 => "p",
            -9 => "n",
            -6 => "μ",
            -3 => "m",
            0 => "",
            3 => "k",
            6 => "M",
            9 => "G",
            12 => "T",
            15 => "P",
            18 => "E",
            21 => "Z",
            24 => "Y",
            _ => return format!("{} × 10^{}", self.mantissa, self.exponent),
        };
        
        format!("{} {}{}", self.mantissa, prefix, if prefix.is_empty() { "" } else { " " })
    }
}

/// Scientific notation utilities.
pub struct ScientificUtils;

impl ScientificUtils {
    /// Convert number to scientific notation string.
    pub fn to_scientific(x: f64, sig_figs: usize) -> String {
        let sci = ScientificNotation::from_f64(x);
        sci.format_with_sig_figs(sig_figs)
    }

    /// Convert number to engineering notation string.
    pub fn to_engineering(x: f64, sig_figs: usize) -> String {
        let eng = EngineeringNotation::from_f64(x);
        let rounded = ScientificNotation {
            mantissa: eng.mantissa,
            exponent: eng.exponent,
        }.round_mantissa(sig_figs);
        
        EngineeringNotation {
            mantissa: rounded.mantissa,
            exponent: eng.exponent,
        }.format_with_prefix()
    }

    /// Parse scientific notation string.
    pub fn parse(s: &str) -> MathResult<ScientificNotation> {
        let s = s.trim();
        
        // Handle E notation
        if let Some(pos) = s.find('E') {
            let mantissa_str = &s[..pos];
            let exponent_str = &s[pos + 1..];
            
            let mantissa: f64 = mantissa_str.parse()
                .map_err(|_| MathError::InvalidArgument("invalid mantissa"))?;
            let exponent: i32 = exponent_str.parse()
                .map_err(|_| MathError::InvalidArgument("invalid exponent"))?;
            
            return Ok(ScientificNotation::new(mantissa, exponent));
        }
        
        // Handle "× 10^" notation
        if let Some(pos) = s.find("× 10^") {
            let mantissa_str = &s[..pos];
            let exponent_str = &s[pos + 5..];
            
            let mantissa: f64 = mantissa_str.parse()
                .map_err(|_| MathError::InvalidArgument("invalid mantissa"))?;
            let exponent: i32 = exponent_str.parse()
                .map_err(|_| MathError::InvalidArgument("invalid exponent"))?;
            
            return Ok(ScientificNotation::new(mantissa, exponent));
        }
        
        // Plain number
        let x: f64 = s.parse()
            .map_err(|_| MathError::InvalidArgument("invalid number"))?;
        Ok(ScientificNotation::from_f64(x))
    }

    /// Compare two numbers in scientific notation.
    pub fn compare(a: f64, b: f64) -> std::cmp::Ordering {
        let sci_a = ScientificNotation::from_f64(a);
        let sci_b = ScientificNotation::from_f64(b);
        
        if sci_a.exponent != sci_b.exponent {
            sci_a.exponent.cmp(&sci_b.exponent)
        } else {
            sci_a.mantissa.partial_cmp(&sci_b.mantissa).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    /// Get order of magnitude (exponent in scientific notation).
    pub fn order_of_magnitude(x: f64) -> i32 {
        if x == 0.0 {
            return 0;
        }
        ScientificNotation::from_f64(x).exponent
    }

    /// Check if numbers are within same order of magnitude.
    pub fn same_order_of_magnitude(a: f64, b: f64) -> bool {
        Self::order_of_magnitude(a) == Self::order_of_magnitude(b)
    }

    /// Significant figures count.
    pub fn count_significant_figures(x: f64) -> usize {
        if x == 0.0 {
            return 0;
        }
        
        let sci = ScientificNotation::from_f64(x);
        let mantissa_str = format!("{:.15}", sci.mantissa.abs());
        
        // Count non-zero digits and zeros between non-zero digits
        let mut count = 0;
        let mut found_non_zero = false;
        let mut found_decimal = false;
        
        for c in mantissa_str.chars() {
            if c == '.' {
                found_decimal = true;
                continue;
            }
            
            if c != '0' {
                found_non_zero = true;
                count += 1;
            } else if found_non_zero {
                count += 1;
            }
        }
        
        count
    }

    /// Round to significant figures.
    pub fn round_to_sig_figs(x: f64, sig_figs: usize) -> f64 {
        if x == 0.0 {
            return 0.0;
        }
        
        let sci = ScientificNotation::from_f64(x);
        let rounded = sci.round_mantissa(sig_figs);
        rounded.to_f64()
    }
}

/// Unit prefixes for SI system.
pub struct UnitPrefix;

impl UnitPrefix {
    /// Get SI prefix for exponent.
    pub fn si_prefix(exponent: i32) -> &'static str {
        match exponent {
            -24 => "yocto",
            -21 => "zepto",
            -18 => "atto",
            -15 => "femto",
            -12 => "pico",
            -9 => "nano",
            -6 => "micro",
            -3 => "milli",
            -2 => "centi",
            -1 => "deci",
            0 => "",
            1 => "deca",
            2 => "hecto",
            3 => "kilo",
            6 => "mega",
            9 => "giga",
            12 => "tera",
            15 => "peta",
            18 => "exa",
            21 => "zetta",
            24 => "yotta",
            _ => "unknown",
        }
    }

    /// Get SI prefix symbol.
    pub fn si_symbol(exponent: i32) -> &'static str {
        match exponent {
            -24 => "y",
            -21 => "z",
            -18 => "a",
            -15 => "f",
            -12 => "p",
            -9 => "n",
            -6 => "μ",
            -3 => "m",
            -2 => "c",
            -1 => "d",
            0 => "",
            1 => "da",
            2 => "h",
            3 => "k",
            6 => "M",
            9 => "G",
            12 => "T",
            15 => "P",
            18 => "E",
            21 => "Z",
            24 => "Y",
            _ => "?",
        }
    }

    /// Convert value to prefixed unit.
    pub fn to_prefixed(value: f64, unit: &str) -> String {
        let eng = EngineeringNotation::from_f64(value);
        let prefix = Self::si_symbol(eng.exponent);
        
        if prefix.is_empty() {
            format!("{} {}", eng.mantissa, unit)
        } else {
            format!("{} {}{}", eng.mantissa, prefix, unit)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scientific_notation() {
        let sci = ScientificNotation::from_f64(1234.5);
        assert!((sci.mantissa - 1.2345).abs() < 1e-10);
        assert_eq!(sci.exponent, 3);
        
        let sci_zero = ScientificNotation::from_f64(0.0);
        assert_eq!(sci_zero.mantissa, 0.0);
        assert_eq!(sci_zero.exponent, 0);
    }

    #[test]
    fn test_scientific_conversion() {
        let sci = ScientificNotation::from_f64(1234.5);
        assert!((sci.to_f64() - 1234.5).abs() < 1e-10);
    }

    #[test]
    fn test_scientific_arithmetic() {
        let a = ScientificNotation::from_f64(100.0);
        let b = ScientificNotation::from_f64(200.0);
        
        let sum = a.add(&b);
        assert!((sum.to_f64() - 300.0).abs() < 1e-10);
        
        let product = a.mul(&b);
        assert!((product.to_f64() - 20000.0).abs() < 1e-10);
    }

    #[test]
    fn test_engineering_notation() {
        let eng = EngineeringNotation::from_f64(1234.5);
        assert!((eng.mantissa - 1.2345).abs() < 1e-10);
        assert_eq!(eng.exponent, 3);
    }

    #[test]
    fn test_parse_scientific() {
        let sci = ScientificUtils::parse("1.234E3").unwrap();
        assert!((sci.to_f64() - 1234.0).abs() < 1e-10);
    }

    #[test]
    fn test_order_of_magnitude() {
        assert_eq!(ScientificUtils::order_of_magnitude(1234.5), 3);
        assert_eq!(ScientificUtils::order_of_magnitude(0.001234), -3);
    }

    #[test]
    fn test_round_to_sig_figs() {
        let rounded = ScientificUtils::round_to_sig_figs(1234.5, 3);
        assert!((rounded - 1230.0).abs() < 1e-10);
    }

    #[test]
    fn test_si_prefixes() {
        assert_eq!(UnitPrefix::si_prefix(3), "kilo");
        assert_eq!(UnitPrefix::si_symbol(3), "k");
        assert_eq!(UnitPrefix::si_prefix(-6), "micro");
    }
}
