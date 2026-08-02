//! Rounding-mode-aware operations.
//!
//! Provides explicit rounding modes for cases where the default
//! `f64::round()` behavior is insufficient.

/// Rounding mode for numeric operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round toward zero (truncation).
    TowardZero,
    /// Round toward positive infinity.
    Up,
    /// Round toward negative infinity.
    Down,
    /// Round to nearest, ties away from zero.
    Nearest,
    /// Round to nearest, ties to even (banker's rounding).
    Bankers,
}

/// Round `x` according to the specified mode.
pub fn round_with_mode(x: f64, mode: RoundingMode) -> f64 {
    match mode {
        RoundingMode::TowardZero => x.trunc(),
        RoundingMode::Up => x.ceil(),
        RoundingMode::Down => x.floor(),
        RoundingMode::Nearest => x.round(),
        RoundingMode::Bankers => {
            let r = x.round();
            if (x - r).abs() < 0.5 {
                r
            } else if r.fract().abs() < f64::EPSILON {
                // tie: round to even
                if r % 2.0 == 0.0 { r } else { x.round_ties_even() }
            } else {
                r
            }
        }
    }
}

/// Round `x` to `decimal_places` decimal places using the specified mode.
pub fn round_to_with_mode(x: f64, decimal_places: i32, mode: RoundingMode) -> f64 {
    let factor = 10.0_f64.powi(decimal_places);
    round_with_mode(x * factor, mode) / factor
}

/// Round `x` to the nearest integer, breaking ties to even (IEEE 754 default).
pub fn round_ties_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// Quantize `x` to have the same number of decimal places as `reference`.
pub fn quantize(x: f64, reference: f64) -> f64 {
    let precision = -reference.fract().log10().ceil() as i32;
    if precision < 0 { 0.0 } else { (x * 10.0_f64.powi(precision)).round() / 10.0_f64.powi(precision) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_modes() {
        assert_eq!(round_with_mode(1.5, RoundingMode::Nearest), 2.0);
        assert_eq!(round_with_mode(1.5, RoundingMode::TowardZero), 1.0);
        assert_eq!(round_with_mode(-1.5, RoundingMode::Down), -2.0);
        assert_eq!(round_with_mode(-1.5, RoundingMode::Up), -1.0);
    }

    #[test]
    fn round_to_places() {
        assert!((round_to_with_mode(3.14159, 2, RoundingMode::Nearest) - 3.14).abs() < 1e-12);
        assert!((round_to_with_mode(2.5, 0, RoundingMode::TowardZero) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bankers_rounding() {
        assert_eq!(round_with_mode(0.5, RoundingMode::Bankers), 0.0);
        assert_eq!(round_with_mode(1.5, RoundingMode::Bankers), 2.0);
        assert_eq!(round_with_mode(2.5, RoundingMode::Bankers), 2.0);
        assert_eq!(round_with_mode(3.5, RoundingMode::Bankers), 4.0);
    }

    #[test]
    fn quantize_test() {
        assert_eq!(quantize(1.2345, 0.01), 1.23);
        assert_eq!(quantize(1.2345, 1.0), 1.0);
    }
}
