//! Comparison operations: fuzzy comparison, range checks, ordering.

use mathverse_core::error::{MathError, MathResult};

/// Fuzzy comparison with tolerance.
pub struct FuzzyCompare;

impl FuzzyCompare {
    /// Check if two numbers are approximately equal.
    pub fn equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    /// Check if a is approximately greater than b.
    pub fn greater_than(a: f64, b: f64, tolerance: f64) -> bool {
        a > b + tolerance
    }

    /// Check if a is approximately less than b.
    pub fn less_than(a: f64, b: f64, tolerance: f64) -> bool {
        a < b - tolerance
    }

    /// Check if a is approximately greater than or equal to b.
    pub fn greater_or_equal(a: f64, b: f64, tolerance: f64) -> bool {
        a >= b - tolerance
    }

    /// Check if a is approximately less than or equal to b.
    pub fn less_or_equal(a: f64, b: f64, tolerance: f64) -> bool {
        a <= b + tolerance
    }

    /// Compare with relative tolerance.
    pub fn equal_relative(a: f64, b: f64, relative_tolerance: f64) -> bool {
        if a == b {
            return true;
        }
        
        let diff = (a - b).abs();
        let max_abs = a.abs().max(b.abs());
        
        diff / max_abs < relative_tolerance
    }

    /// Compare with mixed absolute and relative tolerance.
    pub fn equal_mixed(
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
}

/// Range checking operations.
pub struct RangeCheck;

impl RangeCheck {
    /// Check if value is within inclusive range [min, max].
    pub fn in_range(value: f64, min: f64, max: f64) -> bool {
        value >= min && value <= max
    }

    /// Check if value is within exclusive range (min, max).
    pub fn in_range_exclusive(value: f64, min: f64, max: f64) -> bool {
        value > min && value < max
    }

    /// Check if value is within half-open range [min, max).
    pub fn in_range_half_open(value: f64, min: f64, max: f64) -> bool {
        value >= min && value < max
    }

    /// Clamp value to range.
    pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }

    /// Check if value is within range with tolerance.
    pub fn in_range_fuzzy(value: f64, min: f64, max: f64, tolerance: f64) -> bool {
        value >= min - tolerance && value <= max + tolerance
    }

    /// Check if two ranges overlap.
    pub fn ranges_overlap(min1: f64, max1: f64, min2: f64, max2: f64) -> bool {
        !(max1 < min2 || max2 < min1)
    }

    /// Get intersection of two ranges.
    pub fn range_intersection(min1: f64, max1: f64, min2: f64, max2: f64) -> Option<(f64, f64)> {
        let min = min1.max(min2);
        let max = max1.min(max2);
        
        if min <= max {
            Some((min, max))
        } else {
            None
        }
    }

    /// Get union of two ranges.
    pub fn range_union(min1: f64, max1: f64, min2: f64, max2: f64) -> Option<(f64, f64)> {
        if Self::ranges_overlap(min1, max1, min2, max2) {
            Some((min1.min(min2), max1.max(max2)))
        } else {
            None
        }
    }

    /// Check if value is close to any value in a list.
    pub fn close_to_any(value: f64, targets: &[f64], tolerance: f64) -> bool {
        targets.iter().any(|&t| (value - t).abs() < tolerance)
    }

    /// Find closest value in a list.
    pub fn find_closest(value: f64, targets: &[f64]) -> Option<f64> {
        targets.iter().min_by(|a, b| {
            (value - a).abs().partial_cmp(&(value - b).abs()).unwrap()
        }).copied()
    }
}

/// Ordering and sorting utilities.
pub struct Ordering;

impl Ordering {
    /// Compare two values with custom comparator.
    pub fn compare<T: PartialOrd>(a: T, b: T) -> std::cmp::Ordering {
        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Check if values are in non-decreasing order.
    pub fn is_sorted_ascending(values: &[f64]) -> bool {
        values.windows(2).all(|w| w[0] <= w[1])
    }

    /// Check if values are in non-increasing order.
    pub fn is_sorted_descending(values: &[f64]) -> bool {
        values.windows(2).all(|w| w[0] >= w[1])
    }

    /// Check if values are strictly increasing.
    pub fn is_strictly_increasing(values: &[f64]) -> bool {
        values.windows(2).all(|w| w[0] < w[1])
    }

    /// Check if values are strictly decreasing.
    pub fn is_strictly_decreasing(values: &[f64]) -> bool {
        values.windows(2).all(|w| w[0] > w[1])
    }

    /// Find minimum value.
    pub fn min(values: &[f64]) -> Option<f64> {
        values.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Find maximum value.
    pub fn max(values: &[f64]) -> Option<f64> {
        values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Find minimum and maximum.
    pub fn min_max(values: &[f64]) -> Option<(f64, f64)> {
        if values.is_empty() {
            return None;
        }
        
        let mut min_val = values[0];
        let mut max_val = values[0];
        
        for &v in &values[1..] {
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }
        
        Some((min_val, max_val))
    }

    /// Find median of values.
    pub fn median(values: &mut [f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = values.len();
        if n % 2 == 0 {
            Some((values[n / 2 - 1] + values[n / 2]) / 2.0)
        } else {
            Some(values[n / 2])
        }
    }

    /// Find mode (most frequent value).
    pub fn mode(values: &[f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        
        let mut counts = std::collections::HashMap::new();
        
        for &v in values {
            let key = (v * 1e6).round() as i64; // Round to handle floating point
            *counts.entry(key).or_insert(0) += 1;
        }
        
        let max_count = counts.values().max()?;
        let mode_key = counts.iter().find(|(_, &count)| count == *max_count)?.0;
        
        Some(*mode_key as f64 / 1e6)
    }

    /// Find percentile of values.
    pub fn percentile(values: &mut [f64], percentile: f64) -> Option<f64> {
        if values.is_empty() || percentile < 0.0 || percentile > 100.0 {
            return None;
        }
        
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = values.len();
        let index = (percentile / 100.0 * (n - 1) as f64).round() as usize;
        
        Some(values[index])
    }

    /// Find quartiles (Q1, Q2, Q3).
    pub fn quartiles(values: &mut [f64]) -> Option<(f64, f64, f64)> {
        let q1 = Self::percentile(values, 25.0)?;
        let q2 = Self::percentile(values, 50.0)?;
        let q3 = Self::percentile(values, 75.0)?;
        
        Some((q1, q2, q3))
    }
}

/// Interval comparison.
pub struct Interval;

impl Interval {
    /// Create interval from center and radius.
    pub fn from_center_radius(center: f64, radius: f64) -> (f64, f64) {
        (center - radius, center + radius)
    }

    /// Get center of interval.
    pub fn center(min: f64, max: f64) -> f64 {
        (min + max) / 2.0
    }

    /// Get radius of interval.
    pub fn radius(min: f64, max: f64) -> f64 {
        (max - min) / 2.0
    }

    /// Get width of interval.
    pub fn width(min: f64, max: f64) -> f64 {
        max - min
    }

    /// Check if interval contains value.
    pub fn contains(min: f64, max: f64, value: f64) -> bool {
        RangeCheck::in_range(value, min, max)
    }

    /// Check if interval contains another interval.
    pub fn contains_interval(outer_min: f64, outer_max: f64, inner_min: f64, inner_max: f64) -> bool {
        outer_min <= inner_min && outer_max >= inner_max
    }

    /// Expand interval by factor.
    pub fn expand(min: f64, max: f64, factor: f64) -> (f64, f64) {
        let center = Self::center(min, max);
        let new_radius = Self::radius(min, max) * factor;
        Self::from_center_radius(center, new_radius)
    }

    /// Contract interval by factor.
    pub fn contract(min: f64, max: f64, factor: f64) -> (f64, f64) {
        let center = Self::center(min, max);
        let new_radius = Self::radius(min, max) * factor;
        Self::from_center_radius(center, new_radius)
    }

    /// Translate interval by offset.
    pub fn translate(min: f64, max: f64, offset: f64) -> (f64, f64) {
        (min + offset, max + offset)
    }

    /// Scale interval around center.
    pub fn scale(min: f64, max: f64, scale: f64) -> (f64, f64) {
        let center = Self::center(min, max);
        let new_radius = Self::radius(min, max) * scale;
        Self::from_center_radius(center, new_radius)
    }
}

/// Sign and signum operations.
pub struct Sign;

impl Sign {
    /// Get sign of number: -1, 0, or 1.
    pub fn signum(x: f64) -> i32 {
        if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        }
    }

    /// Check if number is positive.
    pub fn is_positive(x: f64) -> bool {
        x > 0.0
    }

    /// Check if number is negative.
    pub fn is_negative(x: f64) -> bool {
        x < 0.0
    }

    /// Check if number is zero.
    pub fn is_zero(x: f64) -> bool {
        x == 0.0
    }

    /// Check if number is non-negative.
    pub fn is_non_negative(x: f64) -> bool {
        x >= 0.0
    }

    /// Check if number is non-positive.
    pub fn is_non_positive(x: f64) -> bool {
        x <= 0.0
    }

    /// Check if number is approximately zero.
    pub fn is_approximately_zero(x: f64, tolerance: f64) -> bool {
        x.abs() < tolerance
    }

    /// Copy sign from one number to another.
    pub fn copy_sign(magnitude: f64, sign_source: f64) -> f64 {
        if sign_source >= 0.0 {
            magnitude.abs()
        } else {
            -magnitude.abs()
        }
    }

    /// Flip sign of number.
    pub fn flip_sign(x: f64) -> f64 {
        -x
    }

    /// Get absolute value.
    pub fn abs(x: f64) -> f64 {
        x.abs()
    }
}

/// Bounded comparison.
pub struct BoundedCompare;

impl BoundedCompare {
    /// Check if value is within bounds with tolerance.
    pub fn within_bounds(value: f64, lower: f64, upper: f64, tolerance: f64) -> bool {
        value >= lower - tolerance && value <= upper + tolerance
    }

    /// Check if value exceeds upper bound.
    pub fn exceeds_upper(value: f64, upper: f64, tolerance: f64) -> bool {
        value > upper + tolerance
    }

    /// Check if value is below lower bound.
    pub fn below_lower(value: f64, lower: f64, tolerance: f64) -> bool {
        value < lower - tolerance
    }

    /// Calculate distance from lower bound.
    pub fn distance_from_lower(value: f64, lower: f64) -> f64 {
        value - lower
    }

    /// Calculate distance from upper bound.
    pub fn distance_from_upper(value: f64, upper: f64) -> f64 {
        upper - value
    }

    /// Calculate distance from nearest bound.
    pub fn distance_from_nearest_bound(value: f64, lower: f64, upper: f64) -> f64 {
        let dist_lower = (value - lower).abs();
        let dist_upper = (upper - value).abs();
        dist_lower.min(dist_upper)
    }

    /// Check if value is closer to lower or upper bound.
    pub fn nearest_bound(value: f64, lower: f64, upper: f64) -> &'static str {
        let dist_lower = (value - lower).abs();
        let dist_upper = (upper - value).abs();
        
        if dist_lower < dist_upper {
            "lower"
        } else if dist_upper < dist_lower {
            "upper"
        } else {
            "equidistant"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_compare() {
        assert!(FuzzyCompare::equal(1.0, 1.0001, 0.001));
        assert!(FuzzyCompare::greater_than(2.0, 1.0, 0.0));
        assert!(FuzzyCompare::equal_relative(1000.0, 1001.0, 0.001));
    }

    #[test]
    fn test_range_check() {
        assert!(RangeCheck::in_range(5.0, 0.0, 10.0));
        assert!(!RangeCheck::in_range_exclusive(0.0, 0.0, 10.0));
        assert_eq!(RangeCheck::clamp(15.0, 0.0, 10.0), 10.0);
        assert!(RangeCheck::ranges_overlap(0.0, 5.0, 3.0, 8.0));
    }

    #[test]
    fn test_ordering() {
        assert!(Ordering::is_sorted_ascending(&[1.0, 2.0, 3.0]));
        assert!(Ordering::is_sorted_descending(&[3.0, 2.0, 1.0]));
        assert_eq!(Ordering::min(&[1.0, 2.0, 3.0]), Some(1.0));
        assert_eq!(Ordering::max(&[1.0, 2.0, 3.0]), Some(3.0));
    }

    #[test]
    fn test_median() {
        let mut values = vec![3.0, 1.0, 2.0];
        assert_eq!(Ordering::median(&mut values), Some(2.0));
        
        let mut values_even = vec![4.0, 1.0, 3.0, 2.0];
        assert_eq!(Ordering::median(&mut values_even), Some(2.5));
    }

    #[test]
    fn test_interval() {
        let (min, max) = Interval::from_center_radius(5.0, 2.0);
        assert_eq!(min, 3.0);
        assert_eq!(max, 7.0);
        
        assert_eq!(Interval::center(3.0, 7.0), 5.0);
        assert_eq!(Interval::width(3.0, 7.0), 4.0);
    }

    #[test]
    fn test_sign() {
        assert_eq!(Sign::signum(5.0), 1);
        assert_eq!(Sign::signum(-5.0), -1);
        assert_eq!(Sign::signum(0.0), 0);
        assert!(Sign::is_positive(5.0));
        assert!(Sign::is_negative(-5.0));
    }

    #[test]
    fn test_bounded_compare() {
        assert!(BoundedCompare::within_bounds(5.0, 0.0, 10.0, 0.0));
        assert!(BoundedCompare::exceeds_upper(15.0, 10.0, 0.0));
        assert!(BoundedCompare::below_lower(-5.0, 0.0, 0.0));
        assert_eq!(BoundedCompare::distance_from_lower(5.0, 0.0), 5.0);
    }
}
