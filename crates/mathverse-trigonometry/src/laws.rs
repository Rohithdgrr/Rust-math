//! Geometric laws: law of sines, law of cosines, Heron's formula.

use mathverse_core::traits::Real;
use crate::util::map_real as f;

/// Law of sines: a/sin(A) = b/sin(B) = c/sin(C).
/// Given angle A and opposite side a, compute side b opposite angle B.
pub fn law_of_sines_side<T: Real>(a: T, angle_a: T, angle_b: T) -> T {
    a * f(angle_b, f64::sin) / f(angle_a, f64::sin)
}

/// Law of sines: given sides a and b and angle A, compute angle B.
/// Returns the angle in radians.
/// Note: This returns only the principal value. For the ambiguous SSA case,
/// use `law_of_sines_angle_both` to get both possible angles.
pub fn law_of_sines_angle<T: Real>(a: T, b: T, angle_a: T) -> T {
    let sin_b = b * f(angle_a, f64::sin) / a;
    f(sin_b, f64::asin)
}

/// Law of sines: given sides a and b and angle A, compute both possible angles B.
/// Handles the ambiguous SSA (side-side-angle) case.
/// Returns (primary_angle, optional_secondary_angle) where:
/// - primary_angle is always the acute solution (π/2 or less)
/// - secondary_angle is Some(π - primary_angle) when two solutions exist
/// - secondary_angle is None when only one solution exists
/// 
/// The SSA case has two solutions when:
/// - a < b (side opposite given angle is shorter than other side)
/// - angle_a is acute (less than π/2)
/// - sin_b < 1 (so both B and π-B are valid)
pub fn law_of_sines_angle_both<T: Real>(a: T, b: T, angle_a: T) -> (T, Option<T>) {
    let sin_b = b * f(angle_a, f64::sin) / a;
    let sin_b_f64 = sin_b.to_f64();
    
    // No solution if sin_b > 1 or sin_b < -1
    if sin_b_f64.abs() > 1.0 {
        return (T::from_f64(f64::NAN), None);
    }
    
    // Primary angle (acute or right)
    let angle_b = f(sin_b, f64::asin);
    
    // Check for ambiguous case: two solutions exist when:
    // 1. sin_b < 1 (not a right triangle)
    // 2. a < b (side opposite given angle is shorter)
    // 3. angle_a is acute
    let has_two_solutions = sin_b_f64.abs() < 1.0 
        && a.to_f64() < b.to_f64() 
        && angle_a.to_f64() < core::f64::consts::FRAC_PI_2
        && angle_a.to_f64() > 0.0;
    
    if has_two_solutions {
        let pi = T::from_f64(core::f64::consts::PI);
        let secondary = pi - angle_b;
        (angle_b, Some(secondary))
    } else {
        (angle_b, None)
    }
}

/// Law of cosines: c² = a² + b² - 2ab·cos(C).
/// Given sides a, b and included angle C, compute opposite side c.
/// Returns NaN for geometrically impossible inputs (e.g., when the computed c² < 0).
pub fn law_of_cosines_side<T: Real>(a: T, b: T, angle_c: T) -> T {
    let c2 = a * a + b * b - T::from_f64(2.0) * a * b * f(angle_c, f64::cos);
    if c2 < T::zero() {
        T::from_f64(f64::NAN)
    } else {
        c2.sqrt()
    }
}

/// Law of cosines: given sides a, b, c, compute the angle C opposite c.
/// Returns the angle in radians.
pub fn law_of_cosines_angle<T: Real>(a: T, b: T, c: T) -> T {
    let cos_c = (a * a + b * b - c * c) / (T::from_f64(2.0) * a * b);
    let cos_c = cos_c.max(-T::one()).min(T::one());
    f(cos_c, f64::acos)
}

/// Heron's formula: area of a triangle from three sides.
/// Returns NaN for invalid triangles (those violating the triangle inequality).
pub fn heron<T: Real>(a: T, b: T, c: T) -> T {
    let s = (a + b + c) / T::from_f64(2.0);
    let product = s * (s - a) * (s - b) * (s - c);
    if product < T::zero() {
        T::from_f64(f64::NAN)
    } else {
        product.sqrt()
    }
}

/// Area of a triangle from two sides and included angle: ½ab·sin(C).
pub fn triangle_area_sas<T: Real>(a: T, b: T, angle_c: T) -> T {
    T::from_f64(0.5) * a * b * f(angle_c, f64::sin)
}

/// Area of a triangle from base and height.
pub fn triangle_area_base_height<T: Real>(base: T, height: T) -> T {
    T::from_f64(0.5) * base * height
}

/// Bearing (forward azimuth) from point A to point B.
/// Returns angle in radians clockwise from north.
pub fn bearing<T: Real>(lat1: T, lon1: T, lat2: T, lon2: T) -> T {
    let dlon = lon2 - lon1;
    let y = f(dlon, f64::sin) * f(lat2, f64::cos);
    let x = f(lat1, f64::cos) * f(lat2, f64::sin)
        - f(lat1, f64::sin) * f(lat2, f64::cos) * f(dlon, f64::cos);
    f(y, |yv| x.to_f64().atan2(yv))
}

/// Haversine distance between two points on a sphere (lat/lon in radians).
/// Returns the great-circle distance.
pub fn haversine_distance<T: Real>(lat1: T, lon1: T, lat2: T, lon2: T, radius: T) -> T {
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = f(dlat / T::from_f64(2.0), f64::sin).powi(2)
        + f(lat1, f64::cos) * f(lat2, f64::cos) * f(dlon / T::from_f64(2.0), f64::sin).powi(2);
    let c = T::from_f64(2.0) * a.sqrt().atan2((T::one() - a).sqrt());
    radius * c
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_3, FRAC_PI_4, FRAC_PI_6};

    const EPS: f64 = 1e-10;

    #[test]
    fn law_of_sines_test() {
        // 30-60-90 triangle: sides 1, √3, 2
        let a = law_of_sines_side(1.0f64, FRAC_PI_6, FRAC_PI_3);
        assert!((a - 3.0f64.sqrt()).abs() < EPS);
        let c = law_of_sines_side(1.0f64, FRAC_PI_6, core::f64::consts::FRAC_PI_2);
        assert!((c - 2.0).abs() < EPS);
    }

    #[test]
    fn law_of_sines_angle_both_test() {
        // Test single solution case (right triangle)
        let (angle, maybe_second) = law_of_sines_angle_both(3.0f64, 4.0f64, FRAC_PI_2);
        assert!((angle - FRAC_PI_3).abs() < EPS);
        assert!(maybe_second.is_none());
        
        // Test ambiguous SSA case (two solutions)
        // a=7, b=10, A=30°: two possible triangles exist
        let (angle, maybe_second) = law_of_sines_angle_both(7.0f64, 10.0f64, FRAC_PI_6);
        assert!(maybe_second.is_some());
        let second = maybe_second.unwrap();
        // The two angles should sum to π
        assert!((angle + second - core::f64::consts::PI).abs() < EPS);
        
        // Test no solution case (impossible triangle)
        let (angle, maybe_second) = law_of_sines_angle_both(3.0f64, 10.0f64, FRAC_PI_6);
        assert!(angle.is_nan());
        assert!(maybe_second.is_none());
        
        // Test case where a >= b (no ambiguity)
        let (angle, maybe_second) = law_of_sines_angle_both(10.0f64, 7.0f64, FRAC_PI_6);
        assert!(!angle.is_nan());
        assert!(maybe_second.is_none());
    }

    #[test]
    fn law_of_cosines_side_test() {
        // Right triangle: a=3, b=4, C=90° → c=5
        let c = law_of_cosines_side(3.0f64, 4.0f64, core::f64::consts::FRAC_PI_2);
        assert!((c - 5.0).abs() < EPS);
    }

    #[test]
    fn law_of_cosines_angle_test() {
        // 3-4-5 right triangle
        let angle = law_of_cosines_angle(3.0f64, 4.0f64, 5.0f64);
        assert!((angle - core::f64::consts::FRAC_PI_2).abs() < EPS);
    }

    #[test]
    fn heron_test() {
        // 3-4-5 triangle area = 6
        let area = heron(3.0f64, 4.0f64, 5.0f64);
        assert!((area - 6.0).abs() < EPS);
    }

    #[test]
    fn triangle_area_sas_test() {
        // a=3, b=4, C=90° → area = 6
        let area = triangle_area_sas(3.0f64, 4.0f64, core::f64::consts::FRAC_PI_2);
        assert!((area - 6.0).abs() < EPS);
    }

    #[test]
    fn haversine_test() {
        // Distance from equator to north pole (quarter circumference)
        let r = 1.0f64;
        let d = haversine_distance(0.0, 0.0, core::f64::consts::FRAC_PI_2, 0.0, r);
        assert!((d - core::f64::consts::FRAC_PI_2).abs() < EPS);
    }

    #[test]
    fn invalid_triangle_tests() {
        // Test law_of_cosines_side with impossible triangle
        // Sides 1, 1 with included angle π should give c² = 1 + 1 - 2*1*1*(-1) = 4
        // But with angle > π, c² can become negative
        let invalid_side = law_of_cosines_side(1.0f64, 1.0f64, core::f64::consts::PI);
        assert!(invalid_side.is_nan());
        
        // Test heron with invalid triangle (violates triangle inequality)
        // a=1, b=1, c=3: cannot form a triangle
        let invalid_area = heron(1.0f64, 1.0f64, 3.0f64);
        assert!(invalid_area.is_nan());
        
        // Another invalid case: a=2, b=3, c=10
        let invalid_area2 = heron(2.0f64, 3.0f64, 10.0f64);
        assert!(invalid_area2.is_nan());
        
        // Valid triangle should still work
        let valid_area = heron(3.0f64, 4.0f64, 5.0f64);
        assert!((valid_area - 6.0).abs() < EPS);
    }
}
