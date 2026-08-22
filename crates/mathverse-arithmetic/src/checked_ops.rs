//! Checked and saturating arithmetic operations for overflow safety.

/// Checked addition: returns `None` on overflow.
#[must_use]
pub fn checked_add(a: f64, b: f64) -> Option<f64> {
    let result = a + b;
    if result.is_infinite() && (a.is_finite() && b.is_finite()) {
        None
    } else {
        Some(result)
    }
}

/// Checked subtraction: returns `None` on overflow.
#[must_use]
pub fn checked_sub(a: f64, b: f64) -> Option<f64> {
    let result = a - b;
    if result.is_infinite() && (a.is_finite() && b.is_finite()) {
        None
    } else {
        Some(result)
    }
}

/// Checked multiplication: returns `None` on overflow.
#[must_use]
pub fn checked_mul(a: f64, b: f64) -> Option<f64> {
    let result = a * b;
    if result.is_infinite() && (a.is_finite() && b.is_finite() && a != 0.0 && b != 0.0) {
        None
    } else {
        Some(result)
    }
}

/// Checked division: returns `None` on overflow or division by zero.
#[must_use]
pub fn checked_div(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 {
        None
    } else {
        let result = a / b;
        if result.is_infinite() && a.is_finite() {
            None
        } else {
            Some(result)
        }
    }
}

/// Saturating addition: clamps to `f64::MAX` / `f64::MIN` on overflow.
#[must_use]
pub fn saturating_add(a: f64, b: f64) -> f64 {
    let result = a + b;
    if result.is_infinite() {
        if (a + b) > 0.0 { f64::MAX } else { f64::MIN }
    } else {
        result
    }
}

/// Saturating subtraction: clamps to `f64::MAX` / `f64::MIN` on overflow.
#[must_use]
pub fn saturating_sub(a: f64, b: f64) -> f64 {
    let result = a - b;
    if result.is_infinite() {
        if (a - b) > 0.0 { f64::MAX } else { f64::MIN }
    } else {
        result
    }
}

/// Saturating multiplication: clamps to `f64::MAX` / `f64::MIN` on overflow.
#[must_use]
pub fn saturating_mul(a: f64, b: f64) -> f64 {
    let result = a * b;
    if result.is_infinite() {
        if (a * b) > 0.0 { f64::MAX } else { f64::MIN }
    } else {
        result
    }
}

/// Wrapping addition (modular arithmetic on the float bit pattern).
#[must_use]
pub fn wrapping_add(a: f64, b: f64) -> f64 {
    a + b
}

/// Returns `true` if `a` and `b` are approximately equal within `tol`.
#[must_use]
pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Returns `true` if `a` and `b` are approximately equal relative to their magnitude.
#[must_use]
pub fn approx_eq_rel(a: f64, b: f64, rel_tol: f64) -> bool {
    if a == b { return true; }
    let diff = (a - b).abs();
    let largest = a.abs().max(b.abs());
    diff / largest <= rel_tol
}

/// Clamps `x` to `[lo, hi]`.
#[must_use]
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.clamp(lo, hi)
}

/// Linear interpolation: `a + t * (b - a)`.
#[must_use]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Inverse linear interpolation: given `y` in `[lerp(a, b, 0), lerp(a, b, 1)]`, returns `t`.
#[must_use]
pub fn inverse_lerp(a: f64, b: f64, y: f64) -> f64 {
    if (b - a).abs() < f64::EPSILON { 0.0 } else { (y - a) / (b - a) }
}

/// Remaps `x` from `[in_lo, in_hi]` to `[out_lo, out_hi]`.
#[must_use]
pub fn remap(x: f64, in_lo: f64, in_hi: f64, out_lo: f64, out_hi: f64) -> f64 {
    let t = inverse_lerp(in_lo, in_hi, x);
    lerp(out_lo, out_hi, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_ops_test() {
        assert_eq!(checked_add(1.0, 2.0), Some(3.0));
        assert_eq!(checked_mul(f64::MAX, 2.0), None);
        assert_eq!(checked_div(1.0, 0.0), None);
    }

    #[test]
    fn saturating_ops_test() {
        assert_eq!(saturating_add(1.0, 2.0), 3.0);
        assert_eq!(saturating_mul(f64::MAX, 2.0), f64::MAX);
    }

    #[test]
    fn approx_eq_test() {
        assert!(approx_eq(1.0, 1.0 + 1e-15, 1e-10));
        assert!(!approx_eq(1.0, 2.0, 1e-10));
        assert!(approx_eq_rel(1000.0, 1000.0001, 1e-6));
    }

    #[test]
    fn lerp_test() {
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-12);
        assert!((inverse_lerp(0.0, 10.0, 5.0) - 0.5).abs() < 1e-12);
        assert!((remap(5.0, 0.0, 10.0, 100.0, 200.0) - 150.0).abs() < 1e-12);
    }
}
