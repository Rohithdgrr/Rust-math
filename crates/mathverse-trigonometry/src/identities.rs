//! Trigonometric identities: double/half angle, sum/difference, product-to-sum, power-reduction.

use mathverse_core::traits::{Real, Trig};

// ---------------------------------------------------------------------------
// Double angle formulas
// ---------------------------------------------------------------------------

/// sin(2x) = 2 sin(x) cos(x).
pub fn sin_double<T: Real + Trig>(x: T) -> T {
    let s = x.sin();
    let c = x.cos();
    s * T::from_f64(2.0) * c
}

/// cos(2x) = cos²(x) - sin²(x) = 2cos²(x) - 1 = 1 - 2sin²(x).
pub fn cos_double<T: Real + Trig>(x: T) -> T {
    let c = x.cos();
    c * c * T::from_f64(2.0) - T::one()
}

/// tan(2x) = 2tan(x) / (1 - tan²(x)).
pub fn tan_double<T: Real + Trig>(x: T) -> T {
    let t = x.tan();
    let t2 = t * t;
    T::from_f64(2.0) * t / (T::one() - t2)
}

// ---------------------------------------------------------------------------
// Half angle formulas
// ---------------------------------------------------------------------------

/// sin(x/2) = ±√((1 - cos(x)) / 2). Uses the sign of sin(x/2).
pub fn sin_half<T: Real + Trig>(x: T) -> T {
    let c = x.cos();
    let val = ((T::one() - c) / T::from_f64(2.0)).sqrt();
    // Sign follows sin(x/2), not signum(x)
    let half_x = x / T::from_f64(2.0);
    let exact = half_x.sin();
    if exact < T::zero() {
        -val
    } else {
        val
    }
}

/// cos(x/2) = ±√((1 + cos(x)) / 2). Uses the sign of cos(x/2).
pub fn cos_half<T: Real + Trig>(x: T) -> T {
    let c = x.cos();
    let val = ((T::one() + c) / T::from_f64(2.0)).sqrt();
    // Sign follows cos(x/2), not always non-negative
    let half_x = x / T::from_f64(2.0);
    let exact = half_x.cos();
    if exact < T::zero() {
        -val
    } else {
        val
    }
}

/// tan(x/2) = sin(x) / (1 + cos(x)) = (1 - cos(x)) / sin(x).
///
/// Picks the numerically stable form: `sin(x)/(1 + cos(x))` when `cos(x) ≥ 0`
/// (avoids the `1 - cos(x)` cancellation near `x ≈ 0`), else `(1 - cos(x))/sin(x)`.
/// The branch also handles `x` a multiple of π: `s == 0, c == 1` → `0`, `s == 0, c == -1` → `±inf`.
pub fn tan_half<T: Real + Trig>(x: T) -> T {
    let s = x.sin();
    let c = x.cos();
    if c >= T::zero() {
        s / (T::one() + c)
    } else {
        (T::one() - c) / s
    }
}

// ---------------------------------------------------------------------------
// Sum/difference formulas
// ---------------------------------------------------------------------------

/// sin(a + b) = sin(a)cos(b) + cos(a)sin(b).
pub fn sin_sum<T: Real + Trig>(a: T, b: T) -> T {
    let (sa, ca) = sin_cos(a);
    let (sb, cb) = sin_cos(b);
    sa * cb + ca * sb
}

/// sin(a - b) = sin(a)cos(b) - cos(a)sin(b).
pub fn sin_diff<T: Real + Trig>(a: T, b: T) -> T {
    let (sa, ca) = sin_cos(a);
    let (sb, cb) = sin_cos(b);
    sa * cb - ca * sb
}

/// cos(a + b) = cos(a)cos(b) - sin(a)sin(b).
pub fn cos_sum<T: Real + Trig>(a: T, b: T) -> T {
    let (sa, ca) = sin_cos(a);
    let (sb, cb) = sin_cos(b);
    ca * cb - sa * sb
}

/// cos(a - b) = cos(a)cos(b) + sin(a)sin(b).
pub fn cos_diff<T: Real + Trig>(a: T, b: T) -> T {
    let (sa, ca) = sin_cos(a);
    let (sb, cb) = sin_cos(b);
    ca * cb + sa * sb
}

/// tan(a + b) = (tan(a) + tan(b)) / (1 - tan(a)tan(b)).
pub fn tan_sum<T: Real + Trig>(a: T, b: T) -> T {
    let ta = a.tan();
    let tb = b.tan();
    (ta + tb) / (T::one() - ta * tb)
}

/// tan(a - b) = (tan(a) - tan(b)) / (1 + tan(a)tan(b)).
pub fn tan_diff<T: Real + Trig>(a: T, b: T) -> T {
    let ta = a.tan();
    let tb = b.tan();
    (ta - tb) / (T::one() + ta * tb)
}

// ---------------------------------------------------------------------------
// Product-to-sum
// ---------------------------------------------------------------------------

/// sin(a)sin(b) = [cos(a-b) - cos(a+b)] / 2.
pub fn sin_sin_product<T: Real + Trig>(a: T, b: T) -> T {
    (cos_diff(a, b) - cos_sum(a, b)) / T::from_f64(2.0)
}

/// cos(a)cos(b) = [cos(a-b) + cos(a+b)] / 2.
pub fn cos_cos_product<T: Real + Trig>(a: T, b: T) -> T {
    (cos_diff(a, b) + cos_sum(a, b)) / T::from_f64(2.0)
}

/// sin(a)cos(b) = [sin(a-b) + sin(a+b)] / 2.
pub fn sin_cos_product<T: Real + Trig>(a: T, b: T) -> T {
    (sin_diff(a, b) + sin_sum(a, b)) / T::from_f64(2.0)
}

// ---------------------------------------------------------------------------
// Sum-to-product
// ---------------------------------------------------------------------------

/// sin(a) + sin(b) = 2 sin((a+b)/2) cos((a-b)/2).
pub fn sin_sum_to_product<T: Real + Trig>(a: T, b: T) -> T {
    let half_sum = (a + b) / T::from_f64(2.0);
    let half_diff = (a - b) / T::from_f64(2.0);
    T::from_f64(2.0) * half_sum.sin() * half_diff.cos()
}

/// sin(a) - sin(b) = 2 cos((a+b)/2) sin((a-b)/2).
pub fn sin_diff_to_product<T: Real + Trig>(a: T, b: T) -> T {
    let half_sum = (a + b) / T::from_f64(2.0);
    let half_diff = (a - b) / T::from_f64(2.0);
    T::from_f64(2.0) * half_sum.cos() * half_diff.sin()
}

/// cos(a) + cos(b) = 2 cos((a+b)/2) cos((a-b)/2).
pub fn cos_sum_to_product<T: Real + Trig>(a: T, b: T) -> T {
    let half_sum = (a + b) / T::from_f64(2.0);
    let half_diff = (a - b) / T::from_f64(2.0);
    T::from_f64(2.0) * half_sum.cos() * half_diff.cos()
}

/// cos(a) - cos(b) = -2 sin((a+b)/2) sin((a-b)/2).
pub fn cos_diff_to_product<T: Real + Trig>(a: T, b: T) -> T {
    let half_sum = (a + b) / T::from_f64(2.0);
    let half_diff = (a - b) / T::from_f64(2.0);
    -T::from_f64(2.0) * half_sum.sin() * half_diff.sin()
}

// ---------------------------------------------------------------------------
// Power reduction
// ---------------------------------------------------------------------------

/// sin²(x) = (1 - cos(2x)) / 2.
pub fn sin_squared<T: Real + Trig>(x: T) -> T {
    (T::one() - cos_double(x)) / T::from_f64(2.0)
}

/// cos²(x) = (1 + cos(2x)) / 2.
pub fn cos_squared<T: Real + Trig>(x: T) -> T {
    (T::one() + cos_double(x)) / T::from_f64(2.0)
}

/// tan²(x) = (1 - cos(2x)) / (1 + cos(2x)).
pub fn tan_squared<T: Real + Trig>(x: T) -> T {
    let c2 = cos_double(x);
    (T::one() - c2) / (T::one() + c2)
}

// ---------------------------------------------------------------------------
// sin_cos helper
// ---------------------------------------------------------------------------

/// Compute (sin(x), cos(x)) simultaneously (more efficient).
pub fn sin_cos<T: Real + Trig>(x: T) -> (T, T) {
    x.sin_cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mathverse_core::constants::PI;

    const EPS: f64 = 1e-12;

    #[test]
    fn double_angle_test() {
        for x in [-1.0f64, 0.3, 1.0, 2.5] {
            assert!((sin_double(x) - (2.0 * x.sin() * x.cos())).abs() < EPS, "sin_double x={x}");
            assert!((cos_double(x) - (2.0 * x.cos().powi(2) - 1.0)).abs() < EPS, "cos_double x={x}");
            assert!((tan_double(x) - (2.0 * x.tan() / (1.0 - x.tan().powi(2)))).abs() < 1e-10, "tan_double x={x}");
        }
    }

    #[test]
    fn half_angle_test() {
        for x in [0.1f64, 1.0, 2.5] {
            assert!((sin_half(x) - (x / 2.0).sin()).abs() < 1e-10, "sin_half x={x}");
            assert!((cos_half(x) - (x / 2.0).cos()).abs() < 1e-10, "cos_half x={x}");
            assert!((tan_half(x) - (x / 2.0).tan()).abs() < 1e-10, "tan_half x={x}");
        }
    }

    #[test]
    fn half_angle_edge_cases() {
        // Test sign-sensitive edge cases that catch the original bugs
        let pi = core::f64::consts::PI;
        
        // sin_half(3π) should equal sin(3π/2) = -1
        assert!((sin_half(3.0 * pi) - (1.5 * pi).sin()).abs() < 1e-10);
        
        // cos_half(2π) should equal cos(π) = -1 (not +1)
        assert!((cos_half(2.0 * pi) - pi.cos()).abs() < 1e-10);
        
        // cos_half(3π) should equal cos(3π/2) = 0
        assert!((cos_half(3.0 * pi) - (1.5 * pi).cos()).abs() < 1e-10);
        
        // sin_half(-3π) should equal sin(-3π/2) = 1
        assert!((sin_half(-3.0 * pi) - (-1.5 * pi).sin()).abs() < 1e-10);
        
        // cos_half(-2π) should equal cos(-π) = -1
        assert!((cos_half(-2.0 * pi) - (-pi).cos()).abs() < 1e-10);
        
        // tan_half(π) diverges at the tan(π/2) asymptote; f64 cannot reach ±inf
        let tan_pi_half = tan_half(pi);
        assert!(tan_pi_half.abs() > 1e14);
        
        // tan_half(2π) should be 0 (tan(π))
        assert!((tan_half(2.0 * pi) - pi.tan()).abs() < 1e-10);
    }

    #[test]
    fn sum_difference_test() {
        let a = 0.5f64;
        let b = 1.2f64;
        assert!((sin_sum(a, b) - (a + b).sin()).abs() < EPS);
        assert!((sin_diff(a, b) - (a - b).sin()).abs() < EPS);
        assert!((cos_sum(a, b) - (a + b).cos()).abs() < EPS);
        assert!((cos_diff(a, b) - (a - b).cos()).abs() < EPS);
    }

    #[test]
    fn product_to_sum_test() {
        let a = 0.5f64;
        let b = 1.2f64;
        assert!((sin_sin_product(a, b) - a.sin() * b.sin()).abs() < EPS);
        assert!((cos_cos_product(a, b) - a.cos() * b.cos()).abs() < EPS);
        assert!((sin_cos_product(a, b) - a.sin() * b.cos()).abs() < EPS);
    }

    #[test]
    fn sum_to_product_test() {
        let a = 0.5f64;
        let b = 1.2f64;
        assert!((sin_sum_to_product(a, b) - (a.sin() + b.sin())).abs() < EPS);
        assert!((cos_sum_to_product(a, b) - (a.cos() + b.cos())).abs() < EPS);
        assert!((sin_diff_to_product(a, b) - (a.sin() - b.sin())).abs() < EPS);
        assert!((cos_diff_to_product(a, b) - (a.cos() - b.cos())).abs() < EPS);
    }

    #[test]
    fn power_reduction_test() {
        for x in [0.0f64, 0.5, 1.0, PI] {
            assert!((sin_squared(x) - x.sin().powi(2)).abs() < EPS, "sin² x={x}");
            assert!((cos_squared(x) - x.cos().powi(2)).abs() < EPS, "cos² x={x}");
        }
    }
}
