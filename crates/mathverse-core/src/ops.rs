//! Generic scalar operations over [`Real`].

use crate::constants::{DEG_TO_GRAD, GRAD_TO_DEG, GRAD_TO_RAD};
use crate::traits::{Num, Real};
use core::ops::Rem;

/// Clamp `x` into `[lo, hi]`.
pub fn clamp<T: Real>(x: T, lo: T, hi: T) -> T {
    x.max(lo).min(hi)
}

/// Linear interpolation between `a` and `b`. `t = 0` -> `a`, `t = 1` -> `b`.
pub fn lerp<T: Real>(a: T, b: T, t: T) -> T {
    a + (b - a) * t
}

/// Inverse linear interpolation: find `t` such that `lerp(a, b, t) == x`.
/// Returns `0` when `a == b`.
pub fn lerp_inv<T: Real>(x: T, a: T, b: T) -> T {
    let d = b - a;
    if d == T::zero() {
        T::zero()
    } else {
        (x - a) / d
    }
}

/// Smoothstep, evaluated on `x` assumed within `[0, 1]`.
pub fn smoothstep<T: Real>(x: T) -> T {
    x * x * (T::from_f64(3.0) - T::from_f64(2.0) * x)
}

/// Smootherstep: C² interpolation with zero derivatives at both ends.
/// `t` assumed within `[0, 1]`.
/// Formula: `6t⁵ - 15t⁴ + 10t³`
pub fn smootherstep<T: Real>(x: T) -> T {
    let x3 = x * x * x;
    x3 * (x * x * T::from_f64(6.0) - x * T::from_f64(15.0) + T::from_f64(10.0))
}

/// Fractional part: `x - floor(x)`. Negative inputs give `[0, 1)`.
pub fn fract<T: Real>(x: T) -> T {
    x - x.floor()
}

/// Integer part: `floor(x)`.
pub fn integer_part<T: Real>(x: T) -> T {
    x.floor()
}

/// `n`-th root of `x` (any real `n`).
pub fn nth_root<T: Real>(x: T, n: i32) -> T {
    x.powf(T::one() / T::from_f64(n as f64))
}

/// Cube root of `x`.
pub fn cbrt<T: Real>(x: T) -> T {
    x.powf(T::from_f64(1.0 / 3.0))
}

/// Overflow-safe `sqrt(a² + b²)`, generic over reals.
///
/// ```
/// use mathverse_core::ops::hypot2;
/// assert_eq!(hypot2(3.0, 4.0), 5.0);
/// ```
pub fn hypot2<T: Real>(a: T, b: T) -> T {
    let (a, b) = (a.abs(), b.abs());
    let m = a.max(b);
    if m == T::zero() {
        T::zero()
    } else {
        m * (T::one() + (a.min(b) / m).powi(2)).sqrt()
    }
}

/// Overflow-safe `sqrt(a² + b² + c²)`, generic over reals.
pub fn hypot3<T: Real>(a: T, b: T, c: T) -> T {
    let m = a.abs().max(b.abs()).max(c.abs());
    if m == T::zero() {
        T::zero()
    } else {
        let (a, b, c) = (a / m, b / m, c / m);
        m * (a * a + b * b + c * c).sqrt()
    }
}

/// Sum of a slice. Empty slice -> 0.
pub fn sum<T: Num>(xs: &[T]) -> T {
    xs.iter().copied().fold(T::zero(), |acc, x| acc + x)
}

/// Product of a slice. Empty slice -> 1.
pub fn product<T: Num>(xs: &[T]) -> T {
    xs.iter().copied().fold(T::one(), |acc, x| acc * x)
}

/// Degrees -> radians.
pub fn deg_to_rad<T: Real>(d: T) -> T {
    d * T::from_f64(crate::constants::DEG_TO_RAD)
}

/// Radians -> degrees.
pub fn rad_to_deg<T: Real>(r: T) -> T {
    r * T::from_f64(crate::constants::RAD_TO_DEG)
}

/// Gradians -> degrees (1 gradian = 0.9°).
pub fn grad_to_deg<T: Real>(g: T) -> T {
    g * T::from_f64(GRAD_TO_DEG)
}

/// Degrees -> gradians.
pub fn deg_to_grad<T: Real>(d: T) -> T {
    d * T::from_f64(DEG_TO_GRAD)
}

/// Gradians -> radians.
pub fn grad_to_rad<T: Real>(g: T) -> T {
    g * T::from_f64(GRAD_TO_RAD)
}

/// Radians -> gradians.
pub fn rad_to_grad<T: Real>(r: T) -> T {
    r * T::from_f64(200.0 / core::f64::consts::PI)
}

/// Absolute value.
pub fn abs<T: Real>(x: T) -> T {
    x.abs()
}

/// Sign function: returns -1, 0, or 1.
pub fn signum<T: Real>(x: T) -> T {
    if x > T::zero() {
        T::one()
    } else if x < T::zero() {
        T::zero() - T::one()
    } else {
        T::zero()
    }
}

/// Copy the sign of `b` to the magnitude of `a`.
pub fn copysign<T: Real>(a: T, b: T) -> T {
    if b.is_negative() {
        a.abs().neg()
    } else {
        a.abs()
    }
}

/// Saturating subtraction: `max(0, a - b)`.
pub fn abs_sub<T: Real>(a: T, b: T) -> T {
    (a - b).max(T::zero())
}

/// Reciprocal: `1 / x`. Returns 0 for x == 0.
pub fn recip<T: Real>(x: T) -> T {
    if x == T::zero() {
        T::zero()
    } else {
        T::one() / x
    }
}

/// Cube root (alias for [`cbrt`]).
pub fn cube_root<T: Real>(x: T) -> T {
    cbrt(x)
}

/// Map a value from one range to another.
/// `map_range(x, 0, 10, 0, 100)` maps 5 -> 50.
pub fn map_range<T: Real>(x: T, in_lo: T, in_hi: T, out_lo: T, out_hi: T) -> T {
    let t = lerp_inv(x, in_lo, in_hi);
    lerp(out_lo, out_hi, t)
}

/// Wrap `x` into the half-open interval `[lo, hi)`.
pub fn wrap<T: Real + Rem<Output = T>>(x: T, lo: T, hi: T) -> T {
    let range = hi - lo;
    if range == T::zero() {
        return lo;
    }
    let v = x - lo;
    let r = v % range;
    if r < T::zero() {
        r + hi
    } else {
        r + lo
    }
}

/// Ping-pong: triangle wave between 0 and `length`.
pub fn ping_pong<T: Real>(x: T, length: T) -> T {
    let t = wrap(x, T::zero(), length);
    length - (length - t).abs()
}

/// Repeat `x` into `[0, length)`.
pub fn repeat<T: Real>(x: T, length: T) -> T {
    wrap(x, T::zero(), length)
}

/// Normalize an angle in radians to `[-π, π]`.
pub fn wrap_angle<T: Real + Rem<Output = T>>(x: T) -> T {
    let pi = T::from_f64(core::f64::consts::PI);
    let two_pi = T::from_f64(core::f64::consts::TAU);
    let v = x % two_pi;
    if v > pi {
        v - two_pi
    } else if v < -pi {
        v + two_pi
    } else {
        v
    }
}

/// Wrap angle in radians to `[0, 2π)`.
pub fn wrap_angle_positive<T: Real + Rem<Output = T>>(x: T) -> T {
    let two_pi = T::from_f64(core::f64::consts::TAU);
    let v = x % two_pi;
    if v < T::zero() {
        v + two_pi
    } else {
        v
    }
}

/// Distance between two values: `|a - b|`.
pub fn distance<T: Real>(a: T, b: T) -> T {
    (a - b).abs()
}

/// Normalize a vector (represented as a slice) to unit length.
/// Returns zero vector if the norm is zero.
pub fn normalize<T: Real>(xs: &[T]) -> Vec<T> {
    let norm_sq: T = xs.iter().copied().fold(T::zero(), |acc, x| acc + x * x);
    let norm = norm_sq.sqrt();
    if norm == T::zero() {
        return vec![T::zero(); xs.len()];
    }
    xs.iter().map(|&x| x / norm).collect()
}

/// Hypotenuse of two values: `sqrt(a² + b²)`.
/// Alias for [`hypot2`].
pub fn hypot<T: Real>(a: T, b: T) -> T {
    hypot2(a, b)
}

/// Sign as integer: -1, 0, or 1.
pub fn sign<T: Real>(x: T) -> i32 {
    if x > T::zero() {
        1
    } else if x < T::zero() {
        -1
    } else {
        0
    }
}

/// Truncate toward zero.
pub fn trunc<T: Real>(x: T) -> T {
    if x >= T::zero() {
        x.floor()
    } else {
        x.ceil()
    }
}

/// Fractional part toward zero: `x - trunc(x)`.
pub fn frac<T: Real>(x: T) -> T {
    x - trunc(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        assert_eq!(clamp(5.0, 0.0, 3.0), 3.0);
        assert_eq!(lerp(0.0, 10.0, 0.25), 2.5);
        assert_eq!(smoothstep(0.5), 0.5);
        assert_eq!(fract(2.75), 0.75);
        assert_eq!(fract(-0.25), 0.75);
        assert_eq!(nth_root(27.0, 3), 3.0);
    }

    #[test]
    fn hypot_no_overflow() {
        let big = 1e150f64;
        assert_eq!(hypot2(big, big), big * 2.0f64.sqrt());
        assert_eq!(hypot2(0.0, 0.0), 0.0);
        assert_eq!(hypot3(3.0, 4.0, 12.0), 13.0);
    }

    #[test]
    fn aggregates() {
        assert_eq!(sum(&[1, 2, 3, 4]), 10);
        assert_eq!(sum::<i32>(&[]), 0);
        assert_eq!(product(&[1, 2, 3, 4]), 24);
        assert_eq!(product::<f64>(&[]), 1.0);
    }

    #[test]
    fn angle_conversions() {
        assert!((rad_to_deg(deg_to_rad(45.0f64)) - 45.0).abs() < 1e-12);
        assert_eq!(grad_to_deg(100.0), 90.0);
        assert_eq!(deg_to_grad(90.0), 100.0);
        assert!((rad_to_grad(3.14159265358979f64) - 200.0f64).abs() < 1e-10);
        assert!((grad_to_rad(200.0) - core::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn new_ops() {
        assert_eq!(cbrt(27.0), 3.0);
        assert_eq!(cube_root(8.0), 2.0);
        assert_eq!(abs(-5.0), 5.0);
        assert_eq!(signum(-5.0), -1.0);
        assert_eq!(signum(0.0), 0.0);
        assert_eq!(copysign(3.0, -1.0), -3.0);
        assert_eq!(abs_sub(5.0, 3.0), 2.0);
        assert_eq!(abs_sub(3.0, 5.0), 0.0);
        assert_eq!(recip(4.0), 0.25);
        assert_eq!(recip(0.0), 0.0);
        assert_eq!(map_range(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
        assert_eq!(lerp_inv(5.0, 0.0, 10.0), 0.5);
        assert_eq!(smootherstep(0.5), 0.5);
        assert_eq!(sign(5.0), 1);
        assert_eq!(sign(-5.0), -1);
        assert_eq!(sign(0.0), 0);
        assert_eq!(trunc(2.7), 2.0);
        assert_eq!(trunc(-2.7), -2.0);
        assert!((frac(2.7f64) - 0.7f64).abs() < 1e-12);
        assert!((frac(-2.7f64) - (-0.7f64)).abs() < 1e-12);
        assert_eq!(distance(3.0, 7.0), 4.0);
    }

    #[test]
    fn wrap_and_angle() {
        assert_eq!(wrap(15.0, 0.0, 10.0), 5.0);
        assert_eq!(wrap(-3.0, 0.0, 10.0), 7.0);
        assert_eq!(repeat(15.0, 10.0), 5.0);
        assert_eq!(ping_pong(15.0, 10.0), 5.0);
        assert_eq!(ping_pong(25.0, 10.0), 5.0);
        assert!((wrap_angle(core::f64::consts::PI + 0.1) - (-core::f64::consts::PI + 0.1)).abs() < 1e-15);
        assert!((wrap_angle_positive(-0.1) - (2.0 * core::f64::consts::PI - 0.1)).abs() < 1e-15);
    }

    #[test]
    fn normalize_vec() {
        let v = normalize(&[3.0, 4.0]);
        assert!((v[0] - 0.6f64).abs() < 1e-12);
        assert!((v[1] - 0.8f64).abs() < 1e-12);
        let zero = normalize(&[0.0, 0.0]);
        assert_eq!(zero, vec![0.0, 0.0]);
    }
}
