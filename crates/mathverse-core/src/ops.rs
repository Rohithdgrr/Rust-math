//! Generic scalar operations over [`Real`].
//!
//! All functions in this module are generic over the [`Real`] trait and work
//! with both `f32` and `f64`.

use crate::constants::{DEG_TO_GRAD, GRAD_TO_DEG, GRAD_TO_RAD};
use crate::traits::{Num, Real};
use core::ops::Rem;

/// Clamp `x` into `[lo, hi]`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::clamp;
///
/// assert_eq!(clamp(5.0, 0.0, 3.0), 3.0);
/// assert_eq!(clamp(1.0, 0.0, 3.0), 1.0);
/// ```
#[must_use]
#[inline]
pub fn clamp<T: Real>(x: T, lo: T, hi: T) -> T {
    x.max(lo).min(hi)
}

/// Linear interpolation between `a` and `b`.
///
/// `t = 0` returns `a`, `t = 1` returns `b`. Values of `t` outside `[0, 1]`
/// extrapolate beyond the interval.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::lerp;
///
/// assert_eq!(lerp(0.0, 10.0, 0.25), 2.5);
/// assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
/// assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
/// ```
#[must_use]
#[inline]
pub fn lerp<T: Real>(a: T, b: T, t: T) -> T {
    a + (b - a) * t
}

/// Inverse linear interpolation: find `t` such that `lerp(a, b, t) == x`.
///
/// Returns `0` when `a == b`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::lerp_inv;
///
/// assert_eq!(lerp_inv(5.0, 0.0, 10.0), 0.5);
/// assert_eq!(lerp_inv(0.0, 0.0, 10.0), 0.0);
/// ```
#[must_use]
#[inline]
pub fn lerp_inv<T: Real>(x: T, a: T, b: T) -> T {
    let d = b - a;
    if d == T::zero() {
        T::zero()
    } else {
        (x - a) / d
    }
}

/// Smoothstep: C1 interpolation with zero derivatives at both ends.
///
/// `x` assumed within `[0, 1]`. Formula: `3t^2 - 2t^3`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::smoothstep;
///
/// assert_eq!(smoothstep(0.0), 0.0);
/// assert_eq!(smoothstep(0.5), 0.5);
/// assert_eq!(smoothstep(1.0), 1.0);
/// ```
#[must_use]
#[inline]
pub fn smoothstep<T: Real>(x: T) -> T {
    x * x * (T::from_f64(3.0) - T::from_f64(2.0) * x)
}

/// Smootherstep: C2 interpolation with zero derivatives at both ends.
///
/// `t` assumed within `[0, 1]`. Formula: `6t^5 - 15t^4 + 10t^3`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::smootherstep;
///
/// assert_eq!(smootherstep(0.0), 0.0);
/// assert_eq!(smootherstep(0.5), 0.5);
/// assert_eq!(smootherstep(1.0), 1.0);
/// ```
#[must_use]
#[inline]
pub fn smootherstep<T: Real>(x: T) -> T {
    let x3 = x * x * x;
    x3 * (x * x * T::from_f64(6.0) - x * T::from_f64(15.0) + T::from_f64(10.0))
}

/// Fractional part: `x - floor(x)`. Negative inputs give `[0, 1)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::fract;
///
/// assert!((fract(2.75_f64) - 0.75).abs() < 1e-12);
/// assert!((fract(-0.25_f64) - 0.75).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn fract<T: Real>(x: T) -> T {
    x - x.floor()
}

/// Integer part: `floor(x)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::integer_part;
///
/// assert_eq!(integer_part(2.7), 2.0);
/// assert_eq!(integer_part(-2.7), -3.0);
/// ```
#[must_use]
#[inline]
pub fn integer_part<T: Real>(x: T) -> T {
    x.floor()
}

/// `n`-th root of `x` (any real `n`).
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::nth_root;
///
/// assert!((nth_root(27.0_f64, 3) - 3.0).abs() < 1e-12);
/// assert!((nth_root(16.0_f64, 4) - 2.0).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn nth_root<T: Real>(x: T, n: i32) -> T {
    if n == 0 {
        return T::one();
    }
    x.powf(T::from_f64(f64::from(n)).recip())
}

/// Cube root of `x`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::cbrt;
///
/// assert!((cbrt(27.0_f64) - 3.0).abs() < 1e-12);
/// assert!((cbrt(8.0_f64) - 2.0).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn cbrt<T: Real>(x: T) -> T {
    if x >= T::zero() {
        x.powf(T::from_f64(1.0 / 3.0))
    } else {
        -(-x).powf(T::from_f64(1.0 / 3.0))
    }
}

/// Overflow-safe `sqrt(a^2 + b^2)`, generic over reals.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::hypot2;
///
/// assert_eq!(hypot2(3.0, 4.0), 5.0);
/// assert_eq!(hypot2(0.0, 0.0), 0.0);
/// ```
#[must_use]
pub fn hypot2<T: Real>(a: T, b: T) -> T {
    let (a, b) = (a.abs(), b.abs());
    let m = a.max(b);
    if m == T::zero() {
        T::zero()
    } else {
        m * (T::one() + (a.min(b) / m).powi(2)).sqrt()
    }
}

/// Overflow-safe `sqrt(a^2 + b^2 + c^2)`, generic over reals.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::hypot3;
///
/// assert_eq!(hypot3(3.0, 4.0, 12.0), 13.0);
/// ```
#[must_use]
pub fn hypot3<T: Real>(a: T, b: T, c: T) -> T {
    let m = a.abs().max(b.abs()).max(c.abs());
    if m == T::zero() {
        T::zero()
    } else {
        let (a, b, c) = (a / m, b / m, c / m);
        m * (a * a + b * b + c * c).sqrt()
    }
}

/// Sum of a slice. Returns `T::zero()` for an empty slice.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::sum;
///
/// assert_eq!(sum(&[1, 2, 3, 4]), 10);
/// assert_eq!(sum::<i32>(&[]), 0);
/// ```
#[must_use]
pub fn sum<T: Num>(xs: &[T]) -> T {
    xs.iter().copied().fold(T::zero(), |acc, x| acc + x)
}

/// Product of a slice. Returns `T::one()` for an empty slice.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::product;
///
/// assert_eq!(product(&[1, 2, 3, 4]), 24);
/// assert_eq!(product::<f64>(&[]), 1.0);
/// ```
#[must_use]
pub fn product<T: Num>(xs: &[T]) -> T {
    xs.iter().copied().fold(T::one(), |acc, x| acc * x)
}

/// Convert degrees to radians.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::deg_to_rad;
///
/// let rad = deg_to_rad(180.0_f64);
/// assert!((rad - core::f64::consts::PI).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn deg_to_rad<T: Real>(d: T) -> T {
    d * T::from_f64(crate::constants::DEG_TO_RAD)
}

/// Convert radians to degrees.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::rad_to_deg;
///
/// let deg = rad_to_deg(core::f64::consts::PI);
/// assert!((deg - 180.0).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn rad_to_deg<T: Real>(r: T) -> T {
    r * T::from_f64(crate::constants::RAD_TO_DEG)
}

/// Convert gradians to degrees (1 gradian = 0.9 degrees).
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::grad_to_deg;
///
/// assert_eq!(grad_to_deg(100.0), 90.0);
/// ```
#[must_use]
#[inline]
pub fn grad_to_deg<T: Real>(g: T) -> T {
    g * T::from_f64(GRAD_TO_DEG)
}

/// Convert degrees to gradians.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::deg_to_grad;
///
/// assert_eq!(deg_to_grad(90.0), 100.0);
/// ```
#[must_use]
#[inline]
pub fn deg_to_grad<T: Real>(d: T) -> T {
    d * T::from_f64(DEG_TO_GRAD)
}

/// Convert gradians to radians.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::grad_to_rad;
///
/// let rad = grad_to_rad(200.0);
/// assert!((rad - core::f64::consts::PI).abs() < 1e-15);
/// ```
#[must_use]
#[inline]
pub fn grad_to_rad<T: Real>(g: T) -> T {
    g * T::from_f64(GRAD_TO_RAD)
}

/// Convert radians to gradians.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::rad_to_grad;
///
/// let grad = rad_to_grad(core::f64::consts::PI);
/// assert!((grad - 200.0).abs() < 1e-10);
/// ```
#[must_use]
#[inline]
pub fn rad_to_grad<T: Real>(r: T) -> T {
    r * T::from_f64(200.0 / core::f64::consts::PI)
}

/// Copy the sign of `b` to the magnitude of `a`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::copysign;
///
/// assert_eq!(copysign(3.0, -1.0), -3.0);
/// assert_eq!(copysign(3.0, 1.0), 3.0);
/// ```
#[must_use]
#[inline]
pub fn copysign<T: Real>(a: T, b: T) -> T {
    if b.is_negative() {
        a.abs().neg()
    } else {
        a.abs()
    }
}

/// Saturating subtraction: `max(0, a - b)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::abs_sub;
///
/// assert_eq!(abs_sub(5.0, 3.0), 2.0);
/// assert_eq!(abs_sub(3.0, 5.0), 0.0);
/// ```
#[must_use]
#[inline]
pub fn abs_sub<T: Real>(a: T, b: T) -> T {
    (a - b).max(T::zero())
}

/// Map a value from one range to another.
///
/// `map_range(x, 0, 10, 0, 100)` maps 5 to 50.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::map_range;
///
/// assert_eq!(map_range(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
/// assert_eq!(map_range(0.0, 0.0, 10.0, 0.0, 100.0), 0.0);
/// ```
#[must_use]
pub fn map_range<T: Real>(x: T, in_lo: T, in_hi: T, out_lo: T, out_hi: T) -> T {
    let t = lerp_inv(x, in_lo, in_hi);
    lerp(out_lo, out_hi, t)
}

/// Wrap `x` into the half-open interval `[lo, hi)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::wrap;
///
/// assert_eq!(wrap(15.0, 0.0, 10.0), 5.0);
/// assert_eq!(wrap(-3.0, 0.0, 10.0), 7.0);
/// ```
#[must_use]
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
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::ping_pong;
///
/// assert_eq!(ping_pong(0.0, 10.0), 0.0);
/// assert_eq!(ping_pong(7.5, 10.0), 7.5);
/// assert_eq!(ping_pong(12.5, 10.0), 7.5);
/// assert_eq!(ping_pong(20.0, 10.0), 0.0);
/// ```
#[must_use]
#[inline]
pub fn ping_pong<T: Real>(x: T, length: T) -> T {
    let t = wrap(x, T::zero(), length * T::from_f64(2.0));
    length - (t - length).abs()
}

/// Repeat `x` into `[0, length)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::repeat;
///
/// assert_eq!(repeat(15.0, 10.0), 5.0);
/// assert_eq!(repeat(3.0, 10.0), 3.0);
/// ```
#[must_use]
#[inline]
pub fn repeat<T: Real>(x: T, length: T) -> T {
    wrap(x, T::zero(), length)
}

/// Normalize an angle in radians to `[-pi, pi]`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::wrap_angle;
///
/// let a = wrap_angle(core::f64::consts::PI + 0.1);
/// assert!((a - (-core::f64::consts::PI + 0.1)).abs() < 1e-15);
/// ```
#[must_use]
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

/// Wrap angle in radians to `[0, 2pi)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::wrap_angle_positive;
///
/// let a = wrap_angle_positive(-0.1);
/// assert!((a - (2.0 * core::f64::consts::PI - 0.1)).abs() < 1e-15);
/// ```
#[must_use]
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
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::distance;
///
/// assert_eq!(distance(3.0, 7.0), 4.0);
/// assert_eq!(distance(5.0, 5.0), 0.0);
/// ```
#[must_use]
#[inline]
pub fn distance<T: Real>(a: T, b: T) -> T {
    (a - b).abs()
}

/// Normalize a vector (represented as a slice) to unit length.
///
/// Returns a zero vector if the norm is zero.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::normalize;
///
/// let v = normalize(&[3.0_f64, 4.0]);
/// assert!((v[0] - 0.6).abs() < 1e-12);
/// assert!((v[1] - 0.8).abs() < 1e-12);
/// ```
pub fn normalize<T: Real>(xs: &[T]) -> Vec<T> {
    let norm_sq: T = xs.iter().copied().fold(T::zero(), |acc, x| acc + x * x);
    let norm = norm_sq.sqrt();
    if norm == T::zero() {
        return vec![T::zero(); xs.len()];
    }
    xs.iter().map(|&x| x / norm).collect()
}

/// Sign as integer: -1, 0, or 1.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::sign;
///
/// assert_eq!(sign(5.0), 1);
/// assert_eq!(sign(-5.0), -1);
/// assert_eq!(sign(0.0), 0);
/// ```
#[must_use]
#[inline]
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
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::trunc;
///
/// assert_eq!(trunc(2.7), 2.0);
/// assert_eq!(trunc(-2.7), -2.0);
/// ```
#[must_use]
#[inline]
pub fn trunc<T: Real>(x: T) -> T {
    if x >= T::zero() {
        x.floor()
    } else {
        x.ceil()
    }
}

/// Fractional part toward zero: `x - trunc(x)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::frac;
///
/// assert!((frac(2.7_f64) - 0.7).abs() < 1e-12);
/// assert!((frac(-2.7_f64) - (-0.7)).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn frac<T: Real>(x: T) -> T {
    x - trunc(x)
}

/// Heaviside step function: `0` if `x < edge`, `1` if `x >= edge`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::step;
///
/// assert_eq!(step(0.0, -1.0), 0.0);
/// assert_eq!(step(0.0, 0.0), 1.0);
/// assert_eq!(step(0.0, 1.0), 1.0);
/// ```
#[must_use]
#[inline]
pub fn step<T: Real>(edge: T, x: T) -> T {
    if x < edge {
        T::zero()
    } else {
        T::one()
    }
}

/// Clamp `x` into `[0, 1]`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::clamp01;
///
/// assert_eq!(clamp01(0.5), 0.5);
/// assert_eq!(clamp01(-1.0), 0.0);
/// assert_eq!(clamp01(2.0), 1.0);
/// ```
#[must_use]
#[inline]
pub fn clamp01<T: Real>(x: T) -> T {
    clamp(x, T::zero(), T::one())
}

/// Clamp `x` into `[-1, 1]`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::clamp11;
///
/// assert_eq!(clamp11(0.5), 0.5);
/// assert_eq!(clamp11(-2.0), -1.0);
/// assert_eq!(clamp11(3.0), 1.0);
/// ```
#[must_use]
#[inline]
pub fn clamp11<T: Real>(x: T) -> T {
    clamp(x, T::zero() - T::one(), T::one())
}

/// Snap `x` to the nearest multiple of `step`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::snap;
///
/// assert_eq!(snap(7.0, 5.0), 5.0);
/// assert_eq!(snap(8.0, 5.0), 10.0);
/// assert_eq!(snap(0.0, 3.0), 0.0);
/// ```
#[must_use]
pub fn snap<T: Real>(x: T, step_size: T) -> T {
    if step_size == T::zero() {
        return x;
    }
    (x / step_size).round() * step_size
}

/// Overflow-safe midpoint: `(a + b) / 2` without intermediate overflow.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::midpoint;
///
/// assert_eq!(midpoint(0.0, 10.0), 5.0);
/// assert_eq!(midpoint(-3.0, 3.0), 0.0);
/// assert_eq!(midpoint(1e200, 1e200), 1e200);
/// ```
#[must_use]
#[inline]
pub fn midpoint<T: Real>(a: T, b: T) -> T {
    let two = T::one() + T::one();
    a + (b - a) / two
}

/// Shortest-path angular interpolation between `a` and `b` in radians.
///
/// `t = 0` returns `a`, `t = 1` returns `b`. Takes the shorter arc.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::lerp_angle;
///
/// let result = lerp_angle(0.0, core::f64::consts::PI, 0.5);
/// assert!((result - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
/// ```
#[must_use]
pub fn lerp_angle<T: Real + Rem<Output = T>>(a: T, b: T, t: T) -> T {
    let pi = T::from_f64(core::f64::consts::PI);
    let two_pi = T::from_f64(core::f64::consts::TAU);
    let mut diff = (b - a) % two_pi;
    if diff > pi {
        diff = diff - two_pi;
    } else if diff < -pi {
        diff = diff + two_pi;
    }
    a + diff * t
}

/// Inverse linear interpolation: find `t` such that `lerp(a, b, t) == x`.
///
/// Alias for [`lerp_inv`].
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::inv_lerp;
///
/// assert_eq!(inv_lerp(5.0, 0.0, 10.0), 0.5);
/// assert_eq!(inv_lerp(0.0, 0.0, 10.0), 0.0);
/// ```
#[must_use]
#[inline]
pub fn inv_lerp<T: Real>(x: T, a: T, b: T) -> T {
    lerp_inv(x, a, b)
}

/// Remap `x` from range `[in_lo, in_hi]` to `[out_lo, out_hi]`.
///
/// Alias for [`map_range`] with clearer naming.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::remap;
///
/// assert_eq!(remap(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
/// assert_eq!(remap(0.0, 0.0, 10.0, 0.0, 100.0), 0.0);
/// ```
#[must_use]
#[inline]
pub fn remap<T: Real>(x: T, in_lo: T, in_hi: T, out_lo: T, out_hi: T) -> T {
    map_range(x, in_lo, in_hi, out_lo, out_hi)
}

/// Returns `true` if `x` is within `[lo, hi]` (inclusive).
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::is_between;
///
/// assert!(is_between(5.0, 0.0, 10.0));
/// assert!(!is_between(15.0, 0.0, 10.0));
/// assert!(is_between(0.0, 0.0, 10.0));
/// assert!(is_between(10.0, 0.0, 10.0));
/// ```
#[must_use]
#[inline]
pub fn is_between<T: Copy + PartialOrd>(x: T, lo: T, hi: T) -> bool {
    x >= lo && x <= hi
}

/// Linear interpolation with `t` clamped to `[0, 1]`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::lerp_clamped;
///
/// assert_eq!(lerp_clamped(0.0, 10.0, 0.25), 2.5);
/// assert_eq!(lerp_clamped(0.0, 10.0, -1.0), 0.0);
/// assert_eq!(lerp_clamped(0.0, 10.0, 2.0), 10.0);
/// ```
#[must_use]
#[inline]
pub fn lerp_clamped<T: Real>(a: T, b: T, t: T) -> T {
    lerp(a, b, clamp01(t))
}

/// Remap `x` from `[in_lo, in_hi]` to `[out_lo, out_hi]` with output clamped.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::remap_clamped;
///
/// assert_eq!(remap_clamped(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
/// assert_eq!(remap_clamped(-5.0, 0.0, 10.0, 0.0, 100.0), 0.0);
/// assert_eq!(remap_clamped(15.0, 0.0, 10.0, 0.0, 100.0), 100.0);
/// ```
#[must_use]
#[inline]
pub fn remap_clamped<T: Real>(x: T, in_lo: T, in_hi: T, out_lo: T, out_hi: T) -> T {
    lerp_clamped(out_lo, out_hi, lerp_inv(x, in_lo, in_hi))
}

/// Bilinear interpolation between four corner values.
///
/// `v00` at `(0,0)`, `v01` at `(0,1)`, `v10` at `(1,0)`, `v11` at `(1,1)`.
/// `x` and `y` are assumed within `[0, 1]`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::bilinear;
///
/// assert_eq!(bilinear(0.0, 0.0, 0.0, 0.0, 0.0, 10.0), 0.0);
/// assert_eq!(bilinear(1.0, 1.0, 0.0, 0.0, 0.0, 10.0), 10.0);
/// assert_eq!(bilinear(0.5, 0.5, 0.0, 10.0, 10.0, 0.0), 5.0);
/// ```
#[must_use]
pub fn bilinear<T: Real>(x: T, y: T, v00: T, v01: T, v10: T, v11: T) -> T {
    let top = lerp(v00, v10, x);
    let bottom = lerp(v01, v11, x);
    lerp(top, bottom, y)
}

/// Check if a slice is sorted in non-decreasing order.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::is_sorted;
///
/// assert!(is_sorted(&[1, 2, 3, 4]));
/// assert!(is_sorted(&[1, 1, 2, 3]));
/// assert!(!is_sorted(&[1, 3, 2, 4]));
/// assert!(is_sorted::<i32>(&[]));
/// assert!(is_sorted(&[42]));
/// ```
#[must_use]
pub fn is_sorted<T: PartialOrd>(xs: &[T]) -> bool {
    xs.windows(2).all(|w| w[0] <= w[1])
}

/// Minimum value in a slice. Returns `None` for an empty slice.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::min_value;
///
/// assert_eq!(min_value(&[3, 1, 4, 1, 5]), Some(&1));
/// assert_eq!(min_value::<i32>(&[]), None);
/// ```
#[must_use]
pub fn min_value<T: PartialOrd>(xs: &[T]) -> Option<&T> {
    xs.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
}

/// Maximum value in a slice. Returns `None` for an empty slice.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::max_value;
///
/// assert_eq!(max_value(&[3, 1, 4, 1, 5]), Some(&5));
/// assert_eq!(max_value::<i32>(&[]), None);
/// ```
#[must_use]
pub fn max_value<T: PartialOrd>(xs: &[T]) -> Option<&T> {
    xs.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
}

/// Mean (average) of a slice. Returns `T::zero()` for an empty slice.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::mean;
///
/// assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0]), 2.5);
/// assert_eq!(mean::<f64>(&[]), 0.0);
/// ```
#[must_use]
pub fn mean<T: Real>(xs: &[T]) -> T {
    if xs.is_empty() {
        return T::zero();
    }
    let sum: T = xs.iter().copied().fold(T::zero(), |acc, x| acc + x);
    sum / T::from_f64(xs.len() as f64)
}

/// Cumulative sum of a slice. Returns an empty vector for an empty input.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::cumsum;
///
/// assert_eq!(cumsum(&[1, 2, 3, 4]), vec![1, 3, 6, 10]);
/// assert_eq!(cumsum::<i32>(&[]), vec![]);
/// ```
#[must_use]
pub fn cumsum<T: Num + Copy + core::ops::Add<Output = T>>(xs: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(xs.len());
    let mut acc = T::zero();
    for &x in xs {
        acc = acc + x;
        result.push(acc);
    }
    result
}

/// Cumulative product of a slice. Returns an empty vector for an empty input.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::cumprod;
///
/// assert_eq!(cumprod(&[1, 2, 3, 4]), vec![1, 2, 6, 24]);
/// assert_eq!(cumprod::<i32>(&[]), vec![]);
/// ```
#[must_use]
pub fn cumprod<T: Num + Copy + core::ops::Mul<Output = T>>(xs: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(xs.len());
    let mut acc = T::one();
    for &x in xs {
        acc = acc * x;
        result.push(acc);
    }
    result
}

/// Dot product of two slices. Returns `T::zero()` if either is empty.
/// Panics if lengths differ.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::dot_product;
///
/// assert_eq!(dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
/// assert_eq!(dot_product(&[2, 3], &[4, 5]), 23);
/// ```
#[must_use]
pub fn dot_product<T: Num + Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T>>(
    a: &[T],
    b: &[T],
) -> T {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).fold(T::zero(), |acc, x| acc + x)
}

/// Parameterized smoothstep: C1 interpolation between `edge0` and `edge1`.
///
/// Returns `0` when `x <= edge0`, `1` when `x >= edge1`, and a smooth
/// Hermite interpolation between them otherwise.
/// Formula: `t = clamp((x - edge0) / (edge1 - edge0), 0, 1)`,
/// result: `t^2 * (3 - 2t)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::ops::smoothstep_between;
///
/// assert_eq!(smoothstep_between(0.0, 1.0, 0.0), 0.0);
/// assert_eq!(smoothstep_between(0.0, 1.0, 0.5), 0.5);
/// assert_eq!(smoothstep_between(0.0, 1.0, 1.0), 1.0);
/// assert_eq!(smoothstep_between(2.0, 5.0, 3.5), 0.5);
/// assert_eq!(smoothstep_between(0.0, 1.0, -1.0), 0.0);
/// assert_eq!(smoothstep_between(0.0, 1.0, 2.0), 1.0);
/// ```
#[must_use]
pub fn smoothstep_between<T: Real>(edge0: T, edge1: T, x: T) -> T {
    let t = clamp((x - edge0) / (edge1 - edge0), T::zero(), T::one());
    t * t * (T::from_f64(3.0) - T::from_f64(2.0) * t)
}

#[cfg(test)]
#[allow(clippy::suboptimal_flops)]
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
        assert_eq!(copysign(3.0, -1.0), -3.0);
        assert_eq!(abs_sub(5.0, 3.0), 2.0);
        assert_eq!(abs_sub(3.0, 5.0), 0.0);
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
    assert_eq!(ping_pong(0.0, 10.0), 0.0);
    assert_eq!(ping_pong(7.5, 10.0), 7.5);
    assert_eq!(ping_pong(12.5, 10.0), 7.5);
    assert_eq!(ping_pong(20.0, 10.0), 0.0);
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

    #[test]
    fn new_ops_extra() {
        assert_eq!(step(0.0, -1.0), 0.0);
        assert_eq!(step(0.0, 0.0), 1.0);
        assert_eq!(step(0.0, 1.0), 1.0);
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(-1.0), 0.0);
        assert_eq!(clamp01(2.0), 1.0);
        assert_eq!(clamp11(0.5), 0.5);
        assert_eq!(clamp11(-2.0), -1.0);
        assert_eq!(clamp11(3.0), 1.0);
        assert_eq!(snap(7.0, 5.0), 5.0);
        assert_eq!(snap(8.0, 5.0), 10.0);
        assert_eq!(snap(0.0, 3.0), 0.0);
        assert_eq!(midpoint(0.0, 10.0), 5.0);
        assert_eq!(midpoint(-3.0, 3.0), 0.0);
        assert_eq!(midpoint(1e200, 1e200), 1e200);
        let result = lerp_angle(0.0, core::f64::consts::PI, 0.5);
        assert!((result - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(inv_lerp(5.0, 0.0, 10.0), 0.5);
        assert_eq!(remap(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
    }

    #[test]
    fn smoothstep_between_tests() {
        assert_eq!(smoothstep_between(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep_between(0.0, 1.0, 0.5), 0.5);
        assert_eq!(smoothstep_between(0.0, 1.0, 1.0), 1.0);
        assert_eq!(smoothstep_between(2.0, 5.0, 3.5), 0.5);
        assert_eq!(smoothstep_between(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep_between(0.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn is_between_tests() {
        assert!(is_between(5.0, 0.0, 10.0));
        assert!(!is_between(15.0, 0.0, 10.0));
        assert!(!is_between(-1.0, 0.0, 10.0));
        assert!(is_between(0.0, 0.0, 10.0));
        assert!(is_between(10.0, 0.0, 10.0));
    }

    #[test]
    fn lerp_clamped_tests() {
        assert_eq!(lerp_clamped(0.0, 10.0, 0.25), 2.5);
        assert_eq!(lerp_clamped(0.0, 10.0, -1.0), 0.0);
        assert_eq!(lerp_clamped(0.0, 10.0, 2.0), 10.0);
    }

    #[test]
    fn remap_clamped_tests() {
        assert_eq!(remap_clamped(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
        assert_eq!(remap_clamped(-5.0, 0.0, 10.0, 0.0, 100.0), 0.0);
        assert_eq!(remap_clamped(15.0, 0.0, 10.0, 0.0, 100.0), 100.0);
    }

    #[test]
    fn bilinear_tests() {
        assert_eq!(bilinear(0.0, 0.0, 0.0, 0.0, 0.0, 10.0), 0.0);
        assert_eq!(bilinear(1.0, 1.0, 0.0, 0.0, 0.0, 10.0), 10.0);
        assert_eq!(bilinear(0.5, 0.5, 0.0, 10.0, 10.0, 0.0), 5.0);
        assert_eq!(bilinear(0.5, 0.0, 0.0, 10.0, 20.0, 30.0), 10.0);
    }

    #[test]
    fn is_sorted_tests() {
        assert!(is_sorted(&[1, 2, 3, 4]));
        assert!(is_sorted(&[1, 1, 2, 3]));
        assert!(!is_sorted(&[1, 3, 2, 4]));
        assert!(is_sorted::<i32>(&[]));
        assert!(is_sorted(&[42]));
    }

    #[test]
    fn min_max_value_tests() {
        assert_eq!(min_value(&[3, 1, 4, 1, 5]), Some(&1));
        assert_eq!(min_value::<i32>(&[]), None);
        assert_eq!(max_value(&[3, 1, 4, 1, 5]), Some(&5));
        assert_eq!(max_value::<i32>(&[]), None);
    }

    #[test]
    fn mean_tests() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(mean::<f64>(&[]), 0.0);
        assert_eq!(mean(&[5.0]), 5.0);
    }

    #[test]
    fn cumsum_tests() {
        assert_eq!(cumsum(&[1, 2, 3, 4]), vec![1, 3, 6, 10]);
        assert_eq!(cumsum::<i32>(&[]), vec![]);
        assert_eq!(cumsum(&[5]), vec![5]);
    }

    #[test]
    fn cumprod_tests() {
        assert_eq!(cumprod(&[1, 2, 3, 4]), vec![1, 2, 6, 24]);
        assert_eq!(cumprod::<i32>(&[]), vec![]);
        assert_eq!(cumprod(&[5]), vec![5]);
    }

    #[test]
    fn dot_product_tests() {
        assert_eq!(dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(dot_product(&[2, 3], &[4, 5]), 23);
        assert_eq!(dot_product::<f64>(&[], &[]), 0.0);
    }
}
