//! Numeric abstraction traits. Every MathVerse crate is generic over these.

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

/// Foundation: anything that can add, subtract, multiply, and has 0/1.
pub trait Num:
    Copy + PartialEq + Debug + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Convert from a signed integer (lossy for floats below 2^53).
    fn from_i64(v: i64) -> Self;
}

/// Signed numbers: negation, absolute value, sign.
pub trait Signed: Num + Neg<Output = Self> + PartialOrd {
    /// Absolute value.
    fn abs(self) -> Self;
    /// -1, 0, or +1.
    fn signum(self) -> Self;
    fn is_negative(self) -> bool;
}

/// Division ring: everything `Num` is, plus division and reciprocal.
pub trait Field: Num + Div<Output = Self> + Rem<Output = Self> {
    fn div(self, other: Self) -> Self;
    /// `1 / self`.
    fn reciprocal(self) -> Self;
}

/// Real numbers: transcendental functions, powers, rounding, classification.
///
/// Implemented for `f32` and `f64`. Functions take radians unless a `*_deg`
/// variant exists.
pub trait Real: Num + Signed + Div<Output = Self> + Rem<Output = Self> + PartialOrd {
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;

    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn powf(self, e: Self) -> Self;
    fn powi(self, e: i32) -> Self;
    fn exp(self) -> Self;
    fn exp_m1(self) -> Self;
    fn ln(self) -> Self;
    fn ln_1p(self) -> Self;
    fn log(self, base: Self) -> Self;
    fn log10(self) -> Self;
    fn log2(self) -> Self;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;

    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;

    fn sin_cos(self) -> (Self, Self);
    fn hypot(self, other: Self) -> Self;
    fn copysign(self, other: Self) -> Self;
    fn abs_sub(self, other: Self) -> Self;
    fn recip(self) -> Self;

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;

    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;

    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_normal(self) -> bool;
    fn is_subnormal(self) -> bool;
    fn is_sign_negative(self) -> bool;
    fn is_sign_positive(self) -> bool;

    fn signum(self) -> Self;
}

macro_rules! impl_num_int {
    ($($t:ty),* $(,)?) => {$(
        impl Num for $t {
            fn zero() -> Self { 0 }
            fn one() -> Self { 1 }
            fn from_i64(v: i64) -> Self { v as $t }
        }
    )*};
}
impl_num_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

macro_rules! impl_num_float {
    ($($t:ty),* $(,)?) => {$(
        impl Num for $t {
            fn zero() -> Self { 0.0 }
            fn one() -> Self { 1.0 }
            fn from_i64(v: i64) -> Self { v as $t }
        }
    )*};
}
impl_num_float!(f32, f64);

macro_rules! impl_signed_int {
    ($($t:ty),* $(,)?) => {$(
        impl Signed for $t {
            fn abs(self) -> Self { self.abs() }
            fn signum(self) -> Self { self.signum() }
            fn is_negative(self) -> bool { self < 0 }
        }
    )*};
}
impl_signed_int!(i8, i16, i32, i64, isize);

macro_rules! impl_signed_float {
    ($($t:ty),* $(,)?) => {$(
        impl Signed for $t {
            fn abs(self) -> Self { self.abs() }
            fn signum(self) -> Self { self.signum() }
            fn is_negative(self) -> bool { self.is_sign_negative() }
        }
    )*};
}
impl_signed_float!(f32, f64);

macro_rules! impl_field_float {
    ($($t:ty),* $(,)?) => {$(
        impl Field for $t {
            fn div(self, other: Self) -> Self { self / other }
            fn reciprocal(self) -> Self { 1.0 / self }
        }
    )*};
}
impl_field_float!(f32, f64);

macro_rules! impl_real {
    ($($t:ty),* $(,)?) => {$(
        impl Real for $t {
            fn from_f64(v: f64) -> Self { v as $t }
            fn to_f64(self) -> f64 { self as f64 }
            fn sqrt(self) -> Self { self.sqrt() }
            fn cbrt(self) -> Self { self.cbrt() }
            fn powf(self, e: Self) -> Self { self.powf(e) }
            fn powi(self, e: i32) -> Self { self.powi(e) }
            fn exp(self) -> Self { self.exp() }
            fn exp_m1(self) -> Self { self.exp_m1() }
            fn ln(self) -> Self { self.ln() }
            fn ln_1p(self) -> Self { self.ln_1p() }
            fn log(self, base: Self) -> Self { self.log(base) }
            fn log10(self) -> Self { self.log10() }
            fn log2(self) -> Self { self.log2() }
            fn sin(self) -> Self { self.sin() }
            fn cos(self) -> Self { self.cos() }
            fn tan(self) -> Self { self.tan() }
            fn asin(self) -> Self { self.asin() }
            fn acos(self) -> Self { self.acos() }
            fn atan(self) -> Self { self.atan() }
            fn atan2(self, other: Self) -> Self { self.atan2(other) }
            fn sinh(self) -> Self { self.sinh() }
            fn cosh(self) -> Self { self.cosh() }
            fn tanh(self) -> Self { self.tanh() }
            fn asinh(self) -> Self { self.asinh() }
            fn acosh(self) -> Self { self.acosh() }
            fn atanh(self) -> Self { self.atanh() }
            fn sin_cos(self) -> (Self, Self) { self.sin_cos() }
            fn hypot(self, other: Self) -> Self { self.hypot(other) }
            fn copysign(self, other: Self) -> Self { self.copysign(other) }
            fn abs_sub(self, other: Self) -> Self { (self - other).max(Self::zero()) }
            fn recip(self) -> Self { self.recip() }
            fn floor(self) -> Self { self.floor() }
            fn ceil(self) -> Self { self.ceil() }
            fn round(self) -> Self { self.round() }
            fn trunc(self) -> Self { self.trunc() }
            fn fract(self) -> Self { self.fract() }
            fn min(self, other: Self) -> Self { self.min(other) }
            fn max(self, other: Self) -> Self { self.max(other) }
            fn is_finite(self) -> bool { self.is_finite() }
            fn is_nan(self) -> bool { self.is_nan() }
            fn is_infinite(self) -> bool { self.is_infinite() }
            fn is_normal(self) -> bool { self.is_normal() }
            fn is_subnormal(self) -> bool { self.is_subnormal() }
            fn is_sign_negative(self) -> bool { self.is_sign_negative() }
            fn is_sign_positive(self) -> bool { self.is_sign_positive() }
            fn signum(self) -> Self { self.signum() }
        }
    )*};
}
impl_real!(f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_laws() {
        for a in 0..100i64 {
            assert_eq!(a + <i64 as Num>::zero(), a);
            assert_eq!(a * <i64 as Num>::one(), a);
        }
    }

    #[test]
    fn signed_abs() {
        assert_eq!((-5i32).abs(), 5);
        assert_eq!((-2.5f64).abs(), 2.5);
        assert_eq!((-2.5f64).signum(), -1.0);
    }

    #[test]
    fn real_ops() {
        let x = 9.0f64;
        assert_eq!(x.sqrt(), 3.0);
        assert_eq!(x.cbrt(), 2.080083823051904);
        assert_eq!(x.powi(2), 81.0);
        assert_eq!(x.powf(0.5), 3.0);
        assert_eq!(x.ln(), 2.0 * 3.0f64.ln());
        assert!((1.0f64.sin().powi(2) + 1.0f64.cos().powi(2) - 1.0).abs() < 1e-12);
        // inverse trig
        assert!((0.5f64.asin() - 0.5f64.asin()).abs() < 1e-12);
        assert!((0.5f64.acos() - 0.5f64.acos()).abs() < 1e-12);
        assert!((1.0f64.atan() - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((1.0f64.atan2(1.0) - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
        // hyperbolic
        assert!((1.0f64.cosh().powi(2) - 1.0f64.sinh().powi(2) - 1.0).abs() < 1e-12);
        assert!((0.5f64.tanh() - 0.5f64.tanh()).abs() < 1e-12);
        assert!((1.0f64.asinh() - 1.0f64.asinh()).abs() < 1e-12);
        // exp_m1, ln_1p
        assert!((0.001f64.exp_m1() - (0.001f64.exp() - 1.0)).abs() < 1e-15);
        assert!((0.001f64.ln_1p() - 0.001f64.ln_1p()).abs() < 1e-15);
        // sin_cos
        let (s, c) = 0.0f64.sin_cos();
        assert!((s - 0.0f64.sin()).abs() < 1e-15);
        assert!((c - 0.0f64.cos()).abs() < 1e-15);
        // hypot, copysign, abs_sub, recip
        assert!((3.0f64.hypot(4.0) - 5.0).abs() < 1e-12);
        assert_eq!(3.0f64.copysign(-1.0), -3.0);
        assert_eq!(Real::abs_sub(5.0f64, 3.0), 2.0);
        assert_eq!(Real::abs_sub(4.0f64, 5.0), 0.0);
        assert_eq!(4.0f64.recip(), 0.25);
        // log2
        assert!((8.0f64.log2() - 3.0).abs() < 1e-12);
        // trunc, fract, signum
        assert_eq!(2.7f64.trunc(), 2.0);
        assert!((2.7f64.fract() - 0.7).abs() < 1e-12);
        assert_eq!((-2.5f64).signum(), -1.0);
        // classification
        assert!(1.0f64.is_normal());
        assert!(!0.0f64.is_normal());
        assert!(1.0f64.is_sign_positive());
        assert!((-0.0f64).is_sign_negative());
    }

    #[test]
    fn field_reciprocal() {
        assert_eq!(4.0f64.reciprocal(), 0.25);
        assert_eq!(Field::div(10.0f32, 4.0), 2.5);
    }
}
