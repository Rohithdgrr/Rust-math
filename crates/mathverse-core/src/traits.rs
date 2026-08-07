//! Numeric abstraction traits.
//!
//! Every `MathVerse` crate is generic over these traits. The hierarchy is:
//!
//! ```text
//!                         ┌─────────┐
//!                         │   Num   │   Integer + Float
//!                         └────┬────┘
//!                              │
//!                     ┌────────┴────────┐
//!                     │                 │
//!                ┌────┴────┐     ┌─────┴─────┐
//!                │ Signed  │     │ Unsigned* │  (*marker, not enforced)
//!                └────┬────┘     └───────────┘
//!                     │
//!            ┌────────┴────────┐
//!            │                 │
//!       ┌────┴────┐     ┌──────┴──────┐
//!       │  Field  │     │   (marker)  │
//!       └────┬────┘     └─────────────┘
//!            │
//!       ┌────┴──────────────────────────────────┐
//!       │                Real                   │  core: conversion, powers, rounding
//!       └────┬────────┬─────────┬───────┬───────┘
//!            │        │         │       │       │
//!   Transcendental  Trig  Hyperbolic  FloatOps  FloatClass
//!            │        │         │       │       │
//!       └────┴────────┴─────────┴───────┴───────┘
//!                          │
//!                    ┌─────┴─────┐
//!                    │ RealFull  │  backward-compat supertrait
//!                    └───────────┘
//! ```

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
    /// Returns `true` if the number is negative.
    fn is_negative(self) -> bool;
}

/// Division ring: everything [`Num`] is, plus division and reciprocal.
pub trait Field: Num + Div<Output = Self> + Rem<Output = Self> {
    /// Divide self by other.
    fn div(self, other: Self) -> Self;
    /// `1 / self`.
    fn reciprocal(self) -> Self;
}

/// Core real number operations: conversion, powers, rounding, min/max.
///
/// Implemented for `f32` and `f64`. This is the minimal trait bound needed
/// by most generic math code.
pub trait Real: Num + Signed + Div<Output = Self> + Rem<Output = Self> + PartialOrd {
    /// Convert from `f64`.
    fn from_f64(v: f64) -> Self;
    /// Convert to `f64`.
    fn to_f64(self) -> f64;
    /// Machine epsilon for this type.
    fn epsilon() -> Self;

    /// Square root.
    fn sqrt(self) -> Self;
    /// Cube root.
    fn cbrt(self) -> Self;
    /// Raise to a floating-point power.
    fn powf(self, e: Self) -> Self;
    /// Raise to an integer power.
    fn powi(self, e: i32) -> Self;
    /// Reciprocal: `1 / self`.
    fn recip(self) -> Self;

    /// Floor: largest integer <= self.
    fn floor(self) -> Self;
    /// Ceiling: smallest integer >= self.
    fn ceil(self) -> Self;
    /// Round to nearest integer.
    fn round(self) -> Self;
    /// Truncate toward zero.
    fn trunc(self) -> Self;
    /// Fractional part: `self - floor(self)`.
    fn fract(self) -> Self;

    /// Copy the sign of `other` to the magnitude of `self`.
    fn copysign(self, other: Self) -> Self;
    /// `max(self - other, 0)`.
    fn abs_sub(self, other: Self) -> Self;

    /// Minimum of self and other.
    fn min(self, other: Self) -> Self;
    /// Maximum of self and other.
    fn max(self, other: Self) -> Self;
}

/// Transcendental functions: exp, log, and their variants.
pub trait Transcendental: Real {
    /// `e^self`.
    fn exp(self) -> Self;
    /// `e^self - 1`, accurate for small `self`.
    fn exp_m1(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// `ln(1 + self)`, accurate for small `self`.
    fn ln_1p(self) -> Self;
    /// Logarithm with arbitrary base.
    fn log(self, base: Self) -> Self;
    /// Log base 10.
    fn log10(self) -> Self;
    /// Log base 2.
    fn log2(self) -> Self;
}

/// Trigonometric functions (radians).
pub trait Trig: Real {
    /// Sine.
    fn sin(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Tangent.
    fn tan(self) -> Self;
    /// Arcsine.
    fn asin(self) -> Self;
    /// Arccosine.
    fn acos(self) -> Self;
    /// Arctangent.
    fn atan(self) -> Self;
    /// Two-argument arctangent: `atan(self / other)`.
    fn atan2(self, other: Self) -> Self;
    /// Sine and cosine simultaneously.
    fn sin_cos(self) -> (Self, Self);
    /// Hypotenuse: `sqrt(self^2 + other^2)`.
    fn hypot(self, other: Self) -> Self;
}

/// Hyperbolic functions.
pub trait Hyperbolic: Real {
    /// Hyperbolic sine.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn cosh(self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// Inverse hyperbolic sine.
    fn asinh(self) -> Self;
    /// Inverse hyperbolic cosine.
    fn acosh(self) -> Self;
    /// Inverse hyperbolic tangent.
    fn atanh(self) -> Self;
}

/// Floating-point classification.
pub trait FloatClass: Real {
    /// Returns `true` if self is neither infinite nor NaN.
    fn is_finite(self) -> bool;
    /// Returns `true` if self is NaN.
    fn is_nan(self) -> bool;
    /// Returns `true` if self is positive or negative infinity.
    fn is_infinite(self) -> bool;
    /// Returns `true` if self is neither zero, infinite, subnormal, nor NaN.
    fn is_normal(self) -> bool;
    /// Returns `true` if self is subnormal (denormal).
    fn is_subnormal(self) -> bool;
    /// Returns `true` if the sign bit is negative.
    fn is_sign_negative(self) -> bool;
    /// Returns `true` if the sign bit is positive (or zero).
    fn is_sign_positive(self) -> bool;
}

/// Backward-compat supertrait combining all sub-traits.
///
/// Prefer using specific sub-traits (`Trig`, `Hyperbolic`, etc.) in new code.
pub trait RealFull:
    Real + Transcendental + Trig + Hyperbolic + FloatClass
{
}
impl<T: Real + Transcendental + Trig + Hyperbolic + FloatClass> RealFull for T {}

/// Normed type: has an absolute value satisfying the triangle inequality.
///
/// `norm(x) >= 0`, `norm(x) == 0` iff `x == 0`, `norm(x + y) <= norm(x) + norm(y)`.
pub trait Normed: Signed {
    /// The norm (absolute value) of this value.
    fn norm(self) -> Self;
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

#[cfg(feature = "std")]
macro_rules! impl_real_core {
    ($($t:ty),* $(,)?) => {$(
        impl Real for $t {
            fn from_f64(v: f64) -> Self { v as $t }
            fn to_f64(self) -> f64 { f64::from(self) }
            fn epsilon() -> Self { Self::from_f64($crate::precision::EPS) }
            fn sqrt(self) -> Self { self.sqrt() }
            fn cbrt(self) -> Self { self.cbrt() }
            fn powf(self, e: Self) -> Self { self.powf(e) }
            fn powi(self, e: i32) -> Self { self.powi(e) }
            fn recip(self) -> Self { self.recip() }
            fn floor(self) -> Self { self.floor() }
            fn ceil(self) -> Self { self.ceil() }
            fn round(self) -> Self { self.round() }
            fn trunc(self) -> Self { self.trunc() }
            fn fract(self) -> Self { self.fract() }
            fn copysign(self, other: Self) -> Self { self.copysign(other) }
            fn abs_sub(self, other: Self) -> Self { (self - other).max(Self::zero()) }
            fn min(self, other: Self) -> Self { self.min(other) }
            fn max(self, other: Self) -> Self { self.max(other) }
        }
    )*};
}

#[cfg(feature = "std")]
macro_rules! impl_transcendental {
    ($($t:ty),* $(,)?) => {$(
        impl Transcendental for $t {
            fn exp(self) -> Self { self.exp() }
            fn exp_m1(self) -> Self { self.exp_m1() }
            fn ln(self) -> Self { self.ln() }
            fn ln_1p(self) -> Self { self.ln_1p() }
            fn log(self, base: Self) -> Self { self.log(base) }
            fn log10(self) -> Self { self.log10() }
            fn log2(self) -> Self { self.log2() }
        }
    )*};
}

#[cfg(feature = "std")]
macro_rules! impl_trig {
    ($($t:ty),* $(,)?) => {$(
        impl Trig for $t {
            fn sin(self) -> Self { self.sin() }
            fn cos(self) -> Self { self.cos() }
            fn tan(self) -> Self { self.tan() }
            fn asin(self) -> Self { self.asin() }
            fn acos(self) -> Self { self.acos() }
            fn atan(self) -> Self { self.atan() }
            fn atan2(self, other: Self) -> Self { self.atan2(other) }
            fn sin_cos(self) -> (Self, Self) { self.sin_cos() }
            fn hypot(self, other: Self) -> Self { self.hypot(other) }
        }
    )*};
}

#[cfg(feature = "std")]
macro_rules! impl_hyperbolic {
    ($($t:ty),* $(,)?) => {$(
        impl Hyperbolic for $t {
            fn sinh(self) -> Self { self.sinh() }
            fn cosh(self) -> Self { self.cosh() }
            fn tanh(self) -> Self { self.tanh() }
            fn asinh(self) -> Self { self.asinh() }
            fn acosh(self) -> Self { self.acosh() }
            fn atanh(self) -> Self { self.atanh() }
        }
    )*};
}

#[cfg(feature = "std")]
macro_rules! impl_float_class {
    ($($t:ty),* $(,)?) => {$(
        impl FloatClass for $t {
            fn is_finite(self) -> bool { self.is_finite() }
            fn is_nan(self) -> bool { self.is_nan() }
            fn is_infinite(self) -> bool { self.is_infinite() }
            fn is_normal(self) -> bool { self.is_normal() }
            fn is_subnormal(self) -> bool { self.is_subnormal() }
            fn is_sign_negative(self) -> bool { self.is_sign_negative() }
            fn is_sign_positive(self) -> bool { self.is_sign_positive() }
        }
    )*};
}

// std impl: uses hardware-accelerated float methods
#[cfg(feature = "std")]
mod real_impls {
    use super::{Num, Real, Transcendental, Trig, Hyperbolic, FloatClass};
    impl_real_core!(f32, f64);
    impl_transcendental!(f32, f64);
    impl_trig!(f32, f64);
    impl_hyperbolic!(f32, f64);
    impl_float_class!(f32, f64);
}

// no_std + libm impl: uses libm software implementations
#[cfg(all(not(feature = "std"), feature = "libm"))]
mod real_impls {
    use super::{Num, Real, Transcendental, Trig, Hyperbolic, FloatClass};

    macro_rules! impl_real_core_libm {
        ($($t:ty),* $(,)?) => {$(
            impl Real for $t {
                fn from_f64(v: f64) -> Self { v as $t }
                fn to_f64(self) -> f64 { f64::from(self) }
                fn epsilon() -> Self { Self::from_f64($crate::precision::EPS) }
                fn sqrt(self) -> Self { crate::libm_fallback::sqrt(self as f64) as $t }
                fn cbrt(self) -> Self { crate::libm_fallback::cbrt(self as f64) as $t }
                fn powf(self, e: Self) -> Self { crate::libm_fallback::powf(self as f64, e as f64) as $t }
                fn powi(self, e: i32) -> Self { crate::libm_fallback::powi(self as f64, e) as $t }
                fn recip(self) -> Self { 1.0 / self }
                fn floor(self) -> Self { crate::libm_fallback::floor(self as f64) as $t }
                fn ceil(self) -> Self { crate::libm_fallback::ceil(self as f64) as $t }
                fn round(self) -> Self { crate::libm_fallback::round(self as f64) as $t }
                fn trunc(self) -> Self { crate::libm_fallback::trunc(self as f64) as $t }
                fn fract(self) -> Self { self - crate::libm_fallback::floor(self as f64) as $t }
                fn copysign(self, other: Self) -> Self {
                    if other.is_sign_negative() { -self.abs() } else { self.abs() }
                }
                fn abs_sub(self, other: Self) -> Self { (self - other).max(Self::zero()) }
                fn min(self, other: Self) -> Self { if self < other { self } else { other } }
                fn max(self, other: Self) -> Self { if self > other { self } else { other } }
            }
        )*};
    }

    macro_rules! impl_transcendental_libm {
        ($($t:ty),* $(,)?) => {$(
            impl Transcendental for $t {
                fn exp(self) -> Self { crate::libm_fallback::exp(self as f64) as $t }
                fn exp_m1(self) -> Self { crate::libm_fallback::exp_m1(self as f64) as $t }
                fn ln(self) -> Self { crate::libm_fallback::ln(self as f64) as $t }
                fn ln_1p(self) -> Self { crate::libm_fallback::ln_1p(self as f64) as $t }
                fn log(self, base: Self) -> Self { crate::libm_fallback::log(self as f64, base as f64) as $t }
                fn log10(self) -> Self { crate::libm_fallback::log10(self as f64) as $t }
                fn log2(self) -> Self { crate::libm_fallback::log2(self as f64) as $t }
            }
        )*};
    }

    macro_rules! impl_trig_libm {
        ($($t:ty),* $(,)?) => {$(
            impl Trig for $t {
                fn sin(self) -> Self { crate::libm_fallback::sin(self as f64) as $t }
                fn cos(self) -> Self { crate::libm_fallback::cos(self as f64) as $t }
                fn tan(self) -> Self { crate::libm_fallback::tan(self as f64) as $t }
                fn asin(self) -> Self { crate::libm_fallback::asin(self as f64) as $t }
                fn acos(self) -> Self { crate::libm_fallback::acos(self as f64) as $t }
                fn atan(self) -> Self { crate::libm_fallback::atan(self as f64) as $t }
                fn atan2(self, other: Self) -> Self { crate::libm_fallback::atan2(self as f64, other as f64) as $t }
                fn sin_cos(self) -> (Self, Self) {
                    let (s, c) = crate::libm_fallback::sin_cos(self as f64);
                    (s as $t, c as $t)
                }
                fn hypot(self, other: Self) -> Self { crate::libm_fallback::hypot(self as f64, other as f64) as $t }
            }
        )*};
    }

    macro_rules! impl_hyperbolic_libm {
        ($($t:ty),* $(,)?) => {$(
            impl Hyperbolic for $t {
                fn sinh(self) -> Self { crate::libm_fallback::sinh(self as f64) as $t }
                fn cosh(self) -> Self { crate::libm_fallback::cosh(self as f64) as $t }
                fn tanh(self) -> Self { crate::libm_fallback::tanh(self as f64) as $t }
                fn asinh(self) -> Self { crate::libm_fallback::asinh(self as f64) as $t }
                fn acosh(self) -> Self { crate::libm_fallback::acosh(self as f64) as $t }
                fn atanh(self) -> Self { crate::libm_fallback::atanh(self as f64) as $t }
            }
        )*};
    }

    macro_rules! impl_float_class_libm {
        ($($t:ty),* $(,)?) => {$(
            impl FloatClass for $t {
                fn is_finite(self) -> bool { self.is_finite() }
                fn is_nan(self) -> bool { self.is_nan() }
                fn is_infinite(self) -> bool { self.is_infinite() }
                fn is_normal(self) -> bool { self.is_normal() }
                fn is_subnormal(self) -> bool { self.is_finite() && self != 0.0 && self.abs() < Self::MIN_POSITIVE }
                fn is_sign_negative(self) -> bool { self.is_sign_negative() }
                fn is_sign_positive(self) -> bool { self.is_sign_positive() }
            }
        )*};
    }

    impl_real_core_libm!(f32, f64);
    impl_transcendental_libm!(f32, f64);
    impl_trig_libm!(f32, f64);
    impl_hyperbolic_libm!(f32, f64);
    impl_float_class_libm!(f32, f64);
}

impl Normed for f32 {
    fn norm(self) -> Self {
        self.abs()
    }
}
impl Normed for f64 {
    fn norm(self) -> Self {
        self.abs()
    }
}
impl Normed for i8 {
    fn norm(self) -> Self {
        self.abs()
    }
}
impl Normed for i16 {
    fn norm(self) -> Self {
        self.abs()
    }
}
impl Normed for i32 {
    fn norm(self) -> Self {
        self.abs()
    }
}
impl Normed for i64 {
    fn norm(self) -> Self {
        self.abs()
    }
}
impl Normed for isize {
    fn norm(self) -> Self {
        self.abs()
    }
}

#[cfg(test)]
#[allow(clippy::suboptimal_flops, clippy::imprecise_flops)]
mod tests {
    use super::{Field, Num, Real};

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
        assert!((0.5f64.asin() - 0.5f64.asin()).abs() < 1e-12);
        assert!((0.5f64.acos() - 0.5f64.acos()).abs() < 1e-12);
        assert!((1.0f64.atan() - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((1.0f64.atan2(1.0) - core::f64::consts::FRAC_PI_4).abs() < 1e-12);
        assert!((1.0f64.cosh().powi(2) - 1.0f64.sinh().powi(2) - 1.0).abs() < 1e-12);
        assert!((0.5f64.tanh() - 0.5f64.tanh()).abs() < 1e-12);
        assert!((1.0f64.asinh() - 1.0f64.asinh()).abs() < 1e-12);
        assert!((0.001f64.exp_m1() - (0.001f64.exp() - 1.0)).abs() < 1e-15);
        assert!((0.001f64.ln_1p() - 0.001f64.ln_1p()).abs() < 1e-15);
        let (s, c) = 0.0f64.sin_cos();
        assert!((s - 0.0f64.sin()).abs() < 1e-15);
        assert!((c - 0.0f64.cos()).abs() < 1e-15);
        assert!((3.0f64.hypot(4.0) - 5.0).abs() < 1e-12);
        assert_eq!(3.0f64.copysign(-1.0), -3.0);
        assert_eq!(Real::abs_sub(5.0f64, 3.0), 2.0);
        assert_eq!(Real::abs_sub(4.0f64, 5.0), 0.0);
        assert_eq!(4.0f64.recip(), 0.25);
        assert!((8.0f64.log2() - 3.0).abs() < 1e-12);
        assert_eq!(2.7f64.trunc(), 2.0);
        assert!((2.7f64.fract() - 0.7).abs() < 1e-12);
        assert_eq!((-2.5f64).signum(), -1.0);
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
