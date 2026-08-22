#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

//! Arithmetic: percentage, powers, roots, modulus, absolute value,
//! rounding modes, financial TVM, and checked/saturating operations.
//!
//! Addition, subtraction, multiplication, division, logarithms,
//! exponentials and rounding live in `mathverse-core` (traits + [`ops`] +
//! [`precision`]); this crate re-exports them and adds what was missing.

use mathverse_core::traits::{Num, Real, Signed};

pub use mathverse_core::ops::{
    clamp, deg_to_grad, deg_to_rad, fract, grad_to_deg, hypot2, lerp, nth_root, product,
    rad_to_deg, smoothstep, sum,
};
pub use mathverse_core::precision::{almost_eq, almost_eq_rel, round_to, significant_figures};

pub mod percentage;
pub mod rounding;
pub mod finance;
pub mod checked_ops;

pub use percentage::{Percentage, ProfitLoss};
pub use rounding::{round_with_mode, round_to_with_mode, round_ties_even, quantize, RoundingMode};
pub use finance::{
    future_value, present_value, annuity_future_value, annuity_present_value,
    perpetuity_present_value, growing_perpetuity, periods_to_reach, rate_for_target,
    continuous_compound, rule_of_72, rule_of_693,
};
pub use checked_ops::{
    checked_add, checked_sub, checked_mul, checked_div,
    saturating_add, saturating_sub, saturating_mul,
    approx_eq, approx_eq_rel, inverse_lerp, remap,
};

/// Compensated (Neumaier) summation of a slice — accurate to near machine
/// precision even for large, cancellative, or widely-scaled inputs.
///
/// Equivalent to Python's `math.fsum`. Returns `0.0` for empty input.
///
/// # Examples
///
/// ```
/// use mathverse_arithmetic::fsum;
///
/// let xs = [1e16, 1.0, -1e16];
/// assert_eq!(fsum(&xs), 1.0);          // naive sum gives 0.0
/// assert_eq!(fsum(&[1.0, 2.0, 3.0]), 6.0);
/// assert_eq!(fsum(&[]), 0.0);
/// ```
#[must_use]
pub fn fsum(xs: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for &x in xs {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            compensation += (sum - t) + x;
        } else {
            compensation += (x - t) + sum;
        }
        sum = t;
    }
    sum + compensation
}

/// Copy the sign of `sign` onto `magnitude`, like Python's `math.copysign`.
///
/// Generic over [`Real`]; works for `f32` and `f64`.
///
/// # Examples
///
/// ```
/// use mathverse_arithmetic::copysign;
///
/// assert_eq!(copysign(3.0f64, -1.0), -3.0);
/// assert_eq!(copysign(-3.0f64, 1.0), 3.0);
/// assert!(copysign(1.0f64, -0.0).is_sign_negative());
/// ```
#[inline]
#[must_use]
pub fn copysign<T: Real>(magnitude: T, sign: T) -> T {
    magnitude.copysign(sign)
}

/// `x` scaled by a percentage: `percentage(200, 10)` = 20.
#[must_use]
pub fn percentage<T: Real>(x: T, percent: T) -> T {
    x * percent / T::from_f64(100.0)
}

/// `part` as a percentage of `whole`. `whole == 0` yields +/-inf.
#[must_use]
pub fn percent_of<T: Real>(part: T, whole: T) -> T {
    part / whole * T::from_f64(100.0)
}

/// Relative change from `from` to `to`, in percent.
#[must_use]
pub fn percent_change<T: Real>(from: T, to: T) -> T {
    percent_of(to - from, from)
}

/// `x²`.
#[must_use]
pub fn square<T: Num>(x: T) -> T {
    x * x
}

/// `x³`.
#[must_use]
pub fn cube<T: Num>(x: T) -> T {
    x * x * x
}

/// `x^(1/2)`; negative input -> NaN (documented std behavior).
#[must_use]
pub fn square_root<T: Real>(x: T) -> T {
    x.sqrt()
}

/// `x^(1/3)`, defined for all reals via the trait's dedicated `cbrt`
/// (which preserves the sign natively).
#[must_use]
pub fn cube_root<T: Real>(x: T) -> T {
    x.cbrt()
}

/// `base^exp` for unsigned integer exponents, by squaring (O(log exp)).
#[must_use]
pub fn pow<T: Num>(base: T, mut exp: u32) -> T {
    let mut acc = T::one();
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * b;
        }
        b = b * b;
        exp >>= 1;
    }
    acc
}

/// Integer remainder `x % m`. `m == 0` -> [`MathError::DivisionByZero`].
pub fn modulus<T: Num + core::ops::Rem<Output = T>>(
    x: T,
    m: T,
) -> mathverse_core::error::MathResult<T> {
    if m == T::zero() {
        return Err(mathverse_core::error::MathError::DivisionByZero);
    }
    Ok(x % m)
}

/// Absolute value.
#[must_use]
pub fn absolute<T: Signed>(x: T) -> T {
    x.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentages() {
        assert_eq!(percentage(200.0, 10.0), 20.0);
        assert_eq!(percent_of(25.0, 200.0), 12.5);
        assert!((percent_change(100.0, 150.0) - 50.0).abs() < 1e-12);
    }

    #[test]
    fn powers_and_roots() {
        assert_eq!(square(5), 25);
        assert_eq!(cube(3), 27);
        assert_eq!(square_root(81.0), 9.0);
        assert!((cube_root(-8.0) + 2.0).abs() < 1e-12);
        // Generic: f32 path via Real::cbrt
        assert!((cube_root(-8.0f32) + 2.0).abs() < 1e-6);
        assert_eq!(pow(2u64, 10), 1024);
        assert_eq!(pow(3.0f64, 0), 1.0);
    }

    #[test]
    fn modulus_and_abs() {
        assert_eq!(modulus(17, 5).unwrap(), 2);
        assert_eq!(modulus(17, 0), Err(mathverse_core::error::MathError::DivisionByZero));
        assert_eq!(absolute(-4.5), 4.5);
    }

    #[test]
    fn compensated_sum() {
        // Cancellation that a naive sum gets wrong.
        let xs = [1e16, 1.0, -1e16];
        assert_eq!(fsum(&xs), 1.0);
        // Sum of 10_000 copies of 0.1 must be exact-ish.
        let many = vec![0.1; 10_000];
        let s = fsum(&many);
        assert!((s - 1000.0).abs() < 1e-9);
        assert_eq!(fsum(&[]), 0.0);
        // Matches naive sum for simple cases.
        assert_eq!(fsum(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn copysign_test() {
        assert_eq!(copysign(3.0f64, -1.0), -3.0);
        assert_eq!(copysign(-3.0f64, 1.0), 3.0);
        assert!(copysign(1.0f64, -0.0).is_sign_negative());
        assert!(copysign(-1.0f64, 0.0).is_sign_positive());
    }
}
