//! Arithmetic: percentage, powers, roots, modulus, absolute value.
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
pub mod power;
pub mod root;
pub mod modulus;
pub mod number_theory;
pub mod scientific;
pub mod rounding;
pub mod comparison;
pub mod complex;
pub mod rational;
pub mod continued_fraction;
pub mod approximation;
pub mod interpolation;
pub mod special_functions;
pub mod numerical;
pub mod sequence;

pub use percentage::{Percentage, ProfitLoss};
pub use power::{Power, LogarithmicIdentities, ExponentialIdentities};
pub use root::{Root, RootProperties, RootSolving};
pub use modulus::{Modulus, ModularArithmetic, Division};
pub use number_theory::{Gcd, Lcm, Primes, Factorial, Fibonacci};
pub use scientific::{ScientificNotation, EngineeringNotation, ScientificUtils, UnitPrefix};
pub use rounding::{Rounding, RoundingMode, Precision, FixedPoint, DecimalFormat};
pub use comparison::{FuzzyCompare, RangeCheck, Ordering, Interval, Sign, BoundedCompare};
pub use complex::{Complex, ComplexOps, ComplexAnalysis};
pub use rational::{Rational, RationalOps, FractionUtils};
pub use continued_fraction::{SimpleContinuedFraction, GeneralizedContinuedFraction, ContinuedFractionOps, ConvergentProperties};
pub use approximation::{TaylorSeries, PadeApproximation, ChebyshevApproximation, FourierSeries, PolynomialApproximation, RationalApproximation};
pub use interpolation::{LinearInterpolation, PolynomialInterpolation, SplineInterpolation, BarycentricInterpolation, MultidimensionalInterpolation, InterpolationUtils};
pub use special_functions::{Gamma, Beta, Erf, Bessel, Airy};
pub use numerical::{Integration, Differentiation, RootFinding, Optimization};
pub use sequence::{ArithmeticSequence, GeometricSequence, HarmonicSequence, FibonacciSequence, LucasSequence, TriangularNumbers, SquareNumbers, PentagonalNumbers, HexagonalNumbers, CatalanNumbers, PrimeSequence, SequenceOps, RecurrenceRelations};

/// `x` scaled by a percentage: `percentage(200, 10)` = 20.
pub fn percentage<T: Real>(x: T, percent: T) -> T {
    x * percent / T::from_f64(100.0)
}

/// `part` as a percentage of `whole`. `whole == 0` yields +/-inf.
pub fn percent_of<T: Real>(part: T, whole: T) -> T {
    part / whole * T::from_f64(100.0)
}

/// Relative change from `from` to `to`, in percent.
pub fn percent_change<T: Real>(from: T, to: T) -> T {
    percent_of(to - from, from)
}

/// `x²`.
pub fn square<T: Num>(x: T) -> T {
    x * x
}

/// `x³`.
pub fn cube<T: Num>(x: T) -> T {
    x * x * x
}

/// `x^(1/2)`; negative input -> NaN (documented std behavior).
pub fn square_root<T: Real>(x: T) -> T {
    x.sqrt()
}

/// `x^(1/3)`, defined for all reals (`powf` is NaN for negative bases,
/// so compute `sign(x) * |x|^(1/3)`).
pub fn cube_root<T: Real>(x: T) -> T {
    let s = if x.is_negative() { -T::one() } else { T::one() };
    s * x.abs().powf(T::from_f64(1.0 / 3.0))
}

/// `base^exp` for unsigned integer exponents, by squaring (O(log exp)).
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
        assert_eq!(pow(2u64, 10), 1024);
        assert_eq!(pow(3.0f64, 0), 1.0);
    }

    #[test]
    fn modulus_and_abs() {
        assert_eq!(modulus(17, 5).unwrap(), 2);
        assert_eq!(modulus(17, 0), Err(mathverse_core::error::MathError::DivisionByZero));
        assert_eq!(absolute(-4.5), 4.5);
    }
}
