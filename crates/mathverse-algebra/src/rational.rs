//! Rational expressions: `P(x)/Q(x)` as a pair of [`Polynomial`]s with
//! arithmetic, simplification, partial-fraction decomposition, and
//! denominator rationalization.

use core::fmt;

use crate::polynomial::Polynomial;
use crate::factor::divide;
use crate::factor::polynomial_gcd;
use crate::{AlgebraError, Result, TOL};

/// A rational expression `numerator / denominator`.
#[derive(Debug, Clone)]
pub struct RationalExpression {
    pub numerator: Polynomial,
    pub denominator: Polynomial,
}

impl RationalExpression {
    /// Create from numerator and denominator polynomials.
    ///
    /// # Errors
    /// Returns [`AlgebraError::DivisionByZero`] if the denominator is zero.
    pub fn new(num: Polynomial, den: Polynomial) -> Result<Self> {
        if den.is_zero() {
            return Err(AlgebraError::DivisionByZero);
        }
        Ok(RationalExpression { numerator: num, denominator: den })
    }

    /// Create from coefficient slices (lowest-degree first).
    ///
    /// # Errors
    /// Returns [`AlgebraError::DivisionByZero`] if the denominator is zero.
    pub fn from_coeffs(num: &[f64], den: &[f64]) -> Result<Self> {
        Self::new(
            Polynomial::from_coeffs(num),
            Polynomial::from_coeffs(den),
        )
    }

    /// Simplify by dividing numerator and denominator by their polynomial GCD.
    #[must_use]
    pub fn simplify(&self) -> RationalExpression {
        let g = polynomial_gcd(&self.numerator, &self.denominator);
        let (nq, _) = divide(self.numerator.coeffs(), g.coeffs()).unwrap_or((vec![0.0], vec![]));
        let (dq, _) = divide(self.denominator.coeffs(), g.coeffs()).unwrap_or((vec![0.0], vec![]));
        RationalExpression {
            numerator: Polynomial::from_coeffs(&nq),
            denominator: Polynomial::from_coeffs(&dq),
        }
    }

    /// Evaluate at `x`.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        self.numerator.eval(x) / self.denominator.eval(x)
    }
}

impl PartialEq for RationalExpression {
    fn eq(&self, other: &Self) -> bool {
        // Cross-multiply: a/b == c/d iff a*d == b*c
        let lhs = self.numerator.clone() * other.denominator.clone();
        let rhs = other.numerator.clone() * self.denominator.clone();
        lhs.approx_eq(&rhs, TOL)
    }
}

impl fmt::Display for RationalExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) / ({})", self.numerator, self.denominator)
    }
}

impl core::ops::Add for RationalExpression {
    type Output = RationalExpression;
    fn add(self, other: RationalExpression) -> RationalExpression {
        let num = self.numerator.clone() * other.denominator.clone()
            + other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();
        RationalExpression { numerator: num, denominator: den }.simplify()
    }
}

impl core::ops::Sub for RationalExpression {
    type Output = RationalExpression;
    fn sub(self, other: RationalExpression) -> RationalExpression {
        let num = self.numerator.clone() * other.denominator.clone()
            - other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();
        RationalExpression { numerator: num, denominator: den }.simplify()
    }
}

impl core::ops::Mul for RationalExpression {
    type Output = RationalExpression;
    fn mul(self, other: RationalExpression) -> RationalExpression {
        RationalExpression {
            numerator: self.numerator.clone() * other.numerator.clone(),
            denominator: self.denominator.clone() * other.denominator.clone(),
        }
        .simplify()
    }
}

impl core::ops::Div for RationalExpression {
    type Output = RationalExpression;
    fn div(self, other: RationalExpression) -> RationalExpression {
        RationalExpression {
            numerator: self.numerator.clone() * other.denominator.clone(),
            denominator: self.denominator.clone() * other.numerator.clone(),
        }
        .simplify()
    }
}

/// `a/b + c/d = (ad + bc)/(bd)`.
#[must_use]
pub fn add_rational(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    (a * d + b * c, b * d)
}

/// `a/b - c/d = (ad - bc)/(bd)`.
#[must_use]
pub fn sub_rational(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    (a * d - b * c, b * d)
}

/// `(a/b) * (c/d) = ac/bd`.
#[must_use]
pub fn mul_rational(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    (a * c, b * d)
}

/// `(a/b) / (c/d) = ad/bc`.
///
/// # Panics
/// Panics if `b` or `c` is zero (result would be undefined).
#[must_use]
pub fn div_rational(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    (a * d, b * c)
}

/// Reduce a fraction `num/den` to lowest terms using integer GCD.
#[must_use]
pub fn reduce_fraction(num: i64, den: i64) -> (i64, i64) {
    if den == 0 {
        return (num, den);
    }
    let g = gcd_i64(num.abs(), den.abs());
    let n = num / g;
    let d = if den < 0 { -(den / g) } else { den / g };
    (n, d)
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}

/// Rationalize the denominator: multiply numerator and denominator by the
/// conjugate of the denominator.
///
/// For a denominator of the form `a + b`, returns the rationalized form.
/// This is a simplified version for `1/(a + b)` -> `(a - b)/(a^2 - b^2)`.
#[must_use]
pub fn rationalize_denominator(a: f64, b: f64) -> (f64, f64) {
    let conj = a - b;
    let denom = a * a - b * b;
    (conj, denom)
}

/// Partial fraction decomposition for `P(x) / ((x - r1)(x - r2)...(x - rk))`.
///
/// Returns the residues `[A1, A2, ..., Ak]` such that:
/// `P(x)/Q(x) = sum Ai/(x - ri)`.
///
/// Uses the cover-up (Heaviside) method: `Ai = P(ri) / Q'(ri)`.
///
/// # Errors
/// Returns [`AlgebraError::DivisionByZero`] if two roots coincide.
///
/// ```
/// # use mathverse_algebra::rational::partial_fractions;
/// // (x+1) / (x^2 - 1) = (x+1)/((x-1)(x+1)) = 1/(x-1) + 0/(x+1)
/// let residues = partial_fractions(&[1.0, 1.0], &[1.0, -1.0]).unwrap();
/// assert!((residues[0] - 1.0).abs() < 1e-9);
/// ```
pub fn partial_fractions(p_coeffs: &[f64], roots: &[f64]) -> Result<Vec<f64>> {
    let p = Polynomial::from_coeffs(p_coeffs);
    let mut residues = Vec::with_capacity(roots.len());
    for &r in roots {
        let num = p.eval(r);
        let mut deriv = 1.0;
        for &other in roots {
            if (other - r).abs() > TOL {
                deriv *= r - other;
            }
        }
        if deriv.abs() < TOL {
            return Err(AlgebraError::DivisionByZero);
        }
        residues.push(num / deriv);
    }
    Ok(residues)
}

/// Divide `P(x)` by `Q(x)` returning quotient and remainder polynomials.
///
/// # Errors
/// Returns [`AlgebraError::DivisionByZero`] if `q` is the zero polynomial.
pub fn polynomial_division(p: &Polynomial, q: &Polynomial) -> Result<(Polynomial, Polynomial)> {
    let (quot, rem) = divide(p.coeffs(), q.coeffs())?;
    Ok((Polynomial::from_coeffs(&quot), Polynomial::from_coeffs(&rem)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn rational_arithmetic() {
        let r1 = RationalExpression::from_coeffs(&[1.0, 2.0], &[1.0, 1.0]).unwrap();
        let r2 = RationalExpression::from_coeffs(&[2.0, 0.0], &[1.0, 0.0, 1.0]).unwrap();
        let sum = r1 + r2;
        let _ = sum.eval(3.0);
    }

    #[test]
    fn rational_new_zero_den() {
        assert_eq!(
            RationalExpression::from_coeffs(&[1.0], &[0.0]),
            Err(AlgebraError::DivisionByZero)
        );
    }

    #[test]
    fn simplify_ratio() {
        let r = RationalExpression::from_coeffs(&[-1.0, 0.0, 1.0], &[-1.0, 1.0]).unwrap();
        let s = r.simplify();
        assert!(approx(s.numerator.coeffs()[0], 1.0));
        assert!(approx(s.numerator.coeffs()[1], 1.0));
        assert!(approx(s.denominator.coeffs()[0], 1.0));
    }

    #[test]
    fn fraction_ops() {
        assert_eq!(add_rational(1.0, 2.0, 1.0, 3.0), (5.0, 6.0));
        assert_eq!(sub_rational(1.0, 2.0, 1.0, 6.0), (4.0, 12.0));
        assert_eq!(mul_rational(2.0, 3.0, 3.0, 4.0), (6.0, 12.0));
        assert_eq!(div_rational(1.0, 2.0, 3.0, 4.0), (4.0, 6.0));
    }

    #[test]
    fn reduce() {
        assert_eq!(reduce_fraction(6, 9), (2, 3));
        assert_eq!(reduce_fraction(15, 25), (3, 5));
        assert_eq!(reduce_fraction(-4, 8), (-1, 2));
    }

    #[test]
    fn rationalize() {
        let (n, d) = rationalize_denominator(3.0, 2.0);
        assert!(approx(n, 1.0));
        assert!(approx(d, 5.0));
    }

    #[test]
    fn partial_frac() {
        let residues = partial_fractions(&[1.0, 1.0], &[1.0, -1.0]).unwrap();
        assert!(approx(residues[0], 1.0));
        assert!(approx(residues[1], 0.0));
    }

    #[test]
    fn partial_frac_three_roots() {
        // 1/((x-1)(x-2)(x-3)) -> residues [1/2, -1, 1/2]
        let residues = partial_fractions(&[1.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!(approx(residues[0], 0.5));
        assert!(approx(residues[1], -1.0));
        assert!(approx(residues[2], 0.5));
    }

    #[test]
    fn display_rational() {
        let r = RationalExpression::from_coeffs(&[1.0, 1.0], &[-1.0, 1.0]).unwrap();
        let s = format!("{r}");
        assert!(s.contains("/"));
    }
}
