//! Rational expressions: `P(x)/Q(x)` as a pair of [`Polynomial`]s with
//! arithmetic, simplification, partial-fraction decomposition, and
//! denominator rationalization.

use crate::polynomial::Polynomial;
use crate::factor::{divide, polynomial_gcd};

const TOL: f64 = 1e-12;

/// A rational expression `numerator / denominator`.
#[derive(Debug, Clone, PartialEq)]
pub struct RationalExpression {
    pub numerator: Polynomial,
    pub denominator: Polynomial,
}

impl RationalExpression {
    /// Create from numerator and denominator polynomials.
    /// Panics if the denominator is zero.
    pub fn new(num: Polynomial, den: Polynomial) -> Self {
        assert!(
            !den.coeffs().iter().all(|c| c.abs() < TOL),
            "denominator must not be zero"
        );
        RationalExpression { numerator: num, denominator: den }
    }

    /// Create from coefficient slices (lowest-degree first).
    pub fn from_coeffs(num: &[f64], den: &[f64]) -> Self {
        Self::new(
            Polynomial::from_coeffs(num),
            Polynomial::from_coeffs(den),
        )
    }

    /// Simplify by dividing numerator and denominator by their polynomial GCD.
    ///
    /// ```
    /// # use mathverse_algebra::rational::RationalExpression;
    /// let r = RationalExpression::from_coeffs(&[-1.0, 0.0, 1.0], &[-1.0, 1.0]); // (xÂ²-1)/(x-1)
    /// let s = r.simplify();
    /// assert!((s.numerator.coeffs()[0] - 1.0).abs() < 1e-12);
    /// assert!((s.numerator.coeffs()[1] - 1.0).abs() < 1e-12);
    /// ```
    pub fn simplify(&self) -> RationalExpression {
        let g = polynomial_gcd(&self.numerator, &self.denominator);
        let (nq, _) = divide(self.numerator.coeffs(), g.coeffs());
        let (dq, _) = divide(self.denominator.coeffs(), g.coeffs());
        RationalExpression::from_coeffs(&nq, &dq)
    }

    /// Evaluate at `x`.
    pub fn eval(&self, x: f64) -> f64 {
        self.numerator.eval(x) / self.denominator.eval(x)
    }
}

impl core::ops::Add for RationalExpression {
    type Output = RationalExpression;
    fn add(self, other: RationalExpression) -> RationalExpression {
        let num = self.numerator.clone() * other.denominator.clone()
            + other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();
        RationalExpression::new(num, den).simplify()
    }
}

impl core::ops::Sub for RationalExpression {
    type Output = RationalExpression;
    fn sub(self, other: RationalExpression) -> RationalExpression {
        let num = self.numerator.clone() * other.denominator.clone()
            - other.numerator.clone() * self.denominator.clone();
        let den = self.denominator.clone() * other.denominator.clone();
        RationalExpression::new(num, den).simplify()
    }
}

impl core::ops::Mul for RationalExpression {
    type Output = RationalExpression;
    fn mul(self, other: RationalExpression) -> RationalExpression {
        RationalExpression::new(
            self.numerator.clone() * other.numerator.clone(),
            self.denominator.clone() * other.denominator.clone(),
        )
        .simplify()
    }
}

impl core::ops::Div for RationalExpression {
    type Output = RationalExpression;
    fn div(self, other: RationalExpression) -> RationalExpression {
        RationalExpression::new(
            self.numerator.clone() * other.denominator.clone(),
            self.denominator.clone() * other.numerator.clone(),
        )
        .simplify()
    }
}

/// `a/b + c/d = (ad + bc)/(bd)`.
pub fn add_rational(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    (a * d + b * c, b * d)
}

/// `a/b - c/d = (ad - bc)/(bd)`.
pub fn sub_rational(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    (a * d - b * c, b * d)
}

/// `(a/b) Â· (c/d) = ac/bd`.
pub fn mul_rational(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    (a * c, b * d)
}

/// `(a/b) Ã· (c/d) = ad/bc`.
pub fn div_rational(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    (a * d, b * c)
}

/// Reduce a fraction `num/den` to lowest terms using integer GCD.
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
/// This is a simplified version for `1/(a + b)` â†’ `(a - b)/(aÂ² - bÂ²)`.
pub fn rationalize_denominator(a: f64, b: f64) -> (f64, f64) {
    let conj = a - b;
    let denom = a * a - b * b;
    (conj, denom)
}

/// Partial fraction decomposition for `P(x) / ((x - râ‚)(x - râ‚‚)â€¦(x - râ‚–))`.
///
/// Returns the residues `[Aâ‚, Aâ‚‚, â€¦, Aâ‚–]` such that:
/// `P(x)/Q(x) = Î£ Aáµ¢/(x - ráµ¢)`.
///
/// Uses the cover-up (Heaviside) method: `Aáµ¢ = P(ráµ¢) / Q'(ráµ¢)`.
///
/// ```
/// # use mathverse_algebra::rational::partial_fractions;
/// // (x+1) / (xÂ² - 1) = (x+1)/((x-1)(x+1)) = 1/(x-1) Â· 1 + 1/(x+1) Â· 1
/// let residues = partial_fractions(&[1.0, 1.0], &[1.0, -1.0]);
/// assert!((residues[0] - 1.0).abs() < 1e-9);
/// ```
pub fn partial_fractions(p_coeffs: &[f64], roots: &[f64]) -> Vec<f64> {
    let p = Polynomial::from_coeffs(p_coeffs);
    let mut residues = Vec::new();
    for &r in roots {
        let num = p.eval(r);
        let mut deriv = 1.0;
        for &other in roots {
            if (other - r).abs() > TOL {
                deriv *= r - other;
            }
        }
        residues.push(num / deriv);
    }
    residues
}

/// Divide `P(x)` by `Q(x)` returning quotient and remainder polynomials.
pub fn polynomial_division(p: &Polynomial, q: &Polynomial) -> (Polynomial, Polynomial) {
    let (quot, rem) = divide(p.coeffs(), q.coeffs());
    (Polynomial::from_coeffs(&quot), Polynomial::from_coeffs(&rem))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn rational_arithmetic() {
        let r1 = RationalExpression::from_coeffs(&[1.0, 2.0], &[1.0, 1.0]); // (2x+1)/(x+1)
        let r2 = RationalExpression::from_coeffs(&[2.0, 0.0], &[1.0, 0.0, 1.0]); // 2/(xÂ²+1)
        let sum = r1 + r2;
        // Just verify it evaluates without panic
        let _ = sum.eval(3.0);
    }

    #[test]
    fn simplify_ratio() {
        let r = RationalExpression::from_coeffs(&[-1.0, 0.0, 1.0], &[-1.0, 1.0]); // (xÂ²-1)/(x-1)
        let s = r.simplify();
        assert!(approx(s.numerator.coeffs()[0], 1.0));
        assert!(approx(s.numerator.coeffs()[1], 1.0));
        assert!(approx(s.denominator.coeffs()[0], 1.0));
    }

    #[test]
    fn fraction_ops() {
        assert_eq!(add_rational(1.0, 2.0, 1.0, 3.0), (5.0, 6.0));
        assert_eq!(sub_rational(1.0, 2.0, 1.0, 6.0), (1.0, 3.0));
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
        assert!(approx(d, 5.0)); // 9 - 4
    }

    #[test]
    fn partial_frac() {
        let residues = partial_fractions(&[1.0, 1.0], &[1.0, -1.0]); // (x+1)/((x-1)(x+1))
        assert!(approx(residues[0], 1.0));
        assert!(approx(residues[1], 1.0));
    }
}