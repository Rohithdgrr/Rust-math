//! # Rational Expressions
//!
//! Symbolic rational functions `p(x)/q(x)` with exact operations, and
//! partial-fraction decomposition.
//!
//! ## Examples
//!
//! ```rust
//! use mathverse_algebra::rational::RationalExpression;
//!
//! let r1 = RationalExpression::new(&[1.0], &[1.0, 1.0]);   // 1/(x+1)
//! let r2 = RationalExpression::new(&[1.0], &[1.0, -1.0]);  // 1/(x-1)
//! let sum = r1 + r2; // (2x)/((x+1)(x-1))
//! ```

use crate::polynomial::Polynomial;
use crate::{AlgebraError, TOL};
use core::fmt;

/// A rational expression `num / den` where `num` and `den` are [`Polynomial`]s.
///
/// Operations automatically reduce by the polynomial GCD. Division by zero
/// returns an error.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::rational::RationalExpression;
///
/// let r = RationalExpression::new(&[1.0], &[1.0, 1.0]); // 1/(x+1)
/// assert_eq!(r.eval(1.0), 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct RationalExpression {
    pub(crate) num: Polynomial,
    pub(crate) den: Polynomial,
}

impl RationalExpression {
    /// Create from numerator and denominator coefficient slices (lowest-degree first).
    ///
    /// Automatically reduces by GCD.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::rational::RationalExpression;
    ///
    /// let r = RationalExpression::new(&[2.0, 2.0], &[1.0, 1.0]);
    /// assert_eq!(r.eval(0.0), 2.0); // (2x+2)/(x+1) = 2
    /// ```
    pub fn new(num: &[f64], den: &[f64]) -> Self {
        let mut r = RationalExpression {
            num: Polynomial::from_coeffs(num),
            den: Polynomial::from_coeffs(den),
        };
        r.reduce();
        r
    }

    /// Reduce numerator and denominator by their GCD.
    pub(crate) fn reduce(&mut self) {
        if self.num.is_zero() {
            self.den = Polynomial::constant(1.0);
            return;
        }
        let g = crate::factor::polynomial_gcd(self.num.coeffs(), self.den.coeffs());
        if g.is_zero() || g.degree() == 0 {
            return;
        }
        let (q_num, _) = crate::factor::divide(self.num.coeffs(), g.coeffs()).unwrap_or_else(|_| (self.num.coeffs().to_vec(), Polynomial::constant(0.0)));
        let (q_den, _) = crate::factor::divide(self.den.coeffs(), g.coeffs()).unwrap_or_else(|_| (self.den.coeffs().to_vec(), Polynomial::constant(0.0)));
        self.num = Polynomial::from_coeffs(&q_num);
        self.den = Polynomial::from_coeffs(&q_den);
    }

    /// Evaluate at `x` if `den(x) != 0`.
    ///
    /// Returns `None` if `den(x) ≈ 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::rational::RationalExpression;
    ///
    /// let r = RationalExpression::new(&[1.0], &[-1.0, 1.0]); // 1/(x-1)
    /// assert_eq!(r.try_eval(2.0), Some(1.0));
    /// assert_eq!(r.try_eval(1.0), None);
    /// ```
    #[must_use]
    pub fn try_eval(&self, x: f64) -> Option<f64> {
        let d = self.den.eval(x);
        if d.abs() < TOL {
            None
        } else {
            Some(self.num.eval(x) / d)
        }
    }

    /// Evaluate at `x`, returning zero if `den(x) ≈ 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::rational::RationalExpression;
    ///
    /// let r = RationalExpression::new(&[1.0], &[-1.0, 1.0]);
    /// assert_eq!(r.eval(2.0), 1.0);
    /// assert_eq!(r.eval(1.0), 0.0); // pole → 0
    /// ```
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        self.try_eval(x).unwrap_or(0.0)
    }

    /// Degree of the numerator.
    #[inline]
    pub fn num_degree(&self) -> usize {
        self.num.degree()
    }

    /// Degree of the denominator.
    #[inline]
    pub fn den_degree(&self) -> usize {
        self.den.degree()
    }

    /// True if this is a polynomial (denominator is a nonzero constant).
    #[inline]
    pub fn is_polynomial(&self) -> bool {
        self.den.degree() == 0
    }

    /// True if this is a proper rational function (num_degree < den_degree).
    #[inline]
    pub fn is_proper(&self) -> bool {
        self.num_degree() < self.den_degree()
    }

    /// Partial-fraction decomposition: `p(x) / ((x-a)(x-b)) = r/(x-a) + s/(x-b)`.
    ///
    /// Requires the denominator to have exactly 2 distinct real linear factors.
    ///
    /// # Errors
    ///
    /// Returns [`AlgebraError::UnsupportedDegree`] if `den` doesn't have exactly
    /// 2 distinct real roots.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::rational::RationalExpression;
    ///
    /// // 1/((x-1)(x-2)) = -1/(x-1) + 1/(x-2)
    /// let r = RationalExpression::new(&[1.0], &[-6.0, 5.0, -1.0]);
    /// let (a, residues) = r.partial_frac().unwrap();
    /// // residues[0] corresponds to factor (x - a[0])
    /// ```
    pub fn partial_frac(&self) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
        let d = self.den.coeffs();
        if d.len() != 3 {
            return Err(AlgebraError::UnsupportedDegree(d.len() as u8 - 1));
        }
        let roots = crate::roots::solve_quadratic(d[2], d[1], d[0]);
        if roots.len() != 2 {
            return Err(AlgebraError::UnsupportedDegree(1));
        }
        let a = roots[0];
        let b = roots[1];
        let n = &self.num;
        let res_a = n.eval(a) / ((a - b) * d[2]);
        let res_b = n.eval(b) / ((b - a) * d[2]);
        Ok((vec![a, b], vec![res_a, res_b]))
    }

    /// Three-root partial-fraction decomposition: `p(x) / ((x-a)(x-b)(x-c))`.
    ///
    /// # Errors
    ///
    /// Returns [`AlgebraError::UnsupportedDegree`] if `den` doesn't have exactly
    /// 3 distinct real roots.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::rational::RationalExpression;
    ///
    /// // 1/((x-1)(x-2)(x-3))
    /// let r = RationalExpression::new(&[1.0], &[-6.0, 11.0, -6.0, 1.0]);
    /// let (roots, residues) = r.partial_frac_three().unwrap();
    /// assert_eq!(roots.len(), 3);
    /// ```
    pub fn partial_frac_three(&self) -> Result<(Vec<f64>, Vec<f64>), AlgebraError> {
        let d = self.den.coeffs();
        if d.len() != 4 {
            return Err(AlgebraError::UnsupportedDegree(d.len() as u8 - 1));
        }
        let roots = crate::roots::solve_cubic(d[3], d[2], d[1], d[0]);
        if roots.len() != 3 {
            return Err(AlgebraError::UnsupportedDegree(2));
        }
        let a = roots[0];
        let b = roots[1];
        let c = roots[2];
        let n = &self.num;
        let res_a = n.eval(a) / ((a - b) * (a - c) * d[3]);
        let res_b = n.eval(b) / ((b - a) * (b - c) * d[3]);
        let res_c = n.eval(c) / ((c - a) * (c - b) * d[3]);
        Ok((vec![a, b, c], vec![res_a, res_b, res_c]))
    }
}

impl PartialEq for RationalExpression {
    fn eq(&self, other: &Self) -> bool {
        if self.num.is_zero() && other.num.is_zero() {
            return true;
        }
        if self.num.is_zero() != other.num.is_zero() {
            return false;
        }
        let self_n = self.num.coeffs();
        let self_d = self.den.coeffs();
        let other_n = other.num.coeffs();
        let other_d = other.den.coeffs();

        let left_len = self_n.len() + other_d.len();
        let right_len = other_n.len() + self_d.len();
        if left_len != right_len {
            return false;
        }
        let mut left = vec![0.0; left_len];
        for (i, &a) in self_n.iter().enumerate() {
            for (j, &b) in other_d.iter().enumerate() {
                left[i + j] += a * b;
            }
        }
        let mut right = vec![0.0; right_len];
        for (i, &a) in other_n.iter().enumerate() {
            for (j, &b) in self_d.iter().enumerate() {
                right[i + j] += a * b;
            }
        }
        left.iter().zip(right.iter()).all(|(x, y)| (x - y).abs() < TOL)
    }
}

impl Eq for RationalExpression {}

impl fmt::Display for RationalExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den.degree() == 0 && (self.den.coeffs()[0] - 1.0).abs() < TOL {
            write!(f, "{}", self.num)
        } else {
            write!(f, "({}) / ({})", self.num, self.den)
        }
    }
}

// Arithmetic

impl core::ops::Add for RationalExpression {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let new_num = Polynomial::from_coeffs(
            &(self.num * rhs.den.clone() + self.den.clone() * rhs.num).coeffs().to_vec(),
        );
        let new_den = self.den * rhs.den;
        Self::new(new_num.coeffs(), new_den.coeffs())
    }
}

impl core::ops::Sub for RationalExpression {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let new_num = Polynomial::from_coeffs(
            &(self.num * rhs.den.clone() - self.den.clone() * rhs.num).coeffs().to_vec(),
        );
        let new_den = self.den * rhs.den;
        Self::new(new_num.coeffs(), new_den.coeffs())
    }
}

impl core::ops::Mul for RationalExpression {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            &(self.num * rhs.num).coeffs(),
            &(self.den * rhs.den).coeffs(),
        )
    }
}

impl core::ops::Div for RationalExpression {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::new(
            &(self.num * rhs.den).coeffs(),
            &(self.den * rhs.num).coeffs(),
        )
    }
}

impl core::ops::Neg for RationalExpression {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(
            &(-self.num).coeffs(),
            self.den.coeffs(),
        )
    }
}

/// Create a unit fraction `1 / (x - c)`.
#[must_use]
pub fn unit_fraction(c: f64) -> RationalExpression {
    RationalExpression::new(&[1.0], &[-c, 1.0])
}

/// Create a polynomial as a rational function (denominator = 1).
#[must_use]
pub fn poly_to_rational(p: &Polynomial) -> RationalExpression {
    RationalExpression::new(p.coeffs(), &[1.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_cross_mul() {
        let r1 = RationalExpression::new(&[1.0, 1.0], &[1.0, 2.0]);
        let r2 = RationalExpression::new(&[2.0, 2.0], &[2.0, 4.0]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn partial_frac_test() {
        let r = RationalExpression::new(&[1.0], &[-6.0, 5.0, -1.0]);
        let (roots, residues) = r.partial_frac().unwrap();
        assert_eq!(roots.len(), 2);
        assert_eq!(residues.len(), 2);
    }

    #[test]
    fn partial_frac_three_test() {
        let r = RationalExpression::new(&[1.0], &[-6.0, 11.0, -6.0, 1.0]);
        let (roots, residues) = r.partial_frac_three().unwrap();
        assert_eq!(roots.len(), 3);
        assert_eq!(residues.len(), 3);
    }

    #[test]
    fn display_test() {
        let r = RationalExpression::new(&[1.0], &[1.0, 1.0]);
        assert_eq!(format!("{r}"), "(1) / (x + 1)");
    }
}
