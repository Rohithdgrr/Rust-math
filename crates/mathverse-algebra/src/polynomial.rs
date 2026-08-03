//! # Polynomial
//!
//! The core type for this crate: a polynomial with real (`f64`) coefficients,
//! stored in **lowest-degree-first** order.
//!
//! ## Representation
//!
//! ```text
//! coeffs = [a₀, a₁, a₂, ..., aₙ]
//! p(x)  = a₀ + a₁x + a₂x² + ... + aₙxⁿ
//! ```
//!
//! Leading zeros are automatically stripped, so the zero polynomial is `[0.0]`.
//!
//! ## Examples
//!
//! ```rust
//! use mathverse_algebra::Polynomial;
//!
//! // x^2 - 5x + 6
//! let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]);
//! assert_eq!(p.degree(), 2);
//! assert_eq!(p.eval(2.0), 0.0);
//!
//! // Scalar operations
//! let q = p.clone() + 1.0;      // x^2 - 5x + 7
//! let r = p.clone() * 2.0;      // 2x^2 - 10x + 12
//!
//! // Display
//! assert_eq!(format!("{p}"), "x^2 - 5x + 6");
//! ```

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::TOL;

/// Polynomial with real coefficients, stored **lowest degree first**:
/// `coeffs[i]` is the coefficient of `x^i`. Leading zeros are stripped.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::Polynomial;
///
/// let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]); // x^2 - 5x + 6
/// assert_eq!(p.degree(), 2);
/// assert_eq!(p.eval(3.0), 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct Polynomial {
    pub(crate) coeffs: Vec<f64>,
}

impl Polynomial {
    /// Build from coefficients, lowest degree first.
    ///
    /// Leading zeros are stripped. The zero polynomial is `[0.0]`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::Polynomial;
    ///
    /// let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]);
    /// assert_eq!(p.degree(), 2);
    /// assert_eq!(p.eval(1.0), 2.0);
    /// ```
    #[inline]
    pub fn from_coeffs(coeffs: &[f64]) -> Self {
        let mut c = coeffs.to_vec();
        while c.len() > 1 && c.last().map_or(false, |x| x.abs() < TOL) {
            c.pop();
        }
        if c.is_empty() {
            c.push(0.0);
        }
        Polynomial { coeffs: c }
    }

    /// Constant polynomial `c`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::Polynomial;
    ///
    /// let p = Polynomial::constant(5.0);
    /// assert_eq!(p.degree(), 0);
    /// assert_eq!(p.eval(100.0), 5.0);
    /// ```
    #[inline]
    pub fn constant(c: f64) -> Self {
        Self::from_coeffs(&[c])
    }

    /// The polynomial `x`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::Polynomial;
    ///
    /// let x = Polynomial::x();
    /// assert_eq!(x.eval(3.0), 3.0);
    /// ```
    #[inline]
    pub fn x() -> Self {
        Polynomial { coeffs: vec![0.0, 1.0] }
    }

    /// Degree of the polynomial (0 for constants, including zero).
    #[inline]
    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    /// Coefficients, lowest degree first.
    #[inline]
    pub fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    /// True if this is the zero polynomial.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::Polynomial;
    ///
    /// assert!(Polynomial::constant(0.0).is_zero());
    /// assert!(!Polynomial::constant(1.0).is_zero());
    /// ```
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.abs() < TOL)
    }

    /// Evaluate at `x` using Horner's method (O(degree)).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::Polynomial;
    ///
    /// let p = Polynomial::from_coeffs(&[1.0, -2.0, 1.0]); // (x-1)^2
    /// assert_eq!(p.eval(1.0), 0.0);
    /// assert_eq!(p.eval(3.0), 4.0);
    /// ```
    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        self.coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
    }

    /// Derivative polynomial.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::Polynomial;
    ///
    /// // d/dx (x^3 - 2x + 1) = 3x^2 - 2
    /// let p = Polynomial::from_coeffs(&[1.0, -2.0, 0.0, 1.0]);
    /// assert_eq!(p.derivative().coeffs(), &[-2.0, 0.0, 3.0]);
    /// ```
    #[must_use]
    pub fn derivative(&self) -> Polynomial {
        let d: Vec<f64> = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| c * i as f64)
            .collect();
        Polynomial::from_coeffs(&d)
    }

    /// Indefinite integral with zero constant term.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::Polynomial;
    ///
    /// let p = Polynomial::from_coeffs(&[-2.0, 2.0]); // 2x - 2
    /// let i = p.integral(); // x^2 - 2x
    /// assert_eq!(i.coeffs(), &[0.0, -2.0, 1.0]);
    /// ```
    #[must_use]
    pub fn integral(&self) -> Polynomial {
        let mut c: Vec<f64> = vec![0.0];
        for (i, &coef) in self.coeffs.iter().enumerate() {
            c.push(coef / (i + 1) as f64);
        }
        Polynomial { coeffs: c }
    }

    /// Real roots via closed-form solvers (degree ≤ 4).
    ///
    /// Returns an empty `Vec` for degree > 4 or constants.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::Polynomial;
    ///
    /// let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]); // (x-2)(x-3)
    /// let mut roots = p.roots();
    /// roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    /// assert_eq!(roots, vec![2.0, 3.0]);
    /// ```
    #[must_use]
    pub fn roots(&self) -> Vec<f64> {
        crate::roots::solve(&self.coeffs)
    }

    /// Leading coefficient.
    #[inline]
    #[must_use]
    pub fn leading(&self) -> f64 {
        self.coeffs[self.degree()]
    }

    /// Approximate equality with tolerance `tol`.
    ///
    /// Two polynomials are equal if they have the same degree and all
    /// coefficients differ by less than `tol`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use mathverse_algebra::Polynomial;
    ///
    /// let a = Polynomial::from_coeffs(&[1.0, 2.0]);
    /// let b = Polynomial::from_coeffs(&[1.0, 2.0 + 1e-15]);
    /// assert!(a.approx_eq(&b, 1e-12));
    /// ```
    #[must_use]
    pub fn approx_eq(&self, other: &Polynomial, tol: f64) -> bool {
        if self.degree() != other.degree() {
            return false;
        }
        self.coeffs.iter().zip(other.coeffs.iter()).all(|(a, b)| (a - b).abs() < tol)
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        self.approx_eq(other, TOL)
    }
}

impl Eq for Polynomial {}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
        for (i, &c) in self.coeffs.iter().enumerate().rev() {
            if c.abs() < TOL {
                continue;
            }
            let sign = if c < 0.0 { "-" } else { "+" };
            if first {
                if c < 0.0 {
                    write!(f, "-")?;
                }
            } else {
                write!(f, " {sign} ")?;
            }
            first = false;
            let abs_c = c.abs();
            match i {
                0 => write!(f, "{abs_c}")?,
                1 => {
                    if (abs_c - 1.0).abs() > TOL {
                        write!(f, "{abs_c}")?;
                    }
                    write!(f, "x")?;
                }
                n => {
                    if (abs_c - 1.0).abs() > TOL {
                        write!(f, "{abs_c}")?;
                    }
                    write!(f, "x^{n}")?;
                }
            }
        }
        Ok(())
    }
}

impl From<f64> for Polynomial {
    fn from(c: f64) -> Self {
        Self::constant(c)
    }
}

impl Neg for Polynomial {
    type Output = Polynomial;
    #[inline]
    fn neg(mut self) -> Polynomial {
        for c in &mut self.coeffs {
            *c = -*c;
        }
        self
    }
}

impl Add for Polynomial {
    type Output = Polynomial;
    fn add(self, other: Polynomial) -> Polynomial {
        let (mut long, short) = if self.coeffs.len() >= other.coeffs.len() {
            (self.coeffs, &other.coeffs)
        } else {
            (other.coeffs, &self.coeffs)
        };
        for (l, &s) in long.iter_mut().zip(short.iter()) {
            *l += s;
        }
        Polynomial::from_coeffs(&long)
    }
}

impl AddAssign for Polynomial {
    fn add_assign(&mut self, other: Polynomial) {
        *self = self.clone() + other;
    }
}

impl Sub for Polynomial {
    type Output = Polynomial;
    #[inline]
    fn sub(self, other: Polynomial) -> Polynomial {
        self + (-other)
    }
}

impl SubAssign for Polynomial {
    fn sub_assign(&mut self, other: Polynomial) {
        *self = self.clone() - other;
    }
}

impl Mul for Polynomial {
    type Output = Polynomial;
    fn mul(self, other: Polynomial) -> Polynomial {
        let mut c = vec![0.0; self.coeffs.len() + other.coeffs.len() - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                c[i + j] += a * b;
            }
        }
        Polynomial::from_coeffs(&c)
    }
}

impl MulAssign<f64> for Polynomial {
    fn mul_assign(&mut self, k: f64) {
        for c in &mut self.coeffs {
            *c *= k;
        }
    }
}

impl Mul<f64> for Polynomial {
    type Output = Polynomial;
    #[inline]
    fn mul(self, k: f64) -> Polynomial {
        Polynomial::from_coeffs(&self.coeffs.iter().map(|&c| c * k).collect::<Vec<f64>>())
    }
}

impl Mul<Polynomial> for f64 {
    type Output = Polynomial;
    #[inline]
    fn mul(self, p: Polynomial) -> Polynomial {
        p * self
    }
}

impl Add<f64> for Polynomial {
    type Output = Polynomial;
    #[inline]
    fn add(self, c: f64) -> Polynomial {
        let mut result = self;
        if !result.coeffs.is_empty() {
            result.coeffs[0] += c;
        } else {
            result.coeffs.push(c);
        }
        result
    }
}

impl Sub<f64> for Polynomial {
    type Output = Polynomial;
    #[inline]
    fn sub(self, c: f64) -> Polynomial {
        self + (-c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_and_derivative() {
        let p = Polynomial::from_coeffs(&[1.0, -2.0, 1.0]); // (x-1)^2
        assert_eq!(p.eval(1.0), 0.0);
        assert_eq!(p.eval(3.0), 4.0);
        assert_eq!(p.derivative().coeffs(), &[-2.0, 2.0]);
        assert_eq!(p.integral().coeffs(), &[0.0, 1.0, -1.0, 1.0 / 3.0]);
        assert_eq!(Polynomial::from_coeffs(&[0.0]).degree(), 0);
        assert_eq!(Polynomial::from_coeffs(&[1.0, 0.0, 0.0]).degree(), 0);
    }

    #[test]
    fn polynomial_arithmetic() {
        let a = Polynomial::from_coeffs(&[1.0, 2.0]); // 2x + 1
        let b = Polynomial::from_coeffs(&[3.0, 4.0]); // 4x + 3
        assert_eq!((a.clone() + b.clone()).coeffs(), &[4.0, 6.0]);
        assert_eq!((b.clone() - a.clone()).coeffs(), &[2.0, 2.0]);
        assert_eq!((a.clone() * b.clone()).coeffs(), &[3.0, 10.0, 8.0]);
        assert_eq!((a * 2.0).coeffs(), &[2.0, 4.0]);
        assert_eq!((-Polynomial::from_coeffs(&[1.0, 2.0])).coeffs(), &[-1.0, -2.0]);
    }

    #[test]
    fn scalar_ops() {
        let p = Polynomial::from_coeffs(&[1.0, 2.0]);
        assert_eq!((p.clone() + 3.0).coeffs(), &[4.0, 2.0]);
        assert_eq!((p.clone() - 1.0).coeffs(), &[0.0, 2.0]);
        assert_eq!((3.0 * p.clone()).coeffs(), &[3.0, 6.0]);
    }

    #[test]
    fn from_f64() {
        let p: Polynomial = 5.0.into();
        assert_eq!(p.coeffs(), &[5.0]);
    }

    #[test]
    fn display() {
        let p = Polynomial::from_coeffs(&[1.0, -2.0, 1.0]);
        assert_eq!(format!("{p}"), "x^2 - 2x + 1");
        assert_eq!(format!("{}", Polynomial::constant(0.0)), "0");
    }

    #[test]
    fn approx_eq_test() {
        let a = Polynomial::from_coeffs(&[1.0, 2.0]);
        let b = Polynomial::from_coeffs(&[1.0, 2.0 + 1e-15]);
        assert!(a.approx_eq(&b, 1e-12));
    }
}
