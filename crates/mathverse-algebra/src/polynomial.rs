//! `Polynomial` type: real-coefficient polynomials, lowest-degree-first.

use core::ops::{Add, Mul, Neg, Sub};

/// Polynomial with real coefficients, stored **lowest degree first**:
/// `coeffs[i]` is the coefficient of `x^i`. Leading zeros are stripped.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    pub(crate) coeffs: Vec<f64>,
}

impl Polynomial {
    /// Build from coefficients, lowest degree first.
    ///
    /// ```
    /// # use mathverse_algebra::Polynomial;
    /// // x^2 - 5x + 6
    /// let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]);
    /// assert_eq!(p.degree(), 2);
    /// assert_eq!(p.eval(1.0), 2.0);
    /// ```
    pub fn from_coeffs(coeffs: &[f64]) -> Self {
        let mut c = coeffs.to_vec();
        while c.len() > 1 && *c.last().unwrap() == 0.0 {
            c.pop();
        }
        if c.is_empty() {
            c.push(0.0);
        }
        Polynomial { coeffs: c }
    }

    /// Constant polynomial `c`.
    pub fn constant(c: f64) -> Self {
        Self::from_coeffs(&[c])
    }

    /// The polynomial `x`.
    pub fn x() -> Self {
        Polynomial { coeffs: vec![0.0, 1.0] }
    }

    /// `degree` of the polynomial (0 for constants, including zero).
    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    /// Coefficients, lowest degree first.
    pub fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    /// Evaluate at `x` (Horner's method, O(degree)).
    pub fn eval(&self, x: f64) -> f64 {
        self.coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
    }

    /// Derivative polynomial.
    ///
    /// ```
    /// # use mathverse_algebra::Polynomial;
    /// // d/dx (x^3 - 2x + 1) = 3x^2 - 2
    /// let p = Polynomial::from_coeffs(&[1.0, -2.0, 0.0, 1.0]);
    /// assert_eq!(p.derivative().coeffs(), &[-2.0, 0.0, 3.0]);
    /// ```
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
    pub fn integral(&self) -> Polynomial {
        let mut c: Vec<f64> = vec![0.0];
        for (i, &coef) in self.coeffs.iter().enumerate() {
            c.push(coef / (i + 1) as f64);
        }
        Polynomial { coeffs: c }
    }

    /// Real roots, numerically stable.
    /// Degree > 3 returns `[]` (use [`solve_quartic`](crate::roots::solve_quartic)).
    pub fn roots(&self) -> Vec<f64> {
        crate::roots::solve(&self.coeffs)
    }

    /// Leading coefficient.
    pub fn leading(&self) -> f64 {
        self.coeffs[self.degree()]
    }
}

impl Neg for Polynomial {
    type Output = Polynomial;
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

impl Sub for Polynomial {
    type Output = Polynomial;
    fn sub(self, other: Polynomial) -> Polynomial {
        self + (-other)
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

impl Mul<f64> for Polynomial {
    type Output = Polynomial;
    fn mul(self, k: f64) -> Polynomial {
        Polynomial::from_coeffs(&self.coeffs.iter().map(|&c| c * k).collect::<Vec<_>>())
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
}
