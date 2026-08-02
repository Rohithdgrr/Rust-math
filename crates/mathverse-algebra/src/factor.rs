//! # Factor
//!
//! Polynomial factorization: synthetic division, GCD, remainder/factor theorem,
//! and rational root candidates.
//!
//! All polynomials use **lowest-degree-first** coefficient order.
//!
//! ## Examples
//!
//! ```rust
//! use mathverse_algebra::factor::{synthetic_division, polynomial_gcd};
//!
//! // (x^3 - 6x^2 + 11x - 6) ÷ (x - 1) → quotient (x^2 - 5x + 6), remainder 0
//! let div = synthetic_division(&[-6.0, 11.0, -6.0, 1.0], 1.0);
//! assert_eq!(div.remainder, 0.0);
//! ```

use crate::polynomial::Polynomial;
use crate::roots::rational_root_candidates;
use crate::{AlgebraError, TOL};

/// Synthetic division: divide the polynomial by `(x - c)`.
///
/// Returns `(quotient, remainder)` where the remainder is a scalar.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::factor::synthetic_division;
///
/// // (x^2 - 4) ÷ (x - 2)
/// let (q, r) = synthetic_division(&[-4.0, 0.0, 1.0], 2.0);
/// assert_eq!(q, vec![2.0, 1.0]); // x + 2
/// assert_eq!(r, 0.0);
/// ```
#[must_use]
pub fn synthetic_division(coeffs: &[f64], c: f64) -> (Vec<f64>, f64) {
    if coeffs.is_empty() {
        return (vec![], 0.0);
    }
    let mut result = Vec::with_capacity(coeffs.len());
    let mut rem = 0.0;
    for &coef in coeffs.iter().rev() {
        let cur = rem * c + coef;
        result.push(cur);
        rem = cur;
    }
    result.pop();
    result.reverse();
    (result, rem)
}

/// Divide two polynomials: `p / q = (quotient, remainder)`.
///
/// Each coefficient slice is **lowest-degree first**.
///
/// # Errors
///
/// Returns [`AlgebraError::DivisionByZero`] if `q` is the zero polynomial.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::factor::divide;
///
/// // (x^2 - 4) ÷ (x - 2) = x + 2
/// let p = &[-4.0, 0.0, 1.0];
/// let q = &[-2.0, 1.0];
/// let (qo, r) = divide(p, q).unwrap();
/// assert_eq!(qo, vec![2.0, 1.0]);
/// assert_eq!(r.coeffs(), &[0.0]);
/// ```
pub fn divide(p: &[f64], q: &[f64]) -> Result<(Vec<f64>, Polynomial), AlgebraError> {
    if q.iter().all(|&x| x.abs() < TOL) {
        return Err(AlgebraError::DivisionByZero);
    }
    if p.len() < q.len() {
        return Ok((vec![0.0], Polynomial::from_coeffs(p)));
    }
    let mut rem: Vec<f64> = p.to_vec();
    let mut quot = vec![0.0; p.len() - q.len() + 1];
    let lc = q.last().unwrap();
    for i in (0..p.len() - q.len() + 1).rev() {
        let r_idx = i + q.len() - 1;
        if r_idx < rem.len() {
            quot[i] = rem[r_idx] / *lc;
            for j in 0..q.len() {
                if i + j < rem.len() {
                    rem[i + j] -= quot[i] * q[j];
                }
            }
        }
    }
    let rem_poly = Polynomial::from_coeffs(&rem[..q.len().saturating_sub(1)]);
    Ok((quot, rem_poly))
}

/// Polynomial GCD via the Euclidean algorithm.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::factor::polynomial_gcd;
///
/// // gcd(x^2 - 1, x - 1) = x - 1
/// let a = &[-1.0, 0.0, 1.0]; // x^2 - 1
/// let b = &[-1.0, 1.0];      // x - 1
/// let g = polynomial_gcd(a, b);
/// assert_eq!(g.coeffs(), &[-1.0, 1.0]);
/// ```
#[must_use]
pub fn polynomial_gcd(p: &[f64], q: &[f64]) -> Polynomial {
    let mut a = Polynomial::from_coeffs(p);
    let mut b = Polynomial::from_coeffs(q);
    while !b.is_zero() {
        let (_, r) = divide(a.coeffs(), b.coeffs()).unwrap();
        a = b;
        b = r;
    }
    if a.is_zero() {
        a
    } else {
        a
    }
}

/// Remainder theorem: evaluate `p(c)` by synthetic division.
///
/// `p(x) = (x - c) * q(x) + p(c)`
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::factor::remainder_theorem;
///
/// let p = [-6.0, 11.0, -6.0, 1.0]; // (x-1)(x-2)(x-3)
/// assert_eq!(remainder_theorem(&p, 2.0), 0.0);
/// assert_eq!(remainder_theorem(&p, 4.0), 6.0);
/// ```
#[inline]
#[must_use]
pub fn remainder_theorem(coeffs: &[f64], c: f64) -> f64 {
    synthetic_division(coeffs, c).1
}

/// Factor theorem: `c` is a root of `p` iff `p(c) == 0`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::factor::factor_theorem;
///
/// let p = [-6.0, 11.0, -6.0, 1.0]; // (x-1)(x-2)(x-3)
/// assert!(factor_theorem(&p, 1.0));
/// assert!(!factor_theorem(&p, 4.0));
/// ```
#[inline]
#[must_use]
pub fn factor_theorem(coeffs: &[f64], c: f64) -> bool {
    remainder_theorem(coeffs, c).abs() < TOL
}

/// Full factorization: extract all rational roots and return `(linear_factors, residual)`.
///
/// The `linear_factors` are `(root, multiplicity)` pairs. The `residual` polynomial
/// has no rational roots (degree may still be > 0).
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::factor::factor;
///
/// // x^2 - 5x + 6 = (x-2)(x-3)
/// let (factors, residual) = factor(&[6.0, -5.0, 1.0]);
/// assert_eq!(factors.len(), 2);
/// assert!(residual.is_zero());
/// ```
#[must_use]
pub fn factor(coeffs: &[f64]) -> (Vec<(f64, usize)>, Polynomial) {
    let p = Polynomial::from_coeffs(coeffs);
    if p.is_zero() {
        return (vec![], p);
    }
    let mut factors = Vec::new();
    let mut current = p;
    let a = current.leading();
    let candidates = rational_root_candidates(a, *current.coeffs().first().unwrap());
    for &c in &candidates {
        if current.degree() == 0 {
            break;
        }
        let mut mult = 0;
        while current.degree() > 0 {
            let (_, rem) = synthetic_division(current.coeffs(), c);
            if rem.abs() > TOL {
                break;
            }
            mult += 1;
            let (q, _) = synthetic_division(current.coeffs(), c);
            current = Polynomial::from_coeffs(&q);
        }
        if mult > 0 {
            factors.push((c, mult));
        }
    }
    (factors, current)
}

/// Long division with degree-zero divisor.
///
/// If the divisor is a nonzero constant, divides all coefficients directly.
/// If the divisor is zero, returns [`AlgebraError::DivisionByZero`].
///
/// # Errors
///
/// Returns [`AlgebraError::DivisionByZero`] if `d` is zero.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::factor::divide_by_scalar;
///
/// let p = [6.0, -5.0, 1.0]; // x^2 - 5x + 6
/// let q = divide_by_scalar(&p, 2.0).unwrap();
/// assert_eq!(q, vec![3.0, -2.5, 0.5]);
/// ```
pub fn divide_by_scalar(coeffs: &[f64], d: f64) -> Result<Vec<f64>, AlgebraError> {
    if d.abs() < TOL {
        return Err(AlgebraError::DivisionByZero);
    }
    Ok(coeffs.iter().map(|&c| c / d).collect())
}

/// Number of rational roots of the polynomial.
///
/// Uses the rational root theorem to check all candidates.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::factor::rational_root_count;
///
/// // x^2 - 5x + 6 = (x-2)(x-3): 2 rational roots
/// assert_eq!(rational_root_count(&[6.0, -5.0, 1.0]), 2);
/// ```
#[must_use]
pub fn rational_root_count(coeffs: &[f64]) -> usize {
    let p = Polynomial::from_coeffs(coeffs);
    if p.is_zero() {
        return 0;
    }
    let a = p.leading();
    let b = coeffs[0];
    let candidates = rational_root_candidates(a, b);
    candidates
        .into_iter()
        .filter(|&c| {
            let (_, rem) = synthetic_division(p.coeffs(), c);
            rem.abs() < TOL
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn poly(c: &[f64]) -> Polynomial {
        Polynomial::from_coeffs(c)
    }

    #[test]
    fn division() {
        let (q, r) = synthetic_division(&[-6.0, 11.0, -6.0, 1.0], 1.0);
        assert_eq!(poly(&q).degree(), 2);
        assert!(r.abs() < 1e-10);
    }

    #[test]
    fn remainder_theorem_test() {
        let p = [-6.0, 11.0, -6.0, 1.0];
        assert_eq!(remainder_theorem(&p, 1.0), 0.0);
        assert_eq!(remainder_theorem(&p, 2.0), 0.0);
        assert_eq!(remainder_theorem(&p, 3.0), 0.0);
    }

    #[test]
    fn factor_theorem_test() {
        let p = [-6.0, 11.0, -6.0, 1.0];
        assert!(factor_theorem(&p, 1.0));
        assert!(!factor_theorem(&p, 4.0));
    }

    #[test]
    fn polynomial_gcd_test() {
        let g = polynomial_gcd(&[-1.0, 0.0, 1.0], &[-1.0, 1.0]);
        assert_eq!(g.degree(), 1);
        assert_eq!(g.coeffs(), &[-1.0, 1.0]);
    }

    #[test]
    fn divide_test() {
        let (q, r) = divide(&[-4.0, 0.0, 1.0], &[-2.0, 1.0]).unwrap();
        assert_eq!(q, vec![2.0, 1.0]);
        assert!(r.is_zero());
    }

    #[test]
    fn factor_test() {
        let (factors, residual) = factor(&[6.0, -5.0, 1.0]);
        assert_eq!(factors.len(), 2);
        assert!(residual.is_zero());
    }

    #[test]
    fn divide_by_zero() {
        assert_eq!(divide_by_scalar(&[1.0, 2.0], 0.0), Err(AlgebraError::DivisionByZero));
    }
}
