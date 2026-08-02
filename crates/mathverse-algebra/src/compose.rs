//! Polynomial composition `f(g(x))`.

use crate::polynomial::Polynomial;

/// Compose two polynomials: `f(g(x))`.
///
/// Uses Horner's method for efficiency.
///
/// ```
/// # use mathverse_algebra::{Polynomial, compose::compose};
/// let f = Polynomial::from_coeffs(&[1.0, 1.0]); // x + 1
/// let g = Polynomial::from_coeffs(&[0.0, 1.0]); // x
/// let h = compose(&f, &g);
/// assert_eq!(h.coeffs(), &[1.0, 1.0]);
/// ```
#[must_use]
pub fn compose(f: &Polynomial, g: &Polynomial) -> Polynomial {
    let mut result = Polynomial::constant(0.0);
    for &c in f.coeffs().iter().rev() {
        result = result * g.clone() + Polynomial::constant(c);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_basic() {
        let f = Polynomial::from_coeffs(&[1.0, 2.0]); // 2x + 1
        let g = Polynomial::from_coeffs(&[0.0, 1.0, 1.0]); // x^2 + x
        let h = compose(&f, &g);
        assert_eq!(h.coeffs(), &[1.0, 2.0, 2.0]);
    }

    #[test]
    fn compose_identity() {
        let f = Polynomial::from_coeffs(&[1.0, 2.0, 3.0]);
        let x = Polynomial::x();
        let h = compose(&f, &x);
        assert_eq!(h.coeffs(), f.coeffs());
    }
}
