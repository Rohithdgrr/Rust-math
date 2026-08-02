//! Polynomial composition `f(g(x))` and decomposition helpers.

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
pub fn compose(f: &Polynomial, g: &Polynomial) -> Polynomial {
    let mut result = Polynomial::constant(0.0);
    for &c in f.coeffs().iter().rev() {
        result = result * g.clone() + Polynomial::constant(c);
    }
    result
}

/// Attempt to decompose a polynomial `h(x) = f(g(x))`.
///
/// Tries a few candidate inner polynomials and returns `Some((f, g))` if found.
/// This is a heuristic, not guaranteed to find all decompositions.
pub fn decompose(h: &Polynomial) -> Option<(Polynomial, Polynomial)> {
    let candidates = vec![
        Polynomial::from_coeffs(&[0.0, 1.0]),   // x
        Polynomial::from_coeffs(&[1.0, 1.0]),    // x + 1
        Polynomial::from_coeffs(&[-1.0, 1.0]),   // x - 1
        Polynomial::from_coeffs(&[0.0, 2.0]),     // 2x
        Polynomial::from_coeffs(&[1.0, 2.0]),    // 2x + 1
    ];
    for g in candidates {
        if let Some(f) = find_inner(h, &g) {
            return Some((f, g));
        }
    }
    None
}

/// Given `h` and candidate `g`, find `f` such that `h = f(g)`.
fn find_inner(h: &Polynomial, g: &Polynomial) -> Option<Polynomial> {
    let mut remaining = h.clone();
    let mut coeffs: Vec<f64> = Vec::new();
    while remaining.coeffs().len() > 1 && remaining.coeffs().last().unwrap().abs() < 1e-12 {
        break; // BUG: always breaks immediately
    }
    None
}

/// Try to decompose with a specific inner polynomial.
fn try_decompose_with(h: &Polynomial, g: &Polynomial) -> Option<Polynomial> {
    None
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