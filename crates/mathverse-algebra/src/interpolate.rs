//! Polynomial interpolation: Lagrange and Newton's divided differences.

use crate::polynomial::Polynomial;

/// Lagrange interpolation: find the unique degree-(n-1) polynomial passing
/// through n points `(xi, yi)`.
///
/// Returns a [`Polynomial`] in coefficient form.
///
/// ```
/// # use mathverse_algebra::interpolate::lagrange;
/// let p = lagrange(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0]); // y = x^2
/// assert!((p.eval(2.5) - 6.25).abs() < 1e-9);
/// ```
#[must_use]
pub fn lagrange(xi: &[f64], yi: &[f64]) -> Polynomial {
    assert_eq!(xi.len(), yi.len(), "xi and yi must have the same length");
    let n = xi.len();
    let mut result = Polynomial::constant(0.0);
    for i in 0..n {
        let mut term = Polynomial::constant(yi[i]);
        for j in 0..n {
            if i == j {
                continue;
            }
            let denom = xi[i] - xi[j];
            let factor = Polynomial::from_coeffs(&[-xi[j] / denom, 1.0 / denom]);
            term = term * factor;
        }
        result = result + term;
    }
    result
}

/// Newton's divided differences interpolation.
///
/// Returns the interpolating polynomial.
///
/// ```
/// # use mathverse_algebra::interpolate::newton;
/// let p = newton(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0]);
/// assert!((p.eval(2.5) - 6.25).abs() < 1e-9);
/// ```
#[must_use]
pub fn newton(xi: &[f64], yi: &[f64]) -> Polynomial {
    assert_eq!(xi.len(), yi.len());
    let n = xi.len();
    let mut divided = yi.to_vec();
    let mut coeffs = vec![divided[0]];
    for j in 1..n {
        for i in (j..n).rev() {
            divided[i] = (divided[i] - divided[i - 1]) / (xi[i] - xi[i - j]);
        }
        coeffs.push(divided[j]);
    }
    let mut result = Polynomial::constant(coeffs[0]);
    let mut product = Polynomial::constant(1.0);
    for i in 1..n {
        product = product * Polynomial::from_coeffs(&[-xi[i - 1], 1.0]);
        result = result + product.clone() * Polynomial::constant(coeffs[i]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lagrange_quadratic() {
        let p = lagrange(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0]);
        assert!((p.eval(1.0) - 1.0).abs() < 1e-9);
        assert!((p.eval(2.0) - 4.0).abs() < 1e-9);
        assert!((p.eval(3.0) - 9.0).abs() < 1e-9);
        assert!((p.eval(2.5) - 6.25).abs() < 1e-9);
    }

    #[test]
    fn newton_quadratic() {
        let p = newton(&[1.0, 2.0, 3.0], &[1.0, 4.0, 9.0]);
        assert!((p.eval(1.0) - 1.0).abs() < 1e-9);
        assert!((p.eval(2.0) - 4.0).abs() < 1e-9);
        assert!((p.eval(3.0) - 9.0).abs() < 1e-9);
        assert!((p.eval(2.5) - 6.25).abs() < 1e-9);
    }

    #[test]
    fn lagrange_linear() {
        let p = lagrange(&[0.0, 1.0], &[0.0, 1.0]);
        assert!((p.eval(0.5) - 0.5).abs() < 1e-9);
    }
}