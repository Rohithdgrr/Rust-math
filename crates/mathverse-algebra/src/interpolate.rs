//! Polynomial interpolation: Lagrange and Newton's divided differences.

use crate::polynomial::Polynomial;

const TOL: f64 = 1e-12;

/// Lagrange interpolation: given points `(xᵢ, yᵢ)`, return the polynomial
/// `P(x)` of degree ≤ n−1 passing through all points.
///
/// ```
/// # use mathverse_algebra::interpolate::lagrange;
/// // Points (0,1), (1,2), (2,5) → P(x) = x² + 1
/// let p = lagrange(&[(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)]);
/// assert!((p.eval(3.0) - 10.0).abs() < 1e-9);
/// ```
pub fn lagrange(points: &[(f64, f64)]) -> Polynomial {
    let n = points.len();
    if n == 0 {
        return Polynomial::constant(0.0);
    }
    let mut result = Polynomial::constant(0.0);
    for i in 0..n {
        let (xi, yi) = points[i];
        let mut term = Polynomial::constant(yi);
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = points[j].0;
            let denom = xi - xj;
            if denom.abs() < TOL {
                continue;
            }
            // Multiply term by (x - xj) / (xi - xj)
            let factor = Polynomial::from_coeffs(&[-xj / denom, 1.0 / denom]);
            term = term * factor;
        }
        result = result + term;
    }
    result
}

/// Newton's divided-difference interpolation.
///
/// Returns the interpolating polynomial in Newton form (expanded to standard
/// form).
///
/// ```
/// # use mathverse_algebra::interpolate::newton;
/// let p = newton(&[(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)]);
/// assert!((p.eval(3.0) - 10.0).abs() < 1e-9);
/// ```
pub fn newton(points: &[(f64, f64)]) -> Polynomial {
    let n = points.len();
    if n == 0 {
        return Polynomial::constant(0.0);
    }

    // Build divided difference table.
    let mut dd: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        dd[i][0] = points[i].1;
    }
    for j in 1..n {
        for i in 0..n - j {
            let denom = points[i + j].0 - points[i].0;
            if denom.abs() < TOL {
                dd[i][j] = dd[i + 1][j - 1];
            } else {
                dd[i][j] = (dd[i + 1][j - 1] - dd[i][j - 1]) / denom;
            }
        }
    }

    // Build polynomial: P(x) = Σ dd[0][k] · Π(x - xⱼ)
    let mut result = Polynomial::constant(dd[0][0]);
    let mut basis = Polynomial::constant(1.0);
    for k in 1..n {
        let xj = points[k - 1].0;
        let factor = Polynomial::from_coeffs(&[-xj, 1.0]); // (x - xj)
        basis = basis.clone() * factor;
        result = result + basis.clone() * dd[0][k];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn lagrange_quadratic() {
        let p = lagrange(&[(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)]);
        assert!(approx(p.eval(0.0), 1.0));
        assert!(approx(p.eval(1.0), 2.0));
        assert!(approx(p.eval(2.0), 5.0));
        assert!(approx(p.eval(3.0), 10.0));
    }

    #[test]
    fn newton_quadratic() {
        let p = newton(&[(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)]);
        assert!(approx(p.eval(0.0), 1.0));
        assert!(approx(p.eval(1.0), 2.0));
        assert!(approx(p.eval(2.0), 5.0));
        assert!(approx(p.eval(3.0), 10.0));
    }

    #[test]
    fn lagrange_linear() {
        let p = lagrange(&[(0.0, 0.0), (2.0, 4.0)]); // y = 2x
        assert!(approx(p.eval(5.0), 10.0));
    }

    #[test]
    fn empty_points() {
        let p = lagrange(&[]);
        assert_eq!(p.degree(), 0);
    }
}
