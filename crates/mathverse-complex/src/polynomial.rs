//! Polynomial evaluation and root finding.

use crate::Complex;
use mathverse_core::traits::RealFull;

/// Evaluate a polynomial at `z` via Horner's scheme.
///
/// `coeffs[0] + coeffs[1]·z + coeffs[2]·z² + …`
pub fn eval_polynomial<T: RealFull>(coeffs: &[Complex<T>], z: Complex<T>) -> Complex<T> {
    coeffs
        .iter()
        .rev()
        .fold(Complex::<T>::zero(), |acc, &c| acc * z + c)
}

/// Find all roots of a polynomial using the Durand–Kerner (Weierstrass)
/// method. `coeffs` are ordered `coeffs[0] + coeffs[1]·z + …`, and the
/// leading coefficient must be non-zero.
///
/// Iterates until every root update is below `tol` or `max_iterations` is
/// reached (returns the best estimate either way).
///
/// ```
/// use mathverse_complex::{polynomial::polynomial_roots, Complex};
/// // z² + 1 = 0  →  ±i
/// let roots = polynomial_roots(&[Complex::one(), Complex::zero(), Complex::one()], 1000, 1e-12);
/// let to_i = |r: &Complex| (*r - Complex::i()).norm();
/// let to_neg_i = |r: &Complex| (*r + Complex::i()).norm();
/// assert!(roots.iter().map(to_i).fold(f64::MAX, f64::min) < 1e-8);
/// assert!(roots.iter().map(to_neg_i).fold(f64::MAX, f64::min) < 1e-8);
/// ```
pub fn polynomial_roots<T: RealFull>(
    coeffs: &[Complex<T>],
    max_iterations: usize,
    tol: T,
) -> Vec<Complex<T>> {
    let n = coeffs.len().saturating_sub(1);
    if n == 0 {
        return Vec::new();
    }
    let tol_f = tol.to_f64();
    let two_pi = T::from_f64(core::f64::consts::TAU);
    let n_t = T::from_f64(n as f64);
    // Distinct starting points spread around a small circle (0.4·e^(2πik/n)).
    let mut roots: Vec<Complex<T>> = (0..n)
        .map(|i| Complex::<T>::polar(T::from_f64(0.4), two_pi * T::from_f64(i as f64) / n_t))
        .collect();

    for _ in 0..max_iterations {
        let prev = roots.clone();
        let mut converged = true;
        for i in 0..n {
            let mut denom = Complex::<T>::one();
            for (j, rj) in prev.iter().enumerate() {
                if j != i {
                    denom = denom * (prev[i] - *rj);
                }
            }
            roots[i] = prev[i] - eval_polynomial(coeffs, prev[i]) / denom;
            if (roots[i] - prev[i]).norm().to_f64() > tol_f {
                converged = false;
            }
        }
        if converged {
            break;
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_horner() {
        // 1 + 2z + 3z² at z = 2 → 1 + 4 + 12 = 17
        let coeffs = [Complex::one(), Complex::real(2.0), Complex::real(3.0)];
        assert_eq!(
            eval_polynomial(&coeffs, Complex::real(2.0)),
            Complex::real(17.0)
        );
        assert_eq!(eval_polynomial(&coeffs, Complex::zero()), Complex::one());
    }

    #[test]
    fn roots_quadratic() {
        // z² - 1 → ±1
        let roots = polynomial_roots(
            &[Complex::real(-1.0), Complex::zero(), Complex::one()],
            1000,
            1e-12,
        );
        let to_one = |r: &Complex| (*r - Complex::one()).norm();
        let to_neg_one = |r: &Complex| (*r + Complex::one()).norm();
        assert!(roots.iter().map(to_one).fold(f64::MAX, f64::min) < 1e-8);
        assert!(roots.iter().map(to_neg_one).fold(f64::MAX, f64::min) < 1e-8);
    }

    #[test]
    fn roots_imaginary_unit() {
        // z² + 1 → ±i
        let roots = polynomial_roots(
            &[Complex::one(), Complex::zero(), Complex::one()],
            1000,
            1e-12,
        );
        let to_i = |r: &Complex| (*r - Complex::i()).norm();
        let to_neg_i = |r: &Complex| (*r + Complex::i()).norm();
        assert!(roots.iter().map(to_i).fold(f64::MAX, f64::min) < 1e-8);
        assert!(roots.iter().map(to_neg_i).fold(f64::MAX, f64::min) < 1e-8);
    }

    #[test]
    fn roots_cubic() {
        // (z-1)(z-2)(z+1) = z³ - 2z² - z + 2
        let coeffs = [
            Complex::real(2.0),
            Complex::real(-1.0),
            Complex::real(-2.0),
            Complex::one(),
        ];
        let roots = polynomial_roots(&coeffs, 2000, 1e-12);
        let expected = [1.0, 2.0, -1.0];
        for e in expected {
            let d = roots
                .iter()
                .map(|r| (*r - Complex::real(e)).norm())
                .fold(f64::MAX, f64::min);
            assert!(d < 1e-6, "root {e} not found, got {roots:?}");
        }
    }
}
