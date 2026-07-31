//! Polynomial factorization helpers: synthetic division, polynomial long
//! division, GCD (Euclidean algorithm), and rational-root candidates.

use crate::polynomial::Polynomial;

/// Tolerance for coefficient comparisons.
const TOL: f64 = 1e-12;

/// Synthetic division of `coeffs` (lowest-degree first) by `(x - c)`.
///
/// Returns `(quotient_coeffs, remainder)`. The quotient has degree one less
/// than the dividend.
///
/// ```
/// # use mathverse_algebra::factor::synthetic_division;
/// // Divide x² - 5x + 6 by (x - 2)  →  quotient x - 3, remainder 0
/// let (q, r) = synthetic_division(&[6.0, -5.0, 1.0], 2.0);
/// assert!((q[0] - (-3.0)).abs() < 1e-12 && (q[1] - 1.0).abs() < 1e-12);
/// assert!(r.abs() < 1e-12);
/// ```
pub fn synthetic_division(coeffs: &[f64], c: f64) -> (Vec<f64>, f64) {
    if coeffs.is_empty() {
        return (vec![], 0.0);
    }
    let mut result = vec![0.0; coeffs.len() - 1];
    let mut carry = coeffs[coeffs.len() - 1];
    for i in (0..result.len()).rev() {
        result[i] = carry;
        carry = carry * c + coeffs[i];
    }
    let remainder = carry;
    (result, remainder)
}

/// Polynomial long division: `dividend ÷ divisor`.
///
/// Both are lowest-degree-first coefficient slices. Returns
/// `(quotient, remainder)` where `deg(remainder) < deg(divisor)`.
///
/// ```
/// # use mathverse_algebra::factor::divide;
/// // (x³ - 2x² - 5x + 6) ÷ (x - 3) = x² + x - 2, remainder 0
/// let (q, r) = divide(&[6.0, -5.0, -2.0, 1.0], &[-3.0, 1.0]);
/// assert!((q[0] - (-2.0)).abs() < 1e-12);
/// assert!(r.iter().all(|v| v.abs() < 1e-12));
/// ```
pub fn divide(dividend: &[f64], divisor: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if divisor.is_empty() || divisor.iter().all(|&c| c.abs() < TOL) {
        panic!("division by zero polynomial");
    }
    let d_deg = divisor.len().saturating_sub(1);
    let d_lead = divisor[d_deg];

    let mut rem: Vec<f64> = dividend.to_vec();
    let mut quot = vec![0.0; dividend.len().saturating_sub(d_deg).max(1)];

    while rem.len() >= divisor.len() && rem.len() > 1 {
        let shift = rem.len() - divisor.len();
        let coeff = rem[rem.len() - 1] / d_lead;
        if shift < quot.len() {
            quot[shift] = coeff;
        }
        for i in 0..divisor.len() {
            let idx = shift + i;
            if idx < rem.len() {
                rem[idx] -= coeff * divisor[i];
            }
        }
        while rem.len() > 1 && rem.last().map(|v| v.abs()).unwrap_or(0.0) < TOL {
            rem.pop();
        }
    }

    while quot.len() > 1 && quot.last().map(|v| v.abs()).unwrap_or(0.0) < TOL {
        quot.pop();
    }
    (quot, rem)
}

/// Greatest common divisor of two polynomials via the Euclidean algorithm.
///
/// Returns the GCD as a monic polynomial.
///
/// ```
/// # use mathverse_algebra::factor::polynomial_gcd;
/// # use mathverse_algebra::Polynomial;
/// // gcd(x²-1, x-1) = x-1
/// let a = Polynomial::from_coeffs(&[-1.0, 0.0, 1.0]);
/// let b = Polynomial::from_coeffs(&[-1.0, 1.0]);
/// let g = polynomial_gcd(&a, &b);
/// assert!((g.coeffs()[0] - (-1.0)).abs() < 1e-12);
/// assert!((g.coeffs()[1] - 1.0).abs() < 1e-12);
/// ```
pub fn polynomial_gcd(a: &Polynomial, b: &Polynomial) -> Polynomial {
    let mut a = a.coeffs().to_vec();
    let mut b = b.coeffs().to_vec();
    while b.len() > 1 || (b.len() == 1 && b[0].abs() > TOL) {
        let (_, r) = divide(&a, &b);
        a = b;
        b = r;
    }
    make_monic(a)
}

/// Make a polynomial monic (leading coefficient = 1).
fn make_monic(coeffs: Vec<f64>) -> Polynomial {
    if coeffs.is_empty() || coeffs.last().map(|c| c.abs()).unwrap_or(0.0) < TOL {
        return Polynomial::constant(0.0);
    }
    let lead = coeffs[coeffs.len() - 1];
    let normalized: Vec<f64> = coeffs.iter().map(|c| c / lead).collect();
    Polynomial::from_coeffs(&normalized)
}

/// Rational root candidates for `aₙxⁿ + … + a₀` with integer-like coefficients.
///
/// Uses the Rational Root Theorem: if `p/q` (in lowest terms) is a root,
/// then `p | a₀` and `q | aₙ`. Returns unique candidates sorted.
///
/// ```
/// # use mathverse_algebra::factor::rational_root_candidates;
/// // x² - 5x + 6 → candidates ±1, ±2, ±3, ±6
/// let c = rational_root_candidates(&[6.0, -5.0, 1.0]);
/// assert!(c.contains(&2.0));
/// assert!(c.contains(&3.0));
/// ```
pub fn rational_root_candidates(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.len() < 2 {
        return vec![];
    }
    let lead = coeffs[coeffs.len() - 1];
    let constant = coeffs[0];
    if lead.abs() < TOL || constant.abs() < TOL {
        return vec![];
    }
    let p_factors = divisors(constant.abs());
    let q_factors = divisors(lead.abs());
    let mut seen = std::collections::HashSet::new();
    let mut candidates = Vec::new();
    for &p in &p_factors {
        for &q in &q_factors {
            let val = p as f64 / q as f64;
            for &v in &[val, -val] {
                if seen.insert(v.to_bits()) {
                    candidates.push(v);
                }
            }
        }
    }
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    candidates
}

/// Positive integer divisors of `n` (rounded to nearest integer).
fn divisors(n: f64) -> Vec<u64> {
    let n = n.round() as u64;
    if n == 0 {
        return vec![];
    }
    let mut divs = Vec::new();
    for i in 1..=(n as f64).sqrt() as u64 {
        if n % i == 0 {
            divs.push(i);
            if i != n / i {
                divs.push(n / i);
            }
        }
    }
    divs.sort_unstable();
    divs
}

/// Remainder Theorem: the remainder of `f(x) ÷ (x - a)` is `f(a)`.
///
/// ```
/// # use mathverse_algebra::factor::remainder_theorem;
/// # use mathverse_algebra::Polynomial;
/// let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]); // x² - 5x + 6
/// assert!((remainder_theorem(&p, 2.0)).abs() < 1e-12);
/// ```
pub fn remainder_theorem(p: &Polynomial, a: f64) -> f64 {
    p.eval(a)
}

/// Factor Theorem: `(x - a)` is a factor of `f(x)` iff `f(a) = 0`.
///
/// ```
/// # use mathverse_algebra::factor::factor_theorem;
/// # use mathverse_algebra::Polynomial;
/// let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]); // x² - 5x + 6 = (x-2)(x-3)
/// assert!(factor_theorem(&p, 2.0));
/// assert!(!factor_theorem(&p, 0.0));
/// ```
pub fn factor_theorem(p: &Polynomial, a: f64) -> bool {
    p.eval(a).abs() < TOL
}

/// Factor out the greatest common divisor of all coefficients.
///
/// Returns `(gcd, quotient)` where `quotient = poly / gcd`.
///
/// ```
/// # use mathverse_algebra::factor::common_factor;
/// # use mathverse_algebra::Polynomial;
/// let p = Polynomial::from_coeffs(&[4.0, 6.0, 2.0]); // 2x² + 6x + 4 = 2(x² + 3x + 2)
/// let (g, q) = common_factor(&p);
/// assert!((g - 2.0).abs() < 1e-12);
/// assert!((q.coeffs()[0] - 2.0).abs() < 1e-12);
/// ```
pub fn common_factor(p: &Polynomial) -> (f64, Polynomial) {
    let coeffs = p.coeffs();
    let nonzero: Vec<f64> = coeffs.iter().copied().filter(|c| c.abs() > TOL).collect();
    if nonzero.is_empty() {
        return (0.0, p.clone());
    }
    let g = integer_gcd_vec(&nonzero);
    let q = p.clone() * (1.0 / g);
    (g, q)
}

/// GCD of a list of f64 values (treated as integers when possible).
fn integer_gcd_vec(vals: &[f64]) -> f64 {
    let ints: Vec<u64> = vals.iter().map(|v| v.abs().round() as u64).collect();
    if ints.is_empty() {
        return 1.0;
    }
    let mut g = ints[0];
    for &n in &ints[1..] {
        g = gcd_u64(g, n);
    }
    g as f64
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn synth_div_exact() {
        let (q, r) = synthetic_division(&[6.0, -5.0, 1.0], 2.0); // (x-2)(x-3) ÷ (x-2)
        assert!(approx(q[0], -3.0));
        assert!(approx(q[1], 1.0));
        assert!(r.abs() < 1e-12);
    }

    #[test]
    fn synth_div_remainder() {
        let (_, r) = synthetic_division(&[6.0, -5.0, 1.0], 1.0); // f(1) = 2
        assert!(approx(r, 2.0));
    }

    #[test]
    fn long_division() {
        let (q, r) = divide(&[6.0, -5.0, -2.0, 1.0], &[-3.0, 1.0]); // ÷ (x-3)
        assert!(approx(q[0], -2.0));
        assert!(approx(q[1], 1.0));
        assert!(approx(q[2], 1.0));
        assert!(r.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn gcd_poly() {
        let a = Polynomial::from_coeffs(&[-1.0, 0.0, 1.0]); // x²-1
        let b = Polynomial::from_coeffs(&[-1.0, 1.0]); // x-1
        let g = polynomial_gcd(&a, &b);
        assert!(approx(g.coeffs()[0], -1.0));
        assert!(approx(g.coeffs()[1], 1.0));
    }

    #[test]
    fn rational_roots() {
        let c = rational_root_candidates(&[6.0, -5.0, 1.0]);
        assert!(c.contains(&2.0));
        assert!(c.contains(&3.0));
    }

    #[test]
    fn theorems() {
        let p = Polynomial::from_coeffs(&[6.0, -5.0, 1.0]);
        assert!(factor_theorem(&p, 2.0));
        assert!(factor_theorem(&p, 3.0));
        assert!(!factor_theorem(&p, 0.0));
        assert!(approx(remainder_theorem(&p, 1.0), 2.0));
    }

    #[test]
    fn common_factor_extract() {
        let p = Polynomial::from_coeffs(&[4.0, 6.0, 2.0]);
        let (g, q) = common_factor(&p);
        assert!(approx(g, 2.0));
        assert!(approx(q.coeffs()[0], 2.0));
        assert!(approx(q.coeffs()[1], 3.0));
        assert!(approx(q.coeffs()[2], 1.0));
    }
}
