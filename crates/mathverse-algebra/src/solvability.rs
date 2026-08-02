//! Solvability by radicals (Galois-theory flavored checks).
//!
//! Answers the question: *can the roots of a polynomial with integer
//! coefficients be expressed with `+`, `-`, `×`, `÷`, and `ⁿ√` from the
//! coefficients?* That is, is its Galois group solvable?
//!
//! ## What this module does
//!
//! - **Degree ≤ 4**: always solvable by radicals (Abel's theorem; explicit
//!   closed forms live in [`roots`](crate::roots)).
//! - **Degree ≥ 5**: the *generic* polynomial is **not** solvable by radicals
//!   (Abel–Ruffini). This module recognizes the classical special families that
//!   *are* solvable despite the general theorem:
//!   - **Binomials** `xⁿ - a` (roots are n-th roots of `a`).
//!   - **Cyclotomic** `xⁿ - 1` and `xⁿ + 1` (Galois group is abelian).
//!   - **Palindromic / reciprocal** polynomials, which reduce to degree `n/2`
//!     via the substitution `y = x + 1/x`.
//!   - **Reducible** polynomials: if an integer root exists, the problem
//!     reduces to the smaller quotient, recursively.
//!
//! Where the answer genuinely cannot be decided without full factorization, the
//! classifier returns [`Solvability::Unknown`] rather than guessing.

/// Outcome of a solvability classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Solvability {
    /// The polynomial is solvable by radicals; the payload states why.
    SolvableByRadicals(&'static str),
    /// The polynomial is provably not solvable by radicals.
    NotSolvableByRadicals(&'static str),
    /// The answer could not be decided with the implemented checks.
    Unknown(&'static str),
}

/// Trim leading zeros and return the degree, or `None` for the zero polynomial.
#[must_use]
pub fn degree(coeffs: &[i64]) -> Option<usize> {
    if coeffs.is_empty() {
        return None;
    }
    let n = coeffs.len();
    let mut i = n;
    while i > 1 && coeffs[i - 1] == 0 {
        i -= 1;
    }
    if i == 1 && coeffs[0] == 0 {
        None
    } else {
        Some(i - 1)
    }
}

/// Classify whether `coeffs` (lowest-degree first, integer coefficients) is
/// solvable by radicals.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::solvability::{solvable_by_radicals, Solvability};
///
/// // x^2 - 5x + 6 — quadratic, always solvable
/// assert!(matches!(
///     solvable_by_radicals(&[6, -5, 1]),
///     Solvability::SolvableByRadicals(_)
/// ));
///
/// // x^5 - 5x + 5 — Eisenstein-irreducible at p = 5 (not solvable)
/// assert!(matches!(
///     solvable_by_radicals(&[5, -5, 0, 0, 0, 1]),
///     Solvability::NotSolvableByRadicals(_)
/// ));
///
/// // x^5 - 32 — binomial, solvable
/// assert!(matches!(
///     solvable_by_radicals(&[-32, 0, 0, 0, 0, 1]),
///     Solvability::SolvableByRadicals(_)
/// ));
/// ```
#[must_use]
pub fn solvable_by_radicals(coeffs: &[i64]) -> Solvability {
    let Some(mut d) = degree(coeffs) else {
        return Solvability::Unknown("zero polynomial");
    };

    if d <= 4 {
        return Solvability::SolvableByRadicals("degree at most 4");
    }

    let mut cs = coeffs.to_vec();

    // Factor out powers of x: the root x = 0 is trivially "solvable".
    while cs.len() > 1 && cs[0] == 0 {
        cs.remove(0);
        d = match degree(&cs) {
            Some(d) => d,
            None => return Solvability::Unknown("zero polynomial"),
        };
        if d <= 4 {
            return Solvability::SolvableByRadicals(
                "degree at most 4 after factoring out x",
            );
        }
    }

    // Normalize so the leading coefficient is positive.
    if cs[cs.len() - 1] < 0 {
        for c in &mut cs {
            *c = -*c;
        }
    }

    if is_binomial(&cs) {
        return Solvability::SolvableByRadicals("binomial x^n - a");
    }
    if is_cyclotomic(&cs) {
        return Solvability::SolvableByRadicals("cyclotomic (x^n ± 1)");
    }
    if is_palindromic(&cs) {
        return match reduce_reciprocal(&cs) {
            Some(reduced) => match solvable_by_radicals(&reduced) {
                Solvability::SolvableByRadicals(_) => Solvability::SolvableByRadicals(
                    "reciprocal reduction to a solvable lower-degree polynomial",
                ),
                other => other,
            },
            None => Solvability::Unknown("palindromic but reduction unavailable"),
        };
    }
    if is_antipalindromic(&cs) {
        // Antipalindromic polynomials always have x = 1 as a root.
        let q = divide_by_linear(&cs, 1);
        return match solvable_by_radicals(&q) {
            Solvability::SolvableByRadicals(_) => Solvability::SolvableByRadicals(
                "antipalindromic: (x - 1) factor leaves a solvable quotient",
            ),
            other => other,
        };
    }

    // Look for an integer root: if found, reduce to the quotient.
    if let Some(r) = integer_root(&cs) {
        let q = divide_by_linear(&cs, r);
        return match solvable_by_radicals(&q) {
            Solvability::SolvableByRadicals(_) => Solvability::SolvableByRadicals(
                "reducible via integer root to a solvable quotient",
            ),
            other => other,
        };
    }

    // No linear factor. Check irreducibility via Eisenstein's criterion.
    if d >= 5 && eisenstein_irreducible(&cs) {
        return Solvability::NotSolvableByRadicals(
            "irreducible polynomial of degree >= 5 (Eisenstein + Abel-Ruffini)",
        );
    }

    Solvability::Unknown(
        "degree >= 5 with no recognized solvable special form; \
         full factorization would be needed for a definitive answer",
    )
}

/// True when the polynomial has the form `a·xⁿ + b` (exactly two terms).
#[must_use]
pub fn is_binomial(coeffs: &[i64]) -> bool {
    let d = match degree(coeffs) {
        Some(d) => d,
        None => return false,
    };
    d >= 1
        && coeffs[0] != 0
        && coeffs[d] != 0
        && (1..d).all(|i| coeffs[i] == 0)
}

/// True for `xⁿ - 1` or `xⁿ + 1`.
#[must_use]
pub fn is_cyclotomic(coeffs: &[i64]) -> bool {
    let d = match degree(coeffs) {
        Some(d) => d,
        None => return false,
    };
    d >= 1 && coeffs[0].abs() == 1 && coeffs[d].abs() == 1 && is_binomial(coeffs)
}

/// True when `coeffs` is palindromic: `a_k = a_{n-k}` for all `k`.
#[must_use]
pub fn is_palindromic(coeffs: &[i64]) -> bool {
    let d = match degree(coeffs) {
        Some(d) => d,
        None => return false,
    };
    (0..=d / 2).all(|k| coeffs[k] == coeffs[d - k])
}

/// True when `coeffs` is antipalindromic: `a_k = -a_{n-k}` for all `k`.
#[must_use]
pub fn is_antipalindromic(coeffs: &[i64]) -> bool {
    let d = match degree(coeffs) {
        Some(d) => d,
        None => return false,
    };
    (0..=d / 2).all(|k| coeffs[k] == -coeffs[d - k])
}

/// Reduce an even-degree palindromic polynomial `p(x)` to `q(y)` of degree
/// `n/2` via `y = x + 1/x`, where `p(x) = x^(n/2) · q(y)`.
///
/// Returns `None` when the polynomial is not an even-degree palindromic.
#[must_use]
pub fn reduce_reciprocal(coeffs: &[i64]) -> Option<Vec<i64>> {
    let d = degree(coeffs)?;
    if d % 2 != 0 || !is_palindromic(coeffs) {
        return None;
    }
    let half = d / 2;
    // p(x) / x^half = a_half + sum_{k=1}^{half} a_{half-k} * (x^k + x^{-k}),
    // and x^k + x^{-k} = P_k(y) with P_0 = 2, P_1 = y, P_k = y*P_{k-1} - P_{k-2}.
    let mut q = vec![0i64; half + 1];
    q[0] += coeffs[half];

    let mut prev2 = vec![2i64]; // P_0
    let mut prev1 = vec![0i64, 1i64]; // P_1 = y
    for k in 1..=half {
        let pk: Vec<i64> = if k == 1 {
            prev1.clone()
        } else {
            let mut np = vec![0i64; prev1.len() + 1];
            for (j, &c) in prev1.iter().enumerate() {
                np[j + 1] += c;
            }
            for (j, &c) in prev2.iter().enumerate() {
                np[j] -= c;
            }
            prev2 = prev1;
            prev1 = np.clone();
            np
        };
        let idx = half - k;
        for (j, &c) in pk.iter().enumerate() {
            q[idx + j] += coeffs[idx] * c;
        }
    }

    trim(&mut q);
    Some(q)
}

/// True if Eisenstein's criterion proves the polynomial irreducible over ℚ.
///
/// Finds a prime `p` with `p | a_0..a_{n-1}`, `p ∤ a_n`, and `p² ∤ a_0`.
/// Only the divisors of `a_0` are candidates for `p`, and once `p > √|a_0|`
/// the `p² ∤ a_0` condition holds automatically, which keeps the arithmetic
/// within `i64` without overflow.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn eisenstein_irreducible(coeffs: &[i64]) -> bool {
    let Some(d) = degree(coeffs) else {
        return false;
    };
    if d == 0 {
        return false;
    }
    let leading = coeffs[d];
    let constant = coeffs[0];
    if leading == 0 || constant == 0 {
        return false;
    }
    let abs_constant = constant.unsigned_abs();
    let sqrt_a0 = abs_constant.isqrt() as i64;
    divisors(abs_constant).into_iter().any(|p| {
        if p < 2 || !is_prime(p) {
            return false;
        }
        let p = p as i64;
        (0..d).all(|i| coeffs[i] % p == 0)
            && leading % p != 0
            && (p > sqrt_a0 || constant % (p * p) != 0)
    })
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

/// First integer root of a polynomial with integer coefficients, if any.
///
/// Candidates are the divisors of the constant term `a_0` (rational-root
/// theorem); `0` is returned immediately when `a_0 == 0`.
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn integer_root(coeffs: &[i64]) -> Option<i64> {
    let a0 = *coeffs.first()?;
    if a0 == 0 {
        return Some(0);
    }
    let mut candidates: Vec<i64> = Vec::new();
    for p in divisors(a0.unsigned_abs()) {
        candidates.push(p as i64);
        candidates.push(-(p as i64));
    }
    candidates.into_iter().find(|&r| eval(coeffs, r) == 0)
}

fn eval(coeffs: &[i64], x: i64) -> i128 {
    let mut acc: i128 = 0;
    for &c in coeffs.iter().rev() {
        acc = acc.wrapping_mul(x as i128).wrapping_add(c as i128);
    }
    acc
}

fn divisors(n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut i = 1;
    while i * i <= n {
        if n % i == 0 {
            out.push(i);
            if i != n / i {
                out.push(n / i);
            }
        }
        i += 1;
    }
    out
}

/// Synthetic division of `coeffs` by `(x - r)`; returns the quotient.
#[must_use]
pub fn divide_by_linear(coeffs: &[i64], r: i64) -> Vec<i64> {
    let n = coeffs.len();
    if n <= 1 {
        return vec![];
    }
    let mut q = vec![0i64; n - 1];
    q[n - 2] = coeffs[n - 1];
    for k in (1..n - 1).rev() {
        q[k - 1] = coeffs[k] + r * q[k];
    }
    trim(&mut q);
    q
}

fn trim(coeffs: &mut Vec<i64>) {
    while coeffs.len() > 1 && coeffs[coeffs.len() - 1] == 0 {
        coeffs.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_low() {
        assert_eq!(degree(&[0]), None);
        assert_eq!(degree(&[5]), Some(0));
        assert_eq!(degree(&[1, 2, 3]), Some(2));
        assert_eq!(degree(&[1, 2, 3, 0, 0]), Some(2));
    }

    #[test]
    fn quadratic_solvable() {
        assert!(matches!(
            solvable_by_radicals(&[6, -5, 1]),
            Solvability::SolvableByRadicals(_)
        ));
    }

    #[test]
    fn eisenstein_quintic_not_solvable() {
        // x^5 - 5x + 5: irreducible by Eisenstein at p = 5.
        assert!(matches!(
            solvable_by_radicals(&[5, -5, 0, 0, 0, 1]),
            Solvability::NotSolvableByRadicals(_)
        ));
    }

    #[test]
    fn generic_quintic_unknown() {
        // x^5 - x - 1 is irreducible but not provable by the built-in checks.
        assert!(matches!(
            solvable_by_radicals(&[-1, -1, 0, 0, 0, 1]),
            Solvability::Unknown(_)
        ));
    }

    #[test]
    fn binomial_solvable() {
        assert!(matches!(
            solvable_by_radicals(&[-32, 0, 0, 0, 0, 1]),
            Solvability::SolvableByRadicals(_)
        ));
    }

    #[test]
    fn cyclotomic_solvable() {
        assert!(matches!(
            solvable_by_radicals(&[-1, 0, 0, 0, 0, 0, 1]),
            Solvability::SolvableByRadicals(_)
        ));
        assert!(is_cyclotomic(&[1, 0, 0, 0, 1]));
        assert!(is_cyclotomic(&[-1, 0, 0, 0, 0, 1]));
    }

    #[test]
    fn palindromic_reduction() {
        // x^6 + x^3 + 1 is palindromic; reduces to y^3 - 3y + 1.
        let cs = vec![1, 0, 0, 1, 0, 0, 1];
        assert!(is_palindromic(&cs));
        let reduced = reduce_reciprocal(&cs).expect("reduces");
        assert_eq!(reduced, vec![1, -3, 0, 1]);
        assert!(matches!(
            solvable_by_radicals(&cs),
            Solvability::SolvableByRadicals(_)
        ));
    }

    #[test]
    fn antipalindromic_has_root_one() {
        // x^4 - 1 is antipalindromic; antipalindromics always have x = 1.
        let cs = vec![-1, 0, 0, 0, 1];
        assert!(is_antipalindromic(&cs));
        assert_eq!(integer_root(&cs), Some(1));
        let q = divide_by_linear(&cs, 1);
        assert_eq!(q, vec![1, 1, 1, 1]);
    }

    #[test]
    fn integer_root_detection() {
        // (x - 2)(x^4 + x + 1) → x^5 - 2x^4 + x^2 - x - 2
        let cs = vec![-2, -1, 1, 0, -2, 1];
        assert_eq!(integer_root(&cs), Some(2));
    }

    #[test]
    fn eisenstein_check() {
        // x^3 + 2x + 2: p = 2 works.
        assert!(eisenstein_irreducible(&[2, 2, 0, 1]));
        assert!(!eisenstein_irreducible(&[6, -5, 1]));
    }

    #[test]
    fn divide_linear_ok() {
        // (x^2 - 5x + 6) / (x - 2) = x - 3
        let q = divide_by_linear(&[6, -5, 1], 2);
        assert_eq!(q, vec![-3, 1]);
    }
}
