//! Modular arithmetic: exponentiation, inverse, CRT, addition/subtraction/multiplication.

/// Modular exponentiation: `base^exp mod m` using square-and-multiply.
///
/// Returns `None` if `m == 0` (division by zero).
/// Returns `0` if `m == 1`.
///
/// ```
/// use mathverse_number_theory::mod_pow;
/// assert_eq!(mod_pow(2, 10, 1000), Some(24));
/// assert_eq!(mod_pow(3, 13, 7), Some(3));
/// assert_eq!(mod_pow(5, 3, 1), Some(0));
/// assert_eq!(mod_pow(2, 10, 0), None);
/// ```
#[must_use]
pub fn mod_pow(base: u64, mut exp: u64, m: u64) -> Option<u64> {
    if m == 0 {
        return None;
    }
    if m == 1 {
        return Some(0);
    }
    let mut result = 1u128;
    let mut b = base as u128 % m as u128;
    let mm = m as u128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % mm;
        }
        b = (b * b) % mm;
        exp >>= 1;
    }
    Some(result as u64)
}

/// Internal modular exponentiation that assumes `m > 1`.
/// Faster than [`mod_pow`] for internal use where the modulus is known
/// to be greater than 1.
#[inline]
pub fn mod_pow_unchecked(base: u64, exp: u64, m: u64) -> u64 {
    let mut result = 1u128;
    let mut b = base as u128 % m as u128;
    let mm = m as u128;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = (result * b) % mm;
        }
        b = (b * b) % mm;
        e >>= 1;
    }
    result as u64
}

/// Modular multiplicative inverse: `a^-1 mod m`.
///
/// Returns `None` if `a` and `m` are not coprime, or if `m == 0`.
///
/// ```
/// use mathverse_number_theory::mod_inverse;
/// assert_eq!(mod_inverse(3, 11), Some(4)); // 3*4 ≡ 1 (mod 11)
/// assert_eq!(mod_inverse(4, 6), None);     // gcd(4,6) ≠ 1
/// assert_eq!(mod_inverse(3, 0), None);     // division by zero
/// ```
#[must_use]
pub fn mod_inverse(a: u64, m: u64) -> Option<u64> {
    if m == 0 {
        return None;
    }
    let (mut r0, mut r1) = (m as i128, a as i128 % m as i128);
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    if r0 != 1 {
        None
    } else {
        Some((t0.rem_euclid(m as i128)) as u64)
    }
}

/// Modular addition: `(a + b) mod m`.
///
/// Returns `0` if `m == 0`.
///
/// ```
/// use mathverse_number_theory::mod_add;
/// assert_eq!(mod_add(5, 7, 3), 0);
/// assert_eq!(mod_add(2, 3, 10), 5);
/// ```
#[must_use]
pub fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let (ra, rb) = (a % m, b % m);
    let sum = ra.saturating_add(rb);
    if sum >= m {
        sum - m
    } else {
        sum
    }
}

/// Modular subtraction: `(a - b) mod m`.
///
/// Returns `0` if `m == 0`.
///
/// ```
/// use mathverse_number_theory::mod_sub;
/// assert_eq!(mod_sub(2, 5, 7), 4);
/// assert_eq!(mod_sub(10, 3, 7), 0);
/// ```
#[must_use]
pub fn mod_sub(a: u64, b: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let (ra, rb) = (a % m, b % m);
    if ra >= rb {
        ra - rb
    } else {
        m - (rb - ra)
    }
}

/// Modular multiplication: `(a * b) mod m` using `u128` to avoid overflow.
///
/// Returns `0` if `m == 0`.
///
/// ```
/// use mathverse_number_theory::mod_mul;
/// assert_eq!(mod_mul(3, 4, 5), 2);
/// ```
#[must_use]
pub fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    (a as u128 * b as u128 % m as u128) as u64
}

/// Modular division: `a / b mod m` = `a * b^-1 mod m`.
///
/// Returns `None` if `b` has no inverse mod `m` (i.e., `gcd(b, m) ≠ 1`).
///
/// ```
/// use mathverse_number_theory::mod_div;
/// assert_eq!(mod_div(6, 3, 11), Some(2)); // 6/3 mod 11 = 2
/// assert_eq!(mod_div(1, 2, 4), None);     // 2 has no inverse mod 4
/// ```
#[must_use]
pub fn mod_div(a: u64, b: u64, m: u64) -> Option<u64> {
    let bi = mod_inverse(b, m)?;
    Some(mod_mul(a, bi, m))
}

/// Chinese Remainder Theorem: find `x` such that `x ≡ rems[i] (mod mods[i])`
/// for all `i`.
///
/// Works with non-coprime moduli: returns `None` if the system is inconsistent.
/// Uses checked arithmetic to avoid overflow for large inputs.
///
/// ```
/// use mathverse_number_theory::crt;
/// assert_eq!(crt(&[2, 3], &[3, 5]), Some(8));
/// assert_eq!(crt(&[1, 2], &[2, 4]), None); // inconsistent
/// assert_eq!(crt(&[], &[]), None);        // empty input
/// ```
#[must_use]
pub fn crt(rems: &[u64], mods: &[u64]) -> Option<u64> {
    if rems.is_empty() || mods.is_empty() || rems.len() != mods.len() {
        return None;
    }
    let mut result = rems[0] as i128;
    let mut lcm = mods[0] as i128;
    for i in 1..rems.len() {
        let (a1, m1) = (result, lcm);
        let (a2, m2) = (rems[i] as i128, mods[i] as i128);
        let g = gcd_i128(m1, m2);
        if (a2 - a1).rem_euclid(g) != 0 {
            return None;
        }
        let (_, s, _) = extended_gcd_i128(m1 / g, m2 / g);
        let diff = (a2 - a1) / g;
        let r = diff.checked_mul(s)?.rem_euclid(m2 / g);
        let m1_r = m1.checked_mul(r)?;
        result = a1.checked_add(m1_r)?;
        lcm = lcm.checked_div(g)?.checked_mul(m2)?;
        result = result.rem_euclid(lcm);
    }
    if result < 0 || result > u64::MAX as i128 {
        None
    } else {
        Some(result as u64)
    }
}

/// Solves the linear congruence `b*x ≡ a (mod m)`.
///
/// Returns all solutions modulo `m` as a sorted `Vec`.
/// A solution exists iff `gcd(b, m)` divides `a`.
///
/// ```
/// use mathverse_number_theory::solve_linear_congruence;
/// // 3x ≡ 1 (mod 7) → x = 5
/// assert_eq!(solve_linear_congruence(1, 3, 7), Some(vec![5]));
/// // 6x ≡ 4 (mod 8) → 2x ≡ 4/2 (mod 8/2) → x ≡ 2 (mod 4) → {2, 6}
/// assert_eq!(solve_linear_congruence(4, 6, 8), Some(vec![2, 6]));
/// // 2x ≡ 3 (mod 4) → no solution (gcd(2,4)=2 ∤ 3)
/// assert_eq!(solve_linear_congruence(3, 2, 4), None);
/// ```
#[must_use]
pub fn solve_linear_congruence(a: u64, b: u64, m: u64) -> Option<Vec<u64>> {
    if m == 0 {
        return None;
    }
    let g = gcd(b, m);
    if a % g != 0 {
        return None;
    }
    let (b_reduced, m_reduced, a_reduced) = (b / g, m / g, a / g);
    let inv = mod_inverse(b_reduced, m_reduced)?;
    let x0 = mod_mul(a_reduced, inv, m_reduced);
    let mut solutions: Vec<u64> = (0..g).map(|k| x0 + k * m_reduced).collect();
    solutions.sort_unstable();
    Some(solutions)
}

/// Checks if the linear congruence `b*x ≡ a (mod m)` has a solution.
///
/// A solution exists iff `gcd(b, m)` divides `a`.
///
/// ```
/// use mathverse_number_theory::linear_congruence_solvable;
/// assert!(linear_congruence_solvable(1, 3, 7));  // 3x ≡ 1 (mod 7)
/// assert!(linear_congruence_solvable(4, 6, 8));  // 6x ≡ 4 (mod 8)
/// assert!(!linear_congruence_solvable(3, 2, 4)); // 2x ≡ 3 (mod 4): gcd(2,4)=2 ∤ 3
/// ```
#[must_use]
pub fn linear_congruence_solvable(a: u64, b: u64, m: u64) -> bool {
    if m == 0 {
        return false;
    }
    let g = gcd(b, m);
    a % g == 0
}

#[inline]
fn gcd_i128(a: i128, b: i128) -> i128 {
    if b == 0 {
        a.abs()
    } else {
        gcd_i128(b, a % b)
    }
}

#[inline]
fn extended_gcd_i128(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x1, y1) = extended_gcd_i128(b % a, a);
    (g, y1 - (b / a) * x1, x1)
}

#[inline]
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow() {
        assert_eq!(mod_pow(2, 10, 1000), Some(24));
        assert_eq!(mod_pow(3, 13, 7), Some(3));
        assert_eq!(mod_pow(2, 10, 0), None);
        assert_eq!(mod_pow(5, 3, 1), Some(0));
    }

    #[test]
    fn inverse() {
        assert_eq!(mod_inverse(3, 11), Some(4));
        assert_eq!(mod_inverse(4, 6), None);
        assert_eq!(mod_inverse(3, 0), None);
    }

    #[test]
    fn crt_test() {
        assert_eq!(crt(&[2, 3], &[3, 5]), Some(8));
        assert_eq!(crt(&[1, 2], &[2, 4]), None);
        assert_eq!(crt(&[], &[]), None);
        assert_eq!(crt(&[2, 3, 2], &[3, 5, 7]), Some(23));
    }

    #[test]
    fn arithmetic() {
        assert_eq!(mod_add(5, 7, 3), 0);
        assert_eq!(mod_sub(2, 5, 7), 4);
        assert_eq!(mod_mul(3, 4, 5), 2);
    }

    #[test]
    fn mod_add_overflow() {
        assert_eq!(mod_add(u64::MAX, 1, 100), (u64::MAX % 100 + 1) % 100);
        assert_eq!(mod_add(u64::MAX, u64::MAX, 7), (u64::MAX % 7 * 2) % 7);
        assert_eq!(mod_add(0, 0, 1), 0);
        assert_eq!(mod_add(5, 3, 0), 0);
    }

    #[test]
    fn mod_sub_underflow() {
        assert_eq!(mod_sub(0, 1, 7), 6);
        assert_eq!(mod_sub(3, 5, 7), 5);
        assert_eq!(mod_sub(u64::MAX, 0, 99), u64::MAX % 99);
        assert_eq!(mod_sub(5, 3, 0), 0);
    }

    #[test]
    fn linear_congruence_test() {
        assert_eq!(solve_linear_congruence(1, 3, 7), Some(vec![5]));
        assert_eq!(solve_linear_congruence(4, 6, 8), Some(vec![2, 6]));
        assert_eq!(solve_linear_congruence(3, 2, 4), None);
        assert!(linear_congruence_solvable(1, 3, 7));
        assert!(linear_congruence_solvable(4, 6, 8));
        assert!(!linear_congruence_solvable(3, 2, 4));
    }

    #[test]
    fn mod_div_test() {
        assert_eq!(mod_div(6, 3, 11), Some(2));
        assert_eq!(mod_div(1, 2, 4), None);
    }
}
