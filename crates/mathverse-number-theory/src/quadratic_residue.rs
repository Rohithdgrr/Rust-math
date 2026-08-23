//! Legendre/Jacobi symbols, Tonelli-Shanks square root mod p, quadratic residues.

/// Legendre symbol `(a/p)` for odd prime `p`.
///
/// Returns `1` if `a` is a quadratic residue mod `p`, `-1` if not,
/// `0` if `p | a`.
///
/// ```
/// use mathverse_number_theory::legendre;
/// assert_eq!(legendre(2, 7), 1);
/// assert_eq!(legendre(3, 7), -1);
/// assert_eq!(legendre(7, 7), 0);
/// ```
#[must_use]
pub fn legendre(a: u64, p: u64) -> i64 {
    if p == 2 {
        return i64::from(!a.is_multiple_of(2));
    }
    if a.is_multiple_of(p) {
        return 0;
    }
    let ls = crate::modular::mod_pow_unchecked(a % p, (p - 1) / 2, p);
    if ls == p - 1 {
        -1
    } else {
        ls as i64
    }
}

/// Jacobi symbol `(a/n)` for odd `n > 0`.
///
/// Generalization of the Legendre symbol to composite moduli.
/// Returns `0` if `gcd(a, n) ≠ 1`.
///
/// ```
/// use mathverse_number_theory::jacobi;
/// assert_eq!(jacobi(1, 9), 1);
/// assert_eq!(jacobi(2, 9), 1); // (2/3)² = (-1)² = 1
/// assert_eq!(jacobi(3, 9), 0);
/// ```
#[must_use]
pub const fn jacobi(mut a: u64, mut n: u64) -> i64 {
    if n == 0 || n.is_multiple_of(2) {
        return 0;
    }
    a %= n;
    let mut result = 1i64;
    while a != 0 {
        while a.is_multiple_of(2) {
            a /= 2;
            if n % 8 == 3 || n % 8 == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }
    if n == 1 {
        result
    } else {
        0
    }
}

/// Tonelli-Shanks algorithm: finds a square root of `n` modulo odd prime `p`.
///
/// Returns `Some(x)` such that `x² ≡ n (mod p)`, or `None` if no solution exists.
/// When a solution exists, the two roots are `x` and `p - x`.
///
/// ```
/// use mathverse_number_theory::tonelli_shanks;
/// let r = tonelli_shanks(2, 7).unwrap();
/// assert_eq!((r * r) % 7, 2);
/// assert_eq!(tonelli_shanks(3, 7), None);
/// ```
#[must_use]
pub fn tonelli_shanks(n: u64, p: u64) -> Option<u64> {
    if p == 2 {
        return Some(n % 2);
    }
    if p.is_multiple_of(2) || !crate::primes::is_prime(p) {
        return None;
    }
    let n = n % p;
    if n == 0 {
        return Some(0);
    }
    if legendre(n, p) != 1 {
        return None;
    }
    if p % 4 == 3 {
        return Some(crate::modular::mod_pow_unchecked(n, (p + 1) / 4, p));
    }
    let mut odd_part = p - 1;
    let mut two_adicity = 0u32;
    while odd_part.is_multiple_of(2) {
        odd_part /= 2;
        two_adicity += 1;
    }
    let mut nonresidue = 2;
    while legendre(nonresidue, p) != -1 {
        nonresidue += 1;
    }
    let mut cur_unity = crate::modular::mod_pow_unchecked(nonresidue, odd_part, p);
    let mut cur_term = crate::modular::mod_pow_unchecked(n, odd_part, p);
    let mut root = crate::modular::mod_pow_unchecked(n, (odd_part + 1).div_ceil(2), p);
    loop {
        if cur_term == 1 {
            return Some(root);
        }
        let mut idx = 1u32;
        let mut t_pow = cur_term;
        while t_pow != 1 {
            t_pow = crate::modular::mod_mul(t_pow, t_pow, p);
            idx += 1;
            if idx > two_adicity {
                return None;
            }
        }
        let step_mult =
            crate::modular::mod_pow_unchecked(cur_unity, 1u64 << (two_adicity - idx - 1), p);
        cur_unity = crate::modular::mod_mul(step_mult, step_mult, p);
        cur_term = crate::modular::mod_mul(cur_term, cur_unity, p);
        root = crate::modular::mod_mul(root, step_mult, p);
    }
}

/// All quadratic residues mod `p` (values `x² mod p` for `x ∈ [0, p)`).
///
/// ```
/// use mathverse_number_theory::quadratic_residues;
/// let r = quadratic_residues(7);
/// assert!(r.contains(&1));
/// assert!(r.contains(&2));
/// assert!(r.contains(&4));
/// ```
#[must_use]
pub fn quadratic_residues(p: u64) -> Vec<u64> {
    let mut res = std::collections::HashSet::new();
    for x in 0..p {
        res.insert(crate::modular::mod_mul(x, x, p));
    }
    let mut v: Vec<u64> = res.into_iter().collect();
    v.sort_unstable();
    v
}

/// Returns `true` if `a` is a quadratic residue mod `p` (p odd prime).
///
/// ```
/// use mathverse_number_theory::is_quadratic_residue;
/// assert!(is_quadratic_residue(2, 7));
/// assert!(!is_quadratic_residue(3, 7));
/// ```
#[must_use]
#[inline]
pub fn is_quadratic_residue(a: u64, p: u64) -> bool {
    legendre(a, p) == 1
}

/// Combines quadratic residue roots via CRT.
///
/// Given `x² ≡ a (mod m)` and `x² ≡ a (mod n)` where `m` and `n` are
/// coprime odd primes, finds `x` such that `x² ≡ a (mod m*n)`.
///
/// Returns `None` if `m` and `n` are not coprime, or if `a` is not a
/// quadratic residue modulo both.
///
/// ```
/// use mathverse_number_theory::chinese_remainder_quadratic;
/// // 1 is a QR mod 3 and mod 5; combine via CRT.
/// assert!(chinese_remainder_quadratic(1, 3, 5).is_some());
/// ```
#[must_use]
pub fn chinese_remainder_quadratic(a: u64, m: u64, n: u64) -> Option<u64> {
    if crate::gcd(m, n) != 1 {
        return None;
    }
    let r1 = tonelli_shanks(a, m)?;
    let r2 = tonelli_shanks(a, n)?;
    crate::crt(&[r1, r2], &[m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legendre_test() {
        assert_eq!(legendre(2, 7), 1);
        assert_eq!(legendre(3, 7), -1);
        assert_eq!(legendre(7, 7), 0);
    }

    #[test]
    fn jacobi_test() {
        assert_eq!(jacobi(1, 9), 1);
        assert_eq!(jacobi(2, 9), 1); // 9 ≡ 1 (mod 8), so (2/9) = 1
        assert_eq!(jacobi(3, 9), 0);
    }

    #[test]
    fn tonelli_test() {
        let r = tonelli_shanks(2, 7).unwrap();
        assert_eq!(crate::modular::mod_mul(r, r, 7), 2);
        assert_eq!(tonelli_shanks(3, 7), None);
    }

    #[test]
    fn residues_test() {
        let r = quadratic_residues(7);
        assert!(r.contains(&1));
        assert!(r.contains(&2));
        assert!(r.contains(&4));
    }

    #[test]
    fn crq_test() {
        assert!(chinese_remainder_quadratic(1, 3, 5).is_some());
        // 2 is NOT a quadratic residue mod 3, so no solution exists
        assert!(chinese_remainder_quadratic(2, 3, 7).is_none());
        assert_eq!(chinese_remainder_quadratic(1, 3, 9), None);
    }
}
