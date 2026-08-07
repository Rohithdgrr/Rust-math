//! Prime factorization, divisors, divisor functions, Möbius and Liouville functions.

/// Returns the prime factors of `n` in non-decreasing order.
///
/// Returns `[0]` for `n = 0`, and an empty vector for `n = 1`.
///
/// ```
/// use mathverse_number_theory::prime_factors;
/// assert_eq!(prime_factors(84), vec![2, 2, 3, 7]);
/// assert!(prime_factors(1).is_empty());
/// ```
#[must_use]
pub fn prime_factors(mut n: u64) -> Vec<u64> {
    if n == 0 {
        return vec![0];
    }
    if n == 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    while n.is_multiple_of(2) {
        out.push(2);
        n /= 2;
    }
    let mut d = 3u64;
    while (d as u128) * (d as u128) <= n as u128 {
        while n.is_multiple_of(d) {
            out.push(d);
            n /= d;
        }
        d += 2;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Returns all positive divisors of `n` in sorted order.
///
/// ```
/// use mathverse_number_theory::divisors;
/// assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
/// ```
#[must_use]
pub fn divisors(n: u64) -> Vec<u64> {
    if n == 0 {
        return Vec::new();
    }
    let mut divs = Vec::new();
    let mut i = 1u64;
    while (i as u128) * (i as u128) <= n as u128 {
        if n.is_multiple_of(i) {
            divs.push(i);
            if i != n / i {
                divs.push(n / i);
            }
        }
        i += 1;
    }
    divs.sort_unstable();
    divs
}

/// Number of divisors of `n` (divisor function σ₀ or d(n)).
///
/// ```
/// use mathverse_number_theory::divisor_count;
/// assert_eq!(divisor_count(12), 6);
/// assert_eq!(divisor_count(1), 1);
/// ```
#[must_use]
pub fn divisor_count(n: u64) -> u64 {
    if n <= 1 {
        return 1;
    }
    let factors = prime_factors(n);
    let mut result = 1u64;
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut count = 0u32;
        while i < factors.len() && factors[i] == p {
            count += 1;
            i += 1;
        }
        result = result.saturating_mul(count as u64 + 1);
    }
    result
}

/// Sum of all positive divisors of `n` (divisor function σ₁).
///
/// ```
/// use mathverse_number_theory::divisor_sum;
/// assert_eq!(divisor_sum(6), 12); // 1+2+3+6
/// ```
#[must_use]
pub fn divisor_sum(n: u64) -> u64 {
    sigma_k(n, 1).unwrap_or(0)
}

/// Sum of the k-th powers of divisors of `n` (divisor function σ_k).
///
/// Returns `None` on overflow.
///
/// σ_k(n) = Σ_{d|n} d^k
///
/// ```
/// use mathverse_number_theory::sigma_k;
/// assert_eq!(sigma_k(12, 0), Some(6));  // number of divisors
/// assert_eq!(sigma_k(12, 1), Some(28)); // sum of divisors
/// assert_eq!(sigma_k(12, 2), Some(210)); // sum of squares
/// ```
#[must_use]
pub fn sigma_k(n: u64, k: u32) -> Option<u64> {
    if n == 0 {
        return Some(0);
    }
    if n == 1 {
        return Some(1);
    }
    let factors = prime_factors(n);
    let mut result = 1u64;
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut count = 0u32;
        while i < factors.len() && factors[i] == p {
            count += 1;
            i += 1;
        }
        let p_k = p.checked_pow(k)?;
        let mut term = 1u64;
        let mut p_pow = 1u64;
        for _ in 0..count {
            p_pow = p_pow.checked_mul(p_k)?;
            term = term.checked_add(p_pow)?;
        }
        result = result.checked_mul(term)?;
    }
    Some(result)
}

/// Möbius function μ(n):
/// - `1` if `n = 1`
/// - `0` if `n` has a squared prime factor
/// - `(-1)^k` where `k` is the number of distinct prime factors otherwise
///
/// ```
/// use mathverse_number_theory::mobius;
/// assert_eq!(mobius(1), 1);
/// assert_eq!(mobius(6), 1);   // 2×3, two distinct primes
/// assert_eq!(mobius(4), 0);   // 2², repeated
/// assert_eq!(mobius(30), -1); // 2×3×5, three distinct primes
/// ```
#[must_use]
pub fn mobius(n: u64) -> i64 {
    if n == 1 {
        return 1;
    }
    let factors = prime_factors(n);
    let mut i = 0;
    let mut distinct = 0u32;
    while i < factors.len() {
        let p = factors[i];
        let mut count = 0;
        while i < factors.len() && factors[i] == p {
            count += 1;
            i += 1;
        }
        if count > 1 {
            return 0;
        }
        distinct += 1;
    }
    if distinct.is_multiple_of(2) {
        1
    } else {
        -1
    }
}

/// Liouville function λ(n) = (-1)^Ω(n) where Ω(n) is the total number
/// of prime factors counted with multiplicity.
///
/// ```
/// use mathverse_number_theory::liouville;
/// assert_eq!(liouville(1), 1);
/// assert_eq!(liouville(6), 1);  // 2×3, Ω=2, even
/// assert_eq!(liouville(8), -1); // 2³, Ω=3, odd
/// assert_eq!(liouville(0), 0);
/// ```
#[must_use]
pub fn liouville(n: u64) -> i64 {
    if n == 0 {
        return 0;
    }
    let factors = prime_factors(n);
    if factors.len().is_multiple_of(2) {
        1
    } else {
        -1
    }
}

/// Returns `true` if the sum of proper divisors of `n` equals `n`.
///
/// ```
/// use mathverse_number_theory::is_perfect_number;
/// assert!(is_perfect_number(6));
/// assert!(is_perfect_number(28));
/// assert!(!is_perfect_number(12));
/// ```
#[must_use]
pub fn is_perfect_number(n: u64) -> bool {
    n >= 2 && sigma_k(n, 1).unwrap_or(0) == 2 * n
}

/// Returns `true` if the sum of proper divisors of `n` exceeds `n`.
///
/// ```
/// use mathverse_number_theory::is_abundant;
/// assert!(is_abundant(12));
/// assert!(!is_abundant(6));
/// ```
#[must_use]
pub fn is_abundant(n: u64) -> bool {
    n >= 2 && sigma_k(n, 1).unwrap_or(0) > 2 * n
}

/// Returns `true` if the sum of proper divisors of `n` is less than `n`.
///
/// ```
/// use mathverse_number_theory::is_deficient;
/// assert!(is_deficient(4));
/// assert!(is_deficient(1));
/// assert!(!is_deficient(6));
/// assert!(!is_deficient(12));
/// ```
#[must_use]
pub fn is_deficient(n: u64) -> bool {
    if n == 0 {
        return false;
    }
    sigma_k(n, 1).unwrap_or(u64::MAX) < 2 * n
}

/// Pollard's Rho algorithm: finds a single non-trivial factor of `n`,
/// or `None` if `n` is prime (or `n ≤ 1`).
///
/// Expected time O(n^(1/4)) — much faster than trial division for large composites.
/// Uses Floyd's cycle detection with a deterministic PRNG seeded from `n`,
/// so it's reproducible (no OS RNG needed).
///
/// ```
/// use mathverse_number_theory::pollard_rho;
/// // 91 = 7 * 13 — finds one factor
/// let f = pollard_rho(91).unwrap();
/// assert!(91 % f == 0 && f > 1 && f < 91);
/// ```
#[must_use]
pub fn pollard_rho(n: u64) -> Option<u64> {
    use crate::is_prime;
    if n <= 1 {
        return None;
    }
    if is_prime(n) {
        return None;
    }
    if n.is_multiple_of(2) {
        return Some(2);
    }
    let mut seed = n.wrapping_mul(0x5DEECE66D);
    'retry: loop {
        let c = 2 + (seed % (n - 3)) as u64;
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let f = |x: u64| ((x as u128 * x as u128 + c as u128) % n as u128) as u64;
        let mut x = f(2u64);
        let mut y = f(f(2u64));
        let mut d = 1u64;
        while d == 1 {
            let diff = (x as i128 - y as i128).unsigned_abs() as u64;
            d = crate::gcd(diff, n);
            if d == n {
                continue 'retry;
            }
            if d != 1 {
                return Some(d);
            }
            x = f(x);
            y = f(f(y));
        }
    }
}

/// Full prime factorization using trial division + Pollard's Rho.
///
/// Returns a sorted vector of prime factors (with multiplicity).
///
/// ```
/// use mathverse_number_theory::factorize;
/// assert_eq!(factorize(84), vec![2, 2, 3, 7]);
/// assert_eq!(factorize(13), vec![13]);
/// assert!(factorize(1).is_empty());
/// ```
#[must_use]
pub fn factorize(n: u64) -> Vec<u64> {
    if n <= 1 {
        return Vec::new();
    }
    if crate::is_prime(n) {
        return vec![n];
    }
    let mut result = Vec::new();
    factorize_inner(n, &mut result);
    result.sort_unstable();
    result
}

fn factorize_inner(n: u64, acc: &mut Vec<u64>) {
    if n == 1 {
        return;
    }
    if crate::is_prime(n) {
        acc.push(n);
        return;
    }
    if let Some(d) = pollard_rho(n) {
        if d > 1 && d < n {
            factorize_inner(d, acc);
            factorize_inner(n / d, acc);
        } else {
            trial_division_collect(n, acc);
        }
    } else {
        trial_division_collect(n, acc);
    }
}

fn trial_division_collect(mut n: u64, acc: &mut Vec<u64>) {
    while n.is_multiple_of(2) {
        acc.push(2);
        n /= 2;
    }
    let mut d = 3u64;
    while (d as u128) * (d as u128) <= n as u128 {
        while n.is_multiple_of(d) {
            acc.push(d);
            n /= d;
        }
        d += 2;
    }
    if n > 1 {
        acc.push(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factors_test() {
        assert_eq!(prime_factors(84), vec![2, 2, 3, 7]);
        assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(divisor_count(12), 6);
        assert_eq!(prime_factors(0), vec![0]);
        assert!(prime_factors(1).is_empty());
    }

    #[test]
    fn sigma_test() {
        assert_eq!(sigma_k(12, 0), Some(6));
        assert_eq!(sigma_k(12, 1), Some(28));
        assert_eq!(sigma_k(12, 2), Some(210));
        assert_eq!(sigma_k(6, 1), Some(12));
        assert_eq!(sigma_k(1, 1), Some(1));
        assert_eq!(sigma_k(1, 0), Some(1));
        // 2^64 overflows u64
        assert_eq!(sigma_k(2, 64), None);
    }

    #[test]
    fn mobius_test() {
        assert_eq!(mobius(1), 1);
        assert_eq!(mobius(6), 1);
        assert_eq!(mobius(4), 0);
        assert_eq!(mobius(30), -1);
    }

    #[test]
    fn liouville_test() {
        assert_eq!(liouville(1), 1);
        assert_eq!(liouville(6), 1);
        assert_eq!(liouville(8), -1);
        assert_eq!(liouville(0), 0);
    }

    #[test]
    fn perfect() {
        assert!(is_perfect_number(6));
        assert!(is_perfect_number(28));
        assert!(!is_perfect_number(12));
    }

    #[test]
    fn abundant_test() {
        assert!(is_abundant(12));
        assert!(is_abundant(18));
        assert!(!is_abundant(4));
        assert!(!is_abundant(6));
        assert!(!is_abundant(1));
    }

    #[test]
    fn deficient_test() {
        assert!(is_deficient(4));
        assert!(is_deficient(1));
        assert!(!is_deficient(6));
        assert!(!is_deficient(12));
    }

    #[test]
    fn pollard_rho_test() {
        let f = pollard_rho(15).unwrap();
        assert!(15 % f == 0 && f > 1 && f < 15);
        let f = pollard_rho(91).unwrap();
        assert!(91 % f == 0 && f > 1 && f < 91);
        assert_eq!(pollard_rho(2), None);
        assert_eq!(pollard_rho(1), None);
    }

    #[test]
    fn factorize_test() {
        assert_eq!(factorize(84), vec![2, 2, 3, 7]);
        assert_eq!(factorize(13), vec![13]);
        assert!(factorize(1).is_empty());
        assert_eq!(factorize(100), vec![2, 2, 5, 5]);
        assert_eq!(factorize(9999991), vec![9999991]); // large prime
    }

    #[test]
    fn divisor_sum_test() {
        assert_eq!(divisor_sum(6), 12);
        assert_eq!(divisor_sum(0), 0);
        assert_eq!(divisor_sum(1), 1);
    }
}
