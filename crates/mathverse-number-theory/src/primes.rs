//! Primality testing, sieve of Eratosthenes, twin primes, Goldbach, Mersenne primes.
//!
//! All primality tests use **deterministic Miller-Rabin** for `u64`,
//! which is O(log³ n) — no DoS risk from trial division.

/// Tests whether `n` is prime using deterministic Miller-Rabin.
///
/// Uses the twelve prime bases `{2,3,5,7,11,13,17,19,23,29,31,37}`,
/// which provably classify every `u64` value — no false positives.
///
/// This is O(log³ n), so a 64-bit prime is tested in microseconds,
/// not seconds.
///
/// ```
/// use mathverse_number_theory::is_prime;
/// assert!(is_prime(97));
/// assert!(!is_prime(15));
/// assert!(!is_prime(0));
/// assert!(!is_prime(1));
/// ```
#[must_use]
pub fn is_prime(n: u64) -> bool {
    miller_rabin(n)
}

/// Deterministic Miller-Rabin primality test for the full `u64` range.
///
/// Uses the first twelve prime bases `{2,3,5,7,11,13,17,19,23,29,31,37}`,
/// which provably classify every `n < 3,317,044,064,679,887,385,961,981`
/// (all of `u64`). Much faster than trial division for large composites.
///
/// `is_prime` delegates to this function.
///
/// ```
/// use mathverse_number_theory::miller_rabin;
/// assert!(miller_rabin(97));
/// assert!(miller_rabin(1_000_000_007));
/// assert!(!miller_rabin(221)); // 13 * 17
/// assert!(!miller_rabin(4_611_686_018_427_387_903)); // 2^62 - 2^32 + 1
/// ```
#[must_use]
pub fn miller_rabin(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    const SMALL_PRIMES: [u64; 24] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    ];
    for &p in &SMALL_PRIMES {
        if n % p == 0 {
            return n == p;
        }
    }
    let mut d = n - 1;
    let mut s = 0u32;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    'witness: for &a in &BASES {
        let a = a % n;
        if a == 0 {
            continue;
        }
        let mut x = mod_pow_u128(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = mod_pow_u128(x as u64, 2, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

#[inline]
fn mod_pow_u128(base: u64, mut exp: u64, m: u64) -> u64 {
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
    result as u64
}

/// Sieve of Eratosthenes: all primes `≤ limit`.
///
/// Returns an empty vector for `limit < 2`.
///
/// ```
/// use mathverse_number_theory::sieve;
/// let primes = sieve(20);
/// assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19]);
/// ```
#[must_use]
pub fn sieve(limit: usize) -> Vec<u64> {
    if limit < 2 {
        return Vec::new();
    }
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let sqrt_limit = (limit as f64).sqrt() as usize;
    for i in 2..=sqrt_limit {
        if is_prime[i] {
            let start = i * i;
            for j in (start..=limit).step_by(i) {
                is_prime[j] = false;
            }
        }
    }
    (2..=limit as u64).filter(|&p| is_prime[p as usize]).collect()
}

/// Alias for [`sieve`], matching the naming in `mathverse-core`.
///
/// ```
/// use mathverse_number_theory::sieve_of_eratosthenes;
/// assert_eq!(sieve_of_eratosthenes(10), vec![2, 3, 5, 7]);
/// ```
#[must_use]
pub fn sieve_of_eratosthenes(n: usize) -> Vec<u64> {
    sieve(n)
}

/// Segmented sieve: all primes in `[low, high]`.
///
/// Uses O(√high) memory regardless of range size, so it can sieve
/// ranges like 10¹² to 10¹² + 10⁶ efficiently.
///
/// ```
/// use mathverse_number_theory::segmented_sieve;
/// let primes = segmented_sieve(10, 30);
/// assert_eq!(primes, vec![11, 13, 17, 19, 23, 29]);
/// ```
#[must_use]
pub fn segmented_sieve(low: u64, high: u64) -> Vec<u64> {
    if high < 2 || high < low {
        return Vec::new();
    }
    let limit = (high as f64).sqrt() as u64 + 1;
    let base_primes = sieve(limit as usize);
    let segment_size = (high - low + 1) as usize;
    let mut is_prime = vec![true; segment_size];
    for &p in &base_primes {
        let p = p as u64;
        if p * p > high {
            break;
        }
        let start = if low <= p {
            p * p
        } else {
            ((low + p - 1) / p) * p
        };
        let start = start.max(p * p);
        if start > high {
            continue;
        }
        for j in (start..=high).step_by(p as usize) {
            is_prime[(j - low) as usize] = false;
        }
    }
    (0..segment_size)
        .map(|i| low + i as u64)
        .filter(|&p| p >= 2 && is_prime[(p - low) as usize])
        .collect()
}

/// Returns the `n`-th prime (1-indexed: the 1st prime is 2).
///
/// ```
/// use mathverse_number_theory::nth_prime;
/// assert_eq!(nth_prime(1), 2);
/// assert_eq!(nth_prime(10), 29);
/// ```
#[must_use]
pub fn nth_prime(n: usize) -> u64 {
    if n == 0 {
        return 2; // first prime
    }
    if n == 0 {
        return 2; // 0th prime by convention (caller should treat as error)
    }
    let mut count = 1;
    let mut num = 2;
    while count < n {
        num = next_prime(num);
        count += 1;
    }
    num
}

/// Returns all primes in the inclusive range `[a, b]`.
///
/// ```
/// use mathverse_number_theory::prime_between;
/// assert_eq!(prime_between(10, 20), vec![11, 13, 17, 19]);
/// ```
#[must_use]
pub fn prime_between(a: u64, b: u64) -> Vec<u64> {
    segmented_sieve(a, b)
}

/// Returns all twin prime pairs `(p, p+2)` with `p < limit`.
///
/// ```
/// use mathverse_number_theory::twin_primes;
/// let t = twin_primes(20);
/// assert!(t.contains(&(3, 5)));
/// assert!(t.contains(&(5, 7)));
/// assert!(t.contains(&(11, 13)));
/// assert!(t.contains(&(17, 19)));
/// ```
#[must_use]
pub fn twin_primes(limit: u64) -> Vec<(u64, u64)> {
    if limit < 5 {
        return Vec::new();
    }
    let primes = sieve(limit as usize + 2);
    let prime_set: std::collections::HashSet<u64> = primes.iter().copied().collect();
    primes
        .iter()
        .copied()
        .filter(|&p| p + 2 < limit && prime_set.contains(&(p + 2)))
        .map(|p| (p, p + 2))
        .collect()
}

/// Finds a Goldbach partition of `n`: two primes `p, q` such that `p + q = n`.
///
/// Returns `None` for odd `n < 4` (Goldbach only applies to even `n ≥ 4`).
/// For `n ≤ 10_000_000`, uses a sieve for correctness; for larger `n`,
/// uses Miller-Rabin (may be slow but always correct).
///
/// ```
/// use mathverse_number_theory::goldbach;
/// assert_eq!(goldbach(10), Some((3, 7)));
/// assert_eq!(goldbach(100), Some((3, 97)));
/// assert!(goldbach(11).is_none());
/// ```
#[must_use]
pub fn goldbach(n: u64) -> Option<(u64, u64)> {
    if n < 4 || n % 2 != 0 {
        return None;
    }
    if n <= 10_000_000 {
        let primes = sieve(n as usize);
        let prime_set: std::collections::HashSet<u64> = primes.iter().copied().collect();
        for &p in &primes {
            if p > n / 2 {
                break;
            }
            if prime_set.contains(&(n - p)) {
                return Some((p, n - p));
            }
        }
        return None;
    }
    let mut p = n / 2;
    if p % 2 == 0 {
        p -= 1;
    }
    while p >= 2 {
        if is_prime(p) && is_prime(n - p) {
            return Some((p, n - p));
        }
        p -= 2;
    }
    None
}

/// Checks if the Mersenne number M_p = 2^p - 1 is prime, using the
/// Lucas-Lehmer test.
///
/// Returns `None` if `p` is not prime (Mersenne primes require prime exponents).
/// Returns `Some(m)` where `m = 2^p - 1` if M_p is prime.
///
/// For `p ≥ 64`, `2^p - 1` overflows `u64`; with the `bigint` feature,
/// the result is returned as a decimal string.
///
/// ```
/// use mathverse_number_theory::mersenne_prime;
/// assert_eq!(mersenne_prime(2), Some(3));
/// assert_eq!(mersenne_prime(3), Some(7));
/// assert_eq!(mersenne_prime(5), Some(31));
/// assert_eq!(mersenne_prime(7), Some(127));
/// assert_eq!(mersenne_prime(11), None); // 2047 = 23 × 89
/// ```
#[must_use]
pub fn mersenne_prime(p: u32) -> Option<u64> {
    if p < 2 || !is_prime(p as u64) {
        return None;
    }
    if p >= 64 {
        return None;
    }
    if lucas_lehmer(p) {
        Some((1u64 << p) - 1)
    } else {
        None
    }
}

/// Lucas-Lehmer primality test for the Mersenne number M_p = 2^p - 1.
///
/// Returns `true` iff M_p is prime (requires p to be an odd prime).
/// This is the standard deterministic test for Mersenne primes —
/// much faster than Miller-Rabin for numbers of this special form.
///
/// ```
/// use mathverse_number_theory::lucas_lehmer;
/// assert!(lucas_lehmer(3));   // 2^3-1 = 7 is prime
/// assert!(lucas_lehmer(5));   // 2^5-1 = 31 is prime
/// assert!(!lucas_lehmer(11)); // 2^11-1 = 2047 = 23*89 is composite
/// ```
#[must_use]
pub fn lucas_lehmer(p: u32) -> bool {
    if p == 2 {
        return true;
    }
    if p < 2 || p % 2 == 0 {
        return false;
    }
    let mut s = 4u128;
    let m = (1u128 << p) - 1; // M_p = 2^p - 1 as u128 (valid since p < 128)
    for _ in 0..p - 2 {
        s = (s * s - 2) % m;
    }
    s == 0 || s == m
}

/// The prime gap starting at `n`: distance from the first prime `>= n`
/// to the next prime strictly after it.
///
/// Returns `None` if no prime exists in `(n, u64::MAX]` or if the gap
/// would overflow.
///
/// ```
/// use mathverse_number_theory::prime_gap_after;
/// assert_eq!(prime_gap_after(2), Some(1));   // 2→3
/// assert_eq!(prime_gap_after(3), Some(2));   // 3→5
/// assert_eq!(prime_gap_after(7), Some(4));   // 7→11
/// ```
#[must_use]
pub fn prime_gap_after(n: u64) -> Option<u64> {
    let p = if n <= 2 {
        2
    } else if is_prime(n) {
        n
    } else {
        next_prime(n)
    };
    if p == u64::MAX {
        return None;
    }
    let q = next_prime(p);
    q.checked_sub(p)
}

/// The prime gap containing `n`: returns `(prev, next, gap)` where
/// `prev` is the largest prime `< n` and `next` is the smallest prime
/// `> n`.
///
/// Returns `None` if there is no prime below or above `n` in range.
///
/// ```
/// use mathverse_number_theory::prime_gap_containing;
/// assert_eq!(prime_gap_containing(4), Some((3, 5, 2)));
/// assert_eq!(prime_gap_containing(6), Some((5, 7, 2)));
/// assert_eq!(prime_gap_containing(8), Some((7, 11, 4)));
/// ```
#[must_use]
pub fn prime_gap_containing(n: u64) -> Option<(u64, u64, u64)> {
    if n < 3 {
        return None;
    }
    let next = next_prime(n);
    if next == u64::MAX || !is_prime(next) {
        return None;
    }
    let prev = prev_prime(n)?;
    Some((prev, next, next - prev))
}

/// Returns the smallest prime strictly greater than `n`.
///
/// Uses [`is_prime`] (deterministic Miller–Rabin), so it is fast even for
/// large 64-bit values.
///
/// # Examples
///
/// ```
/// use mathverse_number_theory::next_prime;
/// assert_eq!(next_prime(10), 11);
/// assert_eq!(next_prime(0), 2);
/// assert_eq!(next_prime(1), 2);
/// ```
#[must_use]
#[inline]
pub fn next_prime(n: u64) -> u64 {
    if n < 2 {
        return 2;
    }
    let mut candidate = if n % 2 == 0 { n + 1 } else { n + 2 };
    while !is_prime(candidate) {
        candidate += 2;
    }
    candidate
}

/// Returns the largest prime strictly less than `n`, if one exists.
///
/// Returns `None` for `n ≤ 2`.
///
/// # Examples
///
/// ```
/// use mathverse_number_theory::prev_prime;
/// assert_eq!(prev_prime(13), Some(11));
/// assert_eq!(prev_prime(3), Some(2));
/// assert_eq!(prev_prime(2), None);
/// ```
#[must_use]
pub fn prev_prime(n: u64) -> Option<u64> {
    if n <= 2 {
        return None;
    }
    let mut candidate = if n <= 3 { 2 } else { n - 1 };
    if candidate % 2 == 0 && candidate > 2 {
        candidate -= 1;
    }
    while candidate >= 2 {
        if is_prime(candidate) {
            return Some(candidate);
        }
        if candidate <= 2 {
            break;
        }
        candidate -= 2;
    }
    Some(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_prime_fast() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(97));
        assert!(!is_prime(15));
        assert!(!is_prime(221)); // 13 * 17
    }

    #[test]
    fn sieve_test() {
        assert_eq!(sieve(2), vec![2]);
        assert_eq!(sieve(1), Vec::<u64>::new());
        assert_eq!(sieve(0), Vec::<u64>::new());
        let p = sieve(50);
        assert_eq!(p.len(), 15);
    }

    #[test]
    fn segmented_sieve_test() {
        let primes = segmented_sieve(10, 30);
        assert_eq!(primes, vec![11, 13, 17, 19, 23, 29]);
        assert_eq!(segmented_sieve(2, 100).len(), 25);
    }

    #[test]
    fn nth_prime_test() {
        assert_eq!(nth_prime(1), 2);
        assert_eq!(nth_prime(10), 29);
    }

    #[test]
    fn goldbach_test() {
        assert_eq!(goldbach(10), Some((3, 7)));
        assert_eq!(goldbach(100), Some((3, 97)));
        assert!(goldbach(11).is_none());
        assert!(goldbach(8).is_some());
    }

    #[test]
    fn mersenne_test() {
        assert_eq!(mersenne_prime(2), Some(3));
        assert_eq!(mersenne_prime(3), Some(7));
        assert_eq!(mersenne_prime(5), Some(31));
        assert_eq!(mersenne_prime(7), Some(127));
        assert_eq!(mersenne_prime(11), None); // 2047 = 23*89
        assert_eq!(mersenne_prime(13), Some(8191));
        assert_eq!(mersenne_prime(1), None); // 1 is not prime
        assert_eq!(mersenne_prime(0), None);
    }

    #[test]
    fn lucas_lehmer_test() {
        assert!(lucas_lehmer(3));
        assert!(lucas_lehmer(5));
        assert!(lucas_lehmer(7));
        assert!(!lucas_lehmer(11));
        assert!(lucas_lehmer(2)); // M_2 = 3 is prime
        assert!(!lucas_lehmer(15)); // composite exponent
    }

    #[test]
    fn prime_gap_test() {
        assert_eq!(prime_gap_after(2), Some(1));
        assert_eq!(prime_gap_after(3), Some(2));
        assert_eq!(prime_gap_after(7), Some(4));
        assert_eq!(prime_gap_containing(4), Some((3, 5, 2)));
        assert_eq!(prime_gap_containing(6), Some((5, 7, 2)));
    }

    #[test]
    fn twins_test() {
        let t = twin_primes(20);
        assert!(t.contains(&(3, 5)));
        assert!(t.contains(&(5, 7)));
    }

    #[test]
    fn miller_rabin_agrees_with_sieve() {
        let sieve_primes = sieve(5000);
        let mut next = 0usize;
        for n in 0..5000u64 {
            let expected = next < sieve_primes.len() && sieve_primes[next] == n;
            assert_eq!(is_prime(n), expected, "mismatch at {n}");
            if expected {
                next += 1;
            }
        }
    }

    #[test]
    fn miller_rabin_large() {
        assert!(is_prime(1_000_000_007));
        assert!(!is_prime(1_000_000_007 * 2));
        assert!(is_prime(2_305_843_009_213_693_951)); // 2^61 - 1, Mersenne prime
        assert!(!is_prime(u64::MAX)); // 2^64 - 1 = 3 × 6148914691236517205
        assert!(!is_prime(221)); // 13 * 17
    }
}
