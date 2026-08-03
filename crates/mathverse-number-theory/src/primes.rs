//! Primality testing, sieve of Eratosthenes, twin primes, Goldbach, Mersenne primes.

/// Tests whether `n` is prime using trial division.
pub fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    let mut i = 5;
    while i * i <= n { if n % i == 0 || n % (i + 2) == 0 { return false; } i += 6; }
    true
}

pub fn sieve(limit: usize) -> Vec<u64> {
    if limit < 2 { return Vec::new(); }
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    let mut i = 2;
    while i * i <= limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit { is_prime[j] = false; j += i; }
        }
        i += 1;
    }
    (2..=limit as u64).filter(|&p| is_prime[p as usize]).collect()
}

pub fn nth_prime(n: usize) -> u64 {
    let mut count = 0;
    let mut num = 1;
    while count < n { num += 1; if is_prime(num) { count += 1; } }
    num
}

pub fn prime_between(a: u64, b: u64) -> Vec<u64> {
    (a..=b).filter(|&p| is_prime(p)).collect()
}

pub fn twin_primes(limit: u64) -> Vec<(u64, u64)> {
    (3..limit).filter(|&p| is_prime(p) && is_prime(p + 2)).map(|p| (p, p + 2)).collect()
}

pub fn goldbach(n: u64) -> Option<(u64, u64)> {
    if n < 4 || n % 2 != 0 { return None; }
    (2..=n/2).find(|&p| is_prime(p) && is_prime(n - p)).map(|p| (p, n - p))
}

pub fn mersenne_prime(p: u64) -> Option<u64> {
    if !is_prime(p) { return None; }
    if p >= 64 { return None; } // 2^p - 1 overflows u64
    let m = (1u64 << p) - 1;
    if is_prime(m) { Some(m) } else { None }
}

pub fn prime_gap(n: u64) -> u64 {
    let mut p = n;
    while !is_prime(p) { p += 1; }
    let mut q = p + 1;
    while !is_prime(q) { q += 1; }
    q - p
}

/// Deterministic Miller-Rabin primality test for the full `u64` range.
///
/// Uses the [first twelve prime bases](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test)
/// `{2,3,5,7,11,13,17,19,23,29,31,37}`, which provably classify every
/// `n < 3,317,044,064,679,887,385,961,981` (all of `u64`). Much faster than
/// trial division for large composites.
///
/// ```
/// use mathverse_number_theory::miller_rabin;
/// assert!(miller_rabin(97));
/// assert!(miller_rabin(1_000_000_007));
/// assert!(!miller_rabin(221)); // 13 * 17
/// assert!(!miller_rabin(4_611_686_018_427_387_903)); // 2^62 - 2^32 + 1 composite
/// ```
pub fn miller_rabin(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    const SMALL_PRIMES: [u64; 24] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
        41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    ];
    for &p in &SMALL_PRIMES {
        if n % p == 0 {
            return n == p;
        }
    }
    // n - 1 = d * 2^s with d odd.
    let mut d = n - 1;
    let mut s = 0;
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
            x = (x as u128 * x as u128 % n as u128) as u64;
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes() {
        assert!(is_prime(2));
        assert!(is_prime(97));
        assert!(!is_prime(15));
        let p = sieve(50);
        assert_eq!(p.len(), 15);
    }

    #[test]
    fn nth() {
        assert_eq!(nth_prime(1), 2);
        assert_eq!(nth_prime(10), 29);
    }

    #[test]
    fn goldbach_test() {
        assert_eq!(goldbach(10), Some((3, 7)));
        assert!(goldbach(11).is_none());
    }

    #[test]
    fn twins() {
        let t = twin_primes(20);
        assert!(t.contains(&(3, 5)));
        assert!(t.contains(&(5, 7)));
    }

    #[test]
    fn mersenne_test() {
        assert_eq!(mersenne_prime(2), Some(3));       // 2^2-1=3
        assert_eq!(mersenne_prime(3), Some(7));       // 2^3-1=7
        assert_eq!(mersenne_prime(5), Some(31));      // 2^5-1=31
        assert_eq!(mersenne_prime(7), Some(127));     // 2^7-1=127
        assert_eq!(mersenne_prime(11), None);         // 2047=23*89
        assert_eq!(mersenne_prime(64), None);         // overflow guard
        assert_eq!(mersenne_prime(1), None);          // 1 is not prime
    }

    #[test]
    fn miller_rabin_agrees_with_sieve() {
        let sieve_primes = sieve(5000);
        let mut next = 0usize;
        for n in 0..5000u64 {
            let expected = next < sieve_primes.len() && sieve_primes[next] == n;
            assert_eq!(miller_rabin(n), expected, "mismatch at {n}");
            if expected {
                next += 1;
            }
        }
    }

    #[test]
    fn miller_rabin_large() {
        assert!(miller_rabin(1_000_000_007));
        assert!(!miller_rabin(1_000_000_007 * 2));
        assert!(miller_rabin(2_305_843_009_213_693_951)); // 2^61 - 1, Mersenne prime
        assert!(!miller_rabin(u64::MAX));                // 2^64 - 1 = 3 × 6148914691236517205
        assert!(!miller_rabin(221));                      // 13 * 17
        assert!(!miller_rabin(0));
        assert!(!miller_rabin(1));
    }
}
