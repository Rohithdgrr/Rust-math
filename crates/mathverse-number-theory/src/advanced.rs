pub fn next_prime(n: u64) -> u64 {
    let mut p = n + 1;
    while !crate::primes::is_prime(p) { p += 1; }
    p
}

pub fn prev_prime(n: u64) -> Option<u64> {
    if n <= 2 { return None; }
    let mut p = n - 1;
    while p >= 2 { if crate::primes::is_prime(p) { return Some(p); } p -= 1; }
    None
}

pub fn prime_factorization(n: u64) -> Vec<(u64, u32)> {
    let factors = crate::factorization::prime_factors(n);
    let mut result = Vec::new();
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut count = 0;
        while i < factors.len() && factors[i] == p { count += 1; i += 1; }
        result.push((p, count));
    }
    result
}

pub fn sum_of_divisors(n: u64) -> u64 { crate::factorization::divisor_sum(n) }

pub fn number_of_divisors(n: u64) -> u64 { crate::factorization::divisor_count(n) }

pub fn radical(n: u64) -> u64 {
    let factors = crate::factorization::prime_factors(n);
    factors.into_iter().collect::<std::collections::HashSet<_>>().into_iter().product()
}

pub fn kiuchi(n: u64) -> u64 {
    (1..=n).filter(|&k| crate::factorization::divisor_count(k) == 2).count() as u64
}

pub fn highly_composite(limit: u64) -> Vec<u64> {
    (1..=limit).filter(|&n| {
        let d = crate::factorization::divisor_count(n);
        (1..n).all(|k| crate::factorization::divisor_count(k) < d)
    }).collect()
}

pub fn perfect_power(n: u64) -> Option<(u64, u32)> {
    if n <= 3 { return None; }
    // Iterate over exponents e from 2..=63, find integer eth root.
    for e in 2u32..=63 {
        // Binary search for b such that b^e == n
        let mut lo = 2u64;
        let mut hi = {
            // upper bound: 2^(ceil(64/e))
            let mut h = 1u64;
            let bits = (64u64 + e as u64 - 1) / e as u64;
            for _ in 0..bits { h = h.saturating_mul(2); }
            h.min(n)
        };
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            // Compute mid^e, bail on overflow
            let mut pow = 1u64;
            let mut overflow = false;
            for _ in 0..e {
                match pow.checked_mul(mid) {
                    Some(v) => pow = v,
                    None => { overflow = true; break; }
                }
            }
            if overflow || pow > n { hi = mid - 1; }
            else if pow < n { lo = mid + 1; }
            else { return Some((mid, e)); }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_prev() {
        assert_eq!(next_prime(10), 11);
        assert_eq!(prev_prime(10), Some(7));
    }

    #[test]
    fn factorization() {
        assert_eq!(prime_factorization(84), vec![(2, 2), (3, 1), (7, 1)]);
    }

    #[test]
    fn radical_test() {
        assert_eq!(radical(12), 6);
    }

    #[test]
    fn perfect_power_test() {
        assert_eq!(perfect_power(8), Some((2, 3)));
        assert_eq!(perfect_power(9), Some((3, 2)));
        assert_eq!(perfect_power(7), None);
    }

    #[test]
    fn perfect_power_overflow() {
        // Should not panic even for very large n
        assert_eq!(perfect_power(u64::MAX), None);
        assert_eq!(perfect_power(u64::MAX - 1), None);
    }
}
