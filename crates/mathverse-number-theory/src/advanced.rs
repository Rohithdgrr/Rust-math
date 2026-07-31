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
    for b in 2..=((n as f64).sqrt() as u64) {
        let mut p = b;
        let mut e = 1u32;
        while p < n { p *= b; e += 1; }
        if p == n { return Some((b, e)); }
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
}
