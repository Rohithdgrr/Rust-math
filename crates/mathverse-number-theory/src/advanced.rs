//! Advanced: next/prev prime, prime factorization, radical, highly composite, perfect powers.

/// Returns the smallest prime greater than `n`.
///
/// ```
/// use mathverse_number_theory::next_prime;
/// assert_eq!(next_prime(10), 11);
/// assert_eq!(next_prime(2), 3);
/// assert_eq!(next_prime(1), 2);
/// ```
#[must_use]
pub fn next_prime(n: u64) -> u64 {
    if n < 2 {
        return 2;
    }
    let mut candidate = n + 1;
    if candidate <= 2 {
        return 2;
    }
    if candidate % 2 == 0 {
        candidate += 1;
    }
    while !crate::is_prime(candidate) {
        candidate += 2;
    }
    candidate
}

/// Returns the largest prime strictly less than `n`, or `None` if `n ≤ 2`.
///
/// ```
/// use mathverse_number_theory::prev_prime;
/// assert_eq!(prev_prime(10), Some(7));
/// assert_eq!(prev_prime(2), None);
/// ```
#[must_use]
pub fn prev_prime(n: u64) -> Option<u64> {
    if n <= 2 {
        return None;
    }
    let mut candidate = n - 1;
    if candidate > 2 && candidate % 2 == 0 {
        candidate -= 1;
    }
    while candidate >= 2 {
        if crate::is_prime(candidate) {
            return Some(candidate);
        }
        if candidate <= 2 {
            break;
        }
        candidate -= 2;
    }
    Some(2)
}

/// Prime factorization of `n` as `(prime, exponent)` pairs in increasing order.
///
/// ```
/// use mathverse_number_theory::prime_factorization;
/// assert_eq!(prime_factorization(84), vec![(2, 2), (3, 1), (7, 1)]);
/// assert_eq!(prime_factorization(1), vec![]);
/// ```
#[must_use]
pub fn prime_factorization(n: u64) -> Vec<(u64, u32)> {
    if n <= 1 {
        return Vec::new();
    }
    let factors = crate::factorize(n);
    let mut result = Vec::new();
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut count = 0u32;
        while i < factors.len() && factors[i] == p {
            count += 1;
            i += 1;
        }
        result.push((p, count));
    }
    result
}

/// Sum of all positive divisors of `n` (σ₁ function).
/// Equivalent to `divisor_sum(n)`.
///
/// ```
/// use mathverse_number_theory::sum_of_divisors;
/// assert_eq!(sum_of_divisors(6), 12); // 1+2+3+6
/// ```
#[must_use]
pub fn sum_of_divisors(n: u64) -> u64 {
    crate::divisor_sum(n)
}

/// Number of positive divisors of `n` (d(n) or τ(n) function).
/// Equivalent to `divisor_count(n)`.
///
/// ```
/// use mathverse_number_theory::number_of_divisors;
/// assert_eq!(number_of_divisors(12), 6);
/// ```
#[must_use]
pub fn number_of_divisors(n: u64) -> u64 {
    crate::divisor_count(n)
}

/// Radical (square-free kernel) of `n`: the product of distinct prime factors.
///
/// ```
/// use mathverse_number_theory::radical;
/// assert_eq!(radical(12), 6);  // 2 × 3
/// assert_eq!(radical(7), 7);
/// assert_eq!(radical(1), 1);
/// ```
#[must_use]
pub fn radical(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let factors = crate::factorize(n);
    let distinct: std::collections::HashSet<u64> = factors.iter().copied().collect();
    distinct.iter().product()
}

/// Count of primes `≤ n` (prime-counting function π(n)).
///
/// ```
/// use mathverse_number_theory::kiuchi;
/// assert_eq!(kiuchi(10), 4); // 2, 3, 5, 7
/// ```
#[must_use]
pub fn kiuchi(n: u64) -> u64 {
    if n < 2 {
        return 0;
    }
    crate::sieve(n as usize).len() as u64
}

/// All highly composite numbers `≤ limit`: numbers `n` where `d(n) > d(k)`
/// for all `k < n` (where `d(k)` is the number of divisors).
///
/// Uses an efficient O(n log log n) sieve approach.
///
/// ```
/// use mathverse_number_theory::highly_composite;
/// let hc = highly_composite(100);
/// assert!(hc.contains(&1));
/// assert!(hc.contains(&2));
/// assert!(hc.contains(&4));
/// assert!(hc.contains(&6));
/// assert!(hc.contains(&12));
/// assert!(hc.contains(&60));
/// ```
#[must_use]
pub fn highly_composite(limit: u64) -> Vec<u64> {
    if limit == 0 {
        return Vec::new();
    }
    // Linear sieve for divisor counts
    let n = limit as usize + 1;
    let mut div_count = vec![0u64; n];
    let mut result = Vec::new();
    let mut max_div = 0u64;
    for i in 1..n {
        for j in (i..n).step_by(i) {
            div_count[j] += 1;
        }
        if div_count[i] > max_div {
            max_div = div_count[i];
            result.push(i as u64);
        }
    }
    result
}

/// Checks if `n` is a perfect power: `n = b^e` for some `b ≥ 2, e ≥ 2`.
///
/// Uses floating-point root estimation for speed, with exact verification.
/// Returns `None` if `n` is not a perfect power.
///
    /// ```
    /// use mathverse_number_theory::perfect_power;
    /// // 8 = 2³, 9 = 3², 64 = 8² (or 2⁶ or 4³)
    /// assert_eq!(perfect_power(8), Some((2, 3)));
    /// assert_eq!(perfect_power(9), Some((3, 2)));
    /// let r = perfect_power(64).unwrap();
    /// assert_eq!(r.0.pow(r.1), 64); // any valid (base, exponent) pair
    /// assert_eq!(perfect_power(7), None);
    /// assert_eq!(perfect_power(1), None);
    /// ```
#[must_use]
pub fn perfect_power(n: u64) -> Option<(u64, u32)> {
    if n <= 3 {
        return None;
    }
    for e in (2u32..=63).rev() {
        if (1u128 << e) > u128::from(n) {
            continue;
        }
        let root = nth_root_u64(n, e);
        for candidate in root.saturating_sub(1)..=root.saturating_add(1) {
            if candidate < 2 {
                continue;
            }
            if let Some(pow) = candidate.checked_pow(e) {
                if pow == n {
                    return Some((candidate, e));
                }
            }
        }
    }
    None
}

/// Integer `k`-th root: `floor(n^(1/k))`.
#[must_use]
fn nth_root_u64(n: u64, k: u32) -> u64 {
    if n <= 1 {
        return n;
    }
    if k == 1 {
        return n;
    }
    let mut lo = 1u64;
    let mut hi = n.min(1u64 << ((64 + k - 1) / k));
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        match mid.checked_pow(k) {
            Some(v) if v <= n => lo = mid,
            _ => hi = mid - 1,
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_prev() {
        assert_eq!(next_prime(10), 11);
        assert_eq!(next_prime(2), 3);
        assert_eq!(next_prime(1), 2);
        assert_eq!(prev_prime(10), Some(7));
        assert_eq!(prev_prime(2), None);
    }

    #[test]
    fn factorization() {
        assert_eq!(prime_factorization(84), vec![(2, 2), (3, 1), (7, 1)]);
        assert_eq!(prime_factorization(1), vec![]);
    }

    #[test]
    fn radical_test() {
        assert_eq!(radical(12), 6);
        assert_eq!(radical(1), 1);
        assert_eq!(radical(7), 7);
    }

    #[test]
    fn kiuchi_test() {
        assert_eq!(kiuchi(10), 4);
        assert_eq!(kiuchi(1), 0);
    }

    #[test]
    fn highly_composite_test() {
        let hc = highly_composite(100);
        assert!(hc.contains(&1));
        assert!(hc.contains(&2));
        assert!(hc.contains(&4));
        assert!(hc.contains(&6));
        assert!(hc.contains(&12));
        assert!(hc.contains(&60));
    }

    #[test]
    fn perfect_power_test() {
        assert_eq!(perfect_power(8), Some((2, 3)));
        assert_eq!(perfect_power(9), Some((3, 2)));
        // 64 = 2^6 = 4^3 = 8^2 — any valid representation is acceptable
        let r = perfect_power(64).unwrap();
        assert_eq!(r.0.pow(r.1), 64);
        assert_eq!(perfect_power(7), None);
        assert_eq!(perfect_power(1), None);
    }

    #[test]
    fn perfect_power_overflow() {
        assert_eq!(perfect_power(u64::MAX), None);
        assert_eq!(perfect_power(u64::MAX - 1), None);
    }

    #[test]
    fn sum_of_divisors_test() {
        assert_eq!(sum_of_divisors(6), 12);
        assert_eq!(sum_of_divisors(12), 28);
    }
}