//! Euler's totient, Carmichael function, primitive roots, multiplicative order.

/// Euler's totient function φ(n): count of integers in `[1, n]` coprime to `n`.
///
/// ```
/// use mathverse_number_theory::euler_totient;
/// assert_eq!(euler_totient(10), 4);
/// assert_eq!(euler_totient(97), 96);
/// assert_eq!(euler_totient(36), 12);
/// assert_eq!(euler_totient(1), 1);
/// ```
#[must_use]
pub fn euler_totient(mut n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    let mut d = 2u64;
    while (d as u128) * (d as u128) <= n as u128 {
        if n.is_multiple_of(d) {
            while n.is_multiple_of(d) {
                n /= d;
            }
            result -= result / d;
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if n > 1 {
        result -= result / n;
    }
    result
}

/// Precompute φ(0), φ(1), ..., φ(limit) as a vector.
///
/// Uses a sieve-like linear-time algorithm.
///
/// ```
/// use mathverse_number_theory::euler_totient_sum;
/// let phis = euler_totient_sum(10);
/// assert_eq!(phis[7], 6);
/// ```
#[must_use]
pub fn euler_totient_sum(limit: u64) -> Vec<u64> {
    if limit == 0 {
        return vec![0];
    }
    let n = limit as usize + 1;
    let mut phi: Vec<u64> = (0..=limit).collect();
    for i in 2..n {
        if phi[i] == i as u64 {
            // i is prime
            for j in (i..n).step_by(i) {
                phi[j] -= phi[j] / i as u64;
            }
        }
    }
    phi
}

/// Carmichael's lambda function λ(n): the smallest positive integer `k`
/// such that `a^k ≡ 1 (mod n)` for all `a` coprime to `n`.
///
/// ```
/// use mathverse_number_theory::carmichael;
/// assert_eq!(carmichael(1), 1);
/// assert_eq!(carmichael(8), 2);
/// assert_eq!(carmichael(7), 6);
/// ```
#[must_use]
pub fn carmichael(n: u64) -> u64 {
    if n <= 2 {
        return n;
    }
    let factors = crate::prime_factors(n);
    let mut lambda = 1u64;
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut pk = 1u64;
        while i < factors.len() && factors[i] == p {
            pk *= p;
            i += 1;
        }
        let cp = if p == 2 && pk >= 8 {
            pk / 4
        } else {
            pk - pk / p
        };
        lambda = crate::lcm(lambda, cp);
    }
    lambda
}

/// Returns `true` if `a` and `b` are coprime (gcd = 1).
///
/// ```
/// use mathverse_number_theory::is_coprime;
/// assert!(is_coprime(8, 9));
/// assert!(!is_coprime(8, 12));
/// ```
#[must_use]
#[inline]
pub fn is_coprime(a: u64, b: u64) -> bool {
    crate::gcd(a, b) == 1
}

/// Returns all integers in `[1, n]` that are coprime to `n`.
///
/// ```
/// use mathverse_number_theory::coprimes_up_to;
/// assert_eq!(coprimes_up_to(6), vec![1, 5]);
/// ```
#[must_use]
pub fn coprimes_up_to(n: u64) -> Vec<u64> {
    (1..=n).filter(|&k| is_coprime(k, n)).collect()
}

/// Checks if a primitive root modulo `n` can exist.
///
/// Primitive roots exist only for `n ∈ {1, 2, 4, p^k, 2·p^k}` where `p`
/// is an odd prime and `k ≥ 1`.
fn has_primitive_root(n: u64) -> bool {
    if n <= 4 {
        return n >= 1;
    }
    // Remove factors of 2
    let mut m = n;
    while m.is_multiple_of(2) {
        m /= 2;
    }
    if m == 1 {
        // n = 2^k with k >= 3: no primitive root
        return false;
    }
    if m != n && m != n / 2 {
        // n has factor 2^k with k >= 2 but also other factors: no primitive root
        return false;
    }
    // m must be 1, or n/2, or n itself must be a prime power of an odd prime
    if !is_prime_power(m) {
        return false;
    }
    true
}

fn is_prime_power(n: u64) -> bool {
    if n <= 1 {
        return false;
    }
    let factors = crate::prime_factors(n);
    factors.iter().all(|&p| p == factors[0])
}

/// Finds a primitive root modulo `n`.
///
/// Returns `None` if no primitive root exists (i.e., `n` is not
/// `2, 4, p^k,` or `2·p^k` for an odd prime `p`).
///
/// Tests small candidates first (2 through 100), then samples randomly
/// for harder cases.
///
/// ```
/// use mathverse_number_theory::primitive_root;
/// assert!(primitive_root(7).is_some());
/// assert!(primitive_root(97).is_some());
/// assert_eq!(primitive_root(8), None); // 2^3: no primitive root
/// ```
#[must_use]
pub fn primitive_root(n: u64) -> Option<u64> {
    if n <= 1 {
        return None;
    }
    if n == 2 {
        return Some(1);
    }
    if !has_primitive_root(n) {
        return None;
    }
    let phi = euler_totient(n);
    let unique_factors = {
        let factors = crate::prime_factors(phi);
        factors.iter().copied().collect::<std::collections::HashSet<_>>()
    };
    for g in 2..=100.min(n - 1) {
        if !is_coprime(g, n) {
            continue;
        }
        if is_primitive_root(g, n, phi, &unique_factors) {
            return Some(g);
        }
    }
    // Deterministic pseudo-random search for larger candidates.
    let mut state = n.wrapping_mul(0x5DEECE66D);
    for _ in 0..10_000 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let g = 101 + (state % (n - 101));
        if !is_coprime(g, n) {
            continue;
        }
        if is_primitive_root(g, n, phi, &unique_factors) {
            return Some(g);
        }
    }
    None
}

fn is_primitive_root(g: u64, n: u64, phi: u64, unique_factors: &std::collections::HashSet<u64>) -> bool {
    for &p in unique_factors {
        if crate::modular::mod_pow_unchecked(g, phi / p, n) == 1 {
            return false;
        }
    }
    true
}

/// Multiplicative order of `a` modulo `n`: the smallest `k > 0` such that
/// `a^k ≡ 1 (mod n)`.
///
/// Returns `None` if `a` and `n` are not coprime.
///
/// ```
/// use mathverse_number_theory::multiplicative_order;
/// assert_eq!(multiplicative_order(2, 7), Some(3));
/// assert_eq!(multiplicative_order(3, 10), Some(4));
/// assert_eq!(multiplicative_order(2, 4), None); // gcd(2,4) = 2
/// ```
#[must_use]
pub fn multiplicative_order(a: u64, n: u64) -> Option<u64> {
    if !is_coprime(a, n) {
        return None;
    }
    if n == 1 {
        return Some(1);
    }
    let phi = euler_totient(n);
    let mut order = phi;
    let factors = crate::prime_factors(phi);
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        while i + 1 < factors.len() && factors[i + 1] == p {
            i += 1;
        }
        i += 1;
        while order % p == 0 && crate::modular::mod_pow_unchecked(a, order / p, n) == 1 {
            order /= p;
        }
    }
    Some(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn totient() {
        assert_eq!(euler_totient(1), 1);
        assert_eq!(euler_totient(10), 4);
        assert_eq!(euler_totient(97), 96);
        assert_eq!(euler_totient(36), 12);
    }

    #[test]
    fn totient_sum() {
        let phis = euler_totient_sum(20);
        assert_eq!(phis[1], 1);
        assert_eq!(phis[7], 6);
        assert_eq!(phis[11], 10);
    }

    #[test]
    fn carmichael_test() {
        assert_eq!(carmichael(1), 1);
        assert_eq!(carmichael(2), 2);
        assert_eq!(carmichael(8), 2);
        assert_eq!(carmichael(7), 6);
        assert_eq!(carmichael(9), 6);
    }

    #[test]
    fn coprime_test() {
        assert!(is_coprime(8, 9));
        assert!(!is_coprime(8, 12));
        assert!(is_coprime(1, 1));
    }

    #[test]
    fn primitive_test() {
        assert!(primitive_root(7).is_some());
        assert!(primitive_root(97).is_some());
        assert_eq!(primitive_root(8), None); // 2^3
    }

    #[test]
    fn primitive_root_known() {
        // 2 is a primitive root mod 7
        let g = primitive_root(7).unwrap();
        let phi = euler_totient(7);
        assert_eq!(crate::modular::mod_pow_unchecked(g, phi, 7), 1);
    }

    #[test]
    fn order_test() {
        assert_eq!(multiplicative_order(2, 7), Some(3));
        assert_eq!(multiplicative_order(3, 10), Some(4));
        assert_eq!(multiplicative_order(2, 4), None);
        assert_eq!(multiplicative_order(1, 5), Some(1));
    }

    #[test]
    fn has_primitive_root_test() {
        assert!(has_primitive_root(1));
        assert!(has_primitive_root(2));
        assert!(has_primitive_root(4));
        assert!(has_primitive_root(7));
        assert!(has_primitive_root(9));
        assert!(has_primitive_root(18)); // 2 * 3^2
        assert!(!has_primitive_root(8)); // 2^3
        assert!(!has_primitive_root(12)); // 4 * 3
    }
}
