//! Compatibility layer mirroring common functions from `sympy.ntheory`.
//!
//! This module provides function names and return shapes that match
//! the Python `sympy` library's `ntheory` module, making it easier to
//! port code from SymPy to Rust.
//!
//! ## Mapping
//!
//! | SymPy | mathverse-number-theory |
//! |-------|------------------------|
//! | `sympy.isprime(n)` | [`isprime`] |
//! | `sympy.factorint(n)` | [`factorint`] |
//! | `sympy.divisors(n)` | [`divisors`] |
//! | `sympy.divisor_count(n)` | [`divisor_count`] |
//! | `sympy.totient(n)` | [`totient`] |
//! | `sympy.primitive_root(n)` | [`primitive_root`] |
//! | `sympy.n_order(a, n)` | [`n_order`] |

use std::collections::BTreeMap;

/// Deterministic primality test (alias for [`crate::is_prime`]).
///
/// Equivalent to `sympy.isprime(n)`.
///
/// ```
/// use mathverse_number_theory::sympy_compat::isprime;
/// assert!(isprime(97));
/// assert!(!isprime(15));
/// ```
#[must_use]
pub fn isprime(n: u64) -> bool {
    crate::is_prime(n)
}

/// Prime factorization as a map of `{prime: exponent}`.
///
/// Equivalent to `sympy.factorint(n)`.
///
/// ```
/// use mathverse_number_theory::sympy_compat::factorint;
/// let f = factorint(84);
/// assert_eq!(f.get(&2), Some(&2));
/// assert_eq!(f.get(&3), Some(&1));
/// assert_eq!(f.get(&7), Some(&1));
/// ```
#[must_use]
pub fn factorint(n: u64) -> BTreeMap<u64, u32> {
    let factors = crate::prime_factors(n);
    let mut result = BTreeMap::new();
    for p in factors {
        *result.entry(p).or_insert(0) += 1;
    }
    result
}

/// All positive divisors of `n`, sorted.
///
/// Equivalent to `sympy.divisors(n)`.
///
/// ```
/// use mathverse_number_theory::sympy_compat::divisors;
/// assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
/// ```
#[must_use]
pub fn divisors(n: u64) -> Vec<u64> {
    crate::divisors(n)
}

/// Number of divisors of `n`.
///
/// Equivalent to `sympy.divisor_count(n)`.
///
/// ```
/// use mathverse_number_theory::sympy_compat::divisor_count;
/// assert_eq!(divisor_count(12), 6);
/// ```
#[must_use]
pub fn divisor_count(n: u64) -> u64 {
    crate::divisor_count(n)
}

/// Euler's totient function.
///
/// Equivalent to `sympy.totient(n)`.
///
/// ```
/// use mathverse_number_theory::sympy_compat::totient;
/// assert_eq!(totient(10), 4);
/// ```
#[must_use]
pub fn totient(n: u64) -> u64 {
    crate::euler_totient(n)
}

/// Finds a primitive root modulo `n`.
///
/// Equivalent to `sympy.primitive_root(n)`.
///
/// ```
/// use mathverse_number_theory::sympy_compat::primitive_root;
/// assert!(primitive_root(7).is_some());
/// assert_eq!(primitive_root(8), None);
/// ```
#[must_use]
pub fn primitive_root(n: u64) -> Option<u64> {
    crate::primitive_root(n)
}

/// Multiplicative order of `a` modulo `n`.
///
/// Equivalent to `sympy.n_order(a, n)`.
///
/// ```
/// use mathverse_number_theory::sympy_compat::n_order;
/// assert_eq!(n_order(2, 7), Some(3));
/// ```
#[must_use]
pub fn n_order(a: u64, n: u64) -> Option<u64> {
    crate::multiplicative_order(a, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isprime_test() {
        assert!(isprime(2));
        assert!(isprime(97));
        assert!(!isprime(1));
        assert!(!isprime(0));
        assert!(!isprime(221));
    }

    #[test]
    fn factorint_test() {
        let f = factorint(84);
        assert_eq!(f.get(&2), Some(&2));
        assert_eq!(f.get(&3), Some(&1));
        assert_eq!(f.get(&7), Some(&1));

        let f = factorint(1);
        assert!(f.is_empty());
    }

    #[test]
    fn divisors_test() {
        assert_eq!(divisors(12), vec![1, 2, 3, 4, 6, 12]);
        assert_eq!(divisors(1), vec![1]);
    }

    #[test]
    fn totient_test() {
        assert_eq!(totient(10), 4);
        assert_eq!(totient(97), 96);
    }

    #[test]
    fn n_order_test() {
        assert_eq!(n_order(2, 7), Some(3));
        assert_eq!(n_order(1, 5), Some(1));
        assert_eq!(n_order(2, 4), None);
    }
}
