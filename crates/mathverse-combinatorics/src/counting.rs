//! Counting functions: combinations, permutations, falling/rising factorials.

use mathverse_core::algorithms::binomial;

/// Number of ways to choose `k` items from `n` without regard to order.
#[must_use]
pub fn combinations(n: u64, k: u64) -> u128 {
    binomial(n, k)
}

/// Number of ordered arrangements of `k` items chosen from `n`.
#[must_use]
pub fn permutations(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let mut acc: u128 = 1;
    for i in (n - k + 1)..=n {
        acc *= u128::from(i);
    }
    acc
}

/// Number of length-`k` strings over an alphabet of size `n` (`n^k`).
#[must_use]
pub fn permutations_with_repetition(n: u64, k: u64) -> u128 {
    u128::from(n).pow(k as u32)
}

/// Number of multisets of size `k` from `n` types ("stars and bars" count).
#[must_use]
pub fn combinations_with_repetition(n: u64, k: u64) -> u128 {
    binomial(n + k - 1, k)
}

/// Falling factorial `n * (n-1) * ... * (n-k+1)`.
#[must_use]
pub fn falling_factorial(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let mut acc: u128 = 1;
    for i in (n - k + 1)..=n {
        acc *= u128::from(i);
    }
    acc
}

/// Rising factorial `n * (n+1) * ... * (n+k-1)`.
#[must_use]
pub fn rising_factorial(n: u64, k: u64) -> u128 {
    let mut acc: u128 = 1;
    for i in n..(n + k) {
        acc *= u128::from(i);
    }
    acc
}

/// Alias for [`combinations_with_repetition`].
#[must_use]
pub fn multichoose(n: u64, k: u64) -> u128 {
    combinations_with_repetition(n, k)
}

/// Alias for [`permutations`].
#[must_use]
pub fn arrangements(n: u64, k: u64) -> u128 {
    permutations(n, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        assert_eq!(combinations(10, 3), 120);
        assert_eq!(permutations(5, 2), 20);
        assert_eq!(permutations(5, 0), 1);
        assert_eq!(permutations(3, 4), 0);
        assert_eq!(permutations_with_repetition(3, 2), 9);
        assert_eq!(combinations_with_repetition(3, 2), 6);
    }

    #[test]
    fn factorials() {
        assert_eq!(falling_factorial(5, 3), 60);
        assert_eq!(rising_factorial(3, 3), 60);
        assert_eq!(falling_factorial(5, 0), 1);
    }

    #[test]
    fn aliases() {
        assert_eq!(multichoose(3, 2), 6);
        assert_eq!(arrangements(5, 2), 20);
    }
}
