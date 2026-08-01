//! Core integer algorithms: GCD, LCM, modular arithmetic, root-finding,
//! and combinatorial primitives.
//!
//! These are the fundamental building blocks used across number theory,
//! combinatorics, and cryptography. Split from `algorithms.rs` to keep
//! the core substrate lean.

/// Greatest common divisor (Euclidean algorithm).
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::gcd;
///
/// assert_eq!(gcd(48, 18), 6);
/// assert_eq!(gcd(17, 5), 1);
/// assert_eq!(gcd(0, 5), 5);
/// ```
#[must_use]
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Greatest common divisor of `n` numbers.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::gcd_n;
///
/// assert_eq!(gcd_n(&[48, 18, 24]), 6);
/// ```
#[must_use]
pub fn gcd_n(xs: &[u64]) -> u64 {
    xs.iter().copied().fold(0, gcd)
}

/// Least common multiple. Returns 0 for `lcm(0, x)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::lcm;
///
/// assert_eq!(lcm(4, 6), 12);
/// assert_eq!(lcm(0, 5), 0);
/// ```
#[must_use]
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        0
    } else {
        a / gcd(a, b) * b
    }
}

/// Least common multiple of `n` numbers.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::lcm_n;
///
/// assert_eq!(lcm_n(&[4, 6, 8]), 24);
/// ```
#[must_use]
pub fn lcm_n(xs: &[u64]) -> u64 {
    xs.iter().copied().fold(1, lcm)
}

/// Extended Euclidean algorithm: returns `(g, x, y)` such that
/// `a*x + b*y = g = gcd(a, b)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::extended_gcd;
///
/// let (g, x, y) = extended_gcd(48, 18);
/// assert_eq!(g, 6);
/// assert_eq!(48 * x + 18 * y, 6);
/// ```
#[must_use]
pub fn extended_gcd(a: i64, b: i64) -> (u64, i64, i64) {
    if b == 0 {
        return (u64::try_from(a).unwrap_or(0), 1, 0);
    }
    let (g, x1, y1) = extended_gcd(b, a % b);
    (g, y1, x1 - (a / b) * y1)
}

/// Bezout coefficients: returns `(x, y)` such that `a*x + b*y = gcd(a, b)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::bezout_coefficients;
///
/// let (x, y) = bezout_coefficients(48, 18);
/// assert_eq!(48 * x + 18 * y, 6);
/// ```
#[must_use]
#[inline]
pub fn bezout_coefficients(a: i64, b: i64) -> (i64, i64) {
    let (_, x, y) = extended_gcd(a, b);
    (x, y)
}

/// Modular multiplicative inverse: `a^-1 mod m`.
///
/// Returns `None` if `a` and `m` are not coprime.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::modular_inverse;
///
/// assert_eq!(modular_inverse(3, 11), Some(4)); // 3 * 4 = 12 ≡ 1 (mod 11)
/// assert_eq!(modular_inverse(2, 4), None);
/// ```
#[must_use]
pub fn modular_inverse(a: u64, m: u64) -> Option<u64> {
    if m == 0 {
        return None;
    }
    let a = a % m;
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g != 1 {
        return None;
    }
    let result = ((x % m as i64) + m as i64) % m as i64;
    Some(result as u64)
}

/// Modular exponentiation: `base^exp mod m` (exponentiation by squaring).
///
/// `m == 0` returns 0.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::mod_pow;
///
/// assert_eq!(mod_pow(2, 10, 1000), 24);
/// assert_eq!(mod_pow(3, 0, 7), 1);
/// ```
#[must_use]
pub fn mod_pow(base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 0 {
        return 0;
    }
    let m = u128::from(m);
    let mut base = u128::from(base) % m;
    let mut result: u128 = 1 % m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % m;
        }
        base = base * base % m;
        exp >>= 1;
    }
    result as u64
}

/// Integer square root: `floor(sqrt(n))`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::isqrt;
///
/// assert_eq!(isqrt(16), 4);
/// assert_eq!(isqrt(17), 4);
/// assert_eq!(isqrt(0), 0);
/// ```
#[must_use]
pub fn isqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = u64::midpoint(x, n / x);
    }
    x
}

/// Integer square root with remainder: returns `(floor(sqrt(n)), n - floor(sqrt(n))^2)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::isqrt_rem;
///
/// assert_eq!(isqrt_rem(17), (4, 1));
/// assert_eq!(isqrt_rem(16), (4, 0));
/// ```
#[must_use]
#[inline]
pub fn isqrt_rem(n: u64) -> (u64, u64) {
    let s = isqrt(n);
    (s, n - s * s)
}

/// Check if `n` is a perfect square.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::is_square;
///
/// assert!(is_square(16));
/// assert!(!is_square(15));
/// ```
#[must_use]
#[inline]
pub fn is_square(n: u64) -> bool {
    if n == 0 {
        return true;
    }
    let s = isqrt(n);
    s * s == n
}

/// Binomial coefficient `n choose k`. `k > n` returns 0.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::binomial;
///
/// assert_eq!(binomial(10, 3), 120);
/// assert_eq!(binomial(5, 0), 1);
/// ```
#[must_use]
pub fn binomial(n: u64, k: u64) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 1..=k {
        acc = acc * u128::from(n - k + i) / u128::from(i);
    }
    acc
}

/// `n!` as `u128` (exact up to `n = 34`; larger wraps silently).
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::factorial;
///
/// assert_eq!(factorial(5), 120);
/// assert_eq!(factorial(0), 1);
/// ```
#[must_use]
pub fn factorial(n: u64) -> u128 {
    let mut acc: u128 = 1;
    for i in 2..=n {
        acc *= u128::from(i);
    }
    acc
}

/// Ceiling integer log base 2: smallest `k` such that `2^k >= n`.
/// Returns 0 for `n == 0`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::log2_ceil;
///
/// assert_eq!(log2_ceil(1), 0);
/// assert_eq!(log2_ceil(2), 1);
/// assert_eq!(log2_ceil(3), 2);
/// assert_eq!(log2_ceil(8), 3);
/// assert_eq!(log2_ceil(9), 4);
/// ```
#[must_use]
pub const fn log2_ceil(n: u64) -> u32 {
    if n <= 1 {
        return 0;
    }
    64 - (n - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_lcm_tests() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(17, 5), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd_n(&[48, 18, 24]), 6);
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(0, 5), 0);
        assert_eq!(lcm_n(&[4, 6, 8]), 24);
    }

    #[test]
    fn extended_gcd_tests() {
        let (g, x, y) = extended_gcd(48, 18);
        assert_eq!(g, 6);
        assert_eq!(48 * x + 18 * y, 6);
    }

    #[test]
    fn bezout_tests() {
        let (x, y) = bezout_coefficients(48, 18);
        assert_eq!(48 * x + 18 * y, 6);
    }

    #[test]
    fn modular_inverse_tests() {
        assert_eq!(modular_inverse(3, 11), Some(4));
        assert_eq!(modular_inverse(2, 4), None);
    }

    #[test]
    fn mod_pow_tests() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 0, 7), 1);
    }

    #[test]
    fn isqrt_tests() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(17), 4);
        assert_eq!(isqrt_rem(17), (4, 1));
    }

    #[test]
    fn is_square_tests() {
        assert!(is_square(0));
        assert!(is_square(1));
        assert!(is_square(16));
        assert!(!is_square(15));
    }

    #[test]
    fn binomial_tests() {
        assert_eq!(binomial(10, 3), 120);
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(3, 5), 0);
    }

    #[test]
    fn factorial_tests() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(5), 120);
    }

    #[test]
    fn log2_ceil_tests() {
        assert_eq!(log2_ceil(0), 0);
        assert_eq!(log2_ceil(1), 0);
        assert_eq!(log2_ceil(2), 1);
        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(8), 3);
        assert_eq!(log2_ceil(9), 4);
    }
}
