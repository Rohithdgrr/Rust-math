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
/// For overflow-safe computation, use [`checked_binomial`].
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
    checked_binomial(n, k).unwrap_or(u128::MAX)
}

/// Binomial coefficient `n choose k`, returning `None` on overflow.
/// `k > n` returns `Some(0)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::checked_binomial;
///
/// assert_eq!(checked_binomial(10, 3), Some(120));
/// assert_eq!(checked_binomial(5, 0), Some(1));
/// assert_eq!(checked_binomial(5, 6), Some(0));
/// ```
#[must_use]
pub fn checked_binomial(n: u64, k: u64) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 1..=k {
        acc = acc.checked_mul(u128::from(n - k + i))?;
        acc /= u128::from(i);
    }
    Some(acc)
}

/// `n!` as `u128` (exact up to `n = 34`; larger wraps silently).
///
/// For overflow-safe computation, use [`checked_factorial`].
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
    checked_factorial(n).unwrap_or(u128::MAX)
}

/// `n!` as `u128`, returning `None` on overflow.
/// Exact for `n <= 34`; returns `None` for `n >= 35`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::checked_factorial;
///
/// assert_eq!(checked_factorial(5), Some(120));
/// assert_eq!(checked_factorial(0), Some(1));
/// assert_eq!(checked_factorial(35), None);
/// ```
#[must_use]
pub fn checked_factorial(n: u64) -> Option<u128> {
    let mut acc: u128 = 1;
    for i in 2..=n {
        acc = acc.checked_mul(u128::from(i))?;
    }
    Some(acc)
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

/// Returns `true` if `n` is even.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::is_even;
///
/// assert!(is_even(4));
/// assert!(!is_even(7));
/// assert!(is_even(0));
/// ```
#[must_use]
#[inline]
pub const fn is_even(n: u64) -> bool {
    n.is_multiple_of(2)
}

/// Returns `true` if `n` is odd.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::is_odd;
///
/// assert!(is_odd(7));
/// assert!(!is_odd(4));
/// assert!(!is_odd(0));
/// ```
#[must_use]
#[inline]
pub const fn is_odd(n: u64) -> bool {
    !n.is_multiple_of(2)
}

/// Check if `n` is a perfect cube: `n = k^3` for some integer `k`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::is_cube;
///
/// assert!(is_cube(27));
/// assert!(is_cube(0));
/// assert!(!is_cube(10));
/// ```
#[must_use]
pub fn is_cube(n: u64) -> bool {
    if n == 0 {
        return true;
    }
    let mut lo = 1u64;
    let mut hi = n.min(u64::MAX / 2 + 1);
    while lo <= hi {
        let mid = u64::midpoint(lo, hi);
        let cube = mid.saturating_mul(mid).saturating_mul(mid);
        if cube == n {
            return true;
        } else if cube < n {
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    false
}

/// Smallest power of two `>= n`. Returns 0 for `n == 0`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::next_power_of_two;
///
/// assert_eq!(next_power_of_two(5), 8);
/// assert_eq!(next_power_of_two(8), 8);
/// ```
#[must_use]
#[inline]
pub const fn next_power_of_two(n: u64) -> u64 {
    n.next_power_of_two()
}

/// Nearest power of two to `n`. Ties round toward even (e.g. 3 -> 4, 5 -> 4).
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::nearest_power_of_two;
///
/// assert_eq!(nearest_power_of_two(1), 1);
/// assert_eq!(nearest_power_of_two(3), 4);
/// assert_eq!(nearest_power_of_two(5), 4);
/// assert_eq!(nearest_power_of_two(6), 8);
/// assert_eq!(nearest_power_of_two(0), 1);
/// ```
#[must_use]
pub fn nearest_power_of_two(n: u64) -> u64 {
    if n <= 1 {
        return 1;
    }
    let floor = 1u64 << (n - 1).ilog2();
    let ceil = floor.saturating_mul(2);
    let d_floor = n - floor;
    let d_ceil = ceil - n;
    match d_floor.cmp(&d_ceil) {
        core::cmp::Ordering::Less => floor,
        _ => ceil,
    }
}

/// Floor integer log base 2: largest `k` such that `2^k <= n`.
/// Returns 0 for `n == 0`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::log2_floor;
///
/// assert_eq!(log2_floor(0), 0);
/// assert_eq!(log2_floor(1), 0);
/// assert_eq!(log2_floor(2), 1);
/// assert_eq!(log2_floor(3), 1);
/// assert_eq!(log2_floor(7), 2);
/// assert_eq!(log2_floor(8), 3);
/// ```
#[must_use]
pub const fn log2_floor(n: u64) -> u32 {
    if n == 0 {
        return 0;
    }
    n.ilog2()
}

/// Integer log base `base`: largest `k` such that `base^k <= n`.
/// Returns 0 for `n == 0` or `base <= 1`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::log_base;
///
/// assert_eq!(log_base(0, 10), 0);
/// assert_eq!(log_base(100, 10), 2);
/// assert_eq!(log_base(26, 26), 1);
/// assert_eq!(log_base(27, 3), 3);
/// ```
#[must_use]
pub fn log_base(n: u64, base: u64) -> u32 {
    if n == 0 || base <= 1 {
        return 0;
    }
    let mut count = 0u32;
    let mut val = n;
    while val >= base {
        val /= base;
        count += 1;
    }
    count
}

/// Population count: number of set bits (1-bits) in `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::popcount;
///
/// assert_eq!(popcount(0), 0);
/// assert_eq!(popcount(7), 3);
/// assert_eq!(popcount(u64::MAX), 64);
/// ```
#[must_use]
#[inline]
pub const fn popcount(n: u64) -> u32 {
    n.count_ones()
}

/// Number of digits of `n` in the given `base` (2-36).
/// `digit_count_base(0, 10)` returns 1.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::digit_count_base;
///
/// assert_eq!(digit_count_base(0, 10), 1);
/// assert_eq!(digit_count_base(255, 16), 2);
/// assert_eq!(digit_count_base(8, 2), 4);
/// ```
#[must_use]
pub fn digit_count_base(n: u64, base: u32) -> usize {
    if base < 2 {
        return 0;
    }
    if n == 0 {
        return 1;
    }
    let mut count = 0;
    let mut val = n;
    while val > 0 {
        val /= u64::from(base);
        count += 1;
    }
    count
}

/// Reverse the digits of `n` in the given `base` (2-36).
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::reverse_digits_base;
///
/// assert_eq!(reverse_digits_base(123, 10), 321);
/// assert_eq!(reverse_digits_base(0b1101, 2), 0b1011);
/// ```
#[must_use]
pub fn reverse_digits_base(mut n: u64, base: u32) -> u64 {
    if base < 2 {
        return n;
    }
    let mut reversed = 0u64;
    while n > 0 {
        reversed = reversed * u64::from(base) + n % u64::from(base);
        n /= u64::from(base);
    }
    reversed
}

/// Check if `n` is a palindrome in the given `base` (2-36).
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::is_palindrome_base;
///
/// assert!(is_palindrome_base(121, 10));
/// assert!(!is_palindrome_base(123, 10));
/// assert!(is_palindrome_base(9, 2)); // 1001 in binary
/// ```
#[must_use]
#[inline]
pub fn is_palindrome_base(n: u64, base: u32) -> bool {
    n == reverse_digits_base(n, base)
}

/// Sum of squares without overflow: `a^2 + b^2` computed in `u128`.
///
/// # Examples
///
/// ```
/// use mathverse_core::integer::sum_of_squares;
///
/// assert_eq!(sum_of_squares(3, 4), 25);
/// assert_eq!(sum_of_squares(0, 5), 25);
/// assert_eq!(sum_of_squares(u64::MAX, 1), u64::MAX as u128 * u64::MAX as u128 + 1);
/// ```
#[must_use]
#[inline]
pub fn sum_of_squares(a: u64, b: u64) -> u128 {
    u128::from(a) * u128::from(a) + u128::from(b) * u128::from(b)
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
        assert_eq!(checked_binomial(10, 3), Some(120));
        assert_eq!(checked_binomial(5, 0), Some(1));
        assert_eq!(checked_binomial(3, 5), Some(0));
    }

    #[test]
    fn factorial_tests() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(checked_factorial(0), Some(1));
        assert_eq!(checked_factorial(5), Some(120));
        assert_eq!(checked_factorial(34), Some(295232799039604140847618609643520000000));
        assert_eq!(checked_factorial(35), None);
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

    #[test]
    fn is_even_is_odd_tests() {
        assert!(is_even(0));
        assert!(is_even(4));
        assert!(!is_even(7));
        assert!(!is_odd(0));
        assert!(is_odd(7));
        assert!(!is_odd(4));
    }

    #[test]
    fn is_cube_tests() {
        assert!(is_cube(0));
        assert!(is_cube(1));
        assert!(is_cube(8));
        assert!(is_cube(27));
        assert!(is_cube(64));
        assert!(!is_cube(10));
        assert!(!is_cube(9));
    }

    #[test]
    fn nearest_power_of_two_tests() {
        assert_eq!(nearest_power_of_two(0), 1);
        assert_eq!(nearest_power_of_two(1), 1);
        assert_eq!(nearest_power_of_two(2), 2);
        assert_eq!(nearest_power_of_two(3), 4);
        assert_eq!(nearest_power_of_two(4), 4);
        assert_eq!(nearest_power_of_two(5), 4);
        assert_eq!(nearest_power_of_two(6), 8);
        assert_eq!(nearest_power_of_two(7), 8);
        assert_eq!(nearest_power_of_two(8), 8);
    }

    #[test]
    fn log2_floor_tests() {
        assert_eq!(log2_floor(0), 0);
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(2), 1);
        assert_eq!(log2_floor(3), 1);
        assert_eq!(log2_floor(7), 2);
        assert_eq!(log2_floor(8), 3);
        assert_eq!(log2_floor(15), 3);
        assert_eq!(log2_floor(16), 4);
    }

    #[test]
    fn log_base_tests() {
        assert_eq!(log_base(0, 10), 0);
        assert_eq!(log_base(1, 10), 0);
        assert_eq!(log_base(9, 1), 0);
        assert_eq!(log_base(100, 10), 2);
        assert_eq!(log_base(26, 26), 1);
        assert_eq!(log_base(27, 3), 3);
        assert_eq!(log_base(63, 2), 5);
        assert_eq!(log_base(64, 2), 6);
    }

    #[test]
    fn popcount_tests() {
        assert_eq!(popcount(0), 0);
        assert_eq!(popcount(1), 1);
        assert_eq!(popcount(7), 3);
        assert_eq!(popcount(8), 1);
        assert_eq!(popcount(u64::MAX), 64);
    }

    #[test]
    fn digit_count_base_tests() {
        assert_eq!(digit_count_base(0, 10), 1);
        assert_eq!(digit_count_base(9, 10), 1);
        assert_eq!(digit_count_base(10, 10), 2);
        assert_eq!(digit_count_base(255, 16), 2);
        assert_eq!(digit_count_base(8, 2), 4);
        assert_eq!(digit_count_base(0, 2), 1);
    }

    #[test]
    fn reverse_digits_base_tests() {
        assert_eq!(reverse_digits_base(123, 10), 321);
        assert_eq!(reverse_digits_base(0, 10), 0);
        assert_eq!(reverse_digits_base(0b1101, 2), 0b1011);
        assert_eq!(reverse_digits_base(0xff, 16), 0xff);
    }

    #[test]
    fn is_palindrome_base_tests() {
        assert!(is_palindrome_base(121, 10));
        assert!(!is_palindrome_base(123, 10));
        assert!(is_palindrome_base(9, 2));
        assert!(is_palindrome_base(0, 10));
        assert!(is_palindrome_base(1, 10));
    }

    #[test]
    fn sum_of_squares_tests() {
        assert_eq!(sum_of_squares(3, 4), 25);
        assert_eq!(sum_of_squares(0, 5), 25);
        assert_eq!(sum_of_squares(0, 0), 0);
    }
}
