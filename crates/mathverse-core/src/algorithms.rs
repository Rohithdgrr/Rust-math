//! Common integer algorithms: number theory basics and combinatorics.
//!
//! All functions in this module operate on `u64` or `u128` integers.
//! They cover GCD/LCM, prime number utilities, combinatorial functions,
//! and digit-based operations.

use alloc::vec;
use alloc::vec::Vec;

/// Greatest common divisor (Euclidean algorithm).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::gcd;
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
/// use mathverse_core::algorithms::gcd_n;
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
/// use mathverse_core::algorithms::lcm;
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
/// use mathverse_core::algorithms::lcm_n;
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
/// use mathverse_core::algorithms::extended_gcd;
///
/// let (g, x, y) = extended_gcd(48, 18);
/// assert_eq!(g, 6);
/// assert_eq!(48 * x + 18 * y, 6);
/// ```
#[must_use]
pub fn extended_gcd(a: i64, b: i64) -> (u64, i64, i64) {
    if b == 0 {
        return (a as u64, 1, 0);
    }
    let (g, x1, y1) = extended_gcd(b, a % b);
    (g, y1, x1 - (a / b) * y1)
}

/// Bezout coefficients: returns `(x, y)` such that `a*x + b*y = gcd(a, b)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::bezout_coefficients;
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
/// use mathverse_core::algorithms::modular_inverse;
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

/// Euler's totient function phi(n): count of integers <= n that are coprime to n.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::euler_phi;
///
/// assert_eq!(euler_phi(9), 6);
/// assert_eq!(euler_phi(7), 6); // prime
/// ```
#[must_use]
pub fn euler_phi(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    let mut temp = n;
    let mut p = 2u64;
    while p * p <= temp {
        if temp.is_multiple_of(p) {
            while temp.is_multiple_of(p) {
                temp /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if temp > 1 {
        result -= result / temp;
    }
    result
}

/// Check if `a` and `b` are coprime (gcd == 1).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_coprime;
///
/// assert!(is_coprime(8, 9));
/// assert!(!is_coprime(8, 12));
/// ```
#[must_use]
#[inline]
pub fn is_coprime(a: u64, b: u64) -> bool {
    gcd(a, b) == 1
}

/// Chinese Remainder Theorem solver.
///
/// Given residues `r` and moduli `m`, find `x` such that `x ≡ r_i (mod m_i)`.
/// Returns `None` if moduli are not pairwise coprime.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::chinese_remainder;
///
/// let r = chinese_remainder(&[2, 3, 2], &[3, 5, 7]);
/// assert_eq!(r, Some(23));
/// ```
#[must_use]
pub fn chinese_remainder(residues: &[u64], moduli: &[u64]) -> Option<u64> {
    if residues.len() != moduli.len() || moduli.is_empty() {
        return None;
    }
    let total_mod: u128 = moduli.iter().map(|&m| u128::from(m)).product();
    let mut result: u128 = 0;
    for i in 0..moduli.len() {
        let m_i = u128::from(moduli[i]);
        let r_i = u128::from(residues[i]);
        let m_div = total_mod / m_i;
        let inv = modular_inverse(m_div as u64, moduli[i])?;
        result = (result + r_i * m_div % total_mod * u128::from(inv) % total_mod) % total_mod;
    }
    Some(result as u64)
}

/// Integer square root: `floor(sqrt(n))`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::isqrt;
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
/// use mathverse_core::algorithms::isqrt_rem;
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
/// use mathverse_core::algorithms::is_square;
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

/// Check if `n` is a perfect power: `n = a^b` for some `a >= 2, b >= 2`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_perfect_power;
///
/// assert!(is_perfect_power(8));  // 2^3
/// assert!(is_perfect_power(16)); // 2^4
/// assert!(!is_perfect_power(10));
/// ```
#[must_use]
pub fn is_perfect_power(n: u64) -> bool {
    if n < 4 {
        return false;
    }
    let mut b = 2u32;
    while u128::from(2u32).pow(b) <= u128::from(n) {
        let root = nth_root_u64(n, b);
        if root >= 2 && root.pow(b) == n {
            return true;
        }
        b += 1;
    }
    false
}

/// Integer `n`-th root: `floor(n^(1/k))`.
fn nth_root_u64(n: u64, k: u32) -> u64 {
    if n == 0 {
        return 0;
    }
    if k == 0 {
        return 1;
    }
    if k == 1 {
        return n;
    }
    let mut lo: u64 = 1;
    let mut hi: u64 = n;
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if mid.pow(k) <= n {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Modular exponentiation: `base^exp mod m` (exponentiation by squaring).
///
/// `m == 0` returns 0.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::mod_pow;
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

/// `n!` as `u128` (exact up to `n = 34`; larger wraps silently).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::factorial;
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

/// Binomial coefficient `n choose k`. `k > n` returns 0.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::binomial;
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

/// `n`-th Fibonacci number (`F_0 = 0, F_1 = 1`) as `u128` (exact up to n = 184).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::fibonacci;
///
/// assert_eq!(fibonacci(0), 0);
/// assert_eq!(fibonacci(1), 1);
/// assert_eq!(fibonacci(10), 55);
/// ```
#[must_use]
pub fn fibonacci(n: u64) -> u128 {
    if n == 0 {
        return 0;
    }
    let (mut a, mut b) = (0u128, 1u128);
    for _ in 1..n {
        let t = a + b;
        a = b;
        b = t;
    }
    b
}

/// Trial-division primality test.
///
/// For large-scale primality testing, prefer [`is_prime_miller_rabin`].
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_prime;
///
/// assert!(is_prime(7));
/// assert!(!is_prime(4));
/// assert!(!is_prime(1));
/// ```
// ponytail: naive O(sqrt(n)); swap to Miller-Rabin when big primes matter.
#[must_use]
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3u64;
    while u128::from(d) * u128::from(d) <= u128::from(n) {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// `true` if `n` is a power of two (1, 2, 4, ...). `0` is not.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_power_of_two;
///
/// assert!(is_power_of_two(16));
/// assert!(!is_power_of_two(10));
/// ```
#[must_use]
#[inline]
pub const fn is_power_of_two(n: u64) -> bool {
    n.is_power_of_two()
}

/// All primes `<= n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::sieve_of_eratosthenes;
///
/// assert_eq!(sieve_of_eratosthenes(20).len(), 8);
/// assert_eq!(sieve_of_eratosthenes(2), vec![2]);
/// ```
#[must_use]
pub fn sieve_of_eratosthenes(n: usize) -> Vec<u64> {
    let mut prime = vec![true; n + 1];
    prime[0] = false;
    if n >= 1 {
        prime[1] = false;
    }
    let mut p = 2usize;
    while p * p <= n {
        if prime[p] {
            let mut m = p * p;
            while m <= n {
                prime[m] = false;
                m += p;
            }
        }
        p += 1;
    }
    (2..=n)
        .filter(|&i| prime[i])
        .map(|i| i as u64)
        .collect()
}

/// Prime factorization of `n`: returns a vector of `(prime, exponent)` pairs.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::prime_factorization;
///
/// assert_eq!(prime_factorization(60), vec![(2, 2), (3, 1), (5, 1)]);
/// assert_eq!(prime_factorization(1), vec![]);
/// ```
#[must_use]
pub fn prime_factorization(n: u64) -> Vec<(u64, u32)> {
    if n <= 1 {
        return vec![];
    }
    let mut factors = vec![];
    let mut n = n;
    let mut p = 2u64;
    while p * p <= n {
        if n.is_multiple_of(p) {
            let mut count = 0u32;
            while n.is_multiple_of(p) {
                n /= p;
                count += 1;
            }
            factors.push((p, count));
        }
        p += 1;
    }
    if n > 1 {
        factors.push((n, 1));
    }
    factors
}

/// Distinct prime factors of `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::prime_factors;
///
/// assert_eq!(prime_factors(60), vec![2, 3, 5]);
/// ```
#[must_use]
#[inline]
pub fn prime_factors(n: u64) -> Vec<u64> {
    prime_factorization(n).into_iter().map(|(p, _)| p).collect()
}

/// Count of primes <= `n` (prime-counting function pi(n)).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::prime_count;
///
/// assert_eq!(prime_count(20), 8);
/// ```
#[must_use]
#[inline]
pub fn prime_count(n: u64) -> usize {
    sieve_of_eratosthenes(n as usize).len()
}

/// Next prime after `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::next_prime;
///
/// assert_eq!(next_prime(10), 11);
/// assert_eq!(next_prime(2), 3);
/// assert_eq!(next_prime(1), 2);
/// ```
#[must_use]
pub fn next_prime(n: u64) -> u64 {
    if n < 2 {
        return 2;
    }
    let mut candidate = if n.is_multiple_of(2) { n + 1 } else { n + 2 };
    while !is_prime(candidate) {
        candidate += 2;
    }
    candidate
}

/// Largest prime strictly less than `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::prev_prime;
///
/// assert_eq!(prev_prime(13), Some(11));
/// assert_eq!(prev_prime(2), None);
/// ```
#[must_use]
pub fn prev_prime(n: u64) -> Option<u64> {
    if n <= 2 {
        return None;
    }
    let mut candidate = if n.is_multiple_of(2) { n - 1 } else { n - 2 };
    loop {
        if is_prime(candidate) {
            return Some(candidate);
        }
        if candidate <= 2 {
            return Some(2);
        }
        candidate -= 2;
    }
}

/// The `n`-th prime (1-indexed: 1st prime = 2).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::nth_prime;
///
/// assert_eq!(nth_prime(1), Some(2));
/// assert_eq!(nth_prime(6), Some(13));
/// assert_eq!(nth_prime(0), None);
/// ```
#[must_use]
pub fn nth_prime(n: usize) -> Option<u64> {
    if n == 0 {
        return None;
    }
    let mut count = 0;
    let mut candidate = 1u64;
    while count < n {
        candidate = next_prime(candidate);
        count += 1;
    }
    Some(candidate)
}

/// Check if `n` is a triangular number: `n = k(k+1)/2`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_triangular;
///
/// assert!(is_triangular(6));
/// assert!(is_triangular(0));
/// assert!(!is_triangular(5));
/// ```
#[must_use]
pub fn is_triangular(n: u64) -> bool {
    if n == 0 {
        return true;
    }
    let d = 8 * n + 1;
    is_square(d) && {
        let s = isqrt(d);
        (s - 1).is_multiple_of(2)
    }
}

/// Check if `n` is a Harshad (Niven) number: divisible by sum of its digits.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_harshad;
///
/// assert!(is_harshad(18)); // 18 / (1+8) = 2
/// assert!(!is_harshad(16));
/// ```
#[must_use]
#[inline]
pub fn is_harshad(n: u64) -> bool {
    if n == 0 {
        return false;
    }
    let s = digit_sum(n);
    n.is_multiple_of(s)
}

/// Check if `n` is an Armstrong (Narcissistic) number:
/// `n = sum of its digits each raised to the power of the number of digits`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_armstrong;
///
/// assert!(is_armstrong(153));  // 1^3 + 5^3 + 3^3 = 153
/// assert!(!is_armstrong(154));
/// ```
#[must_use]
pub fn is_armstrong(n: u64) -> bool {
    let digits = to_digits(n);
    let k = digits.len() as u32;
    let sum: u64 = digits.iter().map(|&d| d.pow(k)).sum();
    sum == n
}

/// Check if `n` is a perfect number: sum of its proper divisors equals `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_perfect_number;
///
/// assert!(is_perfect_number(6));   // 1+2+3 = 6
/// assert!(is_perfect_number(28));  // 1+2+4+7+14 = 28
/// ```
#[must_use]
pub fn is_perfect_number(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    divisor_sum(n) - n == n
}

/// Check if `n` is an abundant number: sum of proper divisors > `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_abundant;
///
/// assert!(is_abundant(12)); // 1+2+3+4+6 = 16 > 12
/// ```
#[must_use]
pub fn is_abundant(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    divisor_sum(n) - n > n
}

/// Check if `n` is a deficient number: sum of proper divisors < `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_deficient;
///
/// assert!(is_deficient(8)); // 1+2+4 = 7 < 8
/// ```
#[must_use]
pub fn is_deficient(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    divisor_sum(n) - n < n
}

/// Check if `n` is a semiprime: product of exactly two primes (not necessarily distinct).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_semiprime;
///
/// assert!(is_semiprime(6));  // 2*3
/// assert!(is_semiprime(9));  // 3*3
/// assert!(!is_semiprime(8)); // 2^3
/// ```
#[must_use]
pub fn is_semiprime(n: u64) -> bool {
    let factors = prime_factorization(n);
    let total: u32 = factors.iter().map(|(_, e)| e).sum();
    total == 2
}

/// Check if `n` is squarefree: no repeated prime factors.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_squarefree;
///
/// assert!(is_squarefree(30)); // 2*3*5
/// assert!(!is_squarefree(18)); // 2*3^2
/// ```
#[must_use]
pub fn is_squarefree(n: u64) -> bool {
    prime_factorization(n).iter().all(|(_, e)| *e == 1)
}

/// Sum of decimal digits of `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::digit_sum;
///
/// assert_eq!(digit_sum(12345), 15);
/// ```
#[must_use]
#[inline]
pub fn digit_sum(n: u64) -> u64 {
    to_digits(n).iter().sum()
}

/// Number of decimal digits in `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::digit_count;
///
/// assert_eq!(digit_count(0), 1);
/// assert_eq!(digit_count(999), 3);
/// ```
#[must_use]
pub fn digit_count(n: u64) -> usize {
    if n == 0 {
        1
    } else {
        (n as f64).log10().floor() as usize + 1
    }
}

/// Reverse the decimal digits of `n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::reverse_digits;
///
/// assert_eq!(reverse_digits(12345), 54321);
/// ```
#[must_use]
pub fn reverse_digits(n: u64) -> u64 {
    let digits = to_digits(n);
    from_digits(&digits.iter().rev().copied().collect::<Vec<_>>())
}

/// Check if `n` is a decimal palindrome.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_palindrome;
///
/// assert!(is_palindrome(12321));
/// assert!(!is_palindrome(12345));
/// ```
#[must_use]
#[inline]
pub fn is_palindrome(n: u64) -> bool {
    n == reverse_digits(n)
}

/// Convert `n` to a vector of decimal digits (most significant first).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::to_digits;
///
/// assert_eq!(to_digits(123), vec![1, 2, 3]);
/// assert_eq!(to_digits(0), vec![0]);
/// ```
#[must_use]
pub fn to_digits(n: u64) -> Vec<u64> {
    if n == 0 {
        return vec![0];
    }
    let mut digits = vec![];
    let mut n = n;
    while n > 0 {
        digits.insert(0, n % 10);
        n /= 10;
    }
    digits
}

/// Reconstruct a number from decimal digits (most significant first).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::from_digits;
///
/// assert_eq!(from_digits(&[1, 2, 3]), 123);
/// ```
#[must_use]
pub fn from_digits(digits: &[u64]) -> u64 {
    digits.iter().fold(0u64, |acc, &d| acc * 10 + d)
}

/// Convert `n` to a string in the given `base` (2-36).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::to_base;
///
/// assert_eq!(to_base(255, 16), "ff");
/// assert_eq!(to_base(10, 2), "1010");
/// ```
#[must_use]
pub fn to_base(n: u64, base: u32) -> String {
    if !(2..=36).contains(&base) {
        return String::new();
    }
    if n == 0 {
        return "0".to_string();
    }
    let chars = "0123456789abcdefghijklmnopqrstuvwxyz";
    let mut digits = vec![];
    let mut n = n;
    while n > 0 {
        digits.push(chars.chars().nth((n % u64::from(base)) as usize).unwrap());
        n /= u64::from(base);
    }
    digits.iter().rev().collect()
}

/// Parse a string in the given `base` (2-36) to `u64`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::from_base;
///
/// assert_eq!(from_base("ff", 16), Some(255));
/// assert_eq!(from_base("1010", 2), Some(10));
/// ```
#[must_use]
pub fn from_base(s: &str, base: u32) -> Option<u64> {
    if !(2..=36).contains(&base) {
        return None;
    }
    let chars = "0123456789abcdefghijklmnopqrstuvwxyz";
    let s = s.to_lowercase();
    let mut result: u64 = 0;
    for c in s.chars() {
        let digit = chars.find(c)? as u64;
        if digit >= u64::from(base) {
            return None;
        }
        result = result * u64::from(base) + digit;
    }
    Some(result)
}

/// Generate the first `n` rows of Pascal's triangle.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::pascal_triangle;
///
/// let pt = pascal_triangle(4);
/// assert_eq!(pt[3], vec![1, 3, 3, 1]);
/// ```
#[must_use]
pub fn pascal_triangle(n: usize) -> Vec<Vec<u128>> {
    let mut triangle: Vec<Vec<u128>> = vec![];
    for i in 0..n {
        let mut row = vec![1u128; i + 1];
        if i >= 2 {
            for j in 1..i {
                row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
            }
        }
        triangle.push(row);
    }
    triangle
}

/// `n`-th Lucas number (`L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}`).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::lucas_number;
///
/// assert_eq!(lucas_number(0), 2);
/// assert_eq!(lucas_number(1), 1);
/// assert_eq!(lucas_number(4), 7);
/// ```
#[must_use]
pub fn lucas_number(n: u64) -> u128 {
    if n == 0 {
        return 2;
    }
    if n == 1 {
        return 1;
    }
    let (mut a, mut b) = (2u128, 1u128);
    for _ in 2..=n {
        let t = a + b;
        a = b;
        b = t;
    }
    b
}

/// `n`-th Tribonacci number (`T_0 = 0, T_1 = 0, T_2 = 1, T_n = T_{n-1} + T_{n-2} + T_{n-3}`).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::tribonacci;
///
/// assert_eq!(tribonacci(0), 0);
/// assert_eq!(tribonacci(3), 1);
/// assert_eq!(tribonacci(4), 2);
/// ```
#[must_use]
pub fn tribonacci(n: u64) -> u128 {
    if n == 0 || n == 1 {
        return 0;
    }
    if n == 2 {
        return 1;
    }
    let (mut a, mut b, mut c) = (0u128, 0u128, 1u128);
    for _ in 3..=n {
        let t = a + b + c;
        a = b;
        b = c;
        c = t;
    }
    c
}

/// `n`-th Catalan number: `C(n) = (1/(n+1)) * C(2n, n)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::catalan_number;
///
/// assert_eq!(catalan_number(0), 1);
/// assert_eq!(catalan_number(3), 5);
/// assert_eq!(catalan_number(5), 42);
/// ```
#[must_use]
pub fn catalan_number(n: u64) -> u128 {
    if n == 0 {
        return 1;
    }
    binomial(2 * n, n) / u128::from(n + 1)
}

/// Number of integer partitions of `n` (partition function p(n)).
///
/// Uses Euler's pentagonal number theorem recurrence.
/// Time complexity: O(n^2). For large `n` (>200), consider memoization.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::partition_number;
///
/// assert_eq!(partition_number(0), 1);
/// assert_eq!(partition_number(4), 5);
/// assert_eq!(partition_number(5), 7);
/// ```
#[must_use]
pub fn partition_number(n: u64) -> u128 {
    if n == 0 {
        return 1;
    }
    let mut partitions = vec![0u128; n as usize + 1];
    partitions[0] = 1;
    for i in 1..=n as usize {
        let mut sum: i128 = 0;
        let mut k = 1u64;
        loop {
            let pent1 = (k * (3 * k - 1) / 2) as usize;
            if pent1 > i {
                break;
            }
            let sign: i128 = if k % 2 == 1 { 1 } else { -1 };
            sum += sign * partitions[i - pent1] as i128;
            let pent2 = (k * (3 * k + 1) / 2) as usize;
            if pent2 <= i {
                sum += sign * partitions[i - pent2] as i128;
            }
            k += 1;
        }
        partitions[i] = sum as u128;
    }
    partitions[n as usize]
}

/// Unsigned Stirling numbers of the first kind: `s(n, k)`.
///
/// Counts permutations of `n` elements with `k` disjoint cycles.
/// Time complexity: O(n*k). Results grow fast — may overflow `u128` for `n > 34`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::stirling_first;
///
/// assert_eq!(stirling_first(4, 2), 11);
/// ```
// ponytail: recursive implementation; memoize if profiling shows hot path.
#[must_use]
pub fn stirling_first(n: u64, k: u64) -> u128 {
    if k > n || k == 0 {
        return u128::from(n == 0 && k == 0);
    }
    if k == n {
        return 1;
    }
    stirling_first(n - 1, k - 1) + u128::from(n - 1) * stirling_first(n - 1, k)
}

/// Stirling numbers of the second kind: `S(n, k)`.
///
/// Counts ways to partition `n` elements into `k` non-empty subsets.
/// Time complexity: O(n*k). Results grow fast — may overflow `u128` for `n > 34`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::stirling_second;
///
/// assert_eq!(stirling_second(4, 2), 7);
/// ```
// ponytail: recursive implementation; memoize if profiling shows hot path.
#[must_use]
pub fn stirling_second(n: u64, k: u64) -> u128 {
    if k > n || k == 0 {
        return u128::from(n == 0 && k == 0);
    }
    if k == 1 || k == n {
        return 1;
    }
    stirling_second(n - 1, k - 1) + u128::from(k) * stirling_second(n - 1, k)
}

/// `n`-th Bell number: total partitions of a set of n elements.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::bell_number;
///
/// assert_eq!(bell_number(0), 1);
/// assert_eq!(bell_number(3), 5);
/// assert_eq!(bell_number(4), 15);
/// ```
#[must_use]
pub fn bell_number(n: u64) -> u128 {
    let mut row = vec![1u128];
    for i in 1..=n {
        let mut next = vec![0u128; (i + 1) as usize];
        next[0] = *row.last().unwrap();
        for j in 1..=i as usize {
            next[j] = next[j - 1] + row[j - 1];
        }
        row = next;
    }
    row[0]
}

/// Subfactorial (derangement number): `!n = n! * sum(k=0..n) (-1)^k / k!`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::subfactorial;
///
/// assert_eq!(subfactorial(0), 1);
/// assert_eq!(subfactorial(1), 0);
/// assert_eq!(subfactorial(4), 9); // !4 = 9
/// ```
#[must_use]
pub fn subfactorial(n: u64) -> u128 {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 0;
    }
    let mut d = vec![0u128; n as usize + 1];
    d[0] = 1;
    d[1] = 0;
    for i in 2..=n as usize {
        d[i] = (i as u128 - 1) * (d[i - 1] + d[i - 2]);
    }
    d[n as usize]
}

/// Number of permutations: `P(n, r) = n! / (n - r)!`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::permutation_count;
///
/// assert_eq!(permutation_count(5, 3), 60);
/// assert_eq!(permutation_count(5, 0), 1);
/// assert_eq!(permutation_count(3, 5), 0);
/// ```
#[must_use]
pub fn permutation_count(n: u64, r: u64) -> u128 {
    if r > n {
        return 0;
    }
    let mut result: u128 = 1;
    for i in (n - r + 1)..=n {
        result *= u128::from(i);
    }
    result
}

/// Number of divisors: `d(n)` or `tau(n)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::divisor_count;
///
/// assert_eq!(divisor_count(12), 6); // 1,2,3,4,6,12
/// ```
#[must_use]
pub fn divisor_count(n: u64) -> u64 {
    let factors = prime_factorization(n);
    factors
        .iter()
        .map(|(_, e)| u64::from(e + 1))
        .product()
}

/// Sum of divisors: `sigma(n)`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::divisor_sum;
///
/// assert_eq!(divisor_sum(6), 12); // 1+2+3+6 = 12
/// ```
#[must_use]
pub fn divisor_sum(n: u64) -> u64 {
    let factors = prime_factorization(n);
    factors
        .iter()
        .map(|(p, e)| {
            let mut sum = 1u64;
            let mut power = 1u64;
            for _ in 0..*e {
                power *= p;
                sum += power;
            }
            sum
        })
        .product()
}

/// Radical (square-free kernel): product of distinct prime factors.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::radical;
///
/// assert_eq!(radical(18), 6); // 18 = 2 * 3^2, radical = 2 * 3 = 6
/// ```
#[must_use]
pub fn radical(n: u64) -> u64 {
    prime_factors(n).iter().product()
}

/// Check if `n` is a smooth number: all prime factors <= `bound`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_smooth;
///
/// assert!(is_smooth(12, 3));  // 12 = 2^2*3, all factors <= 3
/// assert!(!is_smooth(12, 2));
/// ```
#[must_use]
pub fn is_smooth(n: u64, bound: u64) -> bool {
    prime_factors(n).iter().all(|&p| p <= bound)
}

/// Primorial: product of primes <= `p`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::primorial;
///
/// assert_eq!(primorial(5), 30); // 2 * 3 * 5 = 30
/// ```
#[must_use]
pub fn primorial(p: u64) -> u64 {
    sieve_of_eratosthenes(p as usize)
        .iter()
        .copied()
        .product()
}

/// `n`-th Mersenne number: `2^n - 1`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::mersenne_number;
///
/// assert_eq!(mersenne_number(3), 7);
/// ```
#[must_use]
#[inline]
pub const fn mersenne_number(n: u32) -> u64 {
    (1u64 << n) - 1
}

/// `n`-th Fermat number: `2^(2^n) + 1`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::fermat_number;
///
/// assert_eq!(fermat_number(0), 3);
/// assert_eq!(fermat_number(1), 5);
/// ```
#[must_use]
#[inline]
pub const fn fermat_number(n: u32) -> u128 {
    (1u128 << (1u32 << n)) + 1
}

/// Mobius function mu(n):
/// - 1 if n is squarefree with an even number of prime factors
/// - -1 if n is squarefree with an odd number of prime factors
/// - 0 if n has a repeated prime factor
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::mobius;
///
/// assert_eq!(mobius(1), 1);
/// assert_eq!(mobius(2), -1);
/// assert_eq!(mobius(4), 0); // 4 = 2^2, not squarefree
/// ```
#[must_use]
pub fn mobius(n: u64) -> i32 {
    if n == 0 {
        return 0;
    }
    let factors = prime_factorization(n);
    if factors.iter().any(|(_, e)| *e > 1) {
        return 0;
    }
    if factors.len().is_multiple_of(2) {
        1
    } else {
        -1
    }
}

/// Liouville function lambda(n): `(-1)^Omega(n)` where Omega(n) = total number of prime factors with multiplicity.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::liouville;
///
/// assert_eq!(liouville(1), 1);
/// assert_eq!(liouville(6), 1);  // Omega(6) = 2 (even)
/// assert_eq!(liouville(8), -1); // Omega(8) = 3 (odd)
/// ```
#[must_use]
pub fn liouville(n: u64) -> i32 {
    if n == 0 {
        return 0;
    }
    let factors = prime_factorization(n);
    let total: u32 = factors.iter().map(|(_, e)| e).sum();
    if total.is_multiple_of(2) {
        1
    } else {
        -1
    }
}

/// Double factorial: `n!! = n * (n-2) * (n-4) * ... * 1` (or 2).
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::double_factorial;
///
/// assert_eq!(double_factorial(5), 15); // 5 * 3 * 1 = 15
/// assert_eq!(double_factorial(6), 48); // 6 * 4 * 2 = 48
/// ```
#[must_use]
pub fn double_factorial(n: u64) -> u128 {
    if n == 0 || n == 1 {
        return 1;
    }
    let mut result: u128 = 1;
    let mut k = n;
    while k >= 2 {
        result *= u128::from(k);
        k -= 2;
    }
    result
}

/// Miller-Rabin probabilistic primality test.
///
/// Accuracy: `2^(-rounds)` probability of false positive.
/// Uses a simple PRNG seeded from `n` — not cryptographically secure.
/// For full `u64` deterministic primality, use [`is_prime`] (trial division)
/// or a known set of deterministic bases.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_prime_miller_rabin;
///
/// assert!(is_prime_miller_rabin(97, 10));
/// assert!(!is_prime_miller_rabin(100, 10));
/// ```
#[must_use]
pub fn is_prime_miller_rabin(n: u64, rounds: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }
    let mut d = n - 1;
    let r = d.trailing_zeros();
    d >>= r;
    let mut rng_state = n.wrapping_mul(0x5DEECE66D);
    for _ in 0..rounds {
        let a = 2 + (rng_state % (n - 3));
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        let mut composite = true;
        for _ in 1..r {
            x = (u128::from(x) * u128::from(x) % u128::from(n)) as u64;
            if x == n - 1 {
                composite = false;
                break;
            }
        }
        if composite {
            return false;
        }
    }
    true
}

/// Check if `n` is a power of `base`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::is_power_of;
///
/// assert!(is_power_of(8, 2));
/// assert!(is_power_of(27, 3));
/// assert!(!is_power_of(10, 2));
/// ```
#[must_use]
pub const fn is_power_of(n: u64, base: u64) -> bool {
    if n == 0 || base == 0 {
        return false;
    }
    if n == 1 {
        return true;
    }
    let mut n = n;
    while n.is_multiple_of(base) {
        n /= base;
    }
    n == 1
}

/// Smallest power of two `>= n`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::next_power_of_two;
///
/// assert_eq!(next_power_of_two(5), 8);
/// assert_eq!(next_power_of_two(8), 8);
/// ```
#[must_use]
#[inline]
pub const fn next_power_of_two(n: u64) -> u64 {
    n.next_power_of_two()
}

/// Segmented sieve: all primes in `[lo, hi]`.
///
/// # Examples
///
/// ```
/// use mathverse_core::algorithms::segmented_sieve;
///
/// let primes = segmented_sieve(10, 30);
/// assert_eq!(primes, vec![11, 13, 17, 19, 23, 29]);
/// ```
#[must_use]
pub fn segmented_sieve(lo: u64, hi: u64) -> Vec<u64> {
    if hi < 2 || hi < lo {
        return vec![];
    }
    let limit = isqrt(hi) as usize;
    let small_primes = sieve_of_eratosthenes(limit);
    let mut is_prime = vec![true; (hi - lo + 1) as usize];
    for &p in &small_primes {
        let start = if lo <= p {
            p * p
        } else {
            lo.div_ceil(p) * p
        };
        let mut j = start;
        while j <= hi {
            if j >= lo {
                is_prime[(j - lo) as usize] = false;
            }
            j += p;
        }
    }
    (lo..=hi)
        .filter(|&i| i >= 2 && is_prime[(i - lo) as usize])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_lcm() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(17, 5), 1);
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(0, 5), 0);
        assert_eq!(gcd_n(&[48, 18, 24]), 6);
        assert_eq!(lcm_n(&[4, 6, 8]), 24);
    }

    #[test]
    fn extended_gcd_test() {
        let (g, x, y) = extended_gcd(48, 18);
        assert_eq!(g, 6);
        assert_eq!(48 * x + 18 * y, 6);
        let (g2, _, _) = extended_gcd(17, 5);
        assert_eq!(g2, 1);
    }

    #[test]
    fn modular_inverse_test() {
        assert_eq!(modular_inverse(3, 11), Some(4));
        assert_eq!(modular_inverse(10, 17), Some(12));
        assert_eq!(modular_inverse(2, 4), None);
    }

    #[test]
    fn euler_phi_test() {
        assert_eq!(euler_phi(1), 1);
        assert_eq!(euler_phi(7), 6);
        assert_eq!(euler_phi(9), 6);
        assert_eq!(euler_phi(12), 4);
    }

    #[test]
    fn coprime_test() {
        assert!(is_coprime(8, 9));
        assert!(!is_coprime(8, 12));
    }

    #[test]
    fn crt_test() {
        let r = chinese_remainder(&[2, 3, 2], &[3, 5, 7]);
        assert_eq!(r, Some(23));
    }

    #[test]
    fn modular() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 0, 7), 1);
        assert_eq!(mod_pow(7, 4, 5), 1);
    }

    #[test]
    fn isqrt_test() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(17), 4);
        assert_eq!(isqrt(25), 5);
        assert_eq!(isqrt_rem(17), (4, 1));
    }

    #[test]
    fn is_square_test() {
        assert!(is_square(0));
        assert!(is_square(1));
        assert!(is_square(16));
        assert!(is_square(25));
        assert!(!is_square(15));
        assert!(!is_square(17));
    }

    #[test]
    fn perfect_power_test() {
        assert!(is_perfect_power(8));
        assert!(is_perfect_power(16));
        assert!(is_perfect_power(27));
        assert!(!is_perfect_power(10));
        assert!(!is_perfect_power(15));
    }

    #[test]
    fn prime_factorization_test() {
        assert_eq!(prime_factorization(60), vec![(2, 2), (3, 1), (5, 1)]);
        assert_eq!(prime_factorization(1), vec![]);
        assert_eq!(prime_factorization(7), vec![(7, 1)]);
        assert_eq!(prime_factors(60), vec![2, 3, 5]);
    }

    #[test]
    fn prime_utils() {
        assert_eq!(next_prime(10), 11);
        assert_eq!(next_prime(2), 3);
        assert_eq!(next_prime(1), 2);
        assert_eq!(prev_prime(13), Some(11));
        assert_eq!(prev_prime(2), None);
        assert_eq!(nth_prime(1), Some(2));
        assert_eq!(nth_prime(6), Some(13));
        assert_eq!(nth_prime(0), None);
        assert_eq!(prime_count(20), 8);
    }

    #[test]
    fn miller_rabin_test() {
        assert!(is_prime_miller_rabin(97, 10));
        assert!(!is_prime_miller_rabin(100, 10));
        assert!(is_prime_miller_rabin(7919, 10));
        assert!(is_prime_miller_rabin(2, 10));
        assert!(is_prime_miller_rabin(3, 10));
    }

    #[test]
    fn is_power_of_test() {
        assert!(is_power_of(8, 2));
        assert!(is_power_of(27, 3));
        assert!(!is_power_of(10, 2));
        assert!(is_power_of(1, 5));
    }

    #[test]
    fn digit_operations() {
        assert_eq!(digit_sum(12345), 15);
        assert_eq!(digit_count(0), 1);
        assert_eq!(digit_count(999), 3);
        assert_eq!(reverse_digits(12345), 54321);
        assert!(is_palindrome(12321));
        assert!(!is_palindrome(12345));
        assert_eq!(to_digits(123), vec![1, 2, 3]);
        assert_eq!(from_digits(&[1, 2, 3]), 123);
        assert_eq!(to_base(255, 16), "ff");
        assert_eq!(from_base("ff", 16), Some(255));
        assert_eq!(to_base(10, 2), "1010");
    }

    #[test]
    fn triangular_test() {
        assert!(is_triangular(0));
        assert!(is_triangular(1));
        assert!(is_triangular(3));
        assert!(is_triangular(6));
        assert!(is_triangular(10));
        assert!(!is_triangular(5));
        assert!(!is_triangular(11));
    }

    #[test]
    fn harshad_test() {
        assert!(is_harshad(18));
        assert!(!is_harshad(16));
    }

    #[test]
    fn armstrong_test() {
        assert!(is_armstrong(153));
        assert!(is_armstrong(0));
        assert!(is_armstrong(1));
        assert!(!is_armstrong(154));
    }

    #[test]
    fn perfect_abundant_deficient() {
        assert!(is_perfect_number(6));
        assert!(is_perfect_number(28));
        assert!(is_abundant(12));
        assert!(is_deficient(8));
        assert!(!is_perfect_number(12));
    }

    #[test]
    fn semiprime_squarefree() {
        assert!(is_semiprime(6));
        assert!(is_semiprime(9));
        assert!(!is_semiprime(8));
        assert!(is_squarefree(30));
        assert!(!is_squarefree(18));
    }

    #[test]
    fn pascal_lucas_tribonacci_catalan() {
        let pt = pascal_triangle(5);
        assert_eq!(pt[0], vec![1]);
        assert_eq!(pt[1], vec![1, 1]);
        assert_eq!(pt[2], vec![1, 2, 1]);
        assert_eq!(pt[4], vec![1, 4, 6, 4, 1]);
        assert_eq!(lucas_number(0), 2);
        assert_eq!(lucas_number(1), 1);
        assert_eq!(lucas_number(4), 7);
        assert_eq!(tribonacci(0), 0);
        assert_eq!(tribonacci(3), 1);
        assert_eq!(tribonacci(4), 2);
        assert_eq!(catalan_number(0), 1);
        assert_eq!(catalan_number(3), 5);
        assert_eq!(catalan_number(5), 42);
    }

    #[test]
    fn partition_stirling_bell() {
        assert_eq!(partition_number(0), 1);
        assert_eq!(partition_number(4), 5);
        assert_eq!(partition_number(5), 7);
        assert_eq!(stirling_second(4, 2), 7);
        assert_eq!(stirling_first(4, 2), 11);
        assert_eq!(bell_number(0), 1);
        assert_eq!(bell_number(3), 5);
        assert_eq!(bell_number(4), 15);
        assert_eq!(subfactorial(0), 1);
        assert_eq!(subfactorial(1), 0);
        assert_eq!(subfactorial(4), 9);
    }

    #[test]
    fn permutation_divisor_functions() {
        assert_eq!(permutation_count(5, 3), 60);
        assert_eq!(permutation_count(5, 0), 1);
        assert_eq!(permutation_count(3, 5), 0);
        assert_eq!(divisor_count(12), 6);
        assert_eq!(divisor_count(1), 1);
        assert_eq!(divisor_sum(6), 12);
        assert_eq!(divisor_sum(12), 28);
        assert_eq!(radical(18), 6);
        assert_eq!(radical(7), 7);
        assert!(is_smooth(12, 3));
        assert!(!is_smooth(12, 2));
    }

    #[test]
    fn special_numbers() {
        assert_eq!(primorial(5), 30);
        assert_eq!(primorial(2), 2);
        assert_eq!(mersenne_number(3), 7);
        assert_eq!(fermat_number(0), 3);
        assert_eq!(fermat_number(1), 5);
        assert_eq!(double_factorial(5), 15);
        assert_eq!(double_factorial(6), 48);
    }

    #[test]
    fn mobius_liouville() {
        assert_eq!(mobius(1), 1);
        assert_eq!(mobius(2), -1);
        assert_eq!(mobius(4), 0);
        assert_eq!(mobius(6), 1);
        assert_eq!(mobius(30), -1);
        assert_eq!(liouville(1), 1);
        assert_eq!(liouville(6), 1);
        assert_eq!(liouville(8), -1);
    }

    #[test]
    fn segmented_sieve_test() {
        let primes = segmented_sieve(10, 30);
        assert_eq!(primes, vec![11, 13, 17, 19, 23, 29]);
        assert_eq!(segmented_sieve(1, 1), vec![]);
    }
}
