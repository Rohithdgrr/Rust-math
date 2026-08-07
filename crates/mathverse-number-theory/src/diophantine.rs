//! Extended GCD, linear Diophantine equations, Pell equation, square-free test.

/// Extended Euclidean algorithm: returns `(g, x, y)` such that
/// `a·x + b·y = g = gcd(|a|, |b|)`.
///
/// `g` (the GCD) is always non-negative.
///
/// ```
/// use mathverse_number_theory::extended_gcd;
/// let (g, x, y) = extended_gcd(48, 18);
/// assert_eq!(g, 6);
/// assert_eq!(48 * x + 18 * y, 6);
/// ```
#[must_use]
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x, y) = extended_gcd(b % a, a);
    (g, y - (b / a) * x, x)
}

/// Solves the linear Diophantine equation `a·x + b·y = c`.
///
/// Returns `Some((x, y))` — one particular solution — or `None` if no
/// solution exists (i.e., `gcd(a, b)` does not divide `c`).
///
/// ```
/// use mathverse_number_theory::solve_linear_diophantine;
/// let (x, y) = solve_linear_diophantine(6, 10, 14).unwrap();
/// assert_eq!(6 * x + 10 * y, 14);
/// assert!(solve_linear_diophantine(6, 10, 13).is_none());
/// ```
#[must_use]
pub fn solve_linear_diophantine(a: i64, b: i64, c: i64) -> Option<(i64, i64)> {
    let (g, x, y) = extended_gcd(a.abs(), b.abs());
    if c % g != 0 {
        return None;
    }
    let mult = c / g;
    let x = x * mult * a.signum();
    let y = y * mult * b.signum();
    Some((x, y))
}

/// Returns `true` if `n` is square-free (no repeated prime factor).
///
/// ```
/// use mathverse_number_theory::is_square_free;
/// assert!(is_square_free(6));    // 2 × 3
/// assert!(!is_square_free(12));  // 2² × 3
/// assert!(is_square_free(1));    // vacuously true
/// ```
#[must_use]
pub fn is_square_free(n: u64) -> bool {
    if n == 0 {
        return false;
    }
    if n == 1 {
        return true;
    }
    let factors = crate::prime_factors(n);
    let mut i = 0;
    while i < factors.len() {
        let p = factors[i];
        let mut count = 0;
        while i < factors.len() && factors[i] == p {
            count += 1;
            i += 1;
        }
        if count > 1 {
            return false;
        }
    }
    true
}

/// Kronecker symbol `(a/n)` — extension of the Jacobi symbol to all integers.
///
/// ```
/// use mathverse_number_theory::kronecker;
/// assert_eq!(kronecker(1, 1), 1);
/// assert_eq!(kronecker(0, 0), 0);
/// ```
#[must_use]
pub fn kronecker(a: u64, n: u64) -> i64 {
    if n == 0 {
        return if a == 1 { 1 } else { 0 };
    }
    if a == 0 {
        return if n == 1 { 1 } else { 0 };
    }
    crate::jacobi(a, n)
}

/// Quadratic reciprocity: computes `(p/q) * (q/p)` via Legendre symbols.
/// Returns `0` if either argument is `2` or not prime.
///
/// ```
/// use mathverse_number_theory::quadratic_reciprocity;
/// assert_eq!(quadratic_reciprocity(3, 5), 1);
/// assert_eq!(quadratic_reciprocity(3, 7), -1);
/// ```
#[must_use]
pub fn quadratic_reciprocity(p: u64, q: u64) -> i64 {
    if p == 2 || q == 2 {
        return 0;
    }
    crate::legendre(p, q) * crate::legendre(q, p)
}

/// Finds the fundamental solution `(x, y)` to Pell's equation `x² - d·y² = 1`.
///
/// Returns `None` if `d` is a perfect square (no non-trivial solution) or
/// if the solution does not fit in `u128`. With the `bigint` feature,
/// [`pell_fundamental_big`] supports arbitrary precision.
///
/// ```
/// use mathverse_number_theory::pell_fundamental;
/// let (x, y) = pell_fundamental(2).unwrap();
/// assert_eq!(x * x - 2 * y * y, 1);
/// ```
#[must_use]
pub fn pell_fundamental(d: u64) -> Option<(u128, u128)> {
    if d == 0 {
        return Some((1, 0));
    }
    let a0 = crate::isqrt(d);
    if a0 * a0 == d {
        return None; // d is a perfect square
    }
    let d128 = d as u128;
    let a0_128 = a0 as u128;
    let mut m: u128 = 0;
    let mut dd: u128 = 1;
    let mut a: u128 = a0_128;

    let (mut h_prev, mut h_prev2) = (1u128, 0u128);
    let (mut k_prev, mut k_prev2) = (0u128, 1u128);

    for _ in 0..50_000 {
        let h_n = a.checked_mul(h_prev)?.checked_add(h_prev2)?;
        let k_n = a.checked_mul(k_prev)?.checked_add(k_prev2)?;

        // Verify: h_n² - d·k_n² == 1
        // Use wrapping arithmetic for the verification step so that
        // solutions whose square overflows u128 are still detected.
        let h_sq = h_n.wrapping_mul(h_n);
        let k_sq = k_n.wrapping_mul(k_n);
        let d_k_sq = d128.wrapping_mul(k_sq);
        if h_sq.wrapping_sub(d_k_sq) == 1 {
            return Some((h_n, k_n));
        }

        m = dd.checked_mul(a)?.checked_sub(m)?;
        dd = d128.checked_sub(m.checked_mul(m)?)?.checked_div(dd)?;
        a = a0_128.checked_add(m)?.checked_div(dd)?;

        h_prev2 = h_prev;
        h_prev = h_n;
        k_prev2 = k_prev;
        k_prev = k_n;
    }
    None
}

/// Same as `pell_fundamental` but uses `num-bigint` for arbitrary precision.
/// Only available with the `bigint` feature.
///
/// ```
/// use mathverse_number_theory::pell_fundamental_big;
/// let (x, y) = pell_fundamental_big(2).unwrap();
/// assert_eq!(x.to_string(), "3");
/// assert_eq!(y.to_string(), "2");
/// ```
#[cfg(feature = "bigint")]
#[must_use]
pub fn pell_fundamental_big(d: u64) -> Option<(num_bigint::BigUint, num_bigint::BigUint)> {
    use num_bigint::BigUint;
    let a0 = crate::isqrt(d);
    if a0 * a0 == d {
        return None;
    }
    let d_big = BigUint::from(d);
    let a0_big = BigUint::from(a0);

    let mut m = BigUint::from(0u64);
    let mut dd = BigUint::from(1u64);
    let mut a = a0_big.clone();

    let (mut h_prev, mut h_prev2) = (BigUint::from(1u64), BigUint::from(0u64));
    let (mut k_prev, mut k_prev2) = (BigUint::from(0u64), BigUint::from(1u64));

    for _ in 0..10_000 {
        let h_n = &a * &h_prev + &h_prev2;
        let k_n = &a * &k_prev + &k_prev2;

        let h_sq = &h_n * &h_n;
        let d_k_sq = &d_big * &k_n * &k_n;
        if h_sq == d_k_sq + 1u64 {
            return Some((h_n, k_n));
        }

        m = &dd * &a - &m;
        dd = (&d_big - &m * &m) / &dd;
        a = (&a0_big + &m) / &dd;

        h_prev2 = h_prev;
        h_prev = h_n;
        k_prev2 = k_prev;
        k_prev = k_n;
    }
    None
}

/// Euler-Kronecker (also known as the Euler totient form of the Jacobi symbol).
///
/// This is equivalent to `jacobi(a, n)` for positive `n`.
///
/// ```
/// use mathverse_number_theory::euler_kronecker;
/// assert_eq!(euler_kronecker(1, 9), 1);
/// assert_eq!(euler_kronecker(2, 9), 1);
/// ```
#[must_use]
pub fn euler_kronecker(a: i64, n: u64) -> i64 {
    crate::jacobi(a as u64, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diophantine() {
        let (x, y) = solve_linear_diophantine(6, 10, 14).unwrap();
        assert_eq!(6 * x + 10 * y, 14);
    }

    #[test]
    fn square_free() {
        assert!(is_square_free(6));
        assert!(!is_square_free(12));
        assert!(is_square_free(1));
        assert!(!is_square_free(0)); // 0 = 0^2, not squarefree by convention... actually 0 has no prime factorization
    }

    #[test]
    fn pell() {
        let (x, y) = pell_fundamental(2).unwrap();
        assert_eq!(x * x - 2 * y * y, 1);
        assert_eq!(x, 3);
        assert_eq!(y, 2);
    }

    #[test]
    fn pell_larger() {
        let (x, y) = pell_fundamental(61).unwrap();
        assert_eq!(x * x - 61 * y * y, 1);
    }

    #[test]
    fn pell_perfect_square() {
        assert_eq!(pell_fundamental(4), None);
        assert_eq!(pell_fundamental(9), None);
        assert_eq!(pell_fundamental(1), None);
    }

    #[test]
    fn kronecker_test() {
        assert_eq!(kronecker(0, 0), 0);
        assert_eq!(kronecker(1, 0), 1);
        assert_eq!(kronecker(1, 1), 1);
    }

    #[test]
    fn reciprocity_test() {
        assert_eq!(quadratic_reciprocity(3, 5), 1);
        assert_eq!(quadratic_reciprocity(3, 7), -1);
        assert_eq!(quadratic_reciprocity(2, 3), 0);
    }
}