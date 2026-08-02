//! Extended GCD, linear Diophantine equations, Pell equation, square-free test.

/// Extended Euclidean algorithm: returns (gcd, x, y) such that `ax + by = gcd`.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 { return (b, 0, 1); }
    let (g, x, y) = extended_gcd(b % a, a);
    (g, y - (b / a) * x, x)
}

pub fn solve_linear_diophantine(a: i64, b: i64, c: i64) -> Option<(i64, i64)> {
    let (g, x, y) = extended_gcd(a.abs(), b.abs());
    if c % g != 0 { return None; }
    let mult = c / g;
    Some((x * mult * if a < 0 { -1 } else { 1 }, y * mult * if b < 0 { -1 } else { 1 }))
}

pub fn is_square_free(n: u64) -> bool {
    let factors = crate::factorization::prime_factors(n);
    factors.len() == factors.iter().collect::<std::collections::HashSet<_>>().len()
}

pub fn kronecker(a: u64, b: u64) -> i64 {
    if b == 0 { return if a == 1 { 1 } else { 0 }; }
    if a == 0 { return if b == 1 { 1 } else { 0 }; }
    crate::quadratic_residue::jacobi(a, b)
}

pub fn quadratic_reciprocity(p: u64, q: u64) -> i64 {
    if p == 2 || q == 2 { return 0; }
    let ls = crate::quadratic_residue::legendre(p, q);
    ls * crate::quadratic_residue::legendre(q, p)
}

pub fn pell_fundamental(d: u64) -> Option<(u64, u64)> {
    let sqrt_d = (d as f64).sqrt();
    let a0 = sqrt_d.floor() as u64;
    if a0 * a0 == d { return None; }
    let mut m = 0u64;
    let mut dd = 1u64;
    let mut a = a0;
    let (mut h_prev, mut h_prev2) = (1u128, 0u128);
    let (mut k_prev, mut k_prev2) = (0u128, 1u128);
    for _ in 0..10000 {
        let h_n = a as u128 * h_prev + h_prev2;
        let k_n = a as u128 * k_prev + k_prev2;
        if k_n > 0 && h_n * h_n as u128 == d as u128 * k_n * k_n + 1 {
            return Some((h_n as u64, k_n as u64));
        }
        m = dd * a - m;
        dd = (d - m * m) / dd;
        a = (a0 + m) / dd;
        h_prev2 = h_prev; h_prev = h_n;
        k_prev2 = k_prev; k_prev = k_n;
    }
    None
}

pub fn euler_kronecker(a: i64, n: u64) -> i64 {
    crate::quadratic_residue::jacobi(a as u64, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diophantine() {
        let (x, y) = solve_linear_diophantine(6, 10, 14).unwrap();
        assert!(6 * x + 10 * y == 14);
    }

    #[test]
    fn square_free() {
        assert!(is_square_free(6));
        assert!(!is_square_free(12));
    }

    #[test]
    fn pell() {
        let (x, y) = pell_fundamental(2).unwrap();
        assert!(x * x == 2 * y * y + 1);
    }
}
