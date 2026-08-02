//! Algebraic identity expansions and factorizations.
//!
//! Each function returns the *evaluated numeric result* or the *factored
//! components* so callers can inspect the identity directly.

use mathverse_core::algorithms::binomial;

/// Generate Pascal's triangle up to row `n` (inclusive, 0-indexed).
///
/// Row `k` contains `C(k, 0), C(k, 1), …, C(k, k)`.
///
/// ```
/// # use mathverse_algebra::identities::pascal_triangle;
/// let t = pascal_triangle(4);
/// assert_eq!(t[2], vec![1, 2, 1]);
/// assert_eq!(t[4], vec![1, 4, 6, 4, 1]);
/// ```
pub fn pascal_triangle(n: usize) -> Vec<Vec<u64>> {
    (0..=n).map(|k| (0..=k).map(|j| binomial(k as u64, j as u64) as u64).collect()).collect()
}

/// Binomial coefficient `C(n, k)` as `f64`.
pub fn binomial_coeff(n: u64, k: u64) -> f64 {
    binomial(n, k) as f64
}

/// Expand `(a + b)ⁿ` and return the terms `[C(n,0)aⁿ, C(n,1)aⁿ⁻¹b, …, C(n,n)bⁿ]`.
///
/// ```
/// # use mathverse_algebra::identities::binomial_expand;
/// let terms = binomial_expand(1.0, 2.0, 3); // (1+2)³ = 27
/// let sum: f64 = terms.iter().sum();
/// assert!((sum - 27.0).abs() < 1e-12);
/// ```
pub fn binomial_expand(a: f64, b: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|k| {
            let c = binomial(n as u64, k as u64) as f64;
            c * a.powi((n - k) as i32) * b.powi(k as i32)
        })
        .collect()
}

/// `(a + b)² = a² + 2ab + b²` — returns the expanded value.
pub fn square_of_sum(a: f64, b: f64) -> f64 {
    a * a + 2.0 * a * b + b * b
}

/// `(a - b)² = a² - 2ab + b²` — returns the expanded value.
pub fn square_of_difference(a: f64, b: f64) -> f64 {
    a * a - 2.0 * a * b + b * b
}

/// `a² - b² = (a - b)(a + b)` — returns the two factors `(a - b, a + b)`.
pub fn difference_of_squares(a: f64, b: f64) -> (f64, f64) {
    (a - b, a + b)
}

/// `(a + b)³ = a³ + 3a²b + 3ab² + b³` — returns the expanded value.
pub fn cube_of_sum(a: f64, b: f64) -> f64 {
    a.powi(3) + 3.0 * a * a * b + 3.0 * a * b * b + b.powi(3)
}

/// `(a - b)³ = a³ - 3a²b + 3ab² - b³` — returns the expanded value.
pub fn cube_of_difference(a: f64, b: f64) -> f64 {
    a.powi(3) - 3.0 * a * a * b + 3.0 * a * b * b - b.powi(3)
}

/// `a³ + b³ = (a + b)(a² - ab + b²)` — returns the evaluated numeric result.
pub fn sum_of_cubes(a: f64, b: f64) -> f64 {
    a.powi(3) + b.powi(3)
}

/// `a³ - b³ = (a - b)(a² + ab + b²)` — returns the evaluated numeric result.
pub fn difference_of_cubes(a: f64, b: f64) -> f64 {
    a.powi(3) - b.powi(3)
}

/// `aⁿ + bⁿ` factorization for odd `n`: returns `(a + b, remaining_factor)`.
///
/// For even `n`, returns `None` since `aⁿ + bⁿ` is not factorable over the reals.
pub fn sum_of_nth_powers(a: f64, b: f64, n: usize) -> Option<(f64, f64)> {
    if n % 2 == 0 {
        return None;
    }
    let sum = a + b;
    let remaining = (a.powi(n as i32) + b.powi(n as i32)) / sum;
    Some((sum, remaining))
}

/// Factor a quadratic `ax² + bx + c` into `(px + q)(rx + s)` if possible.
///
/// Returns `Some((p, q, r, s))` such that `a = pr`, `c = qs`, `b = ps + qr`.
/// Returns `None` if no integer-like factorization exists.
pub fn factor_quadratic(a: f64, b: f64, c: f64) -> Option<(f64, f64, f64, f64)> {
    let a_pairs = factor_pairs(a);
    let c_pairs = factor_pairs(c);
    for &(p, r) in &a_pairs {
        for &(q, s) in &c_pairs {
            if (p * s + q * r - b).abs() < 1e-9 {
                return Some((p, q, r, s));
            }
        }
    }
    None
}

/// All factor pairs of `n` (including negatives).
fn factor_pairs(n: f64) -> Vec<(f64, f64)> {
    let n_abs = n.abs();
    let limit = (n_abs.sqrt() + 1.0) as i64;
    let mut pairs = Vec::new();
    for i in 1..=limit {
        let fi = i as f64;
        if (n_abs / fi).fract() < 1e-9 {
            let j = n_abs / fi;
            pairs.push((fi, j));
            pairs.push((-fi, -j));
            if fi != j {
                pairs.push((j, fi));
                pairs.push((-j, -fi));
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pascal() {
        let t = pascal_triangle(4);
        assert_eq!(t[0], vec![1]);
        assert_eq!(t[1], vec![1, 1]);
        assert_eq!(t[2], vec![1, 2, 1]);
        assert_eq!(t[3], vec![1, 3, 3, 1]);
        assert_eq!(t[4], vec![1, 4, 6, 4, 1]);
    }

    #[test]
    fn binomial_identity() {
        let terms = binomial_expand(2.0, 3.0, 4);
        let sum: f64 = terms.iter().sum();
        assert!((sum - 625.0).abs() < 1e-9); // (2+3)^4 = 625
    }

    #[test]
    fn square_identities() {
        assert!((square_of_sum(2.0, 3.0) - 25.0).abs() < 1e-9);
        assert!((square_of_difference(5.0, 3.0) - 4.0).abs() < 1e-9);
        let (a, b) = difference_of_squares(5.0, 3.0);
        assert!((a - 2.0).abs() < 1e-9);
        assert!((b - 8.0).abs() < 1e-9);
    }

    #[test]
    fn cube_identities() {
        assert!((cube_of_sum(1.0, 2.0) - 27.0).abs() < 1e-9);
        assert!((cube_of_difference(3.0, 1.0) - 8.0).abs() < 1e-9);
        assert!((sum_of_cubes(2.0, 3.0) - 35.0).abs() < 1e-9);
        assert!((difference_of_cubes(3.0, 2.0) - 19.0).abs() < 1e-9);
    }

    #[test]
    fn sum_nth_powers() {
        let (sum, remaining) = sum_of_nth_powers(2.0, 1.0, 5).unwrap();
        assert!((sum - 3.0).abs() < 1e-9);
        assert!((remaining - 11.0).abs() < 1e-9);
        assert!(sum_of_nth_powers(2.0, 1.0, 4).is_none());
    }

    #[test]
    fn factor_quad() {
        let f = factor_quadratic(1.0, -5.0, 6.0).unwrap(); // x^2 - 5x + 6 = (x-2)(x-3)
        assert!((f.0 * f.2 - 1.0).abs() < 1e-9); // a = p*r
        assert!((f.1 * f.3 - 6.0).abs() < 1e-9); // c = q*s
        assert!((f.0 * f.3 + f.1 * f.2 - (-5.0)).abs() < 1e-9); // b = ps + qr
    }
}