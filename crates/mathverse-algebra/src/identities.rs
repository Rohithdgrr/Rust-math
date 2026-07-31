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

/// `(a - b)² = a² - 2ab + b²`.
pub fn square_of_difference(a: f64, b: f64) -> f64 {
    a * a - 2.0 * a * b + b * b
}

/// Difference of squares: `a² - b² = (a - b)(a + b)`.
///
/// Returns `(a - b, a + b)` — the two linear factors.
///
/// ```
/// # use mathverse_algebra::identities::difference_of_squares;
/// let (f1, f2) = difference_of_squares(5.0, 3.0); // 25 - 9 = 16
/// assert!((f1 * f2 - 16.0).abs() < 1e-12);
/// ```
pub fn difference_of_squares(a: f64, b: f64) -> (f64, f64) {
    (a - b, a + b)
}

/// Perfect square trinomial: `(a ± b)² = a² ± 2ab + b²`.
///
/// Returns `(a + b)²` and `(a - b)²`.
pub fn perfect_square_trinomial(a: f64, b: f64) -> (f64, f64) {
    (square_of_sum(a, b), square_of_difference(a, b))
}

/// Sum of cubes: `a³ + b³ = (a + b)(a² − ab + b²)`.
///
/// Returns `(linear_factor, quadratic_factor_value)`.
///
/// ```
/// # use mathverse_algebra::identities::sum_of_cubes;
/// let (lin, quad) = sum_of_cubes(2.0, 3.0); // 8 + 27 = 35
/// assert!((lin * quad - 35.0).abs() < 1e-12);
/// ```
pub fn sum_of_cubes(a: f64, b: f64) -> (f64, f64) {
    let linear = a + b;
    let quadratic = a * a - a * b + b * b;
    (linear, quadratic)
}

/// Difference of cubes: `a³ - b³ = (a - b)(a² + ab + b²)`.
///
/// Returns `(linear_factor, quadratic_factor_value)`.
pub fn difference_of_cubes(a: f64, b: f64) -> (f64, f64) {
    let linear = a - b;
    let quadratic = a * a + a * b + b * b;
    (linear, quadratic)
}

/// `(a + b)³ = a³ + 3a²b + 3ab² + b³`.
pub fn cube_of_sum(a: f64, b: f64) -> f64 {
    a * a * a + 3.0 * a * a * b + 3.0 * a * b * b + b * b * b
}

/// `(a - b)³ = a³ - 3a²b + 3ab² - b³`.
pub fn cube_of_difference(a: f64, b: f64) -> f64 {
    a * a * a - 3.0 * a * a * b + 3.0 * a * b * b - b * b * b
}

/// `(a + b)⁴ = a⁴ + 4a³b + 6a²b² + 4ab³ + b⁴`.
pub fn fourth_power_sum(a: f64, b: f64) -> f64 {
    a.powi(4) + 4.0 * a.powi(3) * b + 6.0 * a * a * b * b + 4.0 * a * b.powi(3) + b.powi(4)
}

/// `(a + b + c)² = a² + b² + c² + 2ab + 2bc + 2ca`.
pub fn square_of_trinomial(a: f64, b: f64, c: f64) -> f64 {
    a * a + b * b + c * c + 2.0 * a * b + 2.0 * b * c + 2.0 * c * a
}

/// Sum/difference of nth powers for odd `n`:
/// `aⁿ + bⁿ = (a + b)(aⁿ⁻¹ - aⁿ⁻²b + … + bⁿ⁻¹)`.
///
/// Returns the two factors as `(linear, remaining)`.
pub fn sum_of_nth_powers(a: f64, b: f64, n: usize) -> Option<(f64, f64)> {
    if n == 0 || n % 2 == 0 {
        return None;
    }
    let linear = a + b;
    let mut quad = 0.0;
    for k in 0..n {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        quad += sign * a.powi((n - 1 - k) as i32) * b.powi(k as i32);
    }
    Some((linear, quad))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn pascal() {
        let t = pascal_triangle(4);
        assert_eq!(t[0], vec![1]);
        assert_eq!(t[3], vec![1, 3, 3, 1]);
        assert_eq!(t[4], vec![1, 4, 6, 4, 1]);
    }

    #[test]
    fn binomial_exp() {
        let terms = binomial_expand(1.0, 1.0, 4); // (1+1)⁴ = 16
        assert!((terms.iter().sum::<f64>() - 16.0).abs() < 1e-12);
    }

    #[test]
    fn squares() {
        assert!(approx(square_of_sum(3.0, 4.0), 49.0));
        assert!(approx(square_of_difference(3.0, 4.0), 1.0));
    }

    #[test]
    fn diff_sq() {
        let (f1, f2) = difference_of_squares(5.0, 3.0);
        assert!(approx(f1 * f2, 16.0));
    }

    #[test]
    fn cubes() {
        let (l, q) = sum_of_cubes(2.0, 3.0);
        assert!(approx(l * q, 35.0));
        let (l, q) = difference_of_cubes(3.0, 2.0);
        assert!(approx(l * q, 19.0));
    }

    #[test]
    fn cube_powers() {
        assert!(approx(cube_of_sum(2.0, 3.0), 125.0));
        assert!(approx(cube_of_difference(3.0, 2.0), 1.0));
    }

    #[test]
    fn fourth_power() {
        assert!(approx(fourth_power_sum(1.0, 2.0), 81.0));
    }

    #[test]
    fn trinomial_square() {
        assert!(approx(square_of_trinomial(1.0, 2.0, 3.0), 36.0));
    }

    #[test]
    fn nth_powers_odd() {
        let (l, q) = sum_of_nth_powers(2.0, 1.0, 3).unwrap();
        assert!(approx(l * q, 9.0)); // 8 + 1 = 9
    }
}
