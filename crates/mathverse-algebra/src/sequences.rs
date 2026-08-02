//! Arithmetic and geometric sequences and series.

/// The `n`th term of an arithmetic sequence: `aₙ = a₁ + (n-1)d`.
///
/// ```
/// # use mathverse_algebra::sequences::arithmetic_nth_term;
/// assert_eq!(arithmetic_nth_term(2.0, 3.0, 5), 14.0); // 2, 5, 8, 11, 14
/// ```
#[must_use]
pub fn arithmetic_nth_term(a1: f64, d: f64, n: usize) -> f64 {
    a1 + (n as f64 - 1.0) * d
}

/// Sum of first `n` terms of an arithmetic sequence.
///
/// `Sₙ = n/2 · (2a₁ + (n-1)d)`.
///
/// ```
/// # use mathverse_algebra::sequences::arithmetic_sum;
/// assert_eq!(arithmetic_sum(1.0, 1.0, 10), 55.0); // 1+2+...+10
/// ```
#[must_use]
pub fn arithmetic_sum(a1: f64, d: f64, n: usize) -> f64 {
    let n_f = n as f64;
    n_f / 2.0 * (2.0 * a1 + (n_f - 1.0) * d)
}

/// Sum of first `n` natural numbers: `n(n+1)/2`.
///
/// ```
/// # use mathverse_algebra::sequences::sum_natural;
/// assert_eq!(sum_natural(10), 55);
/// ```
#[must_use]
pub fn sum_natural(n: u64) -> u64 {
    n * (n + 1) / 2
}

/// Sum of squares of first `n` natural numbers: `n(n+1)(2n+1)/6`.
///
/// ```
/// # use mathverse_algebra::sequences::sum_squares;
/// assert_eq!(sum_squares(10), 385);
/// ```
#[must_use]
pub fn sum_squares(n: u64) -> u64 {
    n * (n + 1) * (2 * n + 1) / 6
}

/// Sum of cubes of first `n` natural numbers: `[n(n+1)/2]²`.
///
/// ```
/// # use mathverse_algebra::sequences::sum_cubes;
/// assert_eq!(sum_cubes(10), 3025);
/// ```
#[must_use]
pub fn sum_cubes(n: u64) -> u64 {
    let s = n * (n + 1) / 2;
    s * s
}

/// The `n`th term of a geometric sequence: `aₙ = a₁ · r^(n-1)`.
///
/// ```
/// # use mathverse_algebra::sequences::geometric_nth_term;
/// assert!((geometric_nth_term(2.0, 3.0, 4) - 54.0).abs() < 1e-9); // 2, 6, 18, 54
/// ```
#[must_use]
pub fn geometric_nth_term(a1: f64, r: f64, n: usize) -> f64 {
    a1 * r.powi(n as i32 - 1)
}

/// Sum of first `n` terms of a geometric sequence.
///
/// `Sₙ = a₁(1 - rⁿ)/(1 - r)` for `r ≠ 1`.
///
/// ```
/// # use mathverse_algebra::sequences::geometric_sum;
/// assert!((geometric_sum(1.0, 2.0, 10) - 1023.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn geometric_sum(a1: f64, r: f64, n: usize) -> f64 {
    if (r - 1.0).abs() < 1e-12 {
        a1 * n as f64
    } else {
        a1 * (1.0 - r.powi(n as i32)) / (1.0 - r)
    }
}

/// Sum of an infinite geometric series.
///
/// Returns `Some(a₁/(1-r))` if `|r| < 1`, else `None`.
///
/// ```
/// # use mathverse_algebra::sequences::geometric_infinite_sum;
/// assert!((geometric_infinite_sum(1.0, 0.5).unwrap() - 2.0).abs() < 1e-9);
/// assert!(geometric_infinite_sum(1.0, 2.0).is_none());
/// ```
#[must_use]
pub fn geometric_infinite_sum(a1: f64, r: f64) -> Option<f64> {
    if r.abs() < 1.0 {
        Some(a1 / (1.0 - r))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        assert_eq!(arithmetic_nth_term(2.0, 3.0, 5), 14.0);
        assert_eq!(arithmetic_sum(1.0, 1.0, 10), 55.0);
    }

    #[test]
    fn natural_sums() {
        assert_eq!(sum_natural(10), 55);
        assert_eq!(sum_squares(10), 385);
        assert_eq!(sum_cubes(10), 3025);
        for n in 1..=20 {
            assert_eq!(sum_cubes(n), sum_natural(n).pow(2));
        }
    }

    #[test]
    fn geometric() {
        assert!((geometric_nth_term(2.0, 3.0, 4) - 54.0).abs() < 1e-9);
        assert!((geometric_sum(1.0, 2.0, 10) - 1023.0).abs() < 1e-9);
        assert!((geometric_infinite_sum(1.0, 0.5).unwrap() - 2.0).abs() < 1e-9);
        assert!(geometric_infinite_sum(1.0, 2.0).is_none());
    }

    #[test]
    fn geometric_r_eq_1() {
        assert_eq!(geometric_sum(5.0, 1.0, 100), 500.0);
    }
}