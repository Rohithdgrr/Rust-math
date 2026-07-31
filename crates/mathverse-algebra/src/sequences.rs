//! Arithmetic and geometric sequences, finite/infinite series, and
//! closed-form sums of powers.

/// `n`-th term of an arithmetic sequence: `a₁ + (n−1)d`.
///
/// ```
/// # use mathverse_algebra::sequences::arithmetic_nth_term;
/// assert_eq!(arithmetic_nth_term(2.0, 3.0, 5), 14.0); // 2, 5, 8, 11, 14
/// ```
pub fn arithmetic_nth_term(a1: f64, d: f64, n: usize) -> f64 {
    a1 + (n as f64 - 1.0) * d
}

/// `n`-th term of a geometric sequence: `a₁ · rⁿ⁻¹`.
///
/// ```
/// # use mathverse_algebra::sequences::geometric_nth_term;
/// assert_eq!(geometric_nth_term(3.0, 2.0, 4), 24.0); // 3, 6, 12, 24
/// ```
pub fn geometric_nth_term(a1: f64, r: f64, n: usize) -> f64 {
    a1 * r.powi(n as i32 - 1)
}

/// Sum of first `n` terms of an arithmetic sequence: `n/2 · (a₁ + aₙ)`.
///
/// ```
/// # use mathverse_algebra::sequences::arithmetic_sum;
/// assert_eq!(arithmetic_sum(1.0, 10.0, 10), 55.0); // 1+2+…+10
/// ```
pub fn arithmetic_sum(a1: f64, an: f64, n: usize) -> f64 {
    n as f64 / 2.0 * (a1 + an)
}

/// Sum of first `n` terms of a geometric sequence: `a₁(1 − rⁿ)/(1 − r)`.
/// Returns `None` when `r == 1` (use [`arithmetic_sum`] instead).
///
/// ```
/// # use mathverse_algebra::sequences::geometric_sum;
/// assert_eq!(geometric_sum(1.0, 2.0, 4), 15.0); // 1+2+4+8
/// ```
pub fn geometric_sum(a1: f64, r: f64, n: usize) -> Option<f64> {
    if (r - 1.0).abs() < 1e-15 {
        return None;
    }
    Some(a1 * (1.0 - r.powi(n as i32)) / (1.0 - r))
}

/// Infinite geometric series sum: `a₁/(1 − r)` when `|r| < 1`.
/// Returns `None` when the series diverges.
///
/// ```
/// # use mathverse_algebra::sequences::geometric_infinite_sum;
/// assert_eq!(geometric_infinite_sum(1.0, 0.5), Some(2.0)); // 1 + 0.5 + 0.25 + …
/// ```
pub fn geometric_infinite_sum(a1: f64, r: f64) -> Option<f64> {
    if r.abs() >= 1.0 {
        return None;
    }
    Some(a1 / (1.0 - r))
}

/// Sum of first `n` natural numbers: `n(n+1)/2`.
///
/// ```
/// # use mathverse_algebra::sequences::sum_natural;
/// assert_eq!(sum_natural(100), 5050.0);
/// ```
pub fn sum_natural(n: usize) -> f64 {
    n as f64 * (n as f64 + 1.0) / 2.0
}

/// Sum of first `n` squares: `n(n+1)(2n+1)/6`.
///
/// ```
/// # use mathverse_algebra::sequences::sum_squares;
/// assert_eq!(sum_squares(3), 14.0); // 1 + 4 + 9
/// ```
pub fn sum_squares(n: usize) -> f64 {
    let n = n as f64;
    n * (n + 1.0) * (2.0 * n + 1.0) / 6.0
}

/// Sum of first `n` cubes: `[n(n+1)/2]²`.
///
/// ```
/// # use mathverse_algebra::sequences::sum_cubes;
/// assert_eq!(sum_cubes(3), 36.0); // 1 + 8 + 27
/// ```
pub fn sum_cubes(n: usize) -> f64 {
    let s = sum_natural(n);
    s * s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        assert_eq!(arithmetic_nth_term(2.0, 3.0, 5), 14.0);
        assert_eq!(arithmetic_sum(1.0, 10.0, 10), 55.0);
    }

    #[test]
    fn geometric() {
        assert_eq!(geometric_nth_term(3.0, 2.0, 4), 24.0);
        assert_eq!(geometric_sum(1.0, 2.0, 4), Some(15.0));
        assert_eq!(geometric_infinite_sum(1.0, 0.5), Some(2.0));
        assert_eq!(geometric_infinite_sum(1.0, 2.0), None);
        assert_eq!(geometric_sum(1.0, 1.0, 5), None);
    }

    #[test]
    fn power_sums() {
        assert_eq!(sum_natural(100), 5050.0);
        assert_eq!(sum_squares(3), 14.0);
        assert_eq!(sum_cubes(3), 36.0);
        assert_eq!(sum_cubes(10), sum_natural(10) * sum_natural(10));
    }
}
