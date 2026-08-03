//! # Sequences
//!
//! Arithmetic and geometric sequences and series (finite sums).
//!
//! ## Examples
//!
//! ```rust
//! use mathverse_algebra::sequences::*;
//!
//! // Arithmetic: 1, 3, 5, 7, 9
//! assert_eq!(arithmetic_term(1.0, 2.0, 4), 9.0);
//! assert_eq!(arithmetic_sum(1.0, 2.0, 5), 25.0);
//!
//! // Geometric: 1, 2, 4, 8, 16
//! assert_eq!(geometric_term(1.0, 2.0, 4), 16.0);
//! assert_eq!(geometric_sum(1.0, 2.0, 5), 31.0);
//! ```

/// nth term of an arithmetic sequence: `a + n*d`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::sequences::arithmetic_term;
///
/// // 1, 3, 5, 7, 9 → 5th term = 1 + 4*2 = 9
/// assert_eq!(arithmetic_term(1.0, 2.0, 4), 9.0);
/// ```
#[inline]
#[must_use]
pub fn arithmetic_term(a: f64, d: f64, n: usize) -> f64 {
    a + (n as f64) * d
}

/// Sum of the first `n+1` terms of an arithmetic sequence: `(n+1)(2a + nd)/2`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::sequences::arithmetic_sum;
///
/// // 1 + 3 + 5 + 7 + 9 = 25
/// assert_eq!(arithmetic_sum(1.0, 2.0, 5), 25.0);
/// ```
#[inline]
#[must_use]
pub fn arithmetic_sum(a: f64, d: f64, n: usize) -> f64 {
    let count = n as f64;
    count * (2.0 * a + (count - 1.0) * d) / 2.0
}

/// nth term of a geometric sequence: `a * r^n`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::sequences::geometric_term;
///
/// // 1, 2, 4, 8, 16 → 5th term = 1 * 2^4 = 16
/// assert_eq!(geometric_term(1.0, 2.0, 4), 16.0);
/// ```
#[inline]
#[must_use]
pub fn geometric_term(a: f64, r: f64, n: usize) -> f64 {
    a * r.powi(n as i32)
}

/// Sum of the first `n` terms of a geometric sequence: `a(r^n - 1)/(r - 1)`.
///
/// Returns `0.0` for `n = 0`. Handles `r = 1` by returning `a * n`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::sequences::geometric_sum;
///
/// // 1 + 2 + 4 + 8 + 16 = 31
/// assert_eq!(geometric_sum(1.0, 2.0, 5), 31.0);
/// ```
#[must_use]
pub fn geometric_sum(a: f64, r: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let rn = r.powi(n as i32);
    if (r - 1.0).abs() < crate::TOL {
        a * n as f64
    } else {
        a * (rn - 1.0) / (r - 1.0)
    }
}

/// Infinite geometric series: `a / (1 - r)`, valid for `|r| < 1`.
///
/// Returns `None` if `|r| >= 1`.
///
/// # Examples
///
/// ```rust
/// use mathverse_algebra::sequences::geometric_infinite_sum;
///
/// // 1 + 1/2 + 1/4 + 1/8 + ... = 2
/// assert_eq!(geometric_infinite_sum(1.0, 0.5), Some(2.0));
/// ```
#[must_use]
pub fn geometric_infinite_sum(a: f64, r: f64) -> Option<f64> {
    if r.abs() >= 1.0 {
        None
    } else {
        Some(a / (1.0 - r))
    }
}

/// Check if a sequence is arithmetic by verifying constant difference.
#[must_use]
pub fn is_arithmetic(seq: &[f64]) -> bool {
    if seq.len() < 2 {
        return true;
    }
    let d = seq[1] - seq[0];
    seq.windows(2).all(|w| (w[1] - w[0] - d).abs() < crate::TOL)
}

/// Check if a sequence is geometric by verifying constant ratio.
#[must_use]
pub fn is_geometric(seq: &[f64]) -> bool {
    if seq.len() < 2 {
        return true;
    }
    if seq.iter().any(|&x| x.abs() < crate::TOL) {
        return false;
    }
    let r = seq[1] / seq[0];
    seq.windows(2).all(|w| (w[1] / w[0] - r).abs() < crate::TOL)
}

/// Common difference of an arithmetic sequence.
///
/// Returns `None` if the sequence is not arithmetic or has fewer than 2 elements.
#[must_use]
pub fn common_difference(seq: &[f64]) -> Option<f64> {
    if seq.len() < 2 || !is_arithmetic(seq) {
        return None;
    }
    Some(seq[1] - seq[0])
}

/// Common ratio of a geometric sequence.
///
/// Returns `None` if the sequence is not geometric or has fewer than 2 elements.
#[must_use]
pub fn common_ratio(seq: &[f64]) -> Option<f64> {
    if seq.len() < 2 || !is_geometric(seq) {
        return None;
    }
    Some(seq[1] / seq[0])
}

/// Arithmetic mean (average) of a sequence.
#[must_use]
pub fn arithmetic_mean(seq: &[f64]) -> f64 {
    if seq.is_empty() {
        return 0.0;
    }
    seq.iter().sum::<f64>() / seq.len() as f64
}

/// Geometric mean of a positive sequence.
///
/// Returns `None` if any element is non-positive.
#[must_use]
pub fn geometric_mean(seq: &[f64]) -> Option<f64> {
    if seq.is_empty() || seq.iter().any(|&x| x <= 0.0) {
        return None;
    }
    let product: f64 = seq.iter().product();
    Some(product.powf(1.0 / seq.len() as f64))
}

/// Harmonic mean of a positive sequence.
///
/// Returns `None` if any element is non-positive.
#[must_use]
pub fn harmonic_mean(seq: &[f64]) -> Option<f64> {
    if seq.is_empty() || seq.iter().any(|&x| x <= 0.0) {
        return None;
    }
    let sum: f64 = seq.iter().map(|&x| 1.0 / x).sum();
    Some(seq.len() as f64 / sum)
}

/// n-th Fibonacci number.
#[must_use]
pub fn fibonacci(n: usize) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let (mut a, mut b) = (0u64, 1u64);
    for _ in 2..=n {
        let tmp = a + b;
        a = b;
        b = tmp;
    }
    b
}

/// n-th triangular number: `n(n+1)/2`.
#[must_use]
pub fn triangular(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Partial sums of a sequence.
#[must_use]
pub fn partial_sums(seq: &[f64]) -> Vec<f64> {
    let mut sums = Vec::with_capacity(seq.len());
    let mut running = 0.0;
    for &x in seq {
        running += x;
        sums.push(running);
    }
    sums
}

/// Generalized power sequence: `[1^n, 2^n, 3^n, ..., count^n]`.
#[must_use]
pub fn power_sequence(n: u32, count: usize) -> Vec<f64> {
    (1..=count).map(|i| (i as f64).powi(n as i32)).collect()
}

/// Alternating sequence: `[1, -1, 1, -1, ...]` of length `n`.
#[must_use]
pub fn alternating(n: usize) -> Vec<f64> {
    (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect()
}

/// Factorial of `n`.
#[must_use]
pub fn factorial(n: u64) -> u64 {
    match n {
        0 | 1 => 1,
        _ => (2..=n).product(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_tests() {
        assert_eq!(arithmetic_term(1.0, 2.0, 0), 1.0);
        assert_eq!(arithmetic_term(1.0, 2.0, 4), 9.0);
        assert_eq!(arithmetic_sum(1.0, 2.0, 5), 25.0);
        assert!(is_arithmetic(&[1.0, 3.0, 5.0, 7.0, 9.0]));
        assert!(!is_arithmetic(&[1.0, 2.0, 4.0, 8.0]));
        assert_eq!(common_difference(&[1.0, 3.0, 5.0]), Some(2.0));
    }

    #[test]
    fn geometric_tests() {
        assert_eq!(geometric_term(1.0, 2.0, 0), 1.0);
        assert_eq!(geometric_term(1.0, 2.0, 4), 16.0);
        assert_eq!(geometric_sum(1.0, 2.0, 5), 31.0);
        assert_eq!(geometric_infinite_sum(1.0, 0.5), Some(2.0));
        assert_eq!(geometric_infinite_sum(1.0, 1.5), None);
        assert!(is_geometric(&[1.0, 2.0, 4.0, 8.0]));
        assert!(!is_geometric(&[1.0, 2.0, 3.0]));
        assert_eq!(common_ratio(&[1.0, 2.0, 4.0]), Some(2.0));
    }

    #[test]
    fn means() {
        assert_eq!(arithmetic_mean(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(geometric_mean(&[1.0, 2.0, 4.0]).unwrap(), 2.0);
        // Harmonic mean of [1,2,4] = 3 / (1 + 1/2 + 1/4) = 12/7
        assert_eq!(harmonic_mean(&[1.0, 2.0, 4.0]).unwrap(), 12.0 / 7.0);
    }

    #[test]
    fn combinatorics() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(5), 120);
        assert_eq!(fibonacci(10), 55);
        assert_eq!(triangular(10), 55);
    }

    #[test]
    fn power_and_alternating() {
        assert_eq!(power_sequence(2, 3), vec![1.0, 4.0, 9.0]);
        assert_eq!(alternating(4), vec![1.0, -1.0, 1.0, -1.0]);
    }
}
