//! Rayon-based parallel reductions behind the `parallel` feature.
//!
//! Enabling the `parallel` feature routes the crate's O(n) reductions (dot
//! product, sums, magnitudes, distances, mean) through rayon parallel
//! iterators when the input is long enough to amortize thread-pool overhead
//! (see [`THRESHOLD`]). Smaller inputs keep the scalar path.

use rayon::prelude::*;

/// Minimum slice length for which the parallel path is used.
pub const THRESHOLD: usize = 4096;

/// Dot product of two slices, truncating to the shorter length.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).sum()
}

/// Sum of all elements.
pub fn sum(v: &[f64]) -> f64 { v.par_iter().sum() }

/// Sum of squares.
pub fn sum_sq(v: &[f64]) -> f64 { v.par_iter().map(|x| x * x).sum() }

/// Sum of absolute values.
pub fn sum_abs(v: &[f64]) -> f64 { v.par_iter().map(|x| x.abs()).sum() }

/// Sum of squared differences, truncating to the shorter length.
pub fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Sum of absolute differences, truncating to the shorter length.
pub fn dist_abs(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter().zip(b.par_iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reductions_match_scalar() {
        let n = 100_000;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut x = 0.5;
        let mut y = 2.0;
        for _ in 0..n {
            a.push(x);
            b.push(y);
            x = (x + 1.0) % 11.0;
            y = (y * 1.00001) % 13.0;
        }

        let scalar: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - scalar).abs() < 1e-6);

        let scalar: f64 = a.iter().sum();
        assert!((sum(&a) - scalar).abs() < 1e-6);

        let scalar: f64 = a.iter().map(|x| x * x).sum();
        assert!((sum_sq(&a) - scalar).abs() < 1e-6);

        let scalar: f64 = a.iter().map(|x| x.abs()).sum();
        assert!((sum_abs(&a) - scalar).abs() < 1e-6);

        let scalar: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!((dist_sq(&a, &b) - scalar).abs() < 1e-6);

        let scalar: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!((dist_abs(&a, &b) - scalar).abs() < 1e-6);
    }
}
