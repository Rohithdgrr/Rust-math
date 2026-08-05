/// Creates a zero vector of length `n`.
pub fn zeros(n: usize) -> Vec<f64> { vec![0.0; n] }

/// Creates a vector of ones with length `n`.
pub fn ones(n: usize) -> Vec<f64> { vec![1.0; n] }

/// Creates `n` evenly spaced values from `start` to `end` (inclusive).
pub fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![start]; }
    (0..n).map(|i| start + (end - start) * i as f64 / (n - 1) as f64).collect()
}
/// Pseudo-random vector in `[min, max)` using a deterministic hash.
pub fn random(n: usize, min: f64, max: f64) -> Vec<f64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    (0..n).map(|i| {
        let mut h = DefaultHasher::new();
        i.hash(&mut h);
        let r = h.finish() as f64 / u64::MAX as f64;
        min + r * (max - min)
    }).collect()
}
/// Index of the maximum element.
pub fn argmax(v: &[f64]) -> usize { v.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).expect("non-finite value")).expect("empty slice").0 }

/// Index of the minimum element.
pub fn argmin(v: &[f64]) -> usize { v.iter().enumerate().min_by(|a,b| a.1.partial_cmp(b.1).expect("non-finite value")).expect("empty slice").0 }

/// Maximum value in a slice.
pub fn max(v: &[f64]) -> f64 { v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }

/// Minimum value in a slice.
pub fn min(v: &[f64]) -> f64 { v.iter().cloned().fold(f64::INFINITY, f64::min) }

/// Sum of all elements.
pub fn sum(v: &[f64]) -> f64 { crate::operations::sum_fast(v) }

/// Product of all elements.
pub fn prod(v: &[f64]) -> f64 { v.iter().product() }

/// Clamps all elements in-place to `[min_val, max_val]`.
pub fn clip(v: &mut [f64], min_val: f64, max_val: f64) { for x in v.iter_mut() { *x = x.clamp(min_val, max_val); } }

/// Returns a new vector with elements in reverse order.
pub fn reverse(v: &[f64]) -> Vec<f64> { v.iter().rev().copied().collect() }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn linspace_test() {
        let v = linspace(0.0, 1.0, 5);
        assert_eq!(v, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }
    #[test] fn argmax_test() { assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1); }
}
