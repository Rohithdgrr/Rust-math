//! Parallel operations on raw slices.

use rayon::prelude::*;

/// Parallel sum of a slice.
pub fn par_sum_slice(a: &[f64]) -> f64 {
    a.par_iter().sum()
}

/// Parallel dot product of two slices.
pub fn par_dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter()
        .zip(b.par_iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Parallel element-wise addition into output.
pub fn par_add_slice(a: &[f64], b: &[f64], out: &mut [f64]) {
    out.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|(o, (a, b))| *o = a + b);
}

/// Parallel element-wise multiplication into output.
pub fn par_mul_slice(a: &[f64], b: &[f64], out: &mut [f64]) {
    out.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|(o, (a, b))| *o = a * b);
}

/// Parallel map: apply f to each element.
pub fn par_map_slice(a: &[f64], f: impl Fn(f64) -> f64 + Send + Sync, out: &mut [f64]) {
    out.par_iter_mut()
        .zip(a.par_iter())
        .for_each(|(o, &a)| *o = f(a));
}

/// Prefix sum (inclusive scan).
///
/// Currently a sequential scan (rayon has no stable parallel scan), but the
/// interface is identical to the other `par_*` helpers.
pub fn par_prefix_sum(a: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(a.len());
    let mut acc = 0.0;
    for &v in a {
        acc += v;
        result.push(acc);
    }
    result
}

/// Parallel min element.
pub fn par_min(a: &[f64]) -> f64 {
    a.par_iter()
        .copied()
        .reduce(|| f64::INFINITY, f64::min)
}

/// Parallel max element.
pub fn par_max(a: &[f64]) -> f64 {
    a.par_iter()
        .copied()
        .reduce(|| f64::NEG_INFINITY, f64::max)
}

/// Parallel L2 norm.
pub fn par_l2_norm(a: &[f64]) -> f64 {
    a.par_iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_sum_slice() {
        let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        assert!((par_sum_slice(&a) - 499_500.0).abs() < 1e-6);
    }

    #[test]
    fn test_par_dot_slice() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((par_dot_slice(&a, &b) - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_par_min_max() {
        let a = [3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(par_min(&a), 1.0);
        assert_eq!(par_max(&a), 5.0);
    }

    #[test]
    fn test_par_prefix_sum() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(par_prefix_sum(&a), vec![1.0, 3.0, 6.0, 10.0]);
    }
}
