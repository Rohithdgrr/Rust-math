//! no_std compatible math operations.

use alloc::vec::Vec;
use mathverse_core::traits::Real;

/// no_std compatible vector dot product.
pub fn dot_nostd(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// no_std compatible vector addition.
pub fn add_nostd(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// no_std compatible scalar multiply.
pub fn scale_nostd(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

/// no_std compatible L2 norm.
pub fn norm_nostd(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// no_std compatible sum.
pub fn sum_nostd(a: &[f64]) -> f64 {
    a.iter().sum()
}

/// no_std compatible mean.
pub fn mean_nostd(a: &[f64]) -> f64 {
    sum_nostd(a) / a.len() as f64
}

/// no_std compatible variance (population).
pub fn variance_nostd(a: &[f64]) -> f64 {
    let m = mean_nostd(a);
    a.iter().map(|x| (x - m).powi(2)).sum::<f64>() / a.len() as f64
}

/// no_std compatible clamp.
pub fn clamp_nostd(a: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    a.iter().map(|x| x.clamp(lo, hi)).collect()
}

/// no_std compatible lerp.
pub fn lerp_nostd(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x + t * (y - x))
        .collect()
}

/// no_std compatible element-wise multiply.
pub fn mul_nostd(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// no_std compatible sigmoid.
pub fn sigmoid_nostd(a: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|&x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        })
        .collect()
}

/// no_std compatible relu.
pub fn relu_nostd(a: &[f64]) -> Vec<f64> {
    a.iter().map(|&x| x.max(0.0)).collect()
}

/// no_std compatible tanh.
pub fn tanh_nostd(a: &[f64]) -> Vec<f64> {
    a.iter().map(|&x| x.tanh()).collect()
}

/// no_std compatible softmax.
pub fn softmax_nostd(a: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }
    let max_val = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = a.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// no_std compatible matrix multiply (row-major).
pub fn matmul_nostd(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
    out
}

/// no_std compatible prefix sum (inclusive scan).
pub fn prefix_sum_nostd(a: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(a.len());
    let mut acc = 0.0;
    for &v in a {
        acc += v;
        result.push(acc);
    }
    result
}

/// no_std compatible cumulative product.
pub fn cumprod_nostd(a: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(a.len());
    let mut acc = 1.0;
    for &v in a {
        acc *= v;
        result.push(acc);
    }
    result
}

/// no_std compatible windowed moving average.
pub fn moving_average_nostd(a: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || a.len() < window {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(a.len() - window + 1);
    let mut sum: f64 = a[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..a.len() {
        sum += a[i] - a[i - window];
        result.push(sum / window as f64);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_nostd_test() {
        assert_eq!(dot_nostd(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    }

    #[test]
    fn softmax_nostd_test() {
        let result = softmax_nostd(&[1.0, 2.0, 3.0]);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn matmul_nostd_test() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = matmul_nostd(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn moving_average_test() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average_nostd(&data, 3);
        assert_eq!(ma, vec![2.0, 3.0, 4.0]);
    }
}
