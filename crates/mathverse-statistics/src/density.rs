//! Kernel density estimation (Gaussian kernel).

use crate::descriptive::{iqr, std_dev_sample};
use crate::distributions::normal_pdf;

/// Bandwidth selection for KDE.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bandwidth {
    /// Scott's rule: `n^(-1/5) * s`.
    Scott,
    /// Silverman's rule: `0.9 * n^(-1/5) * min(s, IQR / 1.34)`.
    Silverman,
    /// User-supplied bandwidth.
    Fixed(f64),
}

/// Gaussian kernel density estimate at `x` for samples `xs` with bandwidth
/// `h`. Non-finite samples are skipped; empty input or `h <= 0` yields 0.
#[must_use]
pub fn kernel_density(x: f64, xs: &[f64], h: f64) -> f64 {
    if h <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    let mut n = 0.0;
    let mut sum = 0.0;
    for &xi in xs {
        if !xi.is_finite() {
            continue;
        }
        n += 1.0;
        sum += normal_pdf((x - xi) / h);
    }
    if n == 0.0 {
        return 0.0;
    }
    sum / (n * h)
}

/// Scott's rule bandwidth. Returns 0 for degenerate (zero-variance) data.
#[must_use]
pub fn scott_bandwidth(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let s = std_dev_sample(xs);
    if !s.is_finite() || s <= 0.0 {
        return 0.0;
    }
    n.powf(-0.2) * s
}

/// Silverman's rule bandwidth. Returns 0 for degenerate data.
#[must_use]
pub fn silverman_bandwidth(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let s = std_dev_sample(xs);
    if !s.is_finite() || s <= 0.0 {
        return 0.0;
    }
    0.9 * n.powf(-0.2) * s.min(iqr(xs) / 1.34)
}

/// Resolve a [`Bandwidth`] choice into a bandwidth value.
#[must_use]
pub fn resolve_bandwidth(bw: Bandwidth, xs: &[f64]) -> f64 {
    match bw {
        Bandwidth::Fixed(h) => h,
        Bandwidth::Scott => scott_bandwidth(xs),
        Bandwidth::Silverman => silverman_bandwidth(xs),
    }
}

/// KDE sampled on a uniform grid of `n` points over `[lo, hi]` (inclusive).
#[must_use]
pub fn kernel_density_curve(
    xs: &[f64],
    bandwidth: Bandwidth,
    lo: f64,
    hi: f64,
    n: usize,
) -> Vec<(f64, f64)> {
    let h = resolve_bandwidth(bandwidth, xs);
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(lo, kernel_density(lo, xs, h))];
    }
    (0..n)
        .map(|i| {
            let x = lo + (hi - lo) * i as f64 / (n - 1) as f64;
            (x, kernel_density(x, xs, h))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_peaks_at_data_mean() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let h = scott_bandwidth(&xs);
        assert!(h > 0.0);
        assert!(kernel_density(3.0, &xs, h) > kernel_density(10.0, &xs, h));
    }

    #[test]
    fn bandwidth_rules_positive() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let scott = scott_bandwidth(&xs);
        let silver = silverman_bandwidth(&xs);
        assert!((scott - 1.1459).abs() < 0.01);
        assert!((silver - 0.9735).abs() < 0.01);
        assert!(silver < scott);
    }

    #[test]
    fn curve_spans_grid() {
        let xs = [1.0, 2.0, 3.0];
        let curve = kernel_density_curve(&xs, Bandwidth::Silverman, 0.0, 4.0, 5);
        assert_eq!(curve.len(), 5);
        assert_eq!(curve[0].0, 0.0);
        assert_eq!(curve[4].0, 4.0);
    }

    #[test]
    fn degenerate_and_empty_are_safe() {
        assert_eq!(kernel_density(1.0, &[], 1.0), 0.0);
        assert_eq!(kernel_density(1.0, &[2.0, 2.0, 2.0], 0.0), 0.0);
        let curve = kernel_density_curve(&[2.0, 2.0, 2.0], Bandwidth::Scott, 0.0, 4.0, 3);
        assert!(curve.iter().all(|(_, d)| *d == 0.0));
    }
}
