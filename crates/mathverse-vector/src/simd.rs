//! SIMD-accelerated reduction kernels behind the `simd` feature.
//!
//! Enabling the `simd` feature builds these functions on 128-bit vector lanes
//! via the `wide` crate, which compile to SSE2 on x86-64 and NEON on AArch64
//! (both ISA baselines, so no runtime feature detection is required). On other
//! targets the same API degrades to an equivalent scalar implementation.
//!
//! The crate's O(n) reductions ([`crate::operations`], [`crate::norms`],
//! [`crate::statistics`], [`crate::distance`], [`crate::geometry`]) route
//! through these kernels when the feature is enabled. They are also public so
//! they can be called directly.

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod wide_impl {
    use wide::f64x2;

    /// Dot product of two slices, truncating to the shorter length.
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        let mut acc = f64x2::splat(0.0);
        let mut i = 0;
        while i + 1 < n {
            acc += f64x2::from([a[i], a[i + 1]]) * f64x2::from([b[i], b[i + 1]]);
            i += 2;
        }
        let mut s = acc.0[0] + acc.0[1];
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    /// Sum of all elements.
    pub fn sum(v: &[f64]) -> f64 {
        let mut acc = f64x2::splat(0.0);
        let mut i = 0;
        while i + 1 < v.len() {
            acc += f64x2::from([v[i], v[i + 1]]);
            i += 2;
        }
        let mut s = acc.0[0] + acc.0[1];
        while i < v.len() {
            s += v[i];
            i += 1;
        }
        s
    }

    /// Sum of squares.
    pub fn sum_sq(v: &[f64]) -> f64 {
        let mut acc = f64x2::splat(0.0);
        let mut i = 0;
        while i + 1 < v.len() {
            let lane = f64x2::from([v[i], v[i + 1]]);
            acc += lane * lane;
            i += 2;
        }
        let mut s = acc.0[0] + acc.0[1];
        while i < v.len() {
            s += v[i] * v[i];
            i += 1;
        }
        s
    }

    /// Sum of absolute values.
    pub fn sum_abs(v: &[f64]) -> f64 {
        let mut acc = f64x2::splat(0.0);
        let mut i = 0;
        while i + 1 < v.len() {
            acc += f64x2::from([v[i].abs(), v[i + 1].abs()]);
            i += 2;
        }
        let mut s = acc.0[0] + acc.0[1];
        while i < v.len() {
            s += v[i].abs();
            i += 1;
        }
        s
    }

    /// Sum of squared differences, truncating to the shorter length.
    pub fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        let mut acc = f64x2::splat(0.0);
        let mut i = 0;
        while i + 1 < n {
            let d = f64x2::from([a[i], a[i + 1]]) - f64x2::from([b[i], b[i + 1]]);
            acc += d * d;
            i += 2;
        }
        let mut s = acc.0[0] + acc.0[1];
        while i < n {
            let d = a[i] - b[i];
            s += d * d;
            i += 1;
        }
        s
    }

    /// Sum of absolute differences, truncating to the shorter length.
    pub fn dist_abs(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        let mut acc = f64x2::splat(0.0);
        let mut i = 0;
        while i + 1 < n {
            acc += f64x2::from([(a[i] - b[i]).abs(), (a[i + 1] - b[i + 1]).abs()]);
            i += 2;
        }
        let mut s = acc.0[0] + acc.0[1];
        while i < n {
            s += (a[i] - b[i]).abs();
            i += 1;
        }
        s
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod wide_impl {
    /// Dot product of two slices, truncating to the shorter length.
    pub fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }

    /// Sum of all elements.
    pub fn sum(v: &[f64]) -> f64 { v.iter().sum() }

    /// Sum of squares.
    pub fn sum_sq(v: &[f64]) -> f64 { v.iter().map(|x| x * x).sum() }

    /// Sum of absolute values.
    pub fn sum_abs(v: &[f64]) -> f64 { v.iter().map(|x| x.abs()).sum() }

    /// Sum of squared differences, truncating to the shorter length.
    pub fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Sum of absolute differences, truncating to the shorter length.
    pub fn dist_abs(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
    }
}

pub use wide_impl::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn slices() -> (Vec<f64>, Vec<f64>) {
        let n = 999;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut x = 0.5;
        let mut y = 2.0;
        for _ in 0..n {
            a.push(x);
            b.push(y);
            x = (x + 0.7) % 9.1;
            y = (y * 1.3 + 0.1) % 100.0;
        }
        (a, b)
    }

    #[test]
    fn kernels_match_scalar() {
        let (a, b) = slices();

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

    #[test]
    fn truncates_to_shorter_length() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0];
        assert!((dot(&a, &b) - 3.0).abs() < 1e-12);
        assert!((dist_sq(&a, &b) - 1.0).abs() < 1e-12);
        assert!((dist_abs(&a, &b) - 1.0).abs() < 1e-12);
    }
}
