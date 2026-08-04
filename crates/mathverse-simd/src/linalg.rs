//! SIMD-friendly linear algebra kernels.

/// AXPY: `out[i] = alpha * x[i] + y[i]`.
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &[f64], out: &mut [f64]) {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), out.len());

    let chunks = x.len() / 4;
    let remainder = x.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = alpha * x[base] + y[base];
        out[base + 1] = alpha * x[base + 1] + y[base + 1];
        out[base + 2] = alpha * x[base + 2] + y[base + 2];
        out[base + 3] = alpha * x[base + 3] + y[base + 3];
    }

    for i in (chunks * 4)..(chunks * 4 + remainder) {
        out[i] = alpha * x[i] + y[i];
    }
}

/// Matrix-vector product: `out[i] = sum_j(A[i*lda + j] * x[j])`.
///
/// `a` is row-major, `lda` is the leading dimension (number of columns).
#[inline]
pub fn gemv(a: &[f64], lda: usize, x: &[f64], out: &mut [f64]) {
    let m = out.len();
    let n = x.len();
    assert!(lda >= n);
    assert!(a.len() >= m * lda);

    for i in 0..m {
        let row = &a[i * lda..i * lda + n];
        out[i] = super::dot(row, x);
    }
}

/// Blocked dot product for large vectors.
///
/// Splits the computation into blocks of `BLOCK_SIZE` to improve
/// instruction-level parallelism and cache utilization.
#[inline]
pub fn dot_blocked(a: &[f64], b: &[f64]) -> f64 {
    const BLOCK_SIZE: usize = 256;
    assert_eq!(a.len(), b.len());

    let chunks = a.len() / BLOCK_SIZE;
    let mut partials = [0.0f64; BLOCK_SIZE];

    for block in 0..chunks {
        let start = block * BLOCK_SIZE;
        for i in 0..BLOCK_SIZE {
            partials[i] += a[start + i] * b[start + i];
        }
    }

    let mut total: f64 = partials.iter().sum();
    let remainder_start = chunks * BLOCK_SIZE;
    for i in remainder_start..a.len() {
        total += a[i] * b[i];
    }
    total
}

/// Scale a vector in-place: `x[i] *= s`.
#[inline]
pub fn scale_inplace(x: &mut [f64], s: f64) {
    let chunks = x.len() / 4;
    let remainder = x.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        x[base] *= s;
        x[base + 1] *= s;
        x[base + 2] *= s;
        x[base + 3] *= s;
    }

    for i in (chunks * 4)..(chunks * 4 + remainder) {
        x[i] *= s;
    }
}

/// Element-wise add in-place: `a[i] += b[i]`.
#[inline]
pub fn add_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        a[base] += b[base];
        a[base + 1] += b[base + 1];
        a[base + 2] += b[base + 2];
        a[base + 3] += b[base + 3];
    }

    for i in (chunks * 4)..(chunks * 4 + remainder) {
        a[i] += b[i];
    }
}

/// L2 norm: `sqrt(sum(a[i]^2))`.
#[inline]
pub fn l2_norm(a: &[f64]) -> f64 {
    let mut sum_sq = 0.0;
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        let v0 = a[base];
        let v1 = a[base + 1];
        let v2 = a[base + 2];
        let v3 = a[base + 3];
        sum_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }

    for i in (chunks * 4)..(chunks * 4 + remainder) {
        let v = a[i];
        sum_sq += v * v;
    }

    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axpy() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let mut out = [0.0; 3];
        axpy(2.0, &x, &y, &mut out);
        assert_eq!(out, [6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_gemv() {
        // 2x3 matrix times 3-vector
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0, 1.0];
        let mut out = [0.0; 2];
        gemv(&a, 3, &x, &mut out);
        assert_eq!(out, [6.0, 15.0]);
    }

    #[test]
    fn test_dot_blocked() {
        let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let result = dot_blocked(&a, &b);
        let expected: f64 = (0..1000).map(|i| (i * i) as f64).sum();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let a = [3.0, 4.0];
        assert!((l2_norm(&a) - 5.0).abs() < 1e-12);
    }
}
