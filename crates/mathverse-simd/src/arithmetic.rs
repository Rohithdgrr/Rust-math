//! SIMD-friendly arithmetic kernels.

/// Element-wise addition: `out[i] = a[i] + b[i]`.
///
/// # Panics
/// Panics if slices have different lengths.
#[inline]
pub fn add(a: &[f64], b: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), b.len(), "slice length mismatch");
    assert_eq!(a.len(), out.len(), "output length mismatch");

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = a[base] + b[base];
        out[base + 1] = a[base + 1] + b[base + 1];
        out[base + 2] = a[base + 2] + b[base + 2];
        out[base + 3] = a[base + 3] + b[base + 3];
    }

    for i in (chunks * 4)..(chunks * 4 + remainder) {
        out[i] = a[i] + b[i];
    }
}

/// Element-wise subtraction: `out[i] = a[i] - b[i]`.
#[inline]
pub fn sub(a: &[f64], b: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = a[base] - b[base];
        out[base + 1] = a[base + 1] - b[base + 1];
        out[base + 2] = a[base + 2] - b[base + 2];
        out[base + 3] = a[base + 3] - b[base + 3];
    }

    for i in (chunks * 4)..(chunks * 4 + remainder) {
        out[i] = a[i] - b[i];
    }
}

/// Element-wise multiplication: `out[i] = a[i] * b[i]`.
#[inline]
pub fn mul(a: &[f64], b: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = a[base] * b[base];
        out[base + 1] = a[base + 1] * b[base + 1];
        out[base + 2] = a[base + 2] * b[base + 2];
        out[base + 3] = a[base + 3] * b[base + 3];
    }

    for i in (chunks * 4)..(chunks * 4 + remainder) {
        out[i] = a[i] * b[i];
    }
}

/// Scalar multiply: `out[i] = a[i] * s`.
#[inline]
pub fn scale(a: &[f64], s: f64, out: &mut [f64]) {
    assert_eq!(a.len(), out.len());

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let base = i * 4;
        out[base] = a[base] * s;
        out[base + 1] = a[base + 1] * s;
        out[base + 2] = a[base + 2] * s;
        out[base + 3] = a[base + 3] * s;
    }

    for i in (chunks * 4)..(chunks * 4 + remainder) {
        out[i] = a[i] * s;
    }
}

/// Negate: `out[i] = -a[i]`.
#[inline]
pub fn negate(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = -v;
    }
}

/// Sum of all elements.
#[inline]
pub fn sum(a: &[f64]) -> f64 {
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;

    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base];
        s1 += a[base + 1];
        s2 += a[base + 2];
        s3 += a[base + 3];
    }

    let mut total = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        total += a[i];
    }
    total
}

/// Dot product: `sum(a[i] * b[i])`.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;

    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }

    let mut total = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        total += a[i] * b[i];
    }
    total
}

/// Maximum element.
#[inline]
pub fn max(a: &[f64]) -> f64 {
    assert!(!a.is_empty());
    let mut m = a[0];
    for &v in &a[1..] {
        if v > m {
            m = v;
        }
    }
    m
}

/// Minimum element.
#[inline]
pub fn min(a: &[f64]) -> f64 {
    assert!(!a.is_empty());
    let mut m = a[0];
    for &v in &a[1..] {
        if v < m {
            m = v;
        }
    }
    m
}

/// Fused multiply-add: `out[i] = a[i] * b[i] + c[i]`.
#[inline]
pub fn mul_add(a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), c.len());
    assert_eq!(a.len(), out.len());

    for (o, ((&a, &b), &c)) in out.iter_mut().zip(a.iter().zip(b).zip(c)) {
        *o = a * b + c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        let mut out = [0.0; 5];
        add(&a, &b, &mut out);
        assert_eq!(out, [6.0, 6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_dot() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dot(&a, &b), 32.0);
    }

    #[test]
    fn test_sum() {
        assert_eq!(sum(&[1.0, 2.0, 3.0, 4.0, 5.0]), 15.0);
    }

    #[test]
    fn test_scale() {
        let a = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        scale(&a, 2.0, &mut out);
        assert_eq!(out, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_mul_add() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let c = [5.0, 6.0];
        let mut out = [0.0; 2];
        mul_add(&a, &b, &c, &mut out);
        assert_eq!(out, [8.0, 14.0]);
    }
}
