//! SIMD-friendly math function kernels.

/// Element-wise square root: `out[i] = sqrt(a[i])`.
#[inline]
pub fn sqrt(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.sqrt();
    }
}

/// Element-wise absolute value: `out[i] = abs(a[i])`.
#[inline]
pub fn abs(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.abs();
    }
}

/// Element-wise sign: `out[i] = sign(a[i])`.
#[inline]
pub fn sign(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
}

/// Element-wise exp: `out[i] = exp(a[i])`.
#[inline]
pub fn exp(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.exp();
    }
}

/// Element-wise log: `out[i] = ln(a[i])`.
#[inline]
pub fn log(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.ln();
    }
}

/// Element-wise powf: `out[i] = a[i]^e`.
#[inline]
pub fn powf(a: &[f64], e: f64, out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.powf(e);
    }
}

/// Element-wise sin: `out[i] = sin(a[i])`.
#[inline]
pub fn sin(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.sin();
    }
}

/// Element-wise cos: `out[i] = cos(a[i])`.
#[inline]
pub fn cos(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.cos();
    }
}

/// Element-wise tanh: `out[i] = tanh(a[i])`.
#[inline]
pub fn tanh(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.tanh();
    }
}

/// Element-wise exp(-x): `out[i] = exp(-a[i])`.
#[inline]
pub fn exp_neg(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = (-v).exp();
    }
}

/// Element-wise log1p: `out[i] = ln(1 + a[i])`.
#[inline]
pub fn log1p(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.ln_1p();
    }
}

/// Element-wise expm1: `out[i] = exp(a[i]) - 1`.
#[inline]
pub fn expm1(a: &[f64], out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.exp_m1();
    }
}

/// Clamp all elements to [lo, hi].
#[inline]
pub fn clamp(a: &[f64], lo: f64, hi: f64, out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    for (o, &v) in out.iter_mut().zip(a) {
        *o = v.clamp(lo, hi);
    }
}

/// Lerp: `out[i] = a[i] + t * (b[i] - a[i])`.
#[inline]
pub fn lerp(a: &[f64], b: &[f64], t: f64, out: &mut [f64]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    for (o, (&av, &bv)) in out.iter_mut().zip(a.iter().zip(b)) {
        *o = av + t * (bv - av);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt() {
        let a = [4.0, 9.0, 16.0];
        let mut out = [0.0; 3];
        sqrt(&a, &mut out);
        assert_eq!(out, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let a = [1.0, 2.0, 3.0];
        let mut out = [0.0; 3];
        exp(&a, &mut out);
        let mut back = [0.0; 3];
        log(&out, &mut back);
        for i in 0..3 {
            assert!((a[i] - back[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_clamp() {
        let a = [-1.0, 0.5, 2.0];
        let mut out = [0.0; 3];
        clamp(&a, 0.0, 1.0, &mut out);
        assert_eq!(out, [0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_lerp() {
        let a = [0.0, 0.0];
        let b = [10.0, 20.0];
        let mut out = [0.0; 2];
        lerp(&a, &b, 0.5, &mut out);
        assert_eq!(out, [5.0, 10.0]);
    }
}
