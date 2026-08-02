//! Haar discrete wavelet transform: forward and inverse, multi-level.

pub fn haar_dwt(x: &[f64]) -> mathverse_core::error::MathResult<Vec<f64>> {
    if x.is_empty() || !x.len().is_multiple_of(2) {
        return Err(mathverse_core::error::MathError::InvalidArgument("haar_dwt: length must be even and nonzero"));
    }
    let n = x.len();
    let mut out = vec![0.0; n];
    let inv = core::f64::consts::FRAC_1_SQRT_2;
    for (i, chunk) in x.chunks(2).enumerate() {
        out[i] = (chunk[0] + chunk[1]) * inv;
        out[i + n / 2] = (chunk[0] - chunk[1]) * inv;
    }
    Ok(out)
}

pub fn haar_idwt(c: &[f64]) -> mathverse_core::error::MathResult<Vec<f64>> {
    if c.is_empty() || !c.len().is_multiple_of(2) {
        return Err(mathverse_core::error::MathError::InvalidArgument("haar_idwt: length must be even and nonzero"));
    }
    let n = c.len();
    let mut out = vec![0.0; n];
    let inv = core::f64::consts::FRAC_1_SQRT_2;
    for i in 0..n / 2 {
        let (a, d) = (c[i], c[i + n / 2]);
        out[2 * i] = (a + d) * inv;
        out[2 * i + 1] = (a - d) * inv;
    }
    Ok(out)
}

pub fn haar_dwt_multi(x: &[f64], levels: usize) -> mathverse_core::error::MathResult<Vec<f64>> {
    if x.is_empty() || !x.len().is_power_of_two() {
        return Err(mathverse_core::error::MathError::InvalidArgument("haar_dwt_multi: length must be nonzero power of two"));
    }
    let max_levels = x.len().trailing_zeros() as usize;
    if levels > max_levels {
        return Err(mathverse_core::error::MathError::InvalidArgument("haar_dwt_multi: too many levels"));
    }
    let mut data = x.to_vec();
    let mut n = x.len();
    for _ in 0..levels {
        let mut out = vec![0.0; n];
        let inv = core::f64::consts::FRAC_1_SQRT_2;
        for (i, chunk) in data.chunks(2).enumerate() {
            out[i] = (chunk[0] + chunk[1]) * inv;
            out[i + n / 2] = (chunk[0] - chunk[1]) * inv;
        }
        data = out;
        n /= 2;
    }
    Ok(data)
}

pub fn haar_idwt_multi(c: &[f64], levels: usize) -> mathverse_core::error::MathResult<Vec<f64>> {
    let mut data = c.to_vec();
    let mut n = c.len();
    for _ in 0..levels {
        let mut out = vec![0.0; n * 2];
        let inv = core::f64::consts::FRAC_1_SQRT_2;
        for i in 0..n {
            let (a, d) = (data[i], data[i + n]);
            out[2 * i] = (a + d) * inv;
            out[2 * i + 1] = (a - d) * inv;
        }
        data = out;
        n *= 2;
    }
    Ok(data)
}

pub fn haar_energy(c: &[f64]) -> f64 { c.iter().map(|v| v * v).sum() }

pub fn haar_threshold(c: &mut [f64], threshold: f64) {
    for v in c.iter_mut() { if v.abs() < threshold { *v = 0.0; } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haar_roundtrip() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let c = haar_dwt(&x).unwrap();
        let back = haar_idwt(&c).unwrap();
        for (a, b) in x.iter().zip(&back) { assert!((a - b).abs() < 1e-12); }
    }
}
