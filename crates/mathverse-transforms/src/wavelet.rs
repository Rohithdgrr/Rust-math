//! Haar discrete wavelet transform: forward and inverse, multi-level.

/// Single-level forward Haar wavelet transform (orthonormal).
///
/// Returns `N` coefficients: the first `N/2` are the smoothed averages
/// `(x[2i] + x[2i+1])/√2`, the last `N/2` are the detail differences
/// `(x[2i] - x[2i+1])/√2`. Requires an even, nonzero length. Inverted by
/// [`haar_idwt`].
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

/// Single-level inverse Haar wavelet transform (orthonormal).
///
/// Inverts [`haar_dwt`]; given averages in the first half and details in the
/// second half, reconstructs the original signal. Requires an even, nonzero
/// length.
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

/// Multi-level forward Haar transform in the standard pyramid layout.
///
/// Requires a nonzero power-of-two length and `levels ≤ log₂(N)`. Each level
/// transforms only the current approximation block, leaving earlier detail
/// blocks untouched, so the returned `N`-length vector is laid out as
/// `[app_L | det_L | det_{L−1} | … | det_1]` — exactly what [`haar_idwt_multi`]
/// expects and the same convention `pywt.wavedec` uses for its packed form.
pub fn haar_dwt_multi(x: &[f64], levels: usize) -> mathverse_core::error::MathResult<Vec<f64>> {
    if x.is_empty() || !x.len().is_power_of_two() {
        return Err(mathverse_core::error::MathError::InvalidArgument("haar_dwt_multi: length must be nonzero power of two"));
    }
    let max_levels = x.len().trailing_zeros() as usize;
    if levels > max_levels {
        return Err(mathverse_core::error::MathError::InvalidArgument("haar_dwt_multi: too many levels"));
    }
    let inv = core::f64::consts::FRAC_1_SQRT_2;
    let mut data = x.to_vec();
    let mut len = x.len();
    for _ in 0..levels {
        let half = len / 2;
        // Scratch buffer: writing details in place would clobber pairs that
        // have not been averaged yet.
        let mut out = vec![0.0; len];
        for i in 0..half {
            let (a, d) = (data[2 * i], data[2 * i + 1]);
            out[i] = (a + d) * inv;
            out[half + i] = (a - d) * inv;
        }
        data[..len].copy_from_slice(&out);
        len = half;
    }
    Ok(data)
}

/// Multi-level inverse Haar transform.
///
/// Inverts [`haar_dwt_multi`]: treats the first `N / 2^levels` entries as the
/// coarsest approximation and the following blocks as per-level details,
/// reconstructing level by level until the full signal is restored.
pub fn haar_idwt_multi(c: &[f64], levels: usize) -> mathverse_core::error::MathResult<Vec<f64>> {
    if c.is_empty() || !c.len().is_power_of_two() {
        return Err(mathverse_core::error::MathError::InvalidArgument("haar_idwt_multi: length must be nonzero power of two"));
    }
    let max_levels = c.len().trailing_zeros() as usize;
    if levels > max_levels {
        return Err(mathverse_core::error::MathError::InvalidArgument("haar_idwt_multi: too many levels"));
    }
    let inv = core::f64::consts::FRAC_1_SQRT_2;
    let mut data = c.to_vec();
    let mut len = c.len() >> levels;
    for _ in 0..levels {
        let half = len;
        // Scratch buffer: reconstruction writes into the same prefix it
        // still reads approximation coefficients from.
        let mut out = vec![0.0; half * 2];
        for i in 0..half {
            let (a, d) = (data[i], data[half + i]);
            out[2 * i] = (a + d) * inv;
            out[2 * i + 1] = (a - d) * inv;
        }
        data[..half * 2].copy_from_slice(&out);
        len *= 2;
    }
    Ok(data)
}

/// Energy of a coefficient vector: sum of squares. For orthonormal Haar
/// transforms this equals the input signal's energy (Parseval's theorem).
pub fn haar_energy(c: &[f64]) -> f64 { c.iter().map(|v| v * v).sum() }

/// Hard-threshold wavelet coefficients in place: any coefficient with
/// `|c| < threshold` is set to zero.
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
