//! Radix-2 Cooley-Tukey FFT using complex numbers.

use mathverse_complex::Complex;

/// Forward radix-2 Cooley-Tukey FFT, un-normalized.
///
/// Computes `X[k] = Σₙ x[n]·exp(-2πi·k·n/N)`. The input length must be a
/// nonzero power of two. Use [`ifft`] (which applies the `1/N` scale) to
/// invert, so `ifft(fft(x)) == x` exactly up to floating-point error.
pub fn fft(x: &[Complex]) -> mathverse_core::error::MathResult<Vec<Complex>> {
    let n = x.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(mathverse_core::error::MathError::InvalidArgument("fft: length must be nonzero power of two"));
    }
    let mut a = x.to_vec();
    for i in 0..n { let j = i.reverse_bits() >> (usize::BITS - n.trailing_zeros()); if j > i { a.swap(i, j); } }
    let mut len = 2;
    while len <= n {
        let wlen = Complex::polar(1.0, -2.0 * core::f64::consts::PI / len as f64);
        let mut i = 0;
        while i < n {
            let mut w = Complex::real(1.0);
            for k in 0..len / 2 {
                let u = a[i + k];
                let v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w = w * wlen;
            }
            i += len;
        }
        len *= 2;
    }
    Ok(a)
}

/// Inverse FFT, normalized by `1/N`.
///
/// Inverts [`fft`] via the conjugate trick. The input length must be a
/// nonzero power of two.
pub fn ifft(x: &[Complex]) -> mathverse_core::error::MathResult<Vec<Complex>> {
    let n = x.len();
    let conj: Vec<Complex> = x.iter().map(Complex::conjugate).collect();
    let y = fft(&conj)?;
    Ok(y.iter().map(|c| c.conjugate() / Complex::real(n as f64)).collect())
}

/// Direct O(N²) discrete Fourier transform, un-normalized.
///
/// Same convention as [`fft`]: `X[k] = Σₙ x[n]·exp(-2πi·k·n/N)`, but computed
/// directly rather than via the radix-2 algorithm, so it accepts any length
/// (not just powers of two). Use [`idft`] to invert.
pub fn dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    (0..n).map(|k| {
        (0..n).map(|i| {
            let angle = -2.0 * core::f64::consts::PI * k as f64 * i as f64 / n as f64;
            x[i] * Complex::polar(1.0, angle)
        }).fold(Complex::zero(), |a, b| a + b)
    }).collect()
}

/// Direct inverse DFT, normalized by `1/N`.
///
/// Inverts [`dft`]; accepts any input length.
pub fn idft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    (0..n).map(|k| {
        (0..n).map(|i| {
            let angle = 2.0 * core::f64::consts::PI * k as f64 * i as f64 / n as f64;
            x[i] * Complex::polar(1.0, angle)
        }).fold(Complex::zero(), |a, b| a + b) / Complex::real(n as f64)
    }).collect()
}

/// Forward FFT of a real-valued signal, returned as complex coefficients.
///
/// Treats the input as a complex signal with zero imaginary parts. Returns an
/// empty vector if the length is not a nonzero power of two.
pub fn fft_real(x: &[f64]) -> Vec<Complex> {
    let xc: Vec<Complex> = x.iter().map(|&v| Complex::real(v)).collect();
    fft(&xc).unwrap_or_default()
}

/// Power spectrum of a real signal: squared magnitudes `|X[k]|²` of its FFT.
pub fn power_spectrum(x: &[f64]) -> Vec<f64> {
    fft_real(x).iter().map(Complex::norm_sq).collect()
}

/// Cross-correlation of two real signals via FFT-based circular convolution.
///
/// Both inputs are zero-padded to a common power-of-two length (the next
/// power of two at least the longer input). The result has that padded length;
/// for linear (non-circular) correlation the caller should trim to
/// `a.len() + b.len() - 1`.
pub fn cross_correlation(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len().max(b.len()).next_power_of_two();
    let mut fa = vec![Complex::zero(); n];
    let mut fb = vec![Complex::zero(); n];
    for i in 0..a.len() { fa[i] = Complex::real(a[i]); }
    for i in 0..b.len() { fb[i] = Complex::real(b[i]); }
    let pa = fft(&fa).unwrap();
    let pb = fft(&fb).unwrap();
    let product: Vec<Complex> = pa.iter().zip(pb.iter()).map(|(a, b)| *a * b.conjugate()).collect();
    let result = ifft(&product).unwrap();
    result.iter().map(|c| c.re).collect()
}

/// Linear convolution of two real signals via FFT.
///
/// Zero-pads both inputs to the next power of two ≥ `a.len() + b.len() - 1`,
/// multiplies their spectra, and truncates the inverse FFT back to the full
/// linear-convolution length `a.len() + b.len() - 1`.
pub fn convolution(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = (a.len() + b.len() - 1).next_power_of_two();
    let mut fa = vec![Complex::zero(); n];
    let mut fb = vec![Complex::zero(); n];
    for i in 0..a.len() { fa[i] = Complex::real(a[i]); }
    for i in 0..b.len() { fb[i] = Complex::real(b[i]); }
    let pa = fft(&fa).unwrap();
    let pb = fft(&fb).unwrap();
    let product: Vec<Complex> = pa.iter().zip(&pb).map(|(a, b)| a * b).collect();
    let result = ifft(&product).unwrap();
    result.iter().map(|c| c.re).take(a.len() + b.len() - 1).collect()
}

/// Two-dimensional forward FFT, applied separably (rows, then columns).
///
/// Both dimensions must be nonzero powers of two. Returns the `rows × cols`
/// spectrum, where `rows = x.len()` and `cols = x[0].len()`.
///
/// ```
/// use mathverse_complex::Complex;
/// use mathverse_transforms::{fft2, ifft2};
/// let x = vec![
///     vec![Complex::real(1.0), Complex::real(2.0), Complex::real(3.0), Complex::real(4.0)],
///     vec![Complex::real(5.0), Complex::real(6.0), Complex::real(7.0), Complex::real(8.0)],
///     vec![Complex::real(9.0), Complex::real(10.0), Complex::real(11.0), Complex::real(12.0)],
///     vec![Complex::real(13.0), Complex::real(14.0), Complex::real(15.0), Complex::real(16.0)],
/// ];
/// let y = fft2(&x).unwrap();
/// let back = ifft2(&y).unwrap();
/// for r in 0..4 {
///     for c in 0..4 {
///         assert!((x[r][c] - back[r][c]).norm() < 1e-10);
///     }
/// }
/// ```
pub fn fft2(x: &[Vec<Complex>]) -> mathverse_core::error::MathResult<Vec<Vec<Complex>>> {
    let rows = x.len();
    let cols = x.first().map(|r| r.len()).unwrap_or(0);
    if rows == 0 || cols == 0 || !rows.is_power_of_two() || !cols.is_power_of_two() {
        return Err(mathverse_core::error::MathError::InvalidArgument(
            "fft2: both dimensions must be nonzero powers of two",
        ));
    }
    let rows_fft: Vec<Vec<Complex>> = x
        .iter()
        .map(|row| fft(row))
        .collect::<mathverse_core::error::MathResult<_>>()?;
    let mut out = vec![vec![Complex::zero(); cols]; rows];
    for c in 0..cols {
        let col: Vec<Complex> = (0..rows).map(|r| rows_fft[r][c]).collect();
        let transformed = fft(&col)?;
        for r in 0..rows {
            out[r][c] = transformed[r];
        }
    }
    Ok(out)
}

/// Two-dimensional inverse FFT (normalized by `1/(rows·cols)`).
///
/// Inverts [`fft2`]. Both dimensions must be nonzero powers of two.
pub fn ifft2(x: &[Vec<Complex>]) -> mathverse_core::error::MathResult<Vec<Vec<Complex>>> {
    let rows = x.len();
    let cols = x.first().map(|r| r.len()).unwrap_or(0);
    if rows == 0 || cols == 0 || !rows.is_power_of_two() || !cols.is_power_of_two() {
        return Err(mathverse_core::error::MathError::InvalidArgument(
            "ifft2: both dimensions must be nonzero powers of two",
        ));
    }
    let rows_ifft: Vec<Vec<Complex>> = x
        .iter()
        .map(|row| ifft(row))
        .collect::<mathverse_core::error::MathResult<_>>()?;
    let mut out = vec![vec![Complex::zero(); cols]; rows];
    for c in 0..cols {
        let col: Vec<Complex> = (0..rows).map(|r| rows_ifft[r][c]).collect();
        let transformed = ifft(&col)?;
        for r in 0..rows {
            out[r][c] = transformed[r];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip() {
        let x: Vec<Complex> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let y = fft(&x).unwrap();
        let back = ifft(&y).unwrap();
        for (a, b) in x.iter().zip(&back) { assert!((*a - *b).norm() < 1e-12); }
    }

    #[test]
    fn fft2_roundtrip() {
        let x: Vec<Vec<Complex>> = (0..4)
            .map(|r| (0..4).map(|c| Complex::new(r as f64 + 10.0 * c as f64, (r + c) as f64)).collect())
            .collect();
        let y = fft2(&x).unwrap();
        let back = ifft2(&y).unwrap();
        for r in 0..4 {
            for c in 0..4 {
                assert!((x[r][c] - back[r][c]).norm() < 1e-10, "mismatch at ({r}, {c})");
            }
        }
    }

    #[test]
    fn fft2_rejects_bad_dims() {
        let ragged = vec![vec![Complex::zero(); 3]; 4];
        assert!(fft2(&ragged).is_err());
        assert!(fft2(&[]).is_err());
    }

    #[test]
    fn fft2_matches_separable_product() {
        // FFT of a separable signal x[r][c] = a[r] * b[c] equals
        // FFT(a) ⊗ FFT(b) (outer product of the 1-D spectra).
        let a: Vec<Complex> = (0..4).map(|i| Complex::real(i as f64 + 1.0)).collect();
        let b: Vec<Complex> = (0..4).map(|i| Complex::real(2.0 * i as f64 + 1.0)).collect();
        let fa = fft(&a).unwrap();
        let fb = fft(&b).unwrap();
        let x: Vec<Vec<Complex>> = (0..4)
            .map(|r| (0..4).map(|c| a[r] * b[c]).collect())
            .collect();
        let y = fft2(&x).unwrap();
        for r in 0..4 {
            for c in 0..4 {
                assert!((y[r][c] - fa[r] * fb[c]).norm() < 1e-10, "mismatch at ({r}, {c})");
            }
        }
    }
}
