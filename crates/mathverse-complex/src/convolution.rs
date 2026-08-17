//! Complex convolution using FFT acceleration.
//!
//! Provides overlap-save and overlap-add methods for fast complex
//! convolution of long signals, essential for learned filters in
//! audio processing, signal denoising, and complex-valued CNNs.
//!
//! # Methods
//!
//! | Method | Best for |
//! |--------|----------|
//! | [`convolve_fft`] | Direct FFT convolution (signals of any length) |
//! | [`convolve_direct`] | Small kernels (< 64 elements), no FFT overhead |
//! | [`overlap_save`] | Very long signals with shorter kernels |
//! | [`overlap_add`] | Very long signals with shorter kernels (alternative) |

use crate::fft::{fft, ifft};
use crate::Complex;

/// Direct complex convolution via FFT: `y = x * h`.
///
/// Zero-pads both inputs to the next power of two ≥ `len(x) + len(h) - 1`,
/// computes FFT of both, multiplies pointwise, and inverse-FFTs.
///
/// Returns the full convolution result of length `len(x) + len(h) - 1`.
pub fn convolve_fft(x: &[Complex], h: &[Complex]) -> Vec<Complex> {
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let n = x.len() + h.len() - 1;
    let n_fft = n.next_power_of_two();

    let mut x_padded = x.to_vec();
    x_padded.resize(n_fft, Complex::zero());
    let mut h_padded = h.to_vec();
    h_padded.resize(n_fft, Complex::zero());

    let x_freq = fft(&x_padded);
    let h_freq = fft(&h_padded);

    let y_freq: Vec<Complex> = x_freq.iter().zip(h_freq.iter()).map(|(a, b)| *a * *b).collect();
    let y_full = ifft(&y_freq);

    y_full[..n].to_vec()
}

/// Direct convolution for small signals (no FFT overhead).
///
/// Computes `y[n] = Σ_k x[k] · h[n-k]` directly. Preferred when
/// both signals are short (< 64 elements).
pub fn convolve_direct(x: &[Complex], h: &[Complex]) -> Vec<Complex> {
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let n_out = x.len() + h.len() - 1;
    let mut y = vec![Complex::zero(); n_out];
    for (i, &xi) in x.iter().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            y[i + j] = y[i + j] + xi * hj;
        }
    }
    y
}

/// Overlap-save convolution for long signals with shorter kernels.
///
/// Splits `x` into overlapping blocks of size `block_size`, convolves each
/// with `h` using FFT, and keeps only the valid portion of each block's output.
///
/// `block_size` should be ≥ `2 * h.len()` for efficiency (default: next power
/// of two ≥ `4 * h.len()`).
///
/// # Panics
/// If an explicit `block_size` is smaller than `2 * h.len()`.
///
/// Returns the full convolution result.
pub fn overlap_save(x: &[Complex], h: &[Complex], block_size: Option<usize>) -> Vec<Complex> {
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let m = h.len();
    let n = block_size.unwrap_or_else(|| {
        let min = 4 * m;
        min.next_power_of_two()
    });
    assert!(n >= 2 * m, "block_size must be >= 2 * kernel_len");

    let overlap = m - 1;
    let step = n - overlap;

    // Pad kernel to block_size
    let mut h_padded = h.to_vec();
    h_padded.resize(n, Complex::zero());
    let h_freq = fft(&h_padded);

    // Pad input with leading zeros for the first block
    let _total_len = x.len() + overlap;
    let mut x_padded = vec![Complex::zero(); overlap];
    x_padded.extend_from_slice(x);
    x_padded.resize(x_padded.len().next_power_of_two(), Complex::zero());

    let mut y = Vec::with_capacity(x.len() + m - 1);

    let mut pos = 0;
    while pos + n <= x_padded.len() {
        let block = &x_padded[pos..pos + n];
        let x_freq = fft(block);
        let y_freq: Vec<Complex> = x_freq.iter().zip(h_freq.iter()).map(|(a, b)| *a * *b).collect();
        let y_block = ifft(&y_freq);
        // Keep only the last `step` elements (valid convolution)
        let start = m - 1;
        let end = (start + step).min(n);
        y.extend_from_slice(&y_block[start..end]);
        pos += step;
    }

    y.truncate(x.len() + m - 1);
    y
}

/// Overlap-add convolution for long signals with shorter kernels.
///
/// Splits `x` into non-overlapping blocks, convolves each with `h`,
/// and adds the overlapping portions together.
///
/// Returns the full convolution result.
pub fn overlap_add(x: &[Complex], h: &[Complex], block_size: Option<usize>) -> Vec<Complex> {
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let m = h.len();
    let block_len = block_size.unwrap_or_else(|| {
        let min = 4 * m;
        min.next_power_of_two()
    });
    // Pad to at least block_len + m - 1 to avoid circular aliasing
    let n = (block_len + m - 1).next_power_of_two();

    // Pad kernel to n
    let mut h_padded = h.to_vec();
    h_padded.resize(n, Complex::zero());
    let h_freq = fft(&h_padded);

    let n_out = x.len() + m - 1;
    let mut y = vec![Complex::zero(); n_out];

    let mut pos = 0;
    let mut out_pos = 0;
    while pos < x.len() {
        let cur_block_len = (x.len() - pos).min(block_len);
        let mut block = vec![Complex::zero(); n];
        block[..cur_block_len].copy_from_slice(&x[pos..pos + cur_block_len]);

        let x_freq = fft(&block);
        let y_freq: Vec<Complex> = x_freq.iter().zip(h_freq.iter()).map(|(a, b)| *a * *b).collect();
        let y_block = ifft(&y_freq);

        // Add to output (overlap-add): only the first cur_block_len + m - 1 elements are valid
        let write_len = (n_out - out_pos).min(cur_block_len + m - 1);
        for i in 0..write_len {
            y[out_pos + i] = y[out_pos + i] + y_block[i];
        }

        pos += cur_block_len;
        out_pos += cur_block_len;
    }

    y.truncate(n_out);
    y
}

/// 2D complex convolution via FFT.
///
/// Convolves a 2D signal `x` (of shape `h × w`) with a 2D kernel `k`
/// (of shape `kh × kw`). Returns the full result of shape `(h+kh-1) × (w+kw-1)`.
pub fn convolve_2d_fft(x: &[Vec<Complex>], k: &[Vec<Complex>]) -> Vec<Vec<Complex>> {
    let h = x.len();
    let w = if h > 0 { x[0].len() } else { 0 };
    let kh = k.len();
    let kw = if kh > 0 { k[0].len() } else { 0 };

    if h == 0 || w == 0 || kh == 0 || kw == 0 {
        return Vec::new();
    }

    let out_h = h + kh - 1;
    let out_w = w + kw - 1;
    let n_fft_h = out_h.next_power_of_two();
    let n_fft_w = out_w.next_power_of_two();

    // Pad x and k to (n_fft_h, n_fft_w)
    let mut x_pad = vec![vec![Complex::zero(); n_fft_w]; n_fft_h];
    for i in 0..h {
        for j in 0..w {
            x_pad[i][j] = x[i][j];
        }
    }
    let mut k_pad = vec![vec![Complex::zero(); n_fft_w]; n_fft_h];
    for i in 0..kh {
        for j in 0..kw {
            k_pad[i][j] = k[i][j];
        }
    }

    // Row-wise FFT
    let mut x_freq = vec![vec![Complex::zero(); n_fft_w]; n_fft_h];
    let mut k_freq = vec![vec![Complex::zero(); n_fft_w]; n_fft_h];
    for i in 0..n_fft_h {
        x_freq[i] = fft(&x_pad[i]);
        k_freq[i] = fft(&k_pad[i]);
    }

    // Column-wise FFT
    for j in 0..n_fft_w {
        let col_x: Vec<Complex> = (0..n_fft_h).map(|i| x_freq[i][j]).collect();
        let col_k: Vec<Complex> = (0..n_fft_h).map(|i| k_freq[i][j]).collect();
        let x_col_dft = fft(&col_x);
        let k_col_dft = fft(&col_k);
        for i in 0..n_fft_h {
            x_freq[i][j] = x_col_dft[i] * k_col_dft[i];
        }
    }

    // Column-wise IFFT
    let mut y_freq = vec![vec![Complex::zero(); n_fft_w]; n_fft_h];
    for j in 0..n_fft_w {
        let col: Vec<Complex> = (0..n_fft_h).map(|i| x_freq[i][j]).collect();
        let col_ifft = ifft(&col);
        for i in 0..n_fft_h {
            y_freq[i][j] = col_ifft[i];
        }
    }

    // Row-wise IFFT
    let mut result = vec![vec![Complex::zero(); out_w]; out_h];
    for i in 0..n_fft_h {
        let row_ifft = ifft(&y_freq[i]);
        for j in 0..out_w {
            if i < out_h {
                result[i][j] = row_ifft[j];
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn convolve_fft_identity() {
        // Convolution with delta function [1, 0, 0, ...] should return the input
        let x: Vec<Complex> = (0..8).map(|i| Complex::real(f64::from(i))).collect();
        let h = vec![Complex::one(), Complex::zero(), Complex::zero()];
        let y = convolve_fft(&x, &h);
        // Convolution of length 8 and length 3 yields length 10
        assert_eq!(y.len(), 10);
        // First 8 elements should match the input
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).norm() < EPS);
        }
        // Last 2 elements should be zero
        assert!(y[8].norm() < EPS);
        assert!(y[9].norm() < EPS);
    }

    #[test]
    fn convolve_fft_matches_direct() {
        let x: Vec<Complex> = (0..5).map(|i| Complex::new(f64::from(i), f64::from(i) * 0.5)).collect();
        let h: Vec<Complex> = vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.5)];
        let y_fft = convolve_fft(&x, &h);
        let y_direct = convolve_direct(&x, &h);
        assert_eq!(y_fft.len(), y_direct.len());
        for (a, b) in y_fft.iter().zip(y_direct.iter()) {
            assert!((a - b).norm() < EPS);
        }
    }

    #[test]
    fn convolve_direct_known_result() {
        // [1, 2] * [3, 4] = [3, 10, 8]
        let x = vec![Complex::real(1.0), Complex::real(2.0)];
        let h = vec![Complex::real(3.0), Complex::real(4.0)];
        let y = convolve_direct(&x, &h);
        assert_eq!(y.len(), 3);
        assert!((y[0].re - 3.0).abs() < EPS);
        assert!((y[1].re - 10.0).abs() < EPS);
        assert!((y[2].re - 8.0).abs() < EPS);
    }

    #[test]
    fn overlap_save_matches_direct() {
        let x: Vec<Complex> = (0..100).map(|i| Complex::real(f64::from(i))).collect();
        let h: Vec<Complex> = vec![Complex::real(1.0), Complex::real(0.5), Complex::real(-0.25)];
        let y_direct = convolve_direct(&x, &h);
        let y_os = overlap_save(&x, &h, None);
        assert_eq!(y_os.len(), y_direct.len());
        for (i, (a, b)) in y_direct.iter().zip(y_os.iter()).enumerate() {
            assert!(
                (a - b).norm() < 1e-8,
                "mismatch at {i}: got {b}, expected {a}"
            );
        }
    }

    #[test]
    fn overlap_add_matches_direct() {
        let x: Vec<Complex> = (0..100).map(|i| Complex::real(f64::from(i))).collect();
        let h: Vec<Complex> = vec![Complex::real(1.0), Complex::real(0.5), Complex::real(-0.25)];
        let y_direct = convolve_direct(&x, &h);
        let y_oa = overlap_add(&x, &h, None);
        assert_eq!(y_oa.len(), y_direct.len());
        for (i, (a, b)) in y_direct.iter().zip(y_oa.iter()).enumerate() {
            assert!(
                (a - b).norm() < 1e-8,
                "mismatch at {i}: got {b}, expected {a}"
            );
        }
    }

    #[test]
    fn convolve_2d_basic() {
        // 2x2 signal with 2x2 delta kernel
        let x = vec![
            vec![Complex::real(1.0), Complex::real(2.0)],
            vec![Complex::real(3.0), Complex::real(4.0)],
        ];
        let k = vec![
            vec![Complex::real(1.0), Complex::zero()],
            vec![Complex::zero(), Complex::zero()],
        ];
        let y = convolve_2d_fft(&x, &k);
        assert_eq!(y.len(), 3);
        assert_eq!(y[0].len(), 3);
        // Top-left should be 1*1 = 1
        assert!((y[0][0].re - 1.0).abs() < EPS);
    }

    #[test]
    fn convolve_empty() {
        assert!(convolve_fft(&[], &[Complex::one()]).is_empty());
        assert!(convolve_direct(&[], &[Complex::one()]).is_empty());
    }
}
