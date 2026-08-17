//! Fast Fourier Transform (radix-2 Cooley–Tukey) and inverse FFT.
//!
//! Inputs must have power-of-two length. For the extended transform suite
//! (real-input FFT, 2D FFT, convolution, …), see `mathverse-transforms`.

use crate::Complex;

/// Radix-2 Cooley–Tukey FFT of `input`.
///
/// The length must be a power of two; otherwise this panics.
///
/// ```
/// use mathverse_complex::{fft::fft, Complex};
/// let x = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
/// let y = fft(&x);
/// assert_eq!(y.len(), 2);
/// assert!((y[0].re - 3.0).abs() < 1e-12);
/// assert!((y[1].re + 1.0).abs() < 1e-12);
/// ```
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let mut buf = input.to_vec();
    fft_in_place(&mut buf);
    buf
}

/// In-place radix-2 Cooley–Tukey FFT.
///
/// Mutates `buf` in place. Length must be a power of two.
///
/// # Panics
/// If `buf.len()` is not a power of two.
pub fn fft_in_place(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(
        n.is_power_of_two(),
        "fft: input length must be a power of two, got {n}"
    );
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            buf.swap(i, j);
        }
    }

    // Butterfly loops: stage s produces sub-DFTs of size 2^s.
    let pi = core::f64::consts::PI;
    let mut half = 1;
    while half < n {
        let stride = half * 2;
        let angle_step = -pi / half as f64;
        for chunk_start in (0..n).step_by(stride) {
            for k in 0..half {
                let angle = angle_step * k as f64;
                let twiddle = Complex::polar(1.0, angle);
                let even = buf[chunk_start + k];
                let odd = twiddle * buf[chunk_start + k + half];
                buf[chunk_start + k] = even + odd;
                buf[chunk_start + k + half] = even - odd;
            }
        }
        half = stride;
    }
}

/// Inverse FFT: `ifft(fft(x)) ≈ x` (up to floating-point roundoff).
///
/// # Panics
/// If `input.len()` is not a power of two.
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    assert!(
        n.is_power_of_two(),
        "ifft: input length must be a power of two, got {n}"
    );
    let mut buf: Vec<Complex> = input.iter().map(Complex::conjugate).collect();
    fft_in_place(&mut buf);
    for c in &mut buf {
        *c = c.conjugate() / Complex::real(n as f64);
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_single_element() {
        let x = vec![Complex::new(3.5, -1.0)];
        assert_eq!(fft(&x), x);
        assert!((ifft(&x)[0] - x[0]).norm() < 1e-12);
    }

    #[test]
    fn fft_roundtrip() {
        let x: Vec<Complex> = (0..8)
            .map(|i| Complex::new(f64::from(i) * 0.5, f64::from(i).sin()))
            .collect();
        let y = ifft(&fft(&x));
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn fft_matches_dft() {
        // Brute-force DFT comparison on 4 points
        let x: Vec<Complex> = (0..4)
            .map(|i| Complex::new(f64::from(i), 2.0 * f64::from(i)))
            .collect();
        let n = x.len();
        let dft: Vec<Complex> = (0..n)
            .map(|k| {
                (0..n)
                    .map(|j| {
                        x[j] * Complex::polar(
                            1.0,
                            -2.0 * core::f64::consts::PI * (k * j) as f64 / n as f64,
                        )
                    })
                    .fold(Complex::zero(), |acc, v| acc + v)
            })
            .collect();
        let y = fft(&x);
        for (a, b) in y.iter().zip(dft.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn fft_larger_sizes() {
        // 16-point FFT round-trip
        let x: Vec<Complex> = (0..16)
            .map(|i| Complex::new(f64::from(i).cos(), f64::from(i).sin() * 0.1))
            .collect();
        let y = ifft(&fft(&x));
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn fft_in_place_single() {
        let mut buf = vec![Complex::new(5.0, -2.0)];
        fft_in_place(&mut buf);
        assert_eq!(buf, vec![Complex::new(5.0, -2.0)]);
    }

    #[test]
    fn fft_in_place_matches_fft() {
        let x: Vec<Complex> = (0..32)
            .map(|i| Complex::new(f64::from(i) * 0.3, (f64::from(i) * 0.7).sin()))
            .collect();
        let mut buf = x.clone();
        fft_in_place(&mut buf);
        let y = fft(&x);
        for (a, b) in buf.iter().zip(y.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }
}
