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
    let n = input.len();
    assert!(
        n.is_power_of_two(),
        "fft: input length must be a power of two, got {n}"
    );
    if n <= 1 {
        return input.to_vec();
    }
    let even: Vec<Complex> = input.iter().step_by(2).copied().collect();
    let odd: Vec<Complex> = input.iter().skip(1).step_by(2).copied().collect();
    let even_fft = fft(&even);
    let odd_fft = fft(&odd);
    let mut result = vec![Complex::zero(); n];
    for k in 0..n / 2 {
        let twiddle = Complex::polar(1.0, -2.0 * core::f64::consts::PI * k as f64 / n as f64);
        let t = twiddle * odd_fft[k];
        result[k] = even_fft[k] + t;
        result[k + n / 2] = even_fft[k] - t;
    }
    result
}

/// Inverse FFT: `ifft(fft(x)) ≈ x` (up to floating-point roundoff).
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    assert!(
        n.is_power_of_two(),
        "ifft: input length must be a power of two, got {n}"
    );
    let conj: Vec<Complex> = input.iter().map(Complex::conjugate).collect();
    let mut result = fft(&conj);
    for c in &mut result {
        *c = c.conjugate() / Complex::real(n as f64);
    }
    result
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
            .map(|i| Complex::new((i as f64) * 0.5, (i as f64).sin()))
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
            .map(|i| Complex::new(i as f64, 2.0 * i as f64))
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
}
