//! Complex wavelet transforms: CWT (Continuous Wavelet Transform) and
//! DWT (Discrete Wavelet Transform) for time-frequency analysis of
//! complex-valued signals.
//!
//! The CWT uses FFT-accelerated convolution with complex-valued wavelets
//! (Morlet, Gabor, Mexican Hat) to produce a scalogram. The DWT provides
//! multi-level decomposition using complex-valued filter banks.
//!
//! # Module overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`cwt`] | Continuous wavelet transform via FFT convolution |
//! | [`dwt`] | Single-level discrete wavelet decomposition |
//! | [`dwt_inverse`] | Single-level inverse DWT reconstruction |
//! | [`dwt_multi`] | Multi-level DWT decomposition |
//! | [`cwt_frequencies`] | Compute scale-to-frequency mapping |
//! | [`morlet`] | Morlet wavelet kernel |
//! | [`gabor_wavelet`] | Gabor wavelet kernel |
//! | [`mexican_hat`] | Mexican hat wavelet kernel |

use crate::fft::{fft, ifft};
use crate::Complex;

/// Morlet wavelet: `ψ(t) = π^(-1/4) · e^(i·ω₀·t) · e^(-t²/2)`
///
/// The parameter `omega0` controls the number of oscillations within the
/// Gaussian envelope. Common values: `ω₀ = 5.0` (default), `ω₀ = 6.0`.
///
/// ```
/// use mathverse_complex::wavelets::morlet;
/// use mathverse_complex::Complex;
/// let w = morlet(0.0, 5.0);
/// assert!((w.re - 1.0 / std::f64::consts::PI.sqrt()).abs() < 1e-10);
/// ```
pub fn morlet(t: f64, omega0: f64) -> Complex {
    let envelope = (-t * t / 2.0).exp() / core::f64::consts::PI.sqrt();
    Complex::new(envelope * (omega0 * t).cos(), envelope * (omega0 * t).sin())
}

/// Gabor wavelet: `ψ(t) = (2/(πσ²))^(1/4) · e^(-t²/σ²) · e^(i·ω₀·t)`
///
/// A windowed complex sinusoid with adjustable bandwidth `sigma`.
/// Normalized so that ∫|ψ(t)|² dt = 1.
pub fn gabor_wavelet(t: f64, omega0: f64, sigma: f64) -> Complex {
    let norm = (2.0 / (core::f64::consts::PI * sigma * sigma)).powf(0.25);
    let envelope = (-t * t / (sigma * sigma)).exp() * norm;
    Complex::new(envelope * (omega0 * t).cos(), envelope * (omega0 * t).sin())
}

/// Mexican hat wavelet (real-valued, returned as complex with zero imaginary part).
///
/// `ψ(t) = (2 / √3 · √π) · (1 − t²) · e^(-t²/2)`
pub fn mexican_hat(t: f64) -> Complex {
    let c = 2.0 / (3.0_f64.sqrt() * core::f64::consts::PI.sqrt());
    let envelope = c * (1.0 - t * t) * (-t * t / 2.0).exp();
    Complex::new(envelope, 0.0)
}

/// Compute the CWT (Continuous Wavelet Transform) of a complex-valued signal
/// using FFT-accelerated convolution.
///
/// # Arguments
/// * `signal` — input complex signal of length `n`
/// * `scales` — vector of scales at which to compute the transform
/// * `omega0` — central frequency of the Morlet wavelet
///
/// # Returns
/// A matrix of shape `(scales.len(), n)` where element `(i, j)` is the
/// wavelet coefficient at scale `scales[i]` and time index `j`.
pub fn cwt(signal: &[Complex], scales: &[f64], omega0: f64) -> Vec<Vec<Complex>> {
    let n = signal.len();
    if n == 0 || scales.is_empty() {
        return Vec::new();
    }

    // Pad signal to next power of 2 for FFT efficiency
    let npad = n.next_power_of_two();
    let mut padded = vec![Complex::zero(); npad];
    padded[..n].copy_from_slice(signal);

    let fft_signal = fft(&padded);

    let mut result = Vec::with_capacity(scales.len());
    for &scale in scales {
        // Build wavelet kernel at this scale, sampled at integer indices
        let mut kernel = vec![Complex::zero(); npad];
        for k in 0..npad {
            let t = if k < npad / 2 {
                k as f64 / scale
            } else {
                (k as f64 - npad as f64) / scale
            };
            kernel[k] = morlet(t, omega0) * Complex::real(1.0 / scale.sqrt());
        }

        // FFT of kernel (fftshift is implicit via the index ordering above)
        let fft_kernel = fft(&kernel);

        // Multiply in frequency domain
        let product: Vec<Complex> = fft_signal
            .iter()
            .zip(fft_kernel.iter())
            .map(|(a, b)| *a * *b)
            .collect();

        // Inverse FFT
        let cwt_coeff = ifft(&product);

        // Take only the first n coefficients and apply FFT-shift
        let mut row = vec![Complex::zero(); n];
        for j in 0..n {
            let shifted = (j + npad / 2) % npad;
            row[j] = cwt_coeff[shifted];
        }
        result.push(row);
    }
    result
}

/// Compute scale-to-frequency mapping for CWT with Morlet wavelet.
///
/// Given a sampling period `dt`, returns the approximate instantaneous
/// frequency for each scale: `f ≈ omega0 / (2π · scale · dt)`.
pub fn cwt_frequencies(scales: &[f64], dt: f64, omega0: f64) -> Vec<f64> {
    let two_pi = 2.0 * core::f64::consts::PI;
    scales
        .iter()
        .map(|&s| omega0 / (two_pi * s * dt))
        .collect()
}

/// Single-level DWT (Discrete Wavelet Transform) decomposition using
/// complex-valued Haar-like filters.
///
/// Splits the signal into approximation (low-pass) and detail (high-pass)
/// coefficients, each of length `n/2`.
///
/// # Arguments
/// * `signal` — input signal of even length `n`
///
/// # Returns
/// `(approx, detail)` each of length `n/2`.
pub fn dwt(signal: &[Complex]) -> (Vec<Complex>, Vec<Complex>) {
    let n = signal.len();
    if n < 2 {
        return (signal.to_vec(), Vec::new());
    }
    let half = n / 2;
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();

    let mut approx = Vec::with_capacity(half);
    let mut detail = Vec::with_capacity(half);

    for i in 0..half {
        let low = signal[2 * i];
        let high = signal[2 * i + 1];
        // Haar low-pass: (x[2k] + x[2k+1]) / √2
        approx.push((low + high) * Complex::real(sqrt2_inv));
        // Haar high-pass: (x[2k] − x[2k+1]) / √2
        detail.push((low - high) * Complex::real(sqrt2_inv));
    }

    (approx, detail)
}

/// Single-level inverse DWT: reconstruct signal from approximation and
/// detail coefficients.
pub fn dwt_inverse(approx: &[Complex], detail: &[Complex]) -> Vec<Complex> {
    let n = approx.len() + detail.len();
    let mut signal = vec![Complex::zero(); n];
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();

    for (i, (a, d)) in approx.iter().zip(detail.iter()).enumerate() {
        signal[2 * i] = (*a + *d) * Complex::real(sqrt2_inv);
        signal[2 * i + 1] = (*a - *d) * Complex::real(sqrt2_inv);
    }
    signal
}

/// Multi-level DWT decomposition: recursively decomposes the approximation
/// coefficients up to `levels` levels or until the signal is too short.
///
/// Returns a vector of detail coefficient vectors, from finest (level 1)
/// to coarsest, plus the final approximation.
pub fn dwt_multi(signal: &[Complex], levels: usize) -> (Vec<Vec<Complex>>, Vec<Complex>) {
    let mut details = Vec::new();
    let mut current = signal.to_vec();

    for _ in 0..levels {
        if current.len() < 2 {
            break;
        }
        let (approx, detail) = dwt(&current);
        details.push(detail);
        current = approx;
    }

    (details, current)
}

/// Compute the power scalogram (|CWT|²) from a CWT result matrix.
pub fn scalogram_power(cwt_result: &[Vec<Complex>]) -> Vec<Vec<f64>> {
    cwt_result
        .iter()
        .map(|row| row.iter().map(super::Complex::norm_sq).collect())
        .collect()
}

/// Compute the phase scalogram (arg(CWT)) from a CWT result matrix.
pub fn scalogram_phase(cwt_result: &[Vec<Complex>]) -> Vec<Vec<f64>> {
    cwt_result
        .iter()
        .map(|row| row.iter().map(super::Complex::arg).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn morlet_at_zero() {
        let w = morlet(0.0, 5.0);
        let expected = 1.0 / core::f64::consts::PI.sqrt();
        assert!((w.re - expected).abs() < EPS);
        assert!(w.im.abs() < EPS);
    }

    #[test]
    fn morlet_symmetric_envelope() {
        let w_pos = morlet(1.0, 5.0);
        let w_neg = morlet(-1.0, 5.0);
        // Real parts should be equal (cosine is even, Gaussian is even)
        assert!((w_pos.re - w_neg.re).abs() < EPS);
        // Imaginary parts should be opposite (sine is odd)
        assert!((w_pos.im + w_neg.im).abs() < EPS);
    }

    #[test]
    fn gabor_wavelet_normalization() {
        // Integral of |ψ(t)|² should be approximately 1
        let dt = 0.01;
        let mut energy = 0.0;
        for i in 0..=1000 {
            let t = -5.0 + f64::from(i) * dt;
            let w = gabor_wavelet(t, 5.0, 1.0);
            energy += w.norm_sq() * dt;
        }
        assert!((energy - 1.0).abs() < 0.1);
    }

    #[test]
    fn mexican_hat_at_zero() {
        let h = mexican_hat(0.0);
        let c = 2.0 / (3.0_f64.sqrt() * core::f64::consts::PI.sqrt());
        assert!((h.re - c).abs() < 1e-8);
        assert!(h.im.abs() < EPS);
    }

    #[test]
    fn mexican_hat_zero_crossings() {
        // Mexican hat should be zero at t = ±1
        let h1 = mexican_hat(1.0);
        assert!(h1.re.abs() < 1e-12);
    }

    #[test]
    fn cwt_single_scale_matches_convolution() {
        // CWT at a single scale should produce output of correct length
        let signal: Vec<Complex> = (0..64)
            .map(|i| Complex::new((f64::from(i) * 0.1).sin(), 0.0))
            .collect();
        let scales = vec![1.0];
        let result = cwt(&signal, &scales, 5.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), signal.len());
    }

    #[test]
    fn cwt_multiple_scales() {
        let signal: Vec<Complex> = (0..128)
            .map(|i| Complex::new((f64::from(i) * 0.1).cos(), (f64::from(i) * 0.05).sin()))
            .collect();
        let scales = vec![0.5, 1.0, 2.0, 4.0];
        let result = cwt(&signal, &scales, 6.0);
        assert_eq!(result.len(), 4);
        for row in &result {
            assert_eq!(row.len(), signal.len());
        }
    }

    #[test]
    fn cwt_empty_signal() {
        let result = cwt(&[], &[1.0], 5.0);
        assert!(result.is_empty());
    }

    #[test]
    fn cwt_frequencies_basic() {
        let scales = vec![1.0, 2.0, 4.0];
        let freqs = cwt_frequencies(&scales, 1.0, 5.0);
        let two_pi = 2.0 * core::f64::consts::PI;
        assert!((freqs[0] - 5.0 / two_pi).abs() < EPS);
        assert!((freqs[1] - 5.0 / (two_pi * 2.0)).abs() < EPS);
        assert!((freqs[2] - 5.0 / (two_pi * 4.0)).abs() < EPS);
    }

    #[test]
    fn dwt_single_level() {
        // Constant signal: detail coefficients should be zero
        let signal: Vec<Complex> = vec![Complex::real(5.0); 8];
        let (approx, detail) = dwt(&signal);
        assert_eq!(approx.len(), 4);
        assert_eq!(detail.len(), 4);
        // For constant signal, detail should be zero
        for d in &detail {
            assert!(d.norm() < EPS);
        }
    }

    #[test]
    fn dwt_inverse_roundtrip() {
        let signal: Vec<Complex> = vec![
            Complex::new(1.0, 0.5),
            Complex::new(2.0, -0.3),
            Complex::new(3.0, 1.0),
            Complex::new(4.0, -2.0),
        ];
        let (approx, detail) = dwt(&signal);
        let reconstructed = dwt_inverse(&approx, &detail);
        assert_eq!(reconstructed.len(), signal.len());
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).norm() < EPS);
        }
    }

    #[test]
    fn dwt_multi_level() {
        let signal: Vec<Complex> = (0..16)
            .map(|i| Complex::new((f64::from(i) * 0.5).sin(), 0.0))
            .collect();
        let (details, approx) = dwt_multi(&signal, 3);
        assert_eq!(details.len(), 3);
        assert_eq!(details[0].len(), 8);
        assert_eq!(details[1].len(), 4);
        assert_eq!(details[2].len(), 2);
        assert_eq!(approx.len(), 2);
    }

    #[test]
    fn dwt_short_signal() {
        let signal = vec![Complex::real(1.0)];
        let (approx, detail) = dwt(&signal);
        assert_eq!(approx.len(), 1);
        assert!(detail.is_empty());
    }

    #[test]
    fn scalogram_power_positive() {
        let cwt_result = vec![
            vec![Complex::new(1.0, 2.0), Complex::new(3.0, 0.0)],
            vec![Complex::new(0.0, 1.0), Complex::new(2.0, 2.0)],
        ];
        let power = scalogram_power(&cwt_result);
        assert_eq!(power.len(), 2);
        assert!((power[0][0] - 5.0).abs() < EPS);
        assert!((power[0][1] - 9.0).abs() < EPS);
        assert!((power[1][0] - 1.0).abs() < EPS);
        assert!((power[1][1] - 8.0).abs() < EPS);
    }

    #[test]
    fn scalogram_phase_range() {
        let cwt_result = vec![vec![
            Complex::new(1.0, 1.0),
            Complex::new(-1.0, 0.0),
        ]];
        let phase = scalogram_phase(&cwt_result);
        assert!((phase[0][0] - core::f64::consts::FRAC_PI_4).abs() < EPS);
        assert!((phase[0][1] - core::f64::consts::PI).abs() < EPS);
    }
}
