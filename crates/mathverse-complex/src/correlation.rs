//! FFT-accelerated autocorrelation and cross-correlation for complex
//! signals.
//!
//! Correlation measures the similarity between two signals as a function
//! of time shift, essential for time-delay estimation, matched filtering,
//! signal detection, and synchronization in communications.
//!
//! # Module overview
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`autocorrelation`] | Autocorrelation via FFT |
//! | [`cross_correlation`] | Cross-correlation between two signals via FFT |
//! | [`xcorr_lags`] | Cross-correlation with lag vector |
//! | [`normalized_xcorr`] | Normalized cross-correlation coefficient |
//! | [`find_delay`] | Estimate time delay between two signals |

use crate::fft::{fft, ifft};
use crate::Complex;
use mathverse_core::error::{MathError, MathResult};

/// Compute the autocorrelation of a complex signal using FFT.
///
/// `R_xx[k] = Σ_n x[n] · conj(x[n+k])` zero-padded to avoid
/// circular aliasing.
///
/// # Arguments
/// * `signal` — input signal of length `n`
///
/// # Returns
/// Autocorrelation vector of length `2n − 1`, where index `n − 1 + k`
/// corresponds to lag `k`.
pub fn autocorrelation(signal: &[Complex]) -> Vec<Complex> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    let npad = (2 * n).next_power_of_two();

    let mut padded = vec![Complex::zero(); npad];
    padded[..n].copy_from_slice(signal);

    let fft_sig = fft(&padded);

    // Power spectrum: |FFT(x)|²
    let power: Vec<Complex> = fft_sig.iter().map(|f| *f * f.conjugate()).collect();

    let full = ifft(&power);

    // Reorder from circular IFFT output to linear correlation.
    // IFFT(X · conj(Y))[k] = R_xy[-k] = sum_n x[n] · conj(y[n-k])
    // To get R_xy[lag], we need full[(npad - lag) % npad].
    // Output order: lags -(n-1), ..., -1, 0, 1, ..., n-1
    let mut result = vec![Complex::zero(); 2 * n - 1];
    for lag in -(n as i64 - 1)..(n as i64) {
        let out_idx = (n as i64 - 1 + lag) as usize;
        let full_idx = ((npad as i64 - lag) % npad as i64) as usize;
        result[out_idx] = full[full_idx];
    }
    result
}

/// Compute the cross-correlation of two complex signals using FFT.
///
/// `R_xy[k] = Σ_n x[n] · conj(y[n+k])` zero-padded to avoid
/// circular aliasing.
///
/// # Arguments
/// * `x` — first signal of length `n`
/// * `y` — second signal of length `m` (must be ≤ `n`)
///
/// # Returns
/// Cross-correlation vector of length `n + m − 1`, where index `m − 1 + k`
/// corresponds to lag `k` (so lag 0 is at index `m − 1`).
///
/// # Errors
/// Returns an error if `y` is longer than `x`.
pub fn cross_correlation(x: &[Complex], y: &[Complex]) -> MathResult<Vec<Complex>> {
    if y.len() > x.len() {
        return Err(MathError::DimensionMismatch);
    }
    let nx = x.len();
    let ny = y.len();
    let total = nx + ny - 1;
    let npad = total.next_power_of_two();

    let mut px = vec![Complex::zero(); npad];
    let mut py = vec![Complex::zero(); npad];
    px[..nx].copy_from_slice(x);
    py[..ny].copy_from_slice(y);

    let fft_x = fft(&px);
    let fft_y = fft(&py);

    // R_xy = IFFT(FFT(x) · conj(FFT(y)))
    let product: Vec<Complex> = fft_x
        .iter()
        .zip(fft_y.iter())
        .map(|(a, b)| *a * b.conjugate())
        .collect();

    let full = ifft(&product);

    // Reorder from circular IFFT output to linear correlation.
    // IFFT(X · conj(Y))[k] = R_xy[-k] = sum_n x[n] · conj(y[n-k])
    // To get R_xy[lag], we need full[(npad - lag) % npad].
    // Output order: lags -(ny-1), ..., -1, 0, 1, ..., nx-1
    let mut result = vec![Complex::zero(); total];
    for lag in -(ny as i64 - 1)..(nx as i64) {
        let out_idx = (ny as i64 - 1 + lag) as usize;
        let full_idx = ((npad as i64 - lag) % npad as i64) as usize;
        result[out_idx] = full[full_idx];
    }
    Ok(result)
}

/// Compute cross-correlation with explicit lag vector.
///
/// Returns `(lags, corr)` where `lags[k] = k − (ny − 1)` so that
/// `lags[0] = -(ny−1)` and `lags[nx+ny-2] = nx−1`.
pub fn xcorr_lags(
    x: &[Complex],
    y: &[Complex],
) -> MathResult<(Vec<i64>, Vec<Complex>)> {
    let corr = cross_correlation(x, y)?;
    let nx = x.len();
    let ny = y.len();
    let total = nx + ny - 1;

    let lags: Vec<i64> = (0..total).map(|k| k as i64 - (ny as i64 - 1)).collect();
    Ok((lags, corr))
}

/// Compute the normalized cross-correlation coefficient at each lag.
///
/// `ρ_xy[k] = R_xy[k] / sqrt(R_xx[0] · R_yy[0])`
///
/// Result is in `[-1, 1]` for real signals; for complex signals, the
/// magnitude is in `[0, 1]`.
pub fn normalized_xcorr(x: &[Complex], y: &[Complex]) -> MathResult<Vec<f64>> {
    let corr = cross_correlation(x, y)?;
    // R_xx[0] (zero lag) is at index n-1 in the autocorrelation output
    let ac_x = autocorrelation(x);
    let ac_y = autocorrelation(y);
    let energy_x: f64 = ac_x.get(x.len() - 1).map_or(0.0, super::Complex::norm);
    let energy_y: f64 = ac_y.get(y.len() - 1).map_or(0.0, super::Complex::norm);
    let denom = (energy_x * energy_y).sqrt();

    if denom < 1e-15 {
        return Ok(vec![0.0; corr.len()]);
    }

    Ok(corr.iter().map(|c| c.norm() / denom).collect())
}

/// Estimate the time delay between two signals by finding the lag of
/// maximum cross-correlation magnitude.
///
/// # Returns
/// `(delay, max_correlation_magnitude)` where `delay` is the lag index
/// relative to the start of `y` within `x`.
pub fn find_delay(x: &[Complex], y: &[Complex]) -> MathResult<(i64, f64)> {
    let corr = cross_correlation(x, y)?;
    let ny = y.len();

    let mut max_mag = 0.0;
    let mut max_idx = 0;
    for (i, c) in corr.iter().enumerate() {
        let m = c.norm();
        if m > max_mag {
            max_mag = m;
            max_idx = i;
        }
    }

    let delay = max_idx as i64 - (ny as i64 - 1);
    Ok((delay, max_mag))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autocorrelation_peak_at_zero_lag() {
        let signal: Vec<Complex> = (0..64)
            .map(|i| Complex::new((f64::from(i) * 0.3).sin(), 0.0))
            .collect();
        let ac = autocorrelation(&signal);
        assert_eq!(ac.len(), 127);

        // Peak should be at the center (zero lag = index n-1 = 63)
        let peak = ac[63].norm();
        for (i, c) in ac.iter().enumerate() {
            if i != 63 {
                assert!(
                    c.norm() <= peak + 1e-10,
                    "Lag {} ({}) exceeded peak at zero lag ({})",
                    i as i64 - 63,
                    c.norm(),
                    peak
                );
            }
        }
    }

    #[test]
    fn autocorrelation_real_signal_is_real() {
        let signal: Vec<Complex> = vec![
            Complex::real(1.0),
            Complex::real(2.0),
            Complex::real(3.0),
            Complex::real(4.0),
        ];
        let ac = autocorrelation(&signal);
        for c in &ac {
            assert!(c.im.abs() < 1e-10, "Imaginary part: {}", c.im);
        }
    }

    #[test]
    fn cross_correlation_shifted_signal() {
        let n = 32;
        let signal: Vec<Complex> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.2).cos(), 0.0))
            .collect();
        let shift = 5;
        let mut shifted = vec![Complex::zero(); n];
        shifted[shift..].copy_from_slice(&signal[..n - shift]);

        let (lags, corr) = xcorr_lags(&signal, &shifted).unwrap();

        // Find peak lag
        let mut max_mag = 0.0;
        let mut peak_lag = 0i64;
        for (l, c) in lags.iter().zip(corr.iter()) {
            let m = c.norm();
            if m > max_mag {
                max_mag = m;
                peak_lag = *l;
            }
        }
        assert_eq!(peak_lag, shift as i64);
    }

    #[test]
    fn cross_correlation_identical_signals() {
        let signal: Vec<Complex> = vec![
            Complex::new(1.0, 0.5),
            Complex::new(2.0, -0.3),
            Complex::new(3.0, 1.0),
            Complex::new(4.0, -2.0),
        ];
        let corr = cross_correlation(&signal, &signal).unwrap();
        // Peak at zero lag
        let center = signal.len() - 1;
        let peak = corr[center].norm();
        for c in &corr {
            assert!(c.norm() <= peak + 1e-10);
        }
    }

    #[test]
    fn normalized_xcorr_peak_is_one() {
        let signal: Vec<Complex> = (0..16)
            .map(|i| Complex::new(f64::from(i).sin(), (f64::from(i) * 0.5).cos()))
            .collect();
        let nxcorr = normalized_xcorr(&signal, &signal).unwrap();
        let max_val = nxcorr.iter().copied().fold(0.0_f64, f64::max);
        assert!((max_val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn find_delay_correct() {
        let n = 64;
        let signal: Vec<Complex> = (0..n)
            .map(|i| Complex::new(((i as f64) * 0.15).sin(), 0.0))
            .collect();
        let delay = 7;
        let mut delayed = vec![Complex::zero(); n];
        delayed[delay..].copy_from_slice(&signal[..n - delay]);

        let (est_delay, _) = find_delay(&signal, &delayed).unwrap();
        assert_eq!(est_delay, delay as i64);
    }

    #[test]
    fn cross_correlation_different_lengths() {
        let x: Vec<Complex> = (0..8).map(|i| Complex::real(f64::from(i))).collect();
        let y: Vec<Complex> = (0..4).map(|i| Complex::real(f64::from(i))).collect();
        let corr = cross_correlation(&x, &y).unwrap();
        assert_eq!(corr.len(), 11); // 8 + 4 - 1
    }

    #[test]
    fn empty_signals() {
        let ac = autocorrelation(&[]);
        assert!(ac.is_empty());
    }
}
