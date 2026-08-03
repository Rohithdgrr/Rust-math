//! Power spectral density via periodogram, frequency/amplitude/phase spectra.
//!
//! This module uses FFT from [`mathverse_transforms`] for efficient spectral analysis.

use mathverse_complex::Complex;
use mathverse_transforms::fft::fft;

/// Helper to compute next power of two.
#[inline]
fn next_power_of_two(n: usize) -> usize {
    if n == 0 { return 1; }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Compute power spectral density using the periodogram method.
///
/// Uses FFT for O(n log n) performance instead of naive O(n²) DFT.
/// Input is zero-padded to the next power of 2 for FFT efficiency.
///
/// # Examples
/// ```
/// use mathverse_signal::spectrum::periodogram;
/// let signal = vec![1.0, 0.5, -0.5, -1.0];
/// let psd = periodogram(&signal);
/// assert!(psd.len() >= signal.len());
/// ```
pub fn periodogram(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }
    
    let n = x.len();
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    
    // Remove DC component and convert to complex
    let centered: Vec<Complex> = x.iter()
        .map(|&v| Complex::new(v - mean, 0.0))
        .collect();
    
    // Zero-pad to next power of 2 for FFT
    let fft_size = next_power_of_two(n);
    let mut padded = centered;
    padded.resize(fft_size, Complex::new(0.0, 0.0));
    
    // Compute FFT
    let spectrum = match fft(&padded) {
        Ok(s) => s,
        Err(_) => return vec![0.0; n], // Fallback on error
    };
    
    // Compute power: |X[k]|² / n
    spectrum.iter()
        .map(|c| (c.re * c.re + c.im * c.im) / n as f64)
        .collect()
}

/// Welch's method for power spectral density estimation with overlapping segments.
///
/// Reduces variance compared to simple periodogram by averaging multiple overlapped,
/// windowed periodograms. Uses Hamming window by default.
///
/// # Examples
/// ```
/// use mathverse_signal::spectrum::welch_psd;
/// let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
/// let psd = welch_psd(&signal, 64, 32);
/// assert!(psd.len() > 0);
/// ```
pub fn welch_psd(x: &[f64], segment_size: usize, overlap: usize) -> Vec<f64> {
    if x.is_empty() || segment_size == 0 || overlap >= segment_size || segment_size > x.len() {
        return periodogram(x);
    }
    
    let hop = segment_size - overlap;
    let n_segments = (x.len() - segment_size) / hop + 1;
    
    if n_segments == 0 { 
        return periodogram(x); 
    }
    
    // Initialize accumulator for the correct FFT size
    let fft_size = next_power_of_two(segment_size);
    let mut psd = vec![0.0; fft_size];
    
    for seg in 0..n_segments {
        let start = seg * hop;
        let segment = &x[start..start + segment_size];
        
        // Apply Hamming window
        let windowed: Vec<f64> = segment.iter().enumerate().map(|(i, &v)| {
            let w = 0.54 - 0.46 * (2.0 * core::f64::consts::PI * i as f64 / (segment_size - 1) as f64).cos();
            v * w
        }).collect();
        
        let pg = periodogram(&windowed);
        for i in 0..pg.len().min(psd.len()) { 
            psd[i] += pg[i]; 
        }
    }
    
    psd.iter().map(|v| v / n_segments as f64).collect()
}

pub fn autocorrelation(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mean: f64 = x.iter().sum::<f64>() / n as f64;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    if var == 0.0 { return vec![1.0; n]; }
    (0..n).map(|lag| {
        let sum: f64 = (0..n - lag).map(|i| (x[i] - mean) * (x[i + lag] - mean)).sum();
        sum / (n as f64 * var)
    }).collect()
}

pub fn energy(x: &[f64]) -> f64 { x.iter().map(|v| v * v).sum() }

pub fn parseval(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let time_energy: f64 = x.iter().map(|v| v * v).sum();
    let mean: f64 = x.iter().sum::<f64>() / n;
    let freq_energy: f64 = periodogram(x).iter().sum();
    (time_energy - freq_energy).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodogram_test() {
        let x: Vec<f64> = (0..64).map(|i| (2.0 * core::f64::consts::PI * i as f64 / 16.0).sin()).collect();
        let psd = periodogram(&x);
        assert_eq!(psd.len(), 64);
    }
}
