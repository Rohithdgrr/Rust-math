//! 1D convolution, correlation, FIR filter design (low-pass, high-pass, band-pass).

/// Linear convolution of two signals.
pub fn convolve(x: &[f64], h: &[f64]) -> Vec<f64> {
    let (n, m) = (x.len(), h.len());
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0; n + m - 1];
    for (i, &xi) in x.iter().enumerate() {
        for (j, &hj) in h.iter().enumerate() {
            out[i + j] += xi * hj;
        }
    }
    out
}

/// Cross-correlation of two signals (convolution with the reversed kernel).
#[must_use]
pub fn correlate(x: &[f64], h: &[f64]) -> Vec<f64> {
    let rev: Vec<f64> = h.iter().rev().copied().collect();
    convolve(x, &rev)
}

/// Apply an FIR filter with coefficients `b` to a signal.
#[must_use]
pub fn fir(x: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for n in 0..x.len() {
        let mut s = 0.0;
        for (j, &bj) in b.iter().enumerate() {
            if j <= n {
                s += bj * x[n - j];
            }
        }
        out.push(s);
    }
    out
}

/// Design a windowed-sinc low-pass FIR filter at the given cutoff (normalized
/// to the sample rate) using a Hamming window.
#[must_use]
pub fn fir_lowpass(cutoff: f64, taps: usize) -> Vec<f64> {
    let m = taps / 2;
    let mut h = Vec::with_capacity(taps);
    for n in 0..taps {
        let i = n as f64 - m as f64;
        let v = if i == 0.0 {
            2.0 * cutoff
        } else {
            (2.0 * core::f64::consts::PI * cutoff * i).sin() / (core::f64::consts::PI * i)
        };
        let w = 0.54 + 0.46 * (core::f64::consts::PI * i / m as f64).cos();
        h.push(v * w);
    }
    let g: f64 = h.iter().sum();
    h.iter().map(|v| v / g).collect()
}

/// Design a high-pass FIR filter by spectral inversion of a low-pass design.
#[must_use]
pub fn fir_highpass(cutoff: f64, taps: usize) -> Vec<f64> {
    let lp = fir_lowpass(cutoff, taps);
    lp.iter()
        .map(|&v| -v)
        .collect::<Vec<_>>()
        .iter()
        .enumerate()
        .map(|(i, &v)| if i == taps / 2 { 1.0 + v } else { v })
        .collect()
}

/// Design a band-pass FIR filter between `low` and `high` (normalized
/// frequencies) via spectral shifting of a low-pass prototype.
#[must_use]
pub fn fir_bandpass(low: f64, high: f64, taps: usize) -> Vec<f64> {
    let m = taps / 2;
    let mut h = Vec::with_capacity(taps);
    for n in 0..taps {
        let i = n as f64 - m as f64;
        let center = f64::midpoint(high, low);
        let width = (high - low) / 2.0;
        let v = if i == 0.0 {
            2.0 * width
        } else {
            (2.0 * core::f64::consts::PI * width * i).sin() / (core::f64::consts::PI * i)
        };
        let modulator = (2.0 * core::f64::consts::PI * center * i).cos();
        let w = 0.54 + 0.46 * (core::f64::consts::PI * i / m as f64).cos();
        h.push(v * modulator * w);
    }
    let g: f64 = h.iter().sum();
    h.iter().map(|v| v / g).collect()
}

/// Compute a trailing moving average over the given window length.
#[must_use]
pub fn moving_average(x: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || x.is_empty() {
        return x.to_vec();
    }
    let mut out = Vec::with_capacity(x.len());
    let mut sum = 0.0;
    let mut count = 0;
    for (i, &xi) in x.iter().enumerate() {
        sum += xi;
        count += 1;
        if count > window {
            sum -= x[i - window];
            count -= 1;
        }
        out.push(sum / count as f64);
    }
    out
}

/// Apply a sliding-window median filter.
///
/// The window includes the current sample and up to `window - 1` predecessors.
#[must_use]
pub fn median_filter(x: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || x.is_empty() {
        return x.to_vec();
    }
    let mut out = Vec::with_capacity(x.len());
    for i in 0..x.len() {
        let start = i.saturating_sub(window - 1);
        let end = (i + 1).min(x.len());
        let mut w: Vec<f64> = x[start..end].to_vec();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if w.len().is_multiple_of(2) {
            f64::midpoint(w[w.len() / 2 - 1], w[w.len() / 2])
        } else {
            w[w.len() / 2]
        };
        out.push(median);
    }
    out
}

/// Compute the root-mean-square amplitude of a signal.
#[must_use]
pub fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

/// Find local maxima in a signal (strictly greater than both neighbors).
#[must_use]
pub fn find_peaks(x: &[f64]) -> Vec<usize> {
    if x.len() < 3 {
        return Vec::new();
    }
    (1..x.len() - 1)
        .filter(|&i| x[i] > x[i - 1] && x[i] > x[i + 1])
        .collect()
}

/// Find local maxima whose value also exceeds `threshold`.
#[must_use]
pub fn find_peaks_threshold(x: &[f64], threshold: f64) -> Vec<usize> {
    if x.len() < 3 {
        return Vec::new();
    }
    (1..x.len() - 1)
        .filter(|&i| x[i] > x[i - 1] && x[i] > x[i + 1] && x[i] > threshold)
        .collect()
}

/// Compute the difference between the maximum and minimum sample values.
#[must_use]
pub fn peak_to_peak(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - x.iter().copied().fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_test() {
        assert_eq!(
            convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]),
            vec![1.0, 3.0, 5.0, 3.0]
        );
    }

    #[test]
    fn peaks() {
        let x = [1.0, 3.0, 2.0, 4.0, 1.0];
        let p = find_peaks(&x);
        assert!(p.contains(&1));
        assert!(p.contains(&3));
    }
}
