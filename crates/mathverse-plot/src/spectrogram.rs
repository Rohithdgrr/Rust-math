//! Spectrogram / FFT visualization via `mathverse-transforms` + `mathverse-signal`.
//!
//! Renders a time-frequency heatmap (spectrogram) as SVG, using the
//! Welch PSD estimate for smooth spectral averaging and the periodogram
//! for single-shot spectral views.

use mathverse_signal::spectrum::{periodogram, welch_psd};
use mathverse_transforms::fft::fft_real;

use crate::common::PlotConfig;
use crate::heatmap::Colormap;
use crate::error::{PlotError, PlotResult};
use crate::svg::SvgPlot;

/// Spectrogram rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpectrogramMode {
    /// Welch averaged PSD (smooth, recommended for noisy signals).
    #[default]
    Welch,
    /// Single-shot periodogram (higher variance, sharper peaks).
    Periodogram,
    /// Raw FFT magnitude spectrum (complex-valued input).
    FftMagnitude,
}

/// Configuration for a spectrogram plot.
#[derive(Debug, Clone)]
pub struct SpectrogramConfig {
    /// Plot configuration (title, labels, dimensions, colours).
    pub plot_config: PlotConfig,
    /// Spectrogram rendering mode.
    pub mode: SpectrogramMode,
    /// Welch segment size (samples per segment). Ignored for non-Welch modes.
    pub segment_size: usize,
    /// Welch overlap between segments. Ignored for non-Welch modes.
    pub overlap: usize,
    /// Colormap applied to the spectral magnitude.
    pub colormap: Colormap,
    /// Minimum frequency (Hz) for the y-axis; auto-detected when `None`.
    pub f_min: Option<f64>,
    /// Maximum frequency (Hz) for the y-axis; auto-detected when `None`.
    pub f_max: Option<f64>,
    /// Sample rate (Hz) used for frequency axis labelling.
    pub sample_rate: f64,
}

impl SpectrogramConfig {
    /// Create a new spectrogram config with sensible defaults.
    #[must_use]
    pub fn new(sample_rate: f64) -> Self {
        Self {
            plot_config: PlotConfig::new()
                .with_title("Spectrogram".to_string())
                .with_x_label("Time (s)")
                .with_y_label("Frequency (Hz)"),
            mode: SpectrogramMode::Welch,
            segment_size: 256,
            overlap: 128,
            colormap: crate::color::viridis,
            f_min: None,
            f_max: None,
            sample_rate,
        }
    }

    /// Set the spectrogram rendering mode.
    #[must_use]
    pub fn with_mode(mut self, mode: SpectrogramMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the Welch segment size.
    #[must_use]
    pub fn with_segment_size(mut self, size: usize) -> Self {
        self.segment_size = size.max(1);
        self
    }

    /// Set the Welch overlap between segments.
    #[must_use]
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the colormap.
    #[must_use]
    pub fn with_colormap(mut self, colormap: Colormap) -> Self {
        self.colormap = colormap;
        self
    }
}

/// Compute the spectrogram data as a 2D grid of spectral magnitudes.
///
/// Returns `(frequencies, times, magnitudes)` where `magnitudes[freq_idx][time_idx]`
/// is the spectral magnitude at that frequency-time bin.
pub fn compute_spectrogram(
    signal: &[f64],
    config: &SpectrogramConfig,
) -> PlotResult<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> {
    if signal.is_empty() {
        return Err(PlotError::InvalidData("empty signal".into()));
    }

    let sr = config.sample_rate;
    if sr <= 0.0 {
        return Err(PlotError::InvalidData("sample_rate must be positive".into()));
    }

    let (freqs, times, grid) = match config.mode {
        SpectrogramMode::Welch => {
            let segment_size = config.segment_size.min(signal.len());
            let overlap = config.overlap.min(segment_size - 1);
            let hop = segment_size - overlap;
            let n_segments = if signal.len() >= segment_size {
                (signal.len() - segment_size) / hop + 1
            } else {
                0
            };

            let df = sr / config.segment_size as f64;
            let n_fft = next_power_of_two(config.segment_size);
            let freqs: Vec<f64> = (0..n_fft).map(|k| k as f64 * df).collect();

            let times: Vec<f64> = (0..n_segments)
                .map(|i| i as f64 * hop as f64 / sr)
                .collect();

            let mut grid = vec![vec![0.0; n_segments]; n_fft];
            for (t_idx, &t_start) in times.iter().enumerate() {
                let start = (t_start * sr) as usize;
                let end = (start + segment_size).min(signal.len());
                if start >= signal.len() {
                    continue;
                }
                let segment = &signal[start..end];
                let psd = welch_psd(segment, segment_size.min(segment.len()), overlap);
                for (f_idx, mag) in psd.iter().enumerate().take(n_fft) {
                    grid[f_idx][t_idx] = *mag;
                }
            }

            (freqs, times, grid)
        }
        SpectrogramMode::Periodogram => {
            let psd = periodogram(signal);
            let n_fft = psd.len();
            let df = sr / n_fft as f64;
            let freqs: Vec<f64> = (0..n_fft).map(|k| k as f64 * df).collect();
            let times = vec![0.0];
            (freqs, times, vec![psd])
        }
        SpectrogramMode::FftMagnitude => {
            let n_fft = next_power_of_two(signal.len());
            let mut padded = signal.to_vec();
            padded.resize(n_fft, 0.0);
            let spectrum = fft_real(&padded);
            let df = sr / n_fft as f64;
            let freqs: Vec<f64> = (0..n_fft).map(|k| k as f64 * df).collect();
            let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
            let times = vec![0.0];
            (freqs, times, vec![magnitudes])
        }
    };

    Ok((freqs, times, grid))
}

fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        1 << (usize::BITS - (n - 1).leading_zeros())
    }
}

/// Render a spectrogram as an SVG string.
pub fn render_spectrogram_svg(
    signal: &[f64],
    config: SpectrogramConfig,
) -> PlotResult<String> {
    let (_, _, magnitudes) = compute_spectrogram(signal, &config)?;

    let mut plot = SvgPlot::new(config.plot_config);
    plot.add_heatmap("spectrogram", magnitudes, config.colormap)?;

    Ok(plot.generate())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectrogram_rejects_empty_signal() {
        let config = SpectrogramConfig::new(1000.0);
        let result = compute_spectrogram(&[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn spectrogram_rejects_zero_sample_rate() {
        let config = SpectrogramConfig::new(0.0);
        let result = compute_spectrogram(&[1.0, 2.0, 3.0], &config);
        assert!(result.is_err());
    }

    #[test]
    fn welch_spectrogram_produces_grid() {
        let sr = 1000.0;
        let t: Vec<f64> = (0..1000).map(|i| i as f64 / sr).collect();
        let signal: Vec<f64> = t.iter().map(|&t| (2.0 * core::f64::consts::PI * 50.0 * t).sin()).collect();

        let config = SpectrogramConfig::new(sr)
            .with_mode(SpectrogramMode::Welch)
            .with_segment_size(256)
            .with_overlap(128);

        let (freqs, times, magnitudes) = compute_spectrogram(&signal, &config).unwrap();
        assert!(!freqs.is_empty());
        assert!(!times.is_empty());
        assert_eq!(magnitudes.len(), freqs.len());
        assert!(!magnitudes[0].is_empty());
    }

    #[test]
    fn periodogram_mode_returns_single_time_slice() {
        let sr = 1000.0;
        let signal = vec![1.0, 0.5, -0.5, -1.0, -0.5, 0.5];

        let config = SpectrogramConfig::new(sr).with_mode(SpectrogramMode::Periodogram);

        let (freqs, times, magnitudes) = compute_spectrogram(&signal, &config).unwrap();
        assert_eq!(times.len(), 1);
        assert_eq!(magnitudes.len(), 1);
        assert!(!freqs.is_empty());
    }

    #[test]
    fn fft_magnitude_mode_returns_spectrum() {
        let sr = 1000.0;
        let signal = vec![1.0, 0.0, -1.0, 0.0];

        let config = SpectrogramConfig::new(sr).with_mode(SpectrogramMode::FftMagnitude);

        let (freqs, times, magnitudes) = compute_spectrogram(&signal, &config).unwrap();
        assert_eq!(times.len(), 1);
        assert_eq!(magnitudes.len(), 1);
        assert!(!freqs.is_empty());
    }

    #[test]
    fn render_spectrogram_svg_contains_svg_tag() {
        let sr = 1000.0;
        let t: Vec<f64> = (0..500).map(|i| i as f64 / sr).collect();
        let signal: Vec<f64> = t.iter().map(|&t| (2.0 * core::f64::consts::PI * 50.0 * t).sin()).collect();

        let config = SpectrogramConfig::new(sr);
        let svg = render_spectrogram_svg(&signal, config).unwrap();
        assert!(svg.contains("<svg"));
    }
}