//! Simple spectrogram example using FFT-based spectral analysis.

use mathverse_plot::{SpectrogramConfig, SpectrogramMode, render_spectrogram_svg};
use mathverse_plot::save::PlotSaver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a test signal: 100 Hz sine wave with 440 Hz sine wave starting at t=0.5s
    let sample_rate = 1000.0; // Hz
    let duration = 2.0; // seconds
    let n_samples = (sample_rate * duration) as usize;

    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let low_freq = (2.0 * std::f64::consts::PI * 100.0 * t).sin();
            let high_freq = if t > 0.5 {
                0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
            } else {
                0.0
            };
            low_freq + high_freq
        })
        .collect();

    // Create spectrogram config
    let config = SpectrogramConfig::new(sample_rate)
        .with_mode(SpectrogramMode::Welch)
        .with_segment_size(128)
        .with_overlap(64);

    // Render to SVG
    let svg = render_spectrogram_svg(&signal, config)?;
    PlotSaver::new(&svg).save_png("spectrogram.png")?;
    println!("Wrote spectrogram.svg ({} bytes)", svg.len());

    Ok(())
}
