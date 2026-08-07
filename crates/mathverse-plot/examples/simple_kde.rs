//! Histogram with a Gaussian KDE overlay (Silverman bandwidth), both from
//! `mathverse-statistics`. The KDE is scaled by `n * bin_width` so its area
//! matches the histogram.

use mathverse_plot::{BinningMethod, Color, Histogram, PlotConfig, SvgPlot};
use mathverse_plot::save::PlotSaver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Deterministic pseudo-random sample (sum of 4 uniforms ~ normal-ish)
    let mut rng = 1u64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng >> 33) as f64 / (1u64 << 31) as f64
    };
    let data: Vec<f64> = (0..300)
        .map(|_| next() + next() + next() + next() - 2.0)
        .collect();

    let hist = Histogram::bin(&data, BinningMethod::Auto)?;
    let nbins = hist.counts().len();
    let bin_width = (hist.edges()[nbins] - hist.edges()[0]) / nbins as f64;

    let mut plot = SvgPlot::new(
        PlotConfig::new()
            .with_title("Histogram + KDE overlay")
            .with_x_label("value")
            .with_y_label("count"),
    );
    for (i, &count) in hist.counts().iter().enumerate() {
        let (lo, hi) = (hist.edges()[i], hist.edges()[i + 1]);
        plot.add_bar(lo, hi, count as f64, Color::BLUE);
    }

    // Scale factor: KDE area (1.0) scaled up to the histogram area (n * bin_width)
    let scale = data.len() as f64 * bin_width;
    plot.add_kde_overlay(
        "kde",
        &data,
        mathverse_statistics::Bandwidth::Silverman,
        scale,
        Color::RED,
        200,
    )?;

    PlotSaver::new(&plot.generate()).save_png("histogram_kde.png")?;
    println!("wrote histogram_kde.png");
    Ok(())
}
