//! PDF overlay example: theoretical normal PDF on top of sampled data histogram.

use mathverse_plot::{BinningMethod, Color, Histogram, PlotConfig, SvgPlot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Deterministic pseudo-random sample
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

    let mean = mathverse_statistics::mean(&data);
    let std = mathverse_statistics::std_dev_sample(&data);

    let hist = Histogram::bin(&data, BinningMethod::Auto)?;
    let nbins = hist.counts().len();
    let bin_width = (hist.edges()[nbins] - hist.edges()[0]) / nbins as f64;

    let mut plot = SvgPlot::new(
        PlotConfig::new()
            .with_title("Histogram + Normal PDF overlay")
            .with_x_label("value")
            .with_y_label("count"),
    );
    for (i, &count) in hist.counts().iter().enumerate() {
        let (lo, hi) = (hist.edges()[i], hist.edges()[i + 1]);
        plot.add_bar(lo, hi, count as f64, Color::BLUE);
    }

    // PDF scaled by n * bin_width to match histogram area
    let scale = data.len() as f64 * bin_width;
    let mu = mean;
    let sigma = std;
    plot.add_pdf_overlay(
        "N(μ,σ)",
        Box::new(move |x| {
            let z = (x - mu) / sigma;
            (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt()) * scale
        }),
        hist.edges()[0],
        hist.edges()[nbins],
        200,
        Color::RED,
    );

    std::fs::write("pdf_overlay.svg", plot.generate())?;
    println!("wrote pdf_overlay.svg");
    Ok(())
}
