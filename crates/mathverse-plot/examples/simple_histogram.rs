//! Histogram with bins chosen by mathverse-statistics rules.

use mathverse_plot::{BinningMethod, Color, Histogram, PlotConfig, SvgPlot};
use mathverse_plot::save::PlotSaver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A little deterministic pseudo-random sample
    let mut rng = 1u64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (rng >> 33) as f64 / (1u64 << 31) as f64
    };
    let data: Vec<f64> = (0..200)
        .map(|_| next() + next() + next() - 1.5) // approx normal-ish, centered at 0
        .collect();

    for method in [
        BinningMethod::Auto,
        BinningMethod::Sturges,
        BinningMethod::Scott,
        BinningMethod::FreedmanDiaconis,
        BinningMethod::Sqrt,
    ] {
        let hist = Histogram::bin(&data, method)?;
        let mut plot = SvgPlot::new(
            PlotConfig::new()
                .with_title(format!("Histogram ({method:?})"))
                .with_x_label("value")
                .with_y_label("count"),
        );
        for (i, &count) in hist.counts().iter().enumerate() {
            let (lo, hi) = (hist.edges()[i], hist.edges()[i + 1]);
            plot.add_bar(lo, hi, count as f64, Color::BLUE);
        }
  let file = format!("histogram_{method:?}.png").to_lowercase();
  PlotSaver::new(plot.generate()).save_png(&file)?;
  println!("{method:?}: {} bins -> {file}", hist.counts().len());
    }

    Ok(())
}
