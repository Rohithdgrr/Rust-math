//! Histogram with bins chosen by mathverse-statistics rules.

use mathverse_plot::{BinningMethod, Color, Histogram, PlotConfig, SvgPlot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A little deterministic pseudo-random sample
    let mut rng = 1u64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
        let file = format!("histogram_{:?}.svg", method).to_lowercase();
        std::fs::write(&file, plot.generate())?;
        println!("{method:?}: {} bins -> {file}", hist.counts().len());
    }

    Ok(())
}
