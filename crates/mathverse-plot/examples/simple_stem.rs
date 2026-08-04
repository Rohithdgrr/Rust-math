//! Stem plot example: Discrete signal with positive and negative values.

use mathverse_plot::common::DataPoint;
use mathverse_plot::stem::{render_stem_plot, StemConfig};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let points: Vec<DataPoint> = (0..=10)
        .map(|i| {
            let x = i as f64;
            let y = (x * 0.8).sin() * 5.0 + (i % 3) as f64 - 1.0;
            DataPoint::new(x, y)
        })
        .collect();

    let config = StemConfig::new()
        .with_stem_color(Color::rgb(66, 133, 244))
        .with_marker_color(Color::rgb(220, 20, 60))
        .with_marker_radius(5.0);

    let mut cfg = config;
    cfg.plot_config.title = "Discrete Signal".into();
    cfg.plot_config.width = 600;
    cfg.plot_config.height = 400;

    let svg = render_stem_plot(&points, &cfg)?;
    std::fs::write("stem.svg", svg)?;
    println!("wrote stem.svg");

    Ok(())
}
