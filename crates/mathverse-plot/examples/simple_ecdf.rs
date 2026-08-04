//! ECDF plot example: Empirical cumulative distribution.

use mathverse_plot::ecdf::{render_ecdf, EcdfConfig};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    // Generate sample data from a normal-like distribution
    let values: Vec<f64> = (0..100)
        .map(|i| {
            let x = i as f64 * 0.1;
            5.0 + (x.sin() * 2.0) + ((i % 7) as f64 - 3.0) * 0.3
        })
        .collect();

    let mut cfg = EcdfConfig::new()
        .with_confidence(0.95);
    cfg.color = Color::rgb(0, 100, 200);
    cfg.plot_config.title = "ECDF with 95% CI".into();
    cfg.plot_config.width = 600;
    cfg.plot_config.height = 400;

    let svg = render_ecdf(&values, &cfg)?;
    std::fs::write("ecdf.svg", svg)?;
    println!("wrote ecdf.svg");

    Ok(())
}
