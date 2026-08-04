//! Residual plot example.

use mathverse_plot::{render_residplot, ResidConfig, common::DataPoint};

fn main() -> mathverse_plot::PlotResult<()> {
    // Generate data with a linear trend and heteroscedastic noise
    let points: Vec<DataPoint> = (0..40)
        .map(|i| {
            let x = i as f64 * 0.3;
            let noise = (i as f64 * 0.4).sin() * (1.0 + x * 0.2);
            let y = 2.0 * x + 1.0 + noise;
            DataPoint::new(x, y)
        })
        .collect();

    let mut config = ResidConfig::new();
    config.plot_config = config.plot_config
        .with_title("Residual Plot: Fit Diagnostics");

    let svg = render_residplot(&points, &config)?;
    std::fs::write("residplot.svg", svg)?;
    println!("wrote residplot.svg");

    Ok(())
}
