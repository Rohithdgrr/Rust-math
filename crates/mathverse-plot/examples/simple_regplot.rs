//! Regression plot example.

use mathverse_plot::{render_regplot, RegPlotConfig, common::DataPoint};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    // Generate points with a linear trend plus noise
    let points: Vec<DataPoint> = (0..30)
        .map(|i| {
            let x = i as f64 * 0.5;
            let y = 1.5 * x + 2.0 + ((i as f64 * 0.7).sin() * 1.5);
            DataPoint::new(x, y)
        })
        .collect();

    let mut config = RegPlotConfig::new()
        .with_ci(0.95);

    config.plot_config = config.plot_config
        .with_title("Regression Plot: y = 1.5x + 2 + noise");
    config.show_equation = true;

    let svg = render_regplot(&points, &config)?;
    PlotSaver::new(svg).save_png("regplot.png")?;
    println!("wrote regplot.png");

    Ok(())
}
