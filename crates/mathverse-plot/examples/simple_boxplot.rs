//! Box plot example: Tukey box plots for three sample groups.

use mathverse_plot::common::PlotConfig;
use mathverse_plot::style::Color;
use mathverse_plot::SvgPlot;

fn main() -> mathverse_plot::PlotResult<()> {
    let mut plot = SvgPlot::new(
        PlotConfig::new()
            .with_title("Box Plot Example")
            .with_x_label("group")
            .with_y_label("value"),
    );

    // Group A: narrow spread, one outlier
    let mut a: Vec<f64> = (0..50).map(|i| (i as f64 / 10.0).sin() * 2.0).collect();
    a.push(9.0);
    plot.add_box_plot("sin", &a, Color::BLUE)?;

    // Group B: uniform spread
    let b: Vec<f64> = (0..50).map(|i| (i % 7) as f64).collect();
    plot.add_box_plot("uniform", &b, Color::RED)?;

    // Group C: narrow data with one far outlier
    let mut c: Vec<f64> = (0..50)
        .map(|i| 5.0 + ((i % 3) as f64 - 1.0) * 0.5)
        .collect();
    c.push(11.0);
    plot.add_box_plot("narrow", &c, Color::GREEN)?;

    let svg = plot.generate();
    std::fs::write("boxplot.svg", svg)?;
    println!("wrote boxplot.svg");

    Ok(())
}
