//! Simple line plot with tick labels, rendered to SVG and terminal.

use mathverse_plot::{DataPoint, DataSeries, PlotConfig, SvgPlot, TerminalPlot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xs: Vec<f64> = (0..=100).map(|i| i as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
    let points: Vec<DataPoint> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| DataPoint::new(*x, *y))
        .collect();

    let config = PlotConfig::new()
        .with_title("Sine Wave")
        .with_x_label("x (radians)")
        .with_y_label("sin(x)");

    let mut svg = SvgPlot::new(config.clone());
    svg.add_series(DataSeries::new("sin(x)".into(), points.clone()));
    std::fs::write("sine.svg", svg.generate())?;
    println!("wrote sine.svg");

    let mut term = TerminalPlot::new(config).with_dimensions(80, 24);
    term.add_series(DataSeries::new("sin(x)".into(), points));
    print!("{}", term.generate());

    Ok(())
}
