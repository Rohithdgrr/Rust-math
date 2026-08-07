//! Simple line plot with tick labels, rendered to SVG and terminal.

use mathverse_plot::{DataPoint, DataSeries, PlotConfig, SvgPlot, TerminalPlot};
use mathverse_plot::save::PlotSaver;

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
    svg.add_series(DataSeries::new(String::from("sin(x)"), points.clone()));
    PlotSaver::new(&svg.generate()).save_png("sine.png")?;
    println!("wrote sine.png");

    let mut term = TerminalPlot::new(config).with_dimensions(80, 24);
    term.add_series(DataSeries::new(String::from("sin(x)"), points));
    print!("{}", term.generate());

    Ok(())
}
