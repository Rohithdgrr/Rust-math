//! Interactive egui/eframe window with a live line plot.
//!
//! Pan to move the view, scroll to zoom around the pointer. The plot uses the
//! same `PlotData` snapshot abstraction as the SVG/PNG backends.
//!
//! Run: `cargo run -p mathverse-plot --features interactive --example interactive`

use mathverse_plot::common::{DataPoint, DataSeries, PlotConfig};
use mathverse_plot::interactive::run;
use mathverse_plot::PlotData;

fn main() -> eframe::Result<()> {
    let xs: Vec<f64> = (0..1000).map(|i| f64::from(i) * 0.01).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|x| x.sin() * x.cos() * (0.5 * x).sin())
        .collect();
    let points: Vec<DataPoint> = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| DataPoint::new(*x, *y))
        .collect();

    let config = PlotConfig::new()
        .with_title("interactive sin(x)cos(x)sin(x/2)")
        .with_x_label("x")
        .with_y_label("f(x)");

    let mut svg = mathverse_plot::SvgPlot::new(config.clone());
    svg.add_series(DataSeries::new("f", points.clone()));

    // Stash an SVG for comparison with the interactive scene.
    PlotSaver::new(&svg.generate()).save_png("interactive_scene.png").ok();

    let data: PlotData = svg.snapshot();
    run(data, "mathverse-plot interactive", 900.0, 600.0)
}
