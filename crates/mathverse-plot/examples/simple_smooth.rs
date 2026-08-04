//! Smooth Bezier curves example.

use mathverse_plot::{
    DataPoint, DataSeries, Interpolation, PlotConfig, SmoothConfig, SvgPlot,
};
use mathverse_plot::style::PlotStyle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some data points
    let points = vec![
        DataPoint::new(0.0, 0.0),
        DataPoint::new(1.0, 0.8),
        DataPoint::new(2.0, 0.3),
        DataPoint::new(3.0, 0.9),
        DataPoint::new(4.0, 0.2),
        DataPoint::new(5.0, 0.7),
        DataPoint::new(6.0, 0.4),
    ];

    let config = PlotConfig::new()
        .with_title("Smooth Curves Comparison")
        .with_x_label("x")
        .with_y_label("y");

    let mut plot = SvgPlot::new(config);

    // Add original data as scatter
    let scatter_style = PlotStyle::default()
        .with_line_color(mathverse_plot::style::Color::GRAY)
        .with_marker_style(mathverse_plot::style::MarkerStyle::Circle)
        .with_marker_size(4.0);
    plot.add_series(DataSeries::with_style(
        "Original Points".to_string(),
        points.clone(),
        scatter_style,
    ));

    // Add Catmull-Rom smoothed curve
    let catmull_config = SmoothConfig::new()
        .with_interpolation(Interpolation::CatmullRom)
        .with_tension(0.5)
        .with_subdivisions(20);

    let smoothed_points = mathverse_plot::smooth_points(&points, &catmull_config);
    let catmull_style = PlotStyle::default().with_line_color(mathverse_plot::style::Color::BLUE);
    plot.add_series(DataSeries::with_style(
        "Catmull-Rom".to_string(),
        smoothed_points,
        catmull_style,
    ));

    // Add cubic Bezier smoothed curve
    let bezier_config = SmoothConfig::new()
        .with_interpolation(Interpolation::CubicBezier)
        .with_subdivisions(20);

    let bezier_points = mathverse_plot::smooth_points(&points, &bezier_config);
    let bezier_style = PlotStyle::default().with_line_color(mathverse_plot::style::Color::RED);
    plot.add_series(DataSeries::with_style(
        "Cubic Bezier".to_string(),
        bezier_points,
        bezier_style,
    ));

    // Generate SVG
    let svg = plot.generate();
    std::fs::write("smooth.svg", &svg)?;
    println!("Wrote smooth.svg ({} bytes)", svg.len());

    Ok(())
}
