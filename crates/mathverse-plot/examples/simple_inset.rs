//! Inset axes (zoom panel) example.

use mathverse_plot::{
    DataPoint, DataSeries, InsetAxes, InsetConfig, PlotConfig, SvgPlot,
};
use mathverse_plot::style::Color;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data with interesting detail
    let points: Vec<DataPoint> = (0..=200)
        .map(|i| {
            let x = i as f64 * 0.05;
            let y = x.sin() * (x * 0.3).exp() + 0.1 * (x * 10.0).sin();
            DataPoint::new(x, y)
        })
        .collect();

    let config = PlotConfig::new()
        .with_title("Damped Sine with High-Frequency Noise")
        .with_x_label("x")
        .with_y_label("y");

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("Signal".to_string(), points));

    // Generate the main plot SVG
    let mut svg = plot.generate();

    // Create an inset showing detail of the first peak
    let inset_config = InsetConfig::new((1.0, 3.0), (0.0, 1.0))
        .with_position(550.0, 50.0)
        .with_dimensions(180.0, 120.0)
        .with_background(Color::WHITE)
        .with_border(Color::BLACK)
        .with_connector(true);

    let mut inset = InsetAxes::new(inset_config);

    // Add the same data to the inset (it will clip to the zoom region)
    let zoomed_points: Vec<DataPoint> = (0..=200)
        .filter_map(|i| {
            let x = i as f64 * 0.05;
            if x >= 1.0 && x <= 3.0 {
                let y = x.sin() * (x * 0.3).exp() + 0.1 * (x * 10.0).sin();
                Some(DataPoint::new(x, y))
            } else {
                None
            }
        })
        .collect();

    inset.add_series(DataSeries::new("Zoomed".to_string(), zoomed_points));

    // Render the inset
    let inset_svg = inset.render(50.0, 50.0);

    // Add inset to the main SVG
    svg = svg.replace("</svg>", &format!("{}\n</svg>", inset_svg));

    std::fs::write("inset.svg", &svg)?;
    println!("Wrote inset.svg ({} bytes)", svg.len());

    Ok(())
}
