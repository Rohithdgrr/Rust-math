//! High-DPI export example for publication-quality output.

use mathverse_plot::{DataPoint, DataSeries, DpiConfig, PlotConfig, SvgPlot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some data
    let points: Vec<DataPoint> = (0..=50)
        .map(|i| {
            let x = i as f64 * 0.1;
            DataPoint::new(x, x.sin() * (-x * 0.1).exp())
        })
        .collect();

    // Standard resolution (96 DPI)
    let config = PlotConfig::new()
        .with_title("Damped Sine Wave (Standard)")
        .with_x_label("x")
        .with_y_label("y");

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("Standard".to_string(), points.clone()));
    let svg_standard = plot.generate();
    std::fs::write("highdpi_standard.svg", &svg_standard)?;
    println!("Wrote highdpi_standard.svg ({} bytes)", svg_standard.len());

    // High resolution (300 DPI) for print
    let dpi_config = DpiConfig::print();
    let (scaled_w, scaled_h) = dpi_config.scale_dimensions(800, 600);

    let config = PlotConfig::new()
        .with_title("Damped Sine Wave (300 DPI)")
        .with_x_label("x")
        .with_y_label("y")
        .with_dimensions(scaled_w, scaled_h);

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("300 DPI".to_string(), points.clone()));

    // Generate SVG with proper viewBox for resolution independence
    let mut svg_high = plot.generate();

    // Add viewBox attribute for resolution independence
    svg_high = svg_high.replace(
        "<svg",
        &format!(
            "<svg viewBox=\"0 0 800 600\"",
        ),
    );

    std::fs::write("highdpi_300.svg", &svg_high)?;
    println!("Wrote highdpi_300.svg ({} bytes)", svg_high.len());

    // Show physical dimensions
    let meta = mathverse_plot::PngMetadata::new(300, 2400, 1800);
    let (w_in, h_in) = meta.physical_size_inches();
    let (w_mm, h_mm) = meta.physical_size_mm();
    println!("300 DPI image: {:.1}\" x {:.1}\" ({:.0}mm x {:.0}mm)", w_in, h_in, w_mm, h_mm);

    Ok(())
}
