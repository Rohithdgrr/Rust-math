//! Heatmap example: `sin(x) * cos(y)` rendered with Viridis colormap.

use mathverse_plot::color::viridis;
use mathverse_plot::{PlotConfig, SvgPlot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = 20;
    let cols = 30;
    let grid: Vec<Vec<f64>> = (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let x = c as f64 / cols as f64 * 4.0 * std::f64::consts::PI;
                    let y = r as f64 / rows as f64 * 2.0 * std::f64::consts::PI;
                    x.sin() * y.cos()
                })
                .collect()
        })
        .collect();

    let mut plot = SvgPlot::new(
        PlotConfig::new()
            .with_title("Heatmap: sin(x) * cos(y)")
            .with_x_label("x index")
            .with_y_label("y index"),
    );
    plot.add_heatmap("sin*cos", grid, viridis)?;

    std::fs::write("heatmap.svg", plot.generate())?;
    println!("wrote heatmap.svg");
    Ok(())
}
