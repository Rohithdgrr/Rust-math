//! Pie chart example: Market share breakdown.

use mathverse_plot::pie::{render_pie_chart, PieConfig, PieSlice};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let slices = vec![
        PieSlice::new("Chrome", 65.0, Color::rgb(66, 133, 244)),
        PieSlice::new("Safari", 18.0, Color::rgb(0, 200, 83)),
        PieSlice::new("Firefox", 8.0, Color::rgb(255, 150, 50)),
        PieSlice::new("Edge", 5.0, Color::rgb(0, 120, 212)),
        PieSlice::new("Other", 4.0, Color::rgb(150, 150, 150)),
    ];

    let config = PieConfig::new()
        .with_title("Browser Market Share")
        .with_dimensions(600, 400)
        .with_radius(140.0)
        .with_center(300.0, 210.0)
        .with_percentages();

    let svg = render_pie_chart(&slices, &config)?;
    std::fs::write("pie.svg", svg)?;
    println!("wrote pie.svg");

    Ok(())
}
