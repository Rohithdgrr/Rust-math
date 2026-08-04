//! Point plot example.

use mathverse_plot::{render_pointplot, PointCategory, PointConfig};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let categories = vec![
        PointCategory::new("Baseline", vec![3.0, 3.2, 2.9, 3.1, 3.3, 2.8, 3.0], Color::BLUE),
        PointCategory::new("Week 1", vec![4.5, 4.8, 4.2, 4.6, 4.9, 4.3, 4.7], Color::ORANGE),
        PointCategory::new("Week 2", vec![5.8, 6.1, 5.5, 5.9, 6.2, 5.7, 6.0], Color::GREEN),
        PointCategory::new("Week 3", vec![6.5, 6.8, 6.2, 6.6, 6.9, 6.3, 6.7], Color::RED),
    ];

    let mut config = PointConfig::new().with_point_size(7.0);
    config.plot_config = config.plot_config
        .with_title("Point Plot: Mean Response Over Time");

    let svg = render_pointplot(&categories, &config)?;
    std::fs::write("pointplot.svg", svg)?;
    println!("wrote pointplot.svg");

    Ok(())
}
