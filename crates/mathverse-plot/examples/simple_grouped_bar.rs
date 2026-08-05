//! Grouped bar chart example: Sales by region across years.

use mathverse_plot::grouped_bar::{render_grouped_bar, GroupedBarConfig, GroupedSeries};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let series = vec![
        GroupedSeries::new("2023", vec![45.0, 62.0, 38.0, 55.0], Color::rgb(66, 133, 244)),
        GroupedSeries::new("2024", vec![52.0, 58.0, 45.0, 60.0], Color::rgb(0, 200, 83)),
    ];

    let config = GroupedBarConfig::new(vec![
        "North".into(),
        "South".into(),
        "East".into(),
        "West".into(),
    ])
    .with_bar_width(25.0);

    let mut cfg = config;
    cfg.plot_config.title = "Regional Sales ($M)".into();
    cfg.plot_config.width = 600;
    cfg.plot_config.height = 400;

    let svg = render_grouped_bar(&series, &cfg)?;
PlotSaver::new(&svg).save_png("grouped_bar.png")?;
println!("wrote grouped_bar.png");

    Ok(())
}
