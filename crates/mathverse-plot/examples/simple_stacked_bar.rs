//! Stacked bar chart example: Quarterly revenue by product line.

use mathverse_plot::stacked_bar::{render_stacked_bar, StackedBarConfig, StackedSeries};
use mathverse_plot::style::Color;
use mathverse_plot::save::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    let series = vec![
        StackedSeries::new("Cloud", vec![40.0, 55.0, 60.0, 75.0], Color::rgb(66, 133, 244)),
        StackedSeries::new("Enterprise", vec![30.0, 25.0, 35.0, 30.0], Color::rgb(0, 200, 83)),
        StackedSeries::new("Consumer", vec![20.0, 30.0, 25.0, 35.0], Color::rgb(255, 150, 50)),
    ];

    let config = StackedBarConfig::new(vec![
        "Q1".into(),
        "Q2".into(),
        "Q3".into(),
        "Q4".into(),
    ])
    .with_bar_width(50.0);

    let mut cfg = config;
    cfg.plot_config.title = "Quarterly Revenue ($M)".into();
    cfg.plot_config.width = 600;
    cfg.plot_config.height = 400;

    let svg = render_stacked_bar(&series, &cfg)?;
    PlotSaver::new(&svg).save_png("stacked_bar.png")?;
    println!("wrote stacked_bar.png");

    Ok(())
}
