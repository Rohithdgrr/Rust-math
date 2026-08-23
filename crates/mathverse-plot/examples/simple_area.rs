//! Area chart example: Temperature range over a day.

use mathverse_plot::area::{render_area_chart, AreaConfig, AreaSeries};
use mathverse_plot::common::DataPoint;
use mathverse_plot::save::PlotSaver;
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let morning: Vec<DataPoint> = (0..=12)
        .map(|h| DataPoint::new(f64::from(h), 15.0 + f64::from(h) * 1.5))
        .collect();
    let afternoon: Vec<DataPoint> = (12..=24)
        .map(|h| DataPoint::new(f64::from(h), 33.0 - (f64::from(h) - 12.0) * 1.2))
        .collect();

    let series = vec![
        AreaSeries::new("Morning", morning, Color::rgb(255, 165, 0)),
        AreaSeries::new("Afternoon", afternoon, Color::rgb(66, 133, 244)),
    ];

    let config = AreaConfig::new().with_opacity(0.3).with_baseline(0.0);

    let mut cfg = config;
    cfg.plot_config.title = "Temperature Over 24h".into();
    cfg.plot_config.width = 700;
    cfg.plot_config.height = 400;

    let svg = render_area_chart(&series, &cfg)?;
    PlotSaver::new(&svg).save_png("area.png")?;
    println!("wrote area.png");

    Ok(())
}
