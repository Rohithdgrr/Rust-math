//! Horizontal bar chart example: City populations.

use mathverse_plot::hbar::{render_hbar_chart, HBar, HBarConfig};
use mathverse_plot::style::Color;
use mathverse_plot::save::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    let bars = vec![
        HBar::new("Tokyo", 13960000.0, Color::rgb(255, 99, 71)),
        HBar::new("Delhi", 11030000.0, Color::rgb(30, 144, 255)),
        HBar::new("Shanghai", 24870000.0, Color::rgb(50, 205, 50)),
        HBar::new("São Paulo", 12330000.0, Color::rgb(255, 165, 0)),
        HBar::new("Mumbai", 12440000.0, Color::rgb(148, 0, 211)),
        HBar::new("Cairo", 9540000.0, Color::rgb(220, 20, 60)),
    ];

    let config = HBarConfig::new()
        .with_bar_height(35.0)
        .with_values(true);

    let mut cfg = config;
    cfg.plot_config.title = "World City Populations".into();
    cfg.plot_config.width = 600;
    cfg.plot_config.height = 350;

    let svg = render_hbar_chart(&bars, &cfg)?;
PlotSaver::new(&svg).save_png("hbar.png")?;
println!("wrote hbar.png");

    Ok(())
}
