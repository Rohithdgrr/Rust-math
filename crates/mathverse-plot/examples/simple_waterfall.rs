//! Waterfall chart example: Budget breakdown.

use mathverse_plot::style::Color;
use mathverse_plot::waterfall::{render_waterfall, WaterfallBar, WaterfallConfig};

fn main() -> mathverse_plot::PlotResult<()> {
    let bars = vec![
        WaterfallBar::new("Start", 1000.0).as_total(),
        WaterfallBar::new("Sales", 350.0),
        WaterfallBar::new("Services", 200.0),
        WaterfallBar::new("Rent", -150.0),
        WaterfallBar::new("Payroll", -400.0),
        WaterfallBar::new("Other", -50.0),
        WaterfallBar::new("Net", 950.0).as_total(),
    ];

    let config = WaterfallConfig::new().with_colors(
        Color::rgb(0, 200, 83),
        Color::rgb(220, 20, 60),
        Color::rgb(66, 133, 244),
    );

    let mut cfg = config;
    cfg.plot_config.title = "Budget Waterfall".into();
    cfg.plot_config.width = 700;
    cfg.plot_config.height = 400;

    let svg = render_waterfall(&bars, &cfg)?;
    std::fs::write("waterfall.svg", svg)?;
    println!("wrote waterfall.svg");

    Ok(())
}
