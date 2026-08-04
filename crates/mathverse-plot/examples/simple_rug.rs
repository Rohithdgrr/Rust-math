//! Rug plot example: Data point markers along an axis.

use mathverse_plot::rug::{render_rug_plot, RugConfig};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let values = vec![
        0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8, 0.4, 0.6, 0.35, 0.65, 0.15, 0.85, 0.45, 0.55,
    ];

    let config = RugConfig::new()
        .with_side("bottom")
        .with_height(20.0);

    let mut cfg = config;
    cfg.plot_config.title = "Rug Plot".into();
    cfg.plot_config.width = 600;
    cfg.plot_config.height = 200;
    cfg.plot_config.padding = 30.0;

    // Rug plot doesn't use Color directly in config, it's in RugConfig.color
    let mut cfg2 = cfg;
    cfg2.color = Color::rgb(66, 133, 244);

    let svg = render_rug_plot(&values, &cfg2)?;
    std::fs::write("rug.svg", svg)?;
    println!("wrote rug.svg");

    Ok(())
}
