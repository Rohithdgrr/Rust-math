//! Pareto chart example: Defect categories with cumulative line.

use mathverse_plot::pareto::{render_pareto, ParetoBar, ParetoConfig};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let bars = vec![
        ParetoBar::new("Scratches", 45.0),
        ParetoBar::new("Dents", 30.0),
        ParetoBar::new("Discolor", 15.0),
        ParetoBar::new("Cracks", 10.0),
        ParetoBar::new("Other", 5.0),
    ];

    let config = ParetoConfig::new();

    let mut cfg = config;
    cfg.plot_config.title = "Defect Pareto Analysis".into();
    cfg.plot_config.width = 650;
    cfg.plot_config.height = 400;
    cfg.bar_color = Color::rgb(66, 133, 244);
    cfg.line_color = Color::rgb(220, 20, 60);

    let svg = render_pareto(&bars, &cfg)?;
    std::fs::write("pareto.svg", svg)?;
    println!("wrote pareto.svg");

    Ok(())
}
