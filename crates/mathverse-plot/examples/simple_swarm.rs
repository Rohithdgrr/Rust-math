//! Swarm plot example: non-overlapping categorical scatter.

use mathverse_plot::{render_swarm_plot, SwarmCategory, SwarmConfig};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let categories = vec![
        SwarmCategory::new("Low", vec![1.0, 1.2, 1.5, 1.3, 1.1, 1.4, 1.6, 1.2, 1.3, 1.5], Color::BLUE),
        SwarmCategory::new("Medium", vec![3.0, 3.5, 3.2, 3.8, 3.1, 3.6, 3.4, 3.7, 3.3, 3.9], Color::ORANGE),
        SwarmCategory::new("High", vec![5.5, 6.0, 5.8, 6.2, 5.7, 6.1, 5.9, 6.3, 5.6, 6.0], Color::GREEN),
    ];

    let config = SwarmConfig::new()
        .with_point_size(5.0);

    let mut cfg = config;
    cfg.plot_config = cfg.plot_config.with_title("Swarm Plot: Score Distribution");

    let svg = render_swarm_plot(&categories, &cfg)?;
    std::fs::write("swarm.svg", svg)?;
    println!("wrote swarm.svg");

    Ok(())
}
