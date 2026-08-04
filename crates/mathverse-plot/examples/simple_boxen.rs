//! Boxen (letter-value) plot example.

use mathverse_plot::{render_boxen_plot, BoxenData, BoxenConfig};
use mathverse_plot::style::Color;

fn main() -> mathverse_plot::PlotResult<()> {
    let data = vec![
        BoxenData::new(
            "Group A",
            vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
            Color::BLUE,
        ),
        BoxenData::new(
            "Group B",
            vec![2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
            Color::RED,
        ),
        BoxenData::new(
            "Group C",
            vec![3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0],
            Color::GREEN,
        ),
    ];

    let mut config = BoxenConfig::new().with_levels(6);
    config.plot_config = config.plot_config.with_title("Boxen Plot: Value Spread");

    let svg = render_boxen_plot(&data, &config)?;
    std::fs::write("boxen.svg", svg)?;
    println!("wrote boxen.svg");

    Ok(())
}
