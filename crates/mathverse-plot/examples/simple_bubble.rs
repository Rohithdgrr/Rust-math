//! Bubble chart example: Countries by GDP, population, and area.

use mathverse_plot::bubble::{render_bubble_chart, Bubble, BubbleConfig};
use mathverse_plot::style::Color;
use mathverse_plot::save::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    let bubbles = vec![
        Bubble::new(21400.0, 331.0, 9834.0, Color::rgb(66, 133, 244)).with_label("USA"),
        Bubble::new(14700.0, 1412.0, 9597.0, Color::rgb(220, 20, 60)).with_label("China"),
        Bubble::new(2900.0, 1408.0, 3287.0, Color::rgb(255, 150, 50)).with_label("India"),
        Bubble::new(1700.0, 146.0, 378.0, Color::rgb(0, 200, 83)).with_label("Germany"),
        Bubble::new(2900.0, 67.0, 640.0, Color::rgb(148, 0, 211)).with_label("UK"),
    ];

    let config = BubbleConfig::new().with_size_scale(0.01);

    let mut cfg = config;
    cfg.plot_config.title = "Countries: GDP vs Population".into();
    cfg.plot_config.width = 700;
    cfg.plot_config.height = 500;

    let svg = render_bubble_chart(&bubbles, &cfg)?;
    PlotSaver::new(&svg).save_png("bubble.png")?;
    println!("wrote bubble.png");

    Ok(())
}
