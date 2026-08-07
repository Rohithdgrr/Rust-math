//! Strip plot example: categorical scatter with jitter.

use mathverse_plot::{render_strip_plot, StripCategory, StripConfig};
use mathverse_plot::style::Color;
use mathverse_plot::save::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    let categories = vec![
        StripCategory::new("Control", vec![2.1, 2.5, 2.8, 3.0, 2.3, 2.7, 2.9, 3.1, 2.4, 2.6], Color::BLUE),
        StripCategory::new("Treatment A", vec![3.5, 4.0, 3.8, 4.2, 3.9, 4.1, 3.7, 4.3], Color::RED),
        StripCategory::new("Treatment B", vec![5.0, 5.5, 4.8, 5.2, 5.8, 5.1, 5.4, 5.6, 4.9], Color::GREEN),
    ];

    let config = StripConfig::new()
        .with_jitter(0.4)
        .with_marker_size(5.0);

    let mut cfg = config;
    cfg.plot_config = cfg.plot_config.with_title("Strip Plot: Response by Group");

    let svg = render_strip_plot(&categories, &cfg)?;
    PlotSaver::new(&svg).save_png("strip.png")?;
    println!("wrote strip.png");

    Ok(())
}
