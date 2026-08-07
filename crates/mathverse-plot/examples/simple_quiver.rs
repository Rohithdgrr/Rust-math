//! Quiver plot example: Simple vector field.

use mathverse_plot::quiver::{render_quiver, QuiverConfig, QuiverVector};
use mathverse_plot::style::Color;
use mathverse_plot::save::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    let mut vectors = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            let x = i as f64;
            let y = j as f64;
            let u = -y;
            let v = x;
            vectors.push(QuiverVector::new(x, y, u, v));
        }
    }

    let config = QuiverConfig::new()
        .with_scale(20.0)
        .with_color(Color::rgb(30, 100, 200));

    let mut cfg = config;
    cfg.plot_config.title = "Rotation Field".into();
    cfg.plot_config.width = 500;
    cfg.plot_config.height = 500;

    let svg = render_quiver(&vectors, &cfg)?;
    PlotSaver::new(&svg).save_png("quiver.png")?;
    println!("wrote quiver.png");

    Ok(())
}
