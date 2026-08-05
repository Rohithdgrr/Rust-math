//! Contour plot example: 2D Gaussian peak.

use mathverse_plot::contour::{render_contour, ContourConfig};
use mathverse_plot::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    let n = 30;
    let grid: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            (0..n)
                .map(|i| {
                    let x = i as f64 / (n - 1) as f64 * 4.0 - 2.0;
                    let y = j as f64 / (n - 1) as f64 * 4.0 - 2.0;
                    (-x * x - y * y).exp()
                })
                .collect()
        })
        .collect();

    let config = ContourConfig::new().with_levels(8);

    let mut cfg = config;
    cfg.plot_config.title = "2D Gaussian".into();
    cfg.plot_config.width = 500;
    cfg.plot_config.height = 500;

    let svg = render_contour(&grid, (-2.0, 2.0), (-2.0, 2.0), &cfg)?;
PlotSaver::new(&svg).save_png("contour.png")?;
println!("wrote contour.png");

    Ok(())
}
