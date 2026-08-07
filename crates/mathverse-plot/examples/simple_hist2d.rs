//! 2D histogram example: Correlated random data.

use mathverse_plot::hist2d::{render_hist2d, Hist2DConfig};
use mathverse_plot::save::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    // Generate correlated x,y pairs using a simple LCG
    let n = 500;
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);
    let mut state: u64 = 42;
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
        // Box-Muller transform
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
        x_data.push(z0);
        y_data.push(0.7 * z0 + 0.3 * z1); // correlated
    }

    let config = Hist2DConfig::new().with_bins(25, 25);

    let mut cfg = config;
    cfg.plot_config.title = "2D Histogram".into();
    cfg.plot_config.width = 500;
    cfg.plot_config.height = 500;

    let svg = render_hist2d(&x_data, &y_data, &cfg)?;
    PlotSaver::new(&svg).save_png("hist2d.png")?;
    println!("wrote hist2d.png");

    Ok(())
}
