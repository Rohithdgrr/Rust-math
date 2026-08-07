//! Joint plot example.

use mathverse_plot::{render_jointplot, JointConfig};
use mathverse_plot::save::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    // Generate correlated data
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..60 {
        let xi = i as f64 * 0.15;
        let yi = xi * 0.8 + (i as f64 * 0.3).sin() * 1.5;
        x.push(xi);
        y.push(yi);
    }

    let mut config = JointConfig::new().with_bins(15);
    config.plot_config = config.plot_config
        .with_title("Joint Plot: x vs y");

    let svg = render_jointplot(&x, &y, &config)?;
    PlotSaver::new(&svg).save_png("jointplot.png")?;
    println!("wrote jointplot.png");

    Ok(())
}
