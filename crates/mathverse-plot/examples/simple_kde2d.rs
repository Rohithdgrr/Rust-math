//! 2D KDE plot example.

use mathverse_plot::{render_kde_plot, KdeConfig};
use mathverse_plot::save::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    // Generate two correlated clusters
    let mut x = Vec::new();
    let mut y = Vec::new();

    // Cluster 1: centered around (2, 3)
    for i in 0..40 {
        let xi = 2.0 + (i as f64 * 0.3).sin() * 0.8;
        let yi = 3.0 + (i as f64 * 0.5).cos() * 0.6;
        x.push(xi);
        y.push(yi);
    }

    // Cluster 2: centered around (6, 7)
    for i in 0..40 {
        let xi = 6.0 + (i as f64 * 0.4).sin() * 1.0;
        let yi = 7.0 + (i as f64 * 0.6).cos() * 0.8;
        x.push(xi);
        y.push(yi);
    }

    let mut config = KdeConfig::new()
        .with_bandwidth(0.8)
        .with_grid_size(40);

    config.plot_config = config.plot_config.with_title("2D KDE: Two Clusters");

    let svg = render_kde_plot(&x, &y, &config)?;
    PlotSaver::new(&svg).save_png("kde2d.png")?;
    println!("wrote kde2d.png");

    Ok(())
}
