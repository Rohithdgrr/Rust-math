//! Pair plot example.

use mathverse_plot::{render_pairplot, PairConfig};

fn main() -> mathverse_plot::PlotResult<()> {
    let n = 30;

    // Three correlated variables
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.3 + (i as f64 * 0.5).sin() * 0.5).collect();
    let y: Vec<f64> = x.iter().map(|&v| v * 1.2 + 1.0 + (v * 0.7).cos() * 0.4).collect();
    let z: Vec<f64> = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi + yi * 0.5 + (xi * 0.3).sin() * 0.3).collect();

    let data = vec![x, y, z];
    let labels = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];

    let mut config = PairConfig::new();
    config.plot_config = config.plot_config
        .with_title("Pair Plot: Variable Relationships");

    let svg = render_pairplot(&data, &labels, &config)?;
    std::fs::write("pairplot.svg", svg)?;
    println!("wrote pairplot.svg");

    Ok(())
}
