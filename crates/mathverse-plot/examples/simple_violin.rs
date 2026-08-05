//! Violin plot example: Distribution comparison across groups.

use mathverse_plot::style::Color;
use mathverse_plot::violin::{render_violin_plot, ViolinConfig, ViolinData};

fn main() -> mathverse_plot::PlotResult<()> {
    let group_a: Vec<f64> = (0..200).map(|i| 3.0 + (i as f64 * 0.05).sin() * 1.0).collect();
    let group_b: Vec<f64> = (0..200).map(|i| 5.0 + (i as f64 * 0.03).cos() * 1.5).collect();
    let group_c: Vec<f64> = (0..200).map(|i| 4.0 + ((i as f64 * 0.07).sin() * 0.8)).collect();

    let data = vec![
        ViolinData::new("Group A", group_a, Color::rgb(66, 133, 244)),
        ViolinData::new("Group B", group_b, Color::rgb(255, 99, 71)),
        ViolinData::new("Group C", group_c, Color::rgb(0, 200, 83)),
    ];

    let config = ViolinConfig::new().with_width(80.0);

    let mut cfg = config;
    cfg.plot_config.title = "Distribution Comparison".into();
    cfg.plot_config.width = 600;
    cfg.plot_config.height = 400;

    let svg = render_violin_plot(&data, &cfg)?;
    PlotSaver::new(svg).save_png("violin.png")?;
    println!("wrote violin.png");

    Ok(())
}
