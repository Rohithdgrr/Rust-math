//! Step plot example: Piecewise constant signal.

use mathverse_plot::common::DataPoint;
use mathverse_plot::step::{render_step_plot, StepConfig, StepPosition};
use mathverse_plot::style::Color;
use mathverse_plot::save::PlotSaver;

fn main() -> mathverse_plot::PlotResult<()> {
    let points = vec![
        DataPoint::new(0.0, 2.0),
        DataPoint::new(1.0, 5.0),
        DataPoint::new(2.0, 3.0),
        DataPoint::new(3.0, 7.0),
        DataPoint::new(4.0, 4.0),
        DataPoint::new(5.0, 6.0),
        DataPoint::new(6.0, 2.0),
    ];

    let config = StepConfig::new()
        .with_position(StepPosition::Before)
        .with_color(Color::rgb(0, 150, 136));

    let mut cfg = config;
    cfg.plot_config.title = "Step Plot".into();
    cfg.plot_config.width = 600;
    cfg.plot_config.height = 350;

    let svg = render_step_plot(&points, &cfg)?;
    PlotSaver::new(&svg).save_png("step.png")?;
    println!("wrote step.png");

    Ok(())
}
