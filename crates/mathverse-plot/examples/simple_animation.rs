//! Simple animation example generating SVG frames.

use mathverse_plot::{
    AnimationConfig, DataPoint, DataSeries, PlotConfig,
    generate_frames, render_frame, assemble_animated_svg,
};
use mathverse_plot::save::PlotSaver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate frames for a moving sine wave
    let config = AnimationConfig::new(20)
        .with_frame_duration(100)
        .with_dimensions(800, 400)
        .with_plot_config(
            PlotConfig::new()
                .with_title("Animated Sine Wave")
                .with_x_label("x")
                .with_y_label("sin(x + phase)")
        );

    let frames = generate_frames(
        |i, total| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / total as f64;
            let points: Vec<DataPoint> = (0..=100)
                .map(|j| {
                    let x = j as f64 * 0.1;
                    let y = (x + phase).sin();
                    DataPoint::new(x, y)
                })
                .collect();

            let series = DataSeries::new(format!("frame_{}", i), points);
            render_frame(PlotConfig::new(), series)
        },
        config.clone(),
    )?;

    println!("Generated {} frames", frames.len());

    // Assemble into animated SVG
    let animated_svg = assemble_animated_svg(&frames, &config);
    PlotSaver::new(&animated_svg).save_png("animation.png")?;
    println!("Wrote animation.svg ({} bytes)", animated_svg.len());

    Ok(())
}
