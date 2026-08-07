//! Polar chart of a rose curve, rendered to SVG.
//!
//! The theta→Cartesian conversion ships from `mathverse-trigonometry`.
//! Run: `cargo run -p mathverse-plot --example simple_polar`

use mathverse_plot::polar::render_polar_svg;
use mathverse_plot::{PolarData, PolarPoint, PolarSeries};
use mathverse_plot::save::PlotSaver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 360;
    let points: Vec<PolarPoint> = (0..N)
        .map(|i| {
            let theta = i as f64 * std::f64::consts::TAU / N as f64;
            PolarPoint::new(theta, (3.0 * theta).sin().abs())
        })
        .collect();

    let mut data = PolarData::new().with_title("r = |sin(3θ)|");
    data.add_series(PolarSeries::new("rose", points));

    let svg = render_polar_svg(&data, 600, 600);
    PlotSaver::new(&svg).save_png("polar.png")?;
    println!("wrote polar.svg ({} bytes)", svg.len());
    Ok(())
}
