//! Simple 3D wireframe surface example.

use mathverse_plot::{SurfaceConfig, render_surface_wireframe};
use mathverse_plot::save::PlotSaver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate a test surface: z = sin(x) * cos(y)
    let x_range = (-3.0, 3.0);
    let y_range = (-3.0, 3.0);

    // Create surface config
    let config = SurfaceConfig::new()
        .with_resolution(40)
        .with_camera_distance(5.0)
        .with_rotation_y(0.5);

    // Render to SVG using a closure that computes z = f(x, y)
    let svg = render_surface_wireframe(
        |x, y| x.sin() * y.cos(),
        x_range,
        y_range,
        config,
    )?;
    PlotSaver::new(&svg).save_png("surface.png")?;
    println!("Wrote surface.svg ({} bytes)", svg.len());

    Ok(())
}
