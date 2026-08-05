//! Simple Argand diagram and domain coloring example.

use mathverse_complex::Complex;
use mathverse_plot::{
    ComplexPlaneConfig, ComplexPlaneMode, render_argand, render_domain_coloring,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Argand diagram with complex numbers on the unit circle
    let points: Vec<Complex> = (0..36)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 36.0;
            Complex::new(angle.cos(), angle.sin())
        })
        .collect();

    let config = ComplexPlaneConfig::new((-2.0, 2.0), (-2.0, 2.0))
        .with_mode(ComplexPlaneMode::Argand)
        .with_resolution(400);

    let svg = render_argand(&points, config)?;
    PlotSaver::new(svg).save_png("argand.png")?;
    println!("Wrote argand.svg ({} bytes)", svg.len());

    // Example 2: Domain coloring of f(z) = 1/z
    let domain_config = ComplexPlaneConfig::new((-2.0, 2.0), (-2.0, 2.0))
        .with_mode(ComplexPlaneMode::DomainColoring)
        .with_resolution(300);

    let f = |z: Complex| -> Complex {
        // f(z) = 1/z
        let norm = z.re * z.re + z.im * z.im;
        if norm < 1e-10 {
            Complex::new(0.0, 0.0)
        } else {
            Complex::new(z.re / norm, -z.im / norm)
        }
    };

    let svg = render_domain_coloring(f, domain_config)?;
    PlotSaver::new(svg).save_png("domain_coloring.png")?;
    println!("Wrote domain_coloring.svg ({} bytes)", svg.len());

    Ok(())
}
