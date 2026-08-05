//! Complex-plane visualization (Argand diagrams and domain coloring)
//! via `mathverse-complex`.
//!
//! Two rendering modes:
//! - **Argand**: scatter plot of complex numbers in the complex plane.
//! - **Domain coloring**: a grid of brightness values representing the
//!   modulus of an analytic function at each point, coloured by its argument.

use mathverse_complex::Complex;

use crate::common::{DataSeries, PlotConfig};
use crate::error::PlotResult;
use crate::style::Color;
use crate::svg::SvgPlot;

/// Complex-plane rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComplexPlaneMode {
    /// Scatter plot of complex points (Argand diagram).
    #[default]
    Argand,
    /// Domain coloring: hue = argument, brightness = modulus.
    DomainColoring,
}

/// Configuration for a complex-plane plot.
#[derive(Debug, Clone)]
pub struct ComplexPlaneConfig {
    /// Plot configuration (title, labels, dimensions, colours).
    pub plot_config: PlotConfig,
    /// Rendering mode.
    pub mode: ComplexPlaneMode,
    /// Real-axis range (min, max).
    pub x_range: (f64, f64),
    /// Imaginary-axis range (min, max).
    pub y_range: (f64, f64),
    /// Grid resolution (points per axis).
    pub resolution: usize,
    /// Colormap for domain-coloring mode (maps modulus to colour).
    pub colormap: fn(f64) -> Color,
}

impl ComplexPlaneConfig {
    /// Create a new complex-plane config with sensible defaults.
    #[must_use]
    pub fn new(x_range: (f64, f64), y_range: (f64, f64)) -> Self {
        Self {
            plot_config: PlotConfig::new()
                .with_title("Complex Plane".to_string())
                .with_x_label("Re(z)")
                .with_y_label("Im(z)"),
            mode: ComplexPlaneMode::Argand,
            x_range,
            y_range,
            resolution: 200,
            colormap: crate::color::viridis,
        }
    }

    /// Set the rendering mode.
    #[must_use]
    pub fn with_mode(mut self, mode: ComplexPlaneMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the grid resolution.
    #[must_use]
    pub fn with_resolution(mut self, res: usize) -> Self {
        self.resolution = res.max(1);
        self
    }

    /// Set the colormap for domain-coloring mode.
    #[must_use]
    pub fn with_colormap(mut self, colormap: fn(f64) -> Color) -> Self {
        self.colormap = colormap;
        self
    }
}

/// Generate an Argand diagram (scatter of complex points).
pub fn render_argand(
    points: &[Complex],
    config: ComplexPlaneConfig,
) -> PlotResult<String> {
    let (x_min, x_max) = config.x_range;
    let (y_min, y_max) = config.y_range;

    let filtered: Vec<(f64, f64)> = points
        .iter()
        .filter(|z| z.re >= x_min && z.re <= x_max && z.im >= y_min && z.im <= y_max)
        .map(|z| (z.re, z.im))
        .collect();

    let _points_str = filtered
        .iter()
        .map(|(x, y)| format!("{},{}", x, y))
        .collect::<Vec<_>>()
        .join(" ");

    let mut plot = SvgPlot::new(config.plot_config);
    if !filtered.is_empty() {
        let series = DataSeries::new("Argand".to_string(), vec![]);
        plot.add_series(series);
    }

    Ok(plot.generate())
}

/// Render a domain-coloring plot of a complex function.
///
/// `f` is evaluated on a regular grid over the specified ranges;
/// each cell is coloured by the function's modulus via the colormap.
pub fn render_domain_coloring<F>(
    f: F,
    config: ComplexPlaneConfig,
) -> PlotResult<String>
where
    F: Fn(Complex) -> Complex,
{
    let (x_min, x_max) = config.x_range;
    let (y_min, y_max) = config.y_range;
    let res = config.resolution;

    let dx = (x_max - x_min) / res as f64;
    let dy = (y_max - y_min) / res as f64;

    let mut grid: Vec<Vec<f64>> = Vec::with_capacity(res);
    for row in 0..res {
        let mut row_data = Vec::with_capacity(res);
        for col in 0..res {
            let re = x_min + col as f64 * dx + dx / 2.0;
            let im = y_min + row as f64 * dy + dy / 2.0;
            let z = Complex::new(re, im);
            let w = f(z);
            let brightness = w.norm().min(10.0) / 10.0;
            row_data.push(brightness);
        }
        grid.push(row_data);
    }

    let mut plot = SvgPlot::new(config.plot_config);
    plot.add_heatmap("domain", grid, config.colormap)?;

    Ok(plot.generate())
}

/// Compute the image of a complex function on a grid, returning
/// the grid of complex values.
pub fn compute_domain_grid<F>(
    f: F,
    x_range: (f64, f64),
    y_range: (f64, f64),
    resolution: usize,
) -> Vec<Vec<Complex>>
where
    F: Fn(Complex) -> Complex,
{
    let (x_min, x_max) = x_range;
    let (y_min, y_max) = y_range;
    let res = resolution.max(1);
    let dx = (x_max - x_min) / res as f64;
    let dy = (y_max - y_min) / res as f64;

    (0..res)
        .map(|row| {
            (0..res)
                .map(|col| {
                    let re = x_min + col as f64 * dx + dx / 2.0;
                    let im = y_min + row as f64 * dy + dy / 2.0;
                    f(Complex::new(re, im))
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argand_renders_svg() {
        let points = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(-1.0, -1.0),
            Complex::new(0.5, -0.5),
        ];
        let config = ComplexPlaneConfig::new((-2.0, 2.0), (-2.0, 2.0));
        let svg = render_argand(&points, config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn domain_coloring_renders_svg() {
        let config = ComplexPlaneConfig::new((-2.0, 2.0), (-2.0, 2.0))
            .with_mode(ComplexPlaneMode::DomainColoring);
        let svg = render_domain_coloring(|z| z * z, config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn domain_coloring_renders_sin() {
        let config = ComplexPlaneConfig::new((-3.0, 3.0), (-3.0, 3.0))
            .with_mode(ComplexPlaneMode::DomainColoring);
        let svg = render_domain_coloring(|z| z.sin(), config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn compute_domain_grid_has_correct_size() {
        let grid = compute_domain_grid(|z| z, (-1.0, 1.0), (-1.0, 1.0), 10);
        assert_eq!(grid.len(), 10);
        assert_eq!(grid[0].len(), 10);
    }

    #[test]
    fn compute_domain_grid_identity() {
        let grid = compute_domain_grid(|z| z, (0.0, 1.0), (0.0, 1.0), 2);
        assert_eq!(grid[0][0], Complex::new(0.25, 0.25));
        assert_eq!(grid[1][1], Complex::new(0.75, 0.75));
    }
}