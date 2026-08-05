//! 2D KDE (Kernel Density Estimation) plot.

use std::f64::consts::PI;

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};

/// Configuration for a 2D KDE plot.
#[derive(Debug, Clone)]
pub struct KdeConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Number of grid points per axis.
    pub grid_size: usize,
    /// Bandwidth for X axis.
    pub bw_x: f64,
    /// Bandwidth for Y axis.
    pub bw_y: f64,
    /// Number of contour levels.
    pub levels: usize,
    /// Show filled contours.
    pub filled: bool,
    /// Show contour lines.
    pub show_lines: bool,
    /// Show colorbar.
    pub show_colorbar: bool,
}

impl Default for KdeConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            grid_size: 50,
            bw_x: 0.5,
            bw_y: 0.5,
            levels: 10,
            filled: true,
            show_lines: true,
            show_colorbar: true,
        }
    }
}

impl KdeConfig {
    /// Create a new KDE config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set bandwidth.
    pub fn with_bandwidth(mut self, bw: f64) -> Self {
        self.bw_x = bw;
        self.bw_y = bw;
        self
    }

    /// Set grid size.
    pub fn with_grid_size(mut self, size: usize) -> Self {
        self.grid_size = size;
        self
    }
}

/// Gaussian kernel.
fn gaussian_kernel_2d(x: f64, y: f64) -> f64 {
    (-0.5 * (x * x + y * y)).exp() / (2.0 * PI)
}

/// Compute 2D KDE grid.
fn compute_kde_grid(
    x_data: &[f64],
    y_data: &[f64],
    bw_x: f64,
    bw_y: f64,
    grid_size: usize,
) -> Vec<Vec<f64>> {
    let n = x_data.len() as f64;

    let x_min = x_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let x_margin = (x_max - x_min) * 0.2;
    let y_margin = (y_max - y_min) * 0.2;

    let mut grid = vec![vec![0.0; grid_size]; grid_size];

    for j in 0..grid_size {
        for i in 0..grid_size {
            let gx = (x_min - x_margin) + (x_max - x_min + 2.0 * x_margin) * i as f64 / (grid_size - 1) as f64;
            let gy = (y_min - y_margin) + (y_max - y_min + 2.0 * y_margin) * j as f64 / (grid_size - 1) as f64;

            let mut density = 0.0;
            for (&dx, &dy) in x_data.iter().zip(y_data.iter()) {
                let ux = (gx - dx) / bw_x;
                let uy = (gy - dy) / bw_y;
                density += gaussian_kernel_2d(ux, uy);
            }
            density /= n * bw_x * bw_y;
            grid[j][i] = density;
        }
    }

    grid
}

/// Color for density value.
fn kde_color(t: f64) -> String {
    // Viridis-like colormap
    let r = ((0.267 + t * 0.733) * 255.0) as u8;
    let g = if t < 0.5 {
        ((0.0 + t * 2.0 * 0.7) * 255.0) as u8
    } else {
        ((0.7 - (t - 0.5) * 2.0 * 0.3) * 255.0) as u8
    };
    let b = ((0.329 - t * 0.329) * 255.0) as u8;
    format!("#{:02x}{:02x}{:02x}", r, g, b)
}

/// Render a 2D KDE plot as SVG.
pub fn render_kde_plot(
    x_data: &[f64],
    y_data: &[f64],
    config: &KdeConfig,
) -> PlotResult<String> {
    if x_data.len() != y_data.len() {
        return Err(PlotError::InvalidData("x and y must have same length".into()));
    }
    if x_data.is_empty() {
        return Err(PlotError::InvalidData("no data provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    let x_min = x_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let chart_width = width - padding * 2.0 - if config.show_colorbar { 60.0 } else { 0.0 };
    let chart_height = height - padding * 2.0 - 30.0;

    // Compute KDE grid
    let grid = compute_kde_grid(x_data, y_data, config.bw_x, config.bw_y, config.grid_size);

    // Find max density
    let max_density = grid.iter().flat_map(|r| r.iter()).fold(0.0_f64, |a, &b| a.max(b));
    if max_density == 0.0 {
        return Err(PlotError::InvalidData("zero density".into()));
    }

    let cell_w = chart_width / config.grid_size as f64;
    let cell_h = chart_height / config.grid_size as f64;

    let mut svg = String::new();
    svg.push_str("<svg width=\"");
    svg.push_str(&width.to_string());
    svg.push_str("\" height=\"");
    svg.push_str(&height.to_string());
    svg.push_str("\" xmlns=\"http://www.w3.org/2000/svg\">\n");
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    // Draw filled cells
    if config.filled {
        for j in 0..config.grid_size {
            for i in 0..config.grid_size {
                let t = grid[j][i] / max_density;
                if t < 0.01 {
                    continue;
                }
                let x = padding + i as f64 * cell_w;
                let y = padding + 30.0 + (config.grid_size - 1 - j) as f64 * cell_h;
                let color = kde_color(t);

                svg.push_str("  <rect x=\"");
                svg.push_str(&x.to_string());
                svg.push_str("\" y=\"");
                svg.push_str(&y.to_string());
                svg.push_str("\" width=\"");
                svg.push_str(&(cell_w + 1.0).to_string());
                svg.push_str("\" height=\"");
                svg.push_str(&(cell_h + 1.0).to_string());
                svg.push_str("\" fill=\"");
                svg.push_str(&color);
                svg.push_str("\"/>\n");
            }
        }
    }

    // Scatter points overlay
    for (&x, &y) in x_data.iter().zip(y_data.iter()) {
        let sx = padding + (x - x_min) / (x_max - x_min) * chart_width;
        let sy = padding + 30.0 + chart_height * (1.0 - (y - y_min) / (y_max - y_min));

        svg.push_str("  <circle cx=\"");
        svg.push_str(&sx.to_string());
        svg.push_str("\" cy=\"");
        svg.push_str(&sy.to_string());
        svg.push_str("\" r=\"2\" fill=\"black\" opacity=\"0.3\"/>\n");
    }

    // Colorbar
    if config.show_colorbar {
        let cb_x = width - padding - 40.0;
        let cb_y = padding + 30.0;
        let cb_height = chart_height;

        svg.push_str("  <defs><linearGradient id=\"kde_cb\" x1=\"0\" y1=\"1\" x2=\"0\" y2=\"0\">\n");
        svg.push_str("    <stop offset=\"0%\" stop-color=\"");
        svg.push_str(&kde_color(0.0));
        svg.push_str("\"/>\n");
        svg.push_str("    <stop offset=\"100%\" stop-color=\"");
        svg.push_str(&kde_color(1.0));
        svg.push_str("\"/>\n");
        svg.push_str("  </linearGradient></defs>\n");

        svg.push_str("  <rect x=\"");
        svg.push_str(&cb_x.to_string());
        svg.push_str("\" y=\"");
        svg.push_str(&cb_y.to_string());
        svg.push_str("\" width=\"15\" height=\"");
        svg.push_str(&cb_height.to_string());
        svg.push_str("\" fill=\"url(#kde_cb)\"/>\n");
    }

    // Title
    if !config.plot_config.title.is_empty() {
        svg.push_str("  <text x=\"");
        svg.push_str(&(width / 2.0).to_string());
        svg.push_str("\" y=\"25\" text-anchor=\"middle\" font-size=\"20\" font-weight=\"bold\">");
        svg.push_str(&config.plot_config.title);
        svg.push_str("</text>\n");
    }

    svg.push_str("</svg>");
    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kde_plot_renders() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1 + (i % 3) as f64 * 0.5).collect();
        let y: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin() + (i % 5) as f64 * 0.3).collect();
        let config = KdeConfig::new();
        let svg = render_kde_plot(&x, &y, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn kde_plot_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        let config = KdeConfig::new();
        assert!(render_kde_plot(&x, &y, &config).is_err());
    }
}
