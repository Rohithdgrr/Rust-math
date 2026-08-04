//! Contour plot rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Configuration for a contour plot.
#[derive(Debug, Clone)]
pub struct ContourConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Number of contour levels.
    pub num_levels: usize,
    /// Show labels on contours.
    pub show_labels: bool,
    /// Fill contours (filled contour plot).
    pub filled: bool,
    /// Show grid.
    pub show_grid: bool,
}

impl Default for ContourConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            num_levels: 10,
            show_labels: true,
            filled: false,
            show_grid: true,
        }
    }
}

impl ContourConfig {
    /// Create a new contour config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of levels.
    pub fn with_levels(mut self, levels: usize) -> Self {
        self.num_levels = levels;
        self
    }

    /// Enable filled contours.
    pub fn with_filled(mut self) -> Self {
        self.filled = true;
        self
    }
}

/// Compute contour lines using marching squares.
fn marching_squares(
    grid: &[Vec<f64>],
    threshold: f64,
) -> Vec<Vec<(f64, f64)>> {
    let rows = grid.len();
    if rows == 0 {
        return vec![];
    }
    let cols = grid[0].len();
    let mut segments = Vec::new();

    for j in 0..rows - 1 {
        for i in 0..cols - 1 {
            let tl = grid[j][i];
            let tr = grid[j][i + 1];
            let br = grid[j + 1][i + 1];
            let bl = grid[j + 1][i];

            let mut case = 0;
            if tl >= threshold {
                case |= 1;
            }
            if tr >= threshold {
                case |= 2;
            }
            if br >= threshold {
                case |= 4;
            }
            if bl >= threshold {
                case |= 8;
            }

            // Interpolation helper
            let lerp = |a: f64, b: f64| -> f64 {
                let d = b - a;
                if d.abs() < 1e-10 {
                    0.5
                } else {
                    (threshold - a) / d
                }
            };

            let top = (i as f64 + lerp(tl, tr), j as f64);
            let right = ((i + 1) as f64, j as f64 + lerp(tr, br));
            let bottom = (i as f64 + lerp(bl, br), (j + 1) as f64);
            let left = (i as f64, j as f64 + lerp(tl, bl));

            match case {
                1 | 14 => segments.push(vec![left, top]),
                2 | 13 => segments.push(vec![top, right]),
                3 | 12 => segments.push(vec![left, right]),
                4 | 11 => segments.push(vec![right, bottom]),
                5 => {
                    segments.push(vec![left, top]);
                    segments.push(vec![right, bottom]);
                }
                6 | 9 => segments.push(vec![top, bottom]),
                7 | 8 => segments.push(vec![left, bottom]),
                10 => {
                    segments.push(vec![top, left]);
                    segments.push(vec![bottom, right]);
                }
                _ => {}
            }
        }
    }

    segments
}

/// Render a contour plot as SVG.
pub fn render_contour(
    z_grid: &[Vec<f64>],
    x_range: (f64, f64),
    y_range: (f64, f64),
    config: &ContourConfig,
) -> PlotResult<String> {
    if z_grid.is_empty() || z_grid[0].is_empty() {
        return Err(PlotError::InvalidData("empty grid".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;
    let rows = z_grid.len();
    let cols = z_grid[0].len();

    // Find z range
    let z_min = z_grid.iter().flat_map(|r| r.iter()).fold(f64::INFINITY, |a, &b| a.min(b));
    let z_max = z_grid.iter().flat_map(|r| r.iter()).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;

    let to_x = |x: f64| padding + (x - x_range.0) / (x_range.1 - x_range.0) * chart_width;
    let to_y = |y: f64| padding + 30.0 + chart_height * (1.0 - (y - y_range.0) / (y_range.1 - y_range.0));

    // Color map (simple blue to red)
    let color_for_level = |t: f64| -> String {
        let r = (t * 255.0) as u8;
        let b = ((1.0 - t) * 255.0) as u8;
        format!("#{:02x}00{:02x}", r, b)
    };

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
        width as u32, height as u32
    ));
    svg.push('\n');
    svg.push_str(r#"  <rect width="100%" height="100%" fill="white"/>"#);
    svg.push('\n');

    if config.filled {
        // Filled contour: draw filled cells
        for j in 0..rows - 1 {
            for i in 0..cols - 1 {
                let avg = (z_grid[j][i] + z_grid[j][i + 1] + z_grid[j + 1][i] + z_grid[j + 1][i + 1]) / 4.0;
                let t = (avg - z_min) / (z_max - z_min);
                let color = color_for_level(t);

                let x0 = to_x(x_range.0 + (x_range.1 - x_range.0) * i as f64 / (cols - 1) as f64);
                let y0 = to_y(y_range.0 + (y_range.1 - y_range.0) * j as f64 / (rows - 1) as f64);
                let x1 = to_x(x_range.0 + (x_range.1 - x_range.0) * (i + 1) as f64 / (cols - 1) as f64);
                let y1 = to_y(y_range.0 + (y_range.1 - y_range.0) * (j + 1) as f64 / (rows - 1) as f64);

                svg.push_str(&format!(
                    r#"  <rect x="{x0}" y="{y0}" width="{}" height="{}" fill="{color}" stroke="none"/>"#,
                    x1 - x0, y1 - y0
                ));
                svg.push('\n');
            }
        }
    }

    // Compute levels
    let levels: Vec<f64> = (0..config.num_levels)
        .map(|i| z_min + (z_max - z_min) * (i as f64 + 0.5) / config.num_levels as f64)
        .collect();

    // Draw contour lines
    for (level_idx, &level) in levels.iter().enumerate() {
        let segments = marching_squares(z_grid, level);
        let color = color_for_level(level_idx as f64 / (config.num_levels - 1) as f64);

        for seg in &segments {
            if seg.len() < 2 {
                continue;
            }

            // Map grid coords to data coords
            let data_coords: Vec<(f64, f64)> = seg
                .iter()
                .map(|&(gx, gy)| {
                    let x = x_range.0 + (x_range.1 - x_range.0) * gx / (cols - 1) as f64;
                    let y = y_range.0 + (y_range.1 - y_range.0) * gy / (rows - 1) as f64;
                    (x, y)
                })
                .collect();

            let x0 = to_x(data_coords[0].0);
            let y0 = to_y(data_coords[0].1);
            let x1 = to_x(data_coords[1].0);
            let y1 = to_y(data_coords[1].1);

            svg.push_str(&format!(
                r#"  <line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="1.5"/>"#,
            ));
            svg.push('\n');

            // Label
            if config.show_labels {
                let mx = (x0 + x1) / 2.0;
                let my = (y0 + y1) / 2.0;
                svg.push_str(&format!(
                    r#"  <text x="{mx}" y="{my}" font-size="9" fill="{color}" text-anchor="middle" dy="-3">{:.1}</text>"#,
                    level
                ));
                svg.push('\n');
            }
        }
    }

    // Axes labels
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="11">x</text>"#,
        width / 2.0, height - 5.0
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r#"  <text x="10" y="{}" text-anchor="middle" font-size="11" transform="rotate(-90, 10, {})">y</text>"#,
        height / 2.0, height / 2.0
    ));
    svg.push('\n');

    // Title
    if !config.plot_config.title.is_empty() {
        svg.push_str(&format!(
            r#"  <text x="{}" y="25" text-anchor="middle" font-size="20" font-weight="bold">{}</text>"#,
            width / 2.0, config.plot_config.title
        ));
        svg.push('\n');
    }

    svg.push_str("</svg>");
    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contour_plot_renders() {
        let grid: Vec<Vec<f64>> = (0..10)
            .map(|j| (0..10).map(|i| (i as f64 * i as f64 + j as f64 * j as f64) / 100.0).collect())
            .collect();
        let config = ContourConfig::new();
        let svg = render_contour(&grid, (0.0, 1.0), (0.0, 1.0), &config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn contour_plot_empty_error() {
        let grid: Vec<Vec<f64>> = vec![];
        let config = ContourConfig::new();
        assert!(render_contour(&grid, (0.0, 1.0), (0.0, 1.0), &config).is_err());
    }
}
