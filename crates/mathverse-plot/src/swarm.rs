//! Swarm plot (non-overlapping categorical scatter).

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single category with its data points.
#[derive(Debug, Clone)]
pub struct SwarmCategory {
    /// Category label.
    pub label: String,
    /// Data values.
    pub values: Vec<f64>,
    /// Color for this category.
    pub color: Color,
}

impl SwarmCategory {
    /// Create a new swarm category.
    pub fn new(label: impl Into<String>, values: Vec<f64>, color: Color) -> Self {
        Self {
            label: label.into(),
            values,
            color,
        }
    }
}

/// Configuration for a swarm plot.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Maximum point radius.
    pub point_size: f64,
    /// Spacing between points.
    pub point_spacing: f64,
    /// Show grid.
    pub show_grid: bool,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            point_size: 4.0,
            point_spacing: 1.0,
            show_grid: true,
        }
    }
}

impl SwarmConfig {
    /// Create a new swarm config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set point size.
    pub fn with_point_size(mut self, size: f64) -> Self {
        self.point_size = size;
        self
    }
}

/// Compute swarm positions for a set of points.
fn compute_swarm_positions(values: &[f64], max_radius: f64, _width: f64) -> Vec<(f64, f64)> {
    if values.is_empty() {
        return vec![];
    }

    let mut sorted: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(values.len());
    let mut placed: Vec<(f64, f64)> = Vec::new();

    for (orig_idx, val) in &sorted {
        let mut x = 0.0;
        let y = *val;
        let mut attempts = 0;

        loop {
            let mut collision = false;
            for &(px, py) in &placed {
                let dx = x - px;
                let dy = y - py;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < max_radius * 2.0 * config_point_spacing() {
                    collision = true;
                    break;
                }
            }

            if !collision {
                break;
            }

            // Try offset positions
            attempts += 1;
            if attempts % 2 == 1 {
                x = (attempts as f64 + 1.0) / 2.0 * max_radius * 2.0;
            } else {
                x = -(attempts as f64) / 2.0 * max_radius * 2.0;
            }

            if attempts > 100 {
                break;
            }
        }

        placed.push((x, y));
        positions.push((*orig_idx as f64, x));
    }

    // Sort back by original index
    let mut result = vec![(0.0, 0.0); values.len()];
    for (i, (orig_idx, _)) in sorted.iter().enumerate() {
        result[*orig_idx] = positions[i];
    }

    result
}

fn config_point_spacing() -> f64 {
    1.0
}

/// Render a swarm plot as SVG.
pub fn render_swarm_plot(categories: &[SwarmCategory], config: &SwarmConfig) -> PlotResult<String> {
    if categories.is_empty() {
        return Err(PlotError::InvalidData("no categories provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Find global y range
    let all_min = categories.iter().flat_map(|c| &c.values).fold(f64::INFINITY, |a, &b| a.min(b));
    let all_max = categories.iter().flat_map(|c| &c.values).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;
    let y_range = if all_max - all_min > 0.0 { all_max - all_min } else { 1.0 };

    let to_y = |v| padding + 30.0 + chart_height * (1.0 - (v - all_min) / y_range);

    let n = categories.len();
    let spacing = chart_width / (n + 1) as f64;

    let mut svg = String::new();
    svg.push_str("<svg width=\"");
    svg.push_str(&width.to_string());
    svg.push_str("\" height=\"");
    svg.push_str(&height.to_string());
    svg.push_str("\" xmlns=\"http://www.w3.org/2000/svg\">\n");
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    // Grid
    if config.show_grid {
        for i in 0..=5 {
            let y = padding + 30.0 + (i as f64 / 5.0) * chart_height;
            svg.push_str("  <line x1=\"");
            svg.push_str(&padding.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&(width - padding).to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" stroke=\"#eee\"/>\n");
        }
    }

    // Draw categories
    for (idx, cat) in categories.iter().enumerate() {
        let cx = padding + spacing * (idx + 1) as f64;

        // Compute swarm positions
        let positions = compute_swarm_positions(&cat.values, config.point_size, spacing * 0.8);

        for (i, &val) in cat.values.iter().enumerate() {
            let y = to_y(val);
            let (_x_idx, x_offset) = positions[i];
            let x = cx + x_offset;

            svg.push_str("  <circle cx=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" cy=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" r=\"");
            svg.push_str(&config.point_size.to_string());
            svg.push_str("\" fill=\"");
            svg.push_str(&cat.color.to_hex());
            svg.push_str("\" opacity=\"0.7\"/>\n");
        }

        // Category label
        svg.push_str("  <text x=\"");
        svg.push_str(&cx.to_string());
        svg.push_str("\" y=\"");
        svg.push_str(&(height - padding + 15.0).to_string());
        svg.push_str("\" text-anchor=\"middle\" font-size=\"11\">");
        svg.push_str(&cat.label);
        svg.push_str("</text>\n");
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
    fn swarm_plot_renders() {
        let cats = vec![
            SwarmCategory::new("A", vec![1.0, 2.0, 3.0], Color::BLUE),
            SwarmCategory::new("B", vec![2.0, 3.0, 4.0], Color::GREEN),
        ];
        let config = SwarmConfig::new();
        let svg = render_swarm_plot(&cats, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn swarm_plot_empty_error() {
        let cats = vec![];
        let config = SwarmConfig::new();
        assert!(render_swarm_plot(&cats, &config).is_err());
    }
}
