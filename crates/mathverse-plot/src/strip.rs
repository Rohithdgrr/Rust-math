//! Strip plot (categorical scatter with jitter).

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single category with its data points.
#[derive(Debug, Clone)]
pub struct StripCategory {
    /// Category label.
    pub label: String,
    /// Data values.
    pub values: Vec<f64>,
    /// Color for this category.
    pub color: Color,
}

impl StripCategory {
    /// Create a new strip category.
    pub fn new(label: impl Into<String>, values: Vec<f64>, color: Color) -> Self {
        Self {
            label: label.into(),
            values,
            color,
        }
    }
}

/// Configuration for a strip plot.
#[derive(Debug, Clone)]
pub struct StripConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Jitter width (fraction of category width).
    pub jitter: f64,
    /// Marker size.
    pub marker_size: f64,
    /// Marker alpha (opacity).
    pub alpha: f64,
    /// Show grid.
    pub show_grid: bool,
    /// Show means.
    pub show_means: bool,
}

impl Default for StripConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            jitter: 0.3,
            marker_size: 4.0,
            alpha: 0.6,
            show_grid: true,
            show_means: false,
        }
    }
}

impl StripConfig {
    /// Create a new strip config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set jitter.
    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter;
        self
    }

    /// Set marker size.
    pub fn with_marker_size(mut self, size: f64) -> Self {
        self.marker_size = size;
        self
    }
}

/// Render a strip plot as SVG.
pub fn render_strip_plot(categories: &[StripCategory], config: &StripConfig) -> PlotResult<String> {
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
    let y_range = all_max - all_min;

    let to_y = |v| padding + 30.0 + chart_height * (1.0 - (v - all_min) / y_range);

    let n = categories.len();
    let spacing = chart_width / (n + 1) as f64;

    // Simple seeded random for jitter
    let pseudo_random = |seed: usize| -> f64 {
        let x = (seed as f64 * 127.1 + 311.7).sin() * 43758.5453;
        x - x.floor()
    };

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

        // Draw points with jitter
        for (i, &val) in cat.values.iter().enumerate() {
            let y = to_y(val);
            let jitter_x = (pseudo_random(idx * 1000 + i) - 0.5) * config.jitter * spacing;
            let x = cx + jitter_x;

            svg.push_str("  <circle cx=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" cy=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" r=\"");
            svg.push_str(&config.marker_size.to_string());
            svg.push_str("\" fill=\"");
            svg.push_str(&cat.color.to_hex());
            svg.push_str("\" opacity=\"");
            svg.push_str(&config.alpha.to_string());
            svg.push_str("\"/>\n");
        }

        // Mean marker
        if config.show_means && !cat.values.is_empty() {
            let mean = cat.values.iter().sum::<f64>() / cat.values.len() as f64;
            let y = to_y(mean);
            svg.push_str("  <polygon points=\"");
            let size = 6.0;
            svg.push_str(&format!("{},{} {},{} {},{}", cx, y - size, cx - size, y + size, cx + size, y + size));
            svg.push_str("\" fill=\"red\"/>\n");
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
    fn strip_plot_renders() {
        let cats = vec![
            StripCategory::new("A", vec![1.0, 2.0, 3.0, 4.0], Color::BLUE),
            StripCategory::new("B", vec![2.0, 3.0, 4.0, 5.0], Color::GREEN),
        ];
        let config = StripConfig::new();
        let svg = render_strip_plot(&cats, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn strip_plot_empty_error() {
        let cats = vec![];
        let config = StripConfig::new();
        assert!(render_strip_plot(&cats, &config).is_err());
    }
}
