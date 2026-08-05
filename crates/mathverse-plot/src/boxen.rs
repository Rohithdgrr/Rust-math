//! Boxen (letter-value) plot rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single boxen data set.
#[derive(Debug, Clone)]
pub struct BoxenData {
    /// Label for the boxen.
    pub label: String,
    /// Data values.
    pub values: Vec<f64>,
    /// Fill color.
    pub color: Color,
}

impl BoxenData {
    /// Create new boxen data.
    pub fn new(label: impl Into<String>, values: Vec<f64>, color: Color) -> Self {
        Self {
            label: label.into(),
            values,
            color,
        }
    }
}

/// Configuration for a boxen plot.
#[derive(Debug, Clone)]
pub struct BoxenConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Width of each box (pixels).
    pub box_width: f64,
    /// Number of levels (depth of letter-value boxes).
    pub num_levels: usize,
    /// Show grid.
    pub show_grid: bool,
    /// Show median.
    pub show_median: bool,
    /// Font size.
    pub font_size: f64,
}

impl Default for BoxenConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            box_width: 40.0,
            num_levels: 6,
            show_grid: true,
            show_median: true,
            font_size: 11.0,
        }
    }
}

impl BoxenConfig {
    /// Create a new boxen config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of levels.
    pub fn with_levels(mut self, levels: usize) -> Self {
        self.num_levels = levels;
        self
    }
}

/// Compute quantile.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let idx = pos as usize;
    let frac = pos - idx as f64;
    if idx + 1 < sorted.len() {
        sorted[idx] * (1.0 - frac) + sorted[idx + 1] * frac
    } else {
        sorted[idx]
    }
}

/// Render a boxen plot as SVG.
pub fn render_boxen_plot(data: &[BoxenData], config: &BoxenConfig) -> PlotResult<String> {
    if data.is_empty() {
        return Err(PlotError::InvalidData("no data provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Find global y range
    let all_min = data.iter().flat_map(|d| &d.values).fold(f64::INFINITY, |a, &b| a.min(b));
    let all_max = data.iter().flat_map(|d| &d.values).fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;
    let y_range = all_max - all_min;

    let to_y = |v| padding + 30.0 + chart_height * (1.0 - (v - all_min) / y_range);

    let n = data.len();
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

    // Draw boxen
    for (idx, d) in data.iter().enumerate() {
        let cx = padding + spacing * (idx + 1) as f64;

        let mut sorted = d.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Draw letter-value boxes
        for level in 0..config.num_levels {
            let lower_q = 0.5_f64.powi(level as i32 + 1);
            let upper_q = 1.0 - 0.5_f64.powi(level as i32 + 1);

            let lower = quantile(&sorted, lower_q);
            let upper = quantile(&sorted, upper_q);

            let y1 = to_y(upper);
            let y2 = to_y(lower);
            let box_height = y2 - y1;

            // Width decreases with depth
            let width_factor = 1.0 / (level as f64 + 1.0);
            let box_w = config.box_width * width_factor;

            // Lighter color for deeper boxes
            let opacity = 1.0 - level as f64 * 0.1;

            svg.push_str("  <rect x=\"");
            svg.push_str(&(cx - box_w / 2.0).to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&y1.to_string());
            svg.push_str("\" width=\"");
            svg.push_str(&box_w.to_string());
            svg.push_str("\" height=\"");
            svg.push_str(&box_height.to_string());
            svg.push_str("\" fill=\"");
            svg.push_str(&d.color.to_hex());
            svg.push_str("\" opacity=\"");
            svg.push_str(&opacity.to_string());
            svg.push_str("\" stroke=\"black\" stroke-width=\"0.5\"/>\n");
        }

        // Median line
        if config.show_median {
            let median = quantile(&sorted, 0.5);
            let y = to_y(median);
            let box_w = config.box_width * 0.3;

            svg.push_str("  <line x1=\"");
            svg.push_str(&(cx - box_w / 2.0).to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&(cx + box_w / 2.0).to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" stroke=\"white\" stroke-width=\"2\"/>\n");
        }

        // Whiskers (min to max)
        let min_val = sorted[0];
        let max_val = *sorted.last().ok_or_else(|| PlotError::InvalidData("empty data".into()))?;
        svg.push_str("  <line x1=\"");
        svg.push_str(&cx.to_string());
        svg.push_str("\" y1=\"");
        svg.push_str(&to_y(max_val).to_string());
        svg.push_str("\" x2=\"");
        svg.push_str(&cx.to_string());
        svg.push_str("\" y2=\"");
        svg.push_str(&to_y(min_val).to_string());
        svg.push_str("\" stroke=\"black\" stroke-width=\"1\"/>\n");

        // Category label
        svg.push_str("  <text x=\"");
        svg.push_str(&cx.to_string());
        svg.push_str("\" y=\"");
        svg.push_str(&(height - padding + 15.0).to_string());
        svg.push_str("\" text-anchor=\"middle\" font-size=\"11\">");
        svg.push_str(&d.label);
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
    fn boxen_plot_renders() {
        let data = vec![
            BoxenData::new("A", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], Color::BLUE),
            BoxenData::new("B", vec![2.0, 3.0, 4.0, 5.0, 6.0], Color::GREEN),
        ];
        let config = BoxenConfig::new();
        let svg = render_boxen_plot(&data, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn boxen_plot_empty_error() {
        let data = vec![];
        let config = BoxenConfig::new();
        assert!(render_boxen_plot(&data, &config).is_err());
    }
}
