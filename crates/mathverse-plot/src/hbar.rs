//! Horizontal bar chart rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single bar in a horizontal bar chart.
#[derive(Debug, Clone)]
pub struct HBar {
    /// Label for the bar.
    pub label: String,
    /// Length of the bar (horizontal value).
    pub value: f64,
    /// Fill color.
    pub color: Color,
    /// Optional error bar.
    pub error: Option<f64>,
}

impl HBar {
    /// Create a new horizontal bar.
    pub fn new(label: impl Into<String>, value: f64, color: Color) -> Self {
        Self {
            label: label.into(),
            value,
            color,
            error: None,
        }
    }

    /// Set an error bar.
    pub fn with_error(mut self, error: f64) -> Self {
        self.error = Some(error);
        self
    }
}

/// Configuration for a horizontal bar chart.
#[derive(Debug, Clone)]
pub struct HBarConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Width of each bar (pixels).
    pub bar_height: f64,
    /// Spacing between bars (fraction of bar height).
    pub bar_spacing: f64,
    /// Show grid lines.
    pub show_grid: bool,
    /// Show values on bars.
    pub show_values: bool,
    /// Font size for labels.
    pub font_size: f64,
}

impl Default for HBarConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            bar_height: 30.0,
            bar_spacing: 0.2,
            show_grid: true,
            show_values: true,
            font_size: 12.0,
        }
    }
}

impl HBarConfig {
    /// Create a new horizontal bar config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set bar height.
    pub fn with_bar_height(mut self, height: f64) -> Self {
        self.bar_height = height;
        self
    }

    /// Set bar spacing.
    pub fn with_bar_spacing(mut self, spacing: f64) -> Self {
        self.bar_spacing = spacing;
        self
    }

    /// Show/hide values.
    pub fn with_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }
}

/// Render a horizontal bar chart as SVG.
pub fn render_hbar_chart(bars: &[HBar], config: &HBarConfig) -> PlotResult<String> {
    if bars.is_empty() {
        return Err(PlotError::InvalidData("no bars provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Calculate max value
    let max_value = bars.iter().map(|b| b.value).fold(0.0_f64, f64::max);

    // Calculate layout
    let label_width = 80.0;
    let chart_width = width - padding * 2.0 - label_width;
    let total_bar_height = bars.len() as f64 * config.bar_height * (1.0 + config.bar_spacing);
    let start_y = (height - total_bar_height) / 2.0;

    let mut svg = String::new();

    svg.push_str("<svg width=\"");
    svg.push_str(&width.to_string());
    svg.push_str("\" height=\"");
    svg.push_str(&height.to_string());
    svg.push_str("\" xmlns=\"http://www.w3.org/2000/svg\">\n");

    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    // Grid lines
    if config.show_grid {
        let grid_steps = 5;
        for i in 0..=grid_steps {
            let x = label_width + (i as f64 / grid_steps as f64) * chart_width;
            svg.push_str("  <line x1=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&padding.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&(height - padding).to_string());
            svg.push_str("\" stroke=\"#eee\" stroke-width=\"1\"/>\n");
        }
    }

    // Draw bars
    for (i, bar) in bars.iter().enumerate() {
        let y = start_y + i as f64 * config.bar_height * (1.0 + config.bar_spacing);
        let bar_width = (bar.value / max_value) * chart_width;

        // Label
        svg.push_str("  <text x=\"");
        svg.push_str(&(label_width - 5.0).to_string());
        svg.push_str("\" y=\"");
        svg.push_str(&(y + config.bar_height / 2.0).to_string());
        svg.push_str("\" text-anchor=\"end\" font-size=\"");
        svg.push_str(&config.font_size.to_string());
        svg.push_str("\" dominant-baseline=\"middle\">");
        svg.push_str(&bar.label);
        svg.push_str("</text>\n");

        // Bar
        svg.push_str("  <rect x=\"");
        svg.push_str(&label_width.to_string());
        svg.push_str("\" y=\"");
        svg.push_str(&y.to_string());
        svg.push_str("\" width=\"");
        svg.push_str(&bar_width.to_string());
        svg.push_str("\" height=\"");
        svg.push_str(&config.bar_height.to_string());
        svg.push_str("\" fill=\"");
        svg.push_str(&bar.color.to_hex());
        svg.push_str("\" rx=\"3\"/>\n");

        // Value label
        if config.show_values {
            svg.push_str("  <text x=\"");
            svg.push_str(&(label_width + bar_width).to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&(y + config.bar_height / 2.0).to_string());
            svg.push_str("\" text-anchor=\"start\" font-size=\"");
            svg.push_str(&config.font_size.to_string());
            svg.push_str("\" dominant-baseline=\"middle\">  ");
            svg.push_str(&format!("{:.1}", bar.value));
            svg.push_str("</text>\n");
        }

        // Error bar
        if let Some(error) = bar.error {
            let error_width = (error / max_value) * chart_width;
            let x_center = label_width + bar_width;
            let y_center = y + config.bar_height / 2.0;
            let cap_size = 4.0;

            svg.push_str("  <line x1=\"");
            svg.push_str(&(x_center - error_width).to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&y_center.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&(x_center + error_width).to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y_center.to_string());
            svg.push_str("\" stroke=\"black\" stroke-width=\"2\"/>\n");

            // Caps
            svg.push_str("  <line x1=\"");
            svg.push_str(&(x_center - error_width).to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&(y_center - cap_size).to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&(x_center - error_width).to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&(y_center + cap_size).to_string());
            svg.push_str("\" stroke=\"black\" stroke-width=\"2\"/>\n");

            svg.push_str("  <line x1=\"");
            svg.push_str(&(x_center + error_width).to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&(y_center - cap_size).to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&(x_center + error_width).to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&(y_center + cap_size).to_string());
            svg.push_str("\" stroke=\"black\" stroke-width=\"2\"/>\n");
        }
    }

    // Title
    if !config.plot_config.title.is_empty() {
        svg.push_str("  <text x=\"");
        svg.push_str(&(width / 2.0).to_string());
        svg.push_str("\" y=\"30\" text-anchor=\"middle\" font-size=\"20\" font-weight=\"bold\">");
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
    fn hbar_chart_renders_svg() {
        let bars = vec![
            HBar::new("Apple", 45.0, Color::RED),
            HBar::new("Banana", 32.0, Color::YELLOW),
            HBar::new("Cherry", 58.0, Color::GREEN),
        ];
        let config = HBarConfig::new();
        let svg = render_hbar_chart(&bars, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn hbar_chart_empty_error() {
        let bars = vec![];
        let config = HBarConfig::new();
        assert!(render_hbar_chart(&bars, &config).is_err());
    }
}
