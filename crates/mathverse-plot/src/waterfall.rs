//! Waterfall chart rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single bar in a waterfall chart.
#[derive(Debug, Clone)]
pub struct WaterfallBar {
    /// Label for the bar.
    pub label: String,
    /// Value (positive = increase, negative = decrease).
    pub value: f64,
    /// Whether this is a total/subtotal bar.
    pub is_total: bool,
}

impl WaterfallBar {
    /// Create a new waterfall bar.
    pub fn new(label: impl Into<String>, value: f64) -> Self {
        Self {
            label: label.into(),
            value,
            is_total: false,
        }
    }

    /// Mark as total/subtotal.
    pub fn as_total(mut self) -> Self {
        self.is_total = true;
        self
    }
}

/// Configuration for a waterfall chart.
#[derive(Debug, Clone)]
pub struct WaterfallConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Color for increases.
    pub increase_color: Color,
    /// Color for decreases.
    pub decrease_color: Color,
    /// Color for totals.
    pub total_color: Color,
    /// Bar width (pixels).
    pub bar_width: f64,
    /// Show connecting lines.
    pub show_connectors: bool,
    /// Show values on bars.
    pub show_values: bool,
    /// Show grid.
    pub show_grid: bool,
    /// Font size.
    pub font_size: f64,
}

impl Default for WaterfallConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            increase_color: Color::GREEN,
            decrease_color: Color::RED,
            total_color: Color::BLUE,
            bar_width: 40.0,
            show_connectors: true,
            show_values: true,
            show_grid: true,
            font_size: 11.0,
        }
    }
}

impl WaterfallConfig {
    /// Create a new waterfall config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set colors.
    pub fn with_colors(mut self, increase: Color, decrease: Color, total: Color) -> Self {
        self.increase_color = increase;
        self.decrease_color = decrease;
        self.total_color = total;
        self
    }
}

/// Render a waterfall chart as SVG.
pub fn render_waterfall(bars: &[WaterfallBar], config: &WaterfallConfig) -> PlotResult<String> {
    if bars.is_empty() {
        return Err(PlotError::InvalidData("no bars provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Calculate running totals
    let mut running_total: f64 = 0.0;
    let mut min_val: f64 = 0.0;
    let mut max_val: f64 = 0.0;

    for bar in bars {
        if bar.is_total {
            running_total = bar.value;
        } else {
            running_total += bar.value;
        }
        max_val = max_val.max(running_total);
        if bar.is_total {
            min_val = min_val.min(0.0);
            max_val = max_val.max(bar.value);
        }
    }
    max_val = max_val.max(0.0);

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;
    let zero_y = padding + 30.0 + chart_height * (max_val / (max_val - min_val));

    let bar_spacing = config.bar_width * 0.3;
    let total_space = bars.len() as f64 * (config.bar_width + bar_spacing);
    let start_x = padding + (chart_width - total_space) / 2.0;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
        width as u32, height as u32
    ));
    svg.push('\n');
    svg.push_str(r#"  <rect width="100%" height="100%" fill="white"/>"#);
    svg.push('\n');

    // Grid
    if config.show_grid {
        let x_right = width - padding;
        for i in 0..=5 {
            let y = padding + 30.0 + (i as f64 / 5.0) * chart_height;
            svg.push_str("  <line x1=\"");
            svg.push_str(&padding.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&x_right.to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" stroke=\"#eee\"/>\n");
        }
    }

    // Zero line
    {
        let x_right = width - padding;
        svg.push_str("  <line x1=\"");
        svg.push_str(&padding.to_string());
        svg.push_str("\" y1=\"");
        svg.push_str(&zero_y.to_string());
        svg.push_str("\" x2=\"");
        svg.push_str(&x_right.to_string());
        svg.push_str("\" y2=\"");
        svg.push_str(&zero_y.to_string());
        svg.push_str("\" stroke=\"gray\" stroke-dasharray=\"4\"/>\n");
    }
    svg.push('\n');

    // Draw bars
    let mut current_bottom = 0.0;
    let mut prev_top_y = 0.0;

    for (i, bar) in bars.iter().enumerate() {
        let x = start_x + i as f64 * (config.bar_width + bar_spacing);
        let bar_bottom;
        let bar_height;
        let color;

        if bar.is_total {
            // Total bar starts from zero
            bar_bottom = 0.0;
            bar_height = bar.value.abs();
            color = &config.total_color;
            current_bottom = bar.value;
        } else {
            // Incremental bar
            bar_bottom = current_bottom;
            bar_height = bar.value.abs();
            color = if bar.value >= 0.0 {
                &config.increase_color
            } else {
                &config.decrease_color
            };
            current_bottom += bar.value;
        }

        let y_top;
        let y_bottom;

        if bar.is_total || bar.value >= 0.0 {
            y_top = zero_y - (bar_bottom + bar_height) / (max_val - min_val) * chart_height;
            y_bottom = zero_y - bar_bottom / (max_val - min_val) * chart_height;
        } else {
            y_top = zero_y - bar_bottom / (max_val - min_val) * chart_height;
            y_bottom = zero_y - (bar_bottom + bar_height) / (max_val - min_val) * chart_height;
        }

        svg.push_str(&format!(
            r#"  <rect x="{x}" y="{y_top}" width="{}" height="{}" fill="{}" rx="2"/>"#,
            config.bar_width,
            y_bottom - y_top,
            color.to_hex()
        ));
        svg.push('\n');

        // Connector line
        if config.show_connectors && i > 0 {
            let prev_x = start_x + (i - 1) as f64 * (config.bar_width + bar_spacing) + config.bar_width;
            svg.push_str(&format!(
                r#"  <line x1="{prev_x}" y1="{prev_top_y}" x2="{x}" y2="{prev_top_y}" stroke="gray" stroke-dasharray="3" stroke-width="1"/>"#,
            ));
            svg.push('\n');
        }

        // Value label
        if config.show_values {
            let label_y = if bar.value >= 0.0 || bar.is_total {
                y_top - 5.0
            } else {
                y_bottom + 14.0
            };
            svg.push_str(&format!(
                r#"  <text x="{}" y="{label_y}" text-anchor="middle" font-size="{}">{:.0}</text>"#,
                x + config.bar_width / 2.0,
                config.font_size,
                bar.value
            ));
            svg.push('\n');
        }

        // Category label
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" text-anchor="middle" font-size="10">{}</text>"#,
            x + config.bar_width / 2.0,
            height - padding + 15.0,
            bar.label
        ));
        svg.push('\n');

        // Update prev_top_y for connector
        prev_top_y = if bar.is_total || bar.value >= 0.0 {
            y_top
        } else {
            y_bottom
        };
    }

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
    fn waterfall_renders() {
        let bars = vec![
            WaterfallBar::new("Start", 100.0).as_total(),
            WaterfallBar::new("Revenue", 50.0),
            WaterfallBar::new("Cost", -30.0),
            WaterfallBar::new("Tax", -10.0),
            WaterfallBar::new("Net", 110.0).as_total(),
        ];
        let config = WaterfallConfig::new();
        let svg = render_waterfall(&bars, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn waterfall_empty_error() {
        let bars = vec![];
        let config = WaterfallConfig::new();
        assert!(render_waterfall(&bars, &config).is_err());
    }
}
