//! Count plot (categorical counts as bars).

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Configuration for a count plot.
#[derive(Debug, Clone)]
pub struct CountConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Bar color.
    pub color: Color,
    /// Bar width (pixels).
    pub bar_width: f64,
    /// Show values on bars.
    pub show_values: bool,
    /// Font size.
    pub font_size: f64,
    /// Horizontal bars.
    pub horizontal: bool,
}

impl Default for CountConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            color: Color::BLUE,
            bar_width: 40.0,
            show_values: true,
            font_size: 11.0,
            horizontal: false,
        }
    }
}

impl CountConfig {
    /// Create a new count config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set horizontal.
    pub fn with_horizontal(mut self) -> Self {
        self.horizontal = true;
        self
    }
}

/// Render a count plot as SVG.
pub fn render_countplot(
    categories: &[String],
    config: &CountConfig,
) -> PlotResult<String> {
    if categories.is_empty() {
        return Err(PlotError::InvalidData("no categories provided".into()));
    }

    // Count occurrences
    let mut counts = std::collections::HashMap::new();
    for cat in categories {
        *counts.entry(cat.as_str()).or_insert(0u64) += 1;
    }

    let mut sorted: Vec<(&str, u64)> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    let max_count = sorted.iter().map(|(_, c)| *c).max().unwrap_or(1) as f64;

    let mut svg = String::new();
    svg.push_str("<svg width=\"");
    svg.push_str(&width.to_string());
    svg.push_str("\" height=\"");
    svg.push_str(&height.to_string());
    svg.push_str("\" xmlns=\"http://www.w3.org/2000/svg\">\n");
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    if config.horizontal {
        // Horizontal bars
        let chart_height = height - padding * 2.0 - 30.0;
        let bar_height = chart_height / sorted.len() as f64 * 0.7;
        let spacing = chart_height / sorted.len() as f64;
        let chart_width = width - padding * 2.0 - 80.0;

        for (i, (label, count)) in sorted.iter().enumerate() {
            let y = padding + 30.0 + i as f64 * spacing;
            let bar_w = (*count as f64 / max_count) * chart_width;

            // Label
            svg.push_str("  <text x=\"");
            svg.push_str(&(padding + 70.0).to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&(y + bar_height / 2.0).to_string());
            svg.push_str("\" text-anchor=\"end\" font-size=\"");
            svg.push_str(&config.font_size.to_string());
            svg.push_str("\" dominant-baseline=\"middle\">");
            svg.push_str(label);
            svg.push_str("</text>\n");

            // Bar
            svg.push_str("  <rect x=\"");
            svg.push_str(&(padding + 80.0).to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" width=\"");
            svg.push_str(&bar_w.to_string());
            svg.push_str("\" height=\"");
            svg.push_str(&bar_height.to_string());
            svg.push_str("\" fill=\"");
            svg.push_str(&config.color.to_hex());
            svg.push_str("\" rx=\"2\"/>\n");

            // Value
            if config.show_values {
                svg.push_str("  <text x=\"");
                svg.push_str(&(padding + 80.0 + bar_w + 5.0).to_string());
                svg.push_str("\" y=\"");
                svg.push_str(&(y + bar_height / 2.0).to_string());
                svg.push_str("\" font-size=\"");
                svg.push_str(&config.font_size.to_string());
                svg.push_str("\" dominant-baseline=\"middle\">");
                svg.push_str(&count.to_string());
                svg.push_str("</text>\n");
            }
        }
    } else {
        // Vertical bars
        let chart_width = width - padding * 2.0;
        let chart_height = height - padding * 2.0 - 30.0;
        let spacing = chart_width / sorted.len() as f64;

        // Grid
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

        for (i, (label, count)) in sorted.iter().enumerate() {
            let x = padding + (i as f64 + 0.5) * spacing;
            let bar_h = (*count as f64 / max_count) * chart_height;
            let bar_y = padding + 30.0 + chart_height - bar_h;
            let bar_w = config.bar_width;

            // Bar
            svg.push_str("  <rect x=\"");
            svg.push_str(&(x - bar_w / 2.0).to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&bar_y.to_string());
            svg.push_str("\" width=\"");
            svg.push_str(&bar_w.to_string());
            svg.push_str("\" height=\"");
            svg.push_str(&bar_h.to_string());
            svg.push_str("\" fill=\"");
            svg.push_str(&config.color.to_hex());
            svg.push_str("\" rx=\"2\"/>\n");

            // Value
            if config.show_values {
                svg.push_str("  <text x=\"");
                svg.push_str(&x.to_string());
                svg.push_str("\" y=\"");
                svg.push_str(&(bar_y - 5.0).to_string());
                svg.push_str("\" text-anchor=\"middle\" font-size=\"");
                svg.push_str(&config.font_size.to_string());
                svg.push_str("\">");
                svg.push_str(&count.to_string());
                svg.push_str("</text>\n");
            }

            // Label
            svg.push_str("  <text x=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&(height - padding + 15.0).to_string());
            svg.push_str("\" text-anchor=\"middle\" font-size=\"10\">");
            svg.push_str(label);
            svg.push_str("</text>\n");
        }
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
    fn countplot_renders() {
        let cats = vec![
            "A".into(), "B".into(), "A".into(), "C".into(),
            "B".into(), "A".into(), "C".into(), "C".into(),
        ];
        let config = CountConfig::new();
        let svg = render_countplot(&cats, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn countplot_empty_error() {
        let cats = vec![];
        let config = CountConfig::new();
        assert!(render_countplot(&cats, &config).is_err());
    }
}
