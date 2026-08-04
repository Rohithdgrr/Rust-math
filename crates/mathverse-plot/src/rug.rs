//! Rug plot rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Configuration for a rug plot.
#[derive(Debug, Clone)]
pub struct RugConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Height of rug marks (pixels).
    pub rug_height: f64,
    /// Side to place rug marks ("bottom", "top", "both").
    pub side: String,
    /// Color of rug marks.
    pub color: Color,
    /// Line width.
    pub line_width: f64,
    /// Show grid.
    pub show_grid: bool,
}

impl Default for RugConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            rug_height: 15.0,
            side: "bottom".into(),
            color: Color::BLACK,
            line_width: 1.0,
            show_grid: true,
        }
    }
}

impl RugConfig {
    /// Create a new rug config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set side.
    pub fn with_side(mut self, side: impl Into<String>) -> Self {
        self.side = side.into();
        self
    }

    /// Set height.
    pub fn with_height(mut self, height: f64) -> Self {
        self.rug_height = height;
        self
    }
}

/// Render a rug plot as SVG.
pub fn render_rug_plot(values: &[f64], config: &RugConfig) -> PlotResult<String> {
    if values.is_empty() {
        return Err(PlotError::InvalidData("no values provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if min_val == max_val {
        return Err(PlotError::InvalidData("all values are identical".into()));
    }

    let chart_width = width - padding * 2.0;
    let to_x = |v| padding + (v - min_val) / (max_val - min_val) * chart_width;

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
        let y_top = padding;
        let y_bottom = height - padding;
        for i in 0..=5 {
            let x = padding + (i as f64 / 5.0) * chart_width;
            svg.push_str("  <line x1=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&y_top.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y_bottom.to_string());
            svg.push_str("\" stroke=\"#eee\"/>\n");
        }
    }

    // Rug marks
    let base_y_bottom = height - padding;
    let base_y_top = padding;

    for &v in values {
        let x = to_x(v);

        match config.side.as_str() {
            "bottom" => {
                svg.push_str(&format!(
                    r#"  <line x1="{x}" y1="{base_y_bottom}" x2="{x}" y2="{}" stroke="{}" stroke-width="{}"/>"#,
                    base_y_bottom - config.rug_height,
                    config.color.to_hex(),
                    config.line_width
                ));
            }
            "top" => {
                svg.push_str(&format!(
                    r#"  <line x1="{x}" y1="{base_y_top}" x2="{x}" y2="{}" stroke="{}" stroke-width="{}"/>"#,
                    base_y_top + config.rug_height,
                    config.color.to_hex(),
                    config.line_width
                ));
            }
            _ => {
                // Both
                svg.push_str(&format!(
                    r#"  <line x1="{x}" y1="{base_y_bottom}" x2="{x}" y2="{}" stroke="{}" stroke-width="{}"/>"#,
                    base_y_bottom - config.rug_height,
                    config.color.to_hex(),
                    config.line_width
                ));
                svg.push('\n');
                svg.push_str(&format!(
                    r#"  <line x1="{x}" y1="{base_y_top}" x2="{x}" y2="{}" stroke="{}" stroke-width="{}"/>"#,
                    base_y_top + config.rug_height,
                    config.color.to_hex(),
                    config.line_width
                ));
            }
        }
        svg.push('\n');
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
    fn rug_plot_renders() {
        let values = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8];
        let config = RugConfig::new();
        let svg = render_rug_plot(&values, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<line"));
    }

    #[test]
    fn rug_plot_empty_error() {
        let values = vec![];
        let config = RugConfig::new();
        assert!(render_rug_plot(&values, &config).is_err());
    }

    #[test]
    fn rug_plot_top_side() {
        let values = vec![1.0, 2.0, 3.0];
        let config = RugConfig::new().with_side("top");
        let svg = render_rug_plot(&values, &config).unwrap();
        assert!(svg.contains("<line"));
    }
}
