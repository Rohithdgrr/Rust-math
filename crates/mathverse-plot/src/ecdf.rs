//! ECDF (Empirical Cumulative Distribution Function) plot rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Configuration for an ECDF plot.
#[derive(Debug, Clone)]
pub struct EcdfConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Line color.
    pub color: Color,
    /// Line width.
    pub line_width: f64,
    /// Show markers at data points.
    pub show_markers: bool,
    /// Show grid.
    pub show_grid: bool,
    /// Show confidence bands.
    pub show_confidence: bool,
    /// Confidence level (e.g., 0.95 for 95%).
    pub confidence_level: f64,
}

impl Default for EcdfConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            color: Color::BLUE,
            line_width: 2.0,
            show_markers: true,
            show_grid: true,
            show_confidence: false,
            confidence_level: 0.95,
        }
    }
}

impl EcdfConfig {
    /// Create a new ECDF config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable confidence bands.
    pub fn with_confidence(mut self, level: f64) -> Self {
        self.show_confidence = true;
        self.confidence_level = level;
        self
    }
}

/// Render an ECDF plot as SVG.
pub fn render_ecdf(values: &[f64], config: &EcdfConfig) -> PlotResult<String> {
    if values.is_empty() {
        return Err(PlotError::InvalidData("no values provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Sort values
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len() as f64;
    let min_val = sorted[0];
    let max_val = sorted.last().ok_or_else(|| PlotError::InvalidData("empty data".into()))?;

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;

    let to_x = |v| padding + (v - min_val) / (max_val - min_val) * chart_width;
    let to_y = |p| padding + 30.0 + chart_height * (1.0 - p);

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

    // Confidence bands
    if config.show_confidence {
        let z = 1.96; // For 95% CI
        let mut path_upper = String::from("M");
        let mut path_lower = String::from("M");

        for (i, &v) in sorted.iter().enumerate() {
            let p = (i + 1) as f64 / n;
            let se = (p * (1.0 - p) / n).sqrt() * z;
            let x = to_x(v);

            let y_upper = to_y((p + se).min(1.0));
            let y_lower = to_y((p - se).max(0.0));

            if i == 0 {
                path_upper.push_str(&format!(" {x},{y_upper}"));
                path_lower.push_str(&format!(" {x},{y_lower}"));
            } else {
                path_upper.push_str(&format!(" L {x},{y_upper}"));
                path_lower.push_str(&format!(" L {x},{y_lower}"));
            }
        }

        // Fill between
        let mut fill_path = path_upper.clone();
        for (i, &v) in sorted.iter().rev().enumerate() {
            let p = (sorted.len() - i) as f64 / n;
            let se = (p * (1.0 - p) / n).sqrt() * z;
            let x = to_x(v);
            let y_lower = to_y((p - se).max(0.0));
            fill_path.push_str(&format!(" L {x},{y_lower}"));
        }
        fill_path.push_str(" Z");

        svg.push_str(&format!(
            r#"  <path d="{fill_path}" fill="{}" fill-opacity="0.2"/>"#,
            config.color.to_hex()
        ));
        svg.push('\n');
    }

    // ECDF step function
    let mut path = String::from("M");
    let y0 = to_y(0.0);
    path.push_str(&format!(" {},{}", to_x(min_val), y0));

    for (i, &v) in sorted.iter().enumerate() {
        let x = to_x(v);
        let p = (i + 1) as f64 / n;
        let y = to_y(p);

        // Step: vertical then horizontal
        path.push_str(&format!(" L {x},{}", to_y(i as f64 / n)));
        path.push_str(&format!(" L {x},{y}"));
    }

    svg.push_str(&format!(
        r#"  <path d="{path}" fill="none" stroke="{}" stroke-width="{}"/>"#,
        config.color.to_hex(),
        config.line_width
    ));
    svg.push('\n');

    // Markers
    if config.show_markers {
        for (i, &v) in sorted.iter().enumerate() {
            let x = to_x(v);
            let p = (i + 1) as f64 / n;
            let y = to_y(p);
            svg.push_str(&format!(
                r#"  <circle cx="{x}" cy="{y}" r="2" fill="{}"/>"#,
                config.color.to_hex()
            ));
            svg.push('\n');
        }
    }

    // Axes
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="11">x</text>"#,
        width / 2.0, height - 5.0
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r#"  <text x="10" y="{}" text-anchor="middle" font-size="11" transform="rotate(-90, 10, {})">ECDF</text>"#,
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
    fn ecdf_renders() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5];
        let config = EcdfConfig::new();
        let svg = render_ecdf(&values, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn ecdf_empty_error() {
        let values = vec![];
        let config = EcdfConfig::new();
        assert!(render_ecdf(&values, &config).is_err());
    }

    #[test]
    fn ecdf_with_confidence() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = EcdfConfig::new().with_confidence(0.95);
        let svg = render_ecdf(&values, &config).unwrap();
        assert!(svg.contains("fill-opacity"));
    }
}
