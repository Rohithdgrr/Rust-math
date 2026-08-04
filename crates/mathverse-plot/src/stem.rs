//! Stem plot rendering.

use crate::common::{DataPoint, PlotConfig};
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Configuration for a stem plot.
#[derive(Debug, Clone)]
pub struct StemConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Baseline Y value.
    pub baseline: f64,
    /// Stem color.
    pub stem_color: Color,
    /// Marker color.
    pub marker_color: Color,
    /// Stem width.
    pub stem_width: f64,
    /// Marker radius.
    pub marker_radius: f64,
    /// Marker shape ("circle", "square", "diamond").
    pub marker_shape: String,
    /// Show grid.
    pub show_grid: bool,
}

impl Default for StemConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            baseline: 0.0,
            stem_color: Color::BLUE,
            marker_color: Color::BLUE,
            stem_width: 2.0,
            marker_radius: 4.0,
            marker_shape: "circle".into(),
            show_grid: true,
        }
    }
}

impl StemConfig {
    /// Create a new stem config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set stem color.
    pub fn with_stem_color(mut self, color: Color) -> Self {
        self.stem_color = color;
        self
    }

    /// Set marker color.
    pub fn with_marker_color(mut self, color: Color) -> Self {
        self.marker_color = color;
        self
    }

    /// Set marker radius.
    pub fn with_marker_radius(mut self, radius: f64) -> Self {
        self.marker_radius = radius;
        self
    }
}

/// Render a stem plot as SVG.
pub fn render_stem_plot(points: &[DataPoint], config: &StemConfig) -> PlotResult<String> {
    if points.is_empty() {
        return Err(PlotError::InvalidData("no points provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Find bounds
    let min_x = points.iter().map(|p| p.x).fold(f64::MAX, f64::min);
    let max_x = points.iter().map(|p| p.x).fold(f64::MIN, f64::max);
    let all_y: Vec<f64> = points.iter().map(|p| p.y).chain(std::iter::once(config.baseline)).collect();
    let min_y = all_y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_y = all_y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if min_x == max_x || min_y == max_y {
        return Err(PlotError::InvalidData("insufficient data range".into()));
    }

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;

    let to_x = |x| padding + (x - min_x) / (max_x - min_x) * chart_width;
    let to_y = |y| padding + 30.0 + chart_height * (1.0 - (y - min_y) / (max_y - min_y));

    let baseline_y = to_y(config.baseline);

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

    // Baseline
    {
        let x_right = width - padding;
        svg.push_str("  <line x1=\"");
        svg.push_str(&padding.to_string());
        svg.push_str("\" y1=\"");
        svg.push_str(&baseline_y.to_string());
        svg.push_str("\" x2=\"");
        svg.push_str(&x_right.to_string());
        svg.push_str("\" y2=\"");
        svg.push_str(&baseline_y.to_string());
        svg.push_str("\" stroke=\"gray\" stroke-dasharray=\"4\"/>\n");
    }
    svg.push('\n');

    // Stems and markers
    for p in points {
        let x = to_x(p.x);
        let y = to_y(p.y);

        // Stem line
        svg.push_str(&format!(
            r#"  <line x1="{x}" y1="{baseline_y}" x2="{x}" y2="{y}" stroke="{}" stroke-width="{}"/>"#,
            config.stem_color.to_hex(),
            config.stem_width
        ));
        svg.push('\n');

        // Marker
        match config.marker_shape.as_str() {
            "square" => {
                let r = config.marker_radius;
                svg.push_str(&format!(
                    r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
                    x - r, y - r, r * 2.0, r * 2.0,
                    config.marker_color.to_hex()
                ));
            }
            "diamond" => {
                let r = config.marker_radius;
                svg.push_str(&format!(
                    r#"  <polygon points="{},{} {},{} {},{} {},{}" fill="{}"/>"#,
                    x, y - r,
                    x + r, y,
                    x, y + r,
                    x - r, y,
                    config.marker_color.to_hex()
                ));
            }
            _ => {
                // Circle (default)
                svg.push_str(&format!(
                    r#"  <circle cx="{x}" cy="{y}" r="{}" fill="{}"/>"#,
                    config.marker_radius,
                    config.marker_color.to_hex()
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
    fn stem_plot_renders() {
        let points = vec![
            DataPoint::new(0.0, 5.0),
            DataPoint::new(1.0, -3.0),
            DataPoint::new(2.0, 8.0),
            DataPoint::new(3.0, 2.0),
        ];
        let config = StemConfig::new();
        let svg = render_stem_plot(&points, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<line"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn stem_plot_empty_error() {
        let points = vec![];
        let config = StemConfig::new();
        assert!(render_stem_plot(&points, &config).is_err());
    }

    #[test]
    fn stem_plot_diamond_markers() {
        let points = vec![DataPoint::new(0.0, 1.0), DataPoint::new(1.0, 2.0)];
        let config = StemConfig::new().with_marker_radius(5.0);
        let mut config = config;
        config.marker_shape = "diamond".into();
        let svg = render_stem_plot(&points, &config).unwrap();
        assert!(svg.contains("<polygon"));
    }
}
