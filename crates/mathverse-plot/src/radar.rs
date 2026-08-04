//! Radar (spider) chart rendering.

use std::f64::consts::PI;

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single radar data point.
#[derive(Debug, Clone)]
pub struct RadarPoint {
    /// Axis label.
    pub label: String,
    /// Value (0.0 to 1.0 normalized, or will be auto-normalized).
    pub value: f64,
}

impl RadarPoint {
    /// Create a new radar point.
    pub fn new(label: impl Into<String>, value: f64) -> Self {
        Self {
            label: label.into(),
            value,
        }
    }
}

/// A series of radar data.
#[derive(Debug, Clone)]
pub struct RadarSeries {
    /// Label for the series.
    pub label: String,
    /// Data points (one per axis).
    pub points: Vec<RadarPoint>,
    /// Fill color.
    pub color: Color,
    /// Fill opacity.
    pub fill_opacity: f64,
}

impl RadarSeries {
    /// Create a new radar series.
    pub fn new(label: impl Into<String>, points: Vec<RadarPoint>, color: Color) -> Self {
        Self {
            label: label.into(),
            points,
            color,
            fill_opacity: 0.2,
        }
    }

    /// Set fill opacity.
    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.fill_opacity = opacity;
        self
    }
}

/// Configuration for a radar chart.
#[derive(Debug, Clone)]
pub struct RadarConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Radius of the radar (pixels).
    pub radius: f64,
    /// Center X.
    pub center_x: f64,
    /// Center Y.
    pub center_y: f64,
    /// Number of grid rings.
    pub grid_rings: usize,
    /// Show axis lines.
    pub show_axis_lines: bool,
    /// Show grid.
    pub show_grid: bool,
    /// Show labels.
    pub show_labels: bool,
    /// Font size.
    pub font_size: f64,
}

impl Default for RadarConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            radius: 150.0,
            center_x: 300.0,
            center_y: 220.0,
            grid_rings: 5,
            show_axis_lines: true,
            show_grid: true,
            show_labels: true,
            font_size: 12.0,
        }
    }
}

impl RadarConfig {
    /// Create a new radar config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set radius.
    pub fn with_radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }

    /// Set center.
    pub fn with_center(mut self, x: f64, y: f64) -> Self {
        self.center_x = x;
        self.center_y = y;
        self
    }
}

/// Render a radar chart as SVG.
pub fn render_radar_chart(series: &[RadarSeries], config: &RadarConfig) -> PlotResult<String> {
    if series.is_empty() {
        return Err(PlotError::InvalidData("no series provided".into()));
    }

    let num_axes = series[0].points.len();
    if num_axes < 3 {
        return Err(PlotError::InvalidData("need at least 3 axes".into()));
    }

    // Verify all series have same number of points
    for s in series {
        if s.points.len() != num_axes {
            return Err(PlotError::InvalidData(
                "all series must have same number of points".into(),
            ));
        }
    }

    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;
    let angle_step = 2.0 * PI / num_axes as f64;

    // Find max value for normalization
    let max_value = series
        .iter()
        .flat_map(|s| &s.points)
        .map(|p| p.value)
        .fold(0.0_f64, f64::max);

    if max_value == 0.0 {
        return Err(PlotError::InvalidData("max value must be positive".into()));
    }

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
        width as u32, height as u32
    ));
    svg.push('\n');
    svg.push_str(r#"  <rect width="100%" height="100%" fill="white"/>"#);
    svg.push('\n');

    // Grid rings
    if config.show_grid {
        for ring in 1..=config.grid_rings {
            let r = config.radius * ring as f64 / config.grid_rings as f64;
            let mut polygon = String::from("M");
            for i in 0..num_axes {
                let angle = i as f64 * angle_step - PI / 2.0;
                let x = config.center_x + r * angle.cos();
                let y = config.center_y + r * angle.sin();
                if i == 0 {
                    polygon.push_str(&format!(" {x},{y}"));
                } else {
                    polygon.push_str(&format!(" L {x},{y}"));
                }
            }
            polygon.push_str(" Z");
            svg.push_str("  <path d=\"");
            svg.push_str(&polygon);
            svg.push_str("\" fill=\"none\" stroke=\"#ddd\"/>\n");
            svg.push('\n');
        }
    }

    // Axis lines
    if config.show_axis_lines {
        for i in 0..num_axes {
            let angle = i as f64 * angle_step - PI / 2.0;
            let x = config.center_x + config.radius * angle.cos();
            let y = config.center_y + config.radius * angle.sin();
            svg.push_str("  <line x1=\"");
            svg.push_str(&config.center_x.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&config.center_y.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" stroke=\"#ccc\"/>\n");
            svg.push('\n');
        }
    }

    // Labels
    if config.show_labels {
        for (i, label) in series[0].points.iter().map(|p| &p.label).enumerate() {
            let angle = i as f64 * angle_step - PI / 2.0;
            let label_r = config.radius + 20.0;
            let x = config.center_x + label_r * angle.cos();
            let y = config.center_y + label_r * angle.sin();

            let text_anchor = if angle.cos() < -0.1 {
                "end"
            } else if angle.cos() > 0.1 {
                "start"
            } else {
                "middle"
            };

            svg.push_str(&format!(
                r#"  <text x="{x}" y="{y}" text-anchor="{text_anchor}" font-size="{}" dominant-baseline="middle">{}</text>"#,
                config.font_size, label
            ));
            svg.push('\n');
        }
    }

    // Data polygons
    for s in series {
        let mut polygon = String::from("M");
        for (i, p) in s.points.iter().enumerate() {
            let angle = i as f64 * angle_step - PI / 2.0;
            let r = (p.value / max_value) * config.radius;
            let x = config.center_x + r * angle.cos();
            let y = config.center_y + r * angle.sin();
            if i == 0 {
                polygon.push_str(&format!(" {x},{y}"));
            } else {
                polygon.push_str(&format!(" L {x},{y}"));
            }
        }
        polygon.push_str(" Z");

        svg.push_str(&format!(
            r#"  <path d="{polygon}" fill="{}" fill-opacity="{}" stroke="{}" stroke-width="2"/>"#,
            s.color.to_hex(),
            s.fill_opacity,
            s.color.to_hex()
        ));
        svg.push('\n');

        // Data points
        for (i, p) in s.points.iter().enumerate() {
            let angle = i as f64 * angle_step - PI / 2.0;
            let r = (p.value / max_value) * config.radius;
            let x = config.center_x + r * angle.cos();
            let y = config.center_y + r * angle.sin();
            svg.push_str(&format!(
                r#"  <circle cx="{x}" cy="{y}" r="3" fill="{}"/>"#,
                s.color.to_hex()
            ));
            svg.push('\n');
        }
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
    fn radar_chart_renders() {
        let series = vec![RadarSeries::new(
            "Skills",
            vec![
                RadarPoint::new("Math", 0.9),
                RadarPoint::new("Science", 0.8),
                RadarPoint::new("Art", 0.6),
                RadarPoint::new("Music", 0.7),
                RadarPoint::new("Sports", 0.5),
            ],
            Color::BLUE,
        )];
        let config = RadarConfig::new();
        let svg = render_radar_chart(&series, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn radar_chart_empty_error() {
        let series = vec![];
        let config = RadarConfig::new();
        assert!(render_radar_chart(&series, &config).is_err());
    }
}
