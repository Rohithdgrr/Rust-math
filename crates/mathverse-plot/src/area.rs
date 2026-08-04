//! Area plot (fill between) rendering.

use crate::common::{DataPoint, PlotConfig};
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A line series for area chart.
#[derive(Debug, Clone)]
pub struct AreaSeries {
    /// Series label.
    pub label: String,
    /// Data points.
    pub points: Vec<DataPoint>,
    /// Line color.
    pub color: Color,
}

impl AreaSeries {
    /// Create a new area series.
    pub fn new(label: impl Into<String>, points: Vec<DataPoint>, color: Color) -> Self {
        Self {
            label: label.into(),
            points,
            color,
        }
    }
}

/// Configuration for an area plot.
#[derive(Debug, Clone)]
pub struct AreaConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Fill opacity (0.0 to 1.0).
    pub fill_opacity: f64,
    /// Baseline Y value for fill.
    pub baseline: f64,
    /// Show edge line.
    pub show_line: bool,
    /// Show grid.
    pub show_grid: bool,
}

impl Default for AreaConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            fill_opacity: 0.3,
            baseline: 0.0,
            show_line: true,
            show_grid: true,
        }
    }
}

impl AreaConfig {
    /// Create a new area config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set fill opacity.
    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.fill_opacity = opacity;
        self
    }

    /// Set baseline.
    pub fn with_baseline(mut self, baseline: f64) -> Self {
        self.baseline = baseline;
        self
    }
}

/// Render an area chart as SVG.
pub fn render_area_chart(series: &[AreaSeries], config: &AreaConfig) -> PlotResult<String> {
    if series.is_empty() {
        return Err(PlotError::InvalidData("no series provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Find data bounds
    let mut min_x = f64::MAX;
    let mut max_x = f64::MIN;
    let mut min_y = config.baseline;
    let mut max_y = config.baseline;

    for s in series {
        for p in &s.points {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
    }

    if min_x == max_x || min_y == max_y {
        return Err(PlotError::InvalidData("insufficient data range".into()));
    }

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0; // Space for title
    let base_y = padding + 30.0 + chart_height * (1.0 - (config.baseline - min_y) / (max_y - min_y));

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

    // Draw areas
    for s in series {
        if s.points.len() < 2 {
            continue;
        }

        // Create fill path
        let mut fill_path = String::from("M");
        let mut first = true;

        // Down to baseline on left
        let first_x = padding + (s.points[0].x - min_x) / (max_x - min_x) * chart_width;
        fill_path.push_str(&format!(" {first_x},{base_y}"));

        for p in &s.points {
            let x = padding + (p.x - min_x) / (max_x - min_x) * chart_width;
            let y = padding + 30.0 + chart_height * (1.0 - (p.y - min_y) / (max_y - min_y));

            if first {
                fill_path.push_str(&format!(" L {x},{y}"));
                first = false;
            } else {
                fill_path.push_str(&format!(" L {x},{y}"));
            }
        }

        // Close path back to baseline
        let last_x = padding + (s.points.last().unwrap().x - min_x) / (max_x - min_x) * chart_width;
        fill_path.push_str(&format!(" L {last_x},{base_y} Z"));

        svg.push_str(&format!(
            r#"  <path d="{fill_path}" fill="{}" fill-opacity="{}"/>"#,
            s.color.to_hex(),
            config.fill_opacity
        ));
        svg.push('\n');

        // Line on top
        if config.show_line {
            let mut line_path = String::from("M");
            for (i, p) in s.points.iter().enumerate() {
                let x = padding + (p.x - min_x) / (max_x - min_x) * chart_width;
                let y = padding + 30.0 + chart_height * (1.0 - (p.y - min_y) / (max_y - min_y));
                if i == 0 {
                    line_path.push_str(&format!(" {x},{y}"));
                } else {
                    line_path.push_str(&format!(" L {x},{y}"));
                }
            }

            svg.push_str(&format!(
                r#"  <path d="{line_path}" fill="none" stroke="{}" stroke-width="2"/>"#,
                s.color.to_hex()
            ));
            svg.push('\n');
        }
    }

    // Baseline
    {
        let x_right = width - padding;
        svg.push_str("  <line x1=\"");
        svg.push_str(&padding.to_string());
        svg.push_str("\" y1=\"");
        svg.push_str(&base_y.to_string());
        svg.push_str("\" x2=\"");
        svg.push_str(&x_right.to_string());
        svg.push_str("\" y2=\"");
        svg.push_str(&base_y.to_string());
        svg.push_str("\" stroke=\"gray\" stroke-dasharray=\"4\"/>\n");
    }
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
    fn area_chart_renders() {
        let series = vec![AreaSeries::new(
            "Data",
            vec![
                DataPoint::new(0.0, 10.0),
                DataPoint::new(1.0, 25.0),
                DataPoint::new(2.0, 18.0),
                DataPoint::new(3.0, 30.0),
            ],
            Color::BLUE,
        )];
        let config = AreaConfig::new();
        let svg = render_area_chart(&series, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn area_chart_empty_error() {
        let series = vec![];
        let config = AreaConfig::new();
        assert!(render_area_chart(&series, &config).is_err());
    }
}
