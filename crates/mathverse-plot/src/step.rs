//! Step plot rendering.

use crate::common::{DataPoint, PlotConfig};
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Step position type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepPosition {
    /// Step before the point (pre).
    Before,
    /// Step after the point (post).
    After,
    /// Step in the middle (mid).
    Mid,
}

/// Configuration for a step plot.
#[derive(Debug, Clone)]
pub struct StepConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Step position.
    pub position: StepPosition,
    /// Line color.
    pub color: Color,
    /// Line width.
    pub line_width: f64,
    /// Show markers.
    pub show_markers: bool,
    /// Marker radius.
    pub marker_radius: f64,
    /// Show grid.
    pub show_grid: bool,
}

impl Default for StepConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            position: StepPosition::Before,
            color: Color::BLUE,
            line_width: 2.0,
            show_markers: true,
            marker_radius: 3.0,
            show_grid: true,
        }
    }
}

impl StepConfig {
    /// Create a new step config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set step position.
    pub fn with_position(mut self, pos: StepPosition) -> Self {
        self.position = pos;
        self
    }

    /// Set line color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

/// Render a step plot as SVG.
pub fn render_step_plot(points: &[DataPoint], config: &StepConfig) -> PlotResult<String> {
    if points.is_empty() {
        return Err(PlotError::InvalidData("no points provided".into()));
    }
    if points.len() < 2 {
        return Err(PlotError::InvalidData("need at least 2 points".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    let min_x = points.iter().map(|p| p.x).fold(f64::MAX, f64::min);
    let max_x = points.iter().map(|p| p.x).fold(f64::MIN, f64::max);
    let min_y = points.iter().map(|p| p.y).fold(f64::MAX, f64::min);
    let max_y = points.iter().map(|p| p.y).fold(f64::MIN, f64::max);

    if min_x == max_x || min_y == max_y {
        return Err(PlotError::InvalidData("insufficient data range".into()));
    }

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;

    let to_x = |x| padding + (x - min_x) / (max_x - min_x) * chart_width;
    let to_y = |y| padding + 30.0 + chart_height * (1.0 - (y - min_y) / (max_y - min_y));

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

    // Build step path
    let mut path = String::from("M");

    for (i, p) in points.iter().enumerate() {
        let x = to_x(p.x);
        let y = to_y(p.y);

        if i == 0 {
            path.push_str(&format!(" {x},{y}"));
            continue;
        }

        let prev = &points[i - 1];
        let prev_x = to_x(prev.x);
        let prev_y = to_y(prev.y);

        match config.position {
            StepPosition::Before => {
                // Horizontal from prev, then vertical to current
                path.push_str(&format!(" L {x},{prev_y} L {x},{y}"));
            }
            StepPosition::After => {
                // Vertical from prev, then horizontal to current
                path.push_str(&format!(" L {prev_x},{y} L {x},{y}"));
            }
            StepPosition::Mid => {
                let mid_x = (prev_x + x) / 2.0;
                path.push_str(&format!(" L {mid_x},{prev_y} L {mid_x},{y} L {x},{y}"));
            }
        }
    }

    svg.push_str(&format!(
        r#"  <path d="{path}" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"/>"#,
        config.color.to_hex(),
        config.line_width
    ));
    svg.push('\n');

    // Markers
    if config.show_markers {
        for p in points {
            let x = to_x(p.x);
            let y = to_y(p.y);
            svg.push_str(&format!(
                r#"  <circle cx="{x}" cy="{y}" r="{}" fill="{}" stroke="white" stroke-width="1"/>"#,
                config.marker_radius,
                config.color.to_hex()
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
    fn step_plot_renders() {
        let points = vec![
            DataPoint::new(0.0, 10.0),
            DataPoint::new(1.0, 25.0),
            DataPoint::new(2.0, 15.0),
            DataPoint::new(3.0, 30.0),
        ];
        let config = StepConfig::new();
        let svg = render_step_plot(&points, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn step_plot_after_position() {
        let points = vec![DataPoint::new(0.0, 1.0), DataPoint::new(1.0, 2.0)];
        let config = StepConfig::new().with_position(StepPosition::After);
        let svg = render_step_plot(&points, &config).unwrap();
        assert!(svg.contains("<path"));
    }

    #[test]
    fn step_plot_too_few_points() {
        let points = vec![DataPoint::new(0.0, 1.0)];
        let config = StepConfig::new();
        assert!(render_step_plot(&points, &config).is_err());
    }
}
