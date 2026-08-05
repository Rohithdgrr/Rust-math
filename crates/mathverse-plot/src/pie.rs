//! Pie chart rendering.

use std::f64::consts::PI;

use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single slice of a pie chart.
#[derive(Debug, Clone)]
pub struct PieSlice {
    /// Label for the slice.
    pub label: String,
    /// Numeric value (proportional to slice size).
    pub value: f64,
    /// Fill color.
    pub color: Color,
    /// Whether to explode (pull out) this slice.
    pub explode: f64,
}

impl PieSlice {
    /// Create a new pie slice.
    pub fn new(label: impl Into<String>, value: f64, color: Color) -> Self {
        Self {
            label: label.into(),
            value,
            color,
            explode: 0.0,
        }
    }

    /// Set explode distance.
    pub fn with_explode(mut self, distance: f64) -> Self {
        self.explode = distance;
        self
    }
}

/// Configuration for a pie chart.
#[derive(Debug, Clone)]
pub struct PieConfig {
    /// Chart title.
    pub title: String,
    /// Chart width in pixels.
    pub width: u32,
    /// Chart height in pixels.
    pub height: u32,
    /// Radius of the pie (pixels).
    pub radius: f64,
    /// Center X position.
    pub center_x: f64,
    /// Center Y position.
    pub center_y: f64,
    /// Start angle in radians (0 = 3 o'clock, PI/2 = 6 o'clock).
    pub start_angle: f64,
    /// Whether to sort slices by size.
    pub sort_slices: bool,
    /// Whether to show percentages.
    pub show_percentages: bool,
    /// Whether to show values.
    pub show_values: bool,
    /// Whether to show labels.
    pub show_labels: bool,
    /// Font size for labels.
    pub font_size: f64,
    /// Distance of labels from center (as fraction of radius).
    pub label_distance: f64,
    /// Shadow effect.
    pub shadow: bool,
}

impl Default for PieConfig {
    fn default() -> Self {
        Self {
            title: String::new(),
            width: 600,
            height: 400,
            radius: 150.0,
            center_x: 300.0,
            center_y: 200.0,
            start_angle: -PI / 2.0, // Start at 12 o'clock
            sort_slices: false,
            show_percentages: true,
            show_values: false,
            show_labels: true,
            font_size: 12.0,
            label_distance: 1.3,
            shadow: false,
        }
    }
}

impl PieConfig {
    /// Create a new pie config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the dimensions.
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the radius.
    pub fn with_radius(mut self, radius: f64) -> Self {
        self.radius = radius;
        self
    }

    /// Set the center position.
    pub fn with_center(mut self, x: f64, y: f64) -> Self {
        self.center_x = x;
        self.center_y = y;
        self
    }

    /// Set the start angle.
    pub fn with_start_angle(mut self, angle: f64) -> Self {
        self.start_angle = angle;
        self
    }

    /// Sort slices by size.
    pub fn with_sort(mut self) -> Self {
        self.sort_slices = true;
        self
    }

    /// Show percentages on slices.
    pub fn with_percentages(mut self) -> Self {
        self.show_percentages = true;
        self
    }

    /// Show values on slices.
    pub fn with_values(mut self) -> Self {
        self.show_values = true;
        self
    }

    /// Hide labels.
    pub fn without_labels(mut self) -> Self {
        self.show_labels = false;
        self
    }

    /// Set font size.
    pub fn with_font_size(mut self, size: f64) -> Self {
        self.font_size = size;
        self
    }
}

/// Render a pie chart as SVG.
pub fn render_pie_chart(slices: &[PieSlice], config: &PieConfig) -> PlotResult<String> {
    if slices.is_empty() {
        return Err(PlotError::InvalidData("no slices provided".into()));
    }

    let total: f64 = slices.iter().map(|s| s.value).sum();
    if total <= 0.0 {
        return Err(PlotError::InvalidData(
            "total value must be positive".into(),
        ));
    }

    let mut svg = String::new();

    // SVG header
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
        config.width, config.height
    ));
    svg.push('\n');

    // Background
    svg.push_str(&format!(
        r#"  <rect width="100%" height="100%" fill="white"/>"#
    ));
    svg.push('\n');

    // Shadow effect
    if config.shadow {
        svg.push_str(&format!(
            r#"  <defs><filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="3" dy="3" stdDeviation="3" flood-opacity="0.3"/>
    </filter></defs>"#
        ));
        svg.push('\n');
    }

    // Sort slices if requested
    let mut slices: Vec<&PieSlice> = slices.iter().collect();
    if config.sort_slices {
        slices.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Draw slices
    let mut current_angle = config.start_angle;
    for slice in &slices {
        let fraction = slice.value / total;
        let angle = fraction * 2.0 * PI;

        // Calculate arc path
        let end_angle = current_angle + angle;
        let large_arc = if angle > PI { 1 } else { 0 };

        // Explode offset
        let mid_angle = current_angle + angle / 2.0;
        let explode_x = slice.explode * mid_angle.cos();
        let explode_y = slice.explode * mid_angle.sin();

        let cx = config.center_x + explode_x;
        let cy = config.center_y + explode_y;

        // Arc points
        let x1 = cx + config.radius * current_angle.cos();
        let y1 = cy + config.radius * current_angle.sin();
        let x2 = cx + config.radius * end_angle.cos();
        let y2 = cy + config.radius * end_angle.sin();

        // SVG arc path
        svg.push_str("  <path d=\"M ");
        svg.push_str(&cx.to_string());
        svg.push_str(",");
        svg.push_str(&cy.to_string());
        svg.push_str(" L ");
        svg.push_str(&x1.to_string());
        svg.push_str(",");
        svg.push_str(&y1.to_string());
        svg.push_str(" A ");
        svg.push_str(&config.radius.to_string());
        svg.push_str(",");
        svg.push_str(&config.radius.to_string());
        svg.push_str(" 0 ");
        svg.push_str(&large_arc.to_string());
        svg.push_str(",1 ");
        svg.push_str(&x2.to_string());
        svg.push_str(",");
        svg.push_str(&y2.to_string());
        svg.push_str(" Z\" fill=\"");
        svg.push_str(&slice.color.to_hex());
        svg.push_str("\" stroke=\"white\" stroke-width=\"2\"/>\n");
        svg.push('\n');

        current_angle = end_angle;
    }

    // Draw labels
    if config.show_labels {
        current_angle = config.start_angle;
        for slice in &slices {
            let fraction = slice.value / total;
            let angle = fraction * 2.0 * PI;
            let mid_angle = current_angle + angle / 2.0;

            let label_r = config.radius * config.label_distance;
            let lx = config.center_x + label_r * mid_angle.cos();
            let ly = config.center_y + label_r * mid_angle.sin();

            let mut label_text = slice.label.clone();
            if config.show_percentages {
                let pct = fraction * 100.0;
                if config.show_values {
                    label_text = format!("{}: {:.1} ({:.1}%)", slice.label, slice.value, pct);
                } else {
                    label_text = format!("{}: {:.1}%", slice.label, pct);
                }
            } else if config.show_values {
                label_text = format!("{}: {:.1}", slice.label, slice.value);
            }

            let text_anchor = if mid_angle.cos() > 0.0 {
                "start"
            } else {
                "end"
            };

            svg.push_str(&format!(
                r#"  <text x="{lx}" y="{ly}" text-anchor="{text_anchor}" font-size="{}" dominant-baseline="middle">{}</text>"#,
                config.font_size, label_text
            ));
            svg.push('\n');

            current_angle += angle;
        }
    }

    // Title
    if !config.title.is_empty() {
        svg.push_str(&format!(
            r#"  <text x="{}" y="30" text-anchor="middle" font-size="20" font-weight="bold">{}</text>"#,
            config.width as f64 / 2.0, config.title
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
    fn pie_chart_renders_svg() {
        let slices = vec![
            PieSlice::new("A", 30.0, Color::RED),
            PieSlice::new("B", 20.0, Color::BLUE),
            PieSlice::new("C", 50.0, Color::GREEN),
        ];
        let config = PieConfig::new().with_title("Test Pie");
        let svg = render_pie_chart(&slices, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Test Pie"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn pie_chart_empty_error() {
        let slices = vec![];
        let config = PieConfig::new();
        assert!(render_pie_chart(&slices, &config).is_err());
    }

    #[test]
    fn pie_chart_zero_total_error() {
        let slices = vec![PieSlice::new("A", 0.0, Color::RED)];
        let config = PieConfig::new();
        assert!(render_pie_chart(&slices, &config).is_err());
    }

    #[test]
    fn pie_chart_with_explode() {
        let slices = vec![
            PieSlice::new("A", 50.0, Color::RED).with_explode(10.0),
            PieSlice::new("B", 50.0, Color::BLUE),
        ];
        let config = PieConfig::new();
        let svg = render_pie_chart(&slices, &config).unwrap();
        assert!(svg.contains("<path"));
    }
}
