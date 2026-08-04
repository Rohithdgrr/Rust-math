//! Bubble chart rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single bubble in a bubble chart.
#[derive(Debug, Clone)]
pub struct Bubble {
    /// X position.
    pub x: f64,
    /// Y position.
    pub y: f64,
    /// Bubble size (radius in pixels, or scaled).
    pub size: f64,
    /// Fill color.
    pub color: Color,
    /// Optional label.
    pub label: Option<String>,
}

impl Bubble {
    /// Create a new bubble.
    pub fn new(x: f64, y: f64, size: f64, color: Color) -> Self {
        Self {
            x,
            y,
            size,
            color,
            label: None,
        }
    }

    /// Set label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Configuration for a bubble chart.
#[derive(Debug, Clone)]
pub struct BubbleConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Scale factor for bubble sizes.
    pub size_scale: f64,
    /// Maximum bubble radius (pixels).
    pub max_radius: f64,
    /// Show labels on bubbles.
    pub show_labels: bool,
    /// Show grid.
    pub show_grid: bool,
    /// Font size.
    pub font_size: f64,
}

impl Default for BubbleConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            size_scale: 1.0,
            max_radius: 30.0,
            show_labels: true,
            show_grid: true,
            font_size: 10.0,
        }
    }
}

impl BubbleConfig {
    /// Create a new bubble config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set size scale.
    pub fn with_size_scale(mut self, scale: f64) -> Self {
        self.size_scale = scale;
        self
    }
}

/// Render a bubble chart as SVG.
pub fn render_bubble_chart(bubbles: &[Bubble], config: &BubbleConfig) -> PlotResult<String> {
    if bubbles.is_empty() {
        return Err(PlotError::InvalidData("no bubbles provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    let min_x = bubbles.iter().map(|b| b.x).fold(f64::MAX, f64::min);
    let max_x = bubbles.iter().map(|b| b.x).fold(f64::MIN, f64::max);
    let min_y = bubbles.iter().map(|b| b.y).fold(f64::MAX, f64::min);
    let max_y = bubbles.iter().map(|b| b.y).fold(f64::MIN, f64::max);
    let max_size = bubbles.iter().map(|b| b.size).fold(0.0_f64, f64::max);

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
        let y_bottom = height - padding;
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
        for i in 0..=5 {
            let x = padding + (i as f64 / 5.0) * chart_width;
            svg.push_str("  <line x1=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&padding.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y_bottom.to_string());
            svg.push_str("\" stroke=\"#eee\"/>\n");
        }
    }

    // Draw bubbles
    for b in bubbles {
        let cx = to_x(b.x);
        let cy = to_y(b.y);
        let r = (b.size / max_size) * config.max_radius * config.size_scale;

        svg.push_str(&format!(
            r#"  <circle cx="{cx}" cy="{cy}" r="{r}" fill="{}" fill-opacity="0.6" stroke="{}" stroke-width="1"/>"#,
            b.color.to_hex(),
            b.color.to_hex()
        ));
        svg.push('\n');

        // Label
        if config.show_labels {
            if let Some(label) = &b.label {
                svg.push_str(&format!(
                    r#"  <text x="{cx}" y="{}" text-anchor="middle" font-size="{}" dominant-baseline="middle">{}</text>"#,
                    cy - r - 5.0,
                    config.font_size,
                    label
                ));
                svg.push('\n');
            }
        }
    }

    // Axes
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="11">x</text>"#,
        width / 2.0, height - 5.0
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r#"  <text x="10" y="{}" text-anchor="middle" font-size="11" transform="rotate(-90, 10, {})">y</text>"#,
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
    fn bubble_chart_renders() {
        let bubbles = vec![
            Bubble::new(1.0, 2.0, 10.0, Color::RED).with_label("A"),
            Bubble::new(3.0, 4.0, 20.0, Color::BLUE).with_label("B"),
            Bubble::new(2.0, 1.0, 15.0, Color::GREEN),
        ];
        let config = BubbleConfig::new();
        let svg = render_bubble_chart(&bubbles, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn bubble_chart_empty_error() {
        let bubbles = vec![];
        let config = BubbleConfig::new();
        assert!(render_bubble_chart(&bubbles, &config).is_err());
    }
}
