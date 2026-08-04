//! Automatic colorbar for heatmaps and colormapped data.

use crate::axes::{Range, Scale};
use crate::heatmap::Colormap;
use crate::style::Color;

/// Configuration for a colorbar.
#[derive(Debug, Clone)]
pub struct ColorbarConfig {
    /// Width of the colorbar in pixels.
    pub width: f64,
    /// Height of the colorbar in pixels.
    pub height: f64,
    /// Number of color stops to render.
    pub num_stops: usize,
    /// Number of tick marks.
    pub tick_count: usize,
    /// Font size for tick labels.
    pub font_size: f64,
    /// Optional title for the colorbar.
    pub title: Option<String>,
}

impl Default for ColorbarConfig {
    fn default() -> Self {
        Self {
            width: 20.0,
            height: 200.0,
            num_stops: 64,
            tick_count: 5,
            font_size: 10.0,
            title: None,
        }
    }
}

impl ColorbarConfig {
    /// Create a new colorbar config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the dimensions.
    pub fn with_dimensions(mut self, width: f64, height: f64) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the number of color stops.
    pub fn with_num_stops(mut self, stops: usize) -> Self {
        self.num_stops = stops.max(2);
        self
    }

    /// Set the number of tick marks.
    pub fn with_tick_count(mut self, count: usize) -> Self {
        self.tick_count = count.max(1);
        self
    }

    /// Set the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

/// Render a colorbar as SVG.
///
/// `x`, `y` are the top-left corner of the colorbar in the parent SVG.
/// `data_min`, `data_max` are the data range that the colormap maps from/to.
pub fn render_colorbar(
    x: f64,
    y: f64,
    data_min: f64,
    data_max: f64,
    colormap: Colormap,
    config: &ColorbarConfig,
) -> String {
    let mut svg = String::new();

    // Draw the gradient bar
    let stop_height = config.height / config.num_stops as f64;
    for i in 0..config.num_stops {
        let t = i as f64 / (config.num_stops - 1) as f64;
        let color = colormap(t);
        let sy = y + config.height - (i as f64 + 1.0) * stop_height;
        svg.push_str(&format!(
            r#"  <rect x="{x}" y="{sy}" width="{}" height="{stop_height}" fill="{}"/>"#,
            config.width,
            color.to_hex()
        ));
        svg.push('\n');
    }

    // Draw border
    svg.push_str(&format!(
        r#"  <rect x="{x}" y="{y}" width="{}" height="{}" fill="none" stroke="black" stroke-width="1"/>"#,
        config.width, config.height
    ));
    svg.push('\n');

    // Draw tick marks and labels
    let data_range = Range {
        min: data_min,
        max: data_max,
    };
    let ticks = Scale::Linear.ticks(data_min, data_max, config.tick_count);

    for tick_value in &ticks {
        let t = (*tick_value - data_min) / (data_max - data_min);
        let ty = y + config.height * (1.0 - t);
        let label = format_tick_label(*tick_value);

        // Tick mark
        svg.push_str(&format!(
            r#"  <line x1="{}" y1="{ty}" x2="{}" y2="{ty}" stroke="black" stroke-width="1"/>"#,
            x + config.width,
            x + config.width + 4.0
        ));
        svg.push('\n');

        // Tick label
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" text-anchor="start" font-size="{}">{}</text>"#,
            x + config.width + 6.0,
            ty + config.font_size * 0.35,
            config.font_size,
            label
        ));
        svg.push('\n');
    }

    // Optional title
    if let Some(title) = &config.title {
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" text-anchor="middle" font-size="{}">{}</text>"#,
            x + config.width / 2.0,
            y - 8.0,
            config.font_size + 2.0,
            title
        ));
        svg.push('\n');
    }

    svg
}

/// Format a tick label with appropriate precision.
fn format_tick_label(value: f64) -> String {
    if value.abs() >= 1e6 || (value.abs() < 1e-3 && value != 0.0) {
        format!("{:.2e}", value)
    } else if value == value.round() {
        format!("{}", value as i64)
    } else {
        format!("{:.2}", value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::viridis;

    #[test]
    fn colorbar_renders_svg() {
        let config = ColorbarConfig::new().with_dimensions(20.0, 100.0);
        let svg = render_colorbar(0.0, 0.0, 0.0, 1.0, viridis, &config);
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<text"));
    }

    #[test]
    fn colorbar_with_title() {
        let config = ColorbarConfig::new().with_title("Value");
        let svg = render_colorbar(0.0, 0.0, 0.0, 100.0, viridis, &config);
        assert!(svg.contains("Value"));
    }
}
