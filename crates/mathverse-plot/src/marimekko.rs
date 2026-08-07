//! Marimekko (mosaic) chart rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A segment in a Marimekko chart.
#[derive(Debug, Clone)]
pub struct MarimekkoSegment {
    /// Label for the segment.
    pub label: String,
    /// Value for this segment.
    pub value: f64,
    /// Fill color.
    pub color: Color,
}

impl MarimekkoSegment {
    /// Create a new segment.
    pub fn new(label: impl Into<String>, value: f64, color: Color) -> Self {
        Self {
            label: label.into(),
            value,
            color,
        }
    }
}

/// A column in a Marimekko chart.
#[derive(Debug, Clone)]
pub struct MarimekkoColumn {
    /// Column label.
    pub label: String,
    /// Column width weight.
    pub width_weight: f64,
    /// Segments within this column.
    pub segments: Vec<MarimekkoSegment>,
}

impl MarimekkoColumn {
    /// Create a new column.
    pub fn new(label: impl Into<String>, width_weight: f64, segments: Vec<MarimekkoSegment>) -> Self {
        Self {
            label: label.into(),
            width_weight,
            segments,
        }
    }
}

/// Configuration for a Marimekko chart.
#[derive(Debug, Clone)]
pub struct MarimekkoConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Show values in segments.
    pub show_values: bool,
    /// Show labels.
    pub show_labels: bool,
    /// Font size.
    pub font_size: f64,
}

impl Default for MarimekkoConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            show_values: true,
            show_labels: true,
            font_size: 11.0,
        }
    }
}

impl MarimekkoConfig {
    /// Create a new Marimekko config.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Render a Marimekko chart as SVG.
pub fn render_marimekko(
    columns: &[MarimekkoColumn],
    config: &MarimekkoConfig,
) -> PlotResult<String> {
    if columns.is_empty() {
        return Err(PlotError::InvalidData("no columns provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    let total_weight: f64 = columns.iter().map(|c| c.width_weight).sum();
    if total_weight <= 0.0 {
        return Err(PlotError::InvalidData(
            "column width weights must sum to a positive value".into(),
        ));
    }
    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
        width as u32, height as u32
    ));
    svg.push('\n');
    svg.push_str(r#"  <rect width="100%" height="100%" fill="white"/>"#);
    svg.push('\n');

    let mut x_offset = padding;

    for col in columns {
        let col_width = (col.width_weight / total_weight) * chart_width;
        let total_value: f64 = col.segments.iter().map(|s| s.value).sum();
        if total_value <= 0.0 {
            return Err(PlotError::InvalidData(
                "each column's segment values must sum to a positive value".into(),
            ));
        }

        let mut y_offset = padding + 30.0;

        for seg in &col.segments {
            let seg_height = (seg.value / total_value) * chart_height;

            svg.push_str(&format!(
                r#"  <rect x="{x_offset}" y="{y_offset}" width="{col_width}" height="{seg_height}" fill="{}" stroke="white" stroke-width="2"/>"#,
                seg.color.to_hex()
            ));
            svg.push('\n');

            // Value label
            if config.show_values && seg_height > 15.0 {
                svg.push_str(&format!(
                    r#"  <text x="{}" y="{}" text-anchor="middle" font-size="{}" fill="white" dominant-baseline="middle">{:.1}</text>"#,
                    x_offset + col_width / 2.0,
                    y_offset + seg_height / 2.0,
                    config.font_size,
                    seg.value
                ));
                svg.push('\n');
            }

            y_offset += seg_height;
        }

        // Column label
        if config.show_labels {
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" text-anchor="middle" font-size="11">{}</text>"#,
                x_offset + col_width / 2.0,
                height - padding + 15.0,
                col.label
            ));
            svg.push('\n');
        }

        x_offset += col_width;
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
    fn marimekko_renders() {
        let columns = vec![
            MarimekkoColumn::new(
                "Company A",
                60.0,
                vec![
                    MarimekkoSegment::new("Product 1", 30.0, Color::BLUE),
                    MarimekkoSegment::new("Product 2", 20.0, Color::GREEN),
                ],
            ),
            MarimekkoColumn::new(
                "Company B",
                40.0,
                vec![
                    MarimekkoSegment::new("Product 1", 15.0, Color::RED),
                    MarimekkoSegment::new("Product 2", 25.0, Color::YELLOW),
                ],
            ),
        ];
        let config = MarimekkoConfig::new();
        let svg = render_marimekko(&columns, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn marimekko_empty_error() {
        let columns = vec![];
        let config = MarimekkoConfig::new();
        assert!(render_marimekko(&columns, &config).is_err());
    }

    #[test]
    fn marimekko_zero_weight_error() {
        let columns = vec![MarimekkoColumn::new(
            "A",
            0.0,
            vec![MarimekkoSegment::new("P1", 10.0, Color::BLUE)],
        )];
        let config = MarimekkoConfig::new();
        assert!(render_marimekko(&columns, &config).is_err());
    }

    #[test]
    fn marimekko_zero_value_error() {
        let columns = vec![MarimekkoColumn::new(
            "A",
            10.0,
            vec![MarimekkoSegment::new("P1", 0.0, Color::BLUE)],
        )];
        let config = MarimekkoConfig::new();
        assert!(render_marimekko(&columns, &config).is_err());
    }
}
