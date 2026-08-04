//! Grouped bar chart rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A series of bars for grouped chart.
#[derive(Debug, Clone)]
pub struct GroupedSeries {
    /// Label for the series.
    pub label: String,
    /// Values for each category.
    pub values: Vec<f64>,
    /// Fill color.
    pub color: Color,
}

impl GroupedSeries {
    /// Create a new grouped series.
    pub fn new(label: impl Into<String>, values: Vec<f64>, color: Color) -> Self {
        Self {
            label: label.into(),
            values,
            color,
        }
    }
}

/// Configuration for a grouped bar chart.
#[derive(Debug, Clone)]
pub struct GroupedBarConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Category labels (x-axis).
    pub categories: Vec<String>,
    /// Width of each bar (pixels).
    pub bar_width: f64,
    /// Gap between categories (fraction of bar width).
    pub category_gap: f64,
    /// Show values on bars.
    pub show_values: bool,
    /// Font size.
    pub font_size: f64,
    /// Show legend.
    pub show_legend: bool,
}

impl Default for GroupedBarConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            categories: vec![],
            bar_width: 20.0,
            category_gap: 0.5,
            show_values: true,
            font_size: 11.0,
            show_legend: true,
        }
    }
}

impl GroupedBarConfig {
    /// Create a new config with category labels.
    pub fn new(categories: Vec<String>) -> Self {
        Self {
            categories,
            ..Default::default()
        }
    }

    /// Set bar width.
    pub fn with_bar_width(mut self, width: f64) -> Self {
        self.bar_width = width;
        self
    }
}

/// Render a grouped bar chart as SVG.
pub fn render_grouped_bar(
    series: &[GroupedSeries],
    config: &GroupedBarConfig,
) -> PlotResult<String> {
    if series.is_empty() {
        return Err(PlotError::InvalidData("no series provided".into()));
    }
    if config.categories.is_empty() {
        return Err(PlotError::InvalidData("no categories provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;
    let num_cats = config.categories.len();
    let num_series = series.len();

    // Find max value
    let max_value = series
        .iter()
        .flat_map(|s| &s.values)
        .fold(0.0_f64, |a, b| a.max(*b));

    // Chart area
    let label_area = 50.0;
    let chart_width = width - padding * 2.0 - label_area;
    let chart_height = height - padding * 2.0 - 30.0;

    let group_width = num_series as f64 * config.bar_width;
    let group_spacing = group_width * config.category_gap;
    let total_group_width = group_width + group_spacing;
    let total_space = num_cats as f64 * total_group_width;
    let start_x = label_area + (chart_width - total_space) / 2.0;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
        width as u32, height as u32
    ));
    svg.push('\n');
    svg.push_str(r#"  <rect width="100%" height="100%" fill="white"/>"#);
    svg.push('\n');

    // Grid lines
    let x_right = width - padding;
    for i in 0..=5 {
        let y = padding + 30.0 + (i as f64 / 5.0) * chart_height;
        let val = max_value * (1.0 - i as f64 / 5.0);
        svg.push_str("  <line x1=\"");
        svg.push_str(&label_area.to_string());
        svg.push_str("\" y1=\"");
        svg.push_str(&y.to_string());
        svg.push_str("\" x2=\"");
        svg.push_str(&x_right.to_string());
        svg.push_str("\" y2=\"");
        svg.push_str(&y.to_string());
        svg.push_str("\" stroke=\"#eee\"/>\n");
        svg.push_str("  <text x=\"");
        svg.push_str(&label_area.to_string());
        svg.push_str("\" y=\"");
        svg.push_str(&y.to_string());
        svg.push_str("\" text-anchor=\"end\" font-size=\"10\" dominant-baseline=\"middle\">");
        svg.push_str(&format!("{:.0}", val));
        svg.push_str("</text>\n");
    }

    // Draw grouped bars
    for cat_idx in 0..num_cats {
        let group_x = start_x + cat_idx as f64 * total_group_width;

        for (s_idx, s) in series.iter().enumerate() {
            let val = s.values.get(cat_idx).copied().unwrap_or(0.0);
            let bar_height = (val / max_value) * chart_height;
            let x = group_x + s_idx as f64 * config.bar_width;
            let y = padding + 30.0 + chart_height - bar_height;

            svg.push_str(&format!(
                r#"  <rect x="{x}" y="{y}" width="{}" height="{bar_height}" fill="{}" rx="2"/>"#,
                config.bar_width,
                s.color.to_hex()
            ));
            svg.push('\n');

            // Value label
            if config.show_values && bar_height > 12.0 {
                svg.push_str(&format!(
                    r#"  <text x="{}" y="{}" text-anchor="middle" font-size="{}" fill="black" dominant-baseline="middle">{:.1}</text>"#,
                    x + config.bar_width / 2.0,
                    y - 6.0,
                    config.font_size - 2.0,
                    val
                ));
                svg.push('\n');
            }
        }

        // Category label
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" text-anchor="middle" font-size="11">{}</text>"#,
            group_x + group_width / 2.0,
            height - padding + 15.0,
            config.categories[cat_idx]
        ));
        svg.push('\n');
    }

    // Legend
    if config.show_legend {
        let legend_x = width - padding - 100.0;
        let legend_y = padding + 30.0;
        for (i, s) in series.iter().enumerate() {
            let ly = legend_y + i as f64 * 20.0;
            svg.push_str(&format!(
                r#"  <rect x="{legend_x}" y="{ly}" width="12" height="12" fill="{}"/>"#,
                s.color.to_hex()
            ));
            svg.push('\n');
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" font-size="11" dominant-baseline="middle">{}</text>"#,
                legend_x + 16.0, ly + 6.0, s.label
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
    fn grouped_bar_renders() {
        let series = vec![
            GroupedSeries::new("2023", vec![20.0, 30.0], Color::BLUE),
            GroupedSeries::new("2024", vec![25.0, 35.0], Color::GREEN),
        ];
        let config = GroupedBarConfig::new(vec!["A".into(), "B".into()]);
        let svg = render_grouped_bar(&series, &config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn grouped_bar_empty_error() {
        let series = vec![];
        let config = GroupedBarConfig::new(vec!["A".into()]);
        assert!(render_grouped_bar(&series, &config).is_err());
    }
}
