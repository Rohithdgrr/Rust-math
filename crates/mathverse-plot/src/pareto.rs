//! Pareto chart rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single bar in a Pareto chart.
#[derive(Debug, Clone)]
pub struct ParetoBar {
    /// Label for the bar.
    pub label: String,
    /// Value (frequency or count).
    pub value: f64,
}

impl ParetoBar {
    /// Create a new Pareto bar.
    pub fn new(label: impl Into<String>, value: f64) -> Self {
        Self {
            label: label.into(),
            value,
        }
    }
}

/// Configuration for a Pareto chart.
#[derive(Debug, Clone)]
pub struct ParetoConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Color for bars.
    pub bar_color: Color,
    /// Color for cumulative line.
    pub line_color: Color,
    /// Bar width (pixels).
    pub bar_width: f64,
    /// Show 80% threshold line.
    pub show_80_line: bool,
    /// Show cumulative percentage labels.
    pub show_cumulative: bool,
    /// Show grid.
    pub show_grid: bool,
    /// Font size.
    pub font_size: f64,
}

impl Default for ParetoConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            bar_color: Color::BLUE,
            line_color: Color::RED,
            bar_width: 40.0,
            show_80_line: true,
            show_cumulative: true,
            show_grid: true,
            font_size: 11.0,
        }
    }
}

impl ParetoConfig {
    /// Create a new Pareto config.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Render a Pareto chart as SVG.
pub fn render_pareto(bars: &[ParetoBar], config: &ParetoConfig) -> PlotResult<String> {
    if bars.is_empty() {
        return Err(PlotError::InvalidData("no bars provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Sort by value descending
    let mut sorted: Vec<&ParetoBar> = bars.iter().collect();
    sorted.sort_by(|a, b| b.value.partial_cmp(&a.value).unwrap_or(std::cmp::Ordering::Equal));

    let total: f64 = sorted.iter().map(|b| b.value).sum();
    let max_value = sorted[0].value;

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;
    let bar_spacing = config.bar_width * 0.2;
    let total_space = sorted.len() as f64 * (config.bar_width + bar_spacing);
    let start_x = padding + (chart_width - total_space) / 2.0;

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

    // Draw bars
    let mut cumulative = 0.0;
    let mut cum_points = Vec::new();

    for (i, bar) in sorted.iter().enumerate() {
        let x = start_x + i as f64 * (config.bar_width + bar_spacing);
        let bar_height = (bar.value / max_value) * chart_height;
        let y = padding + 30.0 + chart_height - bar_height;

        svg.push_str(&format!(
            r#"  <rect x="{x}" y="{y}" width="{}" height="{bar_height}" fill="{}" rx="2"/>"#,
            config.bar_width,
            config.bar_color.to_hex()
        ));
        svg.push('\n');

        // Value label
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" text-anchor="middle" font-size="{}">{:.0}</text>"#,
            x + config.bar_width / 2.0,
            y - 5.0,
            config.font_size - 1.0,
            bar.value
        ));
        svg.push('\n');

        // Category label
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" text-anchor="middle" font-size="10">{}</text>"#,
            x + config.bar_width / 2.0,
            height - padding + 15.0,
            bar.label
        ));
        svg.push('\n');

        // Cumulative
        cumulative += bar.value;
        let cum_pct = cumulative / total;
        let cx = x + config.bar_width / 2.0;
        let cy = padding + 30.0 + chart_height * (1.0 - cum_pct);
        cum_points.push((cx, cy));

        if config.show_cumulative {
            svg.push_str(&format!(
                r#"  <text x="{cx}" y="{}" text-anchor="middle" font-size="9" fill="red">{:.0}%</text>"#,
                cy - 8.0,
                cum_pct * 100.0
            ));
            svg.push('\n');
        }
    }

    // Cumulative line
    if cum_points.len() > 1 {
        let mut line_path = String::from("M");
        for (i, (x, y)) in cum_points.iter().enumerate() {
            if i == 0 {
                line_path.push_str(&format!(" {x},{y}"));
            } else {
                line_path.push_str(&format!(" L {x},{y}"));
            }
        }

        svg.push_str(&format!(
            r#"  <path d="{line_path}" fill="none" stroke="{}" stroke-width="2" stroke-dasharray="4"/>"#,
            config.line_color.to_hex()
        ));
        svg.push('\n');
    }

    // 80% threshold line
    if config.show_80_line {
        let y_80 = padding + 30.0 + chart_height * 0.2;
        let x_right = width - padding;
        svg.push_str("  <line x1=\"");
        svg.push_str(&padding.to_string());
        svg.push_str("\" y1=\"");
        svg.push_str(&y_80.to_string());
        svg.push_str("\" x2=\"");
        svg.push_str(&x_right.to_string());
        svg.push_str("\" y2=\"");
        svg.push_str(&y_80.to_string());
        svg.push_str("\" stroke=\"red\" stroke-dasharray=\"6\" stroke-width=\"1\"/>\n");
        svg.push('\n');
        svg.push_str(&format!(
            r#"  <text x="{}" y="{}" font-size="10" fill="red">80%</text>"#,
            width - padding + 5.0, y_80
        ));
        svg.push('\n');
    }

    // Right axis for cumulative %
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" text-anchor="start" font-size="10" fill="red" transform="rotate(-90, {}, {})">Cumulative %</text>"#,
        width - padding + 15.0, height / 2.0,
        width - padding + 15.0, height / 2.0
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
    fn pareto_renders() {
        let bars = vec![
            ParetoBar::new("Defect A", 45.0),
            ParetoBar::new("Defect B", 30.0),
            ParetoBar::new("Defect C", 15.0),
            ParetoBar::new("Defect D", 10.0),
        ];
        let config = ParetoConfig::new();
        let svg = render_pareto(&bars, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn pareto_empty_error() {
        let bars = vec![];
        let config = ParetoConfig::new();
        assert!(render_pareto(&bars, &config).is_err());
    }
}
