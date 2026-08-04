//! Point plot (categorical point estimates with error bars).

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single category with its data points.
#[derive(Debug, Clone)]
pub struct PointCategory {
    /// Category label.
    pub label: String,
    /// Data values.
    pub values: Vec<f64>,
    /// Color for this category.
    pub color: Color,
}

impl PointCategory {
    /// Create a new point category.
    pub fn new(label: impl Into<String>, values: Vec<f64>, color: Color) -> Self {
        Self {
            label: label.into(),
            values,
            color,
        }
    }
}

/// Configuration for a point plot.
#[derive(Debug, Clone)]
pub struct PointConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Point size.
    pub point_size: f64,
    /// Line width.
    pub line_width: f64,
    /// Show error bars.
    pub show_error_bars: bool,
    /// Error bar cap size.
    pub cap_size: f64,
    /// Show grid.
    pub show_grid: bool,
    /// Show connecting line.
    pub show_line: bool,
}

impl Default for PointConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            point_size: 6.0,
            line_width: 2.0,
            show_error_bars: true,
            cap_size: 5.0,
            show_grid: true,
            show_line: true,
        }
    }
}

impl PointConfig {
    /// Create a new point config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set point size.
    pub fn with_point_size(mut self, size: f64) -> Self {
        self.point_size = size;
        self
    }
}

/// Render a point plot as SVG.
pub fn render_pointplot(categories: &[PointCategory], config: &PointConfig) -> PlotResult<String> {
    if categories.is_empty() {
        return Err(PlotError::InvalidData("no categories provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Compute means
    let means: Vec<f64> = categories.iter()
        .map(|c| c.values.iter().sum::<f64>() / c.values.len() as f64)
        .collect();

    // Compute standard errors
    let ses: Vec<f64> = categories.iter()
        .map(|c| {
            let n = c.values.len() as f64;
            let mean = c.values.iter().sum::<f64>() / n;
            let variance = c.values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (variance / n).sqrt()
        })
        .collect();

    let all_min = means.iter().zip(ses.iter())
        .map(|(&m, &s)| m - 2.0 * s)
        .fold(f64::INFINITY, |a, b| a.min(b));
    let all_max = means.iter().zip(ses.iter())
        .map(|(&m, &s)| m + 2.0 * s)
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;
    let y_range = all_max - all_min;

    let to_y = |v| padding + 30.0 + chart_height * (1.0 - (v - all_min) / y_range);

    let n = categories.len();
    let spacing = chart_width / (n + 1) as f64;

    let mut svg = String::new();
    svg.push_str("<svg width=\"");
    svg.push_str(&width.to_string());
    svg.push_str("\" height=\"");
    svg.push_str(&height.to_string());
    svg.push_str("\" xmlns=\"http://www.w3.org/2000/svg\">\n");
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    // Grid
    if config.show_grid {
        for i in 0..=5 {
            let y = padding + 30.0 + (i as f64 / 5.0) * chart_height;
            svg.push_str("  <line x1=\"");
            svg.push_str(&padding.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&(width - padding).to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y.to_string());
            svg.push_str("\" stroke=\"#eee\"/>\n");
        }
    }

    // Connecting line
    if config.show_line && n > 1 {
        let mut path = String::new();
        for (i, &mean) in means.iter().enumerate() {
            let x = padding + spacing * (i + 1) as f64;
            let y = to_y(mean);
            if i == 0 {
                path.push_str(&format!("{x},{y}"));
            } else {
                path.push_str(&format!(" L {x},{y}"));
            }
        }

        svg.push_str("  <polyline points=\"");
        svg.push_str(&path);
        svg.push_str("\" fill=\"none\" stroke=\"black\" stroke-width=\"");
        svg.push_str(&config.line_width.to_string());
        svg.push_str("\"/>\n");
    }

    // Error bars and points
    for (i, cat) in categories.iter().enumerate() {
        let x = padding + spacing * (i + 1) as f64;
        let mean = means[i];
        let se = ses[i];
        let y = to_y(mean);

        // Error bars
        if config.show_error_bars {
            let y_upper = to_y(mean + 2.0 * se);
            let y_lower = to_y(mean - 2.0 * se);

            // Vertical line
            svg.push_str("  <line x1=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&y_upper.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y_lower.to_string());
            svg.push_str("\" stroke=\"");
            svg.push_str(&cat.color.to_hex());
            svg.push_str("\" stroke-width=\"1\"/>\n");

            // Caps
            svg.push_str("  <line x1=\"");
            svg.push_str(&(x - config.cap_size).to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&y_upper.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&(x + config.cap_size).to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y_upper.to_string());
            svg.push_str("\" stroke=\"");
            svg.push_str(&cat.color.to_hex());
            svg.push_str("\" stroke-width=\"1\"/>\n");

            svg.push_str("  <line x1=\"");
            svg.push_str(&(x - config.cap_size).to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&y_lower.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&(x + config.cap_size).to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y_lower.to_string());
            svg.push_str("\" stroke=\"");
            svg.push_str(&cat.color.to_hex());
            svg.push_str("\" stroke-width=\"1\"/>\n");
        }

        // Point
        svg.push_str("  <circle cx=\"");
        svg.push_str(&x.to_string());
        svg.push_str("\" cy=\"");
        svg.push_str(&y.to_string());
        svg.push_str("\" r=\"");
        svg.push_str(&config.point_size.to_string());
        svg.push_str("\" fill=\"");
        svg.push_str(&cat.color.to_hex());
        svg.push_str("\"/>\n");

        // Category label
        svg.push_str("  <text x=\"");
        svg.push_str(&x.to_string());
        svg.push_str("\" y=\"");
        svg.push_str(&(height - padding + 15.0).to_string());
        svg.push_str("\" text-anchor=\"middle\" font-size=\"11\">");
        svg.push_str(&cat.label);
        svg.push_str("</text>\n");
    }

    // Title
    if !config.plot_config.title.is_empty() {
        svg.push_str("  <text x=\"");
        svg.push_str(&(width / 2.0).to_string());
        svg.push_str("\" y=\"25\" text-anchor=\"middle\" font-size=\"20\" font-weight=\"bold\">");
        svg.push_str(&config.plot_config.title);
        svg.push_str("</text>\n");
    }

    svg.push_str("</svg>");
    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointplot_renders() {
        let cats = vec![
            PointCategory::new("A", vec![1.0, 2.0, 3.0, 4.0], Color::BLUE),
            PointCategory::new("B", vec![2.0, 3.0, 4.0, 5.0], Color::GREEN),
            PointCategory::new("C", vec![3.0, 4.0, 5.0, 6.0], Color::RED),
        ];
        let config = PointConfig::new();
        let svg = render_pointplot(&cats, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<line"));
    }

    #[test]
    fn pointplot_empty_error() {
        let cats = vec![];
        let config = PointConfig::new();
        assert!(render_pointplot(&cats, &config).is_err());
    }
}
