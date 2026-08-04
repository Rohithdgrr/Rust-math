//! Residual plot (residuals vs fitted values).

use crate::common::{DataPoint, PlotConfig};
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Configuration for a residual plot.
#[derive(Debug, Clone)]
pub struct ResidConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Point color.
    pub point_color: Color,
    /// Zero line color.
    pub zero_line_color: Color,
    /// Point size.
    pub point_size: f64,
    /// Show grid.
    pub show_grid: bool,
    /// Show loess smoother.
    pub show_smoother: bool,
}

impl Default for ResidConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            point_color: Color::BLUE,
            zero_line_color: Color::RED,
            point_size: 4.0,
            show_grid: true,
            show_smoother: false,
        }
    }
}

impl ResidConfig {
    /// Create a new residual config.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Compute linear regression parameters.
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        ss_xy += (xi - mean_x) * (yi - mean_y);
        ss_xx += (xi - mean_x).powi(2);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    (slope, intercept)
}

/// Render a residual plot as SVG.
pub fn render_residplot(points: &[DataPoint], config: &ResidConfig) -> PlotResult<String> {
    if points.is_empty() {
        return Err(PlotError::InvalidData("no points provided".into()));
    }
    if points.len() < 2 {
        return Err(PlotError::InvalidData("need at least 2 points".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    let x: Vec<f64> = points.iter().map(|p| p.x).collect();
    let y: Vec<f64> = points.iter().map(|p| p.y).collect();

    // Fit regression
    let (slope, intercept) = linear_regression(&x, &y);

    // Compute residuals and fitted values
    let fitted: Vec<f64> = x.iter().map(|&xi| slope * xi + intercept).collect();
    let residuals: Vec<f64> = y.iter().zip(fitted.iter())
        .map(|(&yi, &fi)| yi - fi)
        .collect();

    let fitted_min = fitted.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let fitted_max = fitted.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let resid_min = residuals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let resid_max = residuals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;

    let range_x = fitted_max - fitted_min;
    let range_y = resid_max - resid_min;

    let to_x = |v| padding + (v - fitted_min) / range_x * chart_width;
    let to_y = |v| padding + 30.0 + chart_height * (1.0 - (v - resid_min) / range_y);

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
            let gy = padding + 30.0 + (i as f64 / 5.0) * chart_height;
            svg.push_str("  <line x1=\"");
            svg.push_str(&padding.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&gy.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&(width - padding).to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&gy.to_string());
            svg.push_str("\" stroke=\"#eee\"/>\n");
        }
    }

    // Zero line
    let zero_y = to_y(0.0);
    svg.push_str("  <line x1=\"");
    svg.push_str(&padding.to_string());
    svg.push_str("\" y1=\"");
    svg.push_str(&zero_y.to_string());
    svg.push_str("\" x2=\"");
    svg.push_str(&(width - padding).to_string());
    svg.push_str("\" y2=\"");
    svg.push_str(&zero_y.to_string());
    svg.push_str("\" stroke=\"");
    svg.push_str(&config.zero_line_color.to_hex());
    svg.push_str("\" stroke-dasharray=\"4\"/>\n");

    // Scatter points
    for (&fitted_val, &resid_val) in fitted.iter().zip(residuals.iter()) {
        let sx = to_x(fitted_val);
        let sy = to_y(resid_val);

        svg.push_str("  <circle cx=\"");
        svg.push_str(&sx.to_string());
        svg.push_str("\" cy=\"");
        svg.push_str(&sy.to_string());
        svg.push_str("\" r=\"");
        svg.push_str(&config.point_size.to_string());
        svg.push_str("\" fill=\"");
        svg.push_str(&config.point_color.to_hex());
        svg.push_str("\" opacity=\"0.6\"/>\n");
    }

    // Axes labels
    svg.push_str("  <text x=\"");
    svg.push_str(&(width / 2.0).to_string());
    svg.push_str("\" y=\"");
    svg.push_str(&(height - 5.0).to_string());
    svg.push_str("\" text-anchor=\"middle\" font-size=\"11\">Fitted Values</text>\n");

    svg.push_str("  <text x=\"10\" y=\"");
    svg.push_str(&(height / 2.0).to_string());
    svg.push_str("\" text-anchor=\"middle\" font-size=\"11\" transform=\"rotate(-90, 10, ");
    svg.push_str(&(height / 2.0).to_string());
    svg.push_str(")\">Residuals</text>\n");

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
    fn residplot_renders() {
        let points: Vec<DataPoint> = (0..20)
            .map(|i| DataPoint::new(i as f64, i as f64 * 2.0 + 1.0 + (i as f64 * 0.5).sin()))
            .collect();
        let config = ResidConfig::new();
        let svg = render_residplot(&points, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<line"));
    }

    #[test]
    fn residplot_empty_error() {
        let points = vec![];
        let config = ResidConfig::new();
        assert!(render_residplot(&points, &config).is_err());
    }
}
