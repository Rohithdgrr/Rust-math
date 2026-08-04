//! Regression plot (scatter with regression line and CI).

use crate::common::{DataPoint, PlotConfig};
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Configuration for a regression plot.
#[derive(Debug, Clone)]
pub struct RegPlotConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Scatter point color.
    pub point_color: Color,
    /// Regression line color.
    pub line_color: Color,
    /// Confidence interval color.
    pub ci_color: Color,
    /// Show confidence interval.
    pub show_ci: bool,
    /// CI level (e.g., 0.95 for 95%).
    pub ci_level: f64,
    /// Point size.
    pub point_size: f64,
    /// Line width.
    pub line_width: f64,
    /// Show grid.
    pub show_grid: bool,
    /// Show equation.
    pub show_equation: bool,
}

impl Default for RegPlotConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            point_color: Color::BLUE,
            line_color: Color::RED,
            ci_color: Color::rgb(200, 200, 200),
            show_ci: true,
            ci_level: 0.95,
            point_size: 4.0,
            line_width: 2.0,
            show_grid: true,
            show_equation: false,
        }
    }
}

impl RegPlotConfig {
    /// Create a new regression config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set CI level.
    pub fn with_ci(mut self, level: f64) -> Self {
        self.show_ci = true;
        self.ci_level = level;
        self
    }
}

/// Simple linear regression result.
struct LinReg {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    se: f64,
}

/// Compute linear regression.
fn linear_regression(x: &[f64], y: &[f64]) -> LinReg {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        ss_xy += (xi - mean_x) * (yi - mean_y);
        ss_xx += (xi - mean_x).powi(2);
        ss_yy += (yi - mean_y).powi(2);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    let r_squared = (ss_xy * ss_xy) / (ss_xx * ss_yy);

    // Standard error of estimate
    let sse: f64 = y.iter().zip(x.iter())
        .map(|(&yi, &xi)| (yi - (slope * xi + intercept)).powi(2))
        .sum();
    let se = (sse / (n - 2.0)).sqrt();

    LinReg {
        slope,
        intercept,
        r_squared,
        se,
    }
}

/// Render a regression plot as SVG.
pub fn render_regplot(points: &[DataPoint], config: &RegPlotConfig) -> PlotResult<String> {
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

    let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;

    let to_x = |v| padding + (v - x_min) / (x_max - x_min) * chart_width;
    let to_y = |v| padding + 30.0 + chart_height * (1.0 - (v - y_min) / (y_max - y_min));

    let reg = linear_regression(&x, &y);

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

    // Confidence interval band
    if config.show_ci && points.len() > 2 {
        let n = points.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let ss_xx: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();

        let t_value = 1.96; // Approximate for 95% CI

        let num_steps = 100;
        let mut ci_upper = Vec::new();
        let mut ci_lower = Vec::new();

        for i in 0..=num_steps {
            let xi = x_min + (x_max - x_min) * i as f64 / num_steps as f64;
            let yi = reg.slope * xi + reg.intercept;

            // Confidence interval width
            let se_fit = reg.se * (1.0 / n + (xi - mean_x).powi(2) / ss_xx).sqrt();
            let ci_width = t_value * se_fit;

            ci_upper.push((xi, yi + ci_width));
            ci_lower.push((xi, yi - ci_width));
        }

        // Draw CI polygon
        let mut ci_path = String::from("M");
        for (i, (xi, yi)) in ci_upper.iter().enumerate() {
            let sx = to_x(*xi);
            let sy = to_y(*yi);
            if i == 0 {
                ci_path.push_str(&format!("{sx},{sy}"));
            } else {
                ci_path.push_str(&format!(" L {sx},{sy}"));
            }
        }
        for (xi, yi) in ci_lower.iter().rev() {
            let sx = to_x(*xi);
            let sy = to_y(*yi);
            ci_path.push_str(&format!(" L {sx},{sy}"));
        }
        ci_path.push_str(" Z");

        svg.push_str("  <path d=\"");
        svg.push_str(&ci_path);
        svg.push_str("\" fill=\"");
        svg.push_str(&config.ci_color.to_hex());
        svg.push_str("\" opacity=\"0.3\"/>\n");
    }

    // Regression line
    let x1 = x_min;
    let y1 = reg.slope * x1 + reg.intercept;
    let x2 = x_max;
    let y2 = reg.slope * x2 + reg.intercept;

    svg.push_str("  <line x1=\"");
    svg.push_str(&to_x(x1).to_string());
    svg.push_str("\" y1=\"");
    svg.push_str(&to_y(y1).to_string());
    svg.push_str("\" x2=\"");
    svg.push_str(&to_x(x2).to_string());
    svg.push_str("\" y2=\"");
    svg.push_str(&to_y(y2).to_string());
    svg.push_str("\" stroke=\"");
    svg.push_str(&config.line_color.to_hex());
    svg.push_str("\" stroke-width=\"");
    svg.push_str(&config.line_width.to_string());
    svg.push_str("\"/>\n");

    // Scatter points
    for p in points {
        let sx = to_x(p.x);
        let sy = to_y(p.y);

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

    // Equation
    if config.show_equation {
        let eq_y = padding + 30.0 + 15.0;
        svg.push_str("  <text x=\"");
        svg.push_str(&(padding + 10.0).to_string());
        svg.push_str("\" y=\"");
        svg.push_str(&eq_y.to_string());
        svg.push_str("\" font-size=\"12\" fill=\"black\">y = ");
        svg.push_str(&format!("{:.3}x + {:.3}", reg.slope, reg.intercept));
        svg.push_str("  R² = ");
        svg.push_str(&format!("{:.3}", reg.r_squared));
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
    fn regplot_renders() {
        let points: Vec<DataPoint> = (0..20)
            .map(|i| DataPoint::new(i as f64, i as f64 * 2.0 + 1.0))
            .collect();
        let config = RegPlotConfig::new();
        let svg = render_regplot(&points, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<line"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn regplot_empty_error() {
        let points = vec![];
        let config = RegPlotConfig::new();
        assert!(render_regplot(&points, &config).is_err());
    }
}
