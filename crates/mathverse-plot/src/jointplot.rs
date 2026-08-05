//! Joint plot (bivariate + marginal distributions).

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Configuration for a joint plot.
#[derive(Debug, Clone)]
pub struct JointConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Point color.
    pub point_color: Color,
    /// Marginal color.
    pub marginal_color: Color,
    /// Point size.
    pub point_size: f64,
    /// Number of bins for marginal histograms.
    pub bins: usize,
    /// Show regression line in joint.
    pub show_regression: bool,
}

impl Default for JointConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            point_color: Color::BLUE,
            marginal_color: Color::rgb(100, 100, 200),
            point_size: 3.0,
            bins: 20,
            show_regression: false,
        }
    }
}

impl JointConfig {
    /// Create a new joint config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set bins.
    pub fn with_bins(mut self, bins: usize) -> Self {
        self.bins = bins;
        self
    }
}

/// Render a joint plot as SVG.
pub fn render_jointplot(
    x_data: &[f64],
    y_data: &[f64],
    config: &JointConfig,
) -> PlotResult<String> {
    if x_data.len() != y_data.len() {
        return Err(PlotError::InvalidData("x and y must have same length".into()));
    }
    if x_data.is_empty() {
        return Err(PlotError::InvalidData("no data provided".into()));
    }

    let total_width = config.plot_config.width as f64;
    let total_height = config.plot_config.height as f64;

    let margin_size = 100.0;
    let main_size = total_width - margin_size - 20.0;

    let main_x = margin_size + 10.0;
    let main_y = 10.0;
    let _top_y = 10.0;
    let right_x = main_x + main_size + 10.0;

    let x_min = x_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let to_x = |v| main_x + (v - x_min) / (x_max - x_min) * main_size;
    let to_y = |v| main_y + main_size * (1.0 - (v - y_min) / (y_max - y_min));

    let mut svg = String::new();
    svg.push_str("<svg width=\"");
    svg.push_str(&total_width.to_string());
    svg.push_str("\" height=\"");
    svg.push_str(&total_height.to_string());
    svg.push_str("\" xmlns=\"http://www.w3.org/2000/svg\">\n");
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    // Top marginal histogram
    {
        let mut hist = vec![0usize; config.bins];
        let bin_width = (x_max - x_min) / config.bins as f64;
        for &x in x_data {
            let bin = ((x - x_min) / bin_width) as usize;
            let bin = bin.min(config.bins - 1);
            hist[bin] += 1;
        }
        let max_count = *hist.iter().max().ok_or_else(|| PlotError::InvalidData("empty histogram".into()))? as f64;
        let bar_w = main_size / config.bins as f64;
        let marginal_height = margin_size - 20.0;

        for (i, &count) in hist.iter().enumerate() {
            let bar_h = (count as f64 / max_count) * marginal_height;
            let bx = main_x + i as f64 * bar_w;
            let by = main_y - bar_h;

            svg.push_str("  <rect x=\"");
            svg.push_str(&bx.to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&by.to_string());
            svg.push_str("\" width=\"");
            svg.push_str(&(bar_w - 1.0).to_string());
            svg.push_str("\" height=\"");
            svg.push_str(&bar_h.to_string());
            svg.push_str("\" fill=\"");
            svg.push_str(&config.marginal_color.to_hex());
            svg.push_str("\" opacity=\"0.5\"/>\n");
        }
    }

    // Right marginal histogram
    {
        let mut hist = vec![0usize; config.bins];
        let bin_width = (y_max - y_min) / config.bins as f64;
        for &y in y_data {
            let bin = ((y - y_min) / bin_width) as usize;
            let bin = bin.min(config.bins - 1);
            hist[bin] += 1;
        }
        let max_count = *hist.iter().max().ok_or_else(|| PlotError::InvalidData("empty histogram".into()))? as f64;
        let bar_h = main_size / config.bins as f64;
        let marginal_width = margin_size - 20.0;

        for (i, &count) in hist.iter().enumerate() {
            let bar_w = (count as f64 / max_count) * marginal_width;
            let bx = right_x + marginal_width - bar_w;
            let by = main_y + (config.bins - 1 - i) as f64 * bar_h;

            svg.push_str("  <rect x=\"");
            svg.push_str(&bx.to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&by.to_string());
            svg.push_str("\" width=\"");
            svg.push_str(&bar_w.to_string());
            svg.push_str("\" height=\"");
            svg.push_str(&(bar_h - 1.0).to_string());
            svg.push_str("\" fill=\"");
            svg.push_str(&config.marginal_color.to_hex());
            svg.push_str("\" opacity=\"0.5\"/>\n");
        }
    }

    // Main scatter plot
    for (&x, &y) in x_data.iter().zip(y_data.iter()) {
        let sx = to_x(x);
        let sy = to_y(y);

        svg.push_str("  <circle cx=\"");
        svg.push_str(&sx.to_string());
        svg.push_str("\" cy=\"");
        svg.push_str(&sy.to_string());
        svg.push_str("\" r=\"");
        svg.push_str(&config.point_size.to_string());
        svg.push_str("\" fill=\"");
        svg.push_str(&config.point_color.to_hex());
        svg.push_str("\" opacity=\"0.5\"/>\n");
    }

    // Border around main plot
    svg.push_str("  <rect x=\"");
    svg.push_str(&main_x.to_string());
    svg.push_str("\" y=\"");
    svg.push_str(&main_y.to_string());
    svg.push_str("\" width=\"");
    svg.push_str(&main_size.to_string());
    svg.push_str("\" height=\"");
    svg.push_str(&main_size.to_string());
    svg.push_str("\" fill=\"none\" stroke=\"black\"/>\n");

    // Title
    if !config.plot_config.title.is_empty() {
        svg.push_str("  <text x=\"");
        svg.push_str(&(total_width / 2.0).to_string());
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
    fn jointplot_renders() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let config = JointConfig::new();
        let svg = render_jointplot(&x, &y, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn jointplot_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        let config = JointConfig::new();
        assert!(render_jointplot(&x, &y, &config).is_err());
    }
}
