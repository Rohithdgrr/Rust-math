//! 2D histogram (heatmap-style) rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};

/// Configuration for a 2D histogram.
#[derive(Debug, Clone)]
pub struct Hist2DConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Number of bins in X direction.
    pub bins_x: usize,
    /// Number of bins in Y direction.
    pub bins_y: usize,
    /// Show colorbar.
    pub show_colorbar: bool,
    /// Show grid.
    pub show_grid: bool,
}

impl Default for Hist2DConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            bins_x: 20,
            bins_y: 20,
            show_colorbar: true,
            show_grid: false,
        }
    }
}

impl Hist2DConfig {
    /// Create a new 2D histogram config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set bins.
    pub fn with_bins(mut self, bins_x: usize, bins_y: usize) -> Self {
        self.bins_x = bins_x;
        self.bins_y = bins_y;
        self
    }
}

/// Render a 2D histogram as SVG.
pub fn render_hist2d(
    x_data: &[f64],
    y_data: &[f64],
    config: &Hist2DConfig,
) -> PlotResult<String> {
    if x_data.len() != y_data.len() {
        return Err(PlotError::InvalidData("x and y must have same length".into()));
    }
    if x_data.is_empty() {
        return Err(PlotError::InvalidData("no data provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Find bounds
    let x_min = x_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = x_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = y_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = y_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    // Create bins
    let mut bins = vec![vec![0.0; config.bins_x]; config.bins_y];

    for (&x, &y) in x_data.iter().zip(y_data.iter()) {
        let bx = ((x - x_min) / x_range * config.bins_x as f64) as usize;
        let by = ((y - y_min) / y_range * config.bins_y as f64) as usize;
        let bx = bx.min(config.bins_x - 1);
        let by = by.min(config.bins_y - 1);
        bins[by][bx] += 1.0;
    }

    // Find max count
    let max_count = bins.iter().flat_map(|r| r.iter()).fold(0.0_f64, |a, &b| a.max(b));
    if max_count == 0.0 {
        return Err(PlotError::InvalidData("no data in bins".into()));
    }

    let chart_width = width - padding * 2.0 - if config.show_colorbar { 60.0 } else { 0.0 };
    let chart_height = height - padding * 2.0 - 30.0;

    let cell_w = chart_width / config.bins_x as f64;
    let cell_h = chart_height / config.bins_y as f64;

    // Color function (viridis-like)
    let color_for_val = |val: f64| -> String {
        let t = val / max_count;
        let r = (t * 255.0) as u8;
        let g = ((1.0 - (t - 0.5).abs() * 2.0) * 200.0) as u8;
        let b = ((1.0 - t) * 255.0) as u8;
        format!("#{:02x}{:02x}{:02x}", r, g, b)
    };

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
        width as u32, height as u32
    ));
    svg.push('\n');
    svg.push_str(r#"  <rect width="100%" height="100%" fill="white"/>"#);
    svg.push('\n');

    // Draw bins
    for j in 0..config.bins_y {
        for i in 0..config.bins_x {
            if bins[j][i] == 0.0 {
                continue;
            }
            let x = padding + i as f64 * cell_w;
            let y = padding + 30.0 + (config.bins_y - 1 - j) as f64 * cell_h;
            let color = color_for_val(bins[j][i]);

            svg.push_str(&format!(
                r#"  <rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}"/>"#,
            ));
            svg.push('\n');
        }
    }

    // Colorbar
    if config.show_colorbar {
        let cb_x = width - padding - 40.0;
        let cb_y = padding + 30.0;
        let cb_height = chart_height;
        let cb_width = 15.0;

        // Gradient
        svg.push_str(&format!(
            r#"  <defs><linearGradient id="cb" x1="0" y1="1" x2="0" y2="0">
      <stop offset="0%" stop-color="{}"/>
      <stop offset="100%" stop-color="{}"/>
    </linearGradient></defs>"#,
            color_for_val(0.0),
            color_for_val(max_count)
        ));
        svg.push('\n');

        svg.push_str(&format!(
            r#"  <rect x="{cb_x}" y="{cb_y}" width="{cb_width}" height="{cb_height}" fill="url(#cb)"/>"#,
        ));
        svg.push('\n');

        // Ticks
        for i in 0..=4 {
            let y = cb_y + cb_height * (1.0 - i as f64 / 4.0);
            let val = max_count * i as f64 / 4.0;
            svg.push_str(&format!(
                r#"  <text x="{}" y="{y}" font-size="9" dominant-baseline="middle">{:.0}</text>"#,
                cb_x + cb_width + 5.0, val
            ));
            svg.push('\n');
        }
    }

    // Axes
    svg.push_str(&format!(
        r#"  <text x="{}" y="{}" text-anchor="middle" font-size="11">x</text>"#,
        padding + chart_width / 2.0, height - 5.0
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r#"  <text x="10" y="{}" text-anchor="middle" font-size="11" transform="rotate(-90, 10, {})">y</text>"#,
        padding + 30.0 + chart_height / 2.0, padding + 30.0 + chart_height / 2.0
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
    fn hist2d_renders() {
        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let config = Hist2DConfig::new();
        let svg = render_hist2d(&x, &y, &config).unwrap();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn hist2d_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        let config = Hist2DConfig::new();
        assert!(render_hist2d(&x, &y, &config).is_err());
    }
}
