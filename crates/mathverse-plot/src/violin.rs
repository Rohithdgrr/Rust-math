//! Violin plot rendering.

use std::f64::consts::PI;

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Linear interpolation quantile (Method 2 / Tukey hinges).
/// Returns the p-th quantile (p in [0, 1]) of sorted data.
fn quantile_linear(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let index = p * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = index - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Configuration for a violin plot.
#[derive(Debug, Clone)]
pub struct ViolinConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Width of each violin (pixels).
    pub violin_width: f64,
    /// Number of points to approximate the density curve.
    pub num_points: usize,
    /// Bandwidth for KDE (if None, uses Silverman's rule).
    pub bandwidth: Option<f64>,
    /// Show box plot inside violin.
    pub show_box: bool,
    /// Show median line.
    pub show_median: bool,
    /// Show grid.
    pub show_grid: bool,
}

impl Default for ViolinConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            violin_width: 60.0,
            num_points: 100,
            bandwidth: None,
            show_box: true,
            show_median: true,
            show_grid: true,
        }
    }
}

impl ViolinConfig {
    /// Create a new violin config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set violin width.
    pub fn with_width(mut self, width: f64) -> Self {
        self.violin_width = width;
        self
    }

    /// Set bandwidth.
    pub fn with_bandwidth(mut self, bw: f64) -> Self {
        self.bandwidth = Some(bw);
        self
    }
}

/// A single violin data set.
#[derive(Debug, Clone)]
pub struct ViolinData {
    /// Label for the violin.
    pub label: String,
    /// Data values.
    pub values: Vec<f64>,
    /// Fill color.
    pub color: Color,
}

impl ViolinData {
    /// Create new violin data.
    pub fn new(label: impl Into<String>, values: Vec<f64>, color: Color) -> Self {
        Self {
            label: label.into(),
            values,
            color,
        }
    }
}

/// Gaussian kernel function.
fn gaussian_kernel(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Compute KDE for a dataset.
fn kde(values: &[f64], bandwidth: f64, num_points: usize) -> Vec<(f64, f64)> {
    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;
    let margin = range * 0.2; // Extra margin

    let step = (2.0 * (range + 2.0 * margin)) / num_points as f64;
    let mut result = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let x = min - margin + i as f64 * step;
        let density: f64 = values
            .iter()
            .map(|&v| gaussian_kernel((x - v) / bandwidth))
            .sum::<f64>()
            / (values.len() as f64 * bandwidth);
        result.push((x, density));
    }

    result
}

/// Render a violin plot as SVG.
pub fn render_violin_plot(data: &[ViolinData], config: &ViolinConfig) -> PlotResult<String> {
    if data.is_empty() {
        return Err(PlotError::InvalidData("no data provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    if data.iter().any(|d| d.values.is_empty()) {
        return Err(PlotError::InvalidData(
            "each violin series must contain at least one value".into(),
        ));
    }

    // Find global bounds
    let all_min = data.iter().flat_map(|d| &d.values).fold(f64::INFINITY, |a, &b| a.min(b));
    let all_max = data.iter().flat_map(|d| &d.values).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = if all_max - all_min > 0.0 {
        all_max - all_min
    } else {
        1.0 // constant data: avoid division by zero
    };

    let chart_height = height - padding * 2.0 - 30.0;
    let to_y = |v| padding + 30.0 + chart_height * (1.0 - (v - all_min) / range);

    let n = data.len();
    let total_width = width - padding * 2.0;
    let spacing = total_width / (n + 1) as f64;

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

    // Draw violins
    for (idx, d) in data.iter().enumerate() {
        let center_x = padding + spacing * (idx + 1) as f64;

        // Compute bandwidth (Silverman's rule if not set)
        let bw = config.bandwidth.unwrap_or_else(|| {
            let n = d.values.len() as f64;
            let std = {
                let mean = d.values.iter().sum::<f64>() / n;
                let variance = d.values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
                variance.sqrt()
            };
            1.06 * std * n.powf(-0.2)
        });

        // Compute KDE
        let density = kde(&d.values, bw, config.num_points);
        let max_density = density.iter().map(|(_, d)| d).fold(0.0_f64, |a, &b| a.max(b));

        if max_density == 0.0 {
            continue;
        }

        // Build violin path (right side, then left side mirrored)
        let half_width = config.violin_width / 2.0;

        // Right side
        let mut right_path = String::from("M");
        for (i, (val, dens)) in density.iter().enumerate() {
            let y = to_y(*val);
            let x = center_x + (dens / max_density) * half_width;
            if i == 0 {
                right_path.push_str(&format!(" {x},{y}"));
            } else {
                right_path.push_str(&format!(" L {x},{y}"));
    }
}


        // Left side (mirrored)
        for (val, dens) in density.iter().rev() {
            let y = to_y(*val);
            let x = center_x - (dens / max_density) * half_width;
            right_path.push_str(&format!(" L {x},{y}"));
        }
        right_path.push_str(" Z");

        svg.push_str(&format!(
            r#"  <path d="{right_path}" fill="{}" stroke="{}" stroke-width="1" opacity="0.7"/>"#,
            d.color.to_hex(),
            d.color.to_hex()
        ));
        svg.push('\n');

        // Box plot inside
        if config.show_box {
            let sorted: Vec<f64> = {
                let mut v = d.values.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                v
            };
            let q1 = quantile_linear(&sorted, 0.25);
            let median = quantile_linear(&sorted, 0.5);
            let q3 = quantile_linear(&sorted, 0.75);

            let box_width = 8.0;

            // IQR box
            svg.push_str(&format!(
                r#"  <rect x="{}" y="{}" width="{box_width}" height="{}" fill="white" stroke="black"/>"#,
                center_x - box_width / 2.0,
                to_y(q3),
                to_y(q1) - to_y(q3)
            ));
            svg.push('\n');

            // Median
            if config.show_median {
                svg.push_str(&format!(
                    r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="red" stroke-width="2"/>"#,
                    center_x - box_width / 2.0,
                    to_y(median),
                    center_x + box_width / 2.0,
                    to_y(median)
                ));
                svg.push('\n');
            }

            // Whiskers using 1.5 * IQR rule
            let iqr = q3 - q1;
            let lower_fence = q1 - 1.5 * iqr;
            let upper_fence = q3 + 1.5 * iqr;
            let whisker_low = sorted
                .iter()
                .copied()
                .filter(|&v| v >= lower_fence)
                .fold(f64::INFINITY, f64::min);
            let whisker_high = sorted
                .iter()
                .copied()
                .filter(|&v| v <= upper_fence)
                .fold(f64::NEG_INFINITY, f64::max);
            svg.push_str(&format!(
                r#"  <line x1="{center_x}" y1="{}" x2="{center_x}" y2="{}" stroke="black"/>"#,
                to_y(whisker_high), to_y(whisker_low)
            ));
            svg.push('\n');
        }

        // Label
        svg.push_str(&format!(
            r#"  <text x="{center_x}" y="{}" text-anchor="middle" font-size="11">{}</text>"#,
            height - padding + 15.0, crate::common::xml_escape(&d.label)
        ));
        svg.push('\n');
    }

    // Title
    if !config.plot_config.title.is_empty() {
        svg.push_str(&format!(
            r#"  <text x="{}" y="25" text-anchor="middle" font-size="20" font-weight="bold">{}</text>"#,
            width / 2.0, crate::common::xml_escape(&config.plot_config.title)
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
    fn violin_plot_renders() {
        let data = vec![
            ViolinData::new("A", vec![1.0, 2.0, 3.0, 4.0, 5.0], Color::BLUE),
            ViolinData::new("B", vec![2.0, 3.0, 4.0, 5.0, 6.0], Color::GREEN),
        ];
        let config = ViolinConfig::new();
        let svg = render_violin_plot(&data, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn violin_plot_empty_error() {
        let data = vec![];
        let config = ViolinConfig::new();
        assert!(render_violin_plot(&data, &config).is_err());
    }

    #[test]
    fn violin_plot_empty_series_error() {
        let data = vec![
            ViolinData::new("A", vec![1.0, 2.0], Color::BLUE),
            ViolinData::new("B", vec![], Color::GREEN),
        ];
        let config = ViolinConfig::new();
        assert!(render_violin_plot(&data, &config).is_err());
    }

    #[test]
    fn violin_plot_constant_data() {
        let data = vec![ViolinData::new("A", vec![3.0, 3.0, 3.0], Color::BLUE)];
        let config = ViolinConfig::new();
        let svg = render_violin_plot(&data, &config).unwrap();
        assert!(svg.contains("<svg"));
    }
}
