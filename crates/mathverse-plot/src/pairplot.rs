//! Pair plot (pairwise scatter matrix).

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Configuration for a pair plot.
#[derive(Debug, Clone)]
pub struct PairConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Point color.
    pub point_color: Color,
    /// Point size.
    pub point_size: f64,
    /// Show diagonal KDE/histogram.
    pub show_diag: bool,
    /// Show grid.
    pub show_grid: bool,
}

impl Default for PairConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            point_color: Color::BLUE,
            point_size: 2.0,
            show_diag: true,
            show_grid: false,
        }
    }
}

impl PairConfig {
    /// Create a new pair config.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Render a pair plot as SVG.
pub fn render_pairplot(
    data: &[Vec<f64>],
    labels: &[String],
    config: &PairConfig,
) -> PlotResult<String> {
    if data.is_empty() {
        return Err(PlotError::InvalidData("no data provided".into()));
    }

    let n_vars = data.len();
    if labels.len() != n_vars {
        return Err(PlotError::InvalidData("labels length must match data".into()));
    }

    let total_width = config.plot_config.width as f64;
    let total_height = config.plot_config.height as f64;

    let margin = 60.0;
    let cell_size = (total_width - margin) / n_vars as f64;

    let mut svg = String::new();
    svg.push_str("<svg width=\"");
    svg.push_str(&total_width.to_string());
    svg.push_str("\" height=\"");
    svg.push_str(&total_height.to_string());
    svg.push_str("\" xmlns=\"http://www.w3.org/2000/svg\">\n");
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    for i in 0..n_vars {
        for j in 0..n_vars {
            let cx = margin + j as f64 * cell_size;
            let cy = margin + i as f64 * cell_size;

            // Cell border
            svg.push_str("  <rect x=\"");
            svg.push_str(&cx.to_string());
            svg.push_str("\" y=\"");
            svg.push_str(&cy.to_string());
            svg.push_str("\" width=\"");
            svg.push_str(&cell_size.to_string());
            svg.push_str("\" height=\"");
            svg.push_str(&cell_size.to_string());
            svg.push_str("\" fill=\"none\" stroke=\"#ddd\"/>\n");

            if i == j {
                // Diagonal: histogram
                if config.show_diag {
                    let values = &data[i];
                    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let bins = 15;
                    let bin_width = (max_val - min_val) / bins as f64;

                    if bin_width > 0.0 {
                        let mut hist = vec![0usize; bins];
                        for &v in values {
                            let bin = ((v - min_val) / bin_width) as usize;
                            let bin = bin.min(bins - 1);
                            hist[bin] += 1;
                        }
                        let max_count = *hist.iter().max().ok_or_else(|| PlotError::InvalidData("empty histogram".into()))? as f64;

                        for (b, &count) in hist.iter().enumerate() {
                            let bar_h = (count as f64 / max_count) * (cell_size - 4.0);
                            let bx = cx + 2.0 + b as f64 * (cell_size - 4.0) / bins as f64;
                            let by = cy + cell_size - 2.0 - bar_h;

                            svg.push_str("  <rect x=\"");
                            svg.push_str(&bx.to_string());
                            svg.push_str("\" y=\"");
                            svg.push_str(&by.to_string());
                            svg.push_str("\" width=\"");
                            svg.push_str(&((cell_size - 4.0) / bins as f64 - 1.0).to_string());
                            svg.push_str("\" height=\"");
                            svg.push_str(&bar_h.to_string());
                            svg.push_str("\" fill=\"");
                            svg.push_str(&config.point_color.to_hex());
                            svg.push_str("\" opacity=\"0.5\"/>\n");
                        }
                    }
                }

                // Variable label
                svg.push_str("  <text x=\"");
                svg.push_str(&(cx + cell_size / 2.0).to_string());
                svg.push_str("\" y=\"");
                svg.push_str(&(cy + cell_size / 2.0).to_string());
                svg.push_str("\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"10\">");
                svg.push_str(&labels[i]);
                svg.push_str("</text>\n");
            } else {
                // Scatter plot
                let x_vals = &data[j];
                let y_vals = &data[i];

                let x_min = x_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let x_max = x_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let y_min = y_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let y_max = y_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                let plot_size = cell_size - 8.0;

                for (&x, &y) in x_vals.iter().zip(y_vals.iter()) {
                    let sx = cx + 4.0 + (x - x_min) / (x_max - x_min) * plot_size;
                    let sy = cy + 4.0 + plot_size * (1.0 - (y - y_min) / (y_max - y_min));

                    svg.push_str("  <circle cx=\"");
                    svg.push_str(&sx.to_string());
                    svg.push_str("\" cy=\"");
                    svg.push_str(&sy.to_string());
                    svg.push_str("\" r=\"");
                    svg.push_str(&config.point_size.to_string());
                    svg.push_str("\" fill=\"");
                    svg.push_str(&config.point_color.to_hex());
                    svg.push_str("\" opacity=\"0.4\"/>\n");
                }
            }
        }
    }

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
    fn pairplot_renders() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
            vec![1.0, 3.0, 5.0, 7.0, 9.0],
        ];
        let labels = vec!["X".into(), "Y".into(), "Z".into()];
        let config = PairConfig::new();
        let svg = render_pairplot(&data, &labels, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn pairplot_empty_error() {
        let data = vec![];
        let labels = vec![];
        let config = PairConfig::new();
        assert!(render_pairplot(&data, &labels, &config).is_err());
    }
}
