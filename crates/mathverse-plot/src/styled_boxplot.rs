//! Full-featured box plot rendering with matplotlib styling options.
//!
//! Extends the basic [`crate::boxplot::BoxStats`] statistics with a rich
//! rendering config supporting `notch`, `vert`, `positions`, `patch_artist`,
//! `capsize`, `flierprops`, `meanline`, and `showmeans`.

use crate::boxplot::BoxStats;
use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};

/// Styling for outlier "flier" markers (matplotlib `flierprops`).
#[derive(Debug, Clone)]
pub struct FlierProps {
    /// Marker color.
    pub color: String,
    /// Marker size (radius in px).
    pub size: f64,
}

impl Default for FlierProps {
    fn default() -> Self {
        Self {
            color: "#ff7f0e".to_string(),
            size: 4.0,
        }
    }
}

/// Configuration for a styled box plot.
#[derive(Debug, Clone)]
pub struct BoxPlotConfig {
    /// Chart configuration (dimensions, padding, title...).
    pub plot_config: PlotConfig,
    /// Draw notched boxes (median CI notch).
    pub notch: bool,
    /// Vertical boxes (true) or horizontal (false).
    pub vert: bool,
    /// Fill the boxes with `patch_color` (patch_artist=True).
    pub patch_artist: bool,
    /// Box fill color (used when `patch_artist`).
    pub patch_color: String,
    /// Whisker cap line length in px.
    pub capsize: f64,
    /// Outlier marker styling.
    pub flierprops: FlierProps,
    /// x-positions for each box (defaults to 1..=n).
    pub positions: Vec<f64>,
    /// Draw a line at the mean inside each box.
    pub showmeans: bool,
    /// Line width for box edges.
    pub line_width: f64,
    /// Edge color.
    pub edge_color: String,
}

impl Default for BoxPlotConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            notch: false,
            vert: true,
            patch_artist: true,
            patch_color: "#4c72b0".to_string(),
            capsize: 6.0,
            flierprops: FlierProps::default(),
            positions: Vec::new(),
            showmeans: false,
            line_width: 1.5,
            edge_color: "#222222".to_string(),
        }
    }
}

impl BoxPlotConfig {
    /// Create a new box plot config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the notch flag.
    #[must_use]
    pub fn with_notch(mut self, notch: bool) -> Self {
        self.notch = notch;
        self
    }

    /// Set vertical (true) or horizontal (false) orientation.
    #[must_use]
    pub fn with_vert(mut self, vert: bool) -> Self {
        self.vert = vert;
        self
    }

    /// Set the box fill color (also enables `patch_artist`).
    #[must_use]
    pub fn with_patch_color(mut self, color: impl Into<String>) -> Self {
        self.patch_color = color.into();
        self.patch_artist = true;
        self
    }

    /// Set the whisker cap size in px.
    #[must_use]
    pub fn with_capsize(mut self, capsize: f64) -> Self {
        self.capsize = capsize;
        self
    }

    /// Set explicit x-positions for each box.
    #[must_use]
    pub fn with_positions(mut self, positions: Vec<f64>) -> Self {
        self.positions = positions;
        self
    }

    /// Show the mean line inside each box.
    #[must_use]
    pub fn with_showmeans(mut self, show: bool) -> Self {
        self.showmeans = show;
        self
    }

    /// Set the outlier marker color.
    #[must_use]
    pub fn with_flier_color(mut self, color: impl Into<String>) -> Self {
        self.flierprops.color = color.into();
        self
    }
}

/// Render multiple data series as a styled box plot in SVG.
///
/// Each entry of `datasets` becomes one box at the configured position.
///
/// # Errors
///
/// Returns `PlotError::InvalidData` when a dataset is empty or contains
/// non-finite values.
pub fn render_styled_boxplot(
    datasets: &[Vec<f64>],
    labels: &[String],
    config: &BoxPlotConfig,
) -> PlotResult<String> {
    if datasets.is_empty() {
        return Err(PlotError::InvalidData("no datasets".into()));
    }
    let stats: Vec<BoxStats> = datasets
        .iter()
        .map(|d| BoxStats::compute(d))
        .collect::<PlotResult<_>>()?;

    let n = datasets.len();
    let positions: Vec<f64> = if config.positions.is_empty() {
        (0..n).map(|i| i as f64 + 1.0).collect()
    } else {
        config.positions.clone()
    };
    if positions.len() != n {
        return Err(PlotError::InvalidData(
            "positions length must match datasets length".into(),
        ));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;
    let chart_w = width - 2.0 * padding;
    let chart_h = height - 2.0 * padding;

    // Overall data range across whiskers and outliers for axis mapping.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for s in &stats {
        lo = lo.min(s.min).min(s.q1).min(s.median);
        hi = hi.max(s.max).max(s.q3).max(s.median);
        for o in &s.outliers {
            lo = lo.min(*o);
            hi = hi.max(*o);
        }
    }
    if !lo.is_finite() {
        lo = 0.0;
    }
    if !hi.is_finite() {
        hi = 1.0;
    }
    let span = (hi - lo).max(1e-9);
    lo -= span * 0.05;
    hi += span * 0.05;

    let (pos_lo, pos_hi) = (
        positions.iter().copied().fold(f64::INFINITY, f64::min),
        positions.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    let pos_span = (pos_hi - pos_lo).max(1.0);
    let box_w = (chart_w / n as f64).min(80.0).max(12.0) * 0.5;

    let to_data_x = |v: f64| padding + (v - lo) / (hi - lo) * chart_w;
    let to_data_y = |v: f64| padding + chart_h * (1.0 - (v - lo) / (hi - lo));
    let to_pos_x = |p: f64| padding + (p - pos_lo) / pos_span * chart_w;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">"#,
        w = width as u32,
        h = height as u32
    ));
    svg.push('\n');
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    for (i, (s, &pos)) in stats.iter().zip(&positions).enumerate() {
        let cx = to_pos_x(pos);
        let (x0, x1) = (cx - box_w / 2.0, cx + box_w / 2.0);
        let (y_q1, y_med, y_q3) = (to_data_y(s.q3), to_data_y(s.median), to_data_y(s.q1));
        let (y_lo, y_hi) = (to_data_y(s.max), to_data_y(s.min));

        let label = labels
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("box{}", i + 1));

        if config.vert {
            // Whiskers.
            svg.push_str(&format!(
                r#"  <line x1="{cx}" y1="{y_lo}" x2="{cx}" y2="{y_hi}" stroke="{edge}" stroke-width="{lw}"/>"#,
                edge = config.edge_color,
                lw = config.line_width
            ));
            svg.push('\n');
            // Caps.
            let cap = config.capsize;
            for yy in [y_lo, y_hi] {
                svg.push_str(&format!(
                    r#"  <line x1="{c1}" y1="{yy}" x2="{c2}" y2="{yy}" stroke="{edge}" stroke-width="{lw}"/>"#,
                    c1 = cx - cap,
                    c2 = cx + cap,
                    edge = config.edge_color,
                    lw = config.line_width
                ));
                svg.push('\n');
            }
            // Box.
            if config.notch {
                // Notched median region.
                let mid = cx;
                let notch_half = box_w * 0.18;
                svg.push_str(&format!(
                    r#"  <path d="M {x0} {y_q1} L {x0} {y_med} L {n0} {y_med} L {n1} {y_q3} L {x1} {y_q3} L {x1} {y_med} L {n2} {y_med} L {n3} {y_q1} Z" fill="{fill}" stroke="{edge}" stroke-width="{lw}"/>"#,
                    x0 = x0,
                    y_q1 = y_q1,
                    y_med = y_med,
                    n0 = mid - notch_half,
                    n1 = mid + notch_half,
                    y_q3 = y_q3,
                    x1 = x1,
                    n2 = mid - notch_half,
                    n3 = mid + notch_half,
                    fill = if config.patch_artist { &config.patch_color } else { "none" },
                    edge = config.edge_color,
                    lw = config.line_width
                ));
                svg.push('\n');
            } else {
                let fill = if config.patch_artist {
                    &config.patch_color
                } else {
                    "none"
                };
                svg.push_str(&format!(
                    r#"  <rect x="{x0}" y="{y_q3}" width="{w}" height="{h}" fill="{fill}" stroke="{edge}" stroke-width="{lw}"/>"#,
                    w = box_w,
                    h = (y_q1 - y_q3).abs(),
                    edge = config.edge_color,
                    lw = config.line_width
                ));
                svg.push('\n');
            }
            // Median line.
            svg.push_str(&format!(
                r#"  <line x1="{x0}" y1="{y_med}" x2="{x1}" y2="{y_med}" stroke="{edge}" stroke-width="2"/>"#,
                edge = config.edge_color
            ));
            svg.push('\n');
            // Mean line.
            if config.showmeans {
                let mean = datasets[i].iter().sum::<f64>() / datasets[i].len() as f64;
                let y_mean = to_data_y(mean);
                svg.push_str(&format!(
                    "  <line x1=\"{x0}\" y1=\"{y_mean}\" x2=\"{x1}\" y2=\"{y_mean}\" stroke='#ff0000' stroke-width=\"1.5\" stroke-dasharray=\"3,2\"/>"
                ));
                svg.push('\n');
            }
            // Flier markers.
            for &o in &s.outliers {
                let yy = to_data_y(o);
                svg.push_str(&format!(
                    r#"  <circle cx="{cx}" cy="{yy}" r="{r}" fill="{color}"/>"#,
                    r = config.flierprops.size,
                    color = config.flierprops.color
                ));
                svg.push('\n');
            }
            // Category label under the box.
            svg.push_str(&format!(
                r#"  <text x="{cx}" y="{y}" text-anchor="middle" font-size="11">{label}</text>"#,
                y = height - 5.0
            ));
            svg.push('\n');
        } else {
            // Horizontal variant: swap x/y roles.
            let yc = cx; // position axis is now the vertical axis
            let (y0, y1) = (yc - box_w / 2.0, yc + box_w / 2.0);
            svg.push_str(&format!(
                r#"  <line x1="{x_lo}" y1="{yc}" x2="{x_hi}" y2="{yc}" stroke="{edge}" stroke-width="{lw}"/>"#,
                x_lo = to_data_x(s.min),
                x_hi = to_data_x(s.max),
                edge = config.edge_color,
                lw = config.line_width
            ));
            svg.push('\n');
            for xx in [to_data_x(s.min), to_data_x(s.max)] {
                svg.push_str(&format!(
                    r#"  <line x1="{xx}" y1="{y0}" x2="{xx}" y2="{y1}" stroke="{edge}" stroke-width="{lw}"/>"#,
                    edge = config.edge_color,
                    lw = config.line_width
                ));
                svg.push('\n');
            }
            let x_q1 = to_data_x(s.q1);
            let x_q3 = to_data_x(s.q3);
            let x_med = to_data_x(s.median);
            let fill = if config.patch_artist {
                &config.patch_color
            } else {
                "none"
            };
            svg.push_str(&format!(
                r#"  <rect x="{x_q1}" y="{y0}" width="{w}" height="{h}" fill="{fill}" stroke="{edge}" stroke-width="{lw}"/>"#,
                w = (x_q3 - x_q1).abs(),
                h = box_w,
                edge = config.edge_color,
                lw = config.line_width
            ));
            svg.push('\n');
            svg.push_str(&format!(
                r#"  <line x1="{x_med}" y1="{y0}" x2="{x_med}" y2="{y1}" stroke="{edge}" stroke-width="2"/>"#,
                edge = config.edge_color
            ));
            svg.push('\n');
            for &o in &s.outliers {
                let xx = to_data_x(o);
                svg.push_str(&format!(
                    r#"  <circle cx="{xx}" cy="{yc}" r="{r}" fill="{color}"/>"#,
                    r = config.flierprops.size,
                    color = config.flierprops.color
                ));
                svg.push('\n');
            }
            svg.push_str(&format!(
                r#"  <text x="14" y="{yc}" text-anchor="middle" font-size="11" transform="rotate(-90, 14, {yc})">{label}</text>"#,
                yc = yc
            ));
            svg.push('\n');
        }
    }

    if !config.plot_config.title.is_empty() {
        svg.push_str(&format!(
            r#"  <text x="{cx}" y="25" text-anchor="middle" font-size="20" font-weight="bold">{title}</text>"#,
            cx = width / 2.0,
            title = crate::common::xml_escape(&config.plot_config.title)
        ));
        svg.push('\n');
    }

    svg.push_str("</svg>");
    Ok(svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn datasets() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            vec![5.0, 6.0, 7.0, 8.0, 9.0],
        ]
    }

    #[test]
    fn vertical_notched_patch_render() {
        let cfg = BoxPlotConfig::new().with_notch(true);
        let labels = vec!["A".to_string(), "B".to_string()];
        let svg = render_styled_boxplot(&datasets(), &labels, &cfg).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<path")); // notched box
        assert!(svg.contains("<circle")); // flier
        assert!(svg.contains("#ff7f0e")); // flier color present
        assert!(svg.contains("#4c72b0")); // patch fill present
    }

    #[test]
    fn horizontal_render() {
        let cfg = BoxPlotConfig::new().with_vert(false);
        let labels = vec!["A".to_string(), "B".to_string()];
        let svg = render_styled_boxplot(&datasets(), &labels, &cfg).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn positions_must_match() {
        let cfg = BoxPlotConfig::new().with_positions(vec![1.0]);
        let labels = vec!["A".to_string(), "B".to_string()];
        assert!(render_styled_boxplot(&datasets(), &labels, &cfg).is_err());
    }

    #[test]
    fn empty_datasets_error() {
        assert!(render_styled_boxplot(&[], &[], &BoxPlotConfig::new()).is_err());
    }
}
