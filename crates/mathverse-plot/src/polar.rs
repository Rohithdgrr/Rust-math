//! Polar chart rendering.
//!
//! Uses `mathverse_trigonometry::{sin, cos}` for polar→Cartesian conversion.

use crate::axes::Range;
use crate::style::PlotStyle;

/// A single point in polar coordinates (theta in radians, r ≥ 0).
#[derive(Debug, Clone, Copy)]
pub struct PolarPoint {
    pub theta: f64,
    pub r: f64,
}

impl PolarPoint {
    pub fn new(theta: f64, r: f64) -> Self {
        Self { theta, r }
    }
}

/// A named series of polar points.
#[derive(Debug, Clone)]
pub struct PolarSeries {
    pub name: String,
    pub points: Vec<PolarPoint>,
    pub style: PlotStyle,
}

impl PolarSeries {
    pub fn new(name: impl Into<String>, points: Vec<PolarPoint>) -> Self {
        Self {
            name: name.into(),
            points,
            style: PlotStyle::default(),
        }
    }
}

/// Polar chart configuration.
#[derive(Debug, Clone)]
pub struct PolarData {
    pub series: Vec<PolarSeries>,
    pub title: String,
    pub r_grid_count: usize,
}

impl PolarData {
    pub fn new() -> Self {
        Self {
            series: Vec::new(),
            title: String::new(),
            r_grid_count: 5,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn add_series(&mut self, series: PolarSeries) {
        self.series.push(series);
    }
}

impl Default for PolarData {
    fn default() -> Self {
        Self::new()
    }
}

impl PolarData {
    /// Compute the maximum r across all series.
    pub fn max_r(&self) -> f64 {
        self.series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.r))
            .fold(0.0f64, f64::max)
    }
}

/// Render a polar chart to SVG.
pub fn render_polar_svg(data: &PolarData, width: u32, height: u32) -> String {
    let w = width as f64;
    let h = height as f64;
    let cx = w / 2.0;
    let cy = h / 2.0;
    let max_r = data.max_r();
    let r_range = Range {
        min: 0.0,
        max: max_r,
    }
    .pad(0.05);
    let plot_r = cx.min(cy) - 30.0; // radius in px, leave room for labels

    let to_x = |theta: f64, r: f64| -> f64 {
        cx + (r - r_range.min) / r_range.span() * plot_r * mathverse_trigonometry::cos(theta)
    };
    let to_y = |theta: f64, r: f64| -> f64 {
        cy - (r - r_range.min) / r_range.span() * plot_r * mathverse_trigonometry::sin(theta)
    };

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">"#
    ));

    // Background
    svg.push_str(&format!(
        r#"  <rect width="{w}" height="{h}" fill="white"/>"#
    ));

    // Concentric circles (r grid)
    let r_step = nice_r_step(r_range.span(), data.r_grid_count);
    let mut r_val = 0.0;
    while r_val <= r_range.max {
        let r_px = (r_val - r_range.min) / r_range.span() * plot_r;
        svg.push_str(&format!(
            r##"  <circle cx="{cx}" cy="{cy}" r="{r_px:.2}" fill="none" stroke="#ddd" stroke-width="0.5"/>"##
        ));
        // Label
        if r_val > 0.0 {
            let ly = cy - r_px - 2.0;
            svg.push_str(&format!(
                r##"  <text x="{cx}" y="{ly:.2}" font-size="10" text-anchor="middle" fill="#666">{r_val:.1}</text>"##
            ));
        }
        r_val += r_step;
    }

    // Radial lines (theta grid) every 30°
    for deg in (0..360).step_by(30) {
        let theta = deg as f64 * std::f64::consts::PI / 180.0;
        let x2 = cx + plot_r * mathverse_trigonometry::cos(theta);
        let y2 = cy - plot_r * mathverse_trigonometry::sin(theta);
        svg.push_str(&format!(
            r##"  <line x1="{cx}" y1="{cy}" x2="{x2:.2}" y2="{y2:.2}" stroke="#ddd" stroke-width="0.5"/>"##
        ));
        // Degree label
        let lx = cx + (plot_r + 12.0) * mathverse_trigonometry::cos(theta);
        let ly = cy - (plot_r + 12.0) * mathverse_trigonometry::sin(theta);
        svg.push_str(&format!(
            r##"  <text x="{lx:.2}" y="{ly:.2}" font-size="9" text-anchor="middle" fill="#888">{deg}°</text>"##
        ));
    }

    // Axes
    let axis_top = cy - plot_r;
    let axis_bot = cy + plot_r;
    let axis_left = cx - plot_r;
    let axis_right = cx + plot_r;
    svg.push_str(&format!(
        r##"  <line x1="{cx}" y1="{axis_top:.2}" x2="{cx}" y2="{axis_bot:.2}" stroke="black" stroke-width="1"/>"##
    ));
    svg.push_str(&format!(
        r##"  <line x1="{axis_left:.2}" y1="{cy}" x2="{axis_right:.2}" y2="{cy}" stroke="black" stroke-width="1"/>"##
    ));

    // Data series
    for series in &data.series {
        let c = series.style.line_color.to_hex();
        let lw = series.style.line_width;

        // Line path
        if series.points.len() >= 2 {
            let mut path = String::new();
            for (i, pt) in series.points.iter().enumerate() {
                let x = to_x(pt.theta, pt.r);
                let y = to_y(pt.theta, pt.r);
                if i == 0 {
                    path.push_str(&format!("M{x:.2},{y:.2}"));
                } else {
                    path.push_str(&format!("L{x:.2},{y:.2}"));
                }
            }
            // Close the path if it looks like a closed shape
            if series.points.len() > 2 {
                let first = &series.points[0];
                let last = series.points.last().expect("series.points is non-empty");
                let dtheta = (first.theta - last.theta).abs();
                if dtheta < 0.1 || (dtheta - 2.0 * std::f64::consts::PI).abs() < 0.1 {
                    path.push('Z');
                }
            }
            svg.push_str(&format!(
                r#"  <path d="{path}" fill="none" stroke="{c}" stroke-width="{lw}"/>"#
            ));
        }

        // Scatter dots
        for pt in &series.points {
            let x = to_x(pt.theta, pt.r);
            let y = to_y(pt.theta, pt.r);
            svg.push_str(&format!(
                r#"  <circle cx="{x:.2}" cy="{y:.2}" r="3" fill="{c}"/>"#
            ));
        }
    }

    // Title
    if !data.title.is_empty() {
        svg.push_str(&format!(
            r#"  <text x="{cx}" y="18" font-size="14" text-anchor="middle" font-weight="bold">{}</text>"#,
            escape_xml(&data.title)
        ));
    }

    svg.push_str("</svg>");
    svg
}

/// Nice step size for r-axis grid.
fn nice_r_step(range: f64, target_ticks: usize) -> f64 {
    if range <= 0.0 || target_ticks == 0 {
        return 1.0;
    }
    let raw = range / target_ticks as f64;
    let mag = 10.0_f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = if norm < 1.5 {
        1.0
    } else if norm < 3.5 {
        2.0
    } else if norm < 7.5 {
        5.0
    } else {
        10.0
    };
    step * mag
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polar_render_produces_svg() {
        let mut data = PolarData::new().with_title("Polar Test");
        let pts = vec![
            PolarPoint::new(0.0, 1.0),
            PolarPoint::new(std::f64::consts::FRAC_PI_2, 2.0),
            PolarPoint::new(std::f64::consts::PI, 1.5),
        ];
        data.add_series(PolarSeries::new("s1", pts));
        let svg = render_polar_svg(&data, 400, 400);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Polar Test"));
        assert!(svg.contains("M")); // line path exists
    }

    #[test]
    fn polar_max_r() {
        let mut data = PolarData::new();
        data.add_series(PolarSeries::new(
            "s",
            vec![PolarPoint::new(0.0, 3.0), PolarPoint::new(1.0, 5.0)],
        ));
        assert_eq!(data.max_r(), 5.0);
    }

    #[test]
    fn nice_r_step_basic() {
        assert!(nice_r_step(10.0, 5) > 0.0);
        assert!(nice_r_step(0.0, 5) > 0.0);
    }
}
