//! Quiver (vector field) plot rendering.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single vector in a quiver plot.
#[derive(Debug, Clone)]
pub struct QuiverVector {
    /// X position.
    pub x: f64,
    /// Y position.
    pub y: f64,
    /// X component of the vector.
    pub u: f64,
    /// Y component of the vector.
    pub v: f64,
}

impl QuiverVector {
    /// Create a new quiver vector.
    pub fn new(x: f64, y: f64, u: f64, v: f64) -> Self {
        Self { x, y, u, v }
    }
}

/// Configuration for a quiver plot.
#[derive(Debug, Clone)]
pub struct QuiverConfig {
    /// Chart configuration.
    pub plot_config: PlotConfig,
    /// Arrow color.
    pub color: Color,
    /// Arrow width.
    pub arrow_width: f64,
    /// Scale factor for arrows.
    pub scale: f64,
    /// Show grid.
    pub show_grid: bool,
    /// Normalize arrows (all same length).
    pub normalize: bool,
}

impl Default for QuiverConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            color: Color::BLUE,
            arrow_width: 2.0,
            scale: 30.0,
            show_grid: true,
            normalize: false,
        }
    }
}

impl QuiverConfig {
    /// Create a new quiver config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set scale.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Set color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }
}

/// Render a quiver (vector field) plot as SVG.
pub fn render_quiver(vectors: &[QuiverVector], config: &QuiverConfig) -> PlotResult<String> {
    if vectors.is_empty() {
        return Err(PlotError::InvalidData("no vectors provided".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;

    // Find bounds
    let min_x = vectors.iter().map(|v| v.x).fold(f64::MAX, f64::min);
    let max_x = vectors.iter().map(|v| v.x).fold(f64::MIN, f64::max);
    let min_y = vectors.iter().map(|v| v.y).fold(f64::MAX, f64::min);
    let max_y = vectors.iter().map(|v| v.y).fold(f64::MIN, f64::max);

    // Find max magnitude for normalization
    let max_mag = vectors
        .iter()
        .map(|v| (v.u * v.u + v.v * v.v).sqrt())
        .fold(0.0_f64, f64::max);

    if max_mag == 0.0 {
        return Err(PlotError::InvalidData("all vectors have zero magnitude".into()));
    }

    let chart_width = width - padding * 2.0;
    let chart_height = height - padding * 2.0 - 30.0;

    let to_x = |x| padding + (x - min_x) / (max_x - min_x) * chart_width;
    let to_y = |y| padding + 30.0 + chart_height * (1.0 - (y - min_y) / (max_y - min_y));

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
        let y_bottom = height - padding;
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
        for i in 0..=5 {
            let x = padding + (i as f64 / 5.0) * chart_width;
            svg.push_str("  <line x1=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y1=\"");
            svg.push_str(&padding.to_string());
            svg.push_str("\" x2=\"");
            svg.push_str(&x.to_string());
            svg.push_str("\" y2=\"");
            svg.push_str(&y_bottom.to_string());
            svg.push_str("\" stroke=\"#eee\"/>\n");
        }
    }

    // Draw vectors
    for v in vectors {
        let cx = to_x(v.x);
        let cy = to_y(v.y);

        let (u_scaled, v_scaled) = if config.normalize {
            let mag = (v.u * v.u + v.v * v.v).sqrt();
            if mag > 0.0 {
                (v.u / mag * config.scale, v.v / mag * config.scale)
            } else {
                (0.0, 0.0)
            }
        } else {
            let mag = (v.u * v.u + v.v * v.v).sqrt();
            let factor = mag / max_mag * config.scale;
            (v.u / mag * factor, v.v / mag * factor)
        };

        // Arrow endpoint
        let ex = cx + u_scaled;
        let ey = cy - v_scaled; // SVG Y is inverted

        // Arrow line
        svg.push_str(&format!(
            r#"  <line x1="{cx}" y1="{cy}" x2="{ex}" y2="{ey}" stroke="{}" stroke-width="{}"/>"#,
            config.color.to_hex(),
            config.arrow_width
        ));
        svg.push('\n');

        // Arrowhead
        let angle = (-v_scaled).atan2(u_scaled);
        let head_len = 8.0;
        let head_angle = 0.5; // ~30 degrees

        let hx1 = ex - head_len * (angle - head_angle).cos();
        let hy1 = ey - head_len * (angle - head_angle).sin();
        let hx2 = ex - head_len * (angle + head_angle).cos();
        let hy2 = ey - head_len * (angle + head_angle).sin();

        svg.push_str(&format!(
            r#"  <polygon points="{ex},{ey} {hx1},{hy1} {hx2},{hy2}" fill="{}"/>"#,
            config.color.to_hex()
        ));
        svg.push('\n');
    }

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
    fn quiver_plot_renders() {
        let vectors = vec![
            QuiverVector::new(0.0, 0.0, 1.0, 1.0),
            QuiverVector::new(1.0, 0.0, 0.0, 1.0),
            QuiverVector::new(0.0, 1.0, -1.0, 0.0),
            QuiverVector::new(1.0, 1.0, 0.5, -0.5),
        ];
        let config = QuiverConfig::new();
        let svg = render_quiver(&vectors, &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<line"));
        assert!(svg.contains("<polygon"));
    }

    #[test]
    fn quiver_plot_empty_error() {
        let vectors = vec![];
        let config = QuiverConfig::new();
        assert!(render_quiver(&vectors, &config).is_err());
    }
}
