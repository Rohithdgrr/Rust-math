//! Streamplot — trace streamlines through a 2D vector field, the analogue of
//! matplotlib's `plt.streamplot`.
//!
//! Given a grid of `(u, v)` velocity components, streamlines are integrated
//! forward and backward from seed points with a fixed step, then emitted as
//! SVG polylines. Direction of flow is shown by arrowheads near the midpoint
//! of each line.

use crate::common::PlotConfig;
use crate::error::{PlotError, PlotResult};

/// Configuration for a streamplot.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Chart configuration (dimensions, padding, title...).
    pub plot_config: PlotConfig,
    /// Integration step size in data units per step.
    pub step: f64,
    /// Maximum number of integration steps per streamline (line length cap).
    pub max_steps: usize,
    /// Line color as `#rrggbb`.
    pub color: String,
    /// Line width.
    pub line_width: f64,
    /// Number of seed rows (seeds are laid out on a `seeds x seeds` grid).
    pub seeds: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            plot_config: PlotConfig::new(),
            step: 0.05,
            max_steps: 200,
            color: "#1f77b4".to_string(),
            line_width: 1.2,
            seeds: 6,
        }
    }
}

impl StreamConfig {
    /// Create a new streamplot config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the integration step size.
    #[must_use]
    pub fn with_step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    /// Set the color.
    #[must_use]
    pub fn with_color(mut self, color: impl Into<String>) -> Self {
        self.color = color.into();
        self
    }

    /// Set the seed grid density.
    #[must_use]
    pub fn with_seeds(mut self, seeds: usize) -> Self {
        self.seeds = seeds.max(2);
        self
    }
}

/// Trace one streamline from `(x0, y0)` by Euler integration of the field,
/// returning the sampled points (already clipped to the data range).
fn trace_stream(
    x0: f64,
    y0: f64,
    u: &[Vec<f64>],
    v: &[Vec<f64>],
    x_range: (f64, f64),
    y_range: (f64, f64),
    step: f64,
    max_steps: usize,
) -> Vec<(f64, f64)> {
    let rows = v.len();
    let cols = u[0].len();
    let dx = (x_range.1 - x_range.0) / (cols - 1).max(1) as f64;
    let dy = (y_range.1 - y_range.0) / (rows - 1).max(1) as f64;

    let sample = |x: f64, y: f64| -> (f64, f64) {
        let c = ((x - x_range.0) / dx).floor() as isize;
        let r = ((y - y_range.0) / dy).floor() as isize;
        let c = c.clamp(0, cols as isize - 1) as usize;
        let r = r.clamp(0, rows as isize - 1) as usize;
        (u[r][c], v[r][c])
    };

    let mut pts = Vec::with_capacity(max_steps + 1);
    let (mut x, mut y) = (x0, y0);
    for _ in 0..max_steps {
        pts.push((x, y));
        let (ux, uy) = sample(x, y);
        let norm = (ux * ux + uy * uy).sqrt();
        if norm < 1e-12 {
            break; // stagnation point
        }
        x += ux / norm * step;
        y += uy / norm * step;
        if x < x_range.0 || x > x_range.1 || y < y_range.0 || y > y_range.1 {
            break; // left the domain
        }
    }
    pts
}

/// Render a streamplot as SVG.
///
/// # Errors
///
/// Returns `PlotError::InvalidData` for empty or ragged grids.
pub fn render_streamplot(
    u: &[Vec<f64>],
    v: &[Vec<f64>],
    x_range: (f64, f64),
    y_range: (f64, f64),
    config: &StreamConfig,
) -> PlotResult<String> {
    if u.is_empty() || u[0].is_empty() {
        return Err(PlotError::InvalidData("empty vector field grid".into()));
    }
    if u.len() != v.len() || u[0].len() != v[0].len() {
        return Err(PlotError::InvalidData("u and v grids must match".into()));
    }
    if u.iter().any(|row| row.len() != u[0].len()) || v.iter().any(|row| row.len() != v[0].len()) {
        return Err(PlotError::InvalidData("ragged vector field grid".into()));
    }

    let padding = config.plot_config.padding;
    let width = config.plot_config.width as f64;
    let height = config.plot_config.height as f64;
    let chart_width = width - 2.0 * padding;
    let chart_height = height - 2.0 * padding;

    let to_x = |x: f64| padding + (x - x_range.0) / (x_range.1 - x_range.0) * chart_width;
    let to_y = |y: f64| {
        padding + chart_height * (1.0 - (y - y_range.0) / (y_range.1 - y_range.0))
    };

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">"#,
        w = width as u32,
        h = height as u32
    ));
    svg.push('\n');
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");

    let n = config.seeds;
    for si in 0..n {
        for sj in 0..n {
            // Seed slightly inside the domain to avoid edge artifacts.
            let fx = (sj as f64 + 0.5) / n as f64;
            let fy = (si as f64 + 0.5) / n as f64;
            let x0 = x_range.0 + fx * (x_range.1 - x_range.0);
            let y0 = y_range.0 + fy * (y_range.1 - y_range.0);
            let mut line: Vec<(f64, f64)> =
                trace_stream(x0, y0, u, v, x_range, y_range, config.step, config.max_steps);

            // Arrowheads: a short segment near the end pointing along flow.
            if line.len() >= 4 {
                let n_pts = line.len();
                let tip = line[n_pts - 1];
                let base = line[n_pts - 3];
                let (dx, dy) = (tip.0 - base.0, tip.1 - base.1);
                let len = (dx * dx + dy * dy).sqrt();
                if len > 1e-9 {
                    let (ux, uy) = (dx / len, dy / len);
                    let (px, py) = (tip.0 - ux * config.step * 2.0, tip.1 - uy * config.step * 2.0);
                    let perp = (-uy, ux);
                    let size = config.step * 1.6;
                    let a1 = (px + perp.0 * size, py + perp.1 * size);
                    let a2 = (px - perp.0 * size, py - perp.1 * size);
                    line.push(a1);
                    line.push(tip);
                    line.push(a2);
                }
            }

            if line.len() < 2 {
                continue;
            }
            let coords: Vec<String> = line
                .iter()
                .map(|&(x, y)| format!("{:.2},{:.2}", to_x(x), to_y(y)))
                .collect();
            svg.push_str(&format!(
                r#"  <polyline points="{}" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"/>{}"#,
                coords.join(" "),
                config.color,
                config.line_width,
                "\n"
            ));
        }
    }

    // Axes labels
    svg.push_str(&format!(
        r#"  <text x="{cx}" y="{cy}" text-anchor="middle" font-size="11">x</text>"#,
        cx = width / 2.0,
        cy = height - 5.0
    ));
    svg.push('\n');
    svg.push_str(&format!(
        r#"  <text x="10" y="{cy}" text-anchor="middle" font-size="11" transform="rotate(-90, 10, {cy})">y</text>"#,
        cy = height / 2.0
    ));
    svg.push('\n');

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

    fn vortex() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = 20;
        let u = (0..n)
            .map(|j| {
                (0..n)
                    .map(|i| {
                        let x = i as f64 / (n - 1) as f64 - 0.5;
                        let y = j as f64 / (n - 1) as f64 - 0.5;
                        -y
                    })
                    .collect()
            })
            .collect();
        let v = (0..n)
            .map(|j| {
                (0..n)
                    .map(|i| {
                        let x = i as f64 / (n - 1) as f64 - 0.5;
                        let y = j as f64 / (n - 1) as f64 - 0.5;
                        x
                    })
                    .collect()
            })
            .collect();
        (u, v)
    }

    #[test]
    fn streamplot_renders_polylines() {
        let (u, v) = vortex();
        let config = StreamConfig::new().with_seeds(4);
        let svg = render_streamplot(&u, &v, (0.0, 1.0), (0.0, 1.0), &config).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("<polyline"));
    }

    #[test]
    fn streamplot_rejects_empty() {
        let u: Vec<Vec<f64>> = vec![];
        let v: Vec<Vec<f64>> = vec![];
        assert!(render_streamplot(&u, &v, (0.0, 1.0), (0.0, 1.0), &StreamConfig::new()).is_err());
    }

    #[test]
    fn streamplot_rejects_mismatched_grids() {
        let u = vec![vec![1.0; 5]; 5];
        let v = vec![vec![1.0; 4]; 5];
        assert!(render_streamplot(&u, &v, (0.0, 1.0), (0.0, 1.0), &StreamConfig::new()).is_err());
    }

    #[test]
    fn trace_stays_in_domain() {
        let (u, v) = vortex();
        let pts = trace_stream(0.1, 0.1, &u, &v, (0.0, 1.0), (0.0, 1.0), 0.02, 100);
        assert!(pts.len() >= 2);
        assert!(pts
            .iter()
            .all(|&(x, y)| x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0));
    }
}
