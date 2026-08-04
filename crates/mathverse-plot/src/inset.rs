//! Inset axes (zoom panels) for showing detail of a region.

use crate::axes::Range;
use crate::common::{DataPoint, DataSeries, PlotConfig};
use crate::style::Color;

/// Configuration for an inset axes panel.
#[derive(Debug, Clone)]
pub struct InsetConfig {
    /// Position of the inset in the parent plot (x, y from top-left).
    pub position: (f64, f64),
    /// Width of the inset in pixels.
    pub width: f64,
    /// Height of the inset in pixels.
    pub height: f64,
    /// X range of the region to zoom into.
    pub x_range: (f64, f64),
    /// Y range of the region to zoom into.
    pub y_range: (f64, f64),
    /// Background color.
    pub background: Color,
    /// Border color.
    pub border: Color,
    /// Border width.
    pub border_width: f64,
    /// Show connector lines from the inset to the main plot.
    pub show_connector: bool,
    /// Connector color.
    pub connector_color: Color,
    /// Padding inside the inset (pixels).
    pub padding: f64,
}

impl Default for InsetConfig {
    fn default() -> Self {
        Self {
            position: (500.0, 50.0),
            width: 150.0,
            height: 120.0,
            x_range: (0.0, 1.0),
            y_range: (0.0, 1.0),
            background: Color::WHITE,
            border: Color::BLACK,
            border_width: 1.5,
            show_connector: true,
            connector_color: Color::GRAY,
            padding: 5.0,
        }
    }
}

impl InsetConfig {
    /// Create a new inset config.
    pub fn new(x_range: (f64, f64), y_range: (f64, f64)) -> Self {
        Self {
            x_range,
            y_range,
            ..Default::default()
        }
    }

    /// Set the position.
    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.position = (x, y);
        self
    }

    /// Set the dimensions.
    pub fn with_dimensions(mut self, width: f64, height: f64) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the background color.
    pub fn with_background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Set the border color.
    pub fn with_border(mut self, color: Color) -> Self {
        self.border = color;
        self
    }

    /// Show/hide connector lines.
    pub fn with_connector(mut self, show: bool) -> Self {
        self.show_connector = show;
        self
    }
}

/// An inset axes panel that shows a zoomed view of a region.
#[derive(Debug, Clone)]
pub struct InsetAxes {
    /// Configuration.
    pub config: InsetConfig,
    /// Data series to display in the inset.
    pub series: Vec<DataSeries>,
}

impl InsetAxes {
    /// Create a new inset axes.
    pub fn new(config: InsetConfig) -> Self {
        Self {
            config,
            series: Vec::new(),
        }
    }

    /// Add a data series to the inset.
    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }

    /// Compute the pixel coordinates for a data point within the inset.
    fn data_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        let (x_min, x_max) = self.config.x_range;
        let (y_min, y_max) = self.config.y_range;
        let pad = self.config.padding;
        let w = self.config.width - 2.0 * pad;
        let h = self.config.height - 2.0 * pad;

        let px = pad + (x - x_min) / (x_max - x_min) * w;
        let py = pad + h - (y - y_min) / (y_max - y_min) * h;
        (px, py)
    }

    /// Render the inset as SVG.
    pub fn render(&self, parent_x: f64, parent_y: f64) -> String {
        let mut svg = String::new();
        let (ix, iy) = self.config.position;
        let w = self.config.width;
        let h = self.config.height;
        let pad = self.config.padding;

        // Connector lines from main plot region to inset
        if self.config.show_connector {
            let (xr0, xr1) = self.config.x_range;
            let (yr0, yr1) = self.config.y_range;

            // Four corners of the zoom region in parent coordinates
            let corners = [
                (parent_x + xr0, parent_y + yr0),
                (parent_x + xr1, parent_y + yr0),
                (parent_x + xr1, parent_y + yr1),
                (parent_x + xr0, parent_y + yr1),
            ];

            // Four corners of the inset
            let inset_corners = [
                (ix, iy + h),
                (ix + w, iy + h),
                (ix + w, iy),
                (ix, iy),
            ];

            for ((px, py), (sx, sy)) in corners.iter().zip(inset_corners.iter()) {
                svg.push_str(&format!(
                    r#"  <line x1="{px}" y1="{py}" x2="{sx}" y2="{sy}" stroke="{}" stroke-width="0.5" stroke-dasharray="3,3"/>"#,
                    self.config.connector_color.to_hex()
                ));
                svg.push('\n');
            }
        }

        // Background
        svg.push_str(&format!(
            r#"  <rect x="{ix}" y="{iy}" width="{w}" height="{h}" fill="{}" stroke="{}" stroke-width="{}"/>"#,
            self.config.background.to_hex(),
            self.config.border.to_hex(),
            self.config.border_width
        ));
        svg.push('\n');

        // Clip path
        let clip_id = format!("inset-clip-{}-{}", ix as u32, iy as u32);
        svg.push_str(&format!(
            r#"  <defs><clipPath id="{clip_id}"><rect x="{ix}" y="{iy}" width="{w}" height="{h}"/></clipPath></defs>"#,
        ));
        svg.push('\n');

        // Render series
        svg.push_str(&format!(r#"  <g clip-path="url(#{clip_id})">"#));
        svg.push('\n');

        for series in &self.series {
            if series.points.len() > 1 {
                let points: Vec<String> = series
                    .points
                    .iter()
                    .filter_map(|p| {
                        let (px, py) = self.data_to_pixel(p.x, p.y);
                        // Only include points within the clip region
                        if px >= 0.0 && px <= w && py >= 0.0 && py <= h {
                            Some(format!("{},{}", ix + px, iy + py))
                        } else {
                            None
                        }
                    })
                    .collect();

                if points.len() > 1 {
                    svg.push_str(&format!(
                        r#"    <polyline points="{}" fill="none" stroke="{}" stroke-width="1.5"/>"#,
                        points.join(" "),
                        series.style.line_color.to_hex()
                    ));
                    svg.push('\n');
                }
            }
        }

        svg.push_str("  </g>\n");

        // Small tick marks
        let (x_min, x_max) = self.config.x_range;
        let (y_min, y_max) = self.config.y_range;

        // X ticks (just min and max)
        for &val in &[x_min, x_max] {
            let (px, _) = self.data_to_pixel(val, y_min);
            let tick_x = ix + px;
            let tick_y = iy + h;
            svg.push_str(&format!(
                r#"  <line x1="{tick_x}" y1="{tick_y}" x2="{tick_x}" y2="{}" stroke="black" stroke-width="0.5"/>"#,
                tick_y + 3.0
            ));
            svg.push('\n');
            svg.push_str(&format!(
                r#"  <text x="{tick_x}" y="{}" text-anchor="middle" font-size="7">{:.1}</text>"#,
                tick_y + 10.0,
                val
            ));
            svg.push('\n');
        }

        // Y ticks (just min and max)
        for &val in &[y_min, y_max] {
            let (_, py) = self.data_to_pixel(x_min, val);
            let tick_x = ix;
            let tick_y = iy + py;
            svg.push_str(&format!(
                r#"  <line x1="{}" y1="{tick_y}" x2="{tick_x}" y2="{tick_y}" stroke="black" stroke-width="0.5"/>"#,
                tick_x - 3.0
            ));
            svg.push('\n');
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" text-anchor="end" font-size="7">{:.1}</text>"#,
                tick_x - 4.0,
                tick_y + 3.0,
                val
            ));
            svg.push('\n');
        }

        svg
    }
}

/// A collection of inset axes for a plot.
#[derive(Debug, Clone, Default)]
pub struct Insets {
    /// List of inset axes.
    pub insets: Vec<InsetAxes>,
}

impl Insets {
    /// Create a new empty insets collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an inset.
    pub fn add(mut self, inset: InsetAxes) -> Self {
        self.insets.push(inset);
        self
    }

    /// Render all insets as SVG.
    pub fn render(&self, parent_x: f64, parent_y: f64) -> String {
        self.insets
            .iter()
            .map(|inset| inset.render(parent_x, parent_y))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::PlotStyle;

    #[test]
    fn inset_basic() {
        let config = InsetConfig::new((0.5, 1.5), (0.5, 1.5));
        let mut inset = InsetAxes::new(config);
        let points = vec![DataPoint::new(0.6, 0.6), DataPoint::new(1.4, 1.4)];
        inset.add_series(DataSeries::new("zoomed".to_string(), points));
        let svg = inset.render(50.0, 50.0);
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<polyline"));
    }

    #[test]
    fn inset_with_connector() {
        let config = InsetConfig::new((0.0, 1.0), (0.0, 1.0)).with_connector(true);
        let inset = InsetAxes::new(config);
        let svg = inset.render(100.0, 100.0);
        assert!(svg.contains("<line"));
    }

    #[test]
    fn insets_collection() {
        let insets = Insets::new()
            .add(InsetAxes::new(InsetConfig::new((0.0, 1.0), (0.0, 1.0))))
            .add(InsetAxes::new(InsetConfig::new((2.0, 3.0), (2.0, 3.0))));
        assert_eq!(insets.insets.len(), 2);
    }
}
