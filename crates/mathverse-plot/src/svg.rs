//! SVG plotting backend

use crate::common::{DataPoint, DataSeries, PlotConfig};
use crate::style::{Color, LineStyle, MarkerStyle};

/// SVG plot generator
pub struct SvgPlot {
    config: PlotConfig,
    series: Vec<DataSeries>,
}

impl SvgPlot {
    /// Create a new SVG plot
    pub fn new(config: PlotConfig) -> Self {
        SvgPlot {
            config,
            series: Vec::new(),
        }
    }

    /// Add a data series to the plot
    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }

    /// Generate the SVG string
    pub fn generate(&self) -> String {
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            self.config.width, self.config.height
        ));
        svg.push('\n');

        // Background
        let bg_color = self.series.first()
            .map(|s| s.style.background_color.to_hex())
            .unwrap_or_else(|| Color::WHITE.to_hex());
        svg.push_str(&format!(
            r#"  <rect width="100%" height="100%" fill="{}"/>"#,
            bg_color
        ));
        svg.push('\n');

        // Calculate plot area
        let padding = self.config.padding;
        let plot_width = self.config.width as f64 - 2.0 * padding;
        let plot_height = self.config.height as f64 - 2.0 * padding;

        // Calculate data ranges
        let (x_min, x_max, y_min, y_max) = self.calculate_ranges();

        // Draw grid if enabled
        if self.config.show_grid {
            svg.push_str(&self.generate_grid(padding, plot_width, plot_height, x_min, x_max, y_min, y_max));
        }

        // Draw axes
        svg.push_str(&self.generate_axes(padding, plot_width, plot_height));

        // Draw data series
        for series in &self.series {
            svg.push_str(&self.generate_series(series, padding, plot_width, plot_height, x_min, x_max, y_min, y_max));
        }

        // Draw title
        if !self.config.title.is_empty() {
            svg.push_str(&format!(
                r#"  <text x="{}" y="30" text-anchor="middle" font-size="20">{}</text>"#,
                self.config.width as f64 / 2.0,
                self.config.title
            ));
            svg.push('\n');
        }

        // Draw axis labels
        if !self.config.x_label.is_empty() {
            svg.push_str(&format!(
                r#"  <text x="{}" y="{}" text-anchor="middle" font-size="14">{}</text>"#,
                self.config.width as f64 / 2.0,
                self.config.height as f64 - 10.0,
                self.config.x_label
            ));
            svg.push('\n');
        }

        if !self.config.y_label.is_empty() {
            svg.push_str(&format!(
                r#"  <text x="20" y="{}" text-anchor="middle" font-size="14" transform="rotate(-90, 20, {})">{}</text>"#,
                self.config.height as f64 / 2.0,
                self.config.height as f64 / 2.0,
                self.config.y_label
            ));
            svg.push('\n');
        }

        // Draw legend if enabled
        if self.config.show_legend && !self.series.is_empty() {
            svg.push_str(&self.generate_legend());
        }

        // SVG footer
        svg.push_str("</svg>");

        svg
    }

    fn calculate_ranges(&self) -> (f64, f64, f64, f64) {
        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for series in &self.series {
            for point in &series.points {
                x_min = x_min.min(point.x);
                x_max = x_max.max(point.x);
                y_min = y_min.min(point.y);
                y_max = y_max.max(point.y);
            }
        }

        // Add some padding to ranges
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        x_min -= x_range * 0.05;
        x_max += x_range * 0.05;
        y_min -= y_range * 0.05;
        y_max += y_range * 0.05;

        (x_min, x_max, y_min, y_max)
    }

    fn generate_grid(&self, padding: f64, width: f64, height: f64, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> String {
        let mut grid = String::new();
        let grid_color = self.series.first()
            .map(|s| s.style.grid_color.to_hex())
            .unwrap_or_else(|| Color::GRAY.to_hex());

        // Vertical grid lines
        let x_steps = 10;
        for i in 0..=x_steps {
            let x = padding + (i as f64 / x_steps as f64) * width;
            grid.push_str(&format!(
                r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="0.5" opacity="0.5"/>"#,
                x, padding, x, padding + height, grid_color
            ));
            grid.push('\n');
        }

        // Horizontal grid lines
        let y_steps = 10;
        for i in 0..=y_steps {
            let y = padding + (i as f64 / y_steps as f64) * height;
            grid.push_str(&format!(
                r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="0.5" opacity="0.5"/>"#,
                padding, y, padding + width, y, grid_color
            ));
            grid.push('\n');
        }

        grid
    }

    fn generate_axes(&self, padding: f64, width: f64, height: f64) -> String {
        let mut axes = String::new();
        let axis_color = Color::BLACK.to_hex();

        // X-axis
        axes.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
            padding, padding + height, padding + width, padding + height, axis_color
        ));
        axes.push('\n');

        // Y-axis
        axes.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
            padding, padding, padding, padding + height, axis_color
        ));
        axes.push('\n');

        axes
    }

    fn generate_series(&self, series: &DataSeries, padding: f64, width: f64, height: f64, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> String {
        let mut output = String::new();
        let style = &series.style;

        // Convert data points to SVG coordinates
        let points: Vec<(f64, f64)> = series.points.iter().map(|p| {
            let x = padding + ((p.x - x_min) / (x_max - x_min)) * width;
            let y = padding + height - ((p.y - y_min) / (y_max - y_min)) * height;
            (x, y)
        }).collect();

        // Draw line
        if points.len() > 1 {
            let line_color = style.line_color.to_hex();
            let line_width = style.line_width;
            let dash_array = match style.line_style {
                LineStyle::Solid => "none",
                LineStyle::Dashed => "5,5",
                LineStyle::Dotted => "2,2",
                LineStyle::DashDot => "5,2,2,2",
            };

            output.push_str(&format!(
                r#"  <polyline points="{}" fill="none" stroke="{}" stroke-width="{}" stroke-dasharray="{}"/>"#,
                points.iter().map(|(x, y)| format!("{},{}", x, y)).collect::<Vec<_>>().join(" "),
                line_color, line_width, dash_array
            ));
            output.push('\n');
        }

        // Draw markers
        if style.marker_style != MarkerStyle::None {
            let marker_color = style.marker_color.to_hex();
            let marker_size = style.marker_size;

            for (x, y) in &points {
                let marker = match style.marker_style {
                    MarkerStyle::Circle => {
                        format!(r#"  <circle cx="{}" cy="{}" r="{}" fill="{}"/>"#, x, y, marker_size, marker_color)
                    }
                    MarkerStyle::Square => {
                        format!(r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#, 
                            x - marker_size, y - marker_size, marker_size * 2.0, marker_size * 2.0, marker_color)
                    }
                    MarkerStyle::Triangle => {
                        format!(r#"  <polygon points="{},{} {},{} {},{}" fill="{}"/>"#,
                            x, y - marker_size, x - marker_size, y + marker_size, x + marker_size, y + marker_size, marker_color)
                    }
                    MarkerStyle::Cross | MarkerStyle::Plus => {
                        format!(r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
                            x - marker_size, y, x + marker_size, y, marker_color)
                    }
                    MarkerStyle::Diamond => {
                        format!(r#"  <polygon points="{},{} {},{} {},{} {},{}" fill="{}"/>"#,
                            x, y - marker_size, x + marker_size, y, x, y + marker_size, x - marker_size, y, marker_color)
                    }
                    MarkerStyle::None => String::new(),
                };
                output.push_str(&marker);
                output.push('\n');
            }
        }

        output
    }

    fn generate_legend(&self) -> String {
        let mut legend = String::new();
        let legend_x = self.config.width as f64 - 150.0;
        let legend_y = 50.0;

        legend.push_str(&format!(r#"  <rect x="{}" y="{}" width="140" height="{}" fill="white" stroke="black" opacity="0.9"/>"#,
            legend_x, legend_y, self.series.len() as f64 * 25.0 + 10.0));
        legend.push('\n');

        for (i, series) in self.series.iter().enumerate() {
            let y = legend_y + 10.0 + i as f64 * 25.0;
            let color = series.style.line_color.to_hex();

            legend.push_str(&format!(
                r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
                legend_x + 10.0, y + 10.0, legend_x + 40.0, y + 10.0, color
            ));
            legend.push('\n');

            legend.push_str(&format!(
                r#"  <text x="{}" y="{}" font-size="12">{}</text>"#,
                legend_x + 50.0, y + 15.0, series.name
            ));
            legend.push('\n');
        }

        legend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::PlotStyle;

    #[test]
    fn test_svg_plot_creation() {
        let config = PlotConfig::new()
            .with_title("Test Plot".to_string())
            .with_dimensions(800, 600);

        let mut plot = SvgPlot::new(config);

        let points = vec![
            DataPoint::new(1.0, 2.0),
            DataPoint::new(2.0, 4.0),
            DataPoint::new(3.0, 6.0),
        ];

        let series = DataSeries::new("Test Series".to_string(), points);
        plot.add_series(series);

        let svg = plot.generate();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Test Plot"));
    }
}
