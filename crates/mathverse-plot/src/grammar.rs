//! Grammar-of-graphics API for building plots declaratively.
//!
//! This module provides a fluent API inspired by ggplot2 and Vega-Lite
//! for constructing plots using a grammar-based approach.
//!
//! # Example
//!
//! ```rust
//! use mathverse_plot::grammar::{Plot, Aes, Geom};
//! use mathverse_plot::style::Color;
//!
//! let svg = Plot::new()
//!     .aes(Aes::new().x("x").y("y").color("group"))
//!     .geom_point()
//!     .theme_seaborn()
//!     .title("Scatter Plot")
//!     .x_label("X Axis")
//!     .y_label("Y Axis")
//!     .render();
//! ```

use std::collections::HashMap;

use crate::axes::Scale;
use crate::common::{DataPoint, DataSeries, PlotConfig};
use crate::style::{Color, PlotStyle};
use crate::svg::SvgPlot;
use crate::theme::{Theme, ThemeConfig};

/// Aesthetic mappings for plot variables.
#[derive(Debug, Clone, Default)]
pub struct Aes {
    /// X-axis field name.
    pub x: Option<String>,
    /// Y-axis field name.
    pub y: Option<String>,
    /// Color/group field name.
    pub color: Option<String>,
    /// Size field name.
    pub size: Option<String>,
    /// Shape/fill field name.
    pub shape: Option<String>,
    /// Alpha/opacity field name.
    pub alpha: Option<String>,
}

impl Aes {
    /// Create a new empty aesthetic mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the x-axis field.
    pub fn x(mut self, field: impl Into<String>) -> Self {
        self.x = Some(field.into());
        self
    }

    /// Set the y-axis field.
    pub fn y(mut self, field: impl Into<String>) -> Self {
        self.y = Some(field.into());
        self
    }

    /// Set the color/group field.
    pub fn color(mut self, field: impl Into<String>) -> Self {
        self.color = Some(field.into());
        self
    }

    /// Set the size field.
    pub fn size(mut self, field: impl Into<String>) -> Self {
        self.size = Some(field.into());
        self
    }

    /// Set the shape field.
    pub fn shape(mut self, field: impl Into<String>) -> Self {
        self.shape = Some(field.into());
        self
    }

    /// Set the alpha field.
    pub fn alpha(mut self, field: impl Into<String>) -> Self {
        self.alpha = Some(field.into());
        self
    }
}

/// Geometry types for the grammar-of-graphics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geom {
    /// Scatter points.
    Point,
    /// Line plot.
    Line,
    /// Bar chart.
    Bar,
    /// Area chart (filled line).
    Area,
    /// Histogram.
    Histogram,
    /// Box plot.
    BoxPlot,
    /// Violin plot.
    Violin,
    /// Heatmap.
    Heatmap,
    /// Error bars.
    ErrorBar,
    /// Smooth trend line.
    Smooth,
    /// Text annotations.
    Text,
    /// Ribbon (confidence interval).
    Ribbon,
}

/// A geometric layer in the plot.
#[derive(Debug, Clone)]
pub struct Layer {
    /// The geometry type.
    pub geom: Geom,
    /// Additional aesthetic mappings for this layer.
    pub aes: Option<Aes>,
    /// Style overrides for this layer.
    pub style: Option<PlotStyle>,
    /// Data for this layer (overrides plot-level data).
    pub data: Option<Vec<HashMap<String, f64>>>,
}

impl Layer {
    /// Create a new layer with the given geometry.
    pub fn new(geom: Geom) -> Self {
        Self {
            geom,
            aes: None,
            style: None,
            data: None,
        }
    }

    /// Set aesthetic mappings for this layer.
    pub fn aes(mut self, aes: Aes) -> Self {
        self.aes = Some(aes);
        self
    }

    /// Set the style for this layer.
    pub fn style(mut self, style: PlotStyle) -> Self {
        self.style = Some(style);
        self
    }

    /// Set data for this layer.
    pub fn data(mut self, data: Vec<HashMap<String, f64>>) -> Self {
        self.data = Some(data);
        self
    }
}

/// A grammar-of-graphics plot builder.
#[derive(Debug, Clone)]
pub struct Plot {
    /// Plot-level aesthetic mappings.
    aes: Aes,
    /// Plot-level data.
    data: Vec<HashMap<String, f64>>,
    /// Layers (geometries) to render.
    layers: Vec<Layer>,
    /// Theme configuration.
    theme: Option<ThemeConfig>,
    /// Plot title.
    title: String,
    /// X-axis label.
    x_label: String,
    /// Y-axis label.
    y_label: String,
    /// X-axis scale.
    x_scale: Scale,
    /// Y-axis scale.
    y_scale: Scale,
    /// Legend configuration.
    legend: Option<crate::legend::LegendConfig>,
    /// Plot dimensions.
    width: u32,
    height: u32,
    /// Whether to show grid lines.
    show_grid: bool,
}

impl Default for Plot {
    fn default() -> Self {
        Self::new()
    }
}

impl Plot {
    /// Create a new empty plot.
    pub fn new() -> Self {
        Self {
            aes: Aes::new(),
            data: Vec::new(),
            layers: Vec::new(),
            theme: None,
            title: String::new(),
            x_label: String::new(),
            y_label: String::new(),
            x_scale: Scale::Linear,
            y_scale: Scale::Linear,
            legend: None,
            width: 800,
            height: 600,
            show_grid: true,
        }
    }

    /// Set the plot-level aesthetic mappings.
    pub fn aes(mut self, aes: Aes) -> Self {
        self.aes = aes;
        self
    }

    /// Set the plot data.
    pub fn data(mut self, data: Vec<HashMap<String, f64>>) -> Self {
        self.data = data;
        self
    }

    /// Add a layer with the given geometry.
    pub fn layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Add a point layer (scatter plot).
    pub fn geom_point(self) -> Self {
        self.layer(Layer::new(Geom::Point))
    }

    /// Add a line layer.
    pub fn geom_line(self) -> Self {
        self.layer(Layer::new(Geom::Line))
    }

    /// Add a bar layer.
    pub fn geom_bar(self) -> Self {
        self.layer(Layer::new(Geom::Bar))
    }

    /// Add an area layer.
    pub fn geom_area(self) -> Self {
        self.layer(Layer::new(Geom::Area))
    }

    /// Add a histogram layer.
    pub fn geom_histogram(self) -> Self {
        self.layer(Layer::new(Geom::Histogram))
    }

    /// Add a box plot layer.
    pub fn geom_boxplot(self) -> Self {
        self.layer(Layer::new(Geom::BoxPlot))
    }

    /// Add a violin plot layer.
    pub fn geom_violin(self) -> Self {
        self.layer(Layer::new(Geom::Violin))
    }

    /// Add a heatmap layer.
    pub fn geom_heatmap(self) -> Self {
        self.layer(Layer::new(Geom::Heatmap))
    }

    /// Add an error bar layer.
    pub fn geom_errorbar(self) -> Self {
        self.layer(Layer::new(Geom::ErrorBar))
    }

    /// Add a smooth trend line layer.
    pub fn geom_smooth(self) -> Self {
        self.layer(Layer::new(Geom::Smooth))
    }

    /// Add a text annotation layer.
    pub fn geom_text(self) -> Self {
        self.layer(Layer::new(Geom::Text))
    }

    /// Add a ribbon layer (confidence interval).
    pub fn geom_ribbon(self) -> Self {
        self.layer(Layer::new(Geom::Ribbon))
    }

    /// Set the plot title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the x-axis label.
    pub fn x_label(mut self, label: impl Into<String>) -> Self {
        self.x_label = label.into();
        self
    }

    /// Set the y-axis label.
    pub fn y_label(mut self, label: impl Into<String>) -> Self {
        self.y_label = label.into();
        self
    }

    /// Set the x-axis scale.
    pub fn x_scale(mut self, scale: Scale) -> Self {
        self.x_scale = scale;
        self
    }

    /// Set the y-axis scale.
    pub fn y_scale(mut self, scale: Scale) -> Self {
        self.y_scale = scale;
        self
    }

    /// Set the plot dimensions.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the legend configuration.
    pub fn legend(mut self, legend: crate::legend::LegendConfig) -> Self {
        self.legend = Some(legend);
        self
    }

    /// Set the theme.
    pub fn theme(mut self, theme: ThemeConfig) -> Self {
        self.theme = Some(theme);
        self
    }

    /// Apply a built-in theme.
    pub fn with_theme_enum(mut self, theme: Theme) -> Self {
        self.theme = Some(ThemeConfig::new(theme));
        self
    }

    /// Apply the seaborn theme.
    pub fn theme_seaborn(self) -> Self {
        self.with_theme_enum(Theme::Seaborn)
    }

    /// Apply the ggplot theme.
    pub fn theme_ggplot(self) -> Self {
        self.with_theme_enum(Theme::Ggplot)
    }

    /// Apply the dark theme.
    pub fn theme_dark(self) -> Self {
        self.with_theme_enum(Theme::Dark)
    }

    /// Apply the minimal theme.
    pub fn theme_minimal(self) -> Self {
        self.with_theme_enum(Theme::Minimal)
    }

    /// Apply the classic theme.
    pub fn theme_classic(self) -> Self {
        self.with_theme_enum(Theme::Classic)
    }

    /// Apply the high contrast theme.
    pub fn theme_high_contrast(self) -> Self {
        self.with_theme_enum(Theme::HighContrast)
    }

    /// Apply the academic theme.
    pub fn theme_academic(self) -> Self {
        self.with_theme_enum(Theme::Academic)
    }

    /// Apply the presentation theme.
    pub fn theme_presentation(self) -> Self {
        self.with_theme_enum(Theme::Presentation)
    }

    /// Apply the blueprint theme.
    pub fn theme_blueprint(self) -> Self {
        self.with_theme_enum(Theme::Blueprint)
    }

    /// Set whether to show grid lines.
    pub fn show_grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Render the plot to SVG.
    pub fn render(&self) -> String {
        // Convert plot data to DataSeries
        let series = self.build_series();

        // Create PlotConfig
        let config = PlotConfig {
            width: self.width,
            height: self.height,
            title: self.title.clone(),
            x_label: self.x_label.clone(),
            y_label: self.y_label.clone(),
            x_scale: self.x_scale,
            y_scale: self.y_scale,
            show_grid: self.show_grid,
            ..Default::default()
        };

        // Build SvgPlot
        let mut svg_plot = SvgPlot::new(config);

        // Apply theme
        if let Some(theme) = &self.theme {
            svg_plot = svg_plot.with_theme(theme.clone());
        }

        // Apply legend
        if let Some(legend) = &self.legend {
            svg_plot = svg_plot.with_legend(legend.clone());
        }

        // Add series
        for s in series {
            svg_plot.add_series(s);
        }

        svg_plot.generate()
    }

    /// Build DataSeries from plot data and aesthetic mappings.
    fn build_series(&self) -> Vec<DataSeries> {
        let mut series_map: HashMap<String, Vec<DataPoint>> = HashMap::new();

        // Extract field names
        let x_field = self.aes.x.as_deref().unwrap_or("x");
        let y_field = self.aes.y.as_deref().unwrap_or("y");
        let color_field = self.aes.color.as_deref();

        // Group data by color field if specified
        for point in &self.data {
            let x = point.get(x_field).copied().unwrap_or(0.0);
            let y = point.get(y_field).copied().unwrap_or(0.0);

            let group_key = if let Some(cf) = color_field {
                point.get(cf).copied().unwrap_or(0.0).to_string()
            } else {
                "default".to_string()
            };

            series_map
                .entry(group_key)
                .or_default()
                .push(DataPoint::new(x, y));
        }

        // Convert to DataSeries with different colors
        let colors = [
            Color::BLUE,
            Color::RED,
            Color::GREEN,
            Color::rgb(0xFF, 0x80, 0x00), // Orange
            Color::rgb(0x80, 0x00, 0xFF), // Purple
            Color::rgb(0x00, 0x80, 0x80), // Teal
            Color::rgb(0xFF, 0x00, 0x80), // Pink
            Color::rgb(0x80, 0x80, 0x00), // Olive
        ];

        series_map
            .into_iter()
            .enumerate()
            .map(|(i, (name, points))| {
                let color = colors[i % colors.len()];
                DataSeries {
                    name,
                    points,
                    style: PlotStyle::default().with_line_color(color).with_marker_color(color),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes_builder() {
        let aes = Aes::new().x("col1").y("col2").color("group");
        assert_eq!(aes.x.as_deref(), Some("col1"));
        assert_eq!(aes.y.as_deref(), Some("col2"));
        assert_eq!(aes.color.as_deref(), Some("group"));
    }

    #[test]
    fn plot_builder() {
        let data = vec![
            HashMap::from([("x".to_string(), 1.0), ("y".to_string(), 2.0)]),
            HashMap::from([("x".to_string(), 2.0), ("y".to_string(), 4.0)]),
        ];

        let svg = Plot::new()
            .data(data)
            .aes(Aes::new().x("x").y("y"))
            .geom_point()
            .geom_line()
            .title("Test Plot")
            .x_label("X")
            .y_label("Y")
            .render();

        assert!(svg.contains("<svg"));
        assert!(svg.contains("Test Plot"));
        assert!(svg.contains("X"));
        assert!(svg.contains("Y"));
    }

    #[test]
    fn plot_with_theme() {
        let data = vec![HashMap::from([
            ("x".to_string(), 1.0),
            ("y".to_string(), 2.0),
        ])];

        let svg = Plot::new()
            .data(data)
            .aes(Aes::new().x("x").y("y"))
            .geom_point()
            .theme_seaborn()
            .render();

        assert!(svg.contains("<svg"));
    }

    #[test]
    fn plot_layers() {
        let layer1 = Layer::new(Geom::Point).aes(Aes::new().x("x").y("y"));
        let layer2 = Layer::new(Geom::Line).aes(Aes::new().x("x").y("y"));

        let plot = Plot::new().layer(layer1).layer(layer2);
        assert_eq!(plot.layers.len(), 2);
    }
}
