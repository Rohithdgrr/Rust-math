//! Common plotting data structures

use crate::style::PlotStyle;

/// Data point
#[derive(Debug, Clone, Copy)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
}

impl DataPoint {
    /// Create a new data point
    pub fn new(x: f64, y: f64) -> Self {
        DataPoint { x, y }
    }
}

/// Data series
#[derive(Debug, Clone)]
pub struct DataSeries {
    pub name: String,
    pub points: Vec<DataPoint>,
    pub style: PlotStyle,
}

impl DataSeries {
    /// Create a new data series
    pub fn new(name: String, points: Vec<DataPoint>) -> Self {
        DataSeries {
            name,
            points,
            style: PlotStyle::default(),
        }
    }

    /// Create a new data series with custom style
    pub fn with_style(name: String, points: Vec<DataPoint>, style: PlotStyle) -> Self {
        DataSeries {
            name,
            points,
            style,
        }
    }

    /// Set the style
    pub fn set_style(&mut self, style: PlotStyle) {
        self.style = style;
    }

    /// Get x values
    pub fn x_values(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.x).collect()
    }

    /// Get y values
    pub fn y_values(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.y).collect()
    }

    /// Get the range of x values
    pub fn x_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let x_vals: Vec<f64> = self.points.iter().map(|p| p.x).collect();
        let min = x_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = x_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }

    /// Get the range of y values
    pub fn y_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let y_vals: Vec<f64> = self.points.iter().map(|p| p.y).collect();
        let min = y_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = y_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }
}

/// Plot configuration
#[derive(Debug, Clone)]
pub struct PlotConfig {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub width: u32,
    pub height: u32,
    pub show_grid: bool,
    pub show_legend: bool,
    pub padding: f64,
}

impl Default for PlotConfig {
    fn default() -> Self {
        PlotConfig {
            title: String::new(),
            x_label: String::new(),
            y_label: String::new(),
            width: 800,
            height: 600,
            show_grid: true,
            show_legend: true,
            padding: 50.0,
        }
    }
}

impl PlotConfig {
    /// Create a new plot configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the title
    pub fn with_title(mut self, title: String) -> Self {
        self.title = title;
        self
    }

    /// Set the x-axis label
    pub fn with_x_label(mut self, label: String) -> Self {
        self.x_label = label;
        self
    }

    /// Set the y-axis label
    pub fn with_y_label(mut self, label: String) -> Self {
        self.y_label = label;
        self
    }

    /// Set the dimensions
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set whether to show the grid
    pub fn with_grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Set whether to show the legend
    pub fn with_legend(mut self, show: bool) -> Self {
        self.show_legend = show;
        self
    }

    /// Set the padding
    pub fn with_padding(mut self, padding: f64) -> Self {
        self.padding = padding;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_point() {
        let point = DataPoint::new(1.0, 2.0);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
    }

    #[test]
    fn test_data_series() {
        let points = vec![
            DataPoint::new(1.0, 2.0),
            DataPoint::new(2.0, 4.0),
            DataPoint::new(3.0, 6.0),
        ];
        let series = DataSeries::new("Test".to_string(), points);
        assert_eq!(series.name, "Test");
        assert_eq!(series.points.len(), 3);
    }

    #[test]
    fn test_data_series_ranges() {
        let points = vec![
            DataPoint::new(1.0, 2.0),
            DataPoint::new(5.0, 10.0),
        ];
        let series = DataSeries::new("Test".to_string(), points);
        assert_eq!(series.x_range(), Some((1.0, 5.0)));
        assert_eq!(series.y_range(), Some((2.0, 10.0)));
    }

    #[test]
    fn test_plot_config() {
        let config = PlotConfig::new()
            .with_title("Test Plot".to_string())
            .with_x_label("X".to_string())
            .with_y_label("Y".to_string())
            .with_dimensions(1000, 800);

        assert_eq!(config.title, "Test Plot");
        assert_eq!(config.width, 1000);
        assert_eq!(config.height, 800);
    }
}
