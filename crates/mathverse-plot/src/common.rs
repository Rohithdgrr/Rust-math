//! Common plotting data structures

use crate::axes::{Range, Scale};
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
        let min = x_vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = x_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some((min, max))
    }

    /// Get the range of y values
    pub fn y_range(&self) -> Option<(f64, f64)> {
        if self.points.is_empty() {
            return None;
        }
        let y_vals: Vec<f64> = self.points.iter().map(|p| p.y).collect();
        let min = y_vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = y_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
    /// X-axis scale (linear by default).
    pub x_scale: Scale,
    /// Y-axis scale (linear by default).
    pub y_scale: Scale,
    /// Target number of tick marks per axis.
    pub tick_count: usize,
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
            x_scale: Scale::Linear,
            y_scale: Scale::Linear,
            tick_count: 6,
        }
    }
}

impl PlotConfig {
    /// Create a new plot configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the x-axis label
    pub fn with_x_label(mut self, label: impl Into<String>) -> Self {
        self.x_label = label.into();
        self
    }

    /// Set the y-axis label
    pub fn with_y_label(mut self, label: impl Into<String>) -> Self {
        self.y_label = label.into();
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

    /// Set the x-axis scale
    pub fn with_x_scale(mut self, scale: Scale) -> Self {
        self.x_scale = scale;
        self
    }

    /// Set the y-axis scale
    pub fn with_y_scale(mut self, scale: Scale) -> Self {
        self.y_scale = scale;
        self
    }

    /// Set the target tick count per axis
    pub fn with_tick_count(mut self, count: usize) -> Self {
        self.tick_count = count;
        self
    }
}

/// Bounds spanning all points of a set of series (x and y separately).
/// Returns `(0..1, 0..1)` for empty input.
pub fn plot_bounds(series: &[DataSeries]) -> (Range, Range) {
    let x = Range::compute(series.iter().flat_map(|s| s.points.iter().map(|p| p.x)))
        .unwrap_or_default();
    let y = Range::compute(series.iter().flat_map(|s| s.points.iter().map(|p| p.y)))
        .unwrap_or_default();
    (x, y)
}

/// Largest-Triangle-Three-Buckets downsampling.
///
/// Reduces `points` to at most `target` points while preserving visual shape.
/// Returns the original data if it's already small enough or empty.
///
/// Reference: Sveinn Steffel & Elmqvist, *IEEE TVCG* 2013.
pub fn downsample_lttb(points: &[DataPoint], target: usize) -> Vec<DataPoint> {
    let n = points.len();
    if target == 0 || target >= n || n <= 2 {
        return points.to_vec();
    }

    let mut result = Vec::with_capacity(target);
    result.push(points[0]);

    let bucket_size = (n - 2) as f64 / (target - 2) as f64;

    let mut a = 0; // index of previously selected point
    for i in 1..target - 1 {
        let range_start = ((i - 1) as f64 * bucket_size + 1.0).floor() as usize;
        let range_end = ((i as f64) * bucket_size + 1.0).ceil().min(n as f64 - 1.0) as usize;

        // Average of next bucket (used for triangle area calc)
        let next_start = (i as f64 * bucket_size + 1.0).floor() as usize;
        let next_end = ((i + 1) as f64 * bucket_size + 1.0)
            .ceil()
            .min(n as f64 - 1.0) as usize;
        let mut avg_x = 0.0;
        let mut avg_y = 0.0;
        let next_count = (next_end - next_start + 1).max(1) as f64;
        for j in next_start..=next_end {
            avg_x += points[j].x;
            avg_y += points[j].y;
        }
        avg_x /= next_count;
        avg_y /= next_count;

        // Pick the point in the current bucket with the largest triangle area
        let mut max_area = -1.0_f64;
        let mut max_idx = range_start;
        for j in range_start..=range_end {
            let area = ((points[a].x - avg_x) * (points[j].y - points[a].y)
                - (points[a].x - points[j].x) * (avg_y - points[a].y))
                .abs();
            if area > max_area {
                max_area = area;
                max_idx = j;
            }
        }

        result.push(points[max_idx]);
        a = max_idx;
    }

    result.push(points[n - 1]);
    result
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
        let points = vec![DataPoint::new(1.0, 2.0), DataPoint::new(5.0, 10.0)];
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

    #[test]
    fn lttb_preserves_endpoints() {
        let pts: Vec<DataPoint> = (0..1000)
            .map(|i| DataPoint::new(i as f64, (i as f64 * 0.1).sin()))
            .collect();
        let ds = downsample_lttb(&pts, 50);
        assert_eq!(ds.len(), 50);
        assert_eq!(ds[0].x, 0.0);
        assert_eq!(ds[49].x, 999.0);
    }

    #[test]
    fn lttb_small_input_unchanged() {
        let pts = vec![DataPoint::new(0.0, 0.0), DataPoint::new(1.0, 1.0)];
        let ds = downsample_lttb(&pts, 100);
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn lttb_empty_input() {
        let ds = downsample_lttb(&[], 50);
        assert!(ds.is_empty());
    }
}
