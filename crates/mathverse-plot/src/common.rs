//! Common plotting data structures

use crate::axes::{Range, Scale};
use crate::backend::PlotData;
use crate::style::PlotStyle;

/// Escape XML special characters in text content for safe SVG insertion.
#[must_use]
pub fn xml_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

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
    pub fn new(name: impl Into<String>, points: Vec<DataPoint>) -> Self {
        DataSeries {
            name: name.into(),
            points,
            style: PlotStyle::default(),
        }
    }

    /// Create a new data series with custom style
    pub fn with_style(name: impl Into<String>, points: Vec<DataPoint>, style: PlotStyle) -> Self {
        DataSeries {
            name: name.into(),
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
    /// Font family stack for text (matplotlib `font.family`).
    pub font_family: String,
    /// Base font size in px (matplotlib `font.size`).
    pub font_size: f64,
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
        padding: 55.0,
        x_scale: Scale::Linear,
        y_scale: Scale::Linear,
        tick_count: 8,
        font_family: "Arial, sans-serif".to_string(),
        font_size: 14.0,
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

    /// Set the font family stack for text rendering.
    pub fn with_font_family(mut self, family: impl Into<String>) -> Self {
        self.font_family = family.into();
        self
    }

    /// Set the base font size in px.
    pub fn with_font_size(mut self, size: f64) -> Self {
        self.font_size = size;
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

/// Compute the x-axis range from all data sources in a `PlotData` snapshot.
/// Falls back to `0..1` when no data is present.
/// Heatmaps use index-based ranges (0..cols).
pub fn compute_x_range(data: &PlotData) -> Range {
    if !data.heatmaps.is_empty() {
        let cols = data.heatmaps[0].cols();
        return Range {
            min: 0.0,
            max: cols as f64,
        };
    }
    data.series
        .iter()
        .flat_map(|s| s.points.iter().map(|p| p.x))
        .chain(data.bars.iter().flat_map(|b| [b.x_lo, b.x_hi]))
        .chain(data.boxes.iter().enumerate().map(|(i, _)| i as f64))
        .chain(data.error_bars.iter().map(|e| e.x))
        .chain(data.images.iter().flat_map(|im| [im.x_extent.0, im.x_extent.1]))
        .chain(data.paths.iter().flat_map(|p| p.points.iter().map(|(x, _)| *x)))
        .chain(data.lines.iter().flat_map(|l| [l.x1, l.x2]))
        .fold(None::<(f64, f64)>, |acc, x| match acc {
            None => Some((x, x)),
            Some((lo, hi)) => Some((lo.min(x), hi.max(x))),
        })
        .map(|(lo, hi)| Range { min: lo, max: hi })
        .unwrap_or(Range { min: 0.0, max: 1.0 })
}

/// Compute the y-axis range from all data sources in a `PlotData` snapshot.
/// Falls back to `0..1` when no data is present.
/// Heatmaps use index-based ranges (0..rows).
pub fn compute_y_range(data: &PlotData) -> Range {
    if !data.heatmaps.is_empty() {
        let rows = data.heatmaps[0].rows();
        return Range {
            min: 0.0,
            max: rows as f64,
        };
    }
    data.series
        .iter()
        .flat_map(|s| s.points.iter().map(|p| p.y))
        .chain(data.bars.iter().map(|b| b.y))
        .chain(
            data.boxes
                .iter()
                .flat_map(|bx| [bx.stats.q1, bx.stats.q3, bx.stats.min, bx.stats.max]
                    .into_iter()
                    .chain(bx.stats.outliers.iter().copied())),
        )
        .chain(data.error_bars.iter().flat_map(|e| [e.bar.lo, e.bar.hi]))
        .chain(data.images.iter().flat_map(|im| [im.y_extent.0, im.y_extent.1]))
        .chain(data.paths.iter().flat_map(|p| p.points.iter().map(|(_, y)| *y)))
        .chain(data.lines.iter().flat_map(|l| [l.y1, l.y2]))
        .fold(None::<(f64, f64)>, |acc, y| match acc {
            None => Some((y, y)),
            Some((lo, hi)) => Some((lo.min(y), hi.max(y))),
        })
        .map(|(lo, hi)| Range { min: lo, max: hi })
        .unwrap_or(Range { min: 0.0, max: 1.0 })
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

/// Encode bytes as a base64 string.
pub fn base64_encode(data: &[u8]) -> String {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = if chunk.len() > 1 { u32::from(chunk[1]) } else { 0 };
        let b2 = if chunk.len() > 2 { u32::from(chunk[2]) } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        let significant = (chunk.len() * 8).div_ceil(6);
        for i in 0..4 {
            if i >= significant {
                out.push('=');
            } else {
                let shift = 18 - i * 6;
                let idx = ((triple >> shift) & 0x3F) as usize;
                out.push(alphabet[idx] as char);
            }
        }
    }
    out
}

/// Decode a base64 string to bytes.
pub fn base64_decode(input: &str) -> Vec<u8> {
    let alphabet: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut buf: Vec<u32> = Vec::new();
    for c in input.bytes() {
        if c == b'=' {
            continue;
        }
        if let Some(val) = alphabet.iter().position(|&b| b == c) {
            buf.push(val as u32);
        }
    }
    let mut bytes = Vec::new();
    for chunk in buf.chunks(4) {
        let mut triple: u32 = 0;
        for &v in chunk {
            triple = (triple << 6) | v;
        }
        // Left-align the accumulated bits: missing 6-bit groups are zero-padded
        // on the right, mirroring the encoder's padding behavior.
        triple <<= (4 - chunk.len()) * 6;
        if chunk.len() >= 2 {
            bytes.push((triple >> 16) as u8);
        }
        if chunk.len() >= 3 {
            bytes.push((triple >> 8) as u8);
        }
        if chunk.len() >= 4 {
            bytes.push(triple as u8);
        }
    }
    bytes
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

    #[test]
    fn base64_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_roundtrip() {
        for len in 0..=512 {
            let data: Vec<u8> = (0..len).map(|i| (i * 31) as u8).collect();
            assert_eq!(base64_decode(&base64_encode(&data)), data, "len={len}");
        }
    }
}
