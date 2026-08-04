//! FacetGrid and trellis plots for multi-panel conditioned visualizations.
//!
//! Creates a matrix of small-multiple plots where each panel shows a subset
//! of the data conditioned on one or two categorical variables.
//!
//! # Example
//!
//! ```rust,ignore
//! use mathverse_plot::facet::{FacetGrid, FacetWrap};
//!
//! let grid = FacetGrid::new()
//!     .col("treatment")
//!     .row("time_point")
//!     .wrap(FacetWrap::Columns(3))
//!     .shared_axes(true, true);
//! ```

use crate::axis_config::AxisConfig;
use crate::common::{DataPoint, DataSeries};
use crate::style::Color;
use crate::theme::ThemeConfig;

/// Data row with categorical fields for faceting.
#[derive(Debug, Clone)]
pub struct FacetData {
    /// Column name -> value mapping for each data point.
    pub columns: std::collections::HashMap<String, String>,
    /// Numeric x value.
    pub x: f64,
    /// Numeric y value.
    pub y: f64,
    /// Optional size value.
    pub size: Option<f64>,
    /// Optional color value.
    pub color: Option<Color>,
}

impl FacetData {
    /// Create a new data row.
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            columns: std::collections::HashMap::new(),
            x,
            y,
            size: None,
            color: None,
        }
    }

    /// Set a categorical column value.
    pub fn with_column(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.columns.insert(name.into(), value.into());
        self
    }

    /// Set the size.
    pub fn with_size(mut self, size: f64) -> Self {
        self.size = Some(size);
        self
    }

    /// Set the color.
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

/// Facet wrap mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FacetWrap {
    /// Wrap into columns (fixed number of columns).
    Columns(usize),
    /// Wrap into rows (fixed number of rows).
    Rows(usize),
    /// Auto-wrap into a square-ish grid.
    Auto,
}

/// Facet scale mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FacetScale {
    /// Fixed scales (same limits across all panels).
    Fixed,
    /// Free scales (each panel has its own limits).
    Free,
    /// Free x, fixed y.
    FreeX,
    /// Fixed x, free y.
    FreeY,
}

/// Facet label position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FacetLabelPos {
    /// Labels on top of each column.
    Top,
    /// Labels on left of each row.
    Left,
    /// Labels as strip at top of each panel.
    StripTop,
    /// Labels as strip at left of each panel.
    StripLeft,
}

/// A facet grid configuration.
#[derive(Debug, Clone)]
pub struct FacetGrid {
    /// Column facet variable name.
    pub col_var: Option<String>,
    /// Row facet variable name.
    pub row_var: Option<String>,
    /// Wrap mode (alternative to row/col).
    pub wrap: Option<FacetWrap>,
    /// Scale mode.
    pub scale: FacetScale,
    /// Label position.
    pub label_pos: FacetLabelPos,
    /// Share x axes.
    pub share_x: bool,
    /// Share y axes.
    pub share_y: bool,
    /// Gap between panels (pixels).
    pub gap: f64,
    /// Panel width (pixels).
    pub panel_width: f64,
    /// Panel height (pixels).
    pub panel_height: f64,
    /// Margin around the entire grid.
    pub margin: f64,
    /// Title for the entire grid.
    pub title: String,
    /// Column variable label.
    pub col_label: String,
    /// Row variable label.
    pub row_label: String,
    /// Theme.
    pub theme: ThemeConfig,
    /// Background color for panels.
    pub panel_background: Color,
    /// Strip background color.
    pub strip_background: Color,
    /// Strip text color.
    pub strip_text_color: Color,
    /// Strip font size.
    pub strip_font_size: f64,
    /// Show axis labels on each panel or only outer.
    pub inner_labels: bool,
    /// Show tick labels on each panel or only outer.
    pub inner_ticks: bool,
}

impl Default for FacetGrid {
    fn default() -> Self {
        Self {
            col_var: None,
            row_var: None,
            wrap: None,
            scale: FacetScale::Fixed,
            label_pos: FacetLabelPos::StripTop,
            share_x: true,
            share_y: true,
            gap: 10.0,
            panel_width: 200.0,
            panel_height: 150.0,
            margin: 40.0,
            title: String::new(),
            col_label: String::new(),
            row_label: String::new(),
            theme: ThemeConfig::default(),
            panel_background: Color::WHITE,
            strip_background: Color::rgb(230, 230, 230),
            strip_text_color: Color::BLACK,
            strip_font_size: 11.0,
            inner_labels: false,
            inner_ticks: false,
        }
    }
}

impl FacetGrid {
    /// Create a new facet grid.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set column facet variable.
    pub fn col(mut self, var: impl Into<String>) -> Self {
        self.col_var = Some(var.into());
        self
    }

    /// Set row facet variable.
    pub fn row(mut self, var: impl Into<String>) -> Self {
        self.row_var = Some(var.into());
        self
    }

    /// Set wrap mode.
    pub fn wrap(mut self, wrap: FacetWrap) -> Self {
        self.wrap = Some(wrap);
        self
    }

    /// Set scale mode.
    pub fn scale(mut self, scale: FacetScale) -> Self {
        self.scale = scale;
        self
    }

    /// Share x and/or y axes.
    pub fn shared_axes(mut self, share_x: bool, share_y: bool) -> Self {
        self.share_x = share_x;
        self.share_y = share_y;
        self
    }

    /// Set gap between panels.
    pub fn with_gap(mut self, gap: f64) -> Self {
        self.gap = gap;
        self
    }

    /// Set panel dimensions.
    pub fn with_panel_size(mut self, width: f64, height: f64) -> Self {
        self.panel_width = width;
        self.panel_height = height;
        self
    }

    /// Set title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set column label.
    pub fn with_col_label(mut self, label: impl Into<String>) -> Self {
        self.col_label = label.into();
        self
    }

    /// Set row label.
    pub fn with_row_label(mut self, label: impl Into<String>) -> Self {
        self.row_label = label.into();
        self
    }

    /// Set theme.
    pub fn with_theme(mut self, theme: ThemeConfig) -> Self {
        self.theme = theme;
        self
    }

    /// Show inner labels on all panels.
    pub fn inner_labels(mut self) -> Self {
        self.inner_labels = true;
        self.inner_ticks = true;
        self
    }

    /// Get unique values for a column.
    pub fn unique_values(data: &[FacetData], var: &str) -> Vec<String> {
        let mut values: Vec<String> = data
            .iter()
            .filter_map(|d| d.columns.get(var).cloned())
            .collect();
        values.sort();
        values.dedup();
        values
    }

    /// Get data subset for a specific facet combination.
    pub fn subset<'a>(
        data: &'a [FacetData],
        col_var: Option<&str>,
        col_val: Option<&str>,
        row_var: Option<&str>,
        row_val: Option<&str>,
    ) -> Vec<&'a FacetData> {
        data.iter()
            .filter(|d| {
                let col_match = match (col_var, col_val) {
                    (Some(var), Some(val)) => d.columns.get(var).map(|v| v == val).unwrap_or(false),
                    (None, None) => true,
                    _ => false,
                };
                let row_match = match (row_var, row_val) {
                    (Some(var), Some(val)) => d.columns.get(var).map(|v| v == val).unwrap_or(false),
                    (None, None) => true,
                    _ => false,
                };
                col_match && row_match
            })
            .collect()
    }

    /// Compute grid dimensions.
    pub fn grid_dims(&self, data: &[FacetData]) -> (usize, usize) {
        if let Some(ref wrap) = self.wrap {
            let total_panels = self.total_panels(data);
            match wrap {
                FacetWrap::Columns(cols) => {
                    let rows = (total_panels + cols - 1) / cols;
                    (rows, *cols)
                }
                FacetWrap::Rows(rows) => {
                    let cols = (total_panels + rows - 1) / rows;
                    (*rows, cols)
                }
                FacetWrap::Auto => {
                    let cols = (total_panels as f64).sqrt().ceil() as usize;
                    let rows = (total_panels + cols - 1) / cols;
                    (rows, cols)
                }
            }
        } else {
            let n_cols = self.col_var.as_ref()
                .map(|v| Self::unique_values(data, v).len())
                .unwrap_or(1);
            let n_rows = self.row_var.as_ref()
                .map(|v| Self::unique_values(data, v).len())
                .unwrap_or(1);
            (n_rows, n_cols)
        }
    }

    /// Total number of panels.
    pub fn total_panels(&self, data: &[FacetData]) -> usize {
        let col_vals = self.col_var.as_ref()
            .map(|v| Self::unique_values(data, v))
            .unwrap_or_default();
        let row_vals = self.row_var.as_ref()
            .map(|v| Self::unique_values(data, v))
            .unwrap_or_default();

        let n_cols = if col_vals.is_empty() { 1 } else { col_vals.len() };
        let n_rows = if row_vals.is_empty() { 1 } else { row_vals.len() };

        n_cols * n_rows
    }

    /// Render the facet grid as SVG.
    pub fn render_svg(&self, data: &[FacetData]) -> String {
        let (n_rows, n_cols) = self.grid_dims(data);
        let col_vals = self.col_var.as_ref()
            .map(|v| Self::unique_values(data, v))
            .unwrap_or_default();
        let row_vals = self.row_var.as_ref()
            .map(|v| Self::unique_values(data, v))
            .unwrap_or_default();

        let strip_height = 20.0;
        let total_width = self.margin * 2.0 + n_cols as f64 * (self.panel_width + self.gap) - self.gap;
        let total_height = self.margin * 2.0 + n_rows as f64 * (self.panel_height + self.gap + strip_height) - self.gap;

        let mut svg = String::new();
        svg.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
            total_width, total_height, total_width, total_height
        ));

        // Title
        if !self.title.is_empty() {
            svg.push_str(&format!(
                "  <text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-size=\"14\" font-weight=\"bold\">{}</text>\n",
                total_width / 2.0, self.title
            ));
        }

        // Render each panel
        for row_idx in 0..n_rows {
            for col_idx in 0..n_cols {
                let x = self.margin + col_idx as f64 * (self.panel_width + self.gap);
                let y = self.margin + row_idx as f64 * (self.panel_height + self.gap + strip_height);

                let col_val = col_vals.get(col_idx);
                let row_val = row_vals.get(row_idx);

                // Strip label
                let strip_label = match (&self.label_pos, col_val, row_val) {
                    (FacetLabelPos::StripTop | FacetLabelPos::Top, Some(val), _) => Some(val.as_str()),
                    (FacetLabelPos::StripLeft | FacetLabelPos::Left, _, Some(val)) => Some(val.as_str()),
                    _ => None,
                };

                if let Some(label) = strip_label {
                    svg.push_str(&format!(
                        "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\"/>\n",
                        x, y, self.panel_width, strip_height, self.strip_background.to_hex()
                    ));
                    svg.push_str(&format!(
                        "  <text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"{}\" fill=\"{}\">{}</text>\n",
                        x + self.panel_width / 2.0,
                        y + strip_height - 5.0,
                        self.strip_font_size,
                        self.strip_text_color.to_hex(),
                        label
                    ));
                }

                // Panel background
                let panel_y = y + strip_height;
                svg.push_str(&format!(
                    "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"#ccc\" stroke-width=\"0.5\"/>\n",
                    x, panel_y, self.panel_width, self.panel_height, self.panel_background.to_hex()
                ));

                // Get subset data
                let subset = Self::subset(
                    data,
                    self.col_var.as_deref(), col_val.map(|s| s.as_str()),
                    self.row_var.as_deref(), row_val.map(|s| s.as_str()),
                );

                // Render data points as scatter
                let (mut x_min, mut y_min) = (0.0, 0.0);
                if !subset.is_empty() {
                    let (xmin, xmax) = subset.iter()
                        .map(|d| d.x)
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
                    let (ymin, ymax) = subset.iter()
                        .map(|d| d.y)
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
                    x_min = xmin;
                    y_min = ymin;

                    let x_range = if (xmax - xmin).abs() < 1e-10 { 1.0 } else { xmax - xmin };
                    let y_range = if (ymax - ymin).abs() < 1e-10 { 1.0 } else { ymax - ymin };

                    for point in &subset {
                        let px = x + ((point.x - xmin) / x_range) * (self.panel_width - 20.0) + 10.0;
                        let py = panel_y + self.panel_height - ((point.y - ymin) / y_range) * (self.panel_height - 20.0) - 10.0;
                        let color = point.color.unwrap_or(Color::BLUE);
                        svg.push_str(&format!(
                            "  <circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"{}\" opacity=\"0.7\"/>\n",
                            px, py, color.to_hex()
                        ));
                    }
                }

                // Axis ticks (minimal)
                if self.inner_ticks || col_idx == 0 {
                    svg.push_str(&format!(
                        "  <text x=\"{}\" y=\"{}\" font-size=\"8\" fill=\"gray\" text-anchor=\"middle\">{:.1}</text>\n",
                        x + self.panel_width / 2.0,
                        panel_y + self.panel_height + 10.0,
                        x_min
                    ));
                }
                if self.inner_ticks || row_idx == n_rows - 1 {
                    svg.push_str(&format!(
                        "  <text x=\"{}\" y=\"{}\" font-size=\"8\" fill=\"gray\" text-anchor=\"end\">{:.1}</text>\n",
                        x - 5.0,
                        panel_y + self.panel_height / 2.0,
                        y_min
                    ));
                }
            }
        }

        // Outer labels
        if !self.col_label.is_empty() && !col_vals.is_empty() {
            let label_y = self.margin - 5.0;
            svg.push_str(&format!(
                "  <text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"12\" font-weight=\"bold\">{}</text>\n",
                total_width / 2.0, label_y, self.col_label
            ));
        }

        if !self.row_label.is_empty() && !row_vals.is_empty() {
            let label_x = self.margin - 5.0;
            svg.push_str(&format!(
                "  <text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-size=\"12\" font-weight=\"bold\" transform=\"rotate(-90, {}, {})\">{}</text>\n",
                label_x, total_height / 2.0, label_x, total_height / 2.0, self.row_label
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Convert facet data to DataSeries for use with existing plot types.
    pub fn to_series(data: &[FacetData]) -> DataSeries {
        DataSeries {
            name: String::new(),
            points: data.iter().map(|d| DataPoint::new(d.x, d.y)).collect(),
            style: crate::style::PlotStyle::default(),
        }
    }
}

/// Builder for creating facet grids from data.
pub struct FacetBuilder;

impl FacetBuilder {
    /// Create a simple column facet grid.
    pub fn col(var: impl Into<String>) -> FacetGrid {
        FacetGrid::new().col(var)
    }

    /// Create a simple row facet grid.
    pub fn row(var: impl Into<String>) -> FacetGrid {
        FacetGrid::new().row(var)
    }

    /// Create a grid faceted by both row and column.
    pub fn grid(col_var: impl Into<String>, row_var: impl Into<String>) -> FacetGrid {
        FacetGrid::new().col(col_var).row(row_var)
    }

    /// Create a wrapped facet grid.
    pub fn wrap(var: impl Into<String>, wrap: FacetWrap) -> FacetGrid {
        FacetGrid::new().col(var).wrap(wrap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<FacetData> {
        vec![
            FacetData::new(1.0, 10.0).with_column("group", "A").with_column("time", "T1"),
            FacetData::new(2.0, 20.0).with_column("group", "B").with_column("time", "T1"),
            FacetData::new(3.0, 15.0).with_column("group", "A").with_column("time", "T2"),
            FacetData::new(4.0, 25.0).with_column("group", "B").with_column("time", "T2"),
        ]
    }

    #[test]
    fn facet_data_creation() {
        let d = FacetData::new(1.0, 2.0)
            .with_column("g", "A")
            .with_size(5.0)
            .with_color(Color::RED);
        assert_eq!(d.x, 1.0);
        assert_eq!(d.y, 2.0);
        assert_eq!(d.columns.get("g").unwrap(), "A");
        assert_eq!(d.size, Some(5.0));
    }

    #[test]
    fn unique_values() {
        let data = sample_data();
        let groups = FacetGrid::unique_values(&data, "group");
        assert_eq!(groups, vec!["A", "B"]);

        let times = FacetGrid::unique_values(&data, "time");
        assert_eq!(times, vec!["T1", "T2"]);
    }

    #[test]
    fn subset_filtering() {
        let data = sample_data();
        let subset_a = FacetGrid::subset(&data, Some("group"), Some("A"), None, None);
        assert_eq!(subset_a.len(), 2);

        let subset_b_t1 = FacetGrid::subset(&data, Some("group"), Some("B"), Some("time"), Some("T1"));
        assert_eq!(subset_b_t1.len(), 1);
        assert_eq!(subset_b_t1[0].x, 2.0);
    }

    #[test]
    fn grid_dims_col() {
        let data = sample_data();
        let grid = FacetGrid::new().col("group");
        let (rows, cols) = grid.grid_dims(&data);
        assert_eq!(rows, 1);
        assert_eq!(cols, 2);
    }

    #[test]
    fn grid_dims_row() {
        let data = sample_data();
        let grid = FacetGrid::new().row("time");
        let (rows, cols) = grid.grid_dims(&data);
        assert_eq!(rows, 2);
        assert_eq!(cols, 1);
    }

    #[test]
    fn grid_dims_wrap() {
        let data = sample_data();
        let grid = FacetGrid::new().col("group").wrap(FacetWrap::Columns(1));
        let (rows, cols) = grid.grid_dims(&data);
        assert_eq!(cols, 1);
        assert_eq!(rows, 2);
    }

    #[test]
    fn render_svg() {
        let data = sample_data();
        let grid = FacetGrid::new()
            .col("group")
            .row("time")
            .with_title("Facet Demo")
            .with_panel_size(150.0, 100.0);
        let svg = grid.render_svg(&data);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Facet Demo"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn to_series() {
        let data = sample_data();
        let series = FacetGrid::to_series(&data);
        assert_eq!(series.points.len(), 4);
    }

    #[test]
    fn facet_builder_presets() {
        let g1 = FacetBuilder::col("treatment");
        assert!(g1.col_var.is_some());

        let g2 = FacetBuilder::row("time");
        assert!(g2.row_var.is_some());

        let g3 = FacetBuilder::grid("treatment", "time");
        assert!(g3.col_var.is_some());
        assert!(g3.row_var.is_some());

        let g4 = FacetBuilder::wrap("group", FacetWrap::Columns(3));
        assert!(g4.wrap.is_some());
    }
}
