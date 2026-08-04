//! Heatmap grid data and rendering.

use crate::colorbar::{render_colorbar, ColorbarConfig};
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Colormap function: `t in [0, 1]` → `Color`.
pub type Colormap = fn(f64) -> Color;

/// A heatmap grid with optional normalization.
#[derive(Debug, Clone)]
pub struct HeatmapData {
    /// 2D grid, row-major. Each inner vec is one row (one y-level).
    pub grid: Vec<Vec<f64>>,
    /// Cell label for the legend.
    pub name: String,
    /// Colormap function.
    pub colormap: Colormap,
}

impl HeatmapData {
    /// Create a heatmap from a 2D grid. Rows are y-levels (top to bottom in
    /// SVG), columns are x-positions (left to right).
    ///
    /// # Errors
    ///
    /// Returns `PlotError::InvalidData` for empty or ragged grids.
    pub fn new(
        name: impl Into<String>,
        grid: Vec<Vec<f64>>,
        colormap: Colormap,
    ) -> PlotResult<Self> {
        if grid.is_empty() || grid[0].is_empty() {
            return Err(PlotError::InvalidData("empty heatmap grid".into()));
        }
        let cols = grid[0].len();
        if grid.iter().any(|row| row.len() != cols) {
            return Err(PlotError::InvalidData("ragged heatmap grid".into()));
        }
        Ok(Self {
            grid,
            name: name.into(),
            colormap,
        })
    }

    /// Number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.grid.len()
    }

    /// Number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.grid[0].len()
    }

    /// Min and max across the entire grid.
    #[must_use]
    pub fn bounds(&self) -> (f64, f64) {
        let min = self
            .grid
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = self
            .grid
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    }

    /// Render a single cell as an SVG `<rect>`.
    #[must_use]
    pub fn render_cell(
        &self,
        row: usize,
        col: usize,
        x_px: &dyn Fn(f64) -> f64,
        y_px: &dyn Fn(f64) -> f64,
    ) -> String {
        let v = self.grid[row][col];
        let (lo, hi) = self.bounds();
        let t = if (hi - lo).abs() < f64::EPSILON {
            0.5
        } else {
            (v - lo) / (hi - lo)
        };
        let color = (self.colormap)(t);
        let x = x_px(col as f64);
        let y = y_px(row as f64);
        let w = x_px(col as f64 + 1.0) - x;
        let h = y_px(row as f64 + 1.0) - y;
        format!(
            r#"  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{}" stroke="white" stroke-width="0.5"/>"#,
            color.to_hex()
        )
    }

    /// Render the heatmap with a colorbar.
    #[must_use]
    pub fn render_with_colorbar(
        &self,
        x_px: &dyn Fn(f64) -> f64,
        y_px: &dyn Fn(f64) -> f64,
        colorbar_x: f64,
        colorbar_y: f64,
    ) -> String {
        let mut svg = String::new();

        // Render heatmap cells
        for r in 0..self.rows() {
            for c in 0..self.cols() {
                svg.push_str(&self.render_cell(r, c, x_px, y_px));
                svg.push('\n');
            }
        }

        // Render colorbar
        let (data_min, data_max) = self.bounds();
        let colorbar_config = ColorbarConfig::new()
            .with_title(&self.name)
            .with_dimensions(20.0, 200.0);
        svg.push_str(&render_colorbar(
            colorbar_x,
            colorbar_y,
            data_min,
            data_max,
            self.colormap,
            &colorbar_config,
        ));
        svg.push('\n');

        svg
    }
}
