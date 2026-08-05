//! Figure — multi-axes layout with aggregated legends.

use crate::axes::Range;
use crate::common::{DataSeries, PlotConfig};
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// A single axis (subplot) within a figure.
#[derive(Debug, Clone)]
pub struct Axes {
    config: PlotConfig,
    series: Vec<DataSeries>,
}

impl Axes {
    pub fn new(config: PlotConfig) -> Self {
        Self {
            config,
            series: Vec::new(),
        }
    }

    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }

    pub fn series(&self) -> &[DataSeries] {
        &self.series
    }

    pub fn config(&self) -> &PlotConfig {
        &self.config
    }

    /// X range over all series (empty → `0..1`).
    pub fn x_range(&self) -> Range {
        self.series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.x))
            .fold(None::<(f64, f64)>, |acc, x| match acc {
                None => Some((x, x)),
                Some((lo, hi)) => Some((lo.min(x), hi.max(x))),
            })
            .map(|(lo, hi)| Range { min: lo, max: hi })
            .unwrap_or(Range { min: 0.0, max: 1.0 })
    }

    /// Y range over all series (empty → `0..1`).
    pub fn y_range(&self) -> Range {
        self.series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.y))
            .fold(None::<(f64, f64)>, |acc, y| match acc {
                None => Some((y, y)),
                Some((lo, hi)) => Some((lo.min(y), hi.max(y))),
            })
            .map(|(lo, hi)| Range { min: lo, max: hi })
            .unwrap_or(Range { min: 0.0, max: 1.0 })
    }
}

/// Multi-axes figure arranged in rows × cols.
///
/// Each `Axes` is a subplot at `(row, col)`. Gaps between subplots are filled
/// with shared tick labels and axis labels when `shared_x` / `shared_y` are set.
#[derive(Debug, Clone)]
pub struct Figure {
    rows: usize,
    cols: usize,
    axes: Vec<Vec<Option<Axes>>>,
    /// Shared X label shown below the grid.
    shared_x_label: Option<String>,
    /// Shared Y label shown left of the grid.
    shared_y_label: Option<String>,
}

impl Figure {
    /// Create an empty figure of the given grid size.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            axes: vec![vec![None; cols]; rows],
            shared_x_label: None,
            shared_y_label: None,
        }
    }

    /// Set an axes at `(row, col)`.
    pub fn set_axes(&mut self, row: usize, col: usize, axes: Axes) -> PlotResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(PlotError::InvalidData("axes position out of bounds".into()));
        }
        self.axes[row][col] = Some(axes);
        Ok(())
    }

    /// Get a reference to the axes at `(row, col)`.
    pub fn axes(&self, row: usize, col: usize) -> Option<&Axes> {
        self.axes[row][col].as_ref()
    }

    /// Set a shared X label (displayed once below the bottom row).
    pub fn with_shared_x_label(mut self, label: impl Into<String>) -> Self {
        self.shared_x_label = Some(label.into());
        self
    }

    /// Set a shared Y label (displayed once left of the leftmost column).
    pub fn with_shared_y_label(mut self, label: impl Into<String>) -> Self {
        self.shared_y_label = Some(label.into());
        self
    }

    /// The legend contains one entry per unique series name across all axes.
    /// Returns `(name, color)` pairs.
    pub fn aggregated_legend(&self) -> Vec<(String, Color)> {
        let mut seen = std::collections::HashSet::new();
        let mut legend = Vec::new();
        for row in &self.axes {
            for axes in row {
                if let Some(a) = axes {
                    for s in a.series() {
                        if seen.insert(s.name.clone()) {
                            let color = s.style.line_color;
                            legend.push((s.name.clone(), color));
                        }
                    }
                }
            }
        }
        legend
    }

    /// Total number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Total number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataPoint;

    fn series(name: &str, pts: Vec<(f64, f64)>) -> DataSeries {
        DataSeries::new(
            name,
            pts.into_iter().map(|(x, y)| DataPoint::new(x, y)).collect(),
        )
    }

    #[test]
    fn axes_range_empty() {
        let a = Axes::new(PlotConfig::new());
        assert_eq!(a.x_range().min, 0.0);
        assert_eq!(a.x_range().max, 1.0);
    }

    #[test]
    fn axes_range_populated() {
        let mut a = Axes::new(PlotConfig::new());
        a.add_series(series("s", vec![(1.0, 5.0), (3.0, 2.0)]));
        assert_eq!(a.x_range().min, 1.0);
        assert_eq!(a.x_range().max, 3.0);
        assert_eq!(a.y_range().min, 2.0);
        assert_eq!(a.y_range().max, 5.0);
    }

    #[test]
    fn figure_out_of_bounds() {
        let mut fig = Figure::new(2, 2);
        assert!(fig.set_axes(0, 0, Axes::new(PlotConfig::new())).is_ok());
        assert!(fig.set_axes(2, 0, Axes::new(PlotConfig::new())).is_err());
    }

    #[test]
    fn figure_aggregated_legend_deduplicates() {
        let mut fig = Figure::new(1, 2);
        let mut a0 = Axes::new(PlotConfig::new());
        a0.add_series(series("A", vec![(0.0, 0.0)]));
        let mut a1 = Axes::new(PlotConfig::new());
        a1.add_series(series("A", vec![(0.0, 0.0)]));
        a1.add_series(series("B", vec![(0.0, 0.0)]));
        fig.set_axes(0, 0, a0).unwrap();
        fig.set_axes(0, 1, a1).unwrap();
        let legend = fig.aggregated_legend();
        assert_eq!(legend.len(), 2);
        assert_eq!(legend[0].0, "A");
        assert_eq!(legend[1].0, "B");
    }
}
