//! Backend trait for rendering plots.

use crate::boxplot::BoxStats;
use crate::common::{DataSeries, PlotConfig};
use crate::errorbar::ErrorBar;
use crate::heatmap::HeatmapData;

/// Read-only snapshot of plot data, decoupled from any specific renderer.
/// New backends receive this instead of reaching into `SvgPlot` internals.
#[derive(Debug, Clone)]
pub struct PlotData {
    /// Plot configuration.
    pub config: PlotConfig,
    /// Line/scatter series.
    pub series: Vec<DataSeries>,
    /// Histogram bars.
    pub bars: Vec<BarSnapshot>,
    /// Tukey box plots.
    pub boxes: Vec<BoxSnapshot>,
    /// Error bars.
    pub error_bars: Vec<ErrorBarSnapshot>,
    /// Heatmap grids.
    pub heatmaps: Vec<HeatmapData>,
}

/// A histogram bar.
#[derive(Debug, Clone, Copy)]
pub struct BarSnapshot {
    pub x_lo: f64,
    pub x_hi: f64,
    pub y: f64,
    pub color: crate::style::Color,
}

/// A Tukey box plot.
#[derive(Debug, Clone)]
pub struct BoxSnapshot {
    pub name: String,
    pub stats: BoxStats,
    pub color: crate::style::Color,
}

/// A vertical error bar at a given x position.
#[derive(Debug, Clone, Copy)]
pub struct ErrorBarSnapshot {
    pub x: f64,
    pub bar: ErrorBar,
    pub color: crate::style::Color,
}

/// A renderable plot backend.
pub trait Backend {
    /// Render the plot and return the output (SVG string, PNG bytes, etc.).
    fn generate(&self, data: &PlotData) -> crate::error::PlotResult<String>;
}
