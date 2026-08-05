//! Backend trait for rendering plots.
use crate::boxplot::BoxStats;
use crate::common::{DataSeries, PlotConfig};
use crate::error::PlotResult;
use crate::errorbar::ErrorBar;
use crate::heatmap::HeatmapData;

#[derive(Debug, Clone)]
pub struct PlotData {
    pub config: PlotConfig,
    pub series: Vec<DataSeries>,
    pub bars: Vec<BarSnapshot>,
    pub boxes: Vec<BoxSnapshot>,
    pub error_bars: Vec<ErrorBarSnapshot>,
    pub heatmaps: Vec<HeatmapData>,
}

#[derive(Debug, Clone, Copy)]
pub struct BarSnapshot {
    pub x_lo: f64, pub x_hi: f64, pub y: f64,
    pub color: crate::style::Color,
}
#[derive(Debug, Clone)]
pub struct BoxSnapshot {
    pub name: String, pub stats: BoxStats,
    pub color: crate::style::Color,
}
#[derive(Debug, Clone, Copy)]
pub struct ErrorBarSnapshot {
    pub x: f64, pub bar: ErrorBar,
    pub color: crate::style::Color,
}
#[derive(Debug, Clone)]
pub enum PlotOutput {
    Svg(String),
    Text(String),
    Binary(Vec<u8>, &'static str),
}
pub trait Backend {
    fn generate(&self, data: &PlotData) -> PlotResult<PlotOutput>;
}
