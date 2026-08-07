#![allow(missing_docs)]
#![allow(clippy::approx_constant)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::float_cmp)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::needless_range_loop)]
//! Plotting with SVG, HTML, and terminal output backends.
//!
//! This crate provides:
//! - SVG plotting backend with scale-aware axes and "nice" tick labels
//! - HTML plotting backend
//! - Terminal (ASCII) plotting backend
//! - Common plot types (line, scatter, bar, histogram)
//! - Histogram binning delegated to `mathverse-statistics`
//!
//! All mathematical computation ships from the `mathverse-*` ecosystem; this
//! crate is rendering glue on top.

pub mod annotations;
pub mod animation;
pub mod area;
pub mod axis_config;
pub mod axes;
pub mod backend;
pub mod boxen;
pub mod boxplot;
pub mod bubble;
pub mod candlestick;
#[cfg(feature = "canvas")]
pub mod canvas;
pub mod categorical;
pub mod color;
pub mod colorbar;
pub mod common;
pub mod complex_plane;
pub mod contour;
pub mod countplot;
pub mod cvd;
pub mod datetime;
pub mod dual_axis;
pub mod ecdf;
pub mod error;
pub mod errorbar;
pub mod export;
pub mod facet;
pub mod figure;
pub mod grammar;
pub mod graph_layout;
pub mod grouped_bar;
pub mod gpu_renderer;
pub mod heatmap;
pub mod hbar;
pub mod highdpi;
pub mod hist2d;
pub mod histogram;
pub mod html;
pub mod inset;
#[cfg(feature = "interactive")]
pub mod interactive;
pub mod interactive_html;
pub mod jointplot;
pub mod kde;
pub mod legend;
pub mod marimekko;
pub mod ml_plots;
pub mod pairplot;
pub mod pareto;
#[cfg(feature = "pdf")]
pub mod pdf_backend;
pub mod pdf_overlay;
#[cfg(feature = "png")]
pub mod png_backend;
pub mod pie;
pub mod plt;
pub mod pointplot;
pub mod polar;
pub mod quiver;
pub mod radar;
pub mod rcparams;
pub mod regplot;
pub mod residplot;
pub mod rug;
pub mod save;
pub mod smooth;
pub mod spectrogram;
pub mod stats_annotate;
pub mod stacked_bar;
pub mod stem;
pub mod step;
pub mod streaming;
pub mod strip;
pub mod style;
pub mod surface;
pub mod svg;
pub mod swarm;
pub mod terminal;
pub mod theme;
pub mod violin;
pub mod waterfall;
pub mod webgl_3d;
pub use annotations::*;
pub use animation::*;
pub use area::{render_area_chart, AreaConfig};
pub use axis_config::{AxisConfig, AxisScale, GridStyle};
pub use axes::{Range, Scale};
pub use backend::{Backend, PlotData};
pub use boxen::{render_boxen_plot, BoxenConfig, BoxenData};
pub use boxplot::BoxStats;
pub use bubble::{render_bubble_chart, Bubble, BubbleConfig};
pub use candlestick::{render_candlestick_svg, Candlestick, CandlestickSeries};
pub use categorical::{CategoryMap, CategoricalAxis};
pub use color::*;
pub use colorbar::*;
pub use common::*;
pub use complex_plane::*;
pub use contour::{render_contour, ContourConfig};
pub use countplot::{render_countplot, CountConfig};
pub use cvd::{simulate_cvd, simulate_palette, CvdType};
pub use datetime::{DateTime, DatetimeAxis};
pub use dual_axis::{AxisBreak, AxisBuilder, BrokenAxis, BreakStyle, DualYAxis};
pub use ecdf::{render_ecdf, EcdfConfig};
pub use error::{PlotError, PlotResult};
pub use errorbar::ErrorBar;
pub use export::{ExportConfig, Margin, Metadata, Watermark, WatermarkPosition};
pub use facet::{FacetBuilder, FacetData, FacetGrid, FacetScale, FacetWrap};
pub use figure::{Axes, Figure};
pub use graph_layout::*;
pub use grouped_bar::{render_grouped_bar, GroupedBarConfig, GroupedSeries};
pub use heatmap::{Colormap, HeatmapData};
pub use hbar::{render_hbar_chart, HBar, HBarConfig};
pub use highdpi::{DpiConfig, PngMetadata};
pub use hist2d::{render_hist2d, Hist2DConfig};
pub use histogram::{BinningMethod, Histogram};
pub use inset::{InsetAxes, InsetConfig, Insets};
pub use interactive_html::{render_interactive_html, InteractiveConfig};
pub use jointplot::{render_jointplot, JointConfig};
pub use kde::{render_kde_plot, KdeConfig};
pub use legend::{LegendConfig, LegendItem, LegendLayout, LegendPosition};
pub use marimekko::{render_marimekko, MarimekkoColumn, MarimekkoConfig, MarimekkoSegment};
pub use ml_plots::*;
pub use pairplot::{render_pairplot, PairConfig};
pub use pareto::{render_pareto, ParetoBar, ParetoConfig};
pub use pie::{render_pie_chart, PieConfig, PieSlice};
pub use pointplot::{render_pointplot, PointCategory, PointConfig};
pub use polar::render_polar_svg;
pub use polar::{PolarData, PolarPoint, PolarSeries};
pub use quiver::{render_quiver, QuiverConfig, QuiverVector};
pub use radar::{render_radar_chart, RadarConfig, RadarPoint, RadarSeries};
pub use regplot::{render_regplot, RegPlotConfig};
pub use residplot::{render_residplot, ResidConfig};
pub use rug::{render_rug_plot, RugConfig};
pub use save::{ExportResult, FormatSet, OutputFormat, PlotSaver};
pub use smooth::{smooth_path, smooth_points, Interpolation, SmoothConfig};
pub use spectrogram::*;
pub use stats_annotate::{StatAnnotation, StatAnnotations, StatTest};
pub use stacked_bar::{render_stacked_bar, StackedBarConfig, StackedSeries};
pub use stem::{render_stem_plot, StemConfig};
pub use step::{render_step_plot, StepConfig, StepPosition};
pub use strip::{render_strip_plot, StripCategory, StripConfig};
pub use style::*;
pub use surface::*;
pub use svg::SvgPlot;
pub use swarm::{render_swarm_plot, SwarmCategory, SwarmConfig};
pub use terminal::TerminalPlot;
pub use theme::{ColorPalette, LineStyle, SpineVisibility, Theme, ThemeConfig};
pub use violin::{render_violin_plot, ViolinConfig, ViolinData};
pub use waterfall::{render_waterfall, WaterfallBar, WaterfallConfig};
