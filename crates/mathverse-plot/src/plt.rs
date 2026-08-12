//! A matplotlib-flavoured imperative plotting API.
//!
//! This mirrors how users drive matplotlib: you build a [`Figure`] of
//! [`Axes`], then mutate the returned axes handles in place — adding layered
//! "artists" (series), setting titles/labels/limits/scales, and letting
//! legends derive from the artists you added. New series are automatically
//! colour-cycled from the global palette (see [`crate::rcparams`]).
//!
//! # Example
//!
//! ```rust
//! use mathverse_plot::plt::Figure;
//!
//! let mut fig = Figure::subplots(1, 2);
//! fig.axes_at(0, 0)
//!     .set_title("Sine")
//!     .set_xlabel("x")
//!     .set_ylabel("sin(x)")
//!     .plot(&[0.0, 1.0, 2.0, 3.0], &[0.0, 0.84, 0.91, 0.14], "sin");
//! let svg = fig.render();
//! assert!(svg.contains("<svg"));
//! ```

use crate::axes::{Range, Scale};
use crate::common::{DataPoint, DataSeries, PlotConfig};
use crate::error::PlotResult;
use crate::heatmap::Colormap;
use crate::imshow::ImageData;
use crate::patches::{LineCollection, LineSnapshot, Patch, PathSnapshot};
use crate::rcparams::rc;
use crate::style::{Color, LineStyle, MarkerStyle, PlotStyle};
use crate::svg::SvgPlot;
use crate::theme::{ColorPalette, Theme, ThemeConfig};

/// A single axes holding layered, internally-mutable series artists.
#[derive(Debug, Clone)]
pub struct Axes {
    config: PlotConfig,
    theme: ThemeConfig,
    palette: ColorPalette,
    color_index: usize,
    series: Vec<DataSeries>,
    images: Vec<ImageData>,
    paths: Vec<PathSnapshot>,
    lines: Vec<LineSnapshot>,
    annotations: crate::annotations::Annotations,
    legend: crate::legend::LegendConfig,
    margin_frac: f64,
    xlim: Option<(f64, f64)>,
    ylim: Option<(f64, f64)>,
    /// Secondary series plotted on a right-side twin axis (matplotlib `twinx`).
    twin_series: Vec<DataSeries>,
    /// Label for the right-side twin axis.
    twin_label: Option<String>,
}

impl Default for Axes {
    fn default() -> Self {
        Self::new()
    }
}

impl Axes {
    /// Create an axes snapshotting the current global [`crate::rcparams`].
    pub fn new() -> Self {
        let params = rc();
        let theme = params.theme_config();
        let config = PlotConfig::new()
            .with_dimensions(params.figsize.0.max(1), params.figsize.1.max(1))
            .with_tick_count(params.tick_count)
            .with_x_scale(params.x_scale)
            .with_y_scale(params.y_scale)
            .with_legend(params.show_legend)
            .with_font_family(params.font_family.clone())
            .with_font_size(params.font_size);
        Self {
            config,
            theme,
            palette: params.palette,
            color_index: 0,
            series: Vec::new(),
            images: Vec::new(),
            paths: Vec::new(),
            lines: Vec::new(),
            annotations: crate::annotations::Annotations::new(),
            legend: crate::legend::LegendConfig::default(),
            margin_frac: params.margin_frac,
            xlim: None,
            ylim: None,
            twin_series: Vec::new(),
            twin_label: None,
        }
    }

    /// The next auto-cycled palette color (advances the internal cycle).
    #[must_use]
    pub fn color(&mut self) -> Color {
        let c = self.palette.get(self.color_index);
        self.color_index = self.color_index.wrapping_add(1);
        c
    }

    /// A mutable view of all layered series (artist mutability).
    pub fn series_mut(&mut self) -> &mut Vec<DataSeries> {
        &mut self.series
    }

    /// Mutable access to a single artist by index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of range.
    pub fn artist(&mut self, index: usize) -> &mut DataSeries {
        &mut self.series[index]
    }

    /// Number of layered artists added so far.
    pub fn len(&self) -> usize {
        self.series.len()
    }

    /// True when no artists have been added.
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// Add a line artist. Returns `&mut self` for chaining.
    pub fn plot(&mut self, xs: &[f64], ys: &[f64], name: impl Into<String>) -> &mut Self {
        self.add_series(xs, ys, name, None)
    }

    /// Add a scatter artist with circle markers auto-cycled to the palette.
    pub fn scatter(&mut self, xs: &[f64], ys: &[f64], name: impl Into<String>) -> &mut Self {
        let color = self.color();
        let style = PlotStyle::default()
            .with_line_color(color)
            .with_marker_color(color)
            .with_marker_style(MarkerStyle::Circle)
            .with_marker_size(5.0);
        self.push_series(name, xs, ys, style)
    }

    /// Add a `step` line artist (thin, no fill).
    pub fn step(&mut self, xs: &[f64], ys: &[f64], name: impl Into<String>) -> &mut Self {
        let color = self.color();
        let style = PlotStyle::default()
            .with_line_color(color)
            .with_line_style(LineStyle::Solid)
            .with_line_width(1.5);
        self.push_series(name, xs, ys, style)
    }

    /// Add a line artist with a solid line and optional style, auto-cycling
    /// the color only when no explicit style is given.
    pub fn add_series(
        &mut self,
        xs: &[f64],
        ys: &[f64],
        name: impl Into<String>,
        opts: Option<&PlotStyle>,
    ) -> &mut Self {
        let style = match opts {
            Some(s) => s.clone(),
            None => PlotStyle::default()
                .with_line_color(self.color())
                .with_line_style(LineStyle::Solid),
        };
        self.push_series(name, xs, ys, style)
    }

    /// Push an already-styled artist directly.
    pub fn push(&mut self, series: DataSeries) -> &mut Self {
        self.series.push(series);
        self
    }

    /// Draw a 2D array as a colormapped image (matplotlib `imshow`).
    ///
    /// # Errors
    ///
    /// Returns `PlotError::InvalidData` for empty or ragged grids.
    pub fn imshow(&mut self, grid: Vec<Vec<f64>>, colormap: Colormap) -> PlotResult<&mut Self> {
        self.images.push(ImageData::new(grid, colormap)?);
        Ok(self)
    }

    /// Add a styled path/patch artist (matplotlib `add_patch`).
    pub fn add_patch(&mut self, patch: &Patch) -> &mut Self {
        self.paths.push(PathSnapshot::from(patch));
        self
    }

    /// Add a batch of line segments (matplotlib `LineCollection`).
    pub fn add_line_collection(&mut self, collection: &LineCollection) -> &mut Self {
        self.lines.extend(Vec::from(collection));
        self
    }

    fn push_series(
        &mut self,
        name: impl Into<String>,
        xs: &[f64],
        ys: &[f64],
        style: PlotStyle,
    ) -> &mut Self {
        let points: Vec<DataPoint> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| DataPoint::new(x, y))
            .collect();
        self.series.push(DataSeries::with_style(name.into(), points, style));
        self
    }

    fn zip_points(&self, xs: &[f64], ys: &[f64]) -> Vec<DataPoint> {
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| DataPoint::new(x, y))
            .collect()
    }

    /// Set the title.
    pub fn set_title(&mut self, title: impl Into<String>) -> &mut Self {
        self.config.title = title.into();
        self
    }

    /// Set the x-axis label.
    pub fn set_xlabel(&mut self, label: impl Into<String>) -> &mut Self {
        self.config.x_label = label.into();
        self
    }

    /// Set the y-axis label.
    pub fn set_ylabel(&mut self, label: impl Into<String>) -> &mut Self {
        self.config.y_label = label.into();
        self
    }

    /// Explicit x-axis limits; `None` clears to auto-scale.
    pub fn set_xlim(&mut self, lo: Option<f64>, hi: Option<f64>) -> &mut Self {
        self.xlim = lims(lo, hi);
        self
    }

    /// Explicit y-axis limits; `None` clears to auto-scale.
    pub fn set_ylim(&mut self, lo: Option<f64>, hi: Option<f64>) -> &mut Self {
        self.ylim = lims(lo, hi);
        self
    }

    /// Set the x-axis scale (linear/log/symlog/sqrt).
    pub fn set_xscale(&mut self, scale: Scale) -> &mut Self {
        self.config.x_scale = scale;
        self
    }

    /// Set the y-axis scale (linear/log/symlog/sqrt).
    pub fn set_yscale(&mut self, scale: Scale) -> &mut Self {
        self.config.y_scale = scale;
        self
    }

    /// Toggle grid rendering.
    pub fn grid(&mut self, on: bool) -> &mut Self {
        self.config.show_grid = on;
        self
    }

    /// Set the theme preset explicitly (also swaps the auto-cycle palette).
    pub fn set_theme(&mut self, theme: Theme) -> &mut Self {
        self.theme = ThemeConfig::new(theme);
        self.palette = self.theme.palette.clone();
        self
    }

    /// Customise the legend position and enable the legend.
    pub fn legend(&mut self, pos: crate::legend::LegendPosition) -> &mut Self {
        self.legend = self.legend.clone().with_position(pos);
        self.config.show_legend = true;
        self
    }

    /// Hide the legend for this axes.
    pub fn no_legend(&mut self) -> &mut Self {
        self.config.show_legend = false;
        self
    }

    /// Fractional data margin applied around the auto-scaled bounds
    /// (defaults to the global `rcparams.margin_frac`).
    pub fn set_margin(&mut self, frac: f64) -> &mut Self {
        self.margin_frac = frac;
        self
    }

    /// The data bounds that will be used on the x axis, after applying any
    /// explicit limit or auto-scaling plus the configured margin.
    #[must_use]
    pub fn effective_x_range(&self) -> Range {
        self.effective_range(self.xlim, true)
    }

    /// The data bounds that will be used on the y axis.
    #[must_use]
    pub fn effective_y_range(&self) -> Range {
        self.effective_range(self.ylim, false)
    }

    fn effective_range(&self, explicit: Option<(f64, f64)>, is_x: bool) -> Range {
        if let Some((lo, hi)) = explicit {
            return Range { min: lo, max: hi }.pad(self.margin_frac);
        }
        let (xr, yr) = crate::common::plot_bounds(&self.series);
        let r = if is_x { xr } else { yr };
        r.pad(self.margin_frac)
    }

    /// The derived legend items (one per artist). This is how legends are
    /// built from the artists you added — no manual declaration.
    #[must_use]
    pub fn legend_items(&self) -> Vec<crate::legend::LegendItem> {
        self.series
            .iter()
            .map(|s| crate::legend::LegendItem::new(s.name.clone(), s.style.line_color))
            .collect()
    }

    /// Render this axes to a standalone SVG string.
    pub fn render(&self) -> String {
        let mut svg = SvgPlot::new(self.config.clone());
        svg = svg.with_theme(self.theme.clone());
        if self.config.show_legend {
            svg = svg.with_legend(self.legend.clone());
        }
        for s in &self.series {
            svg.add_series(s.clone());
        }
        for img in &self.images {
            svg.add_image(img.clone());
        }
        for path in &self.paths {
            svg.add_path_snapshot(path.clone());
        }
        for line in &self.lines {
            svg.add_line_snapshot(*line);
        }
        for s in &self.twin_series {
            svg.add_secondary(s.clone());
        }
        if let Some(ref label) = self.twin_label {
            svg.with_secondary_label(label.clone());
        }
        for line in &self.annotations.lines {
            svg.add_ref_line(line.clone());
        }
        for rect in &self.annotations.rectangles {
            svg.add_rect(rect.clone());
        }
        for arrow in &self.annotations.arrows {
            svg.add_arrow(arrow.clone());
        }
        for text in &self.annotations.texts {
            svg.add_text(text.clone());
        }
        svg.generate()
    }

    /// Add a horizontal reference line at `y` (matplotlib `axhline`).
    pub fn axhline(&mut self, y: f64) -> &mut Self {
        self.annotations
            .lines
            .push(crate::annotations::ReferenceLine::horizontal(y));
        self
    }

    /// Add a series on a secondary right-side axis sharing the x scale
    /// (matplotlib `twinx`). The secondary series uses its own y-range and
    /// its own tick labels on the right edge of the plot.
    pub fn twinx(&mut self, xs: &[f64], ys: &[f64], name: impl Into<String>) -> &mut Self {
        let color = self.color();
        let style = PlotStyle::default()
            .with_line_color(color)
            .with_marker_color(color)
            .with_marker_style(MarkerStyle::Circle)
            .with_marker_size(3.0);
        self.twin_series
            .push(DataSeries::with_style(name.into(), self.zip_points(xs, ys), style));
        self
    }

    /// Set the label for the right-side twin axis (matplotlib `twinx` ylabel).
    pub fn set_twin_ylabel(&mut self, label: impl Into<String>) -> &mut Self {
        self.twin_label = Some(label.into());
        self
    }

    /// Add a vertical reference line at `x` (matplotlib `axvline`).
    pub fn axvline(&mut self, x: f64) -> &mut Self {
        self.annotations
            .lines
            .push(crate::annotations::ReferenceLine::vertical(x));
        self
    }

    /// Add a horizontal span from `ymin` to `ymax` across the full plot
    /// width (matplotlib `axhspan`).
    pub fn axhspan(&mut self, ymin: f64, ymax: f64) -> &mut Self {
        let xr = self.effective_x_range();
        self.annotations
            .rectangles
            .push(
                crate::annotations::Rectangle::new(
                    crate::common::DataPoint::new(xr.min, ymin),
                    xr.max - xr.min,
                    ymax - ymin,
                )
                .with_fill(crate::style::Color::rgb(173, 216, 230))
                .with_stroke(crate::style::Color::BLUE),
            );
        self
    }

    /// Add a vertical span from `xmin` to `xmax` across the full plot
    /// height (matplotlib `axvspan`).
    pub fn axvspan(&mut self, xmin: f64, xmax: f64) -> &mut Self {
        let yr = self.effective_y_range();
        self.annotations
            .rectangles
            .push(
                crate::annotations::Rectangle::new(
                    crate::common::DataPoint::new(xmin, yr.min),
                    xmax - xmin,
                    yr.max - yr.min,
                )
                .with_fill(crate::style::Color::rgb(173, 216, 230))
                .with_stroke(crate::style::Color::BLUE),
            );
        self
    }

    /// Save this axes to files via the unified multi-backend saver.
    pub fn save(
        &self,
        base_path: &str,
        formats: &crate::save::FormatSet,
    ) -> Vec<crate::save::ExportResult> {
        crate::save::PlotSaver::new(&self.render())
            .with_dimensions(self.config.width, self.config.height)
            .with_title(self.config.title.clone())
            .save(base_path, formats)
    }

    /// Run `steps` animation frames, calling `frame` to mutate `self` before
    /// snapshotting each rendered SVG into `store`.
    pub fn animate(
        &mut self,
        steps: usize,
        mut frame: impl FnMut(&mut Axes, usize),
        store: &mut Vec<String>,
    ) {
        for i in 0..steps {
            frame(self, i);
            store.push(self.render());
        }
    }

    /// Wrap this axes in the interactive HTML pan/zoom viewer.
    pub fn interactive(&self) -> crate::error::PlotResult<String> {
        self.interactive_with(&crate::interactive_html::InteractiveConfig::default())
    }

    /// Wrap this axes in the interactive viewer with a custom configuration
    /// (e.g. point-click callbacks).
    pub fn interactive_with(
        &self,
        config: &crate::interactive_html::InteractiveConfig,
    ) -> crate::error::PlotResult<String> {
        let data = crate::backend::PlotData {
            config: self.config.clone(),
            series: self.series.clone(),
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
            images: Vec::new(),
            paths: Vec::new(),
            lines: Vec::new(),
        };
        crate::interactive_html::render_interactive_html(&data, config)
    }
}

fn lims(lo: Option<f64>, hi: Option<f64>) -> Option<(f64, f64)> {
    match (lo, hi) {
        (Some(lo), Some(hi)) => Some((lo, hi)),
        // A single bound behaves as an explicit range against the auto range.
        _ => None,
    }
}

/// A grid of subplots with optional shared axis labels.
#[derive(Debug, Clone)]
pub struct Figure {
    rows: usize,
    cols: usize,
    axes: Vec<Axes>,
    shared_x_label: Option<String>,
    shared_y_label: Option<String>,
    spacing: f64,
    /// Per-row relative heights (GridSpec `height_ratios`).
    height_ratios: Vec<f64>,
    /// Per-column relative widths (GridSpec `width_ratios`).
    width_ratios: Vec<f64>,
    /// Optional row/col span per axes: `(row_span, col_span)` in grid cells.
    spans: Vec<(usize, usize)>,
    /// Extra axes placed by fractional figure coordinates `(l, b, w, h)` in
    /// [0, 1] — the analogue of `fig.add_axes([l, b, w, h])`.
    extra_axes: Vec<(f64, f64, f64, f64, Axes)>,
}

impl Figure {
    /// Create a `rows × cols` grid of subplots.
    pub fn subplots(rows: usize, cols: usize) -> Self {
        let rows = rows.max(1);
        let cols = cols.max(1);
        Self {
            rows,
            cols,
            axes: (0..rows * cols).map(|_| Axes::new()).collect(),
            shared_x_label: None,
            shared_y_label: None,
            spacing: 10.0,
            height_ratios: Vec::new(),
            width_ratios: Vec::new(),
            spans: Vec::new(),
            extra_axes: Vec::new(),
        }
    }

    /// Create a grid with per-row and per-column size ratios (matplotlib
    /// `gridspec.GridSpec(height_ratios=..., width_ratios=...)`).
    pub fn subplots_with_ratios(
        rows: usize,
        cols: usize,
        height_ratios: &[f64],
        width_ratios: &[f64],
    ) -> Self {
        let mut fig = Self::subplots(rows, cols);
        fig.height_ratios = height_ratios.to_vec();
        fig.width_ratios = width_ratios.to_vec();
        fig
    }

    /// Create a figure with a single axes at fractional figure coordinates
    /// `(l, b, w, h)` in `[0, 1]` (matplotlib `fig.add_axes`).
    pub fn add_axes(l: f64, b: f64, w: f64, h: f64) -> Self {
        let mut fig = Self::subplots(1, 1);
        fig.axes.clear();
        fig.extra_axes = vec![(l, b, w, h, Axes::new())];
        fig
    }

    /// Set the span (in grid cells) of the axes at `(row, col)` — the
    /// analogue of `fig.add_subplot(row, col, rowspan, colspan)`.
    pub fn set_span(&mut self, row: usize, col: usize, row_span: usize, col_span: usize) -> &mut Self {
        let idx = row * self.cols + col;
        if row_span > 0 && col_span > 0 {
            if self.spans.len() <= idx {
                self.spans.resize(idx + 1, (1, 1));
            }
            self.spans[idx] = (row_span, col_span);
        }
        self
    }

    /// Add a free-floating axes at fractional figure coordinates `(l, b, w, h)`
    /// in `[0, 1]` (matplotlib `fig.add_axes`). It renders on top of the grid.
    pub fn add_extra_axes(&mut self, l: f64, b: f64, w: f64, h: f64, axes: Axes) -> &mut Self {
        self.extra_axes.push((l, b, w, h, axes));
        self
    }

    /// Mutable access to a free-floating axes placed with [`Self::add_extra_axes`].
    pub fn extra_axes_mut(&mut self, index: usize) -> Option<&mut Axes> {
        self.extra_axes.get_mut(index).map(|(_, _, _, _, a)| a)
    }

    /// Mutable reference to the axes at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if `(row, col)` is out of bounds.
    pub fn axes_at(&mut self, row: usize, col: usize) -> &mut Axes {
        let idx = row * self.cols + col;
        assert!(
            row < self.rows && col < self.cols,
            "axes position ({row},{col}) out of bounds for {rows}x{cols}",
            rows = self.rows,
            cols = self.cols
        );
        &mut self.axes[idx]
    }

    /// Immutable reference to the axes at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> Option<&Axes> {
        let idx = row.checked_mul(self.cols)?.checked_add(col)?;
        (row < self.rows && col < self.cols).then(|| &self.axes[idx])
    }

    /// Set a shared x-axis label shown once below the bottom row.
    pub fn set_shared_xlabel(&mut self, label: impl Into<String>) -> &mut Self {
        self.shared_x_label = Some(label.into());
        self
    }

    /// Set a shared y-axis label shown once left of the leftmost column.
    pub fn set_shared_ylabel(&mut self, label: impl Into<String>) -> &mut Self {
        self.shared_y_label = Some(label.into());
        self
    }

    /// Set the gap between subplots in px.
    pub fn set_spacing(&mut self, px: f64) -> &mut Self {
        self.spacing = px;
        self
    }

    /// Auto-adjust subplot parameters for a tight layout (matplotlib `tight_layout`).
    pub fn tight_layout(&mut self) -> &mut Self {
        // Estimate per-subplot text height for labels and titles
        let text_height = 60.0; // px for title + xlabel + padding
        let text_width = 80.0; // px for ylabel + padding
        let total_w = self.cols as f64 * (800.0 + text_width) + (self.cols - 1) as f64 * self.spacing;
        let total_h = self.rows as f64 * (600.0 + text_height) + (self.rows - 1) as f64 * self.spacing;
        // Adjust each axes config to fit within the computed bounds
        let w = ((total_w - (self.cols + 1) as f64 * 20.0) / self.cols as f64) as u32;
        let h = ((total_h - (self.rows + 1) as f64 * 20.0) / self.rows as f64) as u32;
        for ax in &mut self.axes {
            ax.config.width = w;
            ax.config.height = h;
            ax.config.padding = 40.0;
        }
        self
    }

    /// Adjust subplot parameters (matplotlib `subplots_adjust`).
    pub fn subplots_adjust(
        &mut self,
        left: Option<f64>,
        right: Option<f64>,
        top: Option<f64>,
        bottom: Option<f64>,
        wspace: Option<f64>,
        hspace: Option<f64>,
    ) -> &mut Self {
        let l = left.unwrap_or(0.125);
        let r = right.unwrap_or(0.9);
        let t = top.unwrap_or(0.88);
        let b = bottom.unwrap_or(0.12);
        let ws = wspace.unwrap_or(0.2);
        let hs = hspace.unwrap_or(0.2);
        // Use first axes dimensions as reference
        if let Some(first) = self.axes.first() {
            let total_w = first.config.width as f64;
            let total_h = first.config.height as f64;
            let plot_w = (r - l) * total_w;
            let plot_h = (t - b) * total_h;
            let cell_w = (plot_w - ws * (self.cols - 1) as f64 * total_w) / self.cols as f64;
            let cell_h = (plot_h - hs * (self.rows - 1) as f64 * total_h) / self.rows as f64;
            for ax in &mut self.axes {
                ax.config.width = cell_w as u32;
                ax.config.height = cell_h as u32;
                ax.config.padding = l * total_w;
            }
        }
        self
    }

    /// Apply a theme preset to every axes.
    pub fn set_theme(&mut self, theme: Theme) -> &mut Self {
        for ax in &mut self.axes {
            ax.set_theme(theme);
        }
        self
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Render the whole grid into one SVG page with shared labels.
    pub fn render(&self) -> String {
        let cell_w = self
            .axes
            .first()
            .map(|a| a.config.width as f64)
            .unwrap_or(800.0);
        let cell_h = self
            .axes
            .first()
            .map(|a| a.config.height as f64)
            .unwrap_or(600.0);

        let pad = 60.0;
        let wr: Vec<f64> = if self.width_ratios.is_empty() {
            vec![1.0; self.cols]
        } else {
            self.width_ratios.clone()
        };
        let hr: Vec<f64> = if self.height_ratios.is_empty() {
            vec![1.0; self.rows]
        } else {
            self.height_ratios.clone()
        };
        let wr_sum: f64 = wr.iter().sum::<f64>().max(1e-9);
        let hr_sum: f64 = hr.iter().sum::<f64>().max(1e-9);
        let col_widths: Vec<f64> = wr.iter().map(|r| cell_w * r / wr_sum).collect();
        let row_heights: Vec<f64> = hr.iter().map(|r| cell_h * r / hr_sum).collect();

        let total_w =
            self.cols as f64 * cell_w + (self.cols as f64 - 1.0) * self.spacing + 2.0 * pad;
        let total_h =
            self.rows as f64 * cell_h + (self.rows as f64 - 1.0) * self.spacing + 2.0 * pad;

        let mut out = String::with_capacity(8 * 1024);
        out.push_str(&format!(
            r#"<svg width="{tw}" height="{th}" viewBox="0 0 {tw} {th}" xmlns="http://www.w3.org/2000/svg">"#,
            tw = total_w,
            th = total_h
        ));
        out.push('\n');

        let span_of = |idx: usize| -> (usize, usize) {
            self.spans.get(idx).copied().unwrap_or((1, 1))
        };
        for (row, col) in self.iter_positions() {
            let idx = row * self.cols + col;
            let (row_span, col_span) = span_of(idx);
            let ax_w = if let Some(a) = self.axes.get(idx) {
                a.config.width as f64
            } else {
                cell_w
            };
            let ax_h = if let Some(a) = self.axes.get(idx) {
                a.config.height as f64
            } else {
                cell_h
            };
            // Position of the top-left cell of this axes.
            let x0: f64 = col_widths[..col].iter().sum::<f64>();
            let y0: f64 = row_heights[..row].iter().sum::<f64>();
            let span_w: f64 = col_widths[col..(col + col_span).min(self.cols)].iter().sum::<f64>()
                + col_span.saturating_sub(1) as f64 * self.spacing;
            let span_h: f64 = row_heights[row..(row + row_span).min(self.rows)].iter().sum::<f64>()
                + row_span.saturating_sub(1) as f64 * self.spacing;
            let x = pad + x0;
            let y = pad + y0;
            let w = if col_span > 1 { span_w } else { ax_w };
            let h = if row_span > 1 { span_h } else { ax_h };
            let cell_svg = self.axes[idx].render();
            out.push_str(&format!(
                r#"  <svg x="{x}" y="{y}" width="{w}" height="{h}" overflow="visible">"#
            ));
            out.push('\n');
            out.push_str(&strip_svg_tag(&cell_svg));
            out.push_str("  </svg>\n");
        }

        // Fractional-placement axes (fig.add_axes) drawn on top.
        for (l, b, w, h, axes) in &self.extra_axes {
            let x = l * total_w;
            let y = b * total_h;
            let ew = w * total_w;
            let eh = h * total_h;
            let cell_svg = axes.render();
            out.push_str(&format!(
                r#"  <svg x="{x}" y="{y}" width="{ew}" height="{eh}" overflow="visible">"#
            ));
            out.push('\n');
            out.push_str(&strip_svg_tag(&cell_svg));
            out.push_str("  </svg>\n");
        }

        if let Some(label) = &self.shared_x_label {
            out.push_str(&format!(
                r##"  <text x="{}" y="{}" text-anchor="middle" font-size="14" fill="#000">{}</text>"##,
                total_w / 2.0,
                total_h - 20.0,
                crate::common::xml_escape(label)
            ));
            out.push('\n');
        }
        if let Some(label) = &self.shared_y_label {
            out.push_str(&format!(
                r##"  <text x="20" y="{}" text-anchor="middle" font-size="14" fill="#000" transform="rotate(-90, 20, {})">{}</text>"##,
                total_h / 2.0,
                total_h / 2.0,
                crate::common::xml_escape(label)
            ));
            out.push('\n');
        }

        out.push_str("</svg>");
        out
    }

    /// Save the full figure to files via the unified multi-backend saver.
    pub fn save(
        &self,
        base_path: &str,
        formats: &crate::save::FormatSet,
    ) -> Vec<crate::save::ExportResult> {
        crate::save::PlotSaver::new(&self.render())
            .with_title("Figure")
            .save(base_path, formats)
    }

    fn iter_positions(&self) -> impl Iterator<Item = (usize, usize)> + use<'_> {
        (0..self.rows).flat_map(move |r| (0..self.cols).map(move |c| (r, c)))
    }
}

/// Remove the outer `<svg ...></svg>` wrapper from a rendered cell.
fn strip_svg_tag(svg: &str) -> String {
    let start = match svg.find("<svg") {
        Some(i) => i,
        None => return svg.to_string(),
    };
    let open_end = match svg[start..].find('>') {
        Some(i) => start + i,
        None => return svg.to_string(),
    };
    let body_start = open_end + 1;
    match svg.rfind("</svg>") {
        Some(c) => svg[body_start..c].to_string(),
        None => svg[body_start..].to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_plot_and_mutate_artist() {
        let mut ax = Axes::new();
        ax.plot(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], "a")
            .set_title("T")
            .set_xlabel("x")
            .set_ylabel("y");
        assert_eq!(ax.len(), 1);
        ax.artist(0).name = "renamed".into();
        assert_eq!(ax.artist(0).name, "renamed");
        let svg = ax.render();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("T"));
    }

    #[test]
    fn color_cycling_advances() {
        let mut ax = Axes::new();
        let c0 = ax.color();
        let c1 = ax.color();
        assert_ne!(c0, c1);
        let n = ax.palette.colors.len();
        assert!(n >= 2);
        // Consume the rest of the palette; the next draw wraps to the start.
        for _ in 2..n {
            let _ = ax.color();
        }
        let c_again = ax.color();
        assert_eq!(c0, c_again);
    }

    #[test]
    fn figure_subplots_and_axes() {
        let mut fig = Figure::subplots(1, 2);
        fig.axes_at(0, 0).set_title("left").plot(&[0.0, 1.0], &[0.0, 1.0], "s");
        fig.axes_at(0, 1).set_title("right");
        let svg = fig.render();
        assert!(svg.contains("left"));
        assert!(svg.contains("right"));
    }

    #[test]
    fn explicit_xlim_overrides() {
        let mut ax = Axes::new();
        ax.set_xlim(Some(-5.0), Some(5.0))
            .plot(&[10.0, 11.0], &[0.0, 1.0], "s");
        let r = ax.effective_x_range();
        assert!(r.min <= -5.0 && r.max >= 5.0);
    }

    #[test]
    fn animation_produces_frames() {
        let mut ax = Axes::new();
        let mut frames = Vec::new();
        ax.animate(3, |a, i| {
            a.plot(&[0.0, 1.0], &[i as f64, 0.0], format!("t{i}"));
        }, &mut frames);
        assert_eq!(frames.len(), 3);
        assert_eq!(ax.len(), 3);
    }

    #[test]
    fn legend_items_derive_from_artists() {
        let mut ax = Axes::new();
        ax.plot(&[0.0, 1.0], &[0.0, 1.0], "alpha")
            .plot(&[0.0, 1.0], &[1.0, 0.0], "beta");
        let items = ax.legend_items();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].name, "alpha");
        assert_eq!(items[1].name, "beta");
    }

    #[test]
    fn twinx_renders_right_axis() {
        let mut ax = Axes::new();
        ax.plot(&[0.0, 1.0, 2.0], &[0.0, 1.0, 4.0], "main")
            .twinx(&[0.0, 1.0, 2.0], &[100.0, 50.0, 0.0], "secondary")
            .set_twin_ylabel("right side");
        let svg = ax.render();
        // Right-side axis ticks must appear with the secondary scale.
        assert!(svg.contains("100"));
        assert!(svg.contains("50"));
        assert!(svg.contains("rotate(90"));
    }

    #[test]
    fn figure_ratios_change_geometry() {
        let fig = Figure::subplots_with_ratios(2, 1, &[3.0, 1.0], &[1.0]);
        assert_eq!(fig.rows(), 2);
        let svg = fig.render();
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn figure_span_and_extra_axes_render() {
        let mut fig = Figure::subplots(2, 2);
        fig.set_span(0, 0, 1, 2); // top row spans both columns
        let mut inset = Axes::new();
        inset.plot(&[0.0, 1.0], &[0.0, 1.0], "inset");
        fig.add_extra_axes(0.6, 0.6, 0.3, 0.3, inset);
        let svg = fig.render();
        assert!(svg.contains("<svg"));
    }
}