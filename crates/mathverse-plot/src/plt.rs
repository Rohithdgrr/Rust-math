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
    annotations: crate::annotations::Annotations,
    legend: crate::legend::LegendConfig,
    margin_frac: f64,
    xlim: Option<(f64, f64)>,
    ylim: Option<(f64, f64)>,
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
            .with_legend(params.show_legend);
        Self {
            config,
            theme,
            palette: params.palette,
            color_index: 0,
            series: Vec::new(),
            annotations: crate::annotations::Annotations::new(),
            legend: crate::legend::LegendConfig::default(),
            margin_frac: params.margin_frac,
            xlim: None,
            ylim: None,
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
        let data = crate::backend::PlotData {
            config: self.config.clone(),
            series: self.series.clone(),
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
        };
        crate::interactive_html::render_interactive_html(
            &data,
            &crate::interactive_html::InteractiveConfig::default(),
        )
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
        }
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

        for (row, col) in self.iter_positions() {
            let x = pad + col as f64 * (cell_w + self.spacing);
            let y = pad + row as f64 * (cell_h + self.spacing);
            let cell_svg = self.axes[row * self.cols + col].render();
            out.push_str(&format!(
                r#"  <svg x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" overflow="visible">"#
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
        for _ in 0..n {
            ax.color();
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
}