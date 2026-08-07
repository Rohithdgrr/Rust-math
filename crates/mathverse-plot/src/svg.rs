//! SVG plotting backend

use crate::axes::{axis_kernel, Range, Scale};
use crate::backend::{BarSnapshot, BoxSnapshot, ErrorBarSnapshot, PlotData, PlotOutput};
use crate::boxplot::BoxStats;
use crate::common::{DataSeries, PlotConfig};
use crate::error::PlotResult;
use crate::errorbar::ErrorBar;
use crate::heatmap::{Colormap, HeatmapData};
use crate::style::{Color, LineStyle, MarkerStyle};
use crate::theme::ThemeConfig;

/// A filled rectangle spanning `x_lo..x_hi` with height `y` (histogram bar).
#[derive(Debug, Clone, Copy)]
struct BarData {
    x_lo: f64,
    x_hi: f64,
    y: f64,
    color: Color,
}

/// A rendered box at integer slot `i` (data x = `i`).
#[derive(Debug, Clone)]
struct BoxData {
    name: String,
    stats: BoxStats,
    color: Color,
}

/// A vertical error bar at data x with a center marker.
#[derive(Debug, Clone, Copy)]
struct ErrorBarData {
    x: f64,
    bar: ErrorBar,
    color: Color,
}

/// SVG plot generator
pub struct SvgPlot {
    config: PlotConfig,
    series: Vec<DataSeries>,
    bars: Vec<BarData>,
    boxes: Vec<BoxData>,
    error_bars: Vec<ErrorBarData>,
    heatmaps: Vec<HeatmapData>,
    /// Optional theme for styled rendering.
    theme: Option<ThemeConfig>,
    /// Optional legend configuration.
    legend: Option<crate::legend::LegendConfig>,
    /// Annotations (arrows, text, shapes).
    annotations: crate::annotations::Annotations,
    /// Optional colorbar for heatmap/colormapped data.
    colorbar: Option<ColorbarConfig>,
}

/// Configuration for a colorbar legend.
#[derive(Debug, Clone)]
pub struct ColorbarConfig {
    /// Colormap to use.
    pub colormap: crate::heatmap::Colormap,
    /// Label for the colorbar.
    pub label: String,
    /// Data range for the colorbar.
    pub range: (f64, f64),
}

impl SvgPlot {
    /// Create a new SVG plot
    pub fn new(config: PlotConfig) -> Self {
        SvgPlot {
            config,
            series: Vec::new(),
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
            theme: None,
            legend: None,
            annotations: crate::annotations::Annotations::new(),
            colorbar: None,
        }
    }

    /// Set a theme for styled rendering.
    pub fn with_theme(mut self, theme: ThemeConfig) -> Self {
        self.theme = Some(theme);
        self
    }

    /// Set the legend configuration.
    pub fn with_legend(mut self, legend: crate::legend::LegendConfig) -> Self {
        self.legend = Some(legend);
        self
    }

    /// Set a colorbar for heatmap/colormapped data.
    pub fn with_colorbar(mut self, colormap: crate::heatmap::Colormap, label: impl Into<String>, range: (f64, f64)) -> Self {
        self.colorbar = Some(ColorbarConfig {
            colormap,
            label: label.into(),
            range,
        });
        self
    }

    /// Add an arrow annotation.
    pub fn add_arrow(&mut self, arrow: crate::annotations::Arrow) {
        self.annotations.arrows.push(arrow);
    }

    /// Add a text annotation.
    pub fn add_text(&mut self, text: crate::annotations::TextAnnotation) {
        self.annotations.texts.push(text);
    }

    /// Add a reference line (horizontal or vertical).
    pub fn add_ref_line(&mut self, line: crate::annotations::ReferenceLine) {
        self.annotations.lines.push(line);
    }

    /// Add a rectangle annotation.
    pub fn add_rect(&mut self, rect: crate::annotations::Rectangle) {
        self.annotations.rectangles.push(rect);
    }

    /// Get the legend configuration (if set).
    pub fn legend(&self) -> Option<&crate::legend::LegendConfig> {
        self.legend.as_ref()
    }

    /// Get the current theme (if set).
    pub fn theme(&self) -> Option<&ThemeConfig> {
        self.theme.as_ref()
    }

    /// Add a data series to the plot
    pub fn add_series(&mut self, series: DataSeries) {
        self.series.push(series);
    }

    /// Add a filled bar spanning `x_lo..x_hi` with height `y`.
    pub fn add_bar(&mut self, x_lo: f64, x_hi: f64, y: f64, color: Color) {
        self.bars.push(BarData {
            x_lo,
            x_hi,
            y,
            color,
        });
    }

    /// Add a Tukey box plot for `xs`, centered at slot `i` (data x = `i`).
    ///
    /// # Errors
    ///
    /// Returns `PlotError::InvalidData` for empty or non-finite input.
    pub fn add_box_plot(
        &mut self,
        name: impl Into<String>,
        xs: &[f64],
        color: Color,
    ) -> PlotResult<()> {
        let stats = BoxStats::compute(xs)?;
        self.boxes.push(BoxData {
            name: name.into(),
            stats,
            color,
        });
        Ok(())
    }

    /// Add a vertical error bar at data x (whisker + center marker).
    pub fn add_error_bar(&mut self, x: f64, bar: ErrorBar, color: Color) {
        self.error_bars.push(ErrorBarData { x, bar, color });
    }

    /// Add a Gaussian KDE overlay for `xs` as a line series, sampled on a
    /// uniform grid spanning the data range. All density math comes from
    /// `mathverse_statistics`.
    ///
    /// `scale` multiplies the density: `1.0` gives a probability density
    /// (integrates to 1); `n * bin_width` matches a histogram's area.
    ///
    /// # Errors
    ///
    /// Returns `PlotError::InvalidData` for empty or non-finite input.
    pub fn add_kde_overlay(
        &mut self,
        name: impl Into<String>,
        xs: &[f64],
        bandwidth: mathverse_statistics::Bandwidth,
        scale: f64,
        color: Color,
        points: usize,
    ) -> PlotResult<()> {
        if xs.is_empty() {
            return Err(crate::error::PlotError::InvalidData("empty data".into()));
        }
        if xs.iter().any(|x| !x.is_finite()) {
            return Err(crate::error::PlotError::InvalidData(
                "non-finite data value".into(),
            ));
        }
        let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
        let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let curve = mathverse_statistics::kernel_density_curve(xs, bandwidth, min, max, points);
        let series = DataSeries::with_style(
            name.into(),
            curve
                .into_iter()
                .map(|(x, y)| crate::common::DataPoint::new(x, y * scale))
                .collect(),
            crate::style::PlotStyle::default().with_line_color(color),
        );
        self.series.push(series);
        Ok(())
    }

    /// Add a heatmap grid with `colormap`. Axes show row/column indices.
    ///
    /// # Errors
    ///
    /// Returns `PlotError::InvalidData` for empty or ragged grids.
    pub fn add_heatmap(
        &mut self,
        name: impl Into<String>,
        grid: Vec<Vec<f64>>,
        colormap: Colormap,
    ) -> PlotResult<()> {
        self.heatmaps.push(HeatmapData::new(name, grid, colormap)?);
        Ok(())
    }

    /// Snapshot of all data for the backend trait.
    pub fn snapshot(&self) -> PlotData {
        PlotData {
            config: self.config.clone(),
            series: self.series.clone(),
            bars: self
                .bars
                .iter()
                .map(|b| BarSnapshot {
                    x_lo: b.x_lo,
                    x_hi: b.x_hi,
                    y: b.y,
                    color: b.color,
                })
                .collect(),
            boxes: self
                .boxes
                .iter()
                .map(|b| BoxSnapshot {
                    name: b.name.clone(),
                    stats: b.stats.clone(),
                    color: b.color,
                })
                .collect(),
            error_bars: self
                .error_bars
                .iter()
                .map(|e| ErrorBarSnapshot {
                    x: e.x,
                    bar: e.bar,
                    color: e.color,
                })
                .collect(),
            heatmaps: self.heatmaps.clone(),
        }
    }

    /// Add a theoretical PDF overlay. `pdf` is any `f(x)` function;
    /// sampling over `[lo, hi]` with `n` points creates a line series.
    pub fn add_pdf_overlay(
        &mut self,
        name: impl Into<String>,
        pdf: Box<dyn Fn(f64) -> f64>,
        lo: f64,
        hi: f64,
        n: usize,
        color: Color,
    ) {
        let series = DataSeries::with_style(
            name.into(),
            crate::pdf_overlay::sample_pdf(&pdf, lo, hi, n)
                .into_iter()
                .map(|(x, y)| crate::common::DataPoint::new(x, y))
                .collect(),
            crate::style::PlotStyle::default().with_line_color(color),
        );
        self.series.push(series);
    }

    /// Generate the SVG string
    pub fn generate(&self) -> String {
        let mut svg = String::new();

        // Get theme settings (use defaults if no theme set)
        let (bg_color, text_color, _axis_color, font_family, title_size, label_size, _tick_size) = if let Some(ref theme) = self.theme {
            (
                theme.background_color.to_hex(),
                theme.text_color.to_hex(),
                theme.axis_color.to_hex(),
                theme.font_family.clone(),
                theme.title_size,
                theme.label_size,
                theme.tick_size,
            )
        } else {
            (
                Color::WHITE.to_hex(),
                Color::BLACK.to_hex(),
                Color::BLACK.to_hex(),
                "Arial, sans-serif".to_string(),
                20.0,
                14.0,
                11.0,
            )
        };

        // SVG header with viewBox for responsive scaling
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" shape-rendering="geometricPrecision" text-rendering="optimizeLegibility">"#,
            self.config.width, self.config.height, self.config.width, self.config.height
        ));
        svg.push('\n');

        // Basic tooltip on the plot canvas
        let tooltip_text = if self.config.title.is_empty() {
            "Plot".to_string()
        } else {
            self.config.title.clone()
        };
        svg.push_str(&format!(
            r#"  <title>{}</title>"#,
            crate::common::xml_escape(&tooltip_text)
        ));
        svg.push('\n');

        // SVG defs: clipPath, filters, gradients
        let plot_area_width = self.config.width as f64 - 2.0 * self.config.padding as f64;
        let plot_area_height = self.config.height as f64 - 2.0 * self.config.padding as f64;
        svg.push_str(&format!(
            r##"  <defs>
    <clipPath id="plot-area">
      <rect x="{}" y="{}" width="{}" height="{}"/>
    </clipPath>
    <filter id="drop-shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="1" dy="1" stdDeviation="2" flood-color="#000" flood-opacity="0.15"/>
    </filter>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>"##,
            self.config.padding,
            self.config.padding,
            plot_area_width,
            plot_area_height
        ));
        svg.push('\n');

        // Background
        svg.push_str(&format!(
            r#"  <rect width="100%" height="100%" fill="{}"/>"#,
            bg_color
        ));
        svg.push('\n');

        // Calculate plot area
        let padding = self.config.padding;
        let plot_width = self.config.width as f64 - 2.0 * padding;
        let plot_height = self.config.height as f64 - 2.0 * padding;

        // Calculate data ranges (unpadded; padding happens in kernel space)
        let (xr, yr) = self.calculate_ranges();

        // Scale-aware coordinate mapping (kernel space, NaN-safe fallback)
        let (x_t, x_inv, kx) = axis_kernel(self.config.x_scale, xr);
        let (y_t, y_inv, ky) = axis_kernel(self.config.y_scale, yr);
        let x_px_k = |k: f64| padding + (k - kx.min) / kx.span() * plot_width;
        let y_px_k = |k: f64| padding + plot_height - (k - ky.min) / ky.span() * plot_height;
        let x_px = |x: f64| x_px_k(x_t(x));
        let y_px = |y: f64| y_px_k(y_t(y));

        // Kernel-space tick positions with data-space labels
        let x_ticks: Vec<(f64, String)> = Scale::Linear
            .ticks(kx.min, kx.max, self.config.tick_count)
            .into_iter()
            .map(|k| (k, format_tick(x_inv(k))))
            .collect();
        let y_ticks: Vec<(f64, String)> = Scale::Linear
            .ticks(ky.min, ky.max, self.config.tick_count)
            .into_iter()
            .map(|k| (k, format_tick(y_inv(k))))
            .collect();

    // Draw grid if enabled (theme-driven opacity / dash pattern)
    if self.config.show_grid {
        svg.push_str(&self.generate_grid(
            &x_ticks,
            &y_ticks,
            &x_px_k,
            &y_px_k,
            padding,
            plot_height,
        ));
        svg.push('\n');
    }

        // Draw axes
        svg.push_str(&self.generate_axes(padding, plot_width, plot_height));

        // Tick labels + marks
        svg.push_str(&self.generate_ticks(
            &x_ticks,
            &y_ticks,
            &x_px_k,
            &y_px_k,
            padding,
            plot_height,
        ));

        // Draw heatmaps (before series so they appear as background)
        for h in &self.heatmaps {
            for r in 0..h.rows() {
                for c in 0..h.cols() {
                    svg.push_str(&h.render_cell(r, c, &x_px, &y_px));
                    svg.push('\n');
                }
            }
        }

        // Draw bars (histogram) under series
        for bar in &self.bars {
            svg.push_str(&Self::generate_bar(bar, &x_px, &y_px, padding, plot_height));
        }

        // Draw data series
        for series in &self.series {
            svg.push_str(&self.generate_series(series, &x_px, &y_px));
        }

        // Draw box plots
        for (i, b) in self.boxes.iter().enumerate() {
            svg.push_str(&Self::generate_box(i, b, &x_px, &y_px));
        }

        // Draw error bars (above series so they stay visible)
        for e in &self.error_bars {
            svg.push_str(&Self::generate_error_bar(e, &x_px, &y_px));
        }

        // Draw annotations (arrows, text, shapes)
        svg.push_str(&self.generate_annotations(&x_px, &y_px));

    // Draw title (matplotlib places it centered near the top with bold)
    if !self.config.title.is_empty() {
        svg.push_str(&format!(
            r#" <text x="{cx}" y="{title_y}" text-anchor="middle" font-size="{title_size}" font-family="{font_family}" font-weight="bold" fill="{text_color}">{escaped}</text>"#,
            cx = self.config.width as f64 / 2.0,
            title_y = 32.0,
            escaped = crate::common::xml_escape(&self.config.title),
        ));
        svg.push('\n');
    }

    // X-axis label (bottom center, below ticks — matplotlib default)
    if !self.config.x_label.is_empty() {
        svg.push_str(&format!(
            r#" <text x="{cx}" y="{label_y}" text-anchor="middle" font-size="{label_size}" font-family="{font_family}" fill="{text_color}">{escaped}</text>"#,
            cx = self.config.width as f64 / 2.0,
            label_y = self.config.height as f64 - 2.0,
            escaped = crate::common::xml_escape(&self.config.x_label),
        ));
        svg.push('\n');
    }

    // Y-axis label (rotated 90°, centered vertically — matplotlib default)
    if !self.config.y_label.is_empty() {
        let cy = self.config.height as f64 / 2.0;
        svg.push_str(&format!(
            r#" <text x="14" y="{cy}" text-anchor="middle" dominant-baseline="central" font-size="{label_size}" font-family="{font_family}" fill="{text_color}" transform="rotate(-90, 14, {cy})">{escaped}</text>"#,
            escaped = crate::common::xml_escape(&self.config.y_label),
        ));
        svg.push('\n');
    }

        // Draw legend if enabled
        if self.config.show_legend && (!self.series.is_empty() || !self.boxes.is_empty()) {
            svg.push_str(&self.generate_legend());
        }

        // SVG footer
        svg.push_str("</svg>");

        svg
    }

    /// Bounds over series, bars, boxes and heatmaps. Falls back to `0..1` when empty.
    fn calculate_ranges(&self) -> (Range, Range) {
        (
            crate::common::compute_x_range(&self.snapshot()),
            crate::common::compute_y_range(&self.snapshot()),
        )
    }

    fn generate_grid(
        &self,
        x_ticks: &[(f64, String)],
        y_ticks: &[(f64, String)],
        x_px_k: &dyn Fn(f64) -> f64,
        y_px_k: &dyn Fn(f64) -> f64,
        padding: f64,
        plot_height: f64,
    ) -> String {
        let mut grid = String::new();
        
        // Get grid color from theme or series
    let (grid_color, grid_opacity, dash_array) = if let Some(ref theme) = self.theme {
        let opacity = match theme.grid_style {
            crate::theme::LineStyle::Solid => "0.6",
            _ => "0.45",
        };
        let dash = match theme.grid_style {
            crate::theme::LineStyle::Solid => "none",
            crate::theme::LineStyle::Dashed => "6,4",
            crate::theme::LineStyle::Dotted => "2,2",
            crate::theme::LineStyle::DashDot => "6,2,2,2",
        };
        (theme.grid_color.to_hex(), opacity, dash)
    } else {
        (
            self.series
                .first()
                .map(|s| s.style.grid_color.to_hex())
                .unwrap_or_else(|| Color::GRAY.to_hex()),
            "0.5",
            "none",
        )
    };

    // Vertical grid lines
    for (t, _) in x_ticks {
        let x = x_px_k(*t);
        grid.push_str(&format!(
            r#" <line x1="{x}" y1="{padding}" x2="{x}" y2="{bottom}" stroke="{grid_color}" stroke-width="0.6" opacity="{grid_opacity}" stroke-dasharray="{dash_array}"/>"#,
            bottom = padding + plot_height,
        ));
        grid.push('\n');
    }

    // Horizontal grid lines at y ticks
    for (t, _) in y_ticks {
        let y = y_px_k(*t);
        grid.push_str(&format!(
            r#" <line x1="{padding}" y1="{y}" x2="{right}" y2="{y}" stroke="{grid_color}" stroke-width="0.6" opacity="{grid_opacity}" stroke-dasharray="{dash_array}"/>"#,
            right = padding + (self.config.width as f64 - 2.0 * padding),
        ));
        grid.push('\n');
    }

        grid
    }

fn generate_axes(&self, padding: f64, width: f64, height: f64) -> String {
    let mut axes = String::new();
    let (axis_color, border_width) = if let Some(ref theme) = self.theme {
        (theme.axis_color.to_hex(), theme.border_width)
    } else {
        (Color::BLACK.to_hex(), 1.0)
    };

    // X-axis
    axes.push_str(&format!(
        r#" <line x1="{padding}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="{axis_color}" stroke-width="{border_width}" stroke-linecap="round"/>"#,
        bottom = padding + height,
        right = padding + width,
    ));
    axes.push('\n');

    // Y-axis
    axes.push_str(&format!(
        r#" <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{bottom}" stroke="{axis_color}" stroke-width="{border_width}" stroke-linecap="round"/>"#,
        bottom = padding + height,
    ));
    axes.push('\n');

    axes
}

    fn generate_ticks(
        &self,
        x_ticks: &[(f64, String)],
        y_ticks: &[(f64, String)],
        x_px_k: &dyn Fn(f64) -> f64,
        y_px_k: &dyn Fn(f64) -> f64,
        padding: f64,
        plot_height: f64,
    ) -> String {
        let mut out = String::new();

        // Get theme settings
        let (axis_color, text_color, font_family, tick_size) = if let Some(ref theme) = self.theme {
            (
                theme.axis_color.to_hex(),
                theme.text_color.to_hex(),
                theme.font_family.clone(),
                theme.tick_size,
            )
        } else {
            (
                Color::BLACK.to_hex(),
                Color::BLACK.to_hex(),
                "Arial, sans-serif".to_string(),
                11.0,
            )
        };

    // X tick marks (pointing outward) and labels below the axis
    for (t, label) in x_ticks {
        let x = x_px_k(*t);
        out.push_str(&format!(
            r#" <line x1="{x}" y1="{bottom}" x2="{x}" y2="{tick_end}" stroke="{axis_color}" stroke-width="1" stroke-linecap="round"/>"#,
            bottom = padding + plot_height,
            tick_end = padding + plot_height + 4.0,
        ));
        out.push('\n');
        out.push_str(&format!(
            r#" <text x="{x}" y="{label_y}" text-anchor="middle" font-size="{tick_size}" font-family="{font_family}" fill="{text_color}">{label}</text>"#,
            label_y = padding + plot_height + tick_size + 6.0,
        ));
        out.push('\n');
    }

    // Y tick marks (pointing outward) and labels left of the axis
    for (t, label) in y_ticks {
        let y = y_px_k(*t);
        out.push_str(&format!(
            r#" <line x1="{tick_start}" y1="{y}" x2="{padding}" y2="{y}" stroke="{axis_color}" stroke-width="1" stroke-linecap="round"/>"#,
            tick_start = padding - 4.0,
        ));
        out.push('\n');
        out.push_str(&format!(
            r#" <text x="{label_x}" y="{y_mid}" text-anchor="end" dominant-baseline="central" font-size="{tick_size}" font-family="{font_family}" fill="{text_color}">{label}</text>"#,
            label_x = padding - 7.0,
            y_mid = y + 1.0,
        ));
        out.push('\n');
    }

        out
    }

    fn generate_bar(
        bar: &BarData,
        x_px: &dyn Fn(f64) -> f64,
        y_px: &dyn Fn(f64) -> f64,
        padding: f64,
        plot_height: f64,
    ) -> String {
        let x = x_px(bar.x_lo);
        let w = x_px(bar.x_hi) - x;
        let y = y_px(bar.y);
        let baseline = padding + plot_height;
        // Add rounded corners (rx=2, ry=2) for a modern look
        format!(
            r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="2" ry="2" stroke="{}" stroke-width="0.5"/>"#,
            x,
            y,
            w,
            (baseline - y).max(0.0),
            bar.color.to_hex(),
            bar.color.to_hex()
        ) + "\n"
    }

    fn generate_series(
        &self,
        series: &DataSeries,
        x_px: &dyn Fn(f64) -> f64,
        y_px: &dyn Fn(f64) -> f64,
    ) -> String {
        let mut output = String::new();
        let style = &series.style;

        // Convert data points to SVG coordinates
        let points: Vec<(f64, f64)> = series
            .points
            .iter()
            .map(|p| (x_px(p.x), y_px(p.y)))
            .collect();

        // Draw gradient area fill if fill_color is set
        if let Some(fill_color) = &style.fill_color {
            if points.len() > 1 {
                let fill_hex = fill_color.to_hex();
                let gradient_id = format!("area-grad-{}", crate::common::xml_escape(&series.name).replace(' ', "-"));
                
                // Create gradient definition
                output.push_str(&format!(
                    r##"  <defs>
    <linearGradient id="{}" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{}" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="{}" stop-opacity="0.1"/>
    </linearGradient>
  </defs>"##,
                    gradient_id, fill_hex, fill_hex
                ));
                output.push('\n');
                
                // Create filled path
                let padding = self.config.padding;
                let plot_height = self.config.height as f64 - 2.0 * padding;
                let baseline_y = padding + plot_height; // Bottom of plot area
                
                let mut path_d = format!("M{:.1},{:.1}", points[0].0, baseline_y);
                // Move to first point
                path_d.push_str(&format!(" L{:.1},{:.1}", points[0].0, points[0].1));
                
                // Add smooth curve through points
                for i in 1..points.len() {
                    let prev = points[i - 1];
                    let curr = points[i];
                    let cp1_x = prev.0 + (curr.0 - prev.0) / 3.0;
                    let cp1_y = prev.1;
                    let cp2_x = prev.0 + 2.0 * (curr.0 - prev.0) / 3.0;
                    let cp2_y = curr.1;
                    path_d.push_str(&format!(
                        " C{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}",
                        cp1_x, cp1_y, cp2_x, cp2_y, curr.0, curr.1
                    ));
                }
                
                // Close path back to baseline
                path_d.push_str(&format!(" L{:.1},{:.1}", points.last().expect("points is non-empty").0, baseline_y));
                path_d.push_str(" Z");
                
                output.push_str(&format!(
                    r#"  <path d="{}" fill="url(#{})"/>"#,
                    path_d, gradient_id
                ));
                output.push('\n');
            }
        }

        // Draw line with anti-aliasing
        if points.len() > 1 {
            let line_color = style.line_color.to_hex();
            let line_width = style.line_width;
            let dash_array = match style.line_style {
                LineStyle::Solid => "none",
                LineStyle::Dashed => "5,5",
                LineStyle::Dotted => "2,2",
                LineStyle::DashDot => "5,2,2,2",
            };

            // Use path with smooth curves for better rendering
            if points.len() > 2 {
                // Generate smooth cubic Bezier path
                let mut path_d = format!("M{:.1},{:.1}", points[0].0, points[0].1);
                for i in 1..points.len() {
                    let prev = points[i - 1];
                    let curr = points[i];
                    // Simple smooth curve: use control points at 1/3 and 2/3
                    let cp1_x = prev.0 + (curr.0 - prev.0) / 3.0;
                    let cp1_y = prev.1;
                    let cp2_x = prev.0 + 2.0 * (curr.0 - prev.0) / 3.0;
                    let cp2_y = curr.1;
                    path_d.push_str(&format!(
                        " C{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}",
                        cp1_x, cp1_y, cp2_x, cp2_y, curr.0, curr.1
                    ));
                }

                output.push_str(&format!(
                    r#"  <path d="{}" fill="none" stroke="{}" stroke-width="{}" stroke-dasharray="{}" stroke-linecap="round" stroke-linejoin="round"/>"#,
                    path_d, line_color, line_width, dash_array
                ));
            } else {
                // Fallback to polyline for 2 points
                output.push_str(&format!(
                    r#"  <polyline points="{}" fill="none" stroke="{}" stroke-width="{}" stroke-dasharray="{}" stroke-linecap="round" stroke-linejoin="round"/>"#,
                    points
                        .iter()
                        .map(|(x, y)| format!("{x},{y}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    line_color,
                    line_width,
                    dash_array
                ));
            }
            output.push('\n');
        }

        // Draw markers with stroke for better visibility
        if style.marker_style != MarkerStyle::None {
            let marker_color = style.marker_color.to_hex();
            let marker_size = style.marker_size;

            for (x, y) in &points {
                let marker = match style.marker_style {
                    MarkerStyle::Circle => {
                        format!(
                            r#"  <circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="1"/>"#,
                            x, y, marker_size, marker_color, marker_color
                        )
                    }
                    MarkerStyle::Square => {
                        format!(
                            r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="1" rx="1" ry="1"/>"#,
                            x - marker_size,
                            y - marker_size,
                            marker_size * 2.0,
                            marker_size * 2.0,
                            marker_color,
                            marker_color
                        )
                    }
                    MarkerStyle::Triangle => {
                        format!(
                            r#"  <polygon points="{},{} {},{} {},{}" fill="{}" stroke="{}" stroke-width="1"/>"#,
                            x,
                            y - marker_size,
                            x - marker_size,
                            y + marker_size,
                            x + marker_size,
                            y + marker_size,
                            marker_color,
                            marker_color
                        )
                    }
                    MarkerStyle::Plus => {
                        format!(
                            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
                            x - marker_size,
                            y,
                            x + marker_size,
                            y,
                            marker_color,
                            x,
                            y - marker_size,
                            x,
                            y + marker_size,
                            marker_color
                        )
                    }
                    MarkerStyle::Cross => {
                        format!(
                            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
                            x - marker_size,
                            y - marker_size,
                            x + marker_size,
                            y + marker_size,
                            marker_color,
                            x + marker_size,
                            y - marker_size,
                            x - marker_size,
                            y + marker_size,
                            marker_color
                        )
                    }
                    MarkerStyle::Diamond => {
                        format!(
                            r#"  <polygon points="{},{} {},{} {},{} {},{}" fill="{}"/>"#,
                            x,
                            y - marker_size,
                            x + marker_size,
                            y,
                            x,
                            y + marker_size,
                            x - marker_size,
                            y,
                            marker_color
                        )
                    }
                    MarkerStyle::None => String::new(),
                };
                output.push_str(&marker);
                output.push('\n');
            }
        }

        output
    }

    fn generate_box(
        i: usize,
        b: &BoxData,
        x_px: &dyn Fn(f64) -> f64,
        y_px: &dyn Fn(f64) -> f64,
    ) -> String {
        let cx = x_px(i as f64);
        let half_w = (x_px(i as f64 + 0.35) - x_px(i as f64 - 0.35)) / 2.0;
        let stroke = b.color.to_hex();
        let s = &b.stats;
        let mut out = String::new();

        // Whisker to both caps
        out.push_str(&format!(
            r#"  <line x1="{cx}" y1="{}" x2="{cx}" y2="{}" stroke="{stroke}" stroke-width="1.5"/>"#,
            y_px(s.min),
            y_px(s.max)
        ));
        out.push('\n');
        for y in [s.min, s.max] {
            out.push_str(&format!(
                r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{stroke}" stroke-width="1.5"/>"#,
                cx - half_w,
                y_px(y),
                cx + half_w,
                y_px(y)
            ));
            out.push('\n');
        }

        // Box between quartiles + median line
        out.push_str(&format!(
            r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="white" stroke="{stroke}" stroke-width="1.5"/>"#,
            cx - half_w,
            y_px(s.q3),
            2.0 * half_w,
            (y_px(s.q1) - y_px(s.q3)).max(0.0)
        ));
        out.push('\n');
        out.push_str(&format!(
            r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{stroke}" stroke-width="2.5"/>"#,
            cx - half_w,
            y_px(s.median),
            cx + half_w,
            y_px(s.median)
        ));
        out.push('\n');

        // Outliers
        for o in &s.outliers {
            out.push_str(&format!(
                r#"  <circle cx="{cx}" cy="{}" r="2.5" fill="{stroke}"/>"#,
                y_px(*o)
            ));
            out.push('\n');
        }

        out
    }

    fn generate_error_bar(
        e: &ErrorBarData,
        x_px: &dyn Fn(f64) -> f64,
        y_px: &dyn Fn(f64) -> f64,
    ) -> String {
        let cx = x_px(e.x);
        let stroke = e.color.to_hex();
        let (lo, hi) = (y_px(e.bar.lo), y_px(e.bar.hi));
        let mut out = format!(
            r#"  <line x1="{cx}" y1="{lo}" x2="{cx}" y2="{hi}" stroke="{stroke}" stroke-width="1.5"/>"#,
        );
        out.push('\n');
        for y in [lo, hi] {
            out.push_str(&format!(
                r#"  <line x1="{}" y1="{y}" x2="{}" y2="{y}" stroke="{stroke}" stroke-width="1.5"/>"#,
                cx - 4.0,
                cx + 4.0
            ));
            out.push('\n');
        }
        out.push_str(&format!(
            r#"  <circle cx="{cx}" cy="{}" r="3" fill="{stroke}"/>"#,
            y_px(e.bar.center)
        ));
        out.push('\n');
        out
    }

    /// Render all annotations (arrows, text, reference lines, rectangles).
    fn generate_annotations(
        &self,
        x_px: &dyn Fn(f64) -> f64,
        y_px: &dyn Fn(f64) -> f64,
    ) -> String {
        let mut out = String::new();

        // Reference lines (horizontal/vertical across entire plot)
        for line in &self.annotations.lines {
            let stroke = line.color.to_hex();
            let width = line.width;
            let dash = line.dash.as_deref().unwrap_or("");
            let dash_attr = if dash.is_empty() {
                String::new()
            } else {
                format!(r#" stroke-dasharray="{dash}""#)
            };
            match line.orientation {
                crate::annotations::LineOrientation::Horizontal => {
                    let y = y_px(line.position);
                    out.push_str(&format!(
                        r##"  <line x1="0" y1="{y}" x2="{}" y2="{y}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>"##,
                        self.config.width
                    ));
                }
                crate::annotations::LineOrientation::Vertical => {
                    let x = x_px(line.position);
                    out.push_str(&format!(
                        r##"  <line x1="{x}" y1="0" x2="{x}" y2="{}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>"##,
                        self.config.height
                    ));
                }
            }
            out.push('\n');
        }

        // Rectangle annotations (origin + size in data coordinates)
        for rect in &self.annotations.rectangles {
            let fill = rect.fill.map(|c| c.to_hex()).unwrap_or_else(|| "none".to_string());
            let stroke = rect.stroke.to_hex();
            let x = x_px(rect.origin.x);
            let y = y_px(rect.origin.y + rect.height);
            let w = x_px(rect.origin.x + rect.width) - x;
            let h = y - y_px(rect.origin.y);
            out.push_str(&format!(
                r##"  <rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{}" opacity="0.3"/>"##,
                rect.stroke_width
            ));
            out.push('\n');
        }

        // Arrow annotations
        for arrow in &self.annotations.arrows {
            let stroke = arrow.color.to_hex();
            let width = arrow.width;
            let x1 = x_px(arrow.from.x);
            let y1 = y_px(arrow.from.y);
            let x2 = x_px(arrow.to.x);
            let y2 = y_px(arrow.to.y);
            let dash = arrow.dash.as_deref().unwrap_or("");
            let dash_attr = if dash.is_empty() {
                String::new()
            } else {
                format!(r#" stroke-dasharray="{dash}""#)
            };
            // Line
            out.push_str(&format!(
                r##"  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>"##
            ));
            out.push('\n');
            // Arrowhead
            let dx = x2 - x1;
            let dy = y2 - y1;
            let len = (dx * dx + dy * dy).sqrt();
            if len > 0.0 {
                let hs = arrow.head_size;
                let ux = dx / len;
                let uy = dy / len;
                let px = -uy;
                let py = ux;
out.push_str(&format!(
                    r##"  <polygon points="{x2},{y2} {} {} {} {}" fill="{stroke}"/>"##,
                    x2 - hs * ux + hs * 0.4 * px,
                    y2 - hs * uy + hs * 0.4 * py,
                    x2 - hs * ux - hs * 0.4 * px,
                    y2 - hs * uy - hs * 0.4 * py,
                ));
                out.push('\n');
            }
        }

        // Text annotations
        for text in &self.annotations.texts {
            let fill = text.color.to_hex();
            let x = x_px(text.position.x) + text.x_offset;
            let y = y_px(text.position.y) + text.y_offset;
            let anchor = match text.anchor {
                crate::annotations::TextAnchor::Start => "start",
                crate::annotations::TextAnchor::Middle => "middle",
                crate::annotations::TextAnchor::End => "end",
            };
            let weight = &text.font_weight;
            if let Some(bg_color) = text.background {
                let bg = bg_color.to_hex();
                out.push_str(&format!(
                    r##"  <rect x="{}" y="{}" width="{}" height="{}" fill="{bg}" rx="3"/>"##,
                    x - 4.0,
                    y - text.font_size,
                    text.text.len() as f64 * text.font_size * 0.6 + 8.0,
                    text.font_size + 8.0
                ));
                out.push('\n');
            }
            out.push_str(&format!(
                r##"  <text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{}" font-weight="{weight}" fill="{fill}">{}</text>"##,
                text.font_size,
                crate::common::xml_escape(&text.text)
            ));
            out.push('\n');
        }

        out
    }

    fn generate_legend(&self) -> String {
        let mut legend_items = Vec::new();

        // Collect series into legend items
        for series in &self.series {
            legend_items.push(crate::legend::LegendItem::new(
                series.name.clone(),
                series.style.line_color,
            ));
        }

        // Collect box plots into legend items
        for b in &self.boxes {
            legend_items.push(crate::legend::LegendItem::new(b.name.clone(), b.color));
        }

        if legend_items.is_empty() {
            return String::new();
        }

        // Use custom config or defaults
        let config = self
            .legend
            .clone()
            .unwrap_or_default();

        // Estimate legend size and compute position
        let (legend_w, legend_h) =
            crate::legend::estimate_legend_size(&legend_items, &config);

        // Adjust canvas width for outside positions
        let _extra_width = matches!(
            config.position,
            crate::legend::LegendPosition::OutsideRight | crate::legend::LegendPosition::OutsideLeft
        )
        .then_some(legend_w + 20.0)
        .unwrap_or(0.0);

        let plot_width = self.config.width as f64;
        let plot_height = self.config.height as f64;
        let padding = 50.0; // default padding

        let (x, y) = crate::legend::legend_position(
            &config,
            plot_width,
            plot_height,
            legend_w,
            legend_h,
            padding,
        );

        crate::legend::render_legend(&legend_items, x, y, &config)
    }
}

impl crate::backend::Backend for SvgPlot {
    fn generate(&self, data: &PlotData) -> PlotResult<PlotOutput> {
        let mut svg = SvgPlot::new(data.config.clone());
        for s in &data.series {
            svg.add_series(s.clone());
        }
        for b in &data.bars {
            svg.bars.push(BarData {
                x_lo: b.x_lo,
                x_hi: b.x_hi,
                y: b.y,
                color: b.color,
            });
        }
        for b in &data.boxes {
            svg.boxes.push(BoxData {
                name: b.name.clone(),
                stats: b.stats.clone(),
                color: b.color,
            });
        }
        for e in &data.error_bars {
            svg.error_bars.push(ErrorBarData {
                x: e.x,
                bar: e.bar,
                color: e.color,
            });
        }
        for hm in &data.heatmaps {
            svg.heatmaps.push(hm.clone());
        }
        Ok(PlotOutput::Svg(svg.generate()))
    }
}

/// Compact numeric label: plain decimal below 1e6, scientific notation above.
fn format_tick(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    if v.abs() >= 1e6 || (v.abs() < 1e-4 && v != 0.0) {
        format!("{v:.1e}")
    } else {
        format!("{v:.3}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataPoint;

    #[test]
    fn test_svg_plot_creation() {
        let config = PlotConfig::new()
            .with_title("Test Plot".to_string())
            .with_dimensions(800, 600);

        let mut plot = SvgPlot::new(config);

        let points = vec![
            DataPoint::new(1.0, 2.0),
            DataPoint::new(2.0, 4.0),
            DataPoint::new(3.0, 6.0),
        ];

        let series = DataSeries::new("Test Series".to_string(), points);
        plot.add_series(series);

        let svg = plot.generate();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Test Plot"));
    }

    #[test]
    fn constant_data_no_nan() {
        let config = PlotConfig::new();
        let mut plot = SvgPlot::new(config);
        plot.add_series(DataSeries::new(
            "flat",
            vec![DataPoint::new(3.0, 7.0), DataPoint::new(3.0, 7.0)],
        ));
        let svg = plot.generate();
        assert!(!svg.contains("NaN"));
    }

    #[test]
    fn tick_labels_rendered() {
        let config = PlotConfig::new().with_tick_count(5);
        let mut plot = SvgPlot::new(config);
        plot.add_series(DataSeries::new(
            "line",
            vec![DataPoint::new(0.0, 0.0), DataPoint::new(10.0, 10.0)],
        ));
        let svg = plot.generate();
        assert!(svg.contains(">0<"));
        assert!(svg.contains(">10<"));
    }

    #[test]
    fn log_scale_renders() {
        let config = PlotConfig::new().with_x_scale(Scale::Log);
        let mut plot = SvgPlot::new(config);
        plot.add_series(DataSeries::new(
            "exp",
            vec![
                DataPoint::new(1.0, 1.0),
                DataPoint::new(10.0, 2.0),
                DataPoint::new(100.0, 3.0),
            ],
        ));
        let svg = plot.generate();
        assert!(!svg.contains("NaN"));
        assert!(svg.contains(">1<") || svg.contains(">10<") || svg.contains(">100<"));
    }

    #[test]
    fn bar_rendered() {
        let config = PlotConfig::new();
        let mut plot = SvgPlot::new(config);
        plot.add_bar(0.0, 1.0, 5.0, Color::BLUE);
        let svg = plot.generate();
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn box_plot_rendered() {
        let config = PlotConfig::new().with_title("Boxes");
        let mut plot = SvgPlot::new(config);
        plot.add_box_plot("A", &[1.0, 2.0, 3.0, 4.0, 5.0, 100.0], Color::BLUE)
            .unwrap();
        let svg = plot.generate();
        assert!(svg.contains("Boxes"));
        assert!(svg.contains("<rect"));
        assert!(!svg.contains("NaN"));
        assert!(svg.contains(">A<"));
    }

    #[test]
    fn cross_marker_has_two_lines() {
        let style = crate::style::PlotStyle::default().with_marker_style(MarkerStyle::Cross);
        let config = PlotConfig::new();
        let mut plot = SvgPlot::new(config);
        plot.add_series(DataSeries::with_style(
            String::from("cross"),
            vec![DataPoint::new(1.0, 2.0)],
            style,
        ));
        let svg = plot.generate();
        assert!(svg.matches("<line").count() >= 2);
    }

    #[test]
    fn format_tick_variants() {
        assert_eq!(format_tick(0.0), "0");
        assert_eq!(format_tick(2.0), "2");
        assert_eq!(format_tick(2.5), "2.5");
        assert_eq!(format_tick(1e7), "1.0e7");
    }
}
