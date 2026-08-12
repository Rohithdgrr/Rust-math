//! Interactive egui/eframe backend (behind `interactive` feature flag).
//!
//! Renders a `PlotData` snapshot into an `egui` window: axes, nice ticks, grid,
//! and line/scatter series with mouse zooming/panning. This mirrors the SVG
//! scene geometry so the same data can be explored interactively.

use eframe::egui;
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};

use crate::axes::{Range, Scale};
use crate::backend::PlotData;
use crate::common::plot_bounds;
use crate::style::Color;

/// Interactive plot application state.
pub struct InteractivePlot {
    data: PlotData,
    x_range: Range,
    y_range: Range,
    /// Number of gridlines to aim for along each axis.
    tick_count: usize,
    /// Cached canvas size so resizing re-renders correctly.
    last_size: Vec2,
}

impl InteractivePlot {
    /// Build an interactive app around a `PlotData` snapshot.
    pub fn new(data: PlotData) -> Self {
        let (x_range, y_range) = plot_bounds(&data.series);
        Self {
            data,
            x_range,
            y_range,
            tick_count: 6,
            last_size: Vec2::ZERO,
        }
    }

    fn draw(&mut self, ui: &mut egui::Ui) {
        let size = ui.available_size();
        let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());

        if size != self.last_size {
            // Reset any zoom that assumed the previous canvas size.
            self.last_size = size;
        }

        // Pan with a drag, zoom with the scroll wheel around the pointer.
        let drag = response.drag_delta();
        let x_scale = self.x_range.span() / f64::from(size.x);
        let y_scale = self.y_range.span() / f64::from(size.y);
        if response.dragged() {
            self.x_range.min -= f64::from(drag.x) * x_scale;
            self.x_range.max -= f64::from(drag.x) * x_scale;
            self.y_range.min += f64::from(drag.y) * y_scale;
            self.y_range.max += f64::from(drag.y) * y_scale;
        }
        if let Some(hover) = response.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let factor = (f64::from(scroll) * 0.002).exp();
                self.zoom(factor, hover, size);
            }
        }

        let rect = response.rect;
        painter.rect_filled(rect, 0.0, Color32::WHITE);

        let pad = 8.0;
        let plot = Rect::from_min_max(
            Pos2::new(rect.min.x + pad, rect.min.y + pad),
            Pos2::new(rect.max.x - pad, rect.max.y - pad),
        );

        // Grid + ticks + axis lines.
        self.draw_axes(&painter, plot);

        // Series.
        let to_screen = |x: f64, y: f64| -> Pos2 {
            Pos2::new(
                plot.min.x
                    + ((x - self.x_range.min) / self.x_range.span() * f64::from(plot.width()))
                        as f32,
                plot.max.y
                    - ((y - self.y_range.min) / self.y_range.span() * f64::from(plot.height()))
                        as f32,
            )
        };

        for series in &self.data.series {
            let color = color_to_egui(series.style.line_color);
            if series.points.len() >= 2 {
                for w in series.points.windows(2) {
                    let a = to_screen(w[0].x, w[0].y);
                    let b = to_screen(w[1].x, w[1].y);
                    painter.line_segment([a, b], Stroke::new(2.0_f32, color));
                }
            }
            for p in &series.points {
                let c = to_screen(p.x, p.y);
                painter.circle_filled(c, 3.0_f32, color);
            }
        }

        // Title.
        if !self.data.config.title.is_empty() {
            painter.text(
                rect.center_top() + Vec2::new(0.0, 12.0),
                Align2::CENTER_TOP,
                &self.data.config.title,
                FontId::proportional(18.0),
                Color32::BLACK,
            );
        }
        // Range labels.
        let range_label = format!(
            "x: [{:.3}, {:.3}]  y: [{:.3}, {:.3}]",
            self.x_range.min, self.x_range.max, self.y_range.min, self.y_range.max
        );
        painter.text(
            rect.left_bottom() + Vec2::new(4.0, -4.0),
            Align2::LEFT_BOTTOM,
            range_label,
            FontId::monospace(11.0),
            Color32::GRAY,
        );
        ui.ctx().request_repaint();
    }

    /// Zoom around `anchor` by `factor` (axis-aligned).
    fn zoom(&mut self, factor: f64, anchor: Pos2, size: Vec2) {
        let fx = f64::from((anchor.x - size.x * 0.5) / size.x);
        let fy = f64::from((anchor.y - size.y * 0.5) / size.y);
        self.x_range.zoom_at(factor, fx);
        self.y_range.zoom_at(factor, fy);
    }

    fn draw_axes(&self, painter: &egui::Painter, plot: Rect) {
        let x_ticks = Scale::Linear.ticks(self.x_range.min, self.x_range.max, self.tick_count);
        let y_ticks = Scale::Linear.ticks(self.y_range.min, self.y_range.max, self.tick_count);

        let to_x = |x: f64| -> f32 {
            plot.min.x
                + ((x - self.x_range.min) / self.x_range.span() * f64::from(plot.width())) as f32
        };
        let to_y = |y: f64| -> f32 {
            plot.max.y
                - ((y - self.y_range.min) / self.y_range.span() * f64::from(plot.height())) as f32
        };

        // Grid + tick labels.
        for v in x_ticks {
            let x = to_x(v);
            painter.line_segment(
                [Pos2::new(x, plot.min.y), Pos2::new(x, plot.max.y)],
                Stroke::new(1.0_f32, Color32::from_gray(220)),
            );
        }
        for v in y_ticks {
            let y = to_y(v);
            painter.line_segment(
                [Pos2::new(plot.min.x, y), Pos2::new(plot.max.x, y)],
                Stroke::new(1.0_f32, Color32::from_gray(220)),
            );
            painter.text(
                Pos2::new(plot.min.x - 4.0, y),
                Align2::RIGHT_CENTER,
                format_tick(v),
                FontId::proportional(11.0),
                Color32::DARK_GRAY,
            );
        }

        // Axis lines.
        painter.line_segment(
            [
                Pos2::new(plot.min.x, plot.min.y),
                Pos2::new(plot.min.x, plot.max.y),
            ],
            Stroke::new(2.0_f32, Color32::BLACK),
        );
        painter.line_segment(
            [
                Pos2::new(plot.min.x, plot.max.y),
                Pos2::new(plot.max.x, plot.max.y),
            ],
            Stroke::new(2.0_f32, Color32::BLACK),
        );
    }
}

impl eframe::App for InteractivePlot {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Toolbar:");
                if ui.button("⤢ Zoom in").clicked() {
                    self.zoom_toolbar(0.8);
                }
                if ui.button("⤡ Zoom out").clicked() {
                    self.zoom_toolbar(1.25);
                }
                if ui.button("⌂ Home").clicked() {
                    self.reset_view();
                }
                if ui.button("↔ Reset").clicked() {
                    self.reset_view();
                }
                if ui.button("💾 Save PNG").clicked() {
                    #[cfg(feature = "png")]
                    self.save_png_dialog();
                }
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.draw(ui);
        });
    }
}

impl InteractivePlot {
    /// Zoom the view around the canvas center by `factor` (<1 = in).
    fn zoom_toolbar(&mut self, factor: f64) {
        self.x_range.zoom_at(factor, 0.0);
        self.y_range.zoom_at(factor, 0.0);
    }

    /// Restore the full-data view (matplotlib's Home toolbar button).
    fn reset_view(&mut self) {
        let (xr, yr) = plot_bounds(&self.data.series);
        self.x_range = xr;
        self.y_range = yr;
    }

    /// Save the current view to PNG. Writes `mathverse_plot_interactive.png`
    /// in the working directory (matches matplotlib's toolbar Save button
    /// without a native file dialog).
    #[cfg(feature = "png")]
    fn save_png_dialog(&mut self) {
        let svg = {
            let mut plot = crate::svg::SvgPlot::new(self.data.config.clone());
            for s in &self.data.series {
                plot.add_series(s.clone());
            }
            plot.generate()
        };
        let saver = crate::save::PlotSaver::new(svg)
            .with_dimensions(self.data.config.width, self.data.config.height);
        let result = saver.save_as("mathverse_plot_interactive", crate::save::OutputFormat::Png, &crate::save::FormatSet::png());
        if result.success {
            eprintln!("saved PNG -> {:?}", result.path);
        } else {
            eprintln!("save failed: {:?}", result.error);
        }
    }
}

/// Convert a `mathverse_plot::Color` to an `egui::Color32`.
fn color_to_egui(c: Color) -> Color32 {
    let (r, g, b) = c.to_rgb();
    Color32::from_rgb(r, g, b)
}

/// Run the interactive window. `title` is the window title; the plot title
/// comes from the `PlotData` config. Pan with drag, zoom with the scroll wheel.
pub fn run(data: PlotData, title: &str, width: f32, height: f32) -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([width, height])
            .with_title(title),
        ..Default::default()
    };
    eframe::run_native(
        title,
        options,
        Box::new(move |_cc| Ok(Box::new(InteractivePlot::new(data)))),
    )
}

// --- Range extension helpers used by the interactive app ---

trait RangeExt {
    fn zoom_at(&mut self, factor: f64, anchor: f64);
}

impl RangeExt for Range {
    fn zoom_at(&mut self, factor: f64, anchor: f64) {
        let lo = self.min - anchor * self.span();
        let hi = self.max - anchor * self.span();
        let half = (hi - lo) * factor * 0.5;
        let mid = (lo + hi) * 0.5;
        self.min = mid - half;
        self.max = mid + half;
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
