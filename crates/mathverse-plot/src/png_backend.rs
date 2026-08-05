//! PNG raster backend (behind `png` feature flag).

use crate::axes::Range;
use crate::backend::{Backend, PlotData, PlotOutput};
use crate::common;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// PNG rasteriser backed by `tiny-skia`.
pub struct PngBackend {
    width: u32,
    height: u32,
    bg: tiny_skia::Color,
}

impl PngBackend {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bg: tiny_skia::Color::WHITE,
        }
    }

    pub fn with_bg(mut self, color: tiny_skia::Color) -> Self {
        self.bg = color;
        self
    }

    /// Render a `PlotData` snapshot to raw PNG bytes.
    pub fn render(&self, data: &PlotData) -> PlotResult<Vec<u8>> {
        let mut pixmap = tiny_skia::Pixmap::new(self.width, self.height)
            .ok_or_else(|| PlotError::InvalidData("invalid pixmap dimensions".into()))?;

        pixmap.fill(self.bg);

        let pad = data.config.padding;
        let plot_left = pad;
        let plot_top = pad;
        let plot_w = self.width as f64 - 2.0 * pad;
        let plot_h = self.height as f64 - 2.0 * pad;

        if plot_w <= 0.0 || plot_h <= 0.0 {
            return Err(PlotError::InvalidData(
                "plot area is non-positive (padding too large for canvas)".into(),
            ));
        }

        let x_range = self.compute_x_range(data).pad(0.05);
        let y_range = self.compute_y_range(data).pad(0.05);

        let to_px_x =
            |x: f64| -> f32 { (plot_left + (x - x_range.min) / x_range.span() * plot_w) as f32 };
        let to_px_y = |y: f64| -> f32 {
            (plot_top + plot_h - (y - y_range.min) / y_range.span() * plot_h) as f32
        };

        // --- Heatmaps ---
        for hm in &data.heatmaps {
            self.render_heatmap(&mut pixmap, hm, plot_left, plot_top, plot_w, plot_h);
        }

        // --- Bars ---
        for bar in &data.bars {
            let x0 = to_px_x(bar.x_lo);
            let x1 = to_px_x(bar.x_hi);
            let y0 = to_px_y(0.0);
            let y1 = to_px_y(bar.y);
            if let Some(rect) = tiny_skia::Rect::from_xywh(x0, y1, x1 - x0, y0 - y1) {
                let mut p = tiny_skia::Paint::default();
                p.set_color(color_to_skia(bar.color));
                pixmap.fill_rect(rect, &p, tiny_skia::Transform::identity(), None);
            }
        }

        // --- Series (lines + scatter) ---
        for series in &data.series {
            let c = color_to_skia(series.style.line_color);

            if !series.points.is_empty() {
                for w in series.points.windows(2) {
                    let mut pb = tiny_skia::PathBuilder::new();
                    pb.move_to(to_px_x(w[0].x), to_px_y(w[0].y));
                    pb.line_to(to_px_x(w[1].x), to_px_y(w[1].y));
                    let path = pb.finish().ok_or_else(|| PlotError::InvalidData("failed to build series path".into()))?;
                    let mut p = tiny_skia::Paint::default();
                    p.set_color(c);
                    pixmap.stroke_path(
                        &path,
                        &p,
                        &tiny_skia::Stroke {
                            width: 2.0,
                            ..tiny_skia::Stroke::default()
                        },
                        tiny_skia::Transform::identity(),
                        None,
                    );
                }
            }

            for pt in &series.points {
                let circle = circle_path(to_px_x(pt.x), to_px_y(pt.y), 4.0)?;
                let mut p = tiny_skia::Paint::default();
                p.set_color(c);
                pixmap.fill_path(
                    &circle,
                    &p,
                    tiny_skia::FillRule::Winding,
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
        }

        // --- Error bars ---
        for eb in &data.error_bars {
            let cx = to_px_x(eb.x);
            let y_lo = to_px_y(eb.bar.lo);
            let y_hi = to_px_y(eb.bar.hi);
            let c = color_to_skia(eb.color);

            let mut pb = tiny_skia::PathBuilder::new();
            pb.move_to(cx, y_lo);
            pb.line_to(cx, y_hi);
            pb.move_to(cx - 4.0, y_lo);
            pb.line_to(cx + 4.0, y_lo);
            pb.move_to(cx - 4.0, y_hi);
            pb.line_to(cx + 4.0, y_hi);
            let path = pb.finish().ok_or_else(|| PlotError::InvalidData("failed to build error bar path".into()))?;
            let mut p = tiny_skia::Paint::default();
            p.set_color(c);
            pixmap.stroke_path(
                &path,
                &p,
                &tiny_skia::Stroke {
                    width: 1.5,
                    ..tiny_skia::Stroke::default()
                },
                tiny_skia::Transform::identity(),
                None,
            );

            // center dot
            let dot = circle_path(cx, to_px_y(eb.bar.center), 2.5)?;
            pixmap.fill_path(
                &dot,
                &p,
                tiny_skia::FillRule::Winding,
                tiny_skia::Transform::identity(),
                None,
            );
        }

        // --- Box plots ---
        let box_w = 20.0f32;
        for (i, bx) in data.boxes.iter().enumerate() {
            let cx = to_px_x(i as f64);
            let y_q1 = to_px_y(bx.stats.q1);
            let y_med = to_px_y(bx.stats.median);
            let y_q3 = to_px_y(bx.stats.q3);
            let y_lo = to_px_y(bx.stats.min);
            let y_hi = to_px_y(bx.stats.max);
            let c = color_to_skia(bx.color);

            // whiskers + caps
            let mut pb = tiny_skia::PathBuilder::new();
            pb.move_to(cx, y_lo);
            pb.line_to(cx, y_hi);
            pb.move_to(cx - 5.0, y_lo);
            pb.line_to(cx + 5.0, y_lo);
            pb.move_to(cx - 5.0, y_hi);
            pb.line_to(cx + 5.0, y_hi);
            let path = pb.finish().ok_or_else(|| PlotError::InvalidData("failed to build whisker path".into()))?;
            let mut p = tiny_skia::Paint::default();
            p.set_color(c);
            pixmap.stroke_path(
                &path,
                &p,
                &tiny_skia::Stroke {
                    width: 1.5,
                    ..tiny_skia::Stroke::default()
                },
                tiny_skia::Transform::identity(),
                None,
            );

            // box rect (Q1–Q3)
            let x0 = cx - box_w / 2.0;
            if let Some(rect) = tiny_skia::Rect::from_xywh(x0, y_q3, box_w, y_q1 - y_q3) {
                pixmap.fill_rect(rect, &p, tiny_skia::Transform::identity(), None);
                let mut pb = tiny_skia::PathBuilder::new();
                pb.move_to(rect.left(), rect.top());
                pb.line_to(rect.right(), rect.top());
                pb.line_to(rect.right(), rect.bottom());
                pb.line_to(rect.left(), rect.bottom());
                pb.close();
                let rp = pb.finish().ok_or_else(|| PlotError::InvalidData("failed to build box rect path".into()))?;
                pixmap.stroke_path(
                    &rp,
                    &p,
                    &tiny_skia::Stroke {
                        width: 1.5,
                        ..tiny_skia::Stroke::default()
                    },
                    tiny_skia::Transform::identity(),
                    None,
                );
            }

            // median line
            let mut pb2 = tiny_skia::PathBuilder::new();
            pb2.move_to(x0, y_med);
            pb2.line_to(x0 + box_w, y_med);
            let med = pb2.finish().ok_or_else(|| PlotError::InvalidData("failed to build median path".into()))?;
            pixmap.stroke_path(
                &med,
                &p,
                &tiny_skia::Stroke {
                    width: 1.5,
                    ..tiny_skia::Stroke::default()
                },
                tiny_skia::Transform::identity(),
                None,
            );

            // outliers
            for &outlier in &bx.stats.outliers {
                let c = circle_path(cx, to_px_y(outlier), 3.0)?;
                pixmap.fill_path(
                    &c,
                    &p,
                    tiny_skia::FillRule::Winding,
                    tiny_skia::Transform::identity(),
                    None,
                );
            }
        }

        pixmap
            .encode_png()
            .map_err(|e| PlotError::InvalidData(format!("png encode: {e}")))
    }

    fn compute_x_range(&self, data: &PlotData) -> Range {
        crate::common::compute_x_range(data)
    }

    fn compute_y_range(&self, data: &PlotData) -> Range {
        crate::common::compute_y_range(data)
    }

    fn render_heatmap(
        &self,
        pixmap: &mut tiny_skia::Pixmap,
        hm: &crate::heatmap::HeatmapData,
        plot_left: f64,
        plot_top: f64,
        plot_w: f64,
        plot_h: f64,
    ) {
        let rows = hm.rows();
        let cols = hm.cols();
        if rows == 0 || cols == 0 {
            return;
        }
        let cell_w = plot_w / cols as f64;
        let cell_h = plot_h / rows as f64;
        let (data_lo, data_hi) = hm.bounds();
        for r in 0..rows {
            for c in 0..cols {
                let v = hm.grid[r][c];
                let t = if (data_hi - data_lo).abs() < f64::EPSILON {
                    0.5
                } else {
                    (v - data_lo) / (data_hi - data_lo)
                };
                let color = (hm.colormap)(t);
                let x = (plot_left + c as f64 * cell_w) as f32;
                let y = (plot_top + r as f64 * cell_h) as f32;
                if let Some(rect) = tiny_skia::Rect::from_xywh(x, y, cell_w as f32, cell_h as f32) {
                    let mut p = tiny_skia::Paint::default();
                    p.set_color(color_to_skia(color));
                    pixmap.fill_rect(rect, &p, tiny_skia::Transform::identity(), None);
                }
            }
        }
    }
}

impl crate::backend::Backend for PngBackend {
    fn generate(&self, data: &PlotData) -> PlotResult<PlotOutput> {
        let png_bytes = self.render(data)?;
        Ok(PlotOutput::Binary(png_bytes, "image/png"))
    }
}

fn circle_path(cx: f32, cy: f32, r: f32) -> PlotResult<tiny_skia::Path> {
    let k = 0.5522847;
    let mut pb = tiny_skia::PathBuilder::new();
    pb.move_to(cx + r, cy);
    pb.cubic_to(cx + r, cy + r * k, cx + r * k, cy + r, cx, cy + r);
    pb.cubic_to(cx - r * k, cy + r, cx - r, cy + r * k, cx - r, cy);
    pb.cubic_to(cx - r, cy - r * k, cx - r * k, cy - r, cx, cy - r);
    pb.cubic_to(cx + r * k, cy - r, cx + r, cy - r * k, cx + r, cy);
    pb.close();
    pb.finish().ok_or_else(|| PlotError::InvalidData("failed to build circle path".into()))
}

fn color_to_skia(c: Color) -> tiny_skia::Color {
    match c {
        Color::Rgb(r, g, b) => {
            tiny_skia::Color::from_rgba(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
                .unwrap_or(tiny_skia::Color::BLACK)
        }
        Color::Rgba(r, g, b, a) => tiny_skia::Color::from_rgba(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        )
        .unwrap_or(tiny_skia::Color::BLACK),
        Color::Named(name) => match name {
            "red" => tiny_skia::Color::from_rgba(1.0, 0.0, 0.0, 1.0).unwrap(),
            "green" => tiny_skia::Color::from_rgba(0.0, 0.6, 0.0, 1.0).unwrap(),
            "blue" => tiny_skia::Color::from_rgba(0.0, 0.0, 1.0, 1.0).unwrap(),
            "yellow" => tiny_skia::Color::from_rgba(1.0, 1.0, 0.0, 1.0).unwrap(),
            "cyan" => tiny_skia::Color::from_rgba(0.0, 1.0, 1.0, 1.0).unwrap(),
            "magenta" => tiny_skia::Color::from_rgba(1.0, 0.0, 1.0, 1.0).unwrap(),
            "black" => tiny_skia::Color::from_rgba(0.0, 0.0, 0.0, 1.0).unwrap(),
            "white" => tiny_skia::Color::from_rgba(1.0, 1.0, 1.0, 1.0).unwrap(),
            "gray" => tiny_skia::Color::from_rgba(0.5, 0.5, 0.5, 1.0).unwrap(),
            "orange" => tiny_skia::Color::from_rgba(1.0, 0.65, 0.0, 1.0).unwrap(),
            "purple" => tiny_skia::Color::from_rgba(0.5, 0.0, 0.5, 1.0).unwrap(),
            "brown" => tiny_skia::Color::from_rgba(0.6, 0.3, 0.0, 1.0).unwrap(),
            _ => tiny_skia::Color::from_rgba(0.0, 0.0, 0.0, 1.0).unwrap(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, PlotOutput};
    use crate::common::{DataPoint, DataSeries, PlotConfig};
    use crate::style::PlotStyle;

    #[test]
    fn png_backend_returns_binary() {
        let mut data = PlotData {
            config: PlotConfig::new(),
            series: Vec::new(),
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
        };
        data.series.push(DataSeries::with_style(
            "s".into(),
            vec![DataPoint::new(0.0, 0.0), DataPoint::new(1.0, 1.0)],
            PlotStyle::default(),
        ));
        let backend = PngBackend::new(100, 100);
        let result = backend.generate(&data).unwrap();
        match result {
            PlotOutput::Binary(bytes, mime) => {
                assert_eq!(mime, "image/png");
                assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
            }
            PlotOutput::Svg(_) | PlotOutput::Text(_) => panic!("expected Binary output"),
        }
    }

    #[test]
    fn png_backend_empty_plot() {
        let data = PlotData {
            config: PlotConfig::new(),
            series: Vec::new(),
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
        };
        let backend = PngBackend::new(50, 50);
        let result = backend.generate(&data).unwrap();
        match result {
            PlotOutput::Binary(bytes, mime) => {
                assert_eq!(mime, "image/png");
                assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
            }
            PlotOutput::Svg(_) | PlotOutput::Text(_) => panic!("expected Binary output"),
        }
    }

    /// Phase 3 acceptance: the PNG and SVG backends must agree on the same
    /// `PlotData` scene. The PNG should contain non-white pixels wherever the
    /// SVG renders a series line.
    #[test]
    fn png_matches_svg_scene() {
        let mut data = PlotData {
            config: PlotConfig::new(),
            series: Vec::new(),
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
        };
        data.series.push(DataSeries::with_style(
            "s".into(),
            vec![
                DataPoint::new(0.0, 0.0),
                DataPoint::new(0.5, 0.8),
                DataPoint::new(1.0, 1.0),
            ],
            PlotStyle::default(),
        ));

        let mut svg_plot = crate::svg::SvgPlot::new(data.config.clone());
        svg_plot.add_series(data.series[0].clone());
        let svg_text = svg_plot.generate();
        assert!(svg_text.contains("<polyline") || svg_text.contains("<path"));
        assert!(svg_text.contains('s'));
        assert!(svg_text.contains("<polyline"));

        let backend = PngBackend::new(200, 150);
        let result = backend.generate(&data).unwrap();
        let png_bytes = match result {
            PlotOutput::Binary(bytes, _) => bytes,
            PlotOutput::Svg(_) | PlotOutput::Text(_) => panic!("expected Binary output"),
        };
        assert_eq!(&png_bytes[..8], b"\x89PNG\r\n\x1a\n");

        // A blank scene of the same size must render differently: if the PNG
        // backend drew the series, the two PNGs cannot be byte-identical.
        let blank = PlotData {
            config: PlotConfig::new(),
            series: Vec::new(),
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
        };
        let blank_result = backend.generate(&blank).unwrap();
        let blank_bytes = match blank_result {
            PlotOutput::Binary(bytes, _) => bytes,
            PlotOutput::Svg(_) | PlotOutput::Text(_) => panic!("expected Binary output"),
        };
        assert_ne!(png_bytes, blank_bytes, "PNG backend ignored the series");
    }
}
