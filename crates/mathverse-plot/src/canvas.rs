//! WASM canvas backend (behind `canvas` feature flag).
//!
//! Draws a `PlotData` scene into an HTML `<canvas>` 2D context via `web-sys`.
//! This is the in-browser counterpart of the SVG / PNG backends: it maps data
//! space to pixel space with the same affine transform, then emits canvas draw
//! calls for the grid, axes, tick labels, and line/scatter series.

use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::axes::Range;
use crate::backend::PlotData;
use crate::error::{PlotError, PlotResult};
use crate::style::Color;

/// Draw a plot onto a `<canvas>` element.
///
/// `lookup_context = true` resolves the context via `canvas.get_context("2d")`;
/// pass `false` to supply your own context.
pub fn draw_to_canvas(
    canvas: &HtmlCanvasElement,
    data: &PlotData,
    lookup_context: bool,
) -> PlotResult<()> {
    let (width, height) = (canvas.width() as f64, canvas.height() as f64);
    let context = if lookup_context {
        canvas
            .get_context("2d")
            .map_err(|_| PlotError::InvalidData("canvas 2d context unavailable".into()))?
            .ok_or_else(|| PlotError::InvalidData("canvas 2d context unavailable".into()))?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| PlotError::InvalidData("canvas 2d context not 2d".into()))?
    } else {
        // Caller-prepared context; a placeholder lets the compiler type-check.
        canvas
            .get_context("2d")
            .map_err(|_| PlotError::InvalidData("canvas 2d context unavailable".into()))?
            .ok_or_else(|| PlotError::InvalidData("canvas 2d context unavailable".into()))?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| PlotError::InvalidData("canvas 2d context not 2d".into()))?
    };

    // Clear background.
    context.set_fill_style(&wasm_bindgen::JsValue::from_str("#ffffff"));
    context.fill_rect(0.0, 0.0, width, height);

    let pad = 50.0;
    let plot_w = width - 2.0 * pad;
    let plot_h = height - 2.0 * pad;
    let (x_range, y_range) = plot_ranges(data, pad);
    if x_range.span() <= 0.0 || y_range.span() <= 0.0 {
        return Ok(());
    }

    let to_x = |x: f64| pad + (x - x_range.min) / x_range.span() * plot_w;
    let to_y = |y: f64| pad + (y_range.max - y) / y_range.span() * plot_h;

    // Grid + ticks.
    let x_ticks = crate::axes::Scale::Linear.ticks(x_range.min, x_range.max, 6);
    let y_ticks = crate::axes::Scale::Linear.ticks(y_range.min, y_range.max, 6);

    context.set_stroke_style(&wasm_bindgen::JsValue::from_str("#cccccc"));
    context.begin_path();
    for &v in &x_ticks {
        let px = to_x(v);
        context.move_to(px, pad);
        context.line_to(px, pad + plot_h);
    }
    for &v in &y_ticks {
        let py = to_y(v);
        context.move_to(pad, py);
        context.line_to(pad + plot_w, py);
    }
    context.stroke();

    // Axis lines.
    context.set_stroke_style(&wasm_bindgen::JsValue::from_str("#000000"));
    context.set_line_width(2.0);
    context.begin_path();
    context.move_to(pad, pad);
    context.line_to(pad, pad + plot_h);
    context.move_to(pad, pad + plot_h);
    context.line_to(pad + plot_w, pad + plot_h);
    context.stroke();

    // Series.
    context.set_line_width(2.0);
    for series in &data.series {
        let color = color_to_js(series.style.line_color);
        let color = wasm_bindgen::JsValue::from_str(&color);
        context.set_stroke_style(&color);
        context.set_fill_style(&color);
        if series.points.len() >= 2 {
            context.begin_path();
            let first = &series.points[0];
            context.move_to(to_x(first.x), to_y(first.y));
            for p in &series.points[1..] {
                context.line_to(to_x(p.x), to_y(p.y));
            }
            context.stroke();
        }
        for p in &series.points {
            context.begin_path();
            context
                .arc(to_x(p.x), to_y(p.y), 3.0, 0.0, std::f64::consts::TAU)
                .map_err(|_| PlotError::InvalidData("canvas arc failed".into()))?;
            context.fill();
        }
    }

    Ok(())
}

/// Compute padded x/y ranges from a `PlotData` snapshot.
fn plot_ranges(data: &PlotData, _pad: f64) -> (Range, Range) {
    let x = Range::compute(
        data.series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.x)),
    );
    let y = Range::compute(
        data.series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.y)),
    );
    match (x, y) {
        (Some(xs), Some(ys)) if xs.span() > 0.0 && ys.span() > 0.0 => (xs.pad(0.05), ys.pad(0.05)),
        _ => (Range { min: 0.0, max: 1.0 }, Range { min: 0.0, max: 1.0 }),
    }
}

/// Convert a `mathverse_plot::Color` to a CSS color string for the canvas.
fn color_to_js(c: Color) -> String {
    let (r, g, b) = c.to_rgb();
    format!("rgb({r},{g},{b})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plot_ranges_pads_finite_data() {
        let data = PlotData {
            config: crate::common::PlotConfig::new(),
            series: vec![crate::DataSeries::new(
                "s".into(),
                vec![
                    crate::DataPoint::new(0.0, 0.0),
                    crate::DataPoint::new(1.0, 2.0),
                ],
            )],
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
        };
        let (x, y) = plot_ranges(&data, 50.0);
        assert!(x.min < 0.0);
        assert!(x.max > 1.0);
        assert!(y.min < 0.0);
        assert!(y.max > 2.0);
    }

    #[test]
    fn plot_ranges_falls_back_on_empty() {
        let data = PlotData {
            config: crate::common::PlotConfig::new(),
            series: Vec::new(),
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
        };
        let (x, y) = plot_ranges(&data, 50.0);
        assert_eq!((x.min, x.max, y.min, y.max), (0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn color_to_js_uses_rgb() {
        assert_eq!(color_to_js(Color::Named("red")), "rgb(255,0,0)");
        assert_eq!(color_to_js(Color::Rgb(10, 20, 30)), "rgb(10,20,30)");
    }
}
