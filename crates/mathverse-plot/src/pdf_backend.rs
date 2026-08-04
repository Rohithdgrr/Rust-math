//! PDF vector backend (behind `pdf` feature flag).

use crate::axes::Range;
use crate::backend::PlotData;
use crate::error::PlotResult;
use crate::style::Color;

/// PDF vector backend backed by `printpdf`.
pub struct PdfBackend {
    width_mm: f32,
    height_mm: f32,
}

impl PdfBackend {
    /// Create a new PDF backend. Dimensions are in millimeters.
    pub fn new(width_mm: f32, height_mm: f32) -> Self {
        Self {
            width_mm,
            height_mm,
        }
    }

    /// Render `PlotData` to raw PDF bytes.
    #[allow(clippy::too_many_lines)]
    pub fn render(&self, data: &PlotData) -> PlotResult<Vec<u8>> {
        use printpdf::{
            Line, LinePoint, Mm, Op, PdfDocument, PdfPage, PdfSaveOptions, Point, Pt, Rect,
        };

        let pad = (data.config.padding as f32) * 0.352_778; // px → mm at 96dpi
        let plot_w = self.width_mm - 2.0 * pad;
        let plot_h = self.height_mm - 2.0 * pad;

        let x_range = self.compute_x_range(data).pad(0.05);
        let y_range = self.compute_y_range(data).pad(0.05);

        let to_mm_x = |x: f64| -> Mm {
            Mm(pad + ((x as f32 - x_range.min as f32) / (x_range.span() as f32)) * plot_w)
        };
        let to_mm_y = |y: f64| -> Mm {
            Mm(pad + ((y as f32 - y_range.min as f32) / (y_range.span() as f32)) * plot_h)
        };

        let mut ops: Vec<Op> = Vec::new();

        // --- Heatmaps ---
        for hm in &data.heatmaps {
            self.append_heatmap(&mut ops, hm, pad, plot_w, plot_h, &x_range, &y_range);
        }

        // --- Bars ---
        for bar in &data.bars {
            ops.push(Op::SetFillColor {
                col: color_to_printpdf(bar.color),
            });
            let x0 = to_mm_x(bar.x_lo);
            let x1 = to_mm_x(bar.x_hi);
            let y0 = to_mm_y(0.0);
            let y1 = to_mm_y(bar.y);
            let rect = Rect::from_xywh(x0.into(), y1.into(), (x1 - x0).into(), (y0 - y1).into());
            ops.push(Op::DrawRectangle { rectangle: rect });
        }

        // --- Series (lines + scatter) ---
        for series in &data.series {
            let c = color_to_printpdf(series.style.line_color);

            // Line segments
            if series.points.len() >= 2 {
                ops.push(Op::SetOutlineColor { col: c.clone() });
                ops.push(Op::SetOutlineThickness { pt: Pt(1.5) });
                let line = Line {
                    points: series
                        .points
                        .iter()
                        .map(|p| LinePoint {
                            p: Point::new(to_mm_x(p.x), to_mm_y(p.y)),
                            bezier: false,
                        })
                        .collect(),
                    is_closed: false,
                };
                ops.push(Op::DrawLine { line });
            }

            // Scatter dots (approximate circles as polygons)
            ops.push(Op::SetFillColor { col: c.clone() });
            for pt in &series.points {
                let circle = circle_polygon(to_mm_x(pt.x), to_mm_y(pt.y), Mm(1.0));
                ops.push(Op::DrawPolygon { polygon: circle });
            }
        }

        // --- Error bars ---
        for eb in &data.error_bars {
            let c = color_to_printpdf(eb.color);
            ops.push(Op::SetOutlineColor { col: c.clone() });
            ops.push(Op::SetOutlineThickness { pt: Pt(1.0) });
            let cx = to_mm_x(eb.x);
            let y_lo = to_mm_y(eb.bar.lo);
            let y_hi = to_mm_y(eb.bar.hi);
            let cap = Mm(1.5);

            // vertical whisker
            ops.push(Op::DrawLine {
                line: Line {
                    points: vec![
                        LinePoint {
                            p: Point::new(cx, y_lo),
                            bezier: false,
                        },
                        LinePoint {
                            p: Point::new(cx, y_hi),
                            bezier: false,
                        },
                    ],
                    is_closed: false,
                },
            });
            // caps
            for y in [y_lo, y_hi] {
                ops.push(Op::DrawLine {
                    line: Line {
                        points: vec![
                            LinePoint {
                                p: Point::new(cx - cap, y),
                                bezier: false,
                            },
                            LinePoint {
                                p: Point::new(cx + cap, y),
                                bezier: false,
                            },
                        ],
                        is_closed: false,
                    },
                });
            }
            // center dot
            ops.push(Op::SetFillColor { col: c.clone() });
            ops.push(Op::DrawPolygon {
                polygon: circle_polygon(cx, to_mm_y(eb.bar.center), Mm(0.8)),
            });
        }

        // --- Box plots ---
        let box_w = Mm(5.0);
        for (i, bx) in data.boxes.iter().enumerate() {
            let c = color_to_printpdf(bx.color);
            ops.push(Op::SetOutlineColor { col: c.clone() });
            ops.push(Op::SetFillColor { col: c.clone() });
            ops.push(Op::SetOutlineThickness { pt: Pt(1.0) });
            let cx = to_mm_x(i as f64);
            let half = box_w / 2.0;
            let y_q1 = to_mm_y(bx.stats.q1);
            let y_q3 = to_mm_y(bx.stats.q3);
            let y_med = to_mm_y(bx.stats.median);
            let y_lo = to_mm_y(bx.stats.min);
            let y_hi = to_mm_y(bx.stats.max);

            // whiskers
            ops.push(Op::DrawLine {
                line: Line {
                    points: vec![
                        LinePoint {
                            p: Point::new(cx, y_lo),
                            bezier: false,
                        },
                        LinePoint {
                            p: Point::new(cx, y_hi),
                            bezier: false,
                        },
                    ],
                    is_closed: false,
                },
            });
            // caps
            for y in [y_lo, y_hi] {
                ops.push(Op::DrawLine {
                    line: Line {
                        points: vec![
                            LinePoint {
                                p: Point::new(cx - Mm(1.5), y),
                                bezier: false,
                            },
                            LinePoint {
                                p: Point::new(cx + Mm(1.5), y),
                                bezier: false,
                            },
                        ],
                        is_closed: false,
                    },
                });
            }
            // box rect
            ops.push(Op::DrawRectangle {
                rectangle: Rect::from_xywh(
                    (cx - half).into(),
                    y_q3.into(),
                    box_w.into(),
                    (y_q1 - y_q3).into(),
                ),
            });
            // median line
            ops.push(Op::DrawLine {
                line: Line {
                    points: vec![
                        LinePoint {
                            p: Point::new(cx - half, y_med),
                            bezier: false,
                        },
                        LinePoint {
                            p: Point::new(cx + half, y_med),
                            bezier: false,
                        },
                    ],
                    is_closed: false,
                },
            });
            // outliers
            for &outlier in &bx.stats.outliers {
                ops.push(Op::DrawPolygon {
                    polygon: circle_polygon(cx, to_mm_y(outlier), Mm(1.0)),
                });
            }
        }

        let page = PdfPage::new(Mm(self.width_mm), Mm(self.height_mm), ops);
        let mut doc = PdfDocument::new(&data.config.title);
        doc.pages.push(page);
        let mut warnings = Vec::new();
        Ok(doc.save(&PdfSaveOptions::default(), &mut warnings))
    }

    #[allow(clippy::unused_self)]
    fn compute_x_range(&self, data: &PlotData) -> Range {
        data.series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.x))
            .chain(data.boxes.iter().enumerate().map(|(i, _)| i as f64))
            .chain(data.error_bars.iter().map(|e| e.x))
            .fold(None::<(f64, f64)>, |acc, x| match acc {
                None => Some((x, x)),
                Some((lo, hi)) => Some((lo.min(x), hi.max(x))),
            })
            .map_or(Range { min: 0.0, max: 1.0 }, |(lo, hi)| Range {
                min: lo,
                max: hi,
            })
    }

    #[allow(clippy::unused_self)]
    fn compute_y_range(&self, data: &PlotData) -> Range {
        data.series
            .iter()
            .flat_map(|s| s.points.iter().map(|p| p.y))
            .chain(data.bars.iter().map(|b| b.y))
            .chain(data.error_bars.iter().flat_map(|e| [e.bar.lo, e.bar.hi]))
            .chain(
                data.boxes
                    .iter()
                    .flat_map(|bx| [bx.stats.q1, bx.stats.q3, bx.stats.min, bx.stats.max]),
            )
            .fold(None::<(f64, f64)>, |acc, y| match acc {
                None => Some((y, y)),
                Some((lo, hi)) => Some((lo.min(y), hi.max(y))),
            })
            .map_or(Range { min: 0.0, max: 1.0 }, |(lo, hi)| Range {
                min: lo,
                max: hi,
            })
    }

    #[allow(clippy::unused_self)]
    fn append_heatmap(
        &self,
        ops: &mut Vec<printpdf::Op>,
        hm: &crate::heatmap::HeatmapData,
        pad: f32,
        plot_w: f32,
        plot_h: f32,
        _x_range: &crate::axes::Range,
        _y_range: &crate::axes::Range,
    ) {
        let rows = hm.rows();
        let cols = hm.cols();
        if rows == 0 || cols == 0 {
            return;
        }
        let cell_w = plot_w / cols as f32;
        let cell_h = plot_h / rows as f32;
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
                ops.push(printpdf::Op::SetFillColor {
                    col: color_to_printpdf(color),
                });
                let x = printpdf::Mm(pad + c as f32 * cell_w);
                let y = printpdf::Mm(pad + r as f32 * cell_h);
                ops.push(printpdf::Op::DrawRectangle {
                    rectangle: printpdf::Rect::from_xywh(
                        x.into(),
                        y.into(),
                        printpdf::Mm(cell_w).into(),
                        printpdf::Mm(cell_h).into(),
                    ),
                });
            }
        }
    }
}

impl crate::backend::Backend for PdfBackend {
    fn generate(&self, data: &PlotData) -> PlotResult<String> {
        let bytes = self.render(data)?;
        Ok(format!("application/pdf;base64,{}", base64_encode(&bytes)))
    }
}

/// Approximate a circle as a `printpdf::Polygon` with 32 segments.
fn circle_polygon(cx: printpdf::Mm, cy: printpdf::Mm, r: printpdf::Mm) -> printpdf::Polygon {
    use printpdf::{LinePoint, Mm, PaintMode, Point, Polygon, PolygonRing, WindingOrder};
    let n = 32;
    let points: Vec<LinePoint> = (0..n)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * f64::from(i) / f64::from(n);
            LinePoint {
                p: Point::new(
                    Mm(cx.0 + r.0 * angle.cos() as f32),
                    Mm(cy.0 + r.0 * angle.sin() as f32),
                ),
                bezier: false,
            }
        })
        .collect();
    Polygon {
        rings: vec![PolygonRing { points }],
        mode: PaintMode::Fill,
        winding_order: WindingOrder::NonZero,
    }
}

fn color_to_printpdf(c: Color) -> printpdf::Color {
    match c {
        Color::Rgb(r, g, b) | Color::Rgba(r, g, b, _) => printpdf::Color::Rgb(printpdf::Rgb::new(
            f32::from(r) / 255.0,
            f32::from(g) / 255.0,
            f32::from(b) / 255.0,
            None,
        )),
        Color::Named(name) => {
            let (r, g, b) = match name {
                "red" => (1.0, 0.0, 0.0),
                "green" => (0.0, 0.6, 0.0),
                "blue" => (0.0, 0.0, 1.0),
                "yellow" => (1.0, 1.0, 0.0),
                "cyan" => (0.0, 1.0, 1.0),
                "magenta" => (1.0, 0.0, 1.0),
                "white" => (1.0, 1.0, 1.0),
                "gray" => (0.5, 0.5, 0.5),
                "orange" => (1.0, 0.65, 0.0),
                "purple" => (0.5, 0.0, 0.5),
                "brown" => (0.6, 0.3, 0.0),
                _ => (0.0, 0.0, 0.0),
            };
            printpdf::Color::Rgb(printpdf::Rgb::new(r, g, b, None))
        }
    }
}

fn base64_encode(data: &[u8]) -> String {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = if chunk.len() > 1 {
            u32::from(chunk[1])
        } else {
            0
        };
        let b2 = if chunk.len() > 2 {
            u32::from(chunk[2])
        } else {
            0
        };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        for i in (0..4).rev() {
            let idx = ((triple >> (i * 6)) & 0x3F) as usize;
            if i == 1 && chunk.len() == 2 {
                out.push('=');
            } else if i == 0 && chunk.len() == 1 {
                out.push_str("==");
            } else {
                out.push(alphabet[idx] as char);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::common::{DataPoint, DataSeries, PlotConfig};
    use crate::style::PlotStyle;

    #[test]
    fn pdf_backend_returns_data_uri() {
        let mut data = PlotData {
            config: PlotConfig::new().with_title("PDF Test"),
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
        let backend = PdfBackend::new(200.0, 150.0);
        let uri = backend.generate(&data).unwrap();
        assert!(uri.starts_with("application/pdf;base64,"));
        let b64 = &uri["application/pdf;base64,".len()..];
        let decoded = base64_decode(b64);
        assert_eq!(&decoded[..5], b"%PDF-");
    }

    #[test]
    fn pdf_backend_empty_plot() {
        let data = PlotData {
            config: PlotConfig::new(),
            series: Vec::new(),
            bars: Vec::new(),
            boxes: Vec::new(),
            error_bars: Vec::new(),
            heatmaps: Vec::new(),
        };
        let backend = PdfBackend::new(100.0, 100.0);
        let uri = backend.generate(&data).unwrap();
        assert!(uri.starts_with("application/pdf;base64,"));
    }

    fn base64_decode(input: &str) -> Vec<u8> {
        let alphabet: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut buf: Vec<u32> = Vec::new();
        for c in input.bytes() {
            if c == b'=' {
                continue;
            }
            let val = alphabet.iter().position(|&b| b == c).unwrap() as u32;
            buf.push(val);
        }
        let mut bytes = Vec::new();
        for chunk in buf.chunks(4) {
            let mut triple: u32 = 0;
            for &v in chunk {
                triple = (triple << 6) | v;
            }
            if chunk.len() >= 2 {
                bytes.push((triple >> 16) as u8);
            }
            if chunk.len() >= 3 {
                bytes.push((triple >> 8) as u8);
            }
            if chunk.len() >= 4 {
                bytes.push(triple as u8);
            }
        }
        bytes
    }
}
