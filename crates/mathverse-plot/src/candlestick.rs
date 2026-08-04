//! Candlestick chart rendering for OHLC financial data.

use crate::style::Color;

/// A single OHLC candlestick.
#[derive(Debug, Clone, Copy)]
pub struct Candlestick {
    pub x: f64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

impl Candlestick {
    pub fn new(x: f64, open: f64, high: f64, low: f64, close: f64) -> Self {
        Self {
            x,
            open,
            high,
            low,
            close,
        }
    }

    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    pub fn body_top(&self) -> f64 {
        self.open.max(self.close)
    }

    pub fn body_bottom(&self) -> f64 {
        self.open.min(self.close)
    }
}

/// A candlestick series with associated label and colors.
#[derive(Debug, Clone)]
pub struct CandlestickSeries {
    pub name: String,
    pub candles: Vec<Candlestick>,
    pub bullish_color: Color,
    pub bearish_color: Color,
}

impl CandlestickSeries {
    pub fn new(name: impl Into<String>, candles: Vec<Candlestick>) -> Self {
        Self {
            name: name.into(),
            candles,
            bullish_color: Color::Rgb(34, 139, 34), // forest green
            bearish_color: Color::Rgb(220, 20, 20), // red
        }
    }

    pub fn with_colors(mut self, bullish: Color, bearish: Color) -> Self {
        self.bullish_color = bullish;
        self.bearish_color = bearish;
        self
    }
}

/// Render candlestick series to SVG.
pub fn render_candlestick_svg(
    series: &[CandlestickSeries],
    title: &str,
    width: u32,
    height: u32,
) -> String {
    let w = width as f64;
    let h = height as f64;
    let pad = 40.0;
    let plot_w = w - 2.0 * pad;
    let plot_h = h - 2.0 * pad;

    // Compute ranges
    let all_candles: Vec<&Candlestick> = series.iter().flat_map(|s| s.candles.iter()).collect();
    if all_candles.is_empty() {
        return format!(r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"/>"#);
    }

    let x_min = all_candles
        .iter()
        .map(|c| c.x)
        .fold(f64::INFINITY, f64::min);
    let x_max = all_candles
        .iter()
        .map(|c| c.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = all_candles
        .iter()
        .map(|c| c.low)
        .fold(f64::INFINITY, f64::min);
    let y_max = all_candles
        .iter()
        .map(|c| c.high)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min) * 0.05;
    let y_lo = y_min - y_pad;
    let y_hi = y_max + y_pad;
    let x_range = if (x_max - x_min).abs() < f64::EPSILON {
        1.0
    } else {
        x_max - x_min
    };

    let to_x = |x: f64| -> f64 { pad + (x - x_min) / x_range * plot_w };
    let to_y = |y: f64| -> f64 { pad + (y_hi - y) / (y_hi - y_lo) * plot_h };

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">"#
    ));
    svg.push_str(&format!(
        r#"  <rect width="{w}" height="{h}" fill="white"/>"#
    ));

    // Y axis ticks
    let y_step = nice_step(y_hi - y_lo, 6);
    let mut y_tick = (y_lo / y_step).ceil() * y_step;
    let pad_right = w - pad;
    let pad_bottom = h - pad;
    while y_tick <= y_hi {
        let py = to_y(y_tick);
        let label_x = pad - 4.0;
        svg.push_str(&format!(
            r##"  <line x1="{pad}" y1="{py:.2}" x2="{pad_right:.2}" y2="{py:.2}" stroke="#eee" stroke-width="0.5"/>"##
        ));
        svg.push_str(&format!(
            r##"  <text x="{label_x:.2}" y="{py:.2}" font-size="10" text-anchor="end" fill="#666">{y_tick:.2}</text>"##
        ));
        y_tick += y_step;
    }

    // Axes
    svg.push_str(&format!(
        r##"  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad_bottom:.2}" stroke="black" stroke-width="1"/>"##
    ));
    svg.push_str(&format!(
        r##"  <line x1="{pad}" y1="{pad_bottom:.2}" x2="{pad_right:.2}" y2="{pad_bottom:.2}" stroke="black" stroke-width="1"/>"##
    ));

    // Count total candles to compute candle width
    let total_candles: usize = series.iter().map(|s| s.candles.len()).sum();
    let candle_w = (plot_w / total_candles as f64 * 0.6).max(2.0);
    let wick_w = (candle_w * 0.1).max(1.0);

    // Draw candles
    let mut idx = 0;
    for s in series {
        for c in &s.candles {
            let cx = to_x(c.x);
            let body_top = to_y(c.body_top());
            let body_bot = to_y(c.body_bottom());
            let body_h = (body_bot - body_top).max(1.0);
            let color = if c.is_bullish() {
                s.bullish_color.to_hex()
            } else {
                s.bearish_color.to_hex()
            };

            // Wick (high to low)
            svg.push_str(&format!(
                r#"  <line x1="{cx:.2}" y1="{:.2}" x2="{cx:.2}" y2="{:.2}" stroke="{color}" stroke-width="{wick_w:.1}"/>"#,
                to_y(c.high), to_y(c.low)
            ));

            // Body (open to close)
            svg.push_str(&format!(
                r#"  <rect x="{:.2}" y="{body_top:.2}" width="{candle_w:.1}" height="{body_h:.2}" fill="{color}"/>"#,
                cx - candle_w / 2.0
            ));

            idx += 1;
        }
    }

    // Title
    if !title.is_empty() {
        svg.push_str(&format!(
            r#"  <text x="{}" y="18" font-size="14" text-anchor="middle" font-weight="bold">{}</text>"#,
            w / 2.0,
            escape_xml(title)
        ));
    }

    svg.push_str("</svg>");
    svg
}

fn nice_step(range: f64, target_ticks: usize) -> f64 {
    if range <= 0.0 || target_ticks == 0 {
        return 1.0;
    }
    let raw = range / target_ticks as f64;
    let mag = 10.0_f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = if norm < 1.5 {
        1.0
    } else if norm < 3.5 {
        2.0
    } else if norm < 7.5 {
        5.0
    } else {
        10.0
    };
    step * mag
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_bullish_bearish() {
        let bull = Candlestick::new(0.0, 10.0, 15.0, 8.0, 14.0);
        assert!(bull.is_bullish());
        assert_eq!(bull.body_top(), 14.0);
        assert_eq!(bull.body_bottom(), 10.0);

        let bear = Candlestick::new(1.0, 14.0, 16.0, 9.0, 10.0);
        assert!(!bear.is_bullish());
        assert_eq!(bear.body_top(), 14.0);
        assert_eq!(bear.body_bottom(), 10.0);
    }

    #[test]
    fn render_empty() {
        let svg = render_candlestick_svg(&[], "Empty", 400, 300);
        assert!(svg.contains("<svg"));
        assert!(!svg.contains("<rect"));
    }

    #[test]
    fn render_candles() {
        let series = vec![CandlestickSeries::new(
            "AAPL",
            vec![
                Candlestick::new(0.0, 100.0, 110.0, 95.0, 108.0),
                Candlestick::new(1.0, 108.0, 115.0, 105.0, 106.0),
            ],
        )];
        let svg = render_candlestick_svg(&series, "AAPL", 600, 400);
        assert!(svg.contains("AAPL"));
        assert!(svg.contains("<rect")); // body rects exist
        assert!(svg.contains("<line")); // wick lines exist
    }
}
