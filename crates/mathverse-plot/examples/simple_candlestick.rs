//! Candlestick chart of OHLC data, with a `mathverse-finance`-derived return
//! printed to stdout. Renders to SVG.
//!
//! Run: `cargo run -p mathverse-plot --example simple_candlestick`

use mathverse_finance::investment::holding_period_return;
use mathverse_plot::{render_candlestick_svg, Candlestick, CandlestickSeries};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let candles = vec![
        Candlestick::new(0.0, 100.0, 105.0, 98.0, 103.0),
        Candlestick::new(1.0, 103.0, 109.0, 102.0, 108.0),
        Candlestick::new(2.0, 108.0, 107.0, 101.0, 102.0),
        Candlestick::new(3.0, 102.0, 106.0, 100.0, 105.5),
        Candlestick::new(4.0, 105.5, 112.0, 105.0, 111.0),
        Candlestick::new(5.0, 111.0, 110.0, 104.0, 105.0),
        Candlestick::new(6.0, 105.0, 108.0, 103.0, 107.5),
        Candlestick::new(7.0, 107.5, 114.0, 107.0, 113.5),
        Candlestick::new(8.0, 113.5, 115.0, 110.0, 111.0),
        Candlestick::new(9.0, 111.0, 113.0, 108.0, 112.5),
    ];

    let start = candles.first().unwrap().open;
    let end = candles.last().unwrap().close;
    let ret = holding_period_return(start, end, 0.0);
    println!("holding-period return over the window: {ret:.2}%");

    let series = CandlestickSeries::new("ACME", candles);
    let svg = render_candlestick_svg(&[series], "ACME daily OHLC", 640, 420);
    PlotSaver::new(svg).save_png("candlestick.png")?;
    println!("wrote candlestick.svg ({} bytes)", svg.len());
    Ok(())
}
