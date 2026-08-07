//! Flexible legend positioning example.

use mathverse_plot::{
    DataPoint, DataSeries, LegendConfig, LegendPosition, PlotConfig, SvgPlot,
};
use mathverse_plot::style::{Color, PlotStyle};
use mathverse_plot::save::PlotSaver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create multiple data series
    let series1: Vec<DataPoint> = (0..=20)
        .map(|i| {
            let x = i as f64 * 0.5;
            DataPoint::new(x, x.sin())
        })
        .collect();

    let series2: Vec<DataPoint> = (0..=20)
        .map(|i| {
            let x = i as f64 * 0.5;
            DataPoint::new(x, x.cos())
        })
        .collect();

    let series3: Vec<DataPoint> = (0..=20)
        .map(|i| {
            let x = i as f64 * 0.5;
            DataPoint::new(x, (x * 0.5).sin())
        })
        .collect();

    // Example 1: Default legend position (upper right)
    let config = PlotConfig::new()
        .with_title("Legend at Upper Right (Default)")
        .with_x_label("x")
        .with_y_label("y");

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::with_style(
        "sin(x)".to_string(),
        series1.clone(),
        PlotStyle::default().with_line_color(Color::RED),
    ));
    plot.add_series(DataSeries::with_style(
        "cos(x)".to_string(),
        series2.clone(),
        PlotStyle::default().with_line_color(Color::BLUE),
    ));

    let svg = plot.generate();
    PlotSaver::new(&svg).save_png("legend_upper_right.png")?;
    println!("Wrote legend_upper_right.png");

    // Example 2: Legend at bottom left
    let config = PlotConfig::new()
        .with_title("Legend at Lower Left")
        .with_x_label("x")
        .with_y_label("y");

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::with_style(
        "sin(x)".to_string(),
        series1.clone(),
        PlotStyle::default().with_line_color(Color::RED),
    ));
    plot.add_series(DataSeries::with_style(
        "cos(x)".to_string(),
        series2.clone(),
        PlotStyle::default().with_line_color(Color::BLUE),
    ));

    let mut svg = plot.generate();

    // Add legend manually (simplified - in real use, this would be integrated)
    let legend_svg = format!(
        r#"  <rect x="60" y="450" width="120" height="60" fill="white" stroke="gray" rx="4"/>
  <line x1="70" y1="470" x2="90" y2="470" stroke="red" stroke-width="2"/>
  <text x="95" y="474" font-size="11">sin(x)</text>
  <line x1="70" y1="490" x2="90" y2="490" stroke="blue" stroke-width="2"/>
  <text x="95" y="494" font-size="11">cos(x)</text>"#
    );

    svg = svg.replace("</svg>", &format!("{}\n</svg>", legend_svg));
    PlotSaver::new(&svg).save_png("legend_lower_left.png")?;
    println!("Wrote legend_lower_left.png");

    // Example 3: Legend outside right
    let config = PlotConfig::new()
        .with_title("Legend Outside Right")
        .with_x_label("x")
        .with_y_label("y");

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::with_style(
        "sin(x)".to_string(),
        series1.clone(),
        PlotStyle::default().with_line_color(Color::RED),
    ));
    plot.add_series(DataSeries::with_style(
        "cos(x)".to_string(),
        series2.clone(),
        PlotStyle::default().with_line_color(Color::BLUE),
    ));
    plot.add_series(DataSeries::with_style(
        "sin(x/2)".to_string(),
        series3.clone(),
        PlotStyle::default().with_line_color(Color::GREEN),
    ));

    let mut svg = plot.generate();

    // Add legend outside right
    let legend_svg = format!(
        r#"  <rect x="760" y="50" width="100" height="80" fill="white" stroke="gray" rx="4"/>
  <line x1="770" y1="70" x2="790" y2="70" stroke="red" stroke-width="2"/>
  <text x="795" y="74" font-size="11">sin(x)</text>
  <line x1="770" y1="90" x2="790" y2="90" stroke="blue" stroke-width="2"/>
  <text x="795" y="94" font-size="11">cos(x)</text>
  <line x1="770" y1="110" x2="790" y2="110" stroke="green" stroke-width="2"/>
  <text x="795" y="114" font-size="11">sin(x/2)</text>"#
    );

    svg = svg.replace("</svg>", &format!("{}\n</svg>", legend_svg));
    PlotSaver::new(&svg).save_png("legend_outside_right.png")?;
    println!("Wrote legend_outside_right.png");

    Ok(())
}
