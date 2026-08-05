//! Datetime axis example for time series data.

use mathverse_plot::{DataPoint, DataSeries, DateTime, DatetimeAxis, PlotConfig, PlotSaver, SvgPlot};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate time series data: daily temperatures for a month
    let start = DateTime::new(2024, 1, 1, 0, 0, 0);
    let end = DateTime::new(2024, 1, 31, 0, 0, 0);

    let mut points = Vec::new();
    let mut t = start.timestamp;
    let mut day = 0;

    while t <= end.timestamp {
        // Simulate temperature: base 20 + seasonal variation + noise
        let temp = 20.0 + 5.0 * (day as f64 * 0.2).sin() + (day as f64 * 0.7).cos() * 2.0;
        points.push(DataPoint::new(DateTime::from_timestamp(t).to_f64(), temp));
        t += 86400; // 1 day
        day += 1;
    }

    let config = PlotConfig::new()
        .with_title("Daily Temperature - January 2024")
        .with_x_label("Date")
        .with_y_label("Temperature (°C)");

    let mut plot = SvgPlot::new(config);
    plot.add_series(DataSeries::new("Temperature".to_string(), points));

    // Generate SVG
    let svg = plot.generate();

    // Add date labels (simplified - in real use, this would be integrated)
    let datetime_axis = DatetimeAxis::new().with_format("%m-%d");
    let ticks = datetime_axis.ticks(start, end);

    let date_labels: String = ticks
        .iter()
        .map(|(t, label)| {
            let x = 50.0 + (*t - start.to_f64()) / (end.to_f64() - start.to_f64()) * 700.0;
            format!(
                r#"  <text x="{x}" y="580" text-anchor="middle" font-size="10" transform="rotate(-45, {x}, 580)">{label}</text>"#
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut labeled_svg = svg;
    labeled_svg = labeled_svg.replace("</svg>", &format!("{}\n</svg>", date_labels));

PlotSaver::new(&labeled_svg).save_png("datetime.png")?;
println!("Wrote datetime.png ({} bytes)", labeled_svg.len());

    Ok(())
}
