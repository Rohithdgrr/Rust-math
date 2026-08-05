//! Error bar example: mean + 95% CI per group (z = 1.96), from
//! `mathverse_statistics::mean_ci`.

use mathverse_plot::common::{DataPoint, DataSeries, PlotConfig};
use mathverse_plot::errorbar::ErrorBar;
use mathverse_plot::style::Color;
use mathverse_plot::{PlotSaver, SvgPlot};

fn main() -> mathverse_plot::PlotResult<()> {
    let mut plot = SvgPlot::new(
        PlotConfig::new()
            .with_title("Means with 95% CI")
            .with_x_label("group")
            .with_y_label("value"),
    );

    let mut means = Vec::new();
    for i in 0..5 {
        // Pseudo-random-ish group around sin(i), width growing with i.
        let n = 8 + i;
        let group: Vec<f64> = (0..n)
            .map(|j| {
                let noise = ((j * (i + 1)) % 7) as f64 / 7.0 - 0.5;
                (i as f64).sin() * 2.0 + noise * (0.5 + i as f64 / 4.0)
            })
            .collect();

        let bar = ErrorBar::ci(&group, 1.96)?;
        plot.add_error_bar(i as f64, bar, Color::BLUE);
        means.push(DataPoint::new(i as f64, bar.center));
    }

    plot.add_series(DataSeries::with_style(
        "mean".to_string(),
        means,
        mathverse_plot::style::PlotStyle::default()
            .with_marker_style(mathverse_plot::style::MarkerStyle::Circle),
    ));

PlotSaver::new(&plot.generate()).save_png("errorbars.png")?;
println!("wrote errorbars.png");
    Ok(())
}
