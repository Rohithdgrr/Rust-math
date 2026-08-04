//! Generate PNG images for representative plot types.
//! Demonstrates PNG-by-default behavior (PlotSaver::save_png, FormatSet::png()).

use mathverse_plot::*;

fn main() {
    std::fs::create_dir_all("target/plot_images").unwrap();

    // 1. Basic line plot via SvgPlot
    {
        let config = PlotConfig::new().with_title("Line").with_dimensions(600, 400);
        let mut plot = SvgPlot::new(config);
        let xs: Vec<f64> = (0..50).map(|i| i as f64 * 0.2).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let pts: Vec<DataPoint> = xs.iter().zip(ys.iter()).map(|(x, y)| DataPoint::new(*x, *y)).collect();
        plot.add_series(DataSeries::new("sin(x)".to_string(), pts));
        let svg = plot.generate();
        PlotSaver::new(&svg).save_png("target/plot_images/line");
    }

    // 2. Pie chart
    {
        let slices = vec![
            PieSlice::new("A".to_string(), 30.0, Color::RED),
            PieSlice::new("B".to_string(), 45.0, Color::BLUE),
            PieSlice::new("C".to_string(), 25.0, Color::GREEN),
        ];
        let config = PieConfig::new();
        let svg = render_pie_chart(&slices, &config).unwrap();
        PlotSaver::new(&svg).save_png("target/plot_images/pie");
    }

    // 3. Radar chart
    {
        let pts = vec![
            mathverse_plot::radar::RadarPoint::new("A", 5.0),
            mathverse_plot::radar::RadarPoint::new("B", 3.0),
            mathverse_plot::radar::RadarPoint::new("C", 4.0),
        ];
        let series = mathverse_plot::radar::RadarSeries::new("S1".to_string(), pts, mathverse_plot::Color::BLUE);
        let config = RadarConfig::new();
        let svg = render_radar_chart(&[series], &config).unwrap();
        PlotSaver::new(&svg).save_png("target/plot_images/radar");
    }

    // 4. Step plot
    {
        let pts: Vec<DataPoint> = vec![DataPoint::new(0.0, 1.0), DataPoint::new(1.0, 2.0), DataPoint::new(2.0, 1.0)];
        let config = StepConfig::new();
        let svg = render_step_plot(&pts, &config).unwrap();
        PlotSaver::new(&svg).save_png("target/plot_images/step");
    }

    println!("Generated PNG images in target/plot_images/");
}
